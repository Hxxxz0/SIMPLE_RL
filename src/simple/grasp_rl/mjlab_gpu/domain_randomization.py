"""Per-world CUDA domain randomization for the frozen tabletop scene."""

from __future__ import annotations

from typing import Protocol

import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel

from simple.grasp_rl.mjlab_gpu.config import DomainRandomizationConfig
from simple.grasp_rl.mjlab_gpu.simulation import GpuSimulation, _is_descendant
from simple.grasp_rl.schema import JOINT_NAMES

MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "geom_friction",
    "dof_damping",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
)


class _GpuController(Protocol):
    logical_actuator_indices: torch.Tensor
    actuator_strength_scale: torch.Tensor


def _uniform(
    low: float,
    high: float,
    shape: tuple[int, ...],
    *,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(
        float(low), float(high), generator=generator
    )


def _apply_position_focus_mixture(
    translation: torch.Tensor,
    *,
    probability: float,
    jitter_xy: tuple[float, float],
    offset_center_xy: tuple[float, float],
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Mix focused positions into a base distribution without changing legacy RNG."""

    if probability <= 0.0:
        return translation
    sample_shape = translation.shape[:-1]
    device = str(translation.device)
    focused = torch.stack(
        (
            _uniform(
                scale * (offset_center_xy[0] - jitter_xy[0]),
                scale * (offset_center_xy[0] + jitter_xy[0]),
                sample_shape,
                device=device,
                generator=generator,
            ),
            _uniform(
                scale * (offset_center_xy[1] - jitter_xy[1]),
                scale * (offset_center_xy[1] + jitter_xy[1]),
                sample_shape,
                device=device,
                generator=generator,
            ),
        ),
        dim=-1,
    )
    selected = torch.rand(
        sample_shape, device=translation.device, generator=generator
    ) < float(probability)
    return torch.where(selected[..., None], focused, translation)


def _apply_free_joint_pose(
    qpos: torch.Tensor,
    env_ids: torch.Tensor,
    address: int,
    translation_xy: torch.Tensor,
    yaw: torch.Tensor,
) -> None:
    qpos[env_ids, address : address + 2] += translation_xy
    base_quaternion = qpos[env_ids, address + 3 : address + 7].clone()
    half_yaw = 0.5 * yaw
    yaw_quaternion = torch.stack(
        (
            half_yaw.cos(),
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            half_yaw.sin(),
        ),
        dim=-1,
    )
    bw, bx, by, bz = base_quaternion.unbind(-1)
    yw, _, _, yz = yaw_quaternion.unbind(-1)
    qpos[env_ids, address + 3 : address + 7] = torch.stack(
        (
            yw * bw - yz * bz,
            yw * bx - yz * by,
            yw * by + yz * bx,
            yw * bz + yz * bw,
        ),
        dim=-1,
    )


class GpuDomainRandomizer:
    """Randomize reset state and expanded mjlab model fields without CPU fallback."""

    def __init__(
        self,
        gpu: GpuSimulation,
        controller: _GpuController,
        config: DomainRandomizationConfig,
        *,
        seed: int,
    ):
        self.gpu = gpu
        self.sim = gpu.sim
        self.controller = controller
        self.config = config
        self.device = self.sim.device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        model = self.sim.mj_model
        target_name = gpu.bundle.manifest["roles"]["primary"]
        roles = gpu.bundle.manifest["roles"]
        contact_role_names = {
            name
            for role in ("destination", "support", "auxiliary")
            if (name := roles.get(role)) is not None
        }
        self.target_body_id = model.body(target_name).id
        target_joint_id = int(model.body_jntadr[self.target_body_id])
        if (
            target_joint_id < 0
            or model.jnt_type[target_joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError("Randomized target must have a free joint")
        self.target_qpos_address = int(model.jnt_qposadr[target_joint_id])
        destination_name = roles.get("destination")
        self.destination_body_id = (
            None if destination_name is None else model.body(destination_name).id
        )
        destination_joint_id = (
            -1
            if self.destination_body_id is None
            else int(model.body_jntadr[self.destination_body_id])
        )
        self.destination_qpos_address = (
            None
            if destination_joint_id < 0
            or model.jnt_type[destination_joint_id] != mujoco.mjtJoint.mjJNT_FREE
            else int(model.jnt_qposadr[destination_joint_id])
        )
        base_joint_id = model.joint("floating_base_joint").id
        if model.jnt_type[base_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("floating_base_joint must be a free joint")
        self.robot_base_qpos_address = int(model.jnt_qposadr[base_joint_id])
        excluded_body_ids = {
            self.target_body_id,
            self.destination_body_id,
            int(model.jnt_bodyid[base_joint_id]),
        }
        self.distractor_body_ids = []
        self.distractor_qpos_addresses = []
        for body_id in range(1, model.nbody):
            joint_id = int(model.body_jntadr[body_id])
            if (
                body_id in excluded_body_ids
                or joint_id < 0
                or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
            ):
                continue
            self.distractor_body_ids.append(body_id)
            self.distractor_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        destination_pose_enabled = bool(
            any(config.destination_position_jitter_xy) or config.destination_yaw_jitter
        )
        if destination_pose_enabled and self.destination_qpos_address is None:
            raise ValueError("Destination pose DR requires a free destination joint")
        distractor_pose_enabled = bool(
            any(config.distractor_position_jitter_xy) or config.distractor_yaw_jitter
        )
        if distractor_pose_enabled and not self.distractor_qpos_addresses:
            raise ValueError(
                "Distractor pose DR requires at least one free scene object"
            )
        self.target_geom_ids = torch.tensor(
            [
                index
                for index in range(model.ngeom)
                if _is_descendant(
                    model, int(model.geom_bodyid[index]), self.target_body_id
                )
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.contact_geom_ids = torch.tensor(
            [
                index
                for index in range(model.ngeom)
                if any(
                    _is_descendant(
                        model,
                        int(model.geom_bodyid[index]),
                        model.body(name).id,
                    )
                    for name in contact_role_names
                )
            ],
            dtype=torch.long,
            device=self.device,
        )
        joint_ids = [model.joint(name).id for name in JOINT_NAMES]
        self.dof_ids = torch.tensor(
            [int(model.jnt_dofadr[index]) for index in joint_ids],
            dtype=torch.long,
            device=self.device,
        )
        self.actuator_ids = controller.logical_actuator_indices
        self.arm_actuator_ids = self.actuator_ids[15:29]
        self.defaults = {
            field: self.sim.get_default_field(field).clone() for field in MODEL_FIELDS
        }
        self.sim.expand_model_fields(MODEL_FIELDS)
        self.action_delay_steps = torch.zeros(
            self.sim.num_envs, dtype=torch.long, device=self.device
        )
        self.target_mass_scale = torch.ones(self.sim.num_envs, device=self.device)
        self.friction_scale = torch.ones(self.sim.num_envs, device=self.device)
        self.joint_damping_scale = torch.ones(
            self.sim.num_envs, len(self.dof_ids), device=self.device
        )
        self.actuator_strength_scale = torch.ones(
            self.sim.num_envs, len(self.actuator_ids), device=self.device
        )
        self.target_translation_xy = torch.zeros(
            self.sim.num_envs, 2, device=self.device
        )
        self.target_yaw = torch.zeros(self.sim.num_envs, device=self.device)
        self.destination_translation_xy = torch.zeros(
            self.sim.num_envs, 2, device=self.device
        )
        self.destination_yaw = torch.zeros(self.sim.num_envs, device=self.device)
        self.distractor_translation_xy = torch.zeros(
            self.sim.num_envs,
            len(self.distractor_qpos_addresses),
            2,
            device=self.device,
        )
        self.distractor_yaw = torch.zeros(
            self.sim.num_envs,
            len(self.distractor_qpos_addresses),
            device=self.device,
        )
        self.robot_base_translation_xy = torch.zeros(
            self.sim.num_envs, 2, device=self.device
        )
        self.robot_base_yaw = torch.zeros(self.sim.num_envs, device=self.device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "generator_state": self.generator.get_state(),
            "action_delay_steps": self.action_delay_steps.clone(),
            "target_mass_scale": self.target_mass_scale.clone(),
            "friction_scale": self.friction_scale.clone(),
            "joint_damping_scale": self.joint_damping_scale.clone(),
            "actuator_strength_scale": self.actuator_strength_scale.clone(),
            "target_translation_xy": self.target_translation_xy.clone(),
            "target_yaw": self.target_yaw.clone(),
            "destination_translation_xy": self.destination_translation_xy.clone(),
            "destination_yaw": self.destination_yaw.clone(),
            "distractor_translation_xy": self.distractor_translation_xy.clone(),
            "distractor_yaw": self.distractor_yaw.clone(),
            "robot_base_translation_xy": self.robot_base_translation_xy.clone(),
            "robot_base_yaw": self.robot_base_yaw.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"].cpu())
        for name in (
            "target_mass_scale",
            "friction_scale",
            "joint_damping_scale",
            "actuator_strength_scale",
        ):
            value = state.get(name)
            if value is None:
                getattr(self, name).fill_(1.0)
            else:
                getattr(self, name).copy_(value.to(self.device))
        for name in (
            "action_delay_steps",
            "target_translation_xy",
            "target_yaw",
            "destination_translation_xy",
            "destination_yaw",
            "distractor_translation_xy",
            "distractor_yaw",
            "robot_base_translation_xy",
            "robot_base_yaw",
        ):
            value = state.get(name)
            if value is None:
                getattr(self, name).zero_()
            else:
                getattr(self, name).copy_(value.to(self.device))

    def reset(
        self, env_ids: torch.Tensor, *, training: bool, strength: float = 1.0
    ) -> None:
        count = len(env_ids)
        if count == 0:
            return
        cfg = self.config
        scale = min(max(float(strength), 0.0), 1.0)
        enabled = bool(training and cfg.enabled and scale > 0.0)

        def sample_pose(
            position_jitter_xy: tuple[float, float],
            yaw_jitter: float,
            slots: int = 1,
            position_offset_center_xy: tuple[float, float] = (0.0, 0.0),
        ) -> tuple[torch.Tensor, torch.Tensor]:
            translation_shape = (count, 2) if slots == 1 else (count, slots, 2)
            yaw_shape = (count,) if slots == 1 else (count, slots)
            if not enabled or not (
                any(position_jitter_xy)
                or any(position_offset_center_xy)
                or yaw_jitter
            ):
                return (
                    torch.zeros(translation_shape, device=self.device),
                    torch.zeros(yaw_shape, device=self.device),
                )
            max_x, max_y = (
                scale * position_jitter_xy[0],
                scale * position_jitter_xy[1],
            )
            sample_shape = (count,) if slots == 1 else (count, slots)
            translation = torch.stack(
                (
                    _uniform(
                        -max_x,
                        max_x,
                        sample_shape,
                        device=self.device,
                        generator=self.generator,
                    ),
                    _uniform(
                        -max_y,
                        max_y,
                        sample_shape,
                        device=self.device,
                        generator=self.generator,
                    ),
                ),
                dim=-1,
            )
            if any(position_offset_center_xy):
                center = torch.tensor(
                    position_offset_center_xy,
                    dtype=translation.dtype,
                    device=self.device,
                )
                translation.add_(center * scale)
            yaw = _uniform(
                -scale * yaw_jitter,
                scale * yaw_jitter,
                yaw_shape,
                device=self.device,
                generator=self.generator,
            )
            return translation, yaw

        if enabled:

            def scaled_range(bounds: tuple[float, float]) -> tuple[float, float]:
                return (
                    1.0 + scale * (bounds[0] - 1.0),
                    1.0 + scale * (bounds[1] - 1.0),
                )

            mass_scale = _uniform(
                *scaled_range(cfg.target_mass_scale),
                (count,),
                device=self.device,
                generator=self.generator,
            )
            friction_scale = _uniform(
                *scaled_range(cfg.friction_scale),
                (count, 1, 1),
                device=self.device,
                generator=self.generator,
            )
            damping_scale = _uniform(
                *scaled_range(cfg.joint_damping_scale),
                (count, len(self.dof_ids)),
                device=self.device,
                generator=self.generator,
            )
            strength_scale = _uniform(
                *scaled_range(cfg.actuator_strength_scale),
                (count, len(self.actuator_ids)),
                device=self.device,
                generator=self.generator,
            )
            translation, yaw = sample_pose(
                cfg.target_position_jitter_xy,
                cfg.target_yaw_jitter,
                position_offset_center_xy=cfg.target_position_offset_center_xy,
            )
            translation = _apply_position_focus_mixture(
                translation,
                probability=cfg.target_position_focus_probability,
                jitter_xy=cfg.target_position_focus_jitter_xy,
                offset_center_xy=cfg.target_position_focus_offset_center_xy,
                scale=scale,
                generator=self.generator,
            )
            if cfg.action_delay_max_steps:
                delay = (
                    torch.rand((count,), device=self.device, generator=self.generator)
                    < (0.5 * scale)
                ).to(torch.long)
            else:
                delay = torch.zeros(count, dtype=torch.long, device=self.device)
        else:
            mass_scale = torch.ones(count, device=self.device)
            friction_scale = torch.ones(count, 1, 1, device=self.device)
            damping_scale = torch.ones(count, len(self.dof_ids), device=self.device)
            strength_scale = torch.ones(
                count, len(self.actuator_ids), device=self.device
            )
            translation = torch.zeros(count, 2, device=self.device)
            yaw = torch.zeros(count, device=self.device)
            delay = torch.zeros(count, dtype=torch.long, device=self.device)

        destination_translation, destination_yaw = sample_pose(
            cfg.destination_position_jitter_xy, cfg.destination_yaw_jitter
        )
        distractor_translation, distractor_yaw = sample_pose(
            cfg.distractor_position_jitter_xy,
            cfg.distractor_yaw_jitter,
            len(self.distractor_qpos_addresses),
        )
        base_translation, base_yaw = sample_pose(
            cfg.robot_base_position_jitter_xy, cfg.robot_base_yaw_jitter
        )

        model = self.sim.model
        model.body_mass[env_ids, self.target_body_id] = (
            self.defaults["body_mass"][self.target_body_id] * mass_scale
        )
        model.body_inertia[env_ids, self.target_body_id] = (
            self.defaults["body_inertia"][self.target_body_id] * mass_scale[:, None]
        )
        geom_ids = torch.cat((self.target_geom_ids, self.contact_geom_ids)).unique()
        env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
        model.geom_friction[env_grid, geom_grid] = (
            self.defaults["geom_friction"][geom_ids][None] * friction_scale
        )
        env_grid, dof_grid = torch.meshgrid(env_ids, self.dof_ids, indexing="ij")
        model.dof_damping[env_grid, dof_grid] = (
            self.defaults["dof_damping"][self.dof_ids][None] * damping_scale
        )
        env_grid, arm_grid = torch.meshgrid(
            env_ids, self.arm_actuator_ids, indexing="ij"
        )
        arm_strength = strength_scale[:, 15:29, None]
        for field in ("actuator_gainprm", "actuator_biasprm"):
            getattr(model, field)[env_grid, arm_grid] = (
                self.defaults[field][self.arm_actuator_ids][None] * arm_strength
            )
        model.actuator_forcerange[env_grid, arm_grid] = (
            self.defaults["actuator_forcerange"][self.arm_actuator_ids][None]
            * arm_strength
        )
        self.controller.actuator_strength_scale[env_ids] = strength_scale
        self.action_delay_steps[env_ids] = delay
        self.target_mass_scale[env_ids] = mass_scale
        self.friction_scale[env_ids] = friction_scale.flatten()
        self.joint_damping_scale[env_ids] = damping_scale
        self.actuator_strength_scale[env_ids] = strength_scale
        self.target_translation_xy[env_ids] = translation
        self.target_yaw[env_ids] = yaw
        self.destination_translation_xy[env_ids] = destination_translation
        self.destination_yaw[env_ids] = destination_yaw
        self.distractor_translation_xy[env_ids] = distractor_translation
        self.distractor_yaw[env_ids] = distractor_yaw
        self.robot_base_translation_xy[env_ids] = base_translation
        self.robot_base_yaw[env_ids] = base_yaw

        qpos = self.sim.data.qpos
        _apply_free_joint_pose(
            qpos, env_ids, self.target_qpos_address, translation, yaw
        )
        if self.destination_qpos_address is not None:
            _apply_free_joint_pose(
                qpos,
                env_ids,
                self.destination_qpos_address,
                destination_translation,
                destination_yaw,
            )
        for slot, address in enumerate(self.distractor_qpos_addresses):
            _apply_free_joint_pose(
                qpos,
                env_ids,
                address,
                distractor_translation[:, slot],
                distractor_yaw[:, slot],
            )
        _apply_free_joint_pose(
            qpos,
            env_ids,
            self.robot_base_qpos_address,
            base_translation,
            base_yaw,
        )
        self.sim.recompute_constants(RecomputeLevel.set_const)
        self.sim.forward()
