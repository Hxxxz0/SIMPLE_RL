"""Per-world CUDA domain randomization for the frozen tabletop scene."""

from __future__ import annotations

from typing import Protocol

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
        if target_joint_id < 0:
            raise ValueError("Randomized target must have a free joint")
        self.target_qpos_address = int(model.jnt_qposadr[target_joint_id])
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
        self.target_translation_xy = torch.zeros(
            self.sim.num_envs, 2, device=self.device
        )
        self.target_yaw = torch.zeros(self.sim.num_envs, device=self.device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "generator_state": self.generator.get_state(),
            "action_delay_steps": self.action_delay_steps.clone(),
            "target_translation_xy": self.target_translation_xy.clone(),
            "target_yaw": self.target_yaw.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"].cpu())
        for name in (
            "action_delay_steps",
            "target_translation_xy",
            "target_yaw",
        ):
            getattr(self, name).copy_(state[name].to(self.device))

    def reset(
        self, env_ids: torch.Tensor, *, training: bool, strength: float = 1.0
    ) -> None:
        count = len(env_ids)
        if count == 0:
            return
        cfg = self.config
        scale = min(max(float(strength), 0.0), 1.0)
        enabled = bool(training and cfg.enabled and scale > 0.0)
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
            max_x, max_y = (
                scale * cfg.target_position_jitter_xy[0],
                scale * cfg.target_position_jitter_xy[1],
            )
            translation = torch.stack(
                (
                    _uniform(
                        -max_x,
                        max_x,
                        (count,),
                        device=self.device,
                        generator=self.generator,
                    ),
                    _uniform(
                        -max_y,
                        max_y,
                        (count,),
                        device=self.device,
                        generator=self.generator,
                    ),
                ),
                dim=-1,
            )
            yaw = _uniform(
                -scale * cfg.target_yaw_jitter,
                scale * cfg.target_yaw_jitter,
                (count,),
                device=self.device,
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
        self.target_translation_xy[env_ids] = translation
        self.target_yaw[env_ids] = yaw

        qpos = self.sim.data.qpos
        address = self.target_qpos_address
        qpos[env_ids, address : address + 2] += translation
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
        self.sim.recompute_constants(RecomputeLevel.set_const)
        self.sim.forward()
