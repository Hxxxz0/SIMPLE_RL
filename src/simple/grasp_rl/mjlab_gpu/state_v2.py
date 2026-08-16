"""Batched 331-D role-based task state computed entirely on CUDA."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simple.grasp_rl.mjlab_gpu.simulation import (
    GpuSimulation,
    GpuV2SensorLayout,
    _is_descendant,
)
from simple.grasp_rl.mjlab_gpu.state import _quat_to_matrix, _rotation_6d, _to_local
from simple.grasp_rl.schema import ACTOR_OBS_V2_DIM, JOINT_NAMES
from simple.grasp_rl.state_v2 import FAMILIES
from simple.grasp_rl.task_spec import TaskSpecV2, get_task_spec


@dataclass
class GpuEntityStateV2:
    present: bool
    pos_w: torch.Tensor
    rot_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor
    extents: torch.Tensor


@dataclass
class GpuHandStateV2:
    pos_w: torch.Tensor
    rot_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor
    primary_contact: torch.Tensor
    auxiliary_contact: torch.Tensor


@dataclass
class GpuTaskStateV2:
    pelvis_pos_w: torch.Tensor
    pelvis_rot_w: torch.Tensor
    pelvis_height: torch.Tensor
    hands: tuple[GpuHandStateV2, GpuHandStateV2]
    primary: GpuEntityStateV2
    destination: GpuEntityStateV2
    auxiliary: GpuEntityStateV2
    contact_forces_pelvis: torch.Tensor
    distal_pos_w: torch.Tensor
    fingertip_distances: torch.Tensor
    predicates: torch.Tensor
    arm_support_force: torch.Tensor
    articulation: torch.Tensor
    articulation_raw: torch.Tensor
    articulation_present: torch.Tensor
    initial_primary_pos: torch.Tensor
    initial_auxiliary_pos: torch.Tensor


class GpuTaskStateExtractorV2:
    def __init__(self, gpu: GpuSimulation):
        if not isinstance(gpu.sensors, GpuV2SensorLayout):
            raise TypeError("V2 state requires the v2 GPU sensor schema")
        spec = get_task_spec(gpu.bundle.manifest["task"])
        if not isinstance(spec, TaskSpecV2):
            raise TypeError("V2 state requires TaskSpecV2")
        self.gpu = gpu
        self.sim = gpu.sim
        self.sensors = gpu.sensors
        self.spec = spec
        self.device = self.sim.device
        self.num_envs = self.sim.num_envs
        model = self.sim.mj_model
        self.pelvis_id = model.body("pelvis").id
        self.ankle_ids = torch.tensor(
            [model.body("left_ankle_roll_link").id,
             model.body("right_ankle_roll_link").id],
            dtype=torch.long,
            device=self.device,
        )
        self.hand_ids = (model.body("left_wrist_yaw_link").id,
                         model.body("right_wrist_yaw_link").id)
        self.distal_ids = torch.tensor(
            [[model.body(name).id for name in (
                "left_hand_thumb_2_link", "left_hand_index_1_link",
                "left_hand_middle_1_link")],
             [model.body(name).id for name in (
                "right_hand_thumb_2_link", "right_hand_index_1_link",
                "right_hand_middle_1_link")]],
            dtype=torch.long,
            device=self.device,
        )
        roles = gpu.bundle.manifest["roles"]
        self.role_ids = {
            role: (None if not name else model.body(name).id)
            for role, name in roles.items()
        }
        for role in ("primary", "destination", "auxiliary", "support"):
            self.role_ids.setdefault(role, None)
        self.role_extents = {
            role: self._extent(role, body_id)
            for role, body_id in self.role_ids.items()
        }
        qpos, qvel = [], []
        for name in JOINT_NAMES:
            joint_id = model.joint(name).id
            qpos.append(int(model.jnt_qposadr[joint_id]))
            qvel.append(int(model.jnt_dofadr[joint_id]))
        self.qpos_indices = torch.tensor(qpos, dtype=torch.long, device=self.device)
        self.qvel_indices = torch.tensor(qvel, dtype=torch.long, device=self.device)
        reset_action = torch.tensor(
            gpu.bundle.manifest["reset"]["previous_physical_action"],
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_action = reset_action[None].repeat(self.num_envs, 1)
        self._initial_previous_action = reset_action
        self.initial_primary_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_auxiliary_pos = torch.zeros_like(self.initial_primary_pos)
        self.stage_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stage_progress = torch.zeros(self.num_envs, device=self.device)
        self.family_index = FAMILIES.index(spec.family)
        self._articulation_layout()
        self.sync_episode_origin(torch.arange(self.num_envs, device=self.device))

    def _extent(self, role: str, body_id: int | None) -> torch.Tensor:
        object_contract = self.gpu.bundle.manifest.get("object_contract")
        if role == "primary" and object_contract is not None:
            extents = torch.as_tensor(
                object_contract["half_extents_m"],
                dtype=torch.float32,
                device=self.device,
            )
            if extents.shape != (3,) or not torch.isfinite(extents).all() or (
                extents <= 0.0
            ).any():
                raise ValueError("Invalid grasp_anything primary half extents")
            return extents
        if body_id is None:
            radius = 0.0
        else:
            model = self.sim.mj_model
            radius = max(
                (float(model.geom_rbound[index]) for index in range(model.ngeom)
                 if _is_descendant(model, int(model.geom_bodyid[index]), body_id)),
                default=0.0,
            )
        return torch.full((3,), radius, dtype=torch.float32, device=self.device)

    def _articulation_layout(self) -> None:
        model = self.sim.mj_model
        qpos, qvel, limited, ranges, targets = [], [], [], [], []
        goal = self.spec.articulation
        if goal is not None:
            for name, target in zip(goal.joints[:2], goal.targets[:2], strict=True):
                try:
                    joint_id = model.joint(name).id
                except KeyError:
                    continue
                qpos.append(int(model.jnt_qposadr[joint_id]))
                qvel.append(int(model.jnt_dofadr[joint_id]))
                is_limited = bool(model.jnt_limited[joint_id])
                limited.append(is_limited)
                ranges.append(
                    tuple(float(x) for x in model.jnt_range[joint_id])
                    if is_limited else (-max(abs(target), 1.0), max(abs(target), 1.0))
                )
                targets.append(float(target))
        self.articulation_qpos = torch.tensor(qpos, dtype=torch.long, device=self.device)
        self.articulation_qvel = torch.tensor(qvel, dtype=torch.long, device=self.device)
        self.articulation_ranges = torch.tensor(ranges, dtype=torch.float32, device=self.device).reshape(-1, 2)
        self.articulation_targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        self.previous_action[indices] = self._initial_previous_action
        self.stage_index[indices] = 0
        self.stage_progress[indices] = 0.0

    def sync_episode_origin(self, env_ids: torch.Tensor) -> None:
        primary_id = self.role_ids["primary"]
        if primary_id is None:
            raise ValueError("V2 task has no primary body")
        self.initial_primary_pos[env_ids] = self.sim.data.xpos[env_ids, primary_id]
        auxiliary_id = self.role_ids["auxiliary"]
        self.initial_auxiliary_pos[env_ids] = (
            0.0 if auxiliary_id is None else self.sim.data.xpos[env_ids, auxiliary_id]
        )

    def set_previous_action(self, action: torch.Tensor) -> None:
        self.previous_action.copy_(action)

    def set_stage(self, index: torch.Tensor, progress: torch.Tensor) -> None:
        self.stage_index.copy_(index.clamp(0, 7).to(torch.long))
        self.stage_progress.copy_(progress.clamp(0.0, 1.0))

    def _sensor_vectors(self, slices: tuple[slice, ...]) -> torch.Tensor:
        return torch.stack([self.sim.data.sensordata[:, item] for item in slices], dim=1)

    def _contact(self, sensor: slice | None) -> torch.Tensor:
        if sensor is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.sim.data.sensordata[:, sensor].norm(dim=-1) > 1e-8

    def _entity(self, role: str) -> GpuEntityStateV2:
        body_id = self.role_ids[role]
        if body_id is None:
            zeros = torch.zeros(self.num_envs, 3, device=self.device)
            rotation = torch.eye(3, device=self.device)[None].repeat(self.num_envs, 1, 1)
            return GpuEntityStateV2(False, zeros, rotation, zeros, zeros,
                                    self.role_extents[role][None].repeat(self.num_envs, 1))
        return GpuEntityStateV2(
            True,
            self.sim.data.xpos[:, body_id],
            _quat_to_matrix(self.sim.data.xquat[:, body_id]),
            self.sim.data.sensordata[:, self.sensors.linear_velocity[role]],
            self.sim.data.sensordata[:, self.sensors.angular_velocity[role]],
            self.role_extents[role][None].repeat(self.num_envs, 1),
        )

    def _articulation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = torch.zeros(self.num_envs, 8, device=self.device)
        raw = torch.zeros(self.num_envs, 2, device=self.device)
        present = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        for slot in range(len(self.articulation_qpos)):
            q = self.sim.data.qpos[:, self.articulation_qpos[slot]]
            qd = self.sim.data.qvel[:, self.articulation_qvel[slot]]
            low, high = self.articulation_ranges[slot]
            span = (high - low).clamp_min(1e-3)
            offset = 4 * slot
            encoded[:, offset] = 1.0
            encoded[:, offset + 1] = (2.0 * (q - low) / span - 1.0).clamp(-2.0, 2.0)
            encoded[:, offset + 2] = (qd / 5.0).clamp(-2.0, 2.0)
            encoded[:, offset + 3] = ((self.articulation_targets[slot] - q) / span).clamp(-2.0, 2.0)
            raw[:, slot] = q
            present[:, slot] = True
        return encoded, raw, present

    def actor_observation(self) -> tuple[torch.Tensor, GpuTaskStateV2]:
        data = self.sim.data
        pelvis_pos = data.xpos[:, self.pelvis_id]
        pelvis_rot = _quat_to_matrix(data.xquat[:, self.pelvis_id])
        ankle_height = data.xpos[:, self.ankle_ids, 2].mean(dim=-1)
        primary, destination, auxiliary = (
            self._entity(role) for role in ("primary", "destination", "auxiliary")
        )
        force_world = torch.stack(
            tuple(self._sensor_vectors(hand) for hand in self.sensors.primary_force),
            dim=1,
        )
        contact_forces = torch.einsum("bhki,bij->bhkj", force_world, pelvis_rot)
        primary_contacts = force_world.norm(dim=-1).amax(dim=-1) > 1e-8
        if self.sensors.auxiliary_force is None:
            auxiliary_contacts = torch.zeros_like(primary_contacts)
        else:
            auxiliary_force = torch.stack(
                tuple(self._sensor_vectors(hand) for hand in self.sensors.auxiliary_force),
                dim=1,
            )
            auxiliary_contacts = auxiliary_force.norm(dim=-1).amax(dim=-1) > 1e-8
        hands = tuple(
            GpuHandStateV2(
                data.xpos[:, body_id], _quat_to_matrix(data.xquat[:, body_id]),
                data.sensordata[:, self.sensors.linear_velocity[role]],
                data.sensordata[:, self.sensors.angular_velocity[role]],
                primary_contacts[:, index], auxiliary_contacts[:, index],
            )
            for index, (role, body_id) in enumerate(
                zip(("left_hand", "right_hand"), self.hand_ids, strict=True)
            )
        )
        distances = torch.stack(
            [torch.stack(
                [torch.stack([data.sensordata[:, item] for item in finger], dim=-1).amin(dim=-1)[:, 0]
                 for finger in hand], dim=-1)
             for hand in self.sensors.fingertip_distance],
            dim=1,
        ).clamp_min(0.0)
        hand_support = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.sensors.hand_support_force is not None:
            hand_support = torch.stack(
                [
                    self._sensor_vectors(hand).norm(dim=-1).amax(dim=-1) > 1e-8
                    for hand in self.sensors.hand_support_force
                ],
                dim=-1,
            ).any(dim=-1)
        arm_support_force = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )
        if self.sensors.arm_support_force is not None:
            arm_support_force = torch.stack(
                [
                    self._sensor_vectors(arm).norm(dim=-1).amax(dim=-1)
                    for arm in self.sensors.arm_support_force
                ],
                dim=-1,
            )
        predicates = torch.stack(
            (
                primary_contacts[:, 0], primary_contacts[:, 1],
                auxiliary_contacts[:, 0], auxiliary_contacts[:, 1],
                self._contact(self.sensors.primary_destination_force),
                self._contact(self.sensors.primary_auxiliary_force),
                self._contact(self.sensors.auxiliary_destination_force), hand_support,
            ),
            dim=-1,
        ).float()
        articulation, articulation_raw, articulation_present = self._articulation()
        distal_pos = data.xpos[:, self.distal_ids]
        state = GpuTaskStateV2(
            pelvis_pos, pelvis_rot, pelvis_pos[:, 2] - ankle_height, hands,
            primary, destination, auxiliary, contact_forces, distal_pos, distances,
            predicates, arm_support_force, articulation, articulation_raw,
            articulation_present,
            self.initial_primary_pos, self.initial_auxiliary_pos,
        )
        joint_pos = data.qpos[:, self.qpos_indices]
        joint_vel = data.qvel[:, self.qvel_indices]
        gravity = _to_local(
            pelvis_rot,
            torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1),
        )
        pelvis_lin = data.sensordata[:, self.sensors.linear_velocity["pelvis"]]
        pelvis_ang = data.sensordata[:, self.sensors.angular_velocity["pelvis"]]
        root = torch.cat((_to_local(pelvis_rot, pelvis_lin),
                          _to_local(pelvis_rot, pelvis_ang)), dim=-1)
        root = torch.cat((gravity, root, state.pelvis_height[:, None]), dim=-1)
        hand_features = []
        for hand in hands:
            hand_features.extend((
                _to_local(pelvis_rot, hand.pos_w - pelvis_pos),
                _rotation_6d(pelvis_rot.transpose(1, 2) @ hand.rot_w),
                _to_local(pelvis_rot, hand.lin_vel_w),
                _to_local(pelvis_rot, hand.ang_vel_w),
            ))
        entity_features = []
        for entity in (primary, destination, auxiliary):
            if entity.present:
                entity_features.extend((
                    torch.ones(self.num_envs, 1, device=self.device),
                    _to_local(pelvis_rot, entity.pos_w - pelvis_pos),
                    _rotation_6d(pelvis_rot.transpose(1, 2) @ entity.rot_w),
                    _to_local(pelvis_rot, entity.lin_vel_w),
                    _to_local(pelvis_rot, entity.ang_vel_w), entity.extents,
                ))
            else:
                entity_features.append(torch.zeros(self.num_envs, 19, device=self.device))
        primary_in_hands = []
        for hand in hands:
            if primary.present:
                inverse = hand.rot_w.transpose(1, 2)
                primary_in_hands.extend((
                    torch.einsum("bij,bj->bi", inverse, primary.pos_w - hand.pos_w),
                    _rotation_6d(inverse @ primary.rot_w),
                ))
            else:
                primary_in_hands.append(torch.zeros(self.num_envs, 9, device=self.device))
        if primary.present and destination.present:
            primary_to_destination = torch.cat((
                _to_local(pelvis_rot, destination.pos_w - primary.pos_w),
                _rotation_6d(pelvis_rot.transpose(1, 2) @ destination.rot_w),
            ), dim=-1)
        else:
            primary_to_destination = torch.zeros(self.num_envs, 9, device=self.device)
        family = torch.zeros(self.num_envs, 6, device=self.device)
        family[:, self.family_index] = 1.0
        stage = torch.zeros(self.num_envs, 8, device=self.device)
        stage.scatter_(1, self.stage_index[:, None], 1.0)
        context = torch.cat((family, stage, self.stage_progress[:, None]), dim=-1)
        observation = torch.cat((
            joint_pos, joint_vel, root, self.previous_action, *hand_features,
            *entity_features, contact_forces.flatten(1), distances.flatten(1),
            *primary_in_hands, primary_to_destination, predicates, articulation,
            context,
        ), dim=-1).float()
        if observation.shape != (self.num_envs, ACTOR_OBS_V2_DIM):
            raise RuntimeError(f"V2 actor observation shape is {observation.shape}")
        return observation, state
