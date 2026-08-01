"""Batched 192-D legacy actor observation computed entirely on CUDA."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simple.grasp_rl.mjlab_gpu.simulation import GpuSimulation
from simple.grasp_rl.schema import ACTOR_OBS_DIM, JOINT_NAMES


def _quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _to_local(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bj->bi", rotation.transpose(1, 2), vector)


def _rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    return torch.cat((rotation[:, :, 0], rotation[:, :, 1]), dim=-1)


@dataclass
class GpuContactState:
    link_forces_pelvis: torch.Tensor
    link_force_magnitudes: torch.Tensor
    group_forces: torch.Tensor
    object_contact_center_w: torch.Tensor
    has_object_contact_center: torch.Tensor
    hand_table_force: torch.Tensor

    @property
    def group_contacts(self) -> torch.Tensor:
        return self.group_forces > 2.0

    @property
    def is_grasp(self) -> torch.Tensor:
        contacts = self.group_contacts
        return contacts[:, 0] & (contacts[:, 1] | contacts[:, 2])


@dataclass
class GpuTargetState:
    object_pos_w: torch.Tensor
    object_rot_w: torch.Tensor
    object_lin_vel_w: torch.Tensor
    object_ang_vel_w: torch.Tensor
    hand_pos_w: torch.Tensor
    hand_rot_w: torch.Tensor
    hand_lin_vel_w: torch.Tensor
    wrist_pos_w: torch.Tensor
    wrist_lin_vel_w: torch.Tensor
    distal_pos_w: torch.Tensor
    fingertip_surface_distances: torch.Tensor
    table_pos_w: torch.Tensor
    table_rot_w: torch.Tensor
    pelvis_pos_w: torch.Tensor
    pelvis_rot_w: torch.Tensor
    pelvis_height: torch.Tensor
    contact: GpuContactState


class GpuLegacyState:
    def __init__(self, gpu: GpuSimulation):
        if gpu.sensors is None:
            raise ValueError("GpuLegacyState requires the legacy GPU sensor schema")
        self.gpu = gpu
        self.sim = gpu.sim
        self.sensors = gpu.sensors
        self.device = self.sim.device
        model = self.sim.mj_model
        self.pelvis_id = model.body("pelvis").id
        self.ankle_ids = torch.tensor(
            [
                model.body("left_ankle_roll_link").id,
                model.body("right_ankle_roll_link").id,
            ],
            dtype=torch.long,
            device=self.device,
        )
        roles = gpu.bundle.manifest["roles"]
        self.target_id = model.body(roles["primary"]).id
        self.table_id = model.body(roles["destination"]).id
        self.wrist_id = model.body("right_wrist_yaw_link").id
        self.distal_ids = torch.tensor(
            [
                model.body("right_hand_thumb_2_link").id,
                model.body("right_hand_index_1_link").id,
                model.body("right_hand_middle_1_link").id,
            ],
            dtype=torch.long,
            device=self.device,
        )
        qpos_indices = []
        qvel_indices = []
        for name in JOINT_NAMES:
            joint_id = model.joint(name).id
            qpos_indices.append(int(model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        self.qpos_indices = torch.tensor(
            qpos_indices, dtype=torch.long, device=self.device
        )
        self.qvel_indices = torch.tensor(
            qvel_indices, dtype=torch.long, device=self.device
        )
        reset = gpu.bundle.manifest["reset"]
        self._initial_object_pos = torch.tensor(
            reset["initial_object_pos"], dtype=torch.float32, device=self.device
        )
        self._goal_pos = torch.tensor(
            reset["goal_pos"], dtype=torch.float32, device=self.device
        )
        self._goal_offset = self._goal_pos - self._initial_object_pos
        self._initial_previous_action = torch.tensor(
            reset["previous_physical_action"],
            dtype=torch.float32,
            device=self.device,
        )
        self.initial_object_pos = self._initial_object_pos[None].repeat(
            self.sim.num_envs, 1
        )
        self.goal_pos = self._goal_pos[None].repeat(self.sim.num_envs, 1)
        self.previous_action = self._initial_previous_action[None].repeat(
            self.sim.num_envs, 1
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        self.initial_object_pos[indices] = self._initial_object_pos
        self.goal_pos[indices] = self._goal_pos
        self.previous_action[indices] = self._initial_previous_action

    def set_previous_action(self, action: torch.Tensor) -> None:
        if action.shape != self.previous_action.shape:
            raise ValueError("Previous action shape mismatch")
        self.previous_action.copy_(action)

    def sync_episode_origin(self, env_ids: torch.Tensor) -> None:
        """Anchor lift/workspace truth to the randomized reset object pose."""

        current = self.sim.data.xpos[env_ids, self.target_id]
        self.initial_object_pos[env_ids] = current
        self.goal_pos[env_ids] = current + self._goal_offset

    def _sensor_vectors(self, slices: tuple[slice, ...]) -> torch.Tensor:
        return torch.stack(
            [self.sim.data.sensordata[:, item] for item in slices], dim=1
        )

    def _contacts(self, pelvis_rotation: torch.Tensor) -> GpuContactState:
        forces_world = self._sensor_vectors(self.sensors.object_force)
        magnitudes = forces_world.norm(dim=-1)
        forces_pelvis = torch.einsum("bki,bij->bkj", forces_world, pelvis_rotation)
        group_forces = torch.stack(
            (
                magnitudes[:, 1:4].amax(dim=-1),
                magnitudes[:, 4:6].amax(dim=-1),
                magnitudes[:, 6:8].amax(dim=-1),
            ),
            dim=-1,
        )
        positions = self._sensor_vectors(self.sensors.object_position)
        contact_mask = magnitudes > 1e-8
        count = contact_mask.sum(dim=-1, keepdim=True)
        center = (positions * contact_mask[..., None]).sum(dim=1) / count.clamp_min(1)
        center = torch.where(count > 0, center, torch.zeros_like(center))
        table_forces = self._sensor_vectors(self.sensors.table_force)
        hand_table_force = table_forces.norm(dim=-1).sum(dim=-1)
        return GpuContactState(
            link_forces_pelvis=forces_pelvis.clamp(-100.0, 100.0),
            link_force_magnitudes=magnitudes,
            group_forces=group_forces,
            object_contact_center_w=center,
            has_object_contact_center=count[:, 0] > 0,
            hand_table_force=hand_table_force,
        )

    def target_state(self) -> GpuTargetState:
        data = self.sim.data
        pelvis_pos = data.xpos[:, self.pelvis_id]
        object_pos = data.xpos[:, self.target_id]
        table_pos = data.xpos[:, self.table_id]
        wrist_pos = data.xpos[:, self.wrist_id]
        distal_pos = data.xpos[:, self.distal_ids]
        pelvis_rot = _quat_to_matrix(data.xquat[:, self.pelvis_id])
        object_rot = _quat_to_matrix(data.xquat[:, self.target_id])
        table_rot = _quat_to_matrix(data.xquat[:, self.table_id])
        wrist_rot = _quat_to_matrix(data.xquat[:, self.wrist_id])
        object_lin = data.sensordata[:, self.sensors.target_linear_velocity]
        object_ang = data.sensordata[:, self.sensors.target_angular_velocity]
        wrist_lin = data.sensordata[:, self.sensors.wrist_linear_velocity]
        distal_lin = self._sensor_vectors(self.sensors.distal_linear_velocity)
        ankle_height = data.xpos[:, self.ankle_ids, 2].mean(dim=-1)
        distances = torch.stack(
            [
                torch.stack([data.sensordata[:, item] for item in finger], dim=-1).amin(
                    dim=-1
                )[:, 0]
                for finger in self.sensors.fingertip_distance
            ],
            dim=-1,
        ).clamp_min(0.0)
        return GpuTargetState(
            object_pos_w=object_pos,
            object_rot_w=object_rot,
            object_lin_vel_w=object_lin,
            object_ang_vel_w=object_ang,
            hand_pos_w=distal_pos.mean(dim=1),
            hand_rot_w=wrist_rot,
            hand_lin_vel_w=distal_lin.mean(dim=1),
            wrist_pos_w=wrist_pos,
            wrist_lin_vel_w=wrist_lin,
            distal_pos_w=distal_pos,
            fingertip_surface_distances=distances,
            table_pos_w=table_pos,
            table_rot_w=table_rot,
            pelvis_pos_w=pelvis_pos,
            pelvis_rot_w=pelvis_rot,
            pelvis_height=pelvis_pos[:, 2] - ankle_height,
            contact=self._contacts(pelvis_rot),
        )

    def actor_observation(self) -> tuple[torch.Tensor, GpuTargetState]:
        state = self.target_state()
        joint_pos = self.sim.data.qpos[:, self.qpos_indices]
        joint_vel = self.sim.data.qvel[:, self.qvel_indices]
        gravity = _to_local(
            state.pelvis_rot_w,
            torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(
                self.sim.num_envs, -1
            ),
        )
        pelvis_lin = self.sim.data.sensordata[:, self.sensors.pelvis_linear_velocity]
        pelvis_ang = self.sim.data.sensordata[:, self.sensors.pelvis_angular_velocity]
        object_pos_b = _to_local(
            state.pelvis_rot_w, state.object_pos_w - state.pelvis_pos_w
        )
        object_rot_b = state.pelvis_rot_w.transpose(1, 2) @ state.object_rot_w
        object_lin_b = _to_local(state.pelvis_rot_w, state.object_lin_vel_w)
        object_ang_b = _to_local(state.pelvis_rot_w, state.object_ang_vel_w)
        hand_rotation_inv = state.hand_rot_w.transpose(1, 2)
        object_pos_hand = torch.einsum(
            "bij,bj->bi", hand_rotation_inv, state.object_pos_w - state.hand_pos_w
        )
        object_rot_hand = hand_rotation_inv @ state.object_rot_w
        table_pos_b = _to_local(
            state.pelvis_rot_w, state.table_pos_w - state.pelvis_pos_w
        )
        table_rot_b = state.pelvis_rot_w.transpose(1, 2) @ state.table_rot_w
        goal_delta_b = _to_local(state.pelvis_rot_w, self.goal_pos - state.object_pos_w)
        observation = torch.cat(
            (
                joint_pos,
                joint_vel,
                gravity,
                _to_local(state.pelvis_rot_w, pelvis_lin),
                _to_local(state.pelvis_rot_w, pelvis_ang),
                state.pelvis_height[:, None],
                self.previous_action,
                object_pos_b,
                _rotation_6d(object_rot_b),
                object_lin_b,
                object_ang_b,
                object_pos_hand,
                _rotation_6d(object_rot_hand),
                table_pos_b,
                _rotation_6d(table_rot_b),
                state.contact.link_forces_pelvis.flatten(1),
                goal_delta_b,
            ),
            dim=-1,
        ).float()
        if observation.shape != (self.sim.num_envs, ACTOR_OBS_DIM):
            raise RuntimeError(
                f"Actor observation has shape {tuple(observation.shape)}"
            )
        return observation, state
