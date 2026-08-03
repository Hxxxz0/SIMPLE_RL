"""Batched CUDA implementation of SIMPLE's frozen AMO controller state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.action import tracker_hand_targets
from simple.grasp_rl.mjlab_gpu.simulation import GpuSimulation
from simple.grasp_rl.schema import JOINT_NAMES

LOWER_AND_WAIST_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
ARM_INPUT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
AMO_JOINT_NAMES = LOWER_AND_WAIST_NAMES + ARM_INPUT_NAMES


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_inverse(value: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((value[..., :1], -value[..., 1:]), dim=-1)
    return conjugate / value.square().sum(-1, keepdim=True).clamp_min(1e-12)


def _quat_to_rpy(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = 2 * (w * y - z * x)
    pitch = torch.asin(sin_pitch.clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


class BatchedAmoController:
    """Run the released adapter and AMO TorchScript graphs for every env."""

    def __init__(self, gpu: GpuSimulation):
        self.gpu = gpu
        self.sim = gpu.sim
        self.device = self.sim.device
        if self.device != "cuda:0":
            raise ValueError(
                "The released AMO TorchScript graph hard-codes logical cuda:0; "
                "select the physical GPU with CUDA_VISIBLE_DEVICES"
            )
        controller = gpu.bundle.manifest.get("controller_bundle")
        if gpu.bundle.manifest.get("controller") != "amo" or controller is None:
            raise ValueError("BatchedAmoController requires an AMO asset bundle")
        self.bundle_config = controller
        artifacts = {
            Path(item["bundle_path"]).name: gpu.bundle.root / item["bundle_path"]
            for item in controller["artifacts"]
        }
        self.policy = torch.jit.load(
            str(artifacts["amo_jit.pt"]), map_location=self.device
        ).eval()
        self.adapter = torch.jit.load(
            str(artifacts["adapter_jit.pt"]), map_location=self.device
        ).eval()
        stats = torch.load(
            artifacts["adapter_norm_stats.pt"],
            map_location=self.device,
            weights_only=False,
        )
        self.input_mean = torch.as_tensor(
            stats["input_mean"], dtype=torch.float32, device=self.device
        )
        self.input_std = torch.as_tensor(
            stats["input_std"], dtype=torch.float32, device=self.device
        )
        self.output_mean = torch.as_tensor(
            stats["output_mean"], dtype=torch.float32, device=self.device
        )
        self.output_std = torch.as_tensor(
            stats["output_std"], dtype=torch.float32, device=self.device
        )
        self.action_scale = float(controller["action_scale"])
        self.gait_frequency = float(controller["gait_frequency"])
        self._build_model_indices()
        self._load_initial_state()

    def _build_model_indices(self) -> None:
        model = self.sim.mj_model
        qpos = []
        qvel = []
        for name in AMO_JOINT_NAMES:
            joint_id = model.joint(name).id
            qpos.append(int(model.jnt_qposadr[joint_id]))
            qvel.append(int(model.jnt_dofadr[joint_id]))
        self.qpos_indices = torch.tensor(qpos, dtype=torch.long, device=self.device)
        self.qvel_indices = torch.tensor(qvel, dtype=torch.long, device=self.device)

        logical_qpos = []
        logical_qvel = []
        logical_actuators = []
        for name in JOINT_NAMES:
            joint_id = model.joint(name).id
            logical_qpos.append(int(model.jnt_qposadr[joint_id]))
            logical_qvel.append(int(model.jnt_dofadr[joint_id]))
            logical_actuators.append(int(model.actuator(name).id))
        self.logical_qpos_indices = torch.tensor(
            logical_qpos, dtype=torch.long, device=self.device
        )
        self.logical_qvel_indices = torch.tensor(
            logical_qvel, dtype=torch.long, device=self.device
        )
        self.logical_actuator_indices = torch.tensor(
            logical_actuators, dtype=torch.long, device=self.device
        )
        parameters = self.bundle_config["joint_parameters"]
        if tuple(parameters["names"]) != JOINT_NAMES:
            raise ValueError("AMO joint parameter schema does not match JOINT_NAMES")
        self.stiffness = torch.tensor(
            parameters["stiffness"], dtype=torch.float32, device=self.device
        )
        self.damping = torch.tensor(
            parameters["damping"], dtype=torch.float32, device=self.device
        )
        self.torque_limits = torch.tensor(
            parameters["torque_limits"], dtype=torch.float32, device=self.device
        )
        self.actuator_strength_scale = torch.ones(
            self.sim.num_envs, len(JOINT_NAMES), device=self.device
        )

        orientation_id = model.sensor("orientation").id
        angular_velocity_id = model.sensor("angular-velocity").id
        if int(model.sensor_dim[orientation_id]) != 4:
            raise ValueError("orientation sensor must be a quaternion")
        if int(model.sensor_dim[angular_velocity_id]) != 3:
            raise ValueError("angular-velocity sensor must have three values")
        self.orientation_slice = slice(
            int(model.sensor_adr[orientation_id]),
            int(model.sensor_adr[orientation_id]) + 4,
        )
        self.angular_velocity_slice = slice(
            int(model.sensor_adr[angular_velocity_id]),
            int(model.sensor_adr[angular_velocity_id]) + 3,
        )

    @staticmethod
    def metadata() -> dict[str, object]:
        return {"backend": "amo", "mapping_schema_version": 1}

    def _load_initial_state(self) -> None:
        state_path = self.gpu.bundle.root / self.bundle_config["state_file"]
        with np.load(state_path, allow_pickle=False) as saved:
            self.fixture = {
                name: torch.as_tensor(saved[name], device=self.device)
                for name in saved.files
                if name.startswith("parity_")
            }
            self._initial = {
                name: torch.as_tensor(saved[name], device=self.device)
                for name in (
                    "initial_quat",
                    "gait_cycle",
                    "last_action_for_policy",
                    "proprio_history",
                    "extra_history",
                    "last_commands",
                    "in_place_stand_flag",
                    "last_target_yaw",
                    "target_yaw",
                )
            }
        self.reset()

    def _expanded(self, name: str) -> torch.Tensor:
        source = self._initial[name]
        return source.unsqueeze(0).expand(self.sim.num_envs, *source.shape).clone()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if not hasattr(self, "initial_quat"):
            self.initial_quat = self._expanded("initial_quat")
            self.gait_cycle = self._expanded("gait_cycle")
            self.last_action_for_policy = self._expanded("last_action_for_policy")
            self.proprio_history = self._expanded("proprio_history")
            self.extra_history = self._expanded("extra_history")
            self.last_commands = self._expanded("last_commands")
            self.in_place_stand_flag = self._expanded("in_place_stand_flag")
            self.last_target_yaw = self._expanded("last_target_yaw")
            self.target_yaw = self._expanded("target_yaw")
            return
        indices = (
            torch.arange(self.sim.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        for name in (
            "initial_quat",
            "gait_cycle",
            "last_action_for_policy",
            "proprio_history",
            "extra_history",
            "last_commands",
            "in_place_stand_flag",
            "last_target_yaw",
            "target_yaw",
        ):
            target = getattr(self, name)
            source = self._initial[name]
            target[indices] = source

    @staticmethod
    def tracker_command(action: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                action[:, 32],
                action[:, 35],
                action[:, 33],
                action[:, 31] - 0.75,
                action[:, 30],
                action[:, 29],
                action[:, 28],
                action[:, 34],
            ),
            dim=-1,
        )

    @torch.no_grad()
    def compute_pd_target(self, physical_action: torch.Tensor) -> torch.Tensor:
        if physical_action.shape != (self.sim.num_envs, 36):
            raise ValueError(
                f"physical_action must have shape ({self.sim.num_envs}, 36)"
            )
        if not physical_action.is_cuda or str(physical_action.device) != self.device:
            raise ValueError(
                "AMO physical_action must be on the simulation CUDA device"
            )
        command = self.tracker_command(physical_action)
        dof_pos = self.sim.data.qpos[:, self.qpos_indices]
        dof_vel = self.sim.data.qvel[:, self.qvel_indices]
        orientation = self.sim.data.sensordata[:, self.orientation_slice]
        relative_quat = _quat_multiply(_quat_inverse(self.initial_quat), orientation)
        rpy = _quat_to_rpy(relative_quat)
        angular_velocity = self.sim.data.sensordata[:, self.angular_velocity_slice]

        target_yaw = command[:, 1]
        dyaw = (
            torch.remainder(rpy[:, 2] - target_yaw + torch.pi, 2 * torch.pi) - torch.pi
        )
        dyaw = torch.where(self.in_place_stand_flag[:, 0], torch.zeros_like(dyaw), dyaw)
        gait_observation = torch.sin(self.gait_cycle * (2 * torch.pi))

        adapter_raw = torch.cat(
            (
                (0.75 + command[:, 3]).unsqueeze(-1),
                command[:, 4:7],
                dof_pos[:, 15:],
            ),
            dim=-1,
        )
        adapter_input = (adapter_raw - self.input_mean) / (self.input_std + 1e-8)
        adapter_output = self.adapter(adapter_input)
        adapter_output = adapter_output * self.output_std + self.output_mean

        obs_prop = torch.cat(
            (
                angular_velocity * 0.25,
                rpy[:, :2],
                torch.sin(dyaw).unsqueeze(-1),
                torch.cos(dyaw).unsqueeze(-1),
                dof_pos,
                dof_vel * 0.05,
                self.last_action_for_policy,
                gait_observation,
                adapter_output,
            ),
            dim=-1,
        )
        if obs_prop.shape[1] != 93:
            raise RuntimeError(f"AMO proprio shape is {tuple(obs_prop.shape)}")
        old_history = self.proprio_history.flatten(1)
        height = (0.75 + command[:, 3]).unsqueeze(-1)
        obs_demo = torch.cat(
            (
                dof_pos[:, 15:],
                command[:, 0:1],
                command[:, 2:3],
                torch.zeros_like(height),
                command[:, 4:7],
                height.expand(-1, 3),
            ),
            dim=-1,
        )
        policy_obs = torch.cat(
            (
                obs_prop,
                obs_demo,
                torch.zeros((self.sim.num_envs, 3), device=self.device),
                old_history,
            ),
            dim=-1,
        ).float()
        if policy_obs.shape[1] != 1043:
            raise RuntimeError(f"AMO policy input shape is {tuple(policy_obs.shape)}")

        stand = (
            (command[:, 0].abs() < 0.1)
            & (command[:, 2].abs() < 0.1)
            & (command[:, 7] < 0.5)
        )
        self.proprio_history = torch.cat(
            (self.proprio_history[:, 1:], obs_prop[:, None, :]), dim=1
        )
        self.extra_history = torch.cat(
            (self.extra_history[:, 1:], obs_prop[:, None, :]), dim=1
        )
        extra_history = self.extra_history.flatten(1).float()
        raw_action = self.policy(policy_obs, extra_history).clamp(-40.0, 40.0)
        self.last_action_for_policy = torch.cat(
            (raw_action, dof_pos[:, 15:] / self.action_scale), dim=-1
        )
        pd_target = raw_action * self.action_scale

        self.gait_cycle = torch.remainder(
            self.gait_cycle + 0.02 * self.gait_frequency, 1.0
        )
        close_to_quarter = (self.gait_cycle - 0.25).abs() < 0.05
        reset_stand = stand & close_to_quarter.any(dim=-1)
        self.gait_cycle = torch.where(
            reset_stand[:, None],
            torch.full_like(self.gait_cycle, 0.25),
            self.gait_cycle,
        )
        start_walking = (~stand) & close_to_quarter.all(dim=-1)
        walking_phase = torch.tensor(
            [0.25, 0.75], device=self.device, dtype=torch.float32
        )
        self.gait_cycle = torch.where(
            start_walking[:, None], walking_phase, self.gait_cycle
        )
        self.in_place_stand_flag[:, 0] = stand
        self.last_commands = command
        self.last_target_yaw[:, 0] = target_yaw
        self.target_yaw[:, 0] = target_yaw
        self.last_adapter_input = adapter_input
        self.last_adapter_output = adapter_output
        self.last_obs_prop = obs_prop
        self.last_policy_obs = policy_obs
        self.last_raw_action = raw_action
        return pd_target

    @torch.no_grad()
    def apply_physical_action(self, physical_action: torch.Tensor) -> torch.Tensor:
        """Apply one 20 ms tracker command using ten 2 ms GPU substeps."""

        pd_target = self.compute_pd_target(physical_action)
        pd_target = pd_target.clone()
        # The public action stores torso roll/pitch/yaw as 28/29/30 while AMO
        # controls waist yaw/roll/pitch in slots 12/13/14.
        pd_target[:, 12:15] = physical_action[:, (30, 28, 29)]

        arm_actuators = self.logical_actuator_indices[15:29]
        self.sim.data.ctrl[:, arm_actuators] = physical_action[:, 14:28]

        hand_qpos = self.sim.data.qpos[:, self.logical_qpos_indices[29:43]]
        hand_qvel = self.sim.data.qvel[:, self.logical_qvel_indices[29:43]]
        hand_target = tracker_hand_targets(physical_action)
        hand_torque = (
            self.stiffness[29:43] * (hand_target - hand_qpos)
            - self.damping[29:43] * hand_qvel
        )
        hand_limits = self.torque_limits[29:43] * self.actuator_strength_scale[:, 29:43]
        hand_torque = torch.clamp(hand_torque, -hand_limits, hand_limits)
        self.sim.data.ctrl[:, self.logical_actuator_indices[29:43]] = hand_torque

        lower_qpos_indices = self.logical_qpos_indices[:15]
        lower_qvel_indices = self.logical_qvel_indices[:15]
        lower_actuators = self.logical_actuator_indices[:15]
        for _ in range(10):
            qpos = self.sim.data.qpos[:, lower_qpos_indices]
            qvel = self.sim.data.qvel[:, lower_qvel_indices]
            torque = self.stiffness[:15] * (pd_target - qpos) - self.damping[:15] * qvel
            limits = self.torque_limits[:15] * self.actuator_strength_scale[:, :15]
            self.sim.data.ctrl[:, lower_actuators] = torch.clamp(
                torque, -limits, limits
            )
            self.sim.step()
        return pd_target
