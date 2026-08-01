"""Batched CUDA implementation of SIMPLE's frozen Sonic controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from simple.grasp_rl.mjlab_gpu.simulation import GpuSimulation
from simple.grasp_rl.schema import JOINT_NAMES


class _SonicPolicyNetwork(nn.Module):
    """Exact Torch form of the reviewed 516-D Sonic ONNX MLP graph."""

    def __init__(self, path: Path, device: str):
        super().__init__()
        weights = torch.load(path, map_location=device, weights_only=True)
        self.estimator = nn.ModuleList(
            (nn.Linear(516, 256), nn.Linear(256, 256), nn.Linear(256, 35))
        )
        self.actor = nn.ModuleList(
            (nn.Linear(121, 512), nn.Linear(512, 256), nn.Linear(256, 256),
             nn.Linear(256, 15))
        )
        for prefix, layers, source_indices in (
            ("estimator", self.estimator, (0, 2, 4)),
            ("actor", self.actor, (0, 2, 4, 6)),
        ):
            for layer, source_index in zip(layers, source_indices, strict=True):
                layer.weight.data.copy_(weights[f"{prefix}.{source_index}.weight"])
                layer.bias.data.copy_(weights[f"{prefix}.{source_index}.bias"])
        self.to(device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        estimate = observation
        for layer in self.estimator[:-1]:
            estimate = F.elu(layer(estimate))
        estimate = self.estimator[-1](estimate)
        latent = estimate[:, 3:]
        latent = latent / latent.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        actor_input = torch.cat((observation[:, -86:], estimate[:, :3], latent), dim=-1)
        action = actor_input
        for layer in self.actor[:-1]:
            action = F.elu(layer(action))
        return self.actor[-1](action)


def _quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
            2 * (x * z + y * w), 2 * (x * y + z * w),
            1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _yaw(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class BatchedSonicController:
    """Run Sonic history, balance/walk MLPs and PD control for every world."""

    def __init__(self, gpu: GpuSimulation):
        self.gpu = gpu
        self.sim = gpu.sim
        self.device = self.sim.device
        bundle = gpu.bundle.manifest.get("controller_bundle")
        if gpu.bundle.manifest.get("controller") != "sonic_wbc" or bundle is None:
            raise ValueError("BatchedSonicController requires a Sonic asset bundle")
        self.bundle_config = bundle
        artifacts = {item["role"]: gpu.bundle.root / item["bundle_path"] for item in bundle["artifacts"]}
        self.balance_policy = _SonicPolicyNetwork(artifacts["balance"], self.device)
        self.walk_policy = _SonicPolicyNetwork(artifacts["walk"], self.device)
        self._build_indices()
        self._load_parameters()
        self._load_initial_state()

    def _build_indices(self) -> None:
        model = self.sim.mj_model
        qpos, qvel, actuators = [], [], []
        for name in JOINT_NAMES:
            joint_id = model.joint(name).id
            qpos.append(int(model.jnt_qposadr[joint_id]))
            qvel.append(int(model.jnt_dofadr[joint_id]))
            actuators.append(int(model.actuator(name).id))
        self.logical_qpos_indices = torch.tensor(qpos, dtype=torch.long, device=self.device)
        self.logical_qvel_indices = torch.tensor(qvel, dtype=torch.long, device=self.device)
        self.logical_actuator_indices = torch.tensor(actuators, dtype=torch.long, device=self.device)
        root_joint = model.joint("floating_base_joint").id
        self.root_qpos_address = int(model.jnt_qposadr[root_joint])
        self.root_qvel_address = int(model.jnt_dofadr[root_joint])

    def _load_parameters(self) -> None:
        bundle = self.bundle_config
        parameters = bundle["joint_parameters"]
        if tuple(parameters["names"]) != JOINT_NAMES:
            raise ValueError("Sonic joint parameter schema does not match JOINT_NAMES")
        def tensor(value: object) -> torch.Tensor:
            return torch.tensor(value, dtype=torch.float32, device=self.device)
        self.stiffness = tensor(parameters["stiffness"])
        self.damping = tensor(parameters["damping"])
        self.torque_limits = tensor(parameters["torque_limits"])
        policy = bundle["policy"]
        defaults = torch.zeros(29, dtype=torch.float32, device=self.device)
        defaults[:15] = tensor(policy["default_angles"])
        self.default_angles = defaults
        self.action_scale = float(policy["action_scale"])
        self.cmd_scale = tensor(policy["cmd_scale"])
        self.dof_pos_scale = float(policy["dof_pos_scale"])
        self.dof_vel_scale = float(policy["dof_vel_scale"])
        self.ang_vel_scale = float(policy["ang_vel_scale"])
        self.control_decimation = int(bundle["control_decimation"])
        if self.control_decimation != 4:
            raise ValueError("Sonic GPU control requires four 5 ms substeps")
        self.upper_action_indices = torch.tensor(
            bundle["upper_action_indices"], dtype=torch.long, device=self.device
        )
        self.upper_output_mapping = torch.tensor(
            bundle["upper_output_mapping"], dtype=torch.long, device=self.device
        )
        upper_names = bundle["upper_names"]
        self.upper_waist_indices = torch.tensor(
            [upper_names.index(name) for name in
             ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")],
            dtype=torch.long,
            device=self.device,
        )
        self.actuator_strength_scale = torch.ones(
            self.sim.num_envs, len(JOINT_NAMES), device=self.device
        )

    def _load_initial_state(self) -> None:
        path = self.gpu.bundle.root / self.bundle_config["state_file"]
        with np.load(path, allow_pickle=False) as saved:
            self._initial = {
                name: torch.as_tensor(saved[name], dtype=torch.float32, device=self.device)
                for name in (
                    "observation_history", "last_lower_action", "command",
                    "base_height", "torso_rpy", "target_yaw", "pending_upper",
                    "pending_navigation", "pending_base_height",
                )
            }
            self.fixture = {
                name: torch.as_tensor(saved[name], dtype=torch.float32, device=self.device)
                for name in ("parity_physical_action", "parity_pd_target")
            }
        self.reset()

    def _expanded(self, name: str) -> torch.Tensor:
        return self._initial[name].unsqueeze(0).expand(
            self.sim.num_envs, *self._initial[name].shape
        ).clone()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        names = (
            "observation_history", "last_lower_action", "command", "base_height",
            "torso_rpy", "target_yaw", "pending_upper", "pending_navigation",
            "pending_base_height",
        )
        if not hasattr(self, "observation_history"):
            for name in names:
                setattr(self, name, self._expanded(name))
            return
        indices = slice(None) if env_ids is None else env_ids
        for name in names:
            getattr(self, name)[indices] = self._initial[name]

    def _single_observation(self) -> torch.Tensor:
        qpos = self.sim.data.qpos[:, self.logical_qpos_indices[:29]]
        qvel = self.sim.data.qvel[:, self.logical_qvel_indices[:29]]
        root = self.root_qpos_address
        velocity = self.root_qvel_address
        quaternion = self.sim.data.qpos[:, root + 3:root + 7]
        omega = self.sim.data.qvel[:, velocity + 3:velocity + 6]
        gravity = torch.einsum(
            "bij,j->bi",
            _quat_to_matrix(quaternion).transpose(1, 2),
            torch.tensor([0.0, 0.0, -1.0], device=self.device),
        )
        observation = torch.cat(
            (
                self.command * self.cmd_scale,
                self.base_height,
                self.torso_rpy,
                omega * self.ang_vel_scale,
                gravity,
                (qpos - self.default_angles) * self.dof_pos_scale,
                qvel * self.dof_vel_scale,
                self.last_lower_action,
            ),
            dim=-1,
        )
        if observation.shape != (self.sim.num_envs, 86):
            raise RuntimeError(f"Sonic single observation shape is {observation.shape}")
        return observation

    def _torso_rpy(self, upper: torch.Tensor) -> torch.Tensor:
        yaw, roll, pitch = upper[:, self.upper_waist_indices].unbind(-1)
        cy, sy = yaw.cos(), yaw.sin()
        cr, sr = roll.cos(), roll.sin()
        cp, sp = pitch.cos(), pitch.sin()
        # Rz(yaw) @ Rx(roll) @ Ry(pitch), matching the frozen Pinocchio chain.
        r20 = -cr * sp
        r21 = sr
        r22 = cr * cp
        r10 = sy * cp + cy * sr * sp
        r00 = cy * cp - sy * sr * sp
        return torch.stack(
            (torch.atan2(r21, r22), torch.asin((-r20).clamp(-1.0, 1.0)),
             torch.atan2(r10, r00)),
            dim=-1,
        )

    @torch.no_grad()
    def compute_pd_target(self, physical_action: torch.Tensor) -> torch.Tensor:
        if physical_action.shape != (self.sim.num_envs, 36):
            raise ValueError("Sonic physical action shape mismatch")
        single = self._single_observation()
        self.observation_history = torch.cat(
            (self.observation_history[:, 1:], single[:, None]), dim=1
        )
        policy_observation = self.observation_history.flatten(1)
        standing = torch.cat((self.command, self.target_yaw), dim=-1).norm(dim=-1) < 0.05
        balance_action = self.balance_policy(policy_observation)
        walk_action = self.walk_policy(policy_observation)
        lower_action = torch.where(standing[:, None], balance_action, walk_action)
        lower_target = lower_action * self.action_scale + self.default_angles[:15]

        current_upper = self.pending_upper.clone()
        current_navigation = self.pending_navigation.clone()
        current_height = self.pending_base_height.clone()
        root = self.root_qpos_address
        current_yaw = _yaw(self.sim.data.qpos[:, root + 3:root + 7])
        target_yaw = current_navigation[:, 3]
        yaw_error = torch.atan2(
            torch.sin(target_yaw - current_yaw), torch.cos(target_yaw - current_yaw)
        )
        yaw_rate = torch.where(
            yaw_error.abs() < 0.01,
            torch.zeros_like(yaw_error),
            (yaw_error / 0.5).clamp(-1.0, 1.0),
        )
        self.command[:, :2] = current_navigation[:, :2]
        self.command[:, 2] = yaw_rate
        self.target_yaw[:, 0] = target_yaw
        self.base_height.copy_(current_height)
        self.torso_rpy.copy_(self._torso_rpy(current_upper))
        self.last_lower_action.copy_(lower_action)
        self.pending_upper.copy_(physical_action[:, self.upper_action_indices])
        self.pending_navigation.copy_(physical_action[:, 32:36])
        self.pending_base_height.copy_(physical_action[:, 31:32])
        return torch.cat((lower_target, current_upper[:, self.upper_output_mapping]), dim=-1)

    @torch.no_grad()
    def apply_physical_action(self, physical_action: torch.Tensor) -> torch.Tensor:
        target = self.compute_pd_target(physical_action)
        for _ in range(self.control_decimation):
            qpos = self.sim.data.qpos[:, self.logical_qpos_indices]
            qvel = self.sim.data.qvel[:, self.logical_qvel_indices]
            torque = self.stiffness * (target - qpos) - self.damping * qvel
            limits = self.torque_limits * self.actuator_strength_scale
            self.sim.data.ctrl[:, self.logical_actuator_indices] = torch.clamp(
                torque, -limits, limits
            )
            self.sim.step()
        return target
