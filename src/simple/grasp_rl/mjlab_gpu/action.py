"""Piecewise-affine 36-D action decoding on the simulation CUDA device."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from simple.grasp_rl.schema import ACTION_DIM, JOINT_NAMES

if TYPE_CHECKING:
    from simple.grasp_rl.mjlab_gpu.simulation import GpuSimulation


def sonic_upper_mappings(upper_names: list[str]) -> tuple[list[int], list[int]]:
    """Resolve Sonic tracker inputs and simulator outputs by joint name."""

    if len(upper_names) != 31 or len(set(upper_names)) != len(upper_names):
        raise ValueError("Sonic upper joint names must contain 31 unique entries")
    required = {
        *JOINT_NAMES[15:],
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    }
    if set(upper_names) != required:
        raise ValueError("Sonic upper joint names do not match the control schema")
    action_index_by_name = {
        **dict(zip(JOINT_NAMES[15:22], range(14, 21), strict=True)),
        **dict(zip(JOINT_NAMES[22:29], range(21, 28), strict=True)),
        **dict(zip(JOINT_NAMES[29:36], (0, 1, 2, 5, 6, 3, 4), strict=True)),
        **dict(zip(JOINT_NAMES[36:43], range(7, 14), strict=True)),
        "waist_yaw_joint": 30,
        "waist_roll_joint": 28,
        "waist_pitch_joint": 29,
    }
    action_indices = [action_index_by_name[name] for name in upper_names]
    output_mapping = [upper_names.index(name) for name in JOINT_NAMES[15:]]
    return action_indices, output_mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracker_hand_targets(physical_action: torch.Tensor) -> torch.Tensor:
    """Map public tracker hand order to ``JOINT_NAMES[29:43]`` order."""

    if physical_action.shape[-1] < 14:
        raise ValueError("physical_action must contain both seven-DoF hands")
    return torch.cat(
        (
            physical_action[..., 0:3],
            physical_action[..., 5:7],
            physical_action[..., 3:5],
            physical_action[..., 7:14],
        ),
        dim=-1,
    )


class GpuActionTransform:
    """Batched legacy decoder with demonstrated slew and optional 1-step delay."""

    def __init__(
        self,
        *,
        center: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        max_delta: torch.Tensor,
        initial_action: torch.Tensor,
        num_envs: int,
        device: str,
    ):
        self.device = device
        self.num_envs = int(num_envs)
        values = {}
        for name, source in (
            ("center", center),
            ("low", low),
            ("high", high),
            ("max_delta", max_delta),
            ("initial_action", initial_action),
        ):
            value = torch.as_tensor(source, dtype=torch.float32, device=device)
            if value.shape != (ACTION_DIM,):
                raise ValueError(f"{name} must have shape ({ACTION_DIM},)")
            values[name] = value
        self.center = values["center"]
        self.low = values["low"]
        self.high = values["high"]
        self.max_delta = values["max_delta"]
        self.initial_action = values["initial_action"]
        if torch.any(self.low > self.center) or torch.any(self.center > self.high):
            raise ValueError("Action bounds must satisfy low <= center <= high")
        if torch.any(self.max_delta <= 0):
            raise ValueError("All action slew limits must be positive")
        self.previous_action = self.initial_action.expand(self.num_envs, -1).clone()
        self.pending_action = self.previous_action.clone()

    @classmethod
    def from_frozen_bundle(cls, gpu: GpuSimulation) -> "GpuActionTransform":
        manifest = gpu.bundle.manifest
        path = (gpu.bundle.root / manifest["action_transform"]).resolve()
        if not path.is_relative_to(gpu.bundle.root):
            raise ValueError("Action transform path escapes frozen bundle")
        if _sha256(path) != manifest["action_transform_sha256"]:
            raise ValueError("Action transform hash mismatch")
        with np.load(path, allow_pickle=False) as saved:
            return cls(
                center=saved["center"],
                low=saved["low"],
                high=saved["high"],
                max_delta=saved["max_delta"],
                initial_action=torch.tensor(
                    manifest["reset"]["previous_physical_action"]
                ),
                num_envs=gpu.sim.num_envs,
                device=gpu.sim.device,
            )

    @property
    def action_span(self) -> torch.Tensor:
        return self.high - self.low

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        self.previous_action[indices] = self.initial_action
        self.pending_action[indices] = self.initial_action

    @torch.no_grad()
    def decode(
        self,
        raw_action: torch.Tensor,
        delay_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.num_envs, ACTION_DIM)
        if raw_action.shape != expected:
            raise ValueError(f"raw_action must have shape {expected}")
        if str(raw_action.device) != self.device:
            raise ValueError("raw_action must be on the transform device")
        clipped = raw_action.clamp(-1.0, 1.0)
        decoded = torch.where(
            clipped >= 0.0,
            self.center + clipped * (self.high - self.center),
            self.center + clipped * (self.center - self.low),
        )
        decoded = torch.maximum(
            torch.minimum(decoded, self.previous_action + self.max_delta),
            self.previous_action - self.max_delta,
        ).clamp(self.low, self.high)
        if delay_steps is None:
            executed = decoded
        else:
            if delay_steps.shape != (self.num_envs,):
                raise ValueError("delay_steps must have one entry per environment")
            if str(delay_steps.device) != self.device:
                raise ValueError("delay_steps must be on the transform device")
            if torch.any((delay_steps < 0) | (delay_steps > 1)):
                raise ValueError("Only zero or one action-delay step is supported")
            executed = torch.where(
                delay_steps.to(torch.bool)[:, None], self.pending_action, decoded
            )
        self.pending_action.copy_(decoded)
        self.previous_action.copy_(executed)
        return executed

    @torch.no_grad()
    def encode(self, physical_action: torch.Tensor) -> torch.Tensor:
        if physical_action.shape[-1] != ACTION_DIM:
            raise ValueError("physical_action has the wrong final dimension")
        clipped = physical_action.clamp(self.low, self.high)
        offset = clipped - self.center
        raw = torch.where(
            offset >= 0.0,
            offset / (self.high - self.center).clamp_min(1e-6),
            offset / (self.center - self.low).clamp_min(1e-6),
        )
        return raw.clamp(-1.0, 1.0)
