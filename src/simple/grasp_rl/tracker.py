"""36-D policy action normalization and SIMPLE AMO tracker adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simple.core.action import ActionCmd
from simple.grasp_rl.schema import ACTION_DIM


def upper_joints_from_tracker(action: np.ndarray) -> np.ndarray:
    """Convert the public 28-D hand/arm order to ``robot.joint_names[15:]``.

    This is the exact mapping used by ``simple.baselines.act_g1``.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] < 28:
        raise ValueError(f"Expected at least 28 action values, got {action.shape}")
    return np.concatenate(
        [
            action[..., 14:28],
            action[..., 0:3],
            action[..., 5:7],
            action[..., 3:5],
            action[..., 7:14],
        ],
        axis=-1,
    )


def tracker_action_to_cmd(action: np.ndarray, robot) -> ActionCmd:
    """Turn one complete physical 36-D tracker input into an ``ActionCmd``."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action.shape}")

    target_qpos = dict(
        zip(robot.joint_names[15:], upper_joints_from_tracker(action), strict=True)
    )
    waist_qpos = {
        "waist_yaw_joint": float(action[30]),
        "waist_roll_joint": float(action[28]),
        "waist_pitch_joint": float(action[29]),
    }
    command = [
        float(action[32]),
        float(action[35]),
        float(action[33]),
        float(action[31] - 0.75),
        float(action[30]),
        float(action[29]),
        float(action[28]),
        float(action[34]),
    ]
    return ActionCmd(
        "eval_move_actuators",
        target_qpos=target_qpos,
        action_command=command,
        waist_qpos=waist_qpos,
    )


def stand_command() -> ActionCmd:
    return ActionCmd(
        "loco_command",
        command=[0.0] * 8,
        motion_type="stand",
        keep_waist_pose=False,
    )


@dataclass
class ActionTransform:
    """Piecewise affine normalized action decoder with demonstrated slew limits."""

    center: np.ndarray
    low: np.ndarray
    high: np.ndarray
    max_delta: np.ndarray

    def __post_init__(self) -> None:
        for name in ("center", "low", "high", "max_delta"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (ACTION_DIM,):
                raise ValueError(f"{name} must have shape ({ACTION_DIM},), got {value.shape}")
            setattr(self, name, value)
        if np.any(self.low > self.center) or np.any(self.center > self.high):
            raise ValueError("Action bounds must satisfy low <= center <= high")
        if np.any(self.max_delta <= 0):
            raise ValueError("All max_delta entries must be positive")

    @classmethod
    def from_npz(cls, path: str | Path) -> "ActionTransform":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                center=data["center"],
                low=data["low"],
                high=data["high"],
                max_delta=data["max_delta"],
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, center=self.center, low=self.low, high=self.high, max_delta=self.max_delta)

    def decode(self, raw: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected final action dimension {ACTION_DIM}, got {raw.shape}")
        clipped = np.clip(raw, -1.0, 1.0)
        positive = self.center + clipped * (self.high - self.center)
        negative = self.center + clipped * (self.center - self.low)
        decoded = np.where(clipped >= 0.0, positive, negative)
        if previous is not None:
            previous = np.asarray(previous, dtype=np.float32)
            decoded = np.clip(decoded, previous - self.max_delta, previous + self.max_delta)
        return np.clip(decoded, self.low, self.high).astype(np.float32)

    def encode(self, physical: np.ndarray) -> np.ndarray:
        """Invert the piecewise affine mapping (before the optional slew limit)."""
        physical = np.asarray(physical, dtype=np.float32)
        if physical.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected final action dimension {ACTION_DIM}, got {physical.shape}")
        clipped = np.clip(physical, self.low, self.high)
        offset = clipped - self.center
        positive_span = np.maximum(self.high - self.center, 1e-6)
        negative_span = np.maximum(self.center - self.low, 1e-6)
        raw = np.where(offset >= 0.0, offset / positive_span, offset / negative_span)
        return np.clip(raw, -1.0, 1.0).astype(np.float32)


def compute_action_transform(actions: list[np.ndarray]) -> ActionTransform:
    """Derive safe physical action bounds from complete demonstration episodes."""
    if not actions:
        raise ValueError("No action episodes supplied")
    arrays = [np.asarray(x, dtype=np.float32) for x in actions]
    if any(x.ndim != 2 or x.shape[1] != ACTION_DIM for x in arrays):
        raise ValueError("Every action episode must have shape (T, 36)")

    flat = np.concatenate(arrays, axis=0)
    first = np.stack([x[0] for x in arrays], axis=0)
    center = np.median(first, axis=0).astype(np.float32)
    low = np.quantile(flat, 0.005, axis=0).astype(np.float32)
    high = np.quantile(flat, 0.995, axis=0).astype(np.float32)
    span = high - low
    low -= 0.1 * span
    high += 0.1 * span

    # Dataset fields 31:36 are constant for this tabletop task. Give the policy
    # narrow, physically meaningful ranges while keeping a complete 36-D output.
    low[31:36] = np.array([0.70, -0.10, -0.10, 0.0, -0.15], dtype=np.float32)
    high[31:36] = np.array([0.80, 0.10, 0.10, 0.49, 0.15], dtype=np.float32)
    center[31:36] = np.array([0.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    deltas = np.concatenate([np.abs(np.diff(x, axis=0)) for x in arrays], axis=0)
    max_delta = np.quantile(deltas, 0.99, axis=0).astype(np.float32)
    floors = np.full(ACTION_DIM, 0.01, dtype=np.float32)
    floors[31:36] = np.array([0.005, 0.03, 0.03, 0.1, 0.03], dtype=np.float32)
    max_delta = np.maximum(max_delta, floors)

    tiny = high - low < 1e-4
    low[tiny] = center[tiny] - 0.01
    high[tiny] = center[tiny] + 0.01
    return ActionTransform(center=center, low=low, high=high, max_delta=max_delta)
