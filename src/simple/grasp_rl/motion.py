"""Executed-motion extraction shared by offline replay and online guidance."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from simple.grasp_rl.schema import (
    JOINT_NAMES,
    MOTION_FEATURE_DIM,
    MOTION_FRAME_DIM,
    MOTION_LINK_NAMES,
    MOTION_WINDOW,
)


def _quat_to_matrix_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _inverse_heading_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    return torch.stack([c, s, zero, -s, c, zero, zero, zero, one], dim=-1).reshape(
        *q.shape[:-1], 3, 3
    )


def frames_to_features(frames: torch.Tensor) -> torch.Tensor:
    """Convert raw kinematics ``(..., W, 80)`` to SMP features ``(..., W, 82)``.

    Every window is anchored at its final pelvis position and yaw. Rotation-6D
    follows the source SMP implementation and stacks matrix columns 0 and 2.
    """
    if frames.shape[-2:] != (MOTION_WINDOW, MOTION_FRAME_DIM):
        raise ValueError(
            f"Expected (..., {MOTION_WINDOW}, {MOTION_FRAME_DIM}), got {tuple(frames.shape)}"
        )

    root_pos = frames[..., 0:3]
    root_quat = frames[..., 3:7]
    root_lin = frames[..., 7:10]
    root_ang = frames[..., 10:13]
    joint_pos = frames[..., 13:56]
    ee_pos = frames[..., 56:80].reshape(*frames.shape[:-1], 8, 3)

    heading_inv = _inverse_heading_matrix(root_quat[..., -1, :])
    root_offset = root_pos - root_pos[..., -1:, :]
    root_local = torch.einsum("...ij,...wj->...wi", heading_inv, root_offset)
    root_local = root_local.clone()
    root_local[..., 2] = root_pos[..., 2]

    root_rot = torch.einsum("...ij,...wjk->...wik", heading_inv, _quat_to_matrix_wxyz(root_quat))
    root_rot_6d = torch.cat([root_rot[..., :, 0], root_rot[..., :, 2]], dim=-1)

    ee_offset = ee_pos - root_pos[..., :, None, :]
    ee_local = torch.einsum("...ij,...wej->...wei", heading_inv, ee_offset).flatten(-2)
    lin_local = torch.einsum("...ij,...wj->...wi", heading_inv, root_lin)
    ang_local = torch.einsum("...ij,...wj->...wi", heading_inv, root_ang)

    output = torch.cat(
        [root_local, root_rot_6d, joint_pos, ee_local, lin_local, ang_local], dim=-1
    )
    if output.shape[-1] != MOTION_FEATURE_DIM:
        raise RuntimeError(f"Internal motion feature mismatch: {output.shape}")
    return output


def numpy_frames_to_features(frames: np.ndarray) -> np.ndarray:
    # ``sliding_window_view`` returns a read-only strided view.  Copying here
    # makes the PyTorch ownership contract explicit and keeps replay warning-free.
    tensor = torch.from_numpy(np.array(frames, dtype=np.float32, copy=True))
    return frames_to_features(tensor).cpu().numpy()


@dataclass
class MotionFrameExtractor:
    """Resolved MuJoCo IDs for fast, deterministic extraction."""

    model: mujoco.MjModel
    data: mujoco.MjData
    robot: object

    def __post_init__(self) -> None:
        model_joint_names = tuple(self.robot.joint_names)
        if model_joint_names != JOINT_NAMES:
            raise ValueError("G1 joint ordering differs from the grasp-RL schema")
        self.root_body_id = self.model.body("pelvis").id
        self.link_body_ids = tuple(self.model.body(name).id for name in MOTION_LINK_NAMES)

    def extract(self) -> np.ndarray:
        root = self.data.body(self.root_body_id)
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.root_body_id,
            spatial_velocity,
            0,
        )
        angular = spatial_velocity[:3]
        linear = spatial_velocity[3:]
        joints = np.fromiter(
            (self.data.joint(name).qpos.item() for name in JOINT_NAMES),
            dtype=np.float64,
            count=len(JOINT_NAMES),
        )
        links = np.concatenate([self.data.body(body_id).xpos for body_id in self.link_body_ids])
        frame = np.concatenate([root.xpos, root.xquat, linear, angular, joints, links])
        if frame.shape != (MOTION_FRAME_DIM,):
            raise RuntimeError(f"Unexpected motion frame shape {frame.shape}")
        return frame.astype(np.float32)


class BatchedMotionBuffer:
    """GPU rolling window used by the vectorized RL environment."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.frames = torch.zeros(
            num_envs, MOTION_WINDOW, MOTION_FRAME_DIM, device=device, dtype=torch.float32
        )
        self.valid = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.frames[env_ids] = frame[:, None, :]
        self.valid[env_ids] = 1

    def update(self, frame: torch.Tensor) -> None:
        self.frames = torch.roll(self.frames, shifts=-1, dims=1)
        self.frames[:, -1] = frame
        self.valid.clamp_max_(MOTION_WINDOW)
        self.valid += (self.valid < MOTION_WINDOW).long()

    def features(self) -> torch.Tensor:
        return frames_to_features(self.frames)

    @property
    def is_full(self) -> torch.Tensor:
        return self.valid >= MOTION_WINDOW
