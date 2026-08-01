"""GPU reference transforms and training-only observation noise."""

from __future__ import annotations

import torch

from simple.grasp_rl.mjlab_gpu.config import ReferenceNoiseConfig
from simple.grasp_rl.schema import (
    ACTION_DIM,
    REFERENCE_CONTEXT_DIM,
    REFERENCE_CONTEXT_V2_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_FRAME_V2_DIM,
    REFERENCE_FUTURE_OFFSETS,
)


def transform_reference_positions(
    positions: torch.Tensor,
    translation_xy: torch.Tensor,
    yaw: torch.Tensor,
    origin_xy: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact sampled scene XY/yaw transform to reference positions."""

    if positions.shape[-1] != 3:
        raise ValueError("positions must end in xyz")
    batch_shape = positions.shape[:-2]
    expected = (*batch_shape, 2)
    if translation_xy.shape != expected or origin_xy.shape != expected:
        raise ValueError("translation_xy/origin_xy must match the position batch")
    if yaw.shape != batch_shape:
        raise ValueError("yaw must match the position batch")
    result = positions.clone()
    relative = positions[..., :2] - origin_xy.unsqueeze(-2)
    cosine = yaw.cos().unsqueeze(-1)
    sine = yaw.sin().unsqueeze(-1)
    x = cosine * relative[..., 0] - sine * relative[..., 1]
    y = sine * relative[..., 0] + cosine * relative[..., 1]
    result[..., 0] = (
        x + origin_xy[..., 0].unsqueeze(-1) + translation_xy[..., 0].unsqueeze(-1)
    )
    result[..., 1] = (
        y + origin_xy[..., 1].unsqueeze(-1) + translation_xy[..., 1].unsqueeze(-1)
    )
    return result


def apply_reference_noise(
    context: torch.Tensor,
    config: ReferenceNoiseConfig,
    *,
    enabled: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Perturb policy reference context without touching reward/termination truth."""

    if context.shape[-1] == REFERENCE_CONTEXT_DIM:
        frame_dim = REFERENCE_FRAME_DIM
        position_slice = slice(ACTION_DIM, ACTION_DIM + 3)
    elif context.shape[-1] == REFERENCE_CONTEXT_V2_DIM:
        frame_dim = REFERENCE_FRAME_V2_DIM
        # root, primary entity and auxiliary entity deltas
        position_slice = slice(ACTION_DIM, ACTION_DIM + 9)
    else:
        raise ValueError(f"unsupported reference context shape {tuple(context.shape)}")
    if not enabled:
        return context

    flat = context.reshape(-1, context.shape[-1])
    result = flat.clone()
    frame_count = len(REFERENCE_FUTURE_OFFSETS)
    frames = result[:, :-1].reshape(-1, frame_count, frame_dim)

    if config.action_std:
        noise = torch.randn(
            frames[..., :ACTION_DIM].shape,
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        frames[..., :ACTION_DIM].add_(noise, alpha=config.action_std).clamp_(-1.0, 1.0)
    if config.position_std:
        positions = frames[..., position_slice]
        noise = torch.randn(
            positions.shape,
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        positions.add_(noise, alpha=config.position_std)
    if config.future_dropout_probability:
        dropout = (
            torch.rand(
                (len(frames), frame_count - 1),
                device=context.device,
                generator=generator,
            )
            < config.future_dropout_probability
        )
        # Hold the last visible future frame.  Zeroing would manufacture a
        # physically invalid command and there is no mask in the frozen schema.
        for index in range(1, frame_count):
            frames[:, index] = torch.where(
                dropout[:, index - 1, None], frames[:, index - 1], frames[:, index]
            )
    if config.phase_std:
        phase_noise = torch.randn(
            result[:, -1].shape,
            device=context.device,
            dtype=context.dtype,
            generator=generator,
        )
        result[:, -1].add_(phase_noise, alpha=config.phase_std).clamp_(0.0, 1.0)
    return result.reshape(context.shape)
