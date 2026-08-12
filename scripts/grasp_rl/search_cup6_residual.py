"""Search constant Cup_6 residuals in parallel without training a policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from simple.grasp_rl.mjlab_gpu.config import (
    DomainRandomizationConfig,
    MjlabPpoConfig,
)
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv
from simple.grasp_rl.schema import ACTION_DIM

HAND_INDICES = tuple(range(7, 14))
ARM_INDICES = tuple(range(21, 28))
ACTIVE_INDICES = HAND_INDICES + ARM_INDICES
BASE_HAND = (0.05600, -0.02387, 0.04154, 0.00924, -0.11615, 0.07049, 0.10106)
BASE_ARM = (-0.43715, -0.19394, 0.06247, 0.38469, 0.00523, 0.26747, -0.27524)


def _candidate_corrections(
    count: int, *, seed: int, device: str
) -> torch.Tensor:
    if count < 16:
        raise ValueError("residual search requires at least 16 candidates")
    base = torch.zeros(ACTION_DIM, dtype=torch.float32)
    base[list(HAND_INDICES)] = torch.tensor(BASE_HAND)
    base[list(ARM_INDICES)] = torch.tensor(BASE_ARM)
    candidates = base.repeat(count, 1)

    engine = torch.quasirandom.SobolEngine(
        dimension=len(ACTIVE_INDICES), scramble=True, seed=seed
    )
    samples = 2.0 * engine.draw(count - 1) - 1.0
    group = torch.arange(count - 1) * 4 // (count - 1)
    hand_scales = torch.tensor((0.01, 0.02, 0.04, 0.08))[group]
    arm_scales = torch.tensor((0.02, 0.05, 0.10, 0.20))[group]
    scales = torch.cat(
        (hand_scales[:, None].expand(-1, 7), arm_scales[:, None].expand(-1, 7)),
        dim=1,
    )
    candidates[1:, list(ACTIVE_INDICES)] += samples * scales
    return candidates.clamp(-0.7, 0.7).to(device)


@torch.inference_mode()
def search(args: argparse.Namespace) -> dict[str, object]:
    config = MjlabPpoConfig(
        task="grasp_anything",
        asset_bundle=str(args.asset.resolve()),
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        smoke_mode=args.num_envs < 2048,
        reference_processed=str(args.reference.resolve()),
        strict_reference_episode=82,
        max_reference_initial_position_offset=0.08,
        reference_reward_weight=0.005,
        max_reference_action_deviation=0.7,
        domain_randomization=replace(DomainRandomizationConfig(), enabled=False),
    )
    env = GpuGraspVecEnv(
        config, training=False, randomization_enabled=False
    )
    corrections = _candidate_corrections(
        args.num_envs, seed=args.seed, device=args.device
    )
    active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
    success = torch.zeros_like(active)
    failure = torch.zeros_like(active)
    timeout = torch.zeros_like(active)
    had_grasp = torch.zeros_like(active)
    max_lift = torch.full(
        (args.num_envs,), -torch.inf, dtype=torch.float32, device=args.device
    )
    max_quality = torch.zeros_like(max_lift)
    terminal_step = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=args.device
    )

    for step in range(args.max_steps):
        actions = env.reference.current_action() + corrections
        _, _, dones, _ = env.step(actions)
        terms = env.last_terms
        assert terms is not None
        max_lift.copy_(
            torch.where(active, torch.maximum(max_lift, terms.lift_height), max_lift)
        )
        max_quality.copy_(
            torch.where(
                active,
                torch.maximum(max_quality, terms.grasp_quality),
                max_quality,
            )
        )
        had_grasp.logical_or_(active & terms.is_grasp)
        finished = active & dones
        success.logical_or_(finished & terms.success)
        failure.logical_or_(finished & terms.failure)
        timeout.logical_or_(finished & terms.timeout)
        terminal_step[finished] = step + 1
        active.logical_and_(~finished)
        if not active.any():
            break

    score = (
        success.float() * 1000.0
        + max_lift.clamp_min(-1.0) * 100.0
        + had_grasp.float()
        + 0.1 * max_quality
    )
    top_ids = torch.topk(score, min(args.top_k, args.num_envs)).indices
    records = []
    for env_id in top_ids.cpu().tolist():
        correction = corrections[env_id]
        records.append(
            {
                "env_id": env_id,
                "success": bool(success[env_id].item()),
                "failure": bool(failure[env_id].item()),
                "timeout": bool(timeout[env_id].item()),
                "unfinished": bool(active[env_id].item()),
                "terminal_step": int(terminal_step[env_id].item()),
                "max_lift_m": float(max_lift[env_id].item()),
                "max_grasp_quality": float(max_quality[env_id].item()),
                "had_grasp": bool(had_grasp[env_id].item()),
                "right_hand_correction": correction[list(HAND_INDICES)]
                .cpu()
                .tolist(),
                "right_arm_correction": correction[list(ARM_INDICES)]
                .cpu()
                .tolist(),
            }
        )
    result = {
        "schema_version": 1,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "completed": int((~active).sum().item()),
        "successes": int(success.sum().item()),
        "grasp_candidates": int(had_grasp.sum().item()),
        "best": records[0],
        "top": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(search(args), indent=2))


if __name__ == "__main__":
    main()
