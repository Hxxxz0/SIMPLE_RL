"""Search constant Cup_6 residuals in parallel without training a policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from simple.grasp_rl.mjlab_gpu.config import (
    DomainRandomizationConfig,
    MjlabPpoConfig,
    ReferenceNoiseConfig,
)
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv
from simple.grasp_rl.schema import ACTION_DIM

HAND_INDICES = tuple(range(7, 14))
ARM_INDICES = tuple(range(21, 28))
ACTIVE_INDICES = HAND_INDICES + ARM_INDICES
BASE_HAND = (0.05600, -0.02387, 0.04154, 0.00924, -0.11615, 0.07049, 0.10106)
BASE_ARM = (-0.43715, -0.19394, 0.06247, 0.38469, 0.00523, 0.26747, -0.27524)


def _candidate_corrections(
    count: int,
    *,
    seed: int,
    device: str,
    base_hand: tuple[float, ...] = BASE_HAND,
    base_arm: tuple[float, ...] = BASE_ARM,
    hand_scale: float = 1.0,
    arm_scale: float = 1.0,
) -> torch.Tensor:
    if count < 16:
        raise ValueError("residual search requires at least 16 candidates")
    if len(base_hand) != len(HAND_INDICES) or len(base_arm) != len(ARM_INDICES):
        raise ValueError("base hand and arm corrections must each contain seven values")
    if hand_scale <= 0.0 or arm_scale <= 0.0:
        raise ValueError("search scales must be positive")
    base = torch.zeros(ACTION_DIM, dtype=torch.float32)
    base[list(HAND_INDICES)] = torch.tensor(base_hand)
    base[list(ARM_INDICES)] = torch.tensor(base_arm)
    candidates = base.repeat(count, 1)

    engine = torch.quasirandom.SobolEngine(
        dimension=len(ACTIVE_INDICES), scramble=True, seed=seed
    )
    samples = 2.0 * engine.draw(count - 1) - 1.0
    group = torch.arange(count - 1) * 4 // (count - 1)
    hand_scales = hand_scale * torch.tensor((0.01, 0.02, 0.04, 0.08))[group]
    arm_scales = arm_scale * torch.tensor((0.02, 0.05, 0.10, 0.20))[group]
    scales = torch.cat(
        (hand_scales[:, None].expand(-1, 7), arm_scales[:, None].expand(-1, 7)),
        dim=1,
    )
    candidates[1:, list(ACTIVE_INDICES)] += samples * scales
    return candidates.clamp(-0.7, 0.7).to(device)


@torch.inference_mode()
def search(args: argparse.Namespace) -> dict[str, object]:
    pose_randomization = any(
        abs(value)
        for value in (
            *args.target_position_offset_center_xy,
            *args.target_position_jitter_xy,
            args.target_yaw_jitter,
        )
    )
    domain_randomization = DomainRandomizationConfig(
        enabled=pose_randomization,
        target_position_jitter_xy=tuple(args.target_position_jitter_xy),
        target_position_offset_center_xy=tuple(
            args.target_position_offset_center_xy
        ),
        target_yaw_jitter=args.target_yaw_jitter,
        destination_position_jitter_xy=(0.0, 0.0),
        destination_yaw_jitter=0.0,
        distractor_position_jitter_xy=(0.0, 0.0),
        distractor_yaw_jitter=0.0,
        robot_base_position_jitter_xy=(0.0, 0.0),
        robot_base_yaw_jitter=0.0,
        target_mass_scale=(1.0, 1.0),
        friction_scale=(1.0, 1.0),
        joint_damping_scale=(1.0, 1.0),
        actuator_strength_scale=(1.0, 1.0),
        action_delay_max_steps=0,
        curriculum_initial_strength=1.0,
        curriculum_warmup_steps=0,
        curriculum_ramp_steps=1,
        reference_noise=ReferenceNoiseConfig(
            action_std=0.0,
            position_std=0.0,
            phase_std=0.0,
            future_dropout_probability=0.0,
        ),
    )
    config = MjlabPpoConfig(
        task="grasp_anything",
        asset_bundle=str(args.asset.resolve()),
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        smoke_mode=args.num_envs < 2048,
        reference_processed=str(args.reference.resolve()),
        strict_reference_episode=82,
        max_reference_initial_position_offset=args.max_reference_initial_position_offset,
        reference_reward_weight=0.005,
        max_reference_action_deviation=0.7,
        reference_target_x_arm_gains=tuple(args.reference_target_x_arm_gains),
        reference_target_y_arm_gains=tuple(args.reference_target_y_arm_gains),
        reference_target_positive_y_arm_gains=(
            None
            if args.reference_target_positive_y_arm_gains is None
            else tuple(args.reference_target_positive_y_arm_gains)
        ),
        reference_target_yaw_arm_gains=tuple(args.reference_target_yaw_arm_gains),
        domain_randomization=domain_randomization,
    )
    env = GpuGraspVecEnv(
        config,
        training=pose_randomization,
        randomization_enabled=pose_randomization,
    )
    corrections = _candidate_corrections(
        args.num_envs,
        seed=args.seed,
        device=args.device,
        base_hand=tuple(args.base_hand),
        base_arm=tuple(args.base_arm),
        hand_scale=args.hand_search_scale,
        arm_scale=args.arm_search_scale,
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
    max_pregrasp = torch.zeros_like(max_lift)
    max_reach = torch.zeros_like(max_lift)
    max_band_reach = torch.zeros_like(max_lift)
    max_opposition = torch.zeros_like(max_lift)
    max_closure = torch.zeros_like(max_lift)
    min_fingertip_distance = torch.full_like(max_lift, torch.inf)
    numerical_failure = torch.zeros_like(active)
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
        max_pregrasp.copy_(
            torch.where(active, torch.maximum(max_pregrasp, terms.pregrasp), max_pregrasp)
        )
        max_reach.copy_(
            torch.where(active, torch.maximum(max_reach, terms.reach), max_reach)
        )
        for maximum, name in (
            (max_band_reach, "last_grasp_band_reach"),
            (max_opposition, "last_finger_opposition"),
            (max_closure, "last_closure"),
        ):
            observed = getattr(env.reward, name, None)
            if observed is not None:
                maximum.copy_(
                    torch.where(active, torch.maximum(maximum, observed), maximum)
                )
        nearest = getattr(env.reward, "last_min_fingertip_distance", None)
        if nearest is not None:
            # VecEnv resets terminal worlds inside step(), and the object reward
            # uses zero as its reset sentinel.  Ignore that sentinel so a
            # timeout cannot erase the real pre-terminal minimum.
            observed_nearest = torch.where(
                nearest > 0.0, nearest, torch.full_like(nearest, torch.inf)
            )
            min_fingertip_distance.copy_(
                torch.where(
                    active,
                    torch.minimum(min_fingertip_distance, observed_nearest),
                    min_fingertip_distance,
                )
            )
        numerical_failure.logical_or_(env.last_numerical_failure)
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
        + had_grasp.float() * 100.0
        + max_lift.clamp_min(-1.0) * 100.0
        + 10.0 * max_quality
        + max_pregrasp
        + max_reach
        + torch.exp(-20.0 * min_fingertip_distance)
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
                "max_pregrasp": float(max_pregrasp[env_id].item()),
                "max_reach": float(max_reach[env_id].item()),
                "max_band_reach": float(max_band_reach[env_id].item()),
                "max_opposition": float(max_opposition[env_id].item()),
                "max_closure": float(max_closure[env_id].item()),
                "min_fingertip_distance_m": float(
                    min_fingertip_distance[env_id].item()
                ),
                "numerical_failure": bool(numerical_failure[env_id].item()),
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
        "schema_version": 2,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "completed": int((~active).sum().item()),
        "successes": int(success.sum().item()),
        "grasp_candidates": int(had_grasp.sum().item()),
        "numerical_failures": int(numerical_failure.sum().item()),
        "target_position_offset_center_xy": list(
            args.target_position_offset_center_xy
        ),
        "target_position_jitter_xy": list(args.target_position_jitter_xy),
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
    parser.add_argument("--max-reference-initial-position-offset", type=float, default=0.08)
    parser.add_argument("--target-position-offset-center-xy", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--target-position-jitter-xy", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--target-yaw-jitter", type=float, default=0.0)
    parser.add_argument("--reference-target-x-arm-gains", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--reference-target-y-arm-gains", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--reference-target-positive-y-arm-gains", type=float, nargs=2)
    parser.add_argument("--reference-target-yaw-arm-gains", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--base-hand", type=float, nargs=7, default=BASE_HAND)
    parser.add_argument("--base-arm", type=float, nargs=7, default=BASE_ARM)
    parser.add_argument("--hand-search-scale", type=float, default=1.0)
    parser.add_argument("--arm-search-scale", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(search(args), indent=2))


if __name__ == "__main__":
    main()
