#!/usr/bin/env python3
"""Reconstruct Cup_6 evaluation DR values and summarize failed worlds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from simple.grasp_rl.mjlab_gpu.cli import (
    _evaluation_world_sha256,
    _seed_torch,
    _set_evaluation_dr_strength,
)
from simple.grasp_rl.mjlab_gpu.config import (
    DomainRandomizationConfig,
    MjlabPpoConfig,
    ReferenceNoiseConfig,
)
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv


def _evaluation_result(path: Path) -> dict[str, object]:
    text = path.read_text()
    marker = '{\n  "result":'
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"evaluation JSON not found in {path}")
    return json.loads(text[start:])["result"]


def _config(path: Path, *, seed: int, num_envs: int) -> MjlabPpoConfig:
    payload = json.loads(path.read_text())["environment"]
    dr_payload = payload.pop("domain_randomization")
    noise = ReferenceNoiseConfig(**dr_payload.pop("reference_noise"))
    domain_randomization = DomainRandomizationConfig(
        **dr_payload, reference_noise=noise
    )
    payload.update(seed=seed, num_envs=num_envs, smoke_mode=True)
    return MjlabPpoConfig(**payload, domain_randomization=domain_randomization)


def _summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
    selected = values[mask].float()
    if selected.ndim == 1:
        selected = selected[:, None]
    return {
        "count": int(mask.sum().item()),
        "mean": selected.mean(dim=0).tolist(),
        "minimum": selected.amin(dim=0).tolist(),
        "maximum": selected.amax(dim=0).tolist(),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-log", type=Path, required=True)
    parser.add_argument("--dr-strength", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = _evaluation_result(args.evaluation_log)
    seed = int(result["policy_seed"])
    num_envs = int(result["episodes"])
    config = _config(args.config, seed=seed, num_envs=num_envs)
    _seed_torch(seed)
    env = GpuGraspVecEnv(config, training=False, randomization_enabled=True)
    _set_evaluation_dr_strength(env, args.dr_strength)
    observations = env.get_observations()
    physical_hash, _, _ = _evaluation_world_sha256(env, observations, num_envs)
    expected_hash = str(result["initial_world_sha256"])
    if physical_hash != expected_hash:
        raise RuntimeError(
            f"reconstructed world hash mismatch: {physical_hash} != {expected_hash}"
        )

    failed_ids = torch.tensor(
        result["failed_world_ids"], dtype=torch.long, device=env.device
    )
    failed = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    failed[failed_ids] = True
    succeeded = ~failed
    randomizer = env.randomizer
    values = {
        "target_translation_xy": randomizer.target_translation_xy,
        "target_yaw": randomizer.target_yaw,
        "robot_base_translation_xy": randomizer.robot_base_translation_xy,
        "robot_base_yaw": randomizer.robot_base_yaw,
        "target_mass_scale": randomizer.target_mass_scale,
        "friction_scale": randomizer.friction_scale,
        "joint_damping_mean": randomizer.joint_damping_scale.mean(dim=1),
        "right_arm_strength_mean": randomizer.actuator_strength_scale[:, 21:28].mean(
            dim=1
        ),
        "right_hand_strength_mean": randomizer.actuator_strength_scale[:, 7:14].mean(
            dim=1
        ),
        "action_delay_steps": randomizer.action_delay_steps.float(),
    }
    selected_worlds = {
        str(world_id): {
            name: value[world_id].detach().cpu().tolist()
            for name, value in values.items()
        }
        for world_id in failed_ids.tolist()
    }
    report = {
        "evaluation_log": str(args.evaluation_log.resolve()),
        "initial_world_sha256": physical_hash,
        "failed_world_ids": failed_ids.tolist(),
        "failed_worlds": selected_worlds,
        "cohorts": {
            name: {
                "failed": _summary(value, failed),
                "succeeded": _summary(value, succeeded),
            }
            for name, value in values.items()
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)


if __name__ == "__main__":
    main()
