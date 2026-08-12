#!/usr/bin/env python3
"""Reconstruct Cup_6 evaluation worlds and summarize pose-conditioned failures."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch


def _evaluation_result(path: Path) -> dict[str, object]:
    text = path.read_text()
    marker = '{\n  "result":'
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"evaluation JSON not found in {path}")
    result = json.loads(text[start:])["result"]
    if not isinstance(result, dict):
        raise ValueError(f"evaluation result is not an object in {path}")
    return result


def _config(
    path: Path,
    *,
    seed: int,
    num_envs: int,
    target_position_focus_probability: float | None,
) -> Any:
    # Keep GPU-only mjlab imports out of CPU statistics tests.
    from simple.grasp_rl.mjlab_gpu.config import (
        DomainRandomizationConfig,
        MjlabPpoConfig,
        ReferenceNoiseConfig,
    )

    payload = json.loads(path.read_text())["environment"]
    dr_payload = payload.pop("domain_randomization")
    noise = ReferenceNoiseConfig(**dr_payload.pop("reference_noise"))
    if target_position_focus_probability is not None:
        dr_payload["target_position_focus_probability"] = (
            target_position_focus_probability
        )
    domain_randomization = DomainRandomizationConfig(
        **dr_payload, reference_noise=noise
    )
    payload.update(seed=seed, num_envs=num_envs, smoke_mode=True)
    return MjlabPpoConfig(**payload, domain_randomization=domain_randomization)


def _summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
    selected = values[mask].float()
    if selected.ndim == 1:
        selected = selected[:, None]
    if not len(selected):
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "count": int(mask.sum().item()),
        "mean": selected.mean(dim=0).tolist(),
        "minimum": selected.amin(dim=0).tolist(),
        "maximum": selected.amax(dim=0).tolist(),
    }


def _point_biserial(success: torch.Tensor, values: torch.Tensor) -> float | None:
    x = success.to(torch.float64)
    y = values.to(torch.float64)
    if len(x) < 2 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return None
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1].item())


def _axis_bins(
    values: torch.Tensor,
    success: torch.Tensor,
    *,
    lower: float,
    upper: float,
    count: int,
) -> list[dict[str, object]]:
    edges = torch.linspace(lower, upper, count + 1, dtype=torch.float64)
    rows = []
    for index in range(count):
        low = float(edges[index].item())
        high = float(edges[index + 1].item())
        selected = values >= low
        selected &= values <= high if index == count - 1 else values < high
        samples = int(selected.sum().item())
        successes = int(success[selected].sum().item())
        rows.append(
            {
                "lower": low,
                "upper": high,
                "samples": samples,
                "successes": successes,
                "failures": samples - successes,
                "success_rate": successes / samples if samples else None,
            }
        )
    return rows


def _gate_summary(
    selected: torch.Tensor, success: torch.Tensor
) -> dict[str, int | float | None]:
    samples = int(selected.sum().item())
    successes = int(success[selected].sum().item())
    failures = samples - successes
    total_failures = int((~success).sum().item())
    return {
        "samples": samples,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / samples if samples else None,
        "failure_share": failures / total_failures if total_failures else None,
    }


def _statistics(
    *,
    target_translation_xy: torch.Tensor,
    target_yaw: torch.Tensor,
    success: torch.Tensor,
    result: dict[str, object],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    yaw_bounds: tuple[float, float],
    bin_count: int,
    far_x_threshold: float | None,
) -> dict[str, object]:
    values = target_translation_xy.to(torch.float64)
    yaw = target_yaw.to(torch.float64)
    success = success.to(torch.bool)
    x = values[:, 0]
    y = values[:, 1]
    if not (len(x) == len(y) == len(yaw) == len(success)):
        raise ValueError("pose and success tensors must have equal lengths")
    if bin_count < 1:
        raise ValueError("bin_count must be positive")

    episodes = len(success)
    successes = int(success.sum().item())
    failures = episodes - successes
    grasp_rate = float(result["grasp_episode_rate"])
    grasp_episodes = int(round(grasp_rate * episodes))
    if not successes <= grasp_episodes <= episodes:
        raise ValueError("grasp_episode_rate is inconsistent with success count")
    threshold = (
        float(far_x_threshold)
        if far_x_threshold is not None
        else x_bounds[1] - (x_bounds[1] - x_bounds[0]) / bin_count
    )
    far = x > threshold
    near = ~far
    extreme_y_threshold = max(abs(y_bounds[0]), abs(y_bounds[1])) * 0.6

    return {
        "episodes": episodes,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / episodes,
        "grasp_episodes": grasp_episodes,
        "grasp_episode_rate": grasp_episodes / episodes,
        # Evaluation JSON contains aggregate grasp rate, not per-world grasp IDs.
        "grasp_not_success": grasp_episodes - successes,
        "no_grasp": episodes - grasp_episodes,
        "configured_ranges": {
            "target_x": list(x_bounds),
            "target_y": list(y_bounds),
            "target_yaw": list(yaw_bounds),
        },
        "observed_ranges": {
            "target_x": [float(x.min().item()), float(x.max().item())],
            "target_y": [float(y.min().item()), float(y.max().item())],
            "target_yaw": [float(yaw.min().item()), float(yaw.max().item())],
        },
        "success_correlations": {
            "target_x": _point_biserial(success, x),
            "target_y": _point_biserial(success, y),
            "absolute_target_y": _point_biserial(success, y.abs()),
            "target_yaw": _point_biserial(success, yaw),
            "absolute_target_yaw": _point_biserial(success, yaw.abs()),
        },
        "bins": {
            "target_x": _axis_bins(
                x, success, lower=x_bounds[0], upper=x_bounds[1], count=bin_count
            ),
            "target_y": _axis_bins(
                y, success, lower=y_bounds[0], upper=y_bounds[1], count=bin_count
            ),
            "target_yaw": _axis_bins(
                yaw,
                success,
                lower=yaw_bounds[0],
                upper=yaw_bounds[1],
                count=bin_count,
            ),
        },
        "far_x_gate": {
            "threshold": threshold,
            "comparison": "target_x > threshold",
            "far": _gate_summary(far, success),
            "near": _gate_summary(near, success),
        },
        "extreme_y_gate": {
            "absolute_threshold": extreme_y_threshold,
            "comparison": "abs(target_y) > threshold",
            "extreme": _gate_summary(y.abs() > extreme_y_threshold, success),
            "central": _gate_summary(y.abs() <= extreme_y_threshold, success),
        },
    }


def _expand(values: list[Any] | None, count: int, name: str) -> list[Any | None]:
    if not values:
        return [None] * count
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} must be provided once or once per evaluation log")
    return values


def _configured_pose_bounds(config: Any, strength: float) -> dict[str, tuple[float, float]]:
    dr = config.domain_randomization
    x_center, y_center = dr.target_position_offset_center_xy
    x_jitter, y_jitter = dr.target_position_jitter_xy
    return {
        "x": (
            strength * (x_center - x_jitter),
            strength * (x_center + x_jitter),
        ),
        "y": (
            strength * (y_center - y_jitter),
            strength * (y_center + y_jitter),
        ),
        "yaw": (-strength * dr.target_yaw_jitter, strength * dr.target_yaw_jitter),
    }


@torch.inference_mode()
def _reconstruct(
    *,
    config_path: Path,
    evaluation_log: Path,
    seed_override: int | None,
    focus_probability: float | None,
    dr_strength: float,
    bin_count: int,
    far_x_threshold: float | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, object]]:
    from simple.grasp_rl.mjlab_gpu.cli import (
        _evaluation_world_sha256,
        _seed_torch,
        _set_evaluation_dr_strength,
    )
    from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv

    result = _evaluation_result(evaluation_log)
    seed = int(result["policy_seed"] if seed_override is None else seed_override)
    num_envs = int(result["episodes"])
    config = _config(
        config_path,
        seed=seed,
        num_envs=num_envs,
        target_position_focus_probability=focus_probability,
    )
    _seed_torch(seed)
    env = GpuGraspVecEnv(config, training=False, randomization_enabled=True)
    _set_evaluation_dr_strength(env, dr_strength)
    observations = env.get_observations()
    physical_hash, _, _ = _evaluation_world_sha256(env, observations, num_envs)
    expected_hash = str(result["initial_world_sha256"])
    if physical_hash != expected_hash:
        raise RuntimeError(
            f"reconstructed world hash mismatch for {evaluation_log}: "
            f"{physical_hash} != {expected_hash}; check config, seed, DR strength, "
            "and --target-position-focus-probability"
        )

    failed_ids = torch.tensor(result["failed_world_ids"], dtype=torch.long)
    failed = torch.zeros(num_envs, dtype=torch.bool)
    failed[failed_ids] = True
    success = ~failed
    randomizer = env.randomizer
    values = {
        "target_translation_xy": randomizer.target_translation_xy.detach().cpu(),
        "target_yaw": randomizer.target_yaw.detach().cpu(),
        "robot_base_translation_xy": (
            randomizer.robot_base_translation_xy.detach().cpu()
        ),
        "robot_base_yaw": randomizer.robot_base_yaw.detach().cpu(),
        "target_mass_scale": randomizer.target_mass_scale.detach().cpu(),
        "friction_scale": randomizer.friction_scale.detach().cpu(),
        "joint_damping_mean": (
            randomizer.joint_damping_scale.mean(dim=1).detach().cpu()
        ),
        "right_arm_strength_mean": (
            randomizer.actuator_strength_scale[:, 21:28].mean(dim=1).detach().cpu()
        ),
        "right_hand_strength_mean": (
            randomizer.actuator_strength_scale[:, 7:14].mean(dim=1).detach().cpu()
        ),
        "action_delay_steps": randomizer.action_delay_steps.float().detach().cpu(),
    }
    bounds = _configured_pose_bounds(config, dr_strength)
    statistics = _statistics(
        target_translation_xy=values["target_translation_xy"],
        target_yaw=values["target_yaw"],
        success=success,
        result=result,
        x_bounds=bounds["x"],
        y_bounds=bounds["y"],
        yaw_bounds=bounds["yaw"],
        bin_count=bin_count,
        far_x_threshold=far_x_threshold,
    )
    selected_worlds = {
        str(world_id): {
            name: value[world_id].tolist() for name, value in values.items()
        }
        for world_id in failed_ids.tolist()
    }
    report = {
        "evaluation_log": str(evaluation_log.resolve()),
        "config": str(config_path.resolve()),
        "seed": seed,
        "initial_world_sha256": physical_hash,
        "focus_probability_override": focus_probability,
        "failed_world_ids": failed_ids.tolist(),
        "failed_worlds": selected_worlds,
        "cohorts": {
            name: {
                "failed": _summary(value, failed),
                "succeeded": _summary(value, success),
            }
            for name, value in values.items()
        },
        "statistics": statistics,
    }
    pooled = {
        "target_translation_xy": values["target_translation_xy"],
        "target_yaw": values["target_yaw"],
        "success": success,
    }
    env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report, pooled, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Training config; repeat per evaluation log or provide once",
    )
    parser.add_argument(
        "--evaluation-log", type=Path, action="append", required=True
    )
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--dr-strength", type=float, required=True)
    parser.add_argument(
        "--target-position-focus-probability",
        type=float,
        action="append",
        help="Override training focus for reconstruction; repeat per log if needed",
    )
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument(
        "--far-x-threshold",
        type=float,
        help="Far-X gate in metres; default is the lower edge of the last X bin",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.dr_strength <= 1.0:
        parser.error("--dr-strength must be in [0, 1]")
    if args.bins < 1:
        parser.error("--bins must be positive")
    if args.target_position_focus_probability and any(
        not 0.0 <= value <= 1.0
        for value in args.target_position_focus_probability
    ):
        parser.error("--target-position-focus-probability must be in [0, 1]")

    log_count = len(args.evaluation_log)
    configs = _expand(args.config, log_count, "--config")
    seeds = _expand(args.seed, log_count, "--seed")
    focuses = _expand(
        args.target_position_focus_probability,
        log_count,
        "--target-position-focus-probability",
    )
    reports = []
    pooled_values: list[dict[str, torch.Tensor]] = []
    results = []
    for config_path, evaluation_log, seed, focus in zip(
        configs, args.evaluation_log, seeds, focuses, strict=True
    ):
        report, pooled, result = _reconstruct(
            config_path=config_path,
            evaluation_log=evaluation_log,
            seed_override=seed,
            focus_probability=focus,
            dr_strength=args.dr_strength,
            bin_count=args.bins,
            far_x_threshold=args.far_x_threshold,
        )
        reports.append(report)
        pooled_values.append(pooled)
        results.append(result)

    translations = torch.cat(
        [value["target_translation_xy"] for value in pooled_values]
    )
    yaws = torch.cat([value["target_yaw"] for value in pooled_values])
    successes = torch.cat([value["success"] for value in pooled_values])
    ranges = [report["statistics"]["configured_ranges"] for report in reports]
    pooled_result = {
        "grasp_episode_rate": sum(
            float(result["grasp_episode_rate"]) * int(result["episodes"])
            for result in results
        )
        / len(successes)
    }
    pooled_summary = _statistics(
        target_translation_xy=translations,
        target_yaw=yaws,
        success=successes,
        result=pooled_result,
        x_bounds=(
            min(float(value["target_x"][0]) for value in ranges),
            max(float(value["target_x"][1]) for value in ranges),
        ),
        y_bounds=(
            min(float(value["target_y"][0]) for value in ranges),
            max(float(value["target_y"][1]) for value in ranges),
        ),
        yaw_bounds=(
            min(float(value["target_yaw"][0]) for value in ranges),
            max(float(value["target_yaw"][1]) for value in ranges),
        ),
        bin_count=args.bins,
        far_x_threshold=args.far_x_threshold,
    )
    document: dict[str, object] = {
        "schema_version": 2,
        "runs": reports,
        "pooled_summary": pooled_summary,
    }
    # Retain the original single-log top-level fields for existing ad-hoc users.
    if len(reports) == 1:
        document.update(reports[0])
    encoded = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)


if __name__ == "__main__":
    main()
