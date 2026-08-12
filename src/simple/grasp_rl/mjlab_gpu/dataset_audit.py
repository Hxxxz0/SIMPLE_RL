"""Fail-closed audit for exported PPO rollout, GR00T, and Psi0 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from simple.grasp_rl.mjlab_gpu.collect import validate_episode_arrays
from simple.grasp_rl.mjlab_gpu.dataset_export import (
    _load_jsonl,
    _video_probe,
    _write_json,
)


def _span(values: list[float], low: float, high: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {
        "min": float(array.min()),
        "max": float(array.max()),
    }
    width = float(high) - float(low)
    if width == 0.0:
        result.update(fixed=True, expected=float(low))
    else:
        result["span_fraction"] = float((array.max() - array.min()) / width)
    return result


def _xy_coverage(
    rows: list[dict[str, Any]], key: str, envelope: list[float]
) -> list[dict[str, Any]]:
    return [
        _span(
            [float(row["randomization"][key][axis]) for row in rows],
            -float(envelope[axis]),
            float(envelope[axis]),
        )
        for axis in range(2)
    ]


def _scalar_coverage(
    rows: list[dict[str, Any]], key: str, low: float, high: float
) -> dict[str, Any]:
    return _span(
        [float(row["randomization"][key]) for row in rows],
        float(low),
        float(high),
    )


def _matrix_coverage(
    rows: list[dict[str, Any]], key: str, low: float, high: float
) -> dict[str, Any]:
    values = np.asarray(
        [row["randomization"][key] for row in rows], dtype=np.float64
    )
    spans = values.max(axis=0) - values.min(axis=0)
    width = float(high) - float(low)
    result: dict[str, Any] = {
        "min": float(values.min()),
        "max": float(values.max()),
        "dimensions": int(values.shape[1]),
    }
    if width == 0.0:
        result.update(fixed=True, expected=float(low))
    else:
        result["minimum_dimension_span_fraction"] = float(spans.min() / width)
    return result


def _randomization_coverage(
    rows: list[dict[str, Any]], resolved: dict[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot audit randomization coverage without attempts")
    coverage: dict[str, Any] = {
        "episodes": len(rows),
        "target_translation_xy": _xy_coverage(
            rows, "target_translation_xy", resolved["target_position_jitter_xy"]
        ),
        "destination_translation_xy": _xy_coverage(
            rows,
            "destination_translation_xy",
            resolved["destination_position_jitter_xy"],
        ),
        "robot_base_translation_xy": _xy_coverage(
            rows,
            "robot_base_translation_xy",
            resolved["robot_base_position_jitter_xy"],
        ),
        "target_yaw": _scalar_coverage(
            rows, "target_yaw", -resolved["target_yaw_jitter"], resolved["target_yaw_jitter"]
        ),
        "destination_yaw": _scalar_coverage(
            rows,
            "destination_yaw",
            -resolved["destination_yaw_jitter"],
            resolved["destination_yaw_jitter"],
        ),
        "robot_base_yaw": _scalar_coverage(
            rows,
            "robot_base_yaw",
            -resolved["robot_base_yaw_jitter"],
            resolved["robot_base_yaw_jitter"],
        ),
        "target_mass_scale": _scalar_coverage(
            rows, "target_mass_scale", *resolved["target_mass_scale"]
        ),
        "friction_scale": _scalar_coverage(
            rows, "friction_scale", *resolved["friction_scale"]
        ),
        "joint_damping_scale": _matrix_coverage(
            rows, "joint_damping_scale", *resolved["joint_damping_scale"]
        ),
        "actuator_strength_scale": _matrix_coverage(
            rows, "actuator_strength_scale", *resolved["actuator_strength_scale"]
        ),
    }
    distractor_envelope = resolved["distractor_position_jitter_xy"]
    distractor_yaw = float(resolved["distractor_yaw_jitter"])
    distractor_names = sorted(
        {
            name
            for row in rows
            for name in row["randomization"].get("distractor_poses", {})
        }
    )
    coverage["distractor_poses"] = {
        name: {
            "translation_xy": [
                _span(
                    [
                        float(
                            row["randomization"]["distractor_poses"][name][
                                "translation_xy"
                            ][axis]
                        )
                        for row in rows
                    ],
                    -float(distractor_envelope[axis]),
                    float(distractor_envelope[axis]),
                )
                for axis in range(2)
            ],
            "yaw": _span(
                [
                    float(row["randomization"]["distractor_poses"][name]["yaw"])
                    for row in rows
                ],
                -distractor_yaw,
                distractor_yaw,
            ),
        }
        for name in distractor_names
    }
    delay_values = [
        int(row["randomization"]["action_delay_steps"]) for row in rows
    ]
    coverage["action_delay_steps"] = {
        str(value): delay_values.count(value) for value in sorted(set(delay_values))
    }
    return coverage


def _coverage_passed(value: object) -> bool:
    if isinstance(value, dict):
        if "span_fraction" in value and float(value["span_fraction"]) < 0.9:
            return False
        if (
            "minimum_dimension_span_fraction" in value
            and float(value["minimum_dimension_span_fraction"]) < 0.9
        ):
            return False
        return all(_coverage_passed(item) for item in value.values())
    if isinstance(value, list):
        return all(_coverage_passed(item) for item in value)
    return True


def audit_ppo_dataset(
    dataset_root: Path,
    *,
    expected_successes: int | None = None,
    expected_task: str | None = None,
    expected_dr_strength: float | None = None,
    require_full_dr_coverage: bool = False,
) -> dict[str, Any]:
    """Validate an exported split and write its reproducible coverage report."""

    root = dataset_root.resolve()
    rollouts = root / "rollouts"
    summary = json.loads((rollouts / "summary.json").read_text())
    rows = _load_jsonl(rollouts / "manifest.jsonl")
    saved_rows = [row for row in rows if row.get("file")]
    successes = int(summary["successes"])
    if not summary["complete"] or successes != len(saved_rows):
        raise ValueError("Rollout summary and successful manifest rows disagree")
    if expected_successes is not None and successes != expected_successes:
        raise ValueError(f"Expected {expected_successes} successes, found {successes}")
    if expected_task is not None and summary["task"] != expected_task:
        raise ValueError(f"Expected task {expected_task}, found {summary['task']}")
    if expected_dr_strength is not None and not np.isclose(
        float(summary["domain_randomization_strength"]), expected_dr_strength
    ):
        raise ValueError(
            f"Expected DR strength {expected_dr_strength}, found "
            f"{summary['domain_randomization_strength']}"
        )
    if not summary["domain_randomization"]:
        raise ValueError("PPO production dataset must use domain randomization")

    rollout_paths = sorted((rollouts / "episodes").glob("episode_*.npz"))
    if len(rollout_paths) != successes:
        raise ValueError("Successful rollout file count does not match summary")
    episode_frames: list[int] = []
    for path in rollout_paths:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        validate_episode_arrays(arrays)
        episode_frames.append(len(arrays["raw_action"]))
    total_frames = sum(episode_frames)

    groot = root / "groot" / "level-0"
    psi = root / "psi0"
    groot_data = sorted(groot.glob("data/chunk-*/episode_*.parquet"))
    psi_data = sorted(psi.glob("data/chunk-*/episode_*.parquet"))
    groot_video = sorted(groot.glob("videos/chunk-*/*/episode_*.mp4"))
    psi_video = sorted(psi.glob("videos/chunk-*/*/episode_*.mp4"))
    counts = {
        "rollout_npz": len(rollout_paths),
        "groot_parquet": len(groot_data),
        "psi0_parquet": len(psi_data),
        "groot_video": len(groot_video),
        "psi0_video": len(psi_video),
    }
    if any(count != successes for count in counts.values()):
        raise ValueError(f"Dataset view file counts disagree: {counts}")

    required_raw = {
        "observation.state",
        "observation.object_poses",
        "policy.raw_action",
        "policy.physical_action",
        "observation.policy_input",
        "next.done",
    }
    required_psi = {"states", "action", "next.done"}
    for index, (raw_path, psi_path, frames) in enumerate(
        zip(groot_data, psi_data, episode_frames, strict=True)
    ):
        raw_file = pq.ParquetFile(raw_path)
        psi_file = pq.ParquetFile(psi_path)
        if raw_file.metadata.num_rows != frames or psi_file.metadata.num_rows != frames:
            raise ValueError(f"Parquet frame mismatch at episode {index}")
        if not required_raw.issubset(raw_file.schema_arrow.names):
            raise ValueError(f"GR00T schema is incomplete at episode {index}")
        if not required_psi.issubset(psi_file.schema_arrow.names):
            raise ValueError(f"Psi0 schema is incomplete at episode {index}")
        if groot_video[index].stat().st_size != psi_video[index].stat().st_size:
            raise ValueError(f"Copied video size mismatch at episode {index}")

    raw_info = json.loads((groot / "meta" / "info.json").read_text())
    psi_info = json.loads((psi / "meta" / "info.json").read_text())
    for name, info in (("GR00T", raw_info), ("Psi0", psi_info)):
        if int(info["total_episodes"]) != successes or int(info["total_frames"]) != total_frames:
            raise ValueError(f"{name} metadata totals disagree with rollout data")
    raw_tasks = _load_jsonl(groot / "meta" / "tasks.jsonl")
    psi_tasks = _load_jsonl(psi / "meta" / "tasks.jsonl")
    if len(raw_tasks) != 1 or len(psi_tasks) != 1:
        raise ValueError("Dataset must contain exactly one language task")
    if raw_tasks[0]["task"] != psi_tasks[0]["task"]:
        raise ValueError("GR00T and Psi0 task instructions disagree")

    probe = _video_probe(groot_video[0])
    if int(probe["nb_frames"]) != episode_frames[0]:
        raise ValueError("First video and rollout frame counts disagree")
    attempt_coverage = _randomization_coverage(
        rows, summary["resolved_domain_randomization"]
    )
    success_coverage = _randomization_coverage(
        saved_rows, summary["resolved_domain_randomization"]
    )
    delay_max = int(summary["resolved_domain_randomization"]["action_delay_max_steps"])
    delays = {
        int(row["randomization"]["action_delay_steps"]) for row in rows
    }
    attempt_envelope_passed = _coverage_passed(attempt_coverage) and delays == set(
        range(delay_max + 1)
    )
    report = {
        "schema_version": 1,
        "passed": True,
        "attempt_envelope_passed": attempt_envelope_passed,
        "task": summary["task"],
        "episodes": successes,
        "attempts": int(summary["attempts"]),
        "frames": total_frames,
        "instruction": raw_tasks[0]["task"],
        "file_counts": counts,
        "video": probe,
        "randomization_coverage": {
            "attempts": attempt_coverage,
            "successful_episodes": success_coverage,
        },
    }
    if require_full_dr_coverage and not attempt_envelope_passed:
        raise ValueError("Attempt randomization coverage did not span the DR envelope")
    _write_json(root / "audit" / "randomization_coverage.json", report)
    return report
