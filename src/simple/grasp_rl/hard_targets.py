"""Failure-target mining with an explicit final-test leakage barrier."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


HARD_TARGET_SCHEMA_VERSION = 1
TRAINING_SOURCE_SPLITS = {"train", "val"}


@dataclass(frozen=True)
class HardTarget:
    target_offset_xy: tuple[float, float]
    base_episode: int
    rollout_index: int


@dataclass(frozen=True)
class HardTargetManifest:
    targets: tuple[HardTarget, ...]
    source_evaluation: str
    source_split: str
    source_seed: int | None


def load_hard_target_manifest(path: str | Path) -> HardTargetManifest:
    source = Path(path)
    records = [
        json.loads(line)
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "manifest":
        raise ValueError("hard-target JSONL must start with a manifest record")
    header = records[0]
    if header.get("schema_version") != HARD_TARGET_SCHEMA_VERSION:
        raise ValueError(
            f"hard-target schema_version must be {HARD_TARGET_SCHEMA_VERSION}"
        )
    source_split = str(header.get("source_split", ""))
    if source_split not in TRAINING_SOURCE_SPLITS:
        raise ValueError(
            "hard targets may only come from train/val screening, never test/final"
        )
    if bool(header.get("final_test", False)):
        raise ValueError("final-test targets must never be used for training")

    targets = []
    for record in records[1:]:
        if record.get("record_type") != "target":
            raise ValueError("unknown hard-target JSONL record")
        offset = record.get("target_offset_xy")
        if not isinstance(offset, list) or len(offset) != 2:
            raise ValueError("target_offset_xy must contain exactly two values")
        targets.append(
            HardTarget(
                target_offset_xy=(float(offset[0]), float(offset[1])),
                base_episode=int(record["base_episode"]),
                rollout_index=int(record["rollout_index"]),
            )
        )
    if not targets:
        raise ValueError("hard-target manifest contains no targets")
    return HardTargetManifest(
        targets=tuple(targets),
        source_evaluation=str(header["source_evaluation"]),
        source_split=source_split,
        source_seed=(
            None if header.get("source_seed") is None else int(header["source_seed"])
        ),
    )


def mine_hard_targets(
    evaluation_summary: str | Path,
    output_path: str | Path,
    *,
    limit: int = 256,
) -> HardTargetManifest:
    """Write failed train/val targets, hardest first, to auditable JSONL."""

    if limit < 1:
        raise ValueError("limit must be positive")
    summary_path = Path(evaluation_summary)
    summary = json.loads(summary_path.read_text())
    source_split = str(summary.get("evaluation_split", ""))
    if source_split not in TRAINING_SOURCE_SPLITS:
        raise ValueError(
            "hard-target mining requires evaluation_split=train or val"
        )
    if bool(summary.get("final_test", False)):
        raise ValueError("refusing to mine a final-test evaluation")
    failures = []
    for result_index, source_row in enumerate(summary.get("results", [])):
        if source_row.get("success"):
            continue
        row = dict(source_row)
        row.setdefault("rollout_index", result_index)
        failures.append(row)
    failures.sort(
        key=lambda row: (
            float(row.get("max_grasp_quality", 0.0)),
            float(row.get("max_lift", 0.0)),
            int(row.get("rollout_index", 0)),
        )
    )
    failures = failures[:limit]
    if not failures:
        raise ValueError("evaluation contains no failed targets to mine")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record_type": "manifest",
        "schema_version": HARD_TARGET_SCHEMA_VERSION,
        "source_evaluation": str(summary_path.resolve()),
        "source_split": source_split,
        "source_seed": summary.get("seed"),
        "final_test": False,
    }
    lines = [json.dumps(header, sort_keys=True)]
    for row in failures:
        offset = row.get("target_offset")
        if not isinstance(offset, list) or len(offset) < 2:
            raise ValueError("evaluation result is missing target_offset")
        lines.append(
            json.dumps(
                {
                    "record_type": "target",
                    "target_offset_xy": [float(offset[0]), float(offset[1])],
                    "base_episode": int(row["episode"]),
                    "rollout_index": int(row["rollout_index"]),
                },
                sort_keys=True,
            )
        )
    output.write_text("\n".join(lines) + "\n")
    return load_hard_target_manifest(output)
