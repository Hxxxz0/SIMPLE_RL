import json

import pytest

from simple.grasp_rl.hard_targets import (
    load_hard_target_manifest,
    mine_hard_targets,
)


def test_mine_hard_targets_keeps_only_screening_failures(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "evaluation_split": "val",
                "seed": 17,
                "results": [
                    {
                        "rollout_index": 0,
                        "episode": 3,
                        "success": True,
                        "target_offset": [0.01, 0.02, 0.0],
                    },
                    {
                        "episode": 4,
                        "success": False,
                        "max_grasp_quality": 0.0,
                        "max_lift": 0.0,
                        "target_offset": [-0.02, 0.03, 0.0],
                    },
                ],
            }
        )
    )
    output = tmp_path / "hard.jsonl"
    manifest = mine_hard_targets(summary, output)
    loaded = load_hard_target_manifest(output)

    assert manifest == loaded
    assert loaded.source_split == "val"
    assert loaded.targets[0].target_offset_xy == (-0.02, 0.03)
    assert loaded.targets[0].rollout_index == 1


def test_hard_target_mining_rejects_final_test_split(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"evaluation_split": "test", "results": []}))

    with pytest.raises(ValueError, match="train or val"):
        mine_hard_targets(summary, tmp_path / "hard.jsonl")
