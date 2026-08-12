import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/grasp_rl/diagnose_cup6_worlds.py"
SPEC = importlib.util.spec_from_file_location("diagnose_cup6_worlds", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSE)


def test_statistics_reports_bins_gates_and_grasp_gap() -> None:
    report = DIAGNOSE._statistics(
        target_translation_xy=torch.tensor(
            [[0.0, -1.0], [1.0, -0.2], [2.0, 0.2], [3.0, 1.0]]
        ),
        target_yaw=torch.tensor([-0.3, -0.1, 0.1, 0.3]),
        success=torch.tensor([True, True, False, False]),
        result={"grasp_episode_rate": 0.75},
        x_bounds=(0.0, 4.0),
        y_bounds=(-1.0, 1.0),
        yaw_bounds=(-0.4, 0.4),
        bin_count=2,
        far_x_threshold=2.0,
    )

    assert report["successes"] == 2
    assert report["grasp_episodes"] == 3
    assert report["grasp_not_success"] == 1
    assert report["no_grasp"] == 1
    assert report["bins"]["target_x"] == [
        {
            "lower": 0.0,
            "upper": 2.0,
            "samples": 2,
            "successes": 2,
            "failures": 0,
            "success_rate": 1.0,
        },
        {
            "lower": 2.0,
            "upper": 4.0,
            "samples": 2,
            "successes": 0,
            "failures": 2,
            "success_rate": 0.0,
        },
    ]
    assert report["far_x_gate"]["far"]["samples"] == 1
    assert report["far_x_gate"]["far"]["success_rate"] == 0.0
    assert report["far_x_gate"]["far"]["failure_share"] == 0.5
    assert report["success_correlations"]["target_x"] == pytest.approx(
        -0.8944271909999159
    )


def test_expand_accepts_one_or_one_per_log() -> None:
    assert DIAGNOSE._expand([1], 3, "value") == [1, 1, 1]
    assert DIAGNOSE._expand([1, 2], 2, "value") == [1, 2]
    assert DIAGNOSE._expand(None, 2, "value") == [None, None]
    with pytest.raises(ValueError, match="once or once per"):
        DIAGNOSE._expand([1, 2], 3, "value")
