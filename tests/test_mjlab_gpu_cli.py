from __future__ import annotations

import pytest
import torch

from simple.grasp_rl.mjlab_gpu.cli import (
    _initial_pose_diagnostics,
    _pose_axis_diagnostics,
)


def test_pose_axis_diagnostics_separates_outcomes_and_bins() -> None:
    report = _pose_axis_diagnostics(
        torch.tensor([-0.004, -0.002, 0.002, 0.004]),
        torch.tensor([1, 1, 0, 2]),
        bin_count=2,
    )

    assert report["summaries"]["all"]["samples"] == 4
    assert report["summaries"]["success"]["mean"] == pytest.approx(-0.003)
    assert report["summaries"]["physical_failure"]["mean"] == pytest.approx(0.002)
    assert report["summaries"]["timeout"]["mean"] == pytest.approx(0.004)
    assert report["bins"] == [
        {
            "lower": pytest.approx(-0.004),
            "upper": pytest.approx(0.0),
            "samples": 2,
            "successes": 2,
            "physical_failures": 0,
            "timeouts": 0,
            "success_rate": 1.0,
        },
        {
            "lower": pytest.approx(0.0),
            "upper": pytest.approx(0.004),
            "samples": 2,
            "successes": 0,
            "physical_failures": 1,
            "timeouts": 1,
            "success_rate": 0.0,
        },
    ]


def test_initial_pose_diagnostics_covers_target_and_robot_base() -> None:
    outcomes = torch.tensor([1, 0, 2])
    report = _initial_pose_diagnostics(
        {
            "target_translation_xy": torch.tensor(
                [[-0.01, -0.02], [0.0, 0.0], [0.01, 0.02]]
            ),
            "target_yaw": torch.tensor([-0.1, 0.0, 0.1]),
            "robot_base_translation_xy": torch.tensor(
                [[-0.005, 0.005], [0.0, 0.0], [0.005, -0.005]]
            ),
            "robot_base_yaw": torch.tensor([-0.02, 0.0, 0.02]),
        },
        outcomes,
    )

    assert report["sample_count"] == 3
    assert report["outcome_counts"] == {
        "success": 1,
        "physical_failure": 1,
        "timeout": 1,
    }
    assert report["target_translation_xy"]["x"]["summaries"]["success"][
        "mean"
    ] == pytest.approx(-0.01)
    assert report["robot_base_translation_xy"]["y"]["summaries"]["timeout"][
        "mean"
    ] == pytest.approx(-0.005)
    assert report["target_yaw"]["summaries"]["all"]["samples"] == 3
    assert report["robot_base_yaw"]["summaries"]["all"]["samples"] == 3


def test_pose_axis_diagnostics_rejects_invalid_outcome_code() -> None:
    with pytest.raises(ValueError, match="outcome codes"):
        _pose_axis_diagnostics(torch.tensor([0.0]), torch.tensor([-1]))
