import numpy as np

from simple.grasp_rl.vec_env import rsi_stage_bounds, sample_target_randomization


def test_hard_target_sampling_uses_small_failure_neighbourhood() -> None:
    mode, jitter, center, yaw = sample_target_randomization(
        np.random.default_rng(4),
        {"uniform": 0.0, "hard": 1.0, "native": 0.0},
        [[0.02, -0.03]],
        (0.025, 0.03),
        0.15,
    )
    assert mode == "hard"
    assert jitter == (0.0075, 0.0075)
    assert center == (0.02, -0.03)
    assert yaw == 0.15


def test_native_sampling_stays_near_recorded_pose() -> None:
    mode, jitter, center, yaw = sample_target_randomization(
        np.random.default_rng(8),
        {"uniform": 0.0, "hard": 0.0, "native": 1.0},
        None,
        (0.025, 0.03),
        0.15,
    )
    assert mode == "native"
    assert jitter == (0.005, 0.005)
    assert center == (0.0, 0.0)
    assert yaw == 0.03


def test_place_rsi_windows_follow_ordered_stage_entries() -> None:
    bounds = rsi_stage_bounds(
        trajectory_length=101,
        task_family="place",
        first_grasp=25,
        first_lift=50,
        stage_entries={1: 20, 2: 40, 3: 60, 4: 90},
    )
    assert bounds == {
        "approach": (0, 20),
        "grasp": (20, 39),
        "lift": (40, 59),
        "transport": (60, 89),
    }
