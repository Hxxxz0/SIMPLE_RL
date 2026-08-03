import numpy as np
import pytest

from simple.grasp_rl.mjlab_gpu.collect import (
    _episode_arrays,
    _new_trace,
    validate_episode_arrays,
)


def _trace(steps=2):
    trace = _new_trace()
    for _ in range(steps):
        for name in (
            "raw_action",
            "reference_action",
            "effective_action",
            "physical_action",
        ):
            trace[name].append(np.zeros(36, dtype=np.float32))
        trace["observation"].append(np.zeros(331, dtype=np.float32))
        trace["policy_input"].append(np.zeros(842, dtype=np.float32))
        trace["reward"].append(np.asarray(1.0, dtype=np.float32))
        trace["task_reward"].append(np.asarray(0.9, dtype=np.float32))
        trace["reference_reward"].append(np.asarray(0.1, dtype=np.float32))
        trace["stage_index"].append(np.asarray(2, dtype=np.int64))
        trace["qpos"].append(np.zeros(50, dtype=np.float32))
        trace["qvel"].append(np.zeros(49, dtype=np.float32))
    return trace


def test_successful_trajectory_schema_includes_terminal_state() -> None:
    arrays = _episode_arrays(
        _trace(), np.ones(50, dtype=np.float32), np.ones(49, dtype=np.float32)
    )

    validate_episode_arrays(arrays)
    assert arrays["qpos"].shape == (3, 50)
    assert arrays["qvel"].shape == (3, 49)
    assert arrays["raw_action"].shape == (2, 36)
    assert arrays["done"].tolist() == [False, True]


def test_trajectory_schema_rejects_nonfinite_actions() -> None:
    arrays = _episode_arrays(
        _trace(), np.ones(50, dtype=np.float32), np.ones(49, dtype=np.float32)
    )
    arrays["physical_action"][0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        validate_episode_arrays(arrays)
