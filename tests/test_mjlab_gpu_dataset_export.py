import numpy as np

from simple.grasp_rl.mjlab_gpu.dataset_export import (
    RAW_ACTION_NAMES,
    _physical_to_joint_command,
    _psi_arrays,
)
from simple.grasp_rl.schema import JOINT_NAMES


def _raw_to_psi_action(
    raw_action: np.ndarray, physical_action: np.ndarray
) -> np.ndarray:
    by_name = {
        name: raw_action[:, index] for index, name in enumerate(RAW_ACTION_NAMES)
    }
    simple_order = np.stack([by_name[name] for name in JOINT_NAMES], axis=1)
    return np.concatenate(
        (
            simple_order[:, 29:32],
            simple_order[:, 34:36],
            simple_order[:, 32:34],
            simple_order[:, 36:43],
            simple_order[:, 15:22],
            simple_order[:, 22:29],
            simple_order[:, (13, 14, 12)],
            physical_action[:, 31:36],
        ),
        axis=1,
    )


def test_groot_joint_command_round_trips_to_physical_psi_action() -> None:
    rng = np.random.default_rng(7)
    physical = rng.normal(size=(4, 36)).astype(np.float32)
    joint_target = rng.normal(size=(4, 43)).astype(np.float32)

    command = _physical_to_joint_command(physical, joint_target)
    raw_order = np.asarray([JOINT_NAMES.index(name) for name in RAW_ACTION_NAMES])
    reconstructed = _raw_to_psi_action(command[:, raw_order], physical)

    np.testing.assert_allclose(reconstructed, physical)


def test_psi_arrays_match_public_training_shapes() -> None:
    joint_state = np.zeros((3, 43), dtype=np.float64)
    physical = np.zeros((3, 36), dtype=np.float32)

    result = _psi_arrays(joint_state, physical)

    assert result["states"].shape == (3, 32)
    assert result["action"].shape == (3, 36)
    assert result["observation.hand_joints"].shape == (3, 14)
    assert result["observation.arm_joints"].shape == (3, 14)
    assert result["observation.leg_joints"].shape == (3, 15)
