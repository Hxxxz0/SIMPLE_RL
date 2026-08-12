import math

import torch

from simple.grasp_rl.mjlab_gpu.domain_randomization import (
    _apply_free_joint_pose,
    _apply_position_focus_mixture,
)
from simple.grasp_rl.mjlab_gpu.vec_env import _nonfinite_output_rows


def test_free_joint_pose_randomization_updates_xy_and_yaw() -> None:
    qpos = torch.zeros(2, 7)
    qpos[:, 2] = 0.5
    qpos[:, 3] = 1.0
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    translation = torch.tensor([[0.02, -0.03], [-0.01, 0.04]])
    yaw = torch.tensor([math.pi / 2, -math.pi])

    _apply_free_joint_pose(qpos, env_ids, 0, translation, yaw)

    torch.testing.assert_close(qpos[:, :2], translation)
    torch.testing.assert_close(qpos[:, 2], torch.full((2,), 0.5))
    torch.testing.assert_close(
        qpos[:, 3:],
        torch.tensor(
            [
                [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
                [0.0, 0.0, 0.0, -1.0],
            ]
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_nonfinite_output_rows_unions_fields_and_counts_values() -> None:
    affected, values = _nonfinite_output_rows(
        {
            "qacc": torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
            "sensors": torch.tensor([[1.0], [float("inf")]]),
        }
    )
    assert affected.tolist() == [True, True]
    assert values == 2


def test_disabled_position_focus_preserves_values_and_rng_state() -> None:
    generator = torch.Generator().manual_seed(123)
    before = generator.get_state().clone()
    translation = torch.tensor([[0.01, -0.02], [0.03, 0.04]])

    mixed = _apply_position_focus_mixture(
        translation,
        probability=0.0,
        jitter_xy=(0.1, 0.1),
        offset_center_xy=(0.2, 0.2),
        scale=0.5,
        generator=generator,
    )

    assert mixed is translation
    assert torch.equal(generator.get_state(), before)


def test_position_focus_probability_one_samples_scaled_region() -> None:
    generator = torch.Generator().manual_seed(321)
    translation = torch.full((4096, 2), -1.0)

    mixed = _apply_position_focus_mixture(
        translation,
        probability=1.0,
        jitter_xy=(0.01, 0.02),
        offset_center_xy=(0.04, 0.03),
        scale=0.2,
        generator=generator,
    )

    assert torch.all((mixed[:, 0] >= 0.006) & (mixed[:, 0] <= 0.01))
    assert torch.all((mixed[:, 1] >= 0.002) & (mixed[:, 1] <= 0.01))
