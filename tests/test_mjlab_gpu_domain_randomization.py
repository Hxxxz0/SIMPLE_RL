import math

import pytest
import torch

from simple.grasp_rl.mjlab_gpu.config import DomainRandomizationConfig
from simple.grasp_rl.mjlab_gpu.domain_randomization import (
    _apply_free_joint_pose,
    _apply_position_focus_mixture,
    _apply_position_focus_regions,
    _sample_stratified_positions,
    _stratified_cell_assignments,
    _target_translation_bounds,
    _validate_workspace_support_contract,
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


def test_empty_position_focus_regions_preserve_values_and_rng_state() -> None:
    generator = torch.Generator().manual_seed(456)
    before = generator.get_state().clone()
    translation = torch.tensor([[0.01, -0.02], [0.03, 0.04]])

    mixed = _apply_position_focus_regions(
        translation,
        regions=(),
        scale=1.0,
        generator=generator,
    )

    assert mixed is translation
    assert torch.equal(generator.get_state(), before)


def test_position_focus_regions_sample_each_rectangle_and_base_remainder() -> None:
    generator = torch.Generator().manual_seed(789)
    translation = torch.full((20_000, 2), 0.9)

    mixed = _apply_position_focus_regions(
        translation,
        regions=(
            (-0.06, -0.06, 0.01, 0.02, 0.3),
            (0.03, -0.03, 0.01, 0.01, 0.2),
        ),
        scale=1.0,
        generator=generator,
    )

    left = (mixed[:, 0] >= -0.07) & (mixed[:, 0] <= -0.05)
    left &= (mixed[:, 1] >= -0.08) & (mixed[:, 1] <= -0.04)
    right = (mixed[:, 0] >= 0.02) & (mixed[:, 0] <= 0.04)
    right &= (mixed[:, 1] >= -0.04) & (mixed[:, 1] <= -0.02)
    base = torch.all(mixed == 0.9, dim=1)

    assert left.float().mean().item() == pytest.approx(0.3, abs=0.015)
    assert right.float().mean().item() == pytest.approx(0.2, abs=0.015)
    assert base.float().mean().item() == pytest.approx(0.5, abs=0.015)


def test_stratified_position_sampling_covers_every_cell_and_resumes_cursor() -> None:
    generator = torch.Generator().manual_seed(810)
    positions, cell_ids, cursor = _sample_stratified_positions(
        66,
        jitter_xy=(0.2, 0.1),
        offset_center_xy=(0.01, -0.02),
        grid=(8, 8),
        scale=1.0,
        cursor=63,
        device="cpu",
        generator=generator,
    )

    assert cell_ids[:3].tolist() == [63, 0, 1]
    assert cursor == 1
    assert torch.bincount(cell_ids, minlength=64).min().item() == 1
    assert torch.all((positions[:, 0] >= -0.19) & (positions[:, 0] <= 0.21))
    assert torch.all((positions[:, 1] >= -0.12) & (positions[:, 1] <= 0.08))


def test_stratified_focus_preserves_full_grid_and_oversamples_selected_cells() -> None:
    assignments = _stratified_cell_assignments(
        torch.arange(1024), grid=(8, 8), focus_cell_ids=(0, 8, 63)
    )

    counts = torch.bincount(assignments, minlength=64)
    assert assignments[:64].tolist() == list(range(64))
    assert counts.min().item() == 1
    assert counts[[0, 8, 63]].sum().item() == 963
    assert counts.sum().item() == 1024


def test_stratified_curriculum_can_keep_workspace_center_fixed() -> None:
    positions, _, _ = _sample_stratified_positions(
        4096,
        jitter_xy=(0.2, 0.2),
        offset_center_xy=(0.0, -0.09),
        grid=(8, 8),
        scale=0.1,
        cursor=0,
        device="cpu",
        generator=torch.Generator().manual_seed(811),
        scale_offset_center=False,
    )

    assert positions[:, 0].mean().item() == pytest.approx(0.0, abs=0.001)
    assert positions[:, 1].mean().item() == pytest.approx(-0.09, abs=0.001)
    assert positions[:, 0].amin().item() >= -0.02
    assert positions[:, 0].amax().item() <= 0.02


def test_workspace_contract_checks_base_and_focus_target_ranges() -> None:
    config = DomainRandomizationConfig(
        target_position_jitter_xy=(0.2, 0.2),
        target_position_offset_center_xy=(0.0, -0.09),
        target_position_focus_regions=((0.0, -0.09, 0.05, 0.10, 0.5),),
    )
    assert _target_translation_bounds(config) == (
        pytest.approx((-0.2, 0.2)),
        pytest.approx((-0.29, 0.11)),
    )
    manifest = {
        "workspace_support_contract": {
            "translation_bounds_xy_m": [[-0.2, 0.2], [-0.29, 0.11]]
        }
    }
    _validate_workspace_support_contract(config, manifest)
    too_wide = DomainRandomizationConfig(
        target_position_jitter_xy=(0.21, 0.2),
        target_position_offset_center_xy=(0.0, -0.09),
    )
    with pytest.raises(ValueError, match="exceeds audited support"):
        _validate_workspace_support_contract(too_wide, manifest)
