from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/grasp_rl/search_grasp_anything_staged_residual.py"
)
SPEC = importlib.util.spec_from_file_location("staged_residual_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEARCH)


def test_phase_blend_has_separate_approach_and_close_windows() -> None:
    kwargs = {"arm_start": 10, "capture_step": 20, "close_end": 40}
    assert SEARCH.phase_blend(0, **kwargs) == (0.0, 0.0)
    assert SEARCH.phase_blend(15, **kwargs) == pytest.approx((0.5, 0.0))
    assert SEARCH.phase_blend(20, **kwargs) == (1.0, 0.0)
    assert SEARCH.phase_blend(30, **kwargs) == pytest.approx((1.0, 0.5))
    assert SEARCH.phase_blend(45, **kwargs) == (1.0, 1.0)


def test_staged_action_keeps_hand_and_arm_isolated() -> None:
    reference = torch.zeros(2, 36)
    candidates = torch.zeros(2, 11)
    candidates[:, :7] = 0.2
    candidates[:, 7:] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    action = SEARCH._action(
        reference,
        candidates,
        capture_arm=None,
        arm_weight=1.0,
        close_weight=0.5,
        hand_active=True,
    )
    expected_hand = 0.5 * SEARCH.aperture_hand_target(candidates[:, 7:])
    assert torch.allclose(action[:, 7:14], expected_hand)
    assert torch.allclose(action[:, 21:28], torch.full((2, 7), 0.2))
    inactive = torch.ones(36, dtype=torch.bool)
    inactive[7:14] = False
    inactive[21:28] = False
    assert torch.count_nonzero(action[:, inactive]) == 0


def test_staged_action_can_release_captured_arm_to_reference() -> None:
    reference = torch.zeros(1, 36)
    reference[:, 21:28] = 0.6
    candidates = torch.zeros(1, 11)
    capture = torch.full((1, 7), -0.4)

    action = SEARCH._action(
        reference,
        candidates,
        capture_arm=capture,
        arm_weight=1.0,
        close_weight=0.0,
        hand_active=False,
        arm_release_weight=0.25,
    )

    assert torch.allclose(action[:, 21:28], torch.full((1, 7), -0.15))


def test_invalid_phase_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly ordered"):
        SEARCH.phase_blend(
            0, arm_start=10, capture_step=30, close_end=20
        )


def test_arm_release_blend_is_optional_and_ordered() -> None:
    assert SEARCH.arm_release_blend(
        20, release_start=None, release_end=None
    ) == 0.0
    assert SEARCH.arm_release_blend(20, release_start=10, release_end=30) == 0.5
    with pytest.raises(ValueError, match="strictly ordered"):
        SEARCH.arm_release_blend(20, release_start=30, release_end=10)


def test_aperture_mapping_preserves_joint_group_symmetry() -> None:
    aperture = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    target = SEARCH.aperture_hand_target(aperture)
    assert target.tolist()[0] == pytest.approx(
        [-0.59, -0.69, -0.69, 0.79, 0.79, 0.89, 0.89]
    )


def test_thumb_abduction_search_bound_stays_within_public_action_range() -> None:
    low, high = SEARCH.APERTURE_BOUNDS[0]
    targets = SEARCH.aperture_hand_target(
        torch.tensor([[low, 0.0, 0.0, 0.0], [high, 0.0, 0.0, 0.0]])
    )[:, 0]

    assert targets.tolist() == pytest.approx([-0.89, 0.81])
    assert targets.abs().max() <= 1.0


def test_shell_metrics_require_thumb_and_support_near_same_shell() -> None:
    center = torch.zeros(2, 3)
    distal = torch.tensor(
        [
            [[0.032, 0.0, 0.0], [-0.032, 0.0, 0.0], [0.08, 0.0, 0.0]],
            [[0.08, 0.0, 0.0], [-0.032, 0.0, 0.0], [0.0, 0.032, 0.0]],
        ]
    )

    distances, gaps, balanced = SEARCH.fingertip_shell_metrics(
        distal, center, grip_width_m=0.064
    )

    assert distances[0].tolist() == pytest.approx([0.032, 0.032, 0.08])
    assert gaps[0].tolist() == pytest.approx([0.0, 0.0, 0.048])
    assert balanced.tolist() == pytest.approx([0.0, 0.048])


def test_precontact_score_is_driven_by_balanced_shell_gap() -> None:
    zeros = torch.zeros(2)
    score = SEARCH.candidate_score(
        success=torch.zeros(2, dtype=torch.bool),
        max_grasp_streak=zeros.long(),
        max_bilateral_streak=zeros.long(),
        min_opposing_force=zeros,
        balanced_shell_gap=torch.tensor([0.01, 0.08]),
        finger_opposition=torch.tensor([0.0, 1.0]),
        lift_height=zeros,
    )

    assert score[0] > score[1]


def test_precontact_score_does_not_reward_object_motion_without_grasp() -> None:
    score = SEARCH.candidate_score(
        success=torch.zeros(2, dtype=torch.bool),
        max_grasp_streak=torch.tensor([0, 2]),
        max_bilateral_streak=torch.zeros(2, dtype=torch.long),
        min_opposing_force=torch.zeros(2),
        balanced_shell_gap=torch.full((2,), 0.02),
        finger_opposition=torch.zeros(2),
        lift_height=torch.full((2,), 0.09),
    )

    assert score[0] == pytest.approx(-2.0)
    assert score[1] > score[0]


def test_surface_gap_requires_thumb_and_one_support_near_collision_surface() -> None:
    gaps = SEARCH.balanced_surface_gap(
        torch.tensor(
            [
                [0.002, 0.003, 0.080],
                [0.002, 0.050, 0.060],
                [0.040, 0.001, 0.002],
            ]
        )
    )

    assert gaps.tolist() == pytest.approx([0.003, 0.050, 0.040])


def test_surface_gap_rejects_non_fingertip_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        SEARCH.balanced_surface_gap(torch.zeros(2, 4))


def test_parameter_bound_mask_reports_clamped_dimensions() -> None:
    candidates = torch.tensor(
        [
            [
                -0.75,
                0.0,
                0.4,
                0.5,
                0.0,
                0.2,
                0.0,
                0.3,
                0.2,
                0.2,
                0.2,
            ]
        ]
    )

    mask = SEARCH._parameter_bound_mask(candidates)

    assert mask.shape == candidates.shape
    assert mask[0].nonzero().flatten().tolist() == [0]


def test_candidate_sampling_includes_independent_bound_probes() -> None:
    mean = torch.tensor(
        [-0.5, 0.0, 0.4, 0.5, -0.2, 0.3, 0.0, 0.4, 0.2, 0.2, 0.2]
    )
    std = torch.full_like(mean, 0.01)

    candidates = SEARCH._sample_candidates(
        mean,
        std,
        count=32,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(candidates[0], mean)
    for index, (low, high) in enumerate(
        (*SEARCH.ARM_BOUNDS, *SEARCH.APERTURE_BOUNDS)
    ):
        expected_low = mean.clone()
        expected_low[index] = low
        expected_high = mean.clone()
        expected_high[index] = high
        assert torch.equal(candidates[1 + 2 * index], expected_low)
        assert torch.equal(candidates[2 + 2 * index], expected_high)


def test_export_candidate_ranking_prefers_dense_non_bound_successes() -> None:
    def candidate(
        arm_0: float, *, env_id: int, bound: bool = False
    ) -> dict[str, object]:
        return {
            "round": 0,
            "env_id": env_id,
            "had_success": True,
            "right_arm_correction": [arm_0, 0.0, 0.4, 0.5, -0.2, 0.3, 0.0],
            "aperture": [0.4, 0.2, 0.2, 0.2],
            "candidate_bound_parameters": ["arm_0"] if bound else [],
            "max_grasp_streak": 20,
            "max_bilateral_contact_streak": 20,
            "max_lift_m": 0.12,
            "max_min_opposing_force_n": 5.0,
        }

    dense = [candidate(-0.50 + 0.001 * index, env_id=index) for index in range(6)]
    isolated = candidate(-0.30, env_id=10)
    bound = candidate(-0.75, env_id=11, bound=True)
    result = {
        "rounds": [{"successful_candidates": [*dense, isolated, bound]}],
        "global_best": dense[0],
    }

    ranked = SEARCH.rank_export_candidates(result, limit=8)

    assert [item["env_id"] for item in ranked[:6]] == [2, 3, 1, 4, 0, 5]
    assert ranked[-1]["env_id"] == 11
    assert all("success_neighbor_radius" in item for item in ranked)


def test_export_candidate_ranking_deduplicates_global_best() -> None:
    candidate = {
        "had_success": True,
        "right_arm_correction": [-0.5, 0.0, 0.4, 0.5, -0.2, 0.3, 0.0],
        "aperture": [0.4, 0.2, 0.2, 0.2],
        "candidate_bound_parameters": [],
    }
    result = {
        "rounds": [{"successful_candidates": [candidate]}],
        "global_best": dict(candidate),
    }

    assert len(SEARCH.rank_export_candidates(result, limit=4)) == 1


def test_export_candidate_ranking_requires_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        SEARCH.rank_export_candidates({}, limit=0)


def test_replay_quality_prefers_repeatability_before_physical_margin() -> None:
    def replay(success_rate: float, *, grasp_streak: int) -> dict[str, object]:
        return {
            "independent_replay_success_rate_at_first_terminal_step": success_rate,
            "max_grasp_streak": grasp_streak,
            "max_bilateral_contact_streak": 100,
            "max_lift_m": 0.15,
            "max_min_opposing_force_n": 8.0,
        }

    repeatable = replay(0.12, grasp_streak=90)
    long_grasp = replay(0.08, grasp_streak=120)

    assert SEARCH.replay_quality_key(repeatable) > SEARCH.replay_quality_key(
        long_grasp
    )


def test_replay_quality_uses_physical_margin_to_break_rate_ties() -> None:
    baseline = {
        "independent_replay_success_rate_at_first_terminal_step": 0.12,
        "max_grasp_streak": 100,
        "max_bilateral_contact_streak": 90,
        "max_lift_m": 0.15,
        "max_min_opposing_force_n": 8.0,
    }
    stronger = dict(baseline, max_grasp_streak=101)

    assert SEARCH.replay_quality_key(stronger) > SEARCH.replay_quality_key(baseline)
