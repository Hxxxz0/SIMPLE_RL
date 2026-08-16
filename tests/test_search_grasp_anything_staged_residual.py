from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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


def _required_cli_args() -> list[str]:
    return [
        "--asset",
        "asset",
        "--reference",
        "reference",
        "--output",
        "result.json",
        "--base-arm",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
    ]


def test_reference_episode_defaults_to_legacy_episode_82() -> None:
    args = SEARCH._parser().parse_args(_required_cli_args())

    assert args.reference_episode == 82
    assert SEARCH.reference_episode_path(
        args.reference, args.reference_episode
    ) == Path("reference/bc/episode_000082.npz")
    assert args.target_position_jitter_xy == (0.0, 0.0)
    assert args.robot_base_position_jitter_xy == (0.0, 0.0)
    assert args.target_yaw_jitter == 0.0
    assert args.robot_base_yaw_jitter == 0.0
    assert args.stop_grasp_candidates == 8
    assert args.stop_success_candidates is None
    assert args.grasp_streak_score_cap == 20
    assert args.replay_best_candidate_replicas == 0
    assert args.capture_arm_release_residual_scale == 0.0
    assert not args.export_search_rollout
    assert not args.search_body
    assert SEARCH._body_bounds(args) == ()
    assert args.body_start is None
    assert args.body_full_step is None
    assert not args.search_navigation
    assert SEARCH._navigation_bounds(args) == ()


def test_pose_randomization_is_explicitly_opt_in() -> None:
    args = SEARCH._parser().parse_args(
        [
            *_required_cli_args(),
            "--target-position-jitter-xy",
            "0.0025",
            "0.003",
            "--target-yaw-jitter",
            "0.01",
            "--robot-base-position-jitter-xy",
            "0.001",
            "0.002",
            "--robot-base-yaw-jitter",
            "0.005",
        ]
    )

    assert args.target_position_jitter_xy == [0.0025, 0.003]
    assert args.target_yaw_jitter == pytest.approx(0.01)
    assert args.robot_base_position_jitter_xy == [0.001, 0.002]
    assert args.robot_base_yaw_jitter == pytest.approx(0.005)


def test_reference_episode_can_select_bend_pick_episode_11() -> None:
    args = SEARCH._parser().parse_args(
        [*_required_cli_args(), "--reference-episode", "11"]
    )

    assert args.reference_episode == 11
    assert SEARCH.reference_episode_path(
        args.reference, args.reference_episode
    ) == Path("reference/bc/episode_000011.npz")


def test_reference_episode_path_rejects_negative_episode() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SEARCH.reference_episode_path(Path("reference"), -1)


def test_arm_bounds_default_to_legacy_search_domain() -> None:
    args = SEARCH._parser().parse_args(_required_cli_args())

    assert SEARCH._arm_bounds(args) == SEARCH.ARM_BOUNDS


def test_arm_bounds_allow_an_explicit_isolated_search_domain() -> None:
    values = [value for index in range(7) for value in (-1.0, index + 1.0)]
    args = SEARCH._parser().parse_args(
        [*_required_cli_args(), "--arm-bounds", *map(str, values)]
    )

    assert SEARCH._arm_bounds(args) == tuple((-1.0, index + 1.0) for index in range(7))


def test_arm_bounds_reject_unordered_pair() -> None:
    args = SEARCH._parser().parse_args(_required_cli_args())
    args.arm_bounds = [0.0, 0.0, *args.arm_bounds[2:]]

    with pytest.raises(ValueError, match="strictly ordered"):
        SEARCH._arm_bounds(args)


def test_two_stage_search_uses_independent_capture_arm_parameters() -> None:
    args = SEARCH._parser().parse_args(
        [
            *_required_cli_args(),
            "--search-capture-arm",
            "--base-capture-arm",
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.6",
            "0.7",
        ]
    )

    mean, std = SEARCH._initial_distribution(args)

    assert mean.shape == std.shape == (18,)
    assert SEARCH._candidate_capture_arm(mean).tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    assert SEARCH._candidate_aperture(mean).tolist() == pytest.approx(
        list(args.aperture_mean)
    )
    assert SEARCH._capture_arm_bounds(args) == ((-0.25, 0.25),) * 7


def test_body_search_is_opt_in_and_preserves_unambiguous_layouts() -> None:
    args = SEARCH._parser().parse_args(
        [
            *_required_cli_args(),
            "--search-body",
            "--base-body",
            "0.1",
            "0.2",
            "0.3",
            "0.4",
        ]
    )

    mean, std = SEARCH._initial_distribution(args)

    assert mean.shape == std.shape == (15,)
    assert SEARCH._candidate_capture_arm(mean).tolist() == pytest.approx([0.0] * 7)
    assert SEARCH._candidate_body(mean).tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )
    assert SEARCH._candidate_aperture(mean).tolist() == pytest.approx(
        list(args.aperture_mean)
    )
    assert SEARCH._body_bounds(args) == ((-1.0, 1.0),) * 4


def test_staged_action_blends_and_releases_body_correction() -> None:
    reference = torch.zeros(1, 36)
    candidates = torch.zeros(1, 15)
    candidates[:, 7:11] = 0.4

    action = SEARCH._action(
        reference,
        candidates,
        capture_arm=None,
        arm_weight=0.5,
        close_weight=0.0,
        hand_active=False,
        arm_release_weight=0.25,
    )

    assert torch.allclose(action[:, 28:32], torch.full((1, 4), 0.15))


def test_body_phase_can_finish_before_arm_capture() -> None:
    assert SEARCH.body_phase_blend(
        50, body_start=0, body_full_step=100, fallback_weight=0.1
    ) == pytest.approx(0.5)
    assert SEARCH.body_phase_blend(
        50, body_start=None, body_full_step=None, fallback_weight=0.1
    ) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="strictly ordered"):
        SEARCH.body_phase_blend(
            50, body_start=100, body_full_step=50, fallback_weight=0.1
        )

    reference = torch.zeros(1, 36)
    candidates = torch.zeros(1, 15)
    candidates[:, 7:11] = 0.4
    action = SEARCH._action(
        reference,
        candidates,
        capture_arm=None,
        arm_weight=0.1,
        body_weight=0.5,
        close_weight=0.0,
        hand_active=False,
        arm_release_weight=0.25,
    )
    assert torch.allclose(action[:, 28:32], torch.full((1, 4), 0.15))


def test_navigation_search_is_opt_in_and_releases_before_capture() -> None:
    args = SEARCH._parser().parse_args(
        [
            *_required_cli_args(),
            "--search-navigation",
            "--base-navigation",
            "0.3",
            "-0.2",
        ]
    )
    mean, std = SEARCH._initial_distribution(args)

    assert mean.shape == std.shape == (13,)
    assert SEARCH._candidate_navigation(mean).tolist() == pytest.approx([0.3, -0.2])
    assert SEARCH._candidate_aperture(mean).tolist() == pytest.approx(
        list(args.aperture_mean)
    )
    assert SEARCH._navigation_bounds(args) == ((-1.0, 1.0),) * 2

    action = SEARCH._action(
        torch.zeros(1, 36, device=mean.device),
        mean[None],
        capture_arm=None,
        arm_weight=0.5,
        close_weight=0.0,
        hand_active=False,
        arm_release_weight=0.0,
        navigation_release_weight=0.25,
    )
    assert action[0, 32:34].tolist() == pytest.approx([0.1125, -0.075])


def test_capture_body_navigation_layout_remains_unambiguous() -> None:
    args = SEARCH._parser().parse_args(
        [
            *_required_cli_args(),
            "--search-capture-arm",
            "--search-body",
            "--search-navigation",
        ]
    )
    mean, std = SEARCH._initial_distribution(args)

    assert mean.shape == std.shape == (24,)
    assert SEARCH._candidate_capture_arm(mean).shape == (7,)
    assert SEARCH._candidate_body(mean).shape == (4,)
    assert SEARCH._candidate_navigation(mean).shape == (2,)
    assert SEARCH._candidate_aperture(mean).shape == (4,)


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


def test_staged_action_can_retain_capture_residual_during_lift() -> None:
    reference = torch.zeros(1, 36)
    reference[:, 21:28] = 0.6
    candidates = torch.zeros(1, 18)
    candidates[:, 7:14] = 0.2
    capture = torch.full((1, 7), -0.4)

    action = SEARCH._action(
        reference,
        candidates,
        capture_arm=capture,
        arm_weight=1.0,
        close_weight=0.0,
        hand_active=False,
        arm_release_weight=0.25,
        capture_arm_release_residual_scale=0.5,
    )

    assert torch.allclose(action[:, 21:28], torch.full((1, 7), -0.125))


def test_capture_residual_scale_rejects_extrapolation() -> None:
    with pytest.raises(ValueError, match="residual scale"):
        SEARCH._action(
            torch.zeros(1, 36),
            torch.zeros(1, 18),
            capture_arm=torch.zeros(1, 7),
            arm_weight=1.0,
            close_weight=0.0,
            hand_active=False,
            arm_release_weight=1.0,
            capture_arm_release_residual_scale=1.1,
        )


def test_staged_action_can_release_overhead_detour_before_capture() -> None:
    reference = torch.zeros(1, 36)
    reference[:, 21:28] = 0.6
    candidates = torch.zeros(1, 11)
    candidates[:, :7] = -0.4

    action = SEARCH._action(
        reference,
        candidates,
        capture_arm=None,
        arm_weight=1.0,
        close_weight=0.0,
        hand_active=False,
        arm_release_weight=0.25,
    )

    assert torch.allclose(action[:, 21:28], torch.full((1, 7), 0.3))


def test_invalid_phase_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly ordered"):
        SEARCH.phase_blend(0, arm_start=10, capture_step=30, close_end=20)


def test_arm_release_blend_is_optional_and_ordered() -> None:
    assert SEARCH.arm_release_blend(20, release_start=None, release_end=None) == 0.0
    assert SEARCH.arm_release_blend(20, release_start=10, release_end=30) == 0.5
    with pytest.raises(ValueError, match="strictly ordered"):
        SEARCH.arm_release_blend(20, release_start=30, release_end=10)


def test_two_stage_search_has_independent_overhead_and_capture_release() -> None:
    args = SEARCH._parser().parse_args(
        [
            *_required_cli_args(),
            "--search-capture-arm",
            "--arm-release-start",
            "100",
            "--arm-release-end",
            "180",
            "--capture-arm-release-start",
            "250",
            "--capture-arm-release-end",
            "300",
        ]
    )

    assert SEARCH._effective_arm_release_weight(
        args,
        step=140,
        capture_active=False,
        overhead_release_weight=0.5,
    ) == pytest.approx(0.5)
    assert SEARCH._effective_arm_release_weight(
        args,
        step=275,
        capture_active=True,
        overhead_release_weight=1.0,
    ) == pytest.approx(0.5)


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


def test_long_grasp_score_cap_is_explicitly_opt_in() -> None:
    kwargs = {
        "success": torch.zeros(2, dtype=torch.bool),
        "max_grasp_streak": torch.tensor([20, 40]),
        "max_bilateral_streak": torch.tensor([20, 40]),
        "min_opposing_force": torch.zeros(2),
        "balanced_shell_gap": torch.zeros(2),
        "finger_opposition": torch.zeros(2),
        "lift_height": torch.zeros(2),
    }

    legacy = SEARCH.candidate_score(**kwargs)
    long_hold = SEARCH.candidate_score(**kwargs, grasp_streak_score_cap=120)

    assert legacy[0] == legacy[1]
    assert long_hold[1] > long_hold[0]


def test_native_success_stop_replaces_legacy_grasp_stop() -> None:
    record = {"grasp_candidates": 12, "success_candidates": 0}

    assert SEARCH.should_stop_search(
        record,
        stop_grasp_candidates=8,
        stop_success_candidates=None,
    )
    assert not SEARCH.should_stop_search(
        record,
        stop_grasp_candidates=8,
        stop_success_candidates=1,
    )
    record["success_candidates"] = 1
    assert SEARCH.should_stop_search(
        record,
        stop_grasp_candidates=8,
        stop_success_candidates=1,
    )


def test_search_stop_thresholds_must_be_positive() -> None:
    record = {"grasp_candidates": 0, "success_candidates": 0}

    with pytest.raises(ValueError, match="stop grasp"):
        SEARCH.should_stop_search(
            record,
            stop_grasp_candidates=0,
            stop_success_candidates=None,
        )
    with pytest.raises(ValueError, match="stop success"):
        SEARCH.should_stop_search(
            record,
            stop_grasp_candidates=8,
            stop_success_candidates=0,
        )


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
    mean = torch.tensor([-0.5, 0.0, 0.4, 0.5, -0.2, 0.3, 0.0, 0.4, 0.2, 0.2, 0.2])
    std = torch.full_like(mean, 0.01)

    candidates = SEARCH._sample_candidates(
        mean,
        std,
        count=32,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(candidates[0], mean)
    for index, (low, high) in enumerate((*SEARCH.ARM_BOUNDS, *SEARCH.APERTURE_BOUNDS)):
        expected_low = mean.clone()
        expected_low[index] = low
        expected_high = mean.clone()
        expected_high[index] = high
        assert torch.equal(candidates[1 + 2 * index], expected_low)
        assert torch.equal(candidates[2 + 2 * index], expected_high)


def test_candidate_replay_repeats_a_known_solution_across_worlds() -> None:
    candidates = torch.randn(8, 11)
    known = torch.arange(11, dtype=torch.float32)

    SEARCH.replay_candidate_across_worlds(candidates, known, replicas=3)

    assert torch.equal(candidates[:3], known.expand(3, -1))
    assert not torch.equal(candidates[3], known)


def test_candidate_replay_validates_replica_count_and_shape() -> None:
    candidates = torch.zeros(8, 11)

    with pytest.raises(ValueError, match="non-negative"):
        SEARCH.replay_candidate_across_worlds(candidates, torch.zeros(11), replicas=-1)
    with pytest.raises(ValueError, match="environment count"):
        SEARCH.replay_candidate_across_worlds(candidates, torch.zeros(11), replicas=9)
    with pytest.raises(ValueError, match="parameter shape"):
        SEARCH.replay_candidate_across_worlds(candidates, torch.zeros(18), replicas=1)


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

    assert SEARCH.replay_quality_key(repeatable) > SEARCH.replay_quality_key(long_grasp)


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


def test_search_rollout_path_is_isolated_by_round_and_world(tmp_path: Path) -> None:
    path = SEARCH.search_rollout_path(
        tmp_path / "search.json", round_index=3, env_id=17
    )

    assert path == (
        tmp_path / "search_rollouts" / "round_003_env_0017.npz"
    ).resolve()


def test_search_rollout_path_rejects_negative_indices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SEARCH.search_rollout_path(
            tmp_path / "search.json", round_index=-1, env_id=0
        )


def test_load_search_rollout_requires_strict_native_success(tmp_path: Path) -> None:
    path = tmp_path / "success.npz"
    torch_observations = torch.zeros(14, 331).numpy()
    torch_actions = torch.zeros(14, 36).numpy()
    np.savez_compressed(
        path,
        observations=torch_observations,
        raw_actions=torch_actions,
        terminal_success=np.asarray(True),
        terminal_step=np.asarray(14),
        max_lift_m=np.asarray(0.10),
        max_grasp_streak=np.asarray(14),
        max_bilateral_contact_streak=np.asarray(13),
        max_min_opposing_force_n=np.asarray(5.0),
    )

    loaded = SEARCH.load_search_rollout(
        {"env_id": 7, "search_rollout_path": str(path)}
    )

    assert loaded["trajectory_source"] == "captured_search_rollout"
    assert loaded["successful_world_id"] == 7
    assert loaded["terminal_step"] == 14
    assert loaded["independent_replay_success_rate_at_first_terminal_step"] is None


def test_load_search_rollout_rejects_subthreshold_lift(tmp_path: Path) -> None:
    path = tmp_path / "failure.npz"
    np.savez_compressed(
        path,
        observations=np.zeros((13, 331), dtype=np.float32),
        raw_actions=np.zeros((13, 36), dtype=np.float32),
        terminal_success=np.asarray(True),
        terminal_step=np.asarray(13),
        max_lift_m=np.asarray(0.08),
        max_grasp_streak=np.asarray(13),
        max_bilateral_contact_streak=np.asarray(13),
        max_min_opposing_force_n=np.asarray(5.0),
    )

    with pytest.raises(RuntimeError, match="strict physical margins"):
        SEARCH.load_search_rollout(
            {"env_id": 0, "search_rollout_path": str(path)}
        )
