from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/grasp_rl/grasp_anything_bend_objects.py"
SPEC = importlib.util.spec_from_file_location("grasp_anything_bend_objects", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_bend_catalog_is_opt_in_and_low_object_only() -> None:
    assert set(RUNNER.OBJECTS) == {
        "Apple_1",
        "Bowl_1",
        "Potato_1",
        "Tomato_1",
    }
    assert RUNNER.REFERENCE_EPISODE == 11
    assert RUNNER.ROUTE_VERSION == "xmove_bend_ep11_v1"
    assert RUNNER.OBJECTS["Apple_1"].asset_version == "object_reward_v5_stable"
    assert RUNNER.OBJECTS["Apple_1"].table_clearance_m == pytest.approx(0.0)
    for object_id in RUNNER.OBJECTS:
        assert "xmove_bend_ep11" in str(RUNNER.asset_path(object_id))
        assert "episode82" not in str(RUNNER.asset_path(object_id))
        assert RUNNER.strict_reference_path(object_id) != RUNNER.staged_reference_path(
            object_id
        )


def test_stable_physics_assets_are_explicit_for_legacy_bend_objects() -> None:
    legacy = RUNNER.asset_path("Tomato_1")
    stable = RUNNER.asset_path("Tomato_1", stable_physics=True)
    legacy_arguments = RUNNER.environment_args(RUNNER.OBJECTS["Tomato_1"], "fixed", 123)
    stable_arguments = RUNNER.environment_args(
        RUNNER.OBJECTS["Tomato_1"], "fixed", 123, stable_physics=True
    )

    assert "object_reward_v4" in str(legacy)
    assert "object_reward_v5_stable" in str(stable)
    assert legacy != stable
    assert legacy_arguments[legacy_arguments.index("--asset-bundle") + 1] == str(legacy)
    assert stable_arguments[stable_arguments.index("--asset-bundle") + 1] == str(stable)
    assert RUNNER.asset_path("Apple_1") == RUNNER.asset_path(
        "Apple_1", stable_physics=True
    )
    assert RUNNER.OBJECTS["Tomato_1"].stable_table_clearance_m == pytest.approx(0.0)


def test_pose_profiles_expand_target_then_robot_base_position_dr() -> None:
    assert RUNNER.PROFILES["fixed"].target_jitter_xy_m == (0.0, 0.0)
    assert RUNNER.PROFILES["target_xy_2p5mm"].target_jitter_xy_m == pytest.approx(
        (0.0025, 0.0025)
    )
    assert RUNNER.PROFILES["target_xy_10mm"].target_jitter_xy_m == pytest.approx(
        (0.01, 0.01)
    )
    assert RUNNER.PROFILES["target_xy_25mm"].target_jitter_xy_m == pytest.approx(
        (0.025, 0.025)
    )
    assert RUNNER.PROFILES["target_xy_50mm"].target_jitter_xy_m == pytest.approx(
        (0.05, 0.05)
    )
    assert RUNNER.PROFILES["target_xy_100mm"].target_jitter_xy_m == pytest.approx(
        (0.1, 0.1)
    )
    assert RUNNER.PROFILES["target_xy_200mm"].target_jitter_xy_m == pytest.approx(
        (0.2, 0.2)
    )
    assert RUNNER.PROFILES[
        "target_base_xy_2p5mm"
    ].robot_base_jitter_xy_m == pytest.approx((0.0025, 0.0025))


def test_environment_uses_bend_reference_and_explicit_pose_dr() -> None:
    arguments = RUNNER.environment_args(
        RUNNER.OBJECTS["Apple_1"], "target_base_xy_2p5mm", 123
    )

    assert arguments[arguments.index("--strict-reference-episode") + 1] == "11"
    assert arguments[arguments.index("--target-position-jitter-xy") + 1 :][:2] == [
        "0.0025",
        "0.0025",
    ]
    assert arguments[arguments.index("--robot-base-position-jitter-xy") + 1 :][:2] == [
        "0.0025",
        "0.0025",
    ]
    assert arguments[arguments.index("--target-position-offset-center-xy") + 1 :][
        :2
    ] == ["0.0", "-0.09"]
    assert arguments[arguments.index("--target-mass-scale") + 1 :][:2] == ["1", "1"]


def test_reference_override_is_explicit_and_does_not_change_default() -> None:
    spec = RUNNER.OBJECTS["Apple_1"]
    default = RUNNER.environment_args(spec, "fixed", 123)
    override_path = Path("/tmp/apple_position_dr_reference")
    overridden = RUNNER.environment_args(
        spec,
        "fixed",
        123,
        reference_processed=override_path,
    )

    assert default[default.index("--reference-processed") + 1] == str(
        RUNNER.staged_reference_path("Apple_1")
    )
    assert overridden[overridden.index("--reference-processed") + 1] == str(
        override_path
    )
    assert RUNNER._reference_output_suffix(None) == ""
    assert RUNNER._reference_output_suffix(override_path) == (
        "_reference-apple_position_dr_reference"
    )


def test_train_cli_defaults_to_scratch_without_resume() -> None:
    args = RUNNER._parser().parse_args(
        ["train", "Apple_1", "--run-name", "isolated_test"]
    )

    assert args.profile == "target_xy_2p5mm"
    assert args.num_envs == 8192
    assert args.iterations == 40
    assert args.exploration_std == pytest.approx(0.01)
    assert args.exploration_hold_steps == 1
    assert args.ppo_steps_per_env == 24
    assert not hasattr(args, "checkpoint")
    assert args.reference_processed is None
    assert args.dr_initial_strength == pytest.approx(0.1)
    assert args.dr_ramp_steps == 480
    assert args.resume is None
    assert args.warm_start is None
    assert not args.warm_start_critic
    assert not args.freeze_actor_normalizer
    assert args.goal_potential_scale == pytest.approx(5.0)
    assert args.goal_potential_negative_clip == pytest.approx(0.25)
    assert args.success_bonus == pytest.approx(40.0)
    assert args.reference_reward_weight == pytest.approx(0.005)
    assert args.max_reference_action_deviation == pytest.approx(0.7)
    assert args.target_focus_probability == pytest.approx(0.0)
    assert args.target_focus_jitter_xy_m == pytest.approx((0.0, 0.0))
    assert args.target_focus_region == []
    assert not args.stable_physics


def test_train_cli_exposes_temporally_coherent_exploration() -> None:
    args = RUNNER._parser().parse_args(
        [
            "train",
            "Apple_1",
            "--run-name",
            "coherent-exploration",
            "--exploration-std",
            "0.08",
            "--exploration-hold-steps",
            "4",
            "--ppo-steps-per-env",
            "240",
            "--stable-physics",
            "--freeze-actor-normalizer",
        ]
    )

    assert args.exploration_std == pytest.approx(0.08)
    assert args.exploration_hold_steps == 4
    assert args.ppo_steps_per_env == 240
    assert args.stable_physics
    assert args.freeze_actor_normalizer


def test_train_forwards_long_rollout_and_legacy_default(monkeypatch, tmp_path) -> None:
    spec = RUNNER.OBJECTS["Apple_1"]
    reference = tmp_path / "reference"
    reference.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    calls = []

    monkeypatch.setattr(RUNNER, "run_root", lambda _object_id: tmp_path)
    monkeypatch.setattr(
        RUNNER, "_run_gpu", lambda arguments, **kwargs: calls.append(arguments)
    )
    monkeypatch.setattr(
        RUNNER, "_latest_checkpoint", lambda output: output / "model_19.pt"
    )

    RUNNER.train(
        spec,
        profile="fixed",
        gpu=0,
        seed=123,
        num_envs=1024,
        iterations=20,
        exploration_std=0.08,
        exploration_hold_steps=4,
        ppo_steps_per_env=240,
        run_name="long_rollout",
        reference_processed=reference,
        warm_start=checkpoint,
    )

    arguments = calls[0]
    assert arguments[arguments.index("--ppo-steps-per-env") + 1] == "240"


def test_training_focus_mixture_is_explicit_and_keeps_legacy_args_unchanged() -> None:
    spec = RUNNER.OBJECTS["Apple_1"]
    legacy = RUNNER.environment_args(spec, "target_xy_200mm", 123)
    focused = RUNNER.environment_args(
        spec,
        "target_xy_200mm",
        123,
        target_focus_probability=0.25,
        target_focus_jitter_xy_m=(0.025, 0.025),
    )

    assert "--target-position-focus-probability" not in legacy
    assert focused[focused.index("--target-position-focus-probability") + 1] == "0.25"
    assert focused[focused.index("--target-position-focus-jitter-xy") + 1 :][:2] == [
        "0.025",
        "0.025",
    ]
    assert focused[focused.index("--target-position-focus-offset-center-xy") + 1 :][
        :2
    ] == ["0.0", "-0.09"]


def test_multi_region_focus_is_repeatable_and_keeps_legacy_args_unchanged() -> None:
    spec = RUNNER.OBJECTS["Apple_1"]
    legacy = RUNNER.environment_args(spec, "target_xy_200mm", 123)
    focused = RUNNER.environment_args(
        spec,
        "target_xy_200mm",
        123,
        target_focus_regions=(
            (-0.06, -0.065, 0.025, 0.055, 0.2),
            (0.025, -0.065, 0.025, 0.055, 0.2),
        ),
    )

    assert "--target-position-focus-region" not in legacy
    starts = [
        index
        for index, argument in enumerate(focused)
        if argument == "--target-position-focus-region"
    ]
    assert len(starts) == 2
    assert focused[starts[0] + 1 : starts[0] + 6] == [
        "-0.06",
        "-0.065",
        "0.025",
        "0.055",
        "0.2",
    ]
    assert focused[starts[1] + 1 : starts[1] + 6] == [
        "0.025",
        "-0.065",
        "0.025",
        "0.055",
        "0.2",
    ]

    parsed = RUNNER._parser().parse_args(
        [
            "train",
            "Apple_1",
            "--run-name",
            "frontier",
            "--target-focus-region",
            "-0.06",
            "-0.065",
            "0.025",
            "0.055",
            "0.2",
            "--target-focus-region",
            "0.025",
            "-0.065",
            "0.025",
            "0.055",
            "0.2",
        ]
    )
    assert parsed.target_focus_region == [
        [-0.06, -0.065, 0.025, 0.055, 0.2],
        [0.025, -0.065, 0.025, 0.055, 0.2],
    ]
    assert RUNNER._target_focus_output_suffix(()) == ""
    assert RUNNER._target_focus_output_suffix(
        ((-0.06, -0.065, 0.025, 0.055, 0.2),)
    ).startswith("_focusregions-")
    assert RUNNER._focus_checkpoint_output_suffix(Path("model.pt"), ()) == ""
    assert RUNNER._focus_checkpoint_output_suffix(
        Path("model.pt"), ((-0.06, -0.065, 0.025, 0.055, 0.2),)
    ).startswith("_checkpoint-model-")


def test_evaluate_cli_reward_alignment_is_explicit_and_defaults_stay_legacy() -> None:
    default = RUNNER._parser().parse_args(["evaluate", "Apple_1"])
    aligned = RUNNER._parser().parse_args(
        [
            "evaluate",
            "Apple_1",
            "--goal-potential-scale",
            "20",
            "--goal-potential-negative-clip",
            "1",
            "--success-bonus",
            "80",
            "--reference-reward-weight",
            "0",
            "--max-reference-action-deviation",
            "2",
        ]
    )

    assert default.goal_potential_scale == pytest.approx(5.0)
    assert default.goal_potential_negative_clip == pytest.approx(0.25)
    assert default.success_bonus == pytest.approx(40.0)
    assert aligned.goal_potential_scale == pytest.approx(20.0)
    assert aligned.goal_potential_negative_clip == pytest.approx(1.0)
    assert aligned.success_bonus == pytest.approx(80.0)
    assert aligned.reference_reward_weight == pytest.approx(0.0)
    assert aligned.max_reference_action_deviation == pytest.approx(2.0)


def test_train_cli_exact_resume_is_explicit() -> None:
    args = RUNNER._parser().parse_args(
        [
            "train",
            "Apple_1",
            "--run-name",
            "resume_test",
            "--resume",
            "/tmp/model_399.pt",
        ]
    )

    assert args.resume == Path("/tmp/model_399.pt")


def test_train_cli_reward_aligned_warm_start_is_explicit() -> None:
    args = RUNNER._parser().parse_args(
        [
            "train",
            "Apple_1",
            "--run-name",
            "aligned_test",
            "--warm-start",
            "/tmp/model_399.pt",
            "--goal-potential-scale",
            "20",
            "--goal-potential-negative-clip",
            "1",
            "--success-bonus",
            "80",
        ]
    )

    assert args.warm_start == Path("/tmp/model_399.pt")
    assert args.goal_potential_scale == pytest.approx(20.0)
    assert args.goal_potential_negative_clip == pytest.approx(1.0)
    assert args.success_bonus == pytest.approx(80.0)


def test_latest_checkpoint_handles_resumed_iteration_numbers(tmp_path: Path) -> None:
    for iteration in (400, 405, 799):
        (tmp_path / f"model_{iteration}.pt").touch()
    (tmp_path / "model_initial.pt").touch()

    assert RUNNER._latest_checkpoint(tmp_path) == tmp_path / "model_799.pt"
