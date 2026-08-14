from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/grasp_rl/grasp_anything_bend_objects.py"
)
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
    for object_id in RUNNER.OBJECTS:
        assert "xmove_bend_ep11" in str(RUNNER.asset_path(object_id))
        assert "episode82" not in str(RUNNER.asset_path(object_id))
        assert RUNNER.strict_reference_path(object_id) != RUNNER.staged_reference_path(
            object_id
        )


def test_pose_profiles_expand_target_then_robot_base_position_dr() -> None:
    assert RUNNER.PROFILES["fixed"].target_jitter_xy_m == (0.0, 0.0)
    assert RUNNER.PROFILES["target_xy_2p5mm"].target_jitter_xy_m == pytest.approx(
        (0.0025, 0.0025)
    )
    assert RUNNER.PROFILES["target_xy_10mm"].target_jitter_xy_m == pytest.approx(
        (0.01, 0.01)
    )
    assert RUNNER.PROFILES[
        "target_base_xy_2p5mm"
    ].robot_base_jitter_xy_m == pytest.approx((0.0025, 0.0025))


def test_environment_uses_bend_reference_and_explicit_pose_dr() -> None:
    arguments = RUNNER.environment_args(
        RUNNER.OBJECTS["Apple_1"], "target_base_xy_2p5mm", 123
    )

    assert arguments[arguments.index("--strict-reference-episode") + 1] == "11"
    assert arguments[arguments.index("--target-position-jitter-xy") + 1 :][
        :2
    ] == ["0.0025", "0.0025"]
    assert arguments[arguments.index("--robot-base-position-jitter-xy") + 1 :][
        :2
    ] == ["0.0025", "0.0025"]
    assert arguments[arguments.index("--target-position-offset-center-xy") + 1 :][
        :2
    ] == ["0.0", "-0.09"]
    assert arguments[arguments.index("--target-mass-scale") + 1 :][
        :2
    ] == ["1", "1"]


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


def test_train_cli_has_no_legacy_checkpoint_warm_start() -> None:
    args = RUNNER._parser().parse_args(
        ["train", "Apple_1", "--run-name", "isolated_test"]
    )

    assert args.profile == "target_xy_2p5mm"
    assert args.num_envs == 8192
    assert args.iterations == 40
    assert not hasattr(args, "checkpoint")
    assert args.reference_processed is None
