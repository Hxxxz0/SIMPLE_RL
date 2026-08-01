from simple.grasp_rl.acceptance import (
    paired_release_acceptance,
    wilson_lower_bound,
)
from simple.grasp_rl.audit_v2 import v2_audit_acceptance
from simple.grasp_rl.evaluate import validate_final_protocol
import pytest


def test_153_of_200_has_wilson_lower_above_70_percent() -> None:
    assert wilson_lower_bound(152, 200) < 0.70
    assert wilson_lower_bound(153, 200) > 0.70


def test_release_gate_requires_all_paired_conditions() -> None:
    accepted = paired_release_acceptance(
        episodes=200,
        policy_successes=153,
        reference_successes=143,
        exact_mcnemar_p_value=0.01,
    )
    assert accepted["passed"] is True

    not_significant = paired_release_acceptance(
        episodes=200,
        policy_successes=153,
        reference_successes=143,
        exact_mcnemar_p_value=0.05,
    )
    assert not_significant["passed"] is False

    nonfinal = paired_release_acceptance(
        episodes=200,
        policy_successes=160,
        reference_successes=140,
        exact_mcnemar_p_value=0.01,
        locked_final_protocol=False,
    )
    assert nonfinal["passed"] is False
    assert nonfinal["checks"]["locked_final_protocol"] is False


def test_v2_reward_audit_allows_one_bounded_shuffle_success() -> None:
    scenarios = (
        "expert_hold",
        "expert_repeat",
        "no_motion",
        "open_hand",
        "contact_hold",
        "throw",
        "time_shuffle",
    )
    summary = {
        name: {"success_rate": 1.0 if name.startswith("expert") else 0.0}
        for name in scenarios
    }
    summary["time_shuffle"]["success_rate"] = 0.05
    reports = [
        {"scenario": "expert_hold", "success": True, "return": 12.0},
        {"scenario": "no_motion", "success": False, "return": 0.0},
        {"scenario": "open_hand", "success": False, "return": -1.0},
        {"scenario": "contact_hold", "success": False, "return": -0.5},
        {"scenario": "throw", "success": False, "return": 0.0},
        {"scenario": "time_shuffle", "success": True, "return": 13.0},
    ]
    assert all(v2_audit_acceptance(summary, reports, scenarios).values())


def test_final_protocol_is_locked_to_200_unseen_standard_targets() -> None:
    arguments = dict(
        num_episodes=200,
        initialization_prefix=None,
        initialization_phase=None,
        evaluation_split="test",
        randomize_target=True,
        target_position_jitter_xy=(0.025, 0.03),
        target_position_offset_center_xy=(0.0, 0.0),
        target_yaw_jitter=0.15,
        reference_rank=0,
        reference_splits=("train", "val"),
        fixed_base_episode=7,
    )
    validate_final_protocol(**arguments)
    arguments["num_episodes"] = 199
    with pytest.raises(ValueError, match="num_episodes"):
        validate_final_protocol(**arguments)
    arguments["num_episodes"] = 200
    arguments["reference_splits"] = ("train", "val", "test")
    with pytest.raises(ValueError, match="training_reference_library_only"):
        validate_final_protocol(**arguments)
