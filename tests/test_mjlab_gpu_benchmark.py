import pytest

from simple.grasp_rl.mjlab_gpu.benchmark import compare_paired_results


def _result(successes, *, world="world", context="context"):
    return {
        "episodes": 4,
        "successes": len(successes),
        "success_world_ids": successes,
        "initial_world_sha256": world,
        "initial_policy_state_sha256": "state",
        "initial_proposal_context_sha256": context,
        "policy_seed": 42,
        "evaluation_dr_strength": 1.0,
        "dr_profile": "full",
    }


def test_paired_benchmark_reports_rescue_and_regression() -> None:
    result = compare_paired_results(
        _result([0, 1], context="clean"),
        _result([0, 1]),
        _result([1, 2]),
    )

    assert result["physical_worlds_match"]
    assert result["noise_matched_policy_states_bitwise_match"]
    assert result["ppo_only_success_world_ids"] == [2]
    assert result["proposal_only_success_world_ids"] == [0]
    assert result["both_success_world_ids"] == [1]
    assert result["both_failed_world_ids"] == [3]
    assert result["exact_mcnemar_pvalue"] == 1.0


def test_paired_benchmark_rejects_mismatched_worlds() -> None:
    with pytest.raises(ValueError, match="physical worlds"):
        compare_paired_results(
            _result([0], world="reference"),
            _result([0], world="proposal"),
            _result([0], world="ppo"),
        )


def test_paired_benchmark_rejects_mismatched_policy_proposals() -> None:
    with pytest.raises(ValueError, match="identical proposals"):
        compare_paired_results(
            _result([0], context="clean"),
            _result([0], context="proposal"),
            _result([0], context="ppo"),
        )
