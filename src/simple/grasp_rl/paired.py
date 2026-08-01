"""Strict paired comparison of PPO and reference-only evaluations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from simple.grasp_rl.acceptance import paired_release_acceptance


PAIRED_STATE_FIELDS = (
    "initial_qpos",
    "initial_qvel",
    "target_position",
    "target_quaternion",
    "reference_target_position",
    "base_episode",
    "reference_episode",
)


def exact_mcnemar_p_value(policy_only: int, reference_only: int) -> float:
    """Two-sided exact binomial test over discordant paired outcomes."""

    if policy_only < 0 or reference_only < 0:
        raise ValueError("Discordant counts must be non-negative")
    discordant = policy_only + reference_only
    if discordant == 0:
        return 1.0
    lower = min(policy_only, reference_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare_paired_evaluations(
    policy_evaluation: str | Path,
    reference_evaluation: str | Path,
    output_path: str | Path,
) -> dict:
    """Validate identical starts and compare closed-loop success pair by pair."""

    policy_dir = Path(policy_evaluation)
    reference_dir = Path(reference_evaluation)
    policy = json.loads((policy_dir / "summary.json").read_text())
    reference = json.loads((reference_dir / "summary.json").read_text())
    if policy["task"] != reference["task"]:
        raise ValueError("Paired evaluations belong to different tasks")
    if reference.get("reference_action_override") != "all":
        raise ValueError("Reference evaluation must use reference_action_override=all")
    if policy.get("reference_action_override") != "none":
        raise ValueError("Policy evaluation must not override its actions")
    policy_rows = {int(row["rollout_index"]): row for row in policy["results"]}
    reference_rows = {
        int(row["rollout_index"]): row for row in reference["results"]
    }
    if policy_rows.keys() != reference_rows.keys():
        raise ValueError("Paired evaluations have different rollout indices")

    policy_only: list[int] = []
    reference_only: list[int] = []
    both_success: list[int] = []
    both_failure: list[int] = []
    action_absolute_sum = 0.0
    action_element_count = 0
    action_maximum = 0.0
    unequal_length_pairs = 0
    for index in sorted(policy_rows):
        policy_success = bool(policy_rows[index]["success"])
        reference_success = bool(reference_rows[index]["success"])
        if policy_success and not reference_success:
            policy_only.append(index)
        elif reference_success and not policy_success:
            reference_only.append(index)
        elif policy_success:
            both_success.append(index)
        else:
            both_failure.append(index)

        name = f"episode_{int(policy_rows[index]['episode']):06d}_repeat_{index:06d}.npz"
        policy_trajectory = policy_dir / "trajectories" / name
        reference_trajectory = reference_dir / "trajectories" / name
        with (
            np.load(policy_trajectory, allow_pickle=False) as policy_saved,
            np.load(reference_trajectory, allow_pickle=False) as reference_saved,
        ):
            for field in PAIRED_STATE_FIELDS:
                if not np.array_equal(policy_saved[field], reference_saved[field]):
                    raise ValueError(
                        f"Pair {index} differs in required field {field!r}"
                    )
            policy_actions = policy_saved["actions"]
            reference_actions = reference_saved["actions"]
            common_length = min(len(policy_actions), len(reference_actions))
            if len(policy_actions) != len(reference_actions):
                unequal_length_pairs += 1
            difference = np.abs(
                policy_actions[:common_length] - reference_actions[:common_length]
            )
            action_absolute_sum += float(difference.sum())
            action_element_count += int(difference.size)
            action_maximum = max(action_maximum, float(difference.max(initial=0.0)))

    result = {
        "task": policy["task"],
        "policy_checkpoint": policy["checkpoint"],
        "policy_evaluation": str(policy_dir.resolve()),
        "reference_evaluation": str(reference_dir.resolve()),
        "episodes": len(policy_rows),
        "paired_state_fields_exact": list(PAIRED_STATE_FIELDS),
        "policy_successes": len(policy_only) + len(both_success),
        "reference_successes": len(reference_only) + len(both_success),
        "policy_only_successes": policy_only,
        "reference_only_successes": reference_only,
        "both_successes": both_success,
        "both_failures": both_failure,
        "discordant_pairs": len(policy_only) + len(reference_only),
        "exact_mcnemar_p_value": exact_mcnemar_p_value(
            len(policy_only), len(reference_only)
        ),
        "mean_absolute_action_delta": (
            action_absolute_sum / action_element_count
            if action_element_count
            else 0.0
        ),
        "max_absolute_action_delta": action_maximum,
        "unequal_length_pairs": unequal_length_pairs,
    }
    result["release_acceptance"] = paired_release_acceptance(
        episodes=result["episodes"],
        policy_successes=result["policy_successes"],
        reference_successes=result["reference_successes"],
        exact_mcnemar_p_value=result["exact_mcnemar_p_value"],
        locked_final_protocol=(
            bool(policy.get("final_test", False))
            and bool(reference.get("final_test", False))
        ),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result
