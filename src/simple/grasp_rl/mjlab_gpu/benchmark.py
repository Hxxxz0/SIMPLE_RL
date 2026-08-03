"""Strict paired comparison helpers for mjlab GPU policy evaluation."""

from __future__ import annotations

import math
from typing import Any


def _success_ids(result: dict[str, Any]) -> set[int]:
    return {int(value) for value in result["success_world_ids"]}


def _exact_mcnemar_pvalue(left_only: int, right_only: int) -> float:
    """Two-sided exact binomial test over discordant paired outcomes."""

    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def compare_paired_results(
    reference: dict[str, Any],
    proposal: dict[str, Any],
    ppo: dict[str, Any],
) -> dict[str, Any]:
    """Validate identical worlds and summarize noise-matched PPO gain."""

    results = (reference, proposal, ppo)
    episodes = {int(result["episodes"]) for result in results}
    if len(episodes) != 1:
        raise ValueError("paired modes evaluated different episode counts")
    world_hashes = {str(result["initial_world_sha256"]) for result in results}
    if len(world_hashes) != 1:
        raise ValueError("paired modes did not use identical physical worlds")
    policy_states_match = (
        proposal["initial_policy_state_sha256"]
        == ppo["initial_policy_state_sha256"]
    )
    for key in ("policy_seed", "evaluation_dr_strength", "dr_profile"):
        if len({str(result[key]) for result in results}) != 1:
            raise ValueError(f"paired modes disagree on {key}")
    if (
        proposal["initial_proposal_context_sha256"]
        != ppo["initial_proposal_context_sha256"]
    ):
        raise ValueError("proposal-only and PPO did not receive identical proposals")

    proposal_success = _success_ids(proposal)
    ppo_success = _success_ids(ppo)
    ppo_only = sorted(ppo_success - proposal_success)
    proposal_only = sorted(proposal_success - ppo_success)
    both = sorted(ppo_success & proposal_success)
    total = episodes.pop()
    neither = sorted(set(range(total)) - ppo_success - proposal_success)
    return {
        "schema_version": 1,
        "episodes": total,
        "initial_world_sha256": reference["initial_world_sha256"],
        "initial_policy_state_sha256": proposal["initial_policy_state_sha256"],
        "proposal_context_sha256": proposal["initial_proposal_context_sha256"],
        "physical_worlds_match": True,
        # Contact/sensor reductions in MuJoCo-Warp are not guaranteed to be
        # bitwise stable across independent simulator construction.  The exact
        # physical world and reference proposal hashes are the pairing gates;
        # retain policy-state equality as an explicit diagnostic.
        "noise_matched_policy_states_bitwise_match": policy_states_match,
        "reference_policy_state_matches": (
            reference["initial_policy_state_sha256"]
            == proposal["initial_policy_state_sha256"]
        ),
        "noise_matched_proposals_match": True,
        "reference_successes": int(reference["successes"]),
        "proposal_successes": len(proposal_success),
        "ppo_successes": len(ppo_success),
        "ppo_success_rate": len(ppo_success) / total,
        "proposal_success_rate": len(proposal_success) / total,
        "ppo_minus_proposal": (len(ppo_success) - len(proposal_success)) / total,
        "ppo_only_success_world_ids": ppo_only,
        "proposal_only_success_world_ids": proposal_only,
        "both_success_world_ids": both,
        "both_failed_world_ids": neither,
        "exact_mcnemar_pvalue": _exact_mcnemar_pvalue(
            len(ppo_only), len(proposal_only)
        ),
        "modes": {
            "reference_only": reference,
            "proposal_only": proposal,
            "ppo": ppo,
        },
    }
