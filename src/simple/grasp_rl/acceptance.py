"""Statistical release gates for the 200-target pick PPO protocol."""

from __future__ import annotations

import math


def wilson_lower_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes/trials must satisfy 0 <= successes <= trials")
    rate = successes / trials
    z2 = z * z
    center = rate + z2 / (2.0 * trials)
    radius = z * math.sqrt(
        rate * (1.0 - rate) / trials + z2 / (4.0 * trials * trials)
    )
    return (center - radius) / (1.0 + z2 / trials)


def paired_release_acceptance(
    *,
    episodes: int,
    policy_successes: int,
    reference_successes: int,
    exact_mcnemar_p_value: float,
    locked_final_protocol: bool = True,
    required_episodes: int = 200,
    required_successes: int = 153,
    required_wilson_lower: float = 0.70,
    required_improvement: float = 0.05,
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0 <= policy_successes <= episodes:
        raise ValueError("invalid policy_successes")
    if not 0 <= reference_successes <= episodes:
        raise ValueError("invalid reference_successes")
    lower = wilson_lower_bound(policy_successes, episodes)
    improvement = (policy_successes - reference_successes) / episodes
    checks = {
        "locked_final_protocol": bool(locked_final_protocol),
        "exactly_required_episodes": episodes == required_episodes,
        "minimum_successes": policy_successes >= required_successes,
        "wilson_lower_strictly_above_threshold": lower > required_wilson_lower,
        "minimum_reference_improvement": improvement >= required_improvement,
        "mcnemar_significant": exact_mcnemar_p_value < 0.05,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "wilson_95_lower": lower,
        "policy_success_rate": policy_successes / episodes,
        "reference_success_rate": reference_successes / episodes,
        "improvement": improvement,
        "requirements": {
            "episodes": required_episodes,
            "successes": required_successes,
            "wilson_95_lower_strictly_above": required_wilson_lower,
            "reference_improvement_at_least": required_improvement,
            "exact_mcnemar_p_below": 0.05,
        },
    }
