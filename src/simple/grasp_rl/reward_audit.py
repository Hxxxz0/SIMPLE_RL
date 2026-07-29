"""Counterfactual demonstration replay for grasp-reward validation."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.rewards import GraspReward, TASK_REWARD_PROFILES
from simple.grasp_rl.schema import ACTION_SLICES, MAX_EPISODE_STEPS
from simple.grasp_rl.tracker import ActionTransform


AUDIT_SCENARIOS = (
    "expert_hold",
    "no_motion",
    "halfway_hold",
    "open_hand",
    "contact_hold",
    "release_after_lift",
    "expert_repeat",
)
AUDITED_TERMS = (
    "reach",
    "pregrasp",
    "contact",
    "grasp_quality",
    "finger",
    "lift",
    "stable",
    "hold",
    "progress",
    "progress_bonus",
    "grail_grasp",
    "grail_finger_direction",
    "approach_penalty",
    "table_penalty",
    "action_rate_penalty",
    "joint_limit_penalty",
    "target_reward",
    "penalty",
    "terminal_adjustment",
    "hand_table_force",
)


def _episode_file(dataset: Path, episode: int) -> Path:
    return dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"


def _scenario_action(
    scenario: str,
    actions: np.ndarray,
    step: int,
    frozen_action: np.ndarray | None,
) -> np.ndarray:
    if frozen_action is not None:
        action = frozen_action.copy()
    elif scenario == "no_motion":
        action = actions[0].copy()
    elif scenario == "halfway_hold":
        action = actions[min(step, len(actions) // 2)].copy()
    else:
        action = actions[min(step, len(actions) - 1)].copy()
    if scenario == "open_hand":
        hand = ACTION_SLICES["right_hand"]
        action[hand.start : hand.stop] = actions[0, hand.start : hand.stop]
    return action


def _rollout_scenario(
    env: GraspRlEnv,
    state_dict: dict[str, Any],
    actions: np.ndarray,
    scenario: str,
    profiles: tuple[str, ...],
    task_weight: float,
) -> dict[str, Any]:
    env.reset(state_dict=state_dict)
    assert env.state is not None
    evaluators = {
        profile: GraspReward(
            env.state,
            max_episode_steps=MAX_EPISODE_STEPS,
            profile=profile,
        )
        for profile in profiles
    }
    action_span = env.action_transform.high - env.action_transform.low
    totals = {
        profile: {term: 0.0 for term in AUDITED_TERMS} for profile in profiles
    }
    returns = {profile: 0.0 for profile in profiles}
    frozen_action: np.ndarray | None = None
    max_lift = -float("inf")
    max_grasp_quality = 0.0
    first_grasp_step: int | None = None
    first_lift_step: int | None = None
    final_terms = None

    for step_index in range(MAX_EPISODE_STEPS):
        action = _scenario_action(
            scenario, actions, step_index, frozen_action
        )
        previous_action = env.previous_physical_action.copy()
        env.step_physical(action)
        state = env.state.target_state()
        profile_terms = {
            profile: evaluator.compute(
                state,
                action,
                previous_action,
                action_span,
            )
            for profile, evaluator in evaluators.items()
        }
        terms = profile_terms["grail_release_v1"]
        final_terms = terms
        max_lift = max(max_lift, terms.lift_height)
        max_grasp_quality = max(max_grasp_quality, terms.grasp_quality)
        if terms.is_grasp and first_grasp_step is None:
            first_grasp_step = step_index
        if terms.lift_height >= 0.02 and first_lift_step is None:
            first_lift_step = step_index

        for profile, values in profile_terms.items():
            returns[profile] += (
                task_weight * (values.target_reward - values.penalty)
                + values.terminal_adjustment
            )
            for name in AUDITED_TERMS:
                totals[profile][name] += float(getattr(values, name))

        if scenario == "contact_hold" and frozen_action is None and terms.is_grasp:
            frozen_action = action.copy()
        if (
            scenario == "release_after_lift"
            and frozen_action is None
            and terms.lift_height >= 0.015
        ):
            frozen_action = action.copy()
            hand = ACTION_SLICES["right_hand"]
            frozen_action[hand.start : hand.stop] = actions[
                0, hand.start : hand.stop
            ]
        if terms.success or terms.failure or terms.timeout:
            break

    assert final_terms is not None
    return {
        "scenario": scenario,
        "steps": step_index + 1,
        "success": bool(final_terms.success),
        "failure": bool(final_terms.failure),
        "timeout": bool(final_terms.timeout),
        "max_lift": float(max_lift),
        "max_grasp_quality": float(max_grasp_quality),
        "first_grasp_step": first_grasp_step,
        "first_lift_step": first_lift_step,
        "returns": returns,
        "term_sums": totals,
    }


def _audit_chunk(
    dataset_string: str,
    transform_string: str,
    episode_rows: list[tuple[int, dict[str, Any]]],
    profiles: tuple[str, ...],
    scenarios: tuple[str, ...],
    task_weight: float,
    cuda_device: int,
) -> list[dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    os.environ.setdefault("MUJOCO_GL", "egl")
    dataset = Path(dataset_string)
    transform = ActionTransform.from_npz(transform_string)
    env = GraspRlEnv(
        transform,
        seed=10_000 + episode_rows[0][0],
        task_reward_profile="grail_v3",
        fast_reset=False,
    )
    reports: list[dict[str, Any]] = []
    try:
        for episode, row in episode_rows:
            actions = np.asarray(
                pq.read_table(_episode_file(dataset, episode), columns=["action"])[
                    "action"
                ].to_pylist(),
                dtype=np.float32,
            )
            state_dict = json.loads(row["environment_config"])
            rollouts = [
                _rollout_scenario(
                    env,
                    state_dict,
                    actions,
                    scenario,
                    profiles,
                    task_weight,
                )
                for scenario in scenarios
            ]
            reports.append({"episode": episode, "rollouts": rollouts})
    finally:
        env.close()
    return reports


def _summarize(
    reports: list[dict[str, Any]],
    profiles: tuple[str, ...],
    scenarios: tuple[str, ...],
) -> dict[str, Any]:
    by_scenario = {
        scenario: [
            next(
                rollout
                for rollout in report["rollouts"]
                if rollout["scenario"] == scenario
            )
            for report in reports
        ]
        for scenario in scenarios
    }
    scenario_summary: dict[str, Any] = {}
    for scenario, rows in by_scenario.items():
        scenario_summary[scenario] = {
            "episodes": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "failure_rate": float(np.mean([row["failure"] for row in rows])),
            "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_max_lift": float(np.mean([row["max_lift"] for row in rows])),
            "mean_max_grasp_quality": float(
                np.mean([row["max_grasp_quality"] for row in rows])
            ),
            "return": {
                profile: {
                    "mean": float(
                        np.mean([row["returns"][profile] for row in rows])
                    ),
                    "median": float(
                        np.median([row["returns"][profile] for row in rows])
                    ),
                    "min": float(
                        np.min([row["returns"][profile] for row in rows])
                    ),
                    "max": float(
                        np.max([row["returns"][profile] for row in rows])
                    ),
                }
                for profile in profiles
            },
        }

    expert = by_scenario["expert_hold"]
    paired_ranking: dict[str, Any] = {}
    for profile in profiles:
        expert_returns = np.asarray(
            [row["returns"][profile] for row in expert], dtype=np.float64
        )
        paired_ranking[profile] = {}
        for scenario in scenarios:
            if scenario in {"expert_hold", "expert_repeat"}:
                continue
            comparison = np.asarray(
                [
                    row["returns"][profile]
                    for row in by_scenario[scenario]
                ],
                dtype=np.float64,
            )
            difference = expert_returns - comparison
            paired_ranking[profile][scenario] = {
                "expert_higher_fraction": float(np.mean(difference > 0.0)),
                "mean_margin": float(np.mean(difference)),
                "min_margin": float(np.min(difference)),
            }

    reset_check = None
    if "expert_repeat" in by_scenario:
        original = by_scenario["expert_hold"]
        repeated = by_scenario["expert_repeat"]
        reset_check = {
            profile: {
                "max_abs_return_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(
                                [row["returns"][profile] for row in original]
                            )
                            - np.asarray(
                                [row["returns"][profile] for row in repeated]
                            )
                        )
                    )
                ),
                "outcome_match_fraction": float(
                    np.mean(
                        [
                            (a["success"], a["failure"], a["timeout"])
                            == (b["success"], b["failure"], b["timeout"])
                            for a, b in zip(original, repeated, strict=True)
                        ]
                    )
                ),
            }
            for profile in profiles
        }

    acceptance: dict[str, Any] = {}
    for profile in profiles:
        counterfactuals = [
            scenario
            for scenario in scenarios
            if scenario not in {"expert_hold", "expert_repeat"}
        ]
        successful_expert_returns = [
            row["returns"][profile]
            for row in by_scenario["expert_hold"]
            if row["success"]
        ]
        successful_expert_min = (
            float(np.min(successful_expert_returns))
            if successful_expert_returns
            else -float("inf")
        )
        counterfactual_max = max(
            scenario_summary[scenario]["return"][profile]["max"]
            for scenario in counterfactuals
        )
        paired_pass = all(
            paired_ranking[profile][scenario]["expert_higher_fraction"] >= 0.95
            for scenario in counterfactuals
        )
        outcome_pass = all(
            scenario_summary[scenario]["success_rate"] <= 0.05
            for scenario in counterfactuals
        )
        reset_pass = (
            reset_check is None
            or reset_check[profile]["outcome_match_fraction"] >= 0.99
        )
        checks = {
            "expert_success_rate_at_least_95pct": (
                scenario_summary["expert_hold"]["success_rate"] >= 0.95
            ),
            "counterfactual_success_rate_at_most_5pct": outcome_pass,
            "paired_expert_higher_at_least_95pct": paired_pass,
            "successful_expert_global_return_separation": (
                successful_expert_min > counterfactual_max
            ),
            "repeat_outcome_match_at_least_99pct": reset_pass,
        }
        acceptance[profile] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "successful_expert_min_return": successful_expert_min,
            "max_counterfactual_return": counterfactual_max,
            "global_margin": successful_expert_min - counterfactual_max,
        }

    return {
        "scenario_summary": scenario_summary,
        "paired_ranking": paired_ranking,
        "reset_check": reset_check,
        "acceptance": acceptance,
    }


def audit_reward(
    dataset_dir: str | Path,
    action_transform_path: str | Path,
    output_dir: str | Path,
    episodes: int = 100,
    episode_offset: int = 0,
    workers: int = 7,
    profiles: tuple[str, ...] = TASK_REWARD_PROFILES,
    scenarios: tuple[str, ...] = AUDIT_SCENARIOS,
    task_weight: float = 0.02,
) -> Path:
    """Replay experts and targeted failures, then write a paired reward audit."""
    invalid_profiles = set(profiles) - set(TASK_REWARD_PROFILES)
    if invalid_profiles:
        raise ValueError(f"Unknown profiles: {sorted(invalid_profiles)}")
    invalid_scenarios = set(scenarios) - set(AUDIT_SCENARIOS)
    if invalid_scenarios:
        raise ValueError(f"Unknown scenarios: {sorted(invalid_scenarios)}")
    if "grail_release_v1" not in profiles:
        raise ValueError(
            "grail_release_v1 is required to provide common termination signals"
        )
    if "expert_hold" not in scenarios:
        raise ValueError("expert_hold is required for paired ranking")

    dataset = Path(dataset_dir).resolve()
    transform = Path(action_transform_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
        )
    }
    episode_ids = list(range(episode_offset, episode_offset + episodes))
    missing = [episode for episode in episode_ids if episode not in all_rows]
    if missing:
        raise ValueError(f"Dataset does not contain episodes {missing[:5]}")
    selected = [(episode, all_rows[episode]) for episode in episode_ids]
    chunks = [
        list(chunk)
        for chunk in np.array_split(selected, min(workers, len(selected)))
        if len(chunk)
    ]
    args = [
        (
            str(dataset),
            str(transform),
            chunk,
            profiles,
            scenarios,
            task_weight,
            worker,
        )
        for worker, chunk in enumerate(chunks)
    ]
    if len(args) == 1:
        reports = _audit_chunk(*args[0])
    else:
        with mp.get_context("spawn").Pool(len(args)) as pool:
            reports = [
                report
                for group in pool.starmap(_audit_chunk, args)
                for report in group
            ]
    reports.sort(key=lambda row: row["episode"])
    summary = {
        "dataset": str(dataset),
        "action_transform": str(transform),
        "task_weight": task_weight,
        "profiles": list(profiles),
        "scenarios": list(scenarios),
        "num_episodes": len(reports),
        **_summarize(reports, profiles, scenarios),
        "reports": reports,
    }
    result = output / "reward_audit.json"
    result.write_text(json.dumps(summary, indent=2))
    return result


def refresh_reward_audit(result_path: str | Path) -> Path:
    """Recompute summaries from already collected rollout reports."""
    result = Path(result_path).resolve()
    payload = json.loads(result.read_text())
    profiles = tuple(payload["profiles"])
    scenarios = tuple(payload["scenarios"])
    payload.update(_summarize(payload["reports"], profiles, scenarios))
    result.write_text(json.dumps(payload, indent=2))
    return result
