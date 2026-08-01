"""Counterfactual reward audit for ordered v2 task graphs."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.task_spec import TaskSpecV2, get_task_spec
from simple.grasp_rl.tracker import ActionTransform


V2_AUDIT_SCENARIOS = (
    "expert_hold", "no_motion", "halfway_hold", "open_hand",
    "contact_hold", "release_after_lift", "time_shuffle", "throw",
    "expert_repeat",
)


def v2_audit_acceptance(
    summary: dict[str, dict[str, float]],
    reports: list[dict[str, Any]],
    scenarios: tuple[str, ...],
) -> dict[str, bool]:
    expert_returns = [
        row["return"]
        for row in reports
        if row["scenario"] == "expert_hold" and row["success"]
    ]
    required_failures = tuple(
        name
        for name in ("no_motion", "open_hand", "contact_hold", "throw")
        if name in scenarios
    )
    required_failure_returns = [
        row["return"] for row in reports if row["scenario"] in required_failures
    ]
    return {
        "expert_success": summary.get("expert_hold", {}).get("success_rate", 0.0) >= .9,
        "expert_repeat_success": summary.get("expert_repeat", {}).get("success_rate", 0.0) >= .9,
        "core_counterfactuals_fail": bool(required_failures) and all(
            summary[name]["success_rate"] <= .05 for name in required_failures
        ),
        "all_counterfactuals_bounded": all(
            summary[name]["success_rate"] <= .05 for name in scenarios
            if name not in ("expert_hold", "expert_repeat")
        ),
        "expert_return_separation": bool(expert_returns and required_failure_returns)
        and min(expert_returns) > max(required_failure_returns),
    }


def _worker_devices() -> tuple[int, ...]:
    value = os.environ.get("GRASP_RL_WORKER_DEVICES", "0,1,2,3,4,5,6,7")
    devices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not devices:
        raise ValueError("GRASP_RL_WORKER_DEVICES selected no CUDA devices")
    return devices


def _scenario_actions(name: str, actions: np.ndarray, seed: int) -> np.ndarray:
    result = actions.copy()
    if name in ("expert_hold", "expert_repeat"):
        return result
    if name == "no_motion":
        return np.repeat(result[:1], len(result), axis=0)
    if name in ("halfway_hold", "contact_hold"):
        cut = len(result) // (3 if name == "contact_hold" else 2)
        result[cut:] = result[cut]
        return result
    if name == "open_hand":
        result[:, 7:14] = result[0, 7:14]
        result[:, 0:7] = result[0, 0:7]
        return result
    if name == "release_after_lift":
        cut = int(.55 * len(result))
        result[cut:, 7:14] = result[0, 7:14]
        return result
    if name == "time_shuffle":
        blocks = [result[i:i + 25] for i in range(0, len(result), 25)]
        np.random.default_rng(seed).shuffle(blocks)
        return np.concatenate(blocks)
    if name == "throw":
        result[:] = result[0]
        result[:, 7:14] = result[0, 7:14]
        return result
    raise ValueError(f"Unknown v2 audit scenario {name}")


def _teleport_primary_to_destination(env: GraspRlEnv) -> None:
    state = env.state.actor_observation()[1]
    if not state.primary.present or not state.destination.present:
        return
    body_id = state.primary.body_id
    joint_id = int(env.sim.mjModel.body_jntadr[body_id])
    if joint_id < 0 or env.sim.mjModel.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        return
    address = int(env.sim.mjModel.jnt_qposadr[joint_id])
    env.sim.mjData.qpos[address:address + 3] = state.destination.pos_w + np.array([0., 0., .10])
    env.sim.mjData.qvel[int(env.sim.mjModel.jnt_dofadr[joint_id]):][:6] = 0
    mujoco.mj_forward(env.sim.mjModel, env.sim.mjData)


def _audit_episode_entry(processed_string: str, transform_path: str,
                         task_name: str, episode: int, state_dict: dict[str, Any],
                         scenarios: tuple[str, ...], warmup_steps: int,
                         cuda_device: int) -> list[dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    os.environ.setdefault("MUJOCO_GL", "egl")
    processed = Path(processed_string)
    spec = get_task_spec(task_name)
    assert isinstance(spec, TaskSpecV2)
    transform = ActionTransform.from_npz(transform_path)
    with np.load(processed / "bc" / f"episode_{episode:06d}.npz", allow_pickle=False) as data:
        expert = data["physical_actions"].astype(np.float32)
    env = GraspRlEnv(transform, task=spec, warmup_steps=warmup_steps,
                     max_episode_steps=max(spec.max_episode_steps, len(expert)),
                     enable_renderers=False)
    reports: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            actions = _scenario_actions(scenario, expert, 1000 + episode)
            env.reset(state_dict=state_dict)
            if scenario == "throw":
                _teleport_primary_to_destination(env)
            total_return = 0.0
            success = failure = timeout = False
            max_stage = 0
            terminal_step = None
            for index, action in enumerate(actions):
                step = env.step_physical(action)
                total_return += .02 * (step.terms.target_reward - step.terms.penalty) + step.terms.terminal_adjustment
                max_stage = max(max_stage, int(step.terms.stage_index))
                success |= step.terms.success
                failure |= step.terms.failure
                timeout |= step.terms.timeout
                if step.done:
                    terminal_step = index
                    break
            reports.append({"episode": episode, "scenario": scenario,
                            "success": bool(success), "failure": bool(failure),
                            "timeout": bool(timeout), "return": float(total_return),
                            "max_stage": max_stage, "terminal_step": terminal_step})
    finally:
        env.close()
    return reports


def audit_v2_reward(dataset_dir: str | Path, processed_dir: str | Path,
                    output_dir: str | Path, episodes: int = 10,
                    episode_offset: int = 0,
                    scenarios: tuple[str, ...] = V2_AUDIT_SCENARIOS,
                    task: str | TaskSpecV2 | None = None,
                    warmup_steps: int = 60, workers: int = 1) -> Path:
    dataset, processed, output = Path(dataset_dir), Path(processed_dir), Path(output_dir)
    spec = get_task_spec(task)
    if not isinstance(spec, TaskSpecV2):
        raise ValueError("audit_v2_reward requires a v2 task")
    transform = ActionTransform.from_npz(processed / "action_transform.npz")
    manifest = json.loads((processed / "manifest.json").read_text())
    available = manifest["unique_episodes"][episode_offset:episode_offset + episodes]
    rows = [json.loads(line) for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()]
    reports: list[dict[str, Any]] = []
    if workers > 1:
        devices = _worker_devices()
        args = [
            (str(processed), str(processed / "action_transform.npz"), spec.name,
             episode, json.loads(rows[episode]["environment_config"]), scenarios,
             warmup_steps, devices[worker % len(devices)])
            for worker, episode in enumerate(available)
        ]
        with mp.get_context("spawn").Pool(min(workers, len(args))) as pool:
            reports = [item for group in pool.starmap(_audit_episode_entry, args) for item in group]
        env = None
    else:
        env = GraspRlEnv(transform, task=spec, warmup_steps=warmup_steps,
                         max_episode_steps=max(spec.max_episode_steps, 1200),
                         enable_renderers=False)
    try:
        for episode in available if env is not None else ():
            with np.load(processed / "bc" / f"episode_{episode:06d}.npz", allow_pickle=False) as data:
                expert = data["physical_actions"].astype(np.float32)
            for scenario in scenarios:
                actions = _scenario_actions(scenario, expert, 1000 + episode)
                env.reset(state_dict=json.loads(rows[episode]["environment_config"]))
                if scenario == "throw":
                    _teleport_primary_to_destination(env)
                total_return = 0.0
                success = failure = timeout = False
                max_stage = 0
                terminal_step = None
                for index, action in enumerate(actions):
                    step = env.step_physical(action)
                    total_return += .02 * (step.terms.target_reward - step.terms.penalty) + step.terms.terminal_adjustment
                    max_stage = max(max_stage, int(step.terms.stage_index))
                    success |= step.terms.success
                    failure |= step.terms.failure
                    timeout |= step.terms.timeout
                    if step.done and terminal_step is None:
                        terminal_step = index
                    if step.done:
                        break
                reports.append({"episode": episode, "scenario": scenario,
                                "success": bool(success), "failure": bool(failure),
                                "timeout": bool(timeout), "return": float(total_return),
                                "max_stage": max_stage, "terminal_step": terminal_step})
    finally:
        if env is not None:
            env.close()
    summary = {}
    for scenario in scenarios:
        rows_for_scenario = [row for row in reports if row["scenario"] == scenario]
        summary[scenario] = {
            "episodes": len(rows_for_scenario),
            "success_rate": float(np.mean([row["success"] for row in rows_for_scenario])),
            "mean_return": float(np.mean([row["return"] for row in rows_for_scenario])),
            "max_return": float(np.max([row["return"] for row in rows_for_scenario])),
        }
    checks = v2_audit_acceptance(summary, reports, scenarios)
    passed = all(checks.values())
    output.mkdir(parents=True, exist_ok=True)
    path = output / "reward_audit_v2.json"
    path.write_text(json.dumps({"task": spec.name, "processed": str(processed),
                                "passed": passed, "acceptance": checks,
                                "summary": summary,
                                "reports": reports}, indent=2))
    return path
