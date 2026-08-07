"""Successful trajectory export from the real mjlab GPU PPO loop."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.recording import (
    _checkpoint_provenance,
    _episode_randomization,
)
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv

_STEP_KEYS = (
    "observation",
    "policy_input",
    "raw_action",
    "reference_action",
    "effective_action",
    "physical_action",
    "joint_target",
    "reward",
    "task_reward",
    "reference_reward",
    "stage_index",
)


def _new_trace() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in (*_STEP_KEYS, "qpos", "qvel")}


def _episode_arrays(
    trace: dict[str, list[np.ndarray]],
    terminal_qpos: np.ndarray,
    terminal_qvel: np.ndarray,
) -> dict[str, np.ndarray]:
    steps = len(trace["raw_action"])
    if steps < 1:
        raise ValueError("cannot export an empty episode")
    arrays = {key: np.stack(trace[key]) for key in _STEP_KEYS}
    arrays["qpos"] = np.stack((*trace["qpos"], terminal_qpos))
    arrays["qvel"] = np.stack((*trace["qvel"], terminal_qvel))
    arrays["done"] = np.zeros(steps, dtype=np.bool_)
    arrays["done"][-1] = True
    arrays["success"] = np.asarray(True, dtype=np.bool_)
    return arrays


def validate_episode_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Validate the portable successful-trajectory NPZ schema."""

    missing = {*_STEP_KEYS, "qpos", "qvel", "done", "success"} - arrays.keys()
    if missing:
        raise ValueError(f"trajectory is missing fields: {sorted(missing)}")
    steps = len(arrays["raw_action"])
    if steps < 1 or any(len(arrays[key]) != steps for key in _STEP_KEYS):
        raise ValueError("trajectory step fields have inconsistent lengths")
    if len(arrays["qpos"]) != steps + 1 or len(arrays["qvel"]) != steps + 1:
        raise ValueError("qpos/qvel must include the terminal state")
    if arrays["raw_action"].shape[-1] != 36:
        raise ValueError("trajectory actions must use the 36-D public schema")
    if arrays["joint_target"].shape[-1] != 43:
        raise ValueError("trajectory joint targets must use the 43-D Sonic schema")
    if arrays["done"].shape != (steps,) or not bool(arrays["done"][-1]):
        raise ValueError("trajectory must terminate on its final step")
    if not bool(arrays["success"]):
        raise ValueError("only successful trajectories may be exported")
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"trajectory field {name} contains non-finite values")


@torch.inference_mode()
def collect_successful_trajectories(
    env: GpuGraspVecEnv,
    actor: torch.nn.Module,
    checkpoint: Path,
    output_dir: Path,
    *,
    successes: int,
    max_attempts: int,
    domain_randomization: bool,
    stochastic_policy: bool = False,
) -> dict[str, Any]:
    """Write successful NPZ episodes and audit every completed attempt."""

    if successes < 1:
        raise ValueError("successes must be positive")
    if max_attempts < successes:
        raise ValueError("max-attempts must be at least successes")
    if not env.capture_step_data or not env.capture_terminal_qpos:
        raise ValueError("collection requires step and terminal state capture")

    provenance = _checkpoint_provenance(checkpoint)
    dr_strength = env._domain_randomization_strength()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    observations = env.get_observations()
    traces = [_new_trace() for _ in range(env.num_envs)]
    episode_dr = _episode_randomization(env)
    manifest: list[dict[str, Any]] = []
    attempts = 0
    saved = 0

    while saved < successes and attempts < max_attempts:
        policy_input = observations["policy"].detach().cpu().numpy()
        observation_dim = env.reference.observation_dim
        qpos = env.gpu.sim.data.qpos.detach().cpu().numpy()
        qvel = env.gpu.sim.data.qvel.detach().cpu().numpy()
        reference_action = env.reference.current_action().clone()
        raw_action = actor(observations, stochastic_output=stochastic_policy)
        effective_action = env._bounded_reference_action(raw_action)
        observations, rewards, dones, extras = env.step(raw_action)
        step_data = extras.get("step_data")
        if not isinstance(step_data, dict):
            raise TypeError("environment did not return captured step data")
        physical_action = step_data["physical_action"].detach().cpu().numpy()
        joint_target = step_data["joint_target"].detach().cpu().numpy()
        task_reward = step_data["task_reward"].detach().cpu().numpy()
        reference_reward = step_data["reference_reward"].detach().cpu().numpy()
        stage_index = step_data["stage_index"].detach().cpu().numpy()
        raw = raw_action.detach().cpu().numpy()
        reference = reference_action.detach().cpu().numpy()
        effective = effective_action.detach().cpu().numpy()
        reward = rewards.detach().cpu().numpy()

        for env_id in range(env.num_envs):
            trace = traces[env_id]
            values = {
                "observation": policy_input[env_id, :observation_dim],
                "policy_input": policy_input[env_id],
                "raw_action": raw[env_id],
                "reference_action": reference[env_id],
                "effective_action": effective[env_id],
                "physical_action": physical_action[env_id],
                "joint_target": joint_target[env_id],
                "reward": reward[env_id],
                "task_reward": task_reward[env_id],
                "reference_reward": reference_reward[env_id],
                "stage_index": stage_index[env_id],
                "qpos": qpos[env_id],
                "qvel": qvel[env_id],
            }
            for key, value in values.items():
                trace[key].append(np.asarray(value).copy())

        finished = dones.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        if not finished:
            continue
        terminal_ids = extras["terminal_env_ids"].detach().cpu().tolist()
        terminal_qpos = extras["terminal_qpos"].detach().cpu().numpy()
        terminal_qvel = extras["terminal_qvel"].detach().cpu().numpy()
        terminal = {
            env_id: (terminal_qpos[index], terminal_qvel[index])
            for index, env_id in enumerate(terminal_ids)
        }
        assert env.last_terms is not None
        for env_id in finished:
            if attempts >= max_attempts:
                break
            attempts += 1
            success = bool(env.last_terms.success[env_id])
            record: dict[str, Any] = {
                "attempt": attempts,
                "world_id": env_id,
                "success": success,
                "steps": len(traces[env_id]["raw_action"]),
                "randomization": episode_dr[env_id],
                "file": None,
            }
            if success and saved < successes:
                arrays = _episode_arrays(traces[env_id], *terminal[env_id])
                validate_episode_arrays(arrays)
                path = episodes_dir / f"episode_{saved:06d}.npz"
                np.savez_compressed(path, **arrays)
                record["file"] = str(path.relative_to(output_dir))
                saved += 1
            manifest.append(record)
            traces[env_id] = _new_trace()
            if saved >= successes:
                break
        reset_dr = _episode_randomization(env)
        for env_id in finished:
            episode_dr[env_id] = reset_dr[env_id]

    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    )
    summary = {
        "schema_version": 1,
        "task": env.config.task,
        "checkpoint": provenance,
        "requested_successes": successes,
        "successes": saved,
        "attempts": attempts,
        "attempt_success_rate": saved / max(attempts, 1),
        "complete": saved == successes,
        "deterministic_actor": not stochastic_policy,
        "domain_randomization": domain_randomization,
        "domain_randomization_strength": dr_strength,
        "resolved_domain_randomization": asdict(env.config.domain_randomization),
        "seed": env.config.seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if saved != successes:
        raise RuntimeError(
            f"Collected only {saved}/{successes} successes in {attempts} attempts"
        )
    return summary
