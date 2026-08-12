"""Immutable-source preparation and replay audit for role-based v2 tasks."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.schema import ACTION_DIM, ACTOR_OBS_V2_DIM, schema_dict
from simple.grasp_rl.task_spec import TaskSpecV2, get_task_spec
from simple.grasp_rl.tracker import ActionTransform, compute_action_transform


REPAIR_VERSION = "cross_table_yaw_v1"
PREPARE_VERSION = "task_v2_replay_v2"
SONIC_PREPARE_VERSION = "task_v2_replay_v12_native_lift_hold"
ACTION_TRANSFORM_VERSION = "successful_replay_cover_v1"


def prepare_version(spec: TaskSpecV2) -> str:
    return (
        SONIC_PREPARE_VERSION
        if spec.controller_backend == "sonic_wbc"
        else PREPARE_VERSION
    )


def _successful_replay_transform(actions: list[np.ndarray]) -> ActionTransform:
    """Build a transform whose encode/decode path covers every valid command."""

    arrays = [np.asarray(value, dtype=np.float32) for value in actions]
    flat = np.concatenate(arrays)
    first = np.stack([value[0] for value in arrays])
    center = np.median(first, axis=0).astype(np.float32)
    low = flat.min(axis=0)
    high = flat.max(axis=0)
    span = high - low
    low = (low - .01 * span).astype(np.float32)
    high = (high + .01 * span).astype(np.float32)
    tiny = high - low < 1e-4
    low[tiny] = center[tiny] - .01
    high[tiny] = center[tiny] + .01
    transition = [np.abs(value[0] - center) for value in arrays]
    transition.extend(
        np.abs(np.diff(value, axis=0)).max(axis=0) for value in arrays
    )
    max_delta = np.maximum(np.max(transition, axis=0), .01).astype(np.float32)
    return ActionTransform(center, low, high, max_delta)


def _rewrite_successful_raw_actions(
    output: Path, episode_ids: list[int], transform: ActionTransform
) -> None:
    for episode in episode_ids:
        path = output / "bc" / f"episode_{episode:06d}.npz"
        with np.load(path, allow_pickle=False) as saved:
            payload = {name: saved[name] for name in saved.files}
        payload["raw_actions"] = transform.encode(payload["physical_actions"])
        np.savez_compressed(path, **payload)


def _usable_replay_episode_ids(
    reports: list[dict[str, Any]], minimum: int = 3
) -> list[int]:
    """Select only controller-verified demonstrations or fail closed."""

    usable = [int(report["episode"]) for report in reports if report["success"]]
    if len(usable) < minimum:
        raise RuntimeError(
            f"Fewer than {minimum} usable demonstrations passed replay"
        )
    return usable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_devices() -> tuple[int, ...]:
    value = os.environ.get("GRASP_RL_WORKER_DEVICES", "0,1,2,3,4,5,6,7")
    devices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not devices:
        raise ValueError("GRASP_RL_WORKER_DEVICES selected no CUDA devices")
    return devices


def episode_file(dataset: Path, episode: int) -> Path:
    candidates = sorted((dataset / "data").glob(f"chunk-*/episode_{episode:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Missing episode {episode} below {dataset}")
    return candidates[0]


def read_actions(dataset: Path, episode: int) -> np.ndarray:
    return np.asarray(pq.read_table(episode_file(dataset, episode), columns=["action"])["action"].to_pylist(),
                      dtype=np.float32)


def repair_cross_table_actions(actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore turn intent discarded by the historical postprocessor.

    The affected source has zero dimensions 34/35 and a single ~200 frame
    0.1-m/s marker that concatenates two 100-frame TurnSpecs.
    """

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected [T,{ACTION_DIM}] actions, got {actions.shape}")
    repaired = actions.copy()
    missing = bool(np.max(np.abs(actions[:, 34:36])) < 1e-6)
    marker = np.isclose(actions[:, 32], .1, atol=.015)
    runs: list[tuple[int, int]] = []
    start = None
    for index, active in enumerate(np.r_[marker, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    candidates = [(a, b) for a, b in runs if 180 <= b - a <= 220]
    if not missing:
        return repaired, {"version": REPAIR_VERSION, "applied": False, "reason": "yaw_present"}
    if len(candidates) != 1:
        return repaired, {"version": REPAIR_VERSION, "applied": False,
                          "reason": "turn_marker_not_unique", "runs": runs}
    begin, end = candidates[0]
    middle = begin + (end - begin) // 2
    repaired[begin:middle, 34] = 1.0
    repaired[middle:end, 34] = 1.0
    repaired[begin:middle, 35] = np.linspace(0.0, np.pi / 2, middle - begin, endpoint=True)
    repaired[middle:end, 35] = np.linspace(np.pi / 2, np.pi, end - middle, endpoint=True)
    repaired[end:, 35] = np.pi
    changed = np.flatnonzero(np.any(repaired != actions, axis=0)).tolist()
    if changed != [34, 35]:
        raise AssertionError(f"Repair unexpectedly changed action dimensions {changed}")
    return repaired, {"version": REPAIR_VERSION, "applied": True,
                      "turn_interval": [begin, end], "split": middle,
                      "changed_dimensions": changed}


def repair_actions(spec: TaskSpecV2, actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if spec.name == "locomotion_pick_between_tables":
        return repair_cross_table_actions(actions)
    return np.asarray(actions, dtype=np.float32).copy(), {
        "version": "none", "applied": False, "reason": "no_task_repair"
    }


def _prepare_chunk(dataset_string: str, output_string: str, rows: list[dict[str, Any]],
                   episodes: list[int], transform_payload: dict[str, np.ndarray],
                   task_name: str, cuda_device: int, warmup_steps: int) -> list[dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    os.environ.setdefault("MUJOCO_GL", "egl")
    dataset, output = Path(dataset_string), Path(output_string)
    spec = get_task_spec(task_name)
    assert isinstance(spec, TaskSpecV2)
    expected_prepare_version = prepare_version(spec)
    transform = ActionTransform(**transform_payload)
    env = GraspRlEnv(transform, seed=42 + episodes[0], task=spec,
                     warmup_steps=warmup_steps, enable_renderers=False,
                     max_episode_steps=max(spec.max_episode_steps, 1200))
    reports: list[dict[str, Any]] = []
    try:
        for episode in episodes:
            original = read_actions(dataset, episode)
            actions, repair = repair_actions(spec, original)
            bc_path = output / "bc" / f"episode_{episode:06d}.npz"
            episode_path = output / "episodes" / f"episode_{episode:06d}.npz"
            if bc_path.exists() and episode_path.exists():
                try:
                    with np.load(bc_path, allow_pickle=False) as saved:
                        if str(saved["prepare_version"].item()) != expected_prepare_version:
                            raise KeyError("stale v2 replay")
                        observations = saved["observations"]
                        physical = saved["physical_actions"]
                        sources = saved["sample_sources"]
                        saved_success = bool(saved["terminal_success"].item())
                        saved_failure = bool(saved["terminal_failure"].item())
                        saved_timeout = bool(saved["terminal_timeout"].item())
                        saved_max_stage = int(saved["max_stage"].item())
                        saved_max_lift = float(saved["max_lift"].item())
                        saved_max_grasp = float(saved["max_grasp_quality"].item())
                    stage = int(np.argmax(observations[-1, 322:330]))
                    primary = observations[-1, 163:166]
                    destination = observations[-1, 182:185]
                    reports.append({
                        "episode": episode, "frames": len(observations),
                        "source_frames": int(np.sum(sources == "source")),
                        "completion_frames": int(np.sum(sources == "completion")),
                        "repair": repair, "success": saved_success,
                        "failure": saved_failure, "timeout": saved_timeout,
                        "terminal_step": len(observations),
                        "max_stage": max(stage, saved_max_stage),
                        "max_lift": saved_max_lift,
                        "max_grasp_quality": saved_max_grasp,
                        "final_primary_destination_xy": float(np.linalg.norm((destination - primary)[:2])),
                        "finite": bool(np.isfinite(observations).all() and np.isfinite(physical).all()),
                        "resumed": True,
                    })
                    continue
                except (KeyError, ValueError, EOFError):
                    pass
            observation, _ = env.reset(state_dict=json.loads(rows[episode]["environment_config"]))
            observations, raw_actions, physical_actions, frames, sample_sources = [], [], [], [], []
            right_contact_centers_primary: list[np.ndarray] = []
            right_contact_center_valid: list[bool] = []
            terminal = None
            max_stage, max_lift, max_grasp = 0, -np.inf, 0.0
            def execute(action: np.ndarray, source: str) -> bool:
                nonlocal observation, terminal, max_stage, max_lift, max_grasp
                observations.append(observation.copy())
                raw_actions.append(transform.encode(action))
                physical_actions.append(np.asarray(action, dtype=np.float32).copy())
                sample_sources.append(source)
                assert env.state is not None
                _, current_state = env.state.actor_observation()
                valid_center = bool(current_state.has_primary_contact_center[1])
                right_contact_center_valid.append(valid_center)
                if valid_center:
                    center_primary = current_state.primary.rot_w.T @ (
                        current_state.primary_contact_centers_w[1]
                        - current_state.primary.pos_w
                    )
                else:
                    center_primary = np.zeros(3, dtype=np.float64)
                right_contact_centers_primary.append(
                    np.asarray(center_primary, dtype=np.float32)
                )
                step = env.step_physical(action)
                frames.append(step.motion_frame)
                observation = step.actor_observation
                max_stage = max(max_stage, int(step.terms.stage_index))
                max_lift = max(max_lift, float(step.terms.lift_height))
                max_grasp = max(max_grasp, float(step.terms.grasp_quality))
                if step.done and terminal is None:
                    terminal = step.terms.to_dict()
                return bool(step.done)
            for action in actions:
                if execute(action, "source"):
                    break

            # The canonical ReplayDecoupledAgent holds the last upper-body
            # target after a Sonic episode and zeros the first three navigation
            # channels.  The upstream replay uses ten frames; reward auditing
            # uses thirty so a 13-frame robust terminal hold can be verified.
            # These are controller-settling frames,
            # not fabricated demonstrations, and are labelled as completion so
            # BC can down-weight them.
            if spec.controller_backend == "sonic_wbc" and terminal is None:
                command = actions[-1].copy()
                command[32:35] = 0.0
                for _ in range(30):
                    if execute(command, "completion"):
                        break

            # Historical MP export ends while still holding the object short of
            # table two.  Complete it using only current simulator feedback;
            # these lower-weight frames make the source trajectory executable
            # without pretending they were part of the immutable dataset.
            if spec.name == "locomotion_pick_between_tables" and terminal is None:
                command = actions[-1].copy()
                command[34] = 0.0
                command[35] = np.pi
                for _ in range(260):
                    _, state = env.state.actor_observation()
                    delta = state.destination.pos_w - state.primary.pos_w
                    if np.linalg.norm(delta[:2]) < .045:
                        break
                    local = state.pelvis_rot_w.T @ delta
                    command[32] = np.clip(1.2 * local[0], -.25, .25)
                    command[33] = np.clip(1.2 * local[1], -.25, .25)
                    if execute(command, "completion"):
                        break
                if terminal is None:
                    command[32:34] = 0.0
                    for _ in range(100):
                        if execute(command, "completion"):
                            break
                if terminal is None:
                    closed = command[7:14].copy()
                    opened = actions[0, 7:14].copy()
                    for step_index in range(60):
                        command[7:14] = closed + (opened - closed) * ((step_index + 1) / 60.0)
                        if execute(command, "completion"):
                            break
                if terminal is None:
                    for _ in range(100):
                        if execute(command, "completion"):
                            break
            observation_array = np.asarray(observations, dtype=np.float32)
            raw_array = np.asarray(raw_actions, dtype=np.float32)
            if observation_array.shape != (len(physical_actions), ACTOR_OBS_V2_DIM):
                raise RuntimeError(f"Episode {episode}: observations {observation_array.shape}")
            total_frames = len(observations)
            phase = np.arange(total_frames, dtype=np.float32) / max(total_frames - 1, 1)
            sample_weights = 1.0 + 2.0 * phase ** 2
            sample_weights[np.asarray(sample_sources) == "completion"] *= .5
            np.savez_compressed(bc_path,
                                observations=observation_array, raw_actions=raw_array,
                                physical_actions=np.asarray(physical_actions, dtype=np.float32),
                                right_contact_centers_primary=np.asarray(
                                    right_contact_centers_primary, dtype=np.float32
                                ),
                                right_contact_center_valid=np.asarray(
                                    right_contact_center_valid, dtype=bool
                                ),
                                sample_weights=sample_weights,
                                sample_sources=np.asarray(sample_sources),
                                prepare_version=np.asarray(expected_prepare_version),
                                max_stage=np.asarray(max_stage),
                                max_lift=np.asarray(max_lift),
                                max_grasp_quality=np.asarray(max_grasp),
                                terminal_success=np.asarray(bool(terminal and terminal["success"])),
                                terminal_failure=np.asarray(bool(terminal and terminal["failure"])),
                                terminal_timeout=np.asarray(bool(terminal and terminal["timeout"])))
            np.savez_compressed(episode_path,
                                frames=np.asarray(frames, dtype=np.float32),
                                actions=np.asarray(physical_actions, dtype=np.float32),
                                observations=observation_array)
            final_state = env.state.actor_observation()[1]
            final_xy = (float(np.linalg.norm(final_state.primary.pos_w[:2] - final_state.destination.pos_w[:2]))
                        if final_state.primary.present and final_state.destination.present else None)
            reports.append({
                "episode": episode, "frames": total_frames,
                "source_frames": len(actions),
                "completion_frames": total_frames - len(actions), "repair": repair,
                "success": bool(terminal and terminal["success"]),
                "failure": bool(terminal and terminal["failure"]),
                "timeout": bool(terminal and terminal["timeout"]),
                "terminal_step": None if terminal is None else int(env.reward.step_count),
                "max_stage": max_stage, "max_lift": max_lift,
                "max_grasp_quality": max_grasp, "final_primary_destination_xy": final_xy,
                "finite": bool(np.isfinite(observation_array).all()),
            })
    finally:
        env.close()
    return reports


def prepare_v2_dataset(dataset_dir: str | Path, output_dir: str | Path,
                       num_workers: int = 8, seed: int = 42,
                       task: str | TaskSpecV2 | None = None,
                       episodes: int | None = None,
                       episode_id: int | None = None,
                       warmup_steps: int = 60) -> Path:
    spec = get_task_spec(task)
    if not isinstance(spec, TaskSpecV2):
        raise ValueError("prepare_v2_dataset requires a v2 task")
    dataset, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    (output / "bc").mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()]
    all_episode_ids = list(range(len(rows)))
    matching_episode_ids = [
        episode for episode in all_episode_ids
        if json.loads(rows[episode]["environment_config"]).get("uid") in spec.source_uids
    ]
    excluded = sorted(set(all_episode_ids) - set(matching_episode_ids))
    if not matching_episode_ids:
        raise ValueError(f"No episodes match runtime source UIDs {spec.source_uids}")
    if episodes is not None and episode_id is not None:
        raise ValueError("episodes and episode_id are mutually exclusive")
    if episodes is not None:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        episode_ids = matching_episode_ids[:episodes]
    elif episode_id is not None:
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        if episode_id not in matching_episode_ids:
            raise ValueError(
                f"Episode {episode_id} does not match runtime source UIDs "
                f"{spec.source_uids}"
            )
        episode_ids = [episode_id]
    else:
        episode_ids = matching_episode_ids
    repaired = [repair_actions(spec, read_actions(dataset, i))[0] for i in episode_ids]
    transform = compute_action_transform(repaired, legacy_tabletop_locomotion_bounds=False)
    if spec.name == "locomotion_pick_between_tables":
        transform.low[32:34] = np.minimum(transform.low[32:34], -.25)
        transform.high[32:34] = np.maximum(transform.high[32:34], .35)
        transform.max_delta[32:34] = np.maximum(transform.max_delta[32:34], .05)
    transform_path = output / "action_transform.npz"
    if transform_path.exists():
        existing_transform = ActionTransform.from_npz(transform_path)
        fields = ("center", "low", "high", "max_delta")
        if all(
            np.array_equal(getattr(existing_transform, name), getattr(transform, name))
            for name in fields
        ):
            # Preserve the byte-level hash stored in policy checkpoints when
            # a replay-only regeneration does not change action semantics.
            transform = existing_transform
        else:
            transform.save(transform_path)
    else:
        transform.save(transform_path)
    chunks = [list(map(int, chunk)) for chunk in np.array_split(episode_ids, min(num_workers, len(episode_ids))) if len(chunk)]
    payload = {name: getattr(transform, name) for name in ("center", "low", "high", "max_delta")}
    devices = _worker_devices()
    args = [(str(dataset), str(output), rows, chunk, payload, spec.name,
             devices[worker % len(devices)], warmup_steps)
            for worker, chunk in enumerate(chunks)]
    if len(args) == 1:
        reports = _prepare_chunk(*args[0])
    else:
        with mp.get_context("spawn").Pool(len(args)) as pool:
            reports = [item for group in pool.starmap(_prepare_chunk, args) for item in group]
    reports.sort(key=lambda x: x["episode"])
    replay_success_rate = float(np.mean([r["success"] for r in reports]))
    failed_replay_episodes = [int(r["episode"]) for r in reports if not r["success"]]
    (output / "replay_audit.json").write_text(json.dumps({
        "task": spec.name,
        "prepare_version": prepare_version(spec),
        "reports": reports,
        "replay_success_rate": replay_success_rate,
        "failed_replay_episodes": failed_replay_episodes,
    }, indent=2))
    # Never train a policy on a source episode that the selected controller,
    # simulator and reward graph cannot reproduce.  This gate applies to every
    # v2 task; task registration alone is not evidence of executable data.
    usable_episode_ids = _usable_replay_episode_ids(
        reports, minimum=1 if episode_id is not None else 3
    )
    # The policy path always executes normalized outputs through decode(), so
    # its transform must reproduce every accepted demonstration exactly.  The
    # earlier quantile-based slew limiter clipped rare but intentional Sonic
    # gripper jumps by as much as 1.5 rad, despite direct physical replay
    # succeeding.  Refit on replay-success data and regenerate normalized BC
    # targets without touching the immutable source dataset.
    successful_actions = []
    for episode in usable_episode_ids:
        with np.load(
            output / "bc" / f"episode_{episode:06d}.npz", allow_pickle=False
        ) as saved:
            successful_actions.append(saved["physical_actions"].copy())
    transform = _successful_replay_transform(successful_actions)
    transform.save(transform_path)
    _rewrite_successful_raw_actions(output, usable_episode_ids, transform)
    if len(usable_episode_ids) == 1:
        splits = {"train": usable_episode_ids.copy(), "val": [], "test": []}
    else:
        shuffled = np.asarray(usable_episode_ids)
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled)
        train_stop = max(1, int(.8 * len(shuffled)))
        val_stop = min(
            train_stop + max(1, int(.1 * len(shuffled))), len(shuffled) - 1
        )
        splits = {"train": sorted(shuffled[:train_stop].tolist()),
                  "val": sorted(shuffled[train_stop:val_stop].tolist()),
                  "test": sorted(shuffled[val_stop:].tolist())}
    source_hash = hashlib.sha256()
    for episode in episode_ids:
        source_hash.update(episode_file(dataset, episode).read_bytes())
    manifest = {
        "source": str(dataset), "source_sha256": source_hash.hexdigest(),
        "source_immutable": True, "source_state_vector_dim": 32,
        "action_transform_version": ACTION_TRANSFORM_VERSION,
        "action_transform_sha256": _sha256(transform_path),
        "task": spec.name, "task_metadata": spec.metadata(), "seed": seed,
        "unique_episodes": usable_episode_ids, "source_episode_ids": episode_ids,
        "requested_episode_id": episode_id,
        "splits": splits, "schema": schema_dict(),
        "excluded_episodes": excluded,
        "excluded_reason": "different controller/source UID" if excluded else None,
        "repair_version": REPAIR_VERSION if spec.name == "locomotion_pick_between_tables" else "none",
        "reports": reports,
        "failed_replay_episodes": failed_replay_episodes,
        "replay_success_rate": replay_success_rate,
        "replay_gate_passed": replay_success_rate >= .90,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output


def write_repaired_actions(dataset_dir: str | Path, output_dir: str | Path,
                           task: str | TaskSpecV2) -> Path:
    """Materialize only derived repaired commands plus provenance."""

    dataset, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    spec = get_task_spec(task)
    if not isinstance(spec, TaskSpecV2):
        raise ValueError("repair-data is only available for v2 tasks")
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()]
    reports = []
    for episode in range(len(rows)):
        actions, report = repair_actions(spec, read_actions(dataset, episode))
        np.save(output / f"episode_{episode:06d}.npy", actions)
        reports.append({"episode": episode, **report})
    (output / "repair_manifest.json").write_text(json.dumps({
        "source": str(dataset), "task": spec.name, "version": REPAIR_VERSION,
        "reports": reports,
    }, indent=2))
    return output
