"""Replay recorded tracker inputs and build actual-motion diffusion windows."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.motion import numpy_frames_to_features
from simple.grasp_rl.schema import JOINT_NAMES, MOTION_WINDOW, schema_dict
from simple.grasp_rl.task_spec import GraspTaskSpec, get_task_spec
from simple.grasp_rl.tracker import ActionTransform, compute_action_transform


@dataclass
class ReplayReport:
    episode: int
    frames: int
    joint_rmse: float
    arm_rmse: float
    max_joint_error: float
    max_lift: float
    final_lift: float
    valid_grasp_steps: int
    finite: bool


def _read_actions(path: Path) -> np.ndarray:
    return np.asarray(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float32)


def _recorded_qpos(table: Any) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(table["observation.leg_joints"].to_pylist(), dtype=np.float32),
            np.asarray(table["observation.arm_joints"].to_pylist(), dtype=np.float32),
            np.asarray(table["observation.hand_joints"].to_pylist(), dtype=np.float32),
        ],
        axis=1,
    )


def _episode_file(dataset_dir: Path, episode: int) -> Path:
    return dataset_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"


def _replay_chunk(
    dataset_dir_string: str,
    output_dir_string: str,
    episodes: list[int],
    rows: list[dict[str, Any]],
    transform_payload: dict[str, np.ndarray],
    cuda_device: int,
    task_name: str,
) -> list[dict[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    os.environ.setdefault("MUJOCO_GL", "egl")
    dataset_dir = Path(dataset_dir_string)
    output_dir = Path(output_dir_string)
    transform = ActionTransform(**transform_payload)
    env = GraspRlEnv(
        transform,
        seed=42 + episodes[0],
        warmup_steps=60,
        task=task_name,
    )
    reports: list[dict[str, Any]] = []
    try:
        for episode in episodes:
            path = _episode_file(dataset_dir, episode)
            table = pq.read_table(path)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            recorded = _recorded_qpos(table)
            state_dict = json.loads(rows[episode]["environment_config"])
            env.reset(state_dict=state_dict)
            initial_z = float(env.state.initial_object_pos[2])  # type: ignore[union-attr]
            frames: list[np.ndarray] = []
            replay_qpos: list[np.ndarray] = []
            max_lift = 0.0
            grasp_steps = 0
            for action in actions:
                step = env.step_physical(action)
                frames.append(step.motion_frame)
                replay_qpos.append(
                    np.fromiter(
                        (env.sim.mjData.joint(name).qpos.item() for name in JOINT_NAMES),
                        dtype=np.float32,
                        count=43,
                    )
                )
                max_lift = max(max_lift, step.terms.lift_height)
                grasp_steps += int(step.terms.is_grasp)

            frame_array = np.stack(frames)
            if len(frame_array) < MOTION_WINDOW:
                raise RuntimeError(f"Episode {episode} is shorter than the motion window")
            raw_windows = np.lib.stride_tricks.sliding_window_view(
                frame_array, window_shape=MOTION_WINDOW, axis=0
            ).transpose(0, 2, 1)
            windows = numpy_frames_to_features(raw_windows).astype(np.float32)
            np.savez_compressed(
                output_dir / "episodes" / f"episode_{episode:06d}.npz",
                frames=frame_array,
                windows=windows,
                actions=actions,
            )

            replay = np.stack(replay_qpos)
            error = replay - recorded
            report = ReplayReport(
                episode=episode,
                frames=len(actions),
                joint_rmse=float(np.sqrt(np.mean(error**2))),
                arm_rmse=float(np.sqrt(np.mean(error[:, 15:29] ** 2))),
                max_joint_error=float(np.max(np.abs(error))),
                max_lift=float(max_lift),
                final_lift=float(env.sim.mj_objects["target"].xpos[2] - initial_z),
                valid_grasp_steps=grasp_steps,
                finite=bool(np.isfinite(frame_array).all() and np.isfinite(windows).all()),
            )
            reports.append(asdict(report))
    finally:
        env.close()
    return reports


def prepare_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    num_workers: int = 8,
    seed: int = 42,
    task: str | GraspTaskSpec | None = None,
) -> Path:
    task_spec = get_task_spec(task)
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in (dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines()]
    episodes = list(range(len(rows)))
    action_episodes = [_read_actions(_episode_file(dataset_dir, episode)) for episode in episodes]
    fingerprints: dict[str, int] = {}
    duplicates: dict[int, int] = {}
    unique_episodes = []
    for episode, actions in zip(episodes, action_episodes, strict=True):
        digest = hashlib.sha256()
        digest.update(rows[episode]["environment_config"].encode())
        digest.update(np.ascontiguousarray(actions).tobytes())
        fingerprint = digest.hexdigest()
        if fingerprint in fingerprints:
            duplicates[episode] = fingerprints[fingerprint]
        else:
            fingerprints[fingerprint] = episode
            unique_episodes.append(episode)
    if len(unique_episodes) < 3:
        raise ValueError("At least three unique demonstrations are required")
    transform = compute_action_transform(
        [action_episodes[episode] for episode in unique_episodes],
        legacy_tabletop_locomotion_bounds=(
            task_spec.name == "tabletop_grasp"
        ),
    )
    transform.save(output_dir / "action_transform.npz")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique_episodes)
    rng.shuffle(shuffled)
    train_stop = max(1, int(np.floor(0.8 * len(shuffled))))
    val_count = max(1, int(np.floor(0.1 * len(shuffled))))
    val_stop = min(train_stop + val_count, len(shuffled) - 1)
    splits = {
        "train": sorted(shuffled[:train_stop].tolist()),
        "val": sorted(shuffled[train_stop:val_stop].tolist()),
        "test": sorted(shuffled[val_stop:].tolist()),
    }

    chunks = [
        list(map(int, chunk))
        for chunk in np.array_split(
            unique_episodes, min(num_workers, len(unique_episodes))
        )
        if len(chunk)
    ]
    payload = {
        "center": transform.center,
        "low": transform.low,
        "high": transform.high,
        "max_delta": transform.max_delta,
    }
    if len(chunks) == 1:
        all_reports = _replay_chunk(
            str(dataset_dir),
            str(output_dir),
            chunks[0],
            rows,
            payload,
            0,
            task_spec.name,
        )
    else:
        context = mp.get_context("spawn")
        args = [
            (
                str(dataset_dir),
                str(output_dir),
                chunk,
                rows,
                payload,
                worker % 8,
                task_spec.name,
            )
            for worker, chunk in enumerate(chunks)
        ]
        with context.Pool(len(chunks)) as pool:
            nested = pool.starmap(_replay_chunk, args)
        all_reports = [item for group in nested for item in group]
    all_reports.sort(key=lambda item: item["episode"])

    train_windows = []
    for episode in splits["train"]:
        with np.load(episode_dir / f"episode_{episode:06d}.npz", allow_pickle=False) as data:
            train_windows.append(data["windows"])
    flat = np.concatenate(train_windows, axis=0).reshape(-1, schema_dict()["motion_feature_dim"])
    q_low = np.quantile(flat, 0.01, axis=0).astype(np.float32)
    q_high = np.quantile(flat, 0.99, axis=0).astype(np.float32)
    tiny = q_high - q_low < 1e-6
    q_high[tiny] = q_low[tiny] + 1.0
    np.savez(output_dir / "norm_stats.npz", q_low=q_low, q_high=q_high)

    source_hash = hashlib.sha256()
    for episode in unique_episodes:
        source_hash.update(_episode_file(dataset_dir, episode).read_bytes())
    manifest = {
        "source": str(dataset_dir),
        "source_sha256": source_hash.hexdigest(),
        "task": task_spec.name,
        "task_metadata": task_spec.metadata(),
        "seed": seed,
        "unique_episodes": unique_episodes,
        "duplicates": {str(key): value for key, value in duplicates.items()},
        "splits": splits,
        "schema": schema_dict(),
        "reports": all_reports,
        "num_windows": {
            split: int(
                sum(
                    len(np.load(episode_dir / f"episode_{episode:06d}.npz", allow_pickle=False)["windows"])
                    for episode in ids
                )
            )
            for split, ids in splits.items()
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_dir
