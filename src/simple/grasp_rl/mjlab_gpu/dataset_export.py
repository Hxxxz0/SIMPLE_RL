"""Export successful GPU PPO rollouts as GR00T and Psi0 LeRobot datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from simple.grasp_rl.mjlab_gpu.action import tracker_hand_targets
from simple.grasp_rl.mjlab_gpu.collect import validate_episode_arrays
from simple.grasp_rl.schema import JOINT_NAMES

WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")

# The source GR00T datasets put hands directly after the corresponding arm.
RAW_ACTION_NAMES = (
    *JOINT_NAMES[:22],
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    *JOINT_NAMES[22:29],
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), default=_json_default) + "\n"
            for row in rows
        )
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _source_instruction(source_episode: dict[str, Any]) -> str:
    """Resolve the language label from the frozen source episode."""

    tasks = source_episode.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("Source episode must contain exactly one task instruction")
    instruction = tasks[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Source episode task instruction must be a non-empty string")
    return instruction.strip()


def _stats(value: np.ndarray, *, include_count: bool = True) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    result: dict[str, Any] = {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).astype(np.float32).tolist(),
    }
    if include_count:
        result["count"] = [len(array)]
    return result


def _load_render_model(asset_bundle: Path) -> mujoco.MjModel:
    render_manifest_path = asset_bundle / "render_manifest.json"
    manifest = json.loads(render_manifest_path.read_text())
    unhashed = dict(manifest)
    expected_hash = unhashed.pop("manifest_hash", None)
    actual_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("Render sidecar manifest hash mismatch")
    scene = (asset_bundle / manifest["scene_file"]).resolve()
    if not scene.is_relative_to(asset_bundle.resolve()):
        raise ValueError("Render scene escapes the asset bundle")
    if _sha256(scene) != manifest["scene_sha256"]:
        raise ValueError("Render scene hash mismatch")
    for item in manifest["assets"]:
        path = (asset_bundle / item["bundle_path"]).resolve()
        if (
            not path.is_relative_to(asset_bundle.resolve())
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Render asset hash mismatch: {path}")
    return mujoco.MjModel.from_xml_path(str(scene))


def _joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(
        [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES],
        dtype=np.int64,
    )
    qvel = np.asarray(
        [model.jnt_dofadr[model.joint(name).id] for name in JOINT_NAMES],
        dtype=np.int64,
    )
    return qpos, qvel


def _object_qpos_addresses(
    model: mujoco.MjModel, source_info: dict[str, Any]
) -> np.ndarray:
    """Match source object-pose columns to non-robot free joints in scene order."""

    try:
        shape = source_info["features"]["observation.object_poses"]["shape"]
    except (KeyError, TypeError) as error:
        raise ValueError("Source dataset is missing observation.object_poses") from error
    if not isinstance(shape, list) or len(shape) != 1 or int(shape[0]) % 7:
        raise ValueError("Source object poses must be a flat sequence of 7-D poses")
    expected_objects = int(shape[0]) // 7
    root_joint = model.joint("floating_base_joint").id
    free_joint = int(mujoco.mjtJoint.mjJNT_FREE)
    addresses = [
        int(model.jnt_qposadr[index])
        for index in range(model.njnt)
        if index != root_joint and int(model.jnt_type[index]) == free_joint
    ]
    if len(addresses) != expected_objects:
        raise ValueError(
            "Render scene object topology does not match source metadata: "
            f"found {len(addresses)} free objects, expected {expected_objects}"
        )
    return np.asarray(addresses, dtype=np.int64)


def _eef(data: mujoco.MjData) -> np.ndarray:
    values = []
    for name in WRIST_BODIES:
        body = data.body(name)
        values.extend(body.xpos.tolist())
        values.extend(body.xquat.tolist())
    return np.asarray(values, dtype=np.float64)


def _physical_to_joint_command(
    physical_action: np.ndarray, joint_target: np.ndarray
) -> np.ndarray:
    """Combine Sonic lower targets with the public 36-D upper-body command."""

    command = np.asarray(joint_target, dtype=np.float32).copy()
    command[:, 12] = physical_action[:, 30]
    command[:, 13] = physical_action[:, 28]
    command[:, 14] = physical_action[:, 29]
    command[:, 15:22] = physical_action[:, 14:21]
    command[:, 22:29] = physical_action[:, 21:28]
    hands = tracker_hand_targets(
        torch.as_tensor(physical_action, dtype=torch.float32)
    ).numpy()
    command[:, 29:43] = hands
    return command


def _render_and_derive(
    model: mujoco.MjModel,
    arrays: dict[str, np.ndarray],
    raw_video: Path,
    object_qpos_addresses: np.ndarray,
    *,
    camera: str,
    width: int,
    height: int,
    fps: int,
) -> dict[str, np.ndarray]:
    steps = len(arrays["physical_action"])
    if arrays["qpos"].shape[1] != model.nq or arrays["qvel"].shape[1] != model.nv:
        raise ValueError("Rollout state layout does not match the render model")
    if model.camera(camera).id < 0:
        raise ValueError(f"Render model has no camera named {camera!r}")
    qpos_addresses, _ = _joint_addresses(model)
    root_joint = model.joint("floating_base_joint").id
    root_qpos = int(model.jnt_qposadr[root_joint])
    root_qvel = int(model.jnt_dofadr[root_joint])
    command = _physical_to_joint_command(
        arrays["physical_action"], arrays["joint_target"]
    )
    raw_order = np.asarray([JOINT_NAMES.index(name) for name in RAW_ACTION_NAMES])

    joint_state = arrays["qpos"][:-1, qpos_addresses].astype(np.float64)
    raw_action = command[:, raw_order].astype(np.float64)
    observation_eef = np.empty((steps, 14), dtype=np.float64)
    action_eef = np.empty((steps, 14), dtype=np.float64)
    object_poses = np.empty((steps, 7 * len(object_qpos_addresses)), dtype=np.float64)
    base_pose = arrays["qpos"][:-1, root_qpos : root_qpos + 7].astype(np.float64)
    base_vel = arrays["qvel"][:-1, root_qvel : root_qvel + 6].astype(np.float64)

    raw_video.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = imageio.get_writer(
        raw_video,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=1,
    )
    try:
        for index in range(steps):
            data.qpos[:] = arrays["qpos"][index]
            data.qvel[:] = arrays["qvel"][index]
            mujoco.mj_forward(model, data)
            observation_eef[index] = _eef(data)
            object_poses[index] = np.concatenate(
                [
                    data.qpos[address : address + 7]
                    for address in object_qpos_addresses
                ]
            )
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

            data.qpos[qpos_addresses] = command[index]
            mujoco.mj_forward(model, data)
            action_eef[index] = _eef(data)
    finally:
        writer.close()
        renderer.close()

    return {
        "joint_state": joint_state,
        "raw_action": raw_action,
        "observation_eef": observation_eef,
        "action_eef": action_eef,
        "object_poses": object_poses,
        "base_pose": base_pose,
        "base_vel": base_vel,
    }


def _psi_arrays(
    joint_state: np.ndarray, physical_action: np.ndarray
) -> dict[str, np.ndarray]:
    state = joint_state.astype(np.float32)
    action = physical_action.astype(np.float32)
    previous_height = np.concatenate(
        (np.asarray([0.74], dtype=np.float32), action[:-1, 31])
    )[:, None]
    waist_rpy = state[:, (13, 14, 12)]
    previous_rpy = np.concatenate(
        (np.zeros((1, 3), dtype=np.float32), waist_rpy[:-1]), axis=0
    )
    hand = np.concatenate((state[:, 29:36], state[:, 36:43]), axis=1)
    arm = np.concatenate((state[:, 15:22], state[:, 22:29]), axis=1)
    leg = state[:, :15]
    psi_state = np.concatenate(
        (
            state[:, 29:32],
            state[:, 34:36],
            state[:, 32:34],
            state[:, 36:43],
            state[:, 15:22],
            state[:, 22:29],
            waist_rpy,
            previous_height,
        ),
        axis=1,
    )
    if psi_state.shape[1] != 32 or action.shape[1] != 36:
        raise AssertionError("Psi0 state/action schema mismatch")
    return {
        "states": psi_state,
        "action": action,
        "observation.hand_joints": hand,
        "observation.arm_joints": arm,
        "observation.leg_joints": leg,
        "observation.prev_torso_rpy": previous_rpy,
        "observation.prev_height": previous_height,
    }


def _append_raw_features(info: dict[str, Any]) -> None:
    features = info["features"]
    for name, size in (
        ("policy.raw_action", 36),
        ("policy.reference_action", 36),
        ("policy.effective_action", 36),
        ("policy.physical_action", 36),
        ("observation.task_privileged", 331),
        ("observation.policy_input", 842),
    ):
        features[name] = {"dtype": "float32", "shape": [size], "names": [name]}
    for name, dtype in (
        ("reward", "float32"),
        ("task_reward", "float32"),
        ("reference_reward", "float32"),
        ("stage_index", "int64"),
        ("next.done", "bool"),
    ):
        features[name] = {"dtype": dtype, "shape": [1], "names": [name]}


def _video_probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def _contact_sheet(video: Path, output: Path) -> None:
    import cv2

    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in (0, max(0, count // 2), max(0, count - 1)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not decode frame {index} from {video}")
        frames.append(frame)
    capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.concatenate(frames, axis=1)):
        raise RuntimeError(f"Could not write {output}")


def export_dual_dataset(
    rollouts_dir: Path,
    output_root: Path,
    asset_bundle: Path,
    source_dataset: Path,
    psi0_template: Path,
    *,
    camera: str = "head_stereo_left",
    width: int = 640,
    height: int = 360,
    fps: int = 50,
) -> dict[str, Any]:
    """Export every successful rollout in ``rollouts_dir`` into both views."""

    rollout_summary = json.loads((rollouts_dir / "summary.json").read_text())
    manifest_rows = _load_jsonl(rollouts_dir / "manifest.jsonl")
    manifest_by_file = {row["file"]: row for row in manifest_rows if row.get("file")}
    episode_paths = sorted((rollouts_dir / "episodes").glob("episode_*.npz"))
    if not episode_paths:
        raise ValueError("No successful rollout episodes found")

    groot_root = output_root / "groot" / "level-0"
    psi_root = output_root / "psi0"
    audit_root = output_root / "audit"
    for path in (groot_root, psi_root):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite dataset view: {path}")

    source_info = json.loads((source_dataset / "meta" / "info.json").read_text())
    psi_info_template = json.loads((psi0_template / "meta" / "info.json").read_text())
    source_episodes = _load_jsonl(source_dataset / "meta" / "episodes.jsonl")
    bundle_manifest = json.loads((asset_bundle / "manifest.json").read_text())
    base_episode = int(bundle_manifest["base_episode"])
    source_episode = next(
        (
            row
            for row in source_episodes
            if int(row["episode_index"]) == base_episode
        ),
        None,
    )
    if source_episode is None:
        raise ValueError(f"Source dataset has no base episode {base_episode}")
    if rollout_summary["task"] != bundle_manifest["task"]:
        raise ValueError(
            "Rollout task does not match frozen asset bundle: "
            f"{rollout_summary['task']} != {bundle_manifest['task']}"
        )
    instruction = _source_instruction(source_episode)
    environment_config = source_episode.get("environment_config", "")
    model = _load_render_model(asset_bundle)
    object_qpos_addresses = _object_qpos_addresses(model, source_info)

    raw_tables: list[dict[str, np.ndarray]] = []
    psi_tables: list[dict[str, np.ndarray]] = []
    raw_episode_meta: list[dict[str, Any]] = []
    psi_episode_meta: list[dict[str, Any]] = []
    raw_episode_stats: list[dict[str, Any]] = []
    psi_episode_stats: list[dict[str, Any]] = []
    total_frames = 0

    for episode_index, episode_path in enumerate(episode_paths):
        with np.load(episode_path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        validate_episode_arrays(arrays)
        steps = len(arrays["physical_action"])
        chunk = episode_index // 1000
        raw_video = (
            groot_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / "observation.images.ego_view"
            / f"episode_{episode_index:06d}.mp4"
        )
        derived = _render_and_derive(
            model,
            arrays,
            raw_video,
            object_qpos_addresses,
            camera=camera,
            width=width,
            height=height,
            fps=fps,
        )
        psi_video = (
            psi_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / "egocentric"
            / f"episode_{episode_index:06d}.mp4"
        )
        psi_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_video, psi_video)

        frame_index = np.arange(steps, dtype=np.int64)
        timestamp = frame_index.astype(np.float32) / float(fps)
        episode_column = np.full(steps, episode_index, dtype=np.int64)
        global_index = np.arange(total_frames, total_frames + steps, dtype=np.int64)
        task_index = np.zeros(steps, dtype=np.int64)
        done = np.zeros(steps, dtype=np.bool_)
        done[-1] = True
        raw = {
            "observation.state": derived["joint_state"],
            "observation.eef_state": derived["observation_eef"],
            "action": derived["raw_action"],
            "action.eef": derived["action_eef"],
            "observation.img_state_delta": np.zeros(steps, dtype=np.float32),
            "teleop.navigate_command": arrays["physical_action"][:, 32:36].astype(
                np.float64
            ),
            "teleop.base_height_command": arrays["physical_action"][:, 31].astype(
                np.float64
            ),
            "observation.base_pose": derived["base_pose"],
            "observation.base_vel": derived["base_vel"],
            "observation.object_poses": derived["object_poses"],
            "policy.raw_action": arrays["raw_action"].astype(np.float32),
            "policy.reference_action": arrays["reference_action"].astype(np.float32),
            "policy.effective_action": arrays["effective_action"].astype(np.float32),
            "policy.physical_action": arrays["physical_action"].astype(np.float32),
            "observation.task_privileged": arrays["observation"].astype(np.float32),
            "observation.policy_input": arrays["policy_input"].astype(np.float32),
            "reward": arrays["reward"].astype(np.float32),
            "task_reward": arrays["task_reward"].astype(np.float32),
            "reference_reward": arrays["reference_reward"].astype(np.float32),
            "stage_index": arrays["stage_index"].astype(np.int64),
            "next.done": done,
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": episode_column,
            "index": global_index,
            "task_index": task_index,
        }
        psi = {
            **_psi_arrays(derived["joint_state"], arrays["physical_action"]),
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": episode_column,
            "index": global_index,
            "task_index": task_index,
            "next.done": done,
        }
        raw_tables.append(raw)
        psi_tables.append(psi)

        raw_data = (
            groot_root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        psi_data = (
            psi_root
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        raw_data.parent.mkdir(parents=True, exist_ok=True)
        psi_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({key: value.tolist() for key, value in raw.items()}), raw_data
        )
        pq.write_table(
            pa.table({key: value.tolist() for key, value in psi.items()}), psi_data
        )

        relative_file = str(episode_path.relative_to(rollouts_dir))
        rollout_record = manifest_by_file[relative_file]
        provenance = {
            "rollout": rollout_record,
            "checkpoint": rollout_summary["checkpoint"],
            "asset_manifest_hash": bundle_manifest["manifest_hash"],
            "reward_hash": bundle_manifest["reward_hash"],
            "task_spec_hash": bundle_manifest["task_spec_hash"],
        }
        raw_episode_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": steps,
                "environment_config": environment_config,
                "policy_provenance": provenance,
            }
        )
        psi_episode_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [0],
                "length": steps,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + steps - 1,
                "robot_type": "g1",
                "instruction": {"task_index": 0, "task": instruction},
                "environment_config": environment_config,
                "policy_provenance": provenance,
            }
        )
        raw_episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    key: _stats(value)
                    for key, value in raw.items()
                    if np.asarray(value).dtype.kind in "fciub"
                    and key not in {"index", "episode_index", "task_index"}
                },
            }
        )
        psi_episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "action": _stats(psi["action"]),
                    "timestamp": _stats(timestamp),
                },
            }
        )
        total_frames += steps
        if (episode_index + 1) % 10 == 0 or episode_index + 1 == len(episode_paths):
            print(
                f"[dataset-export] completed {episode_index + 1}/{len(episode_paths)} episodes",
                flush=True,
            )

    raw_info = copy.deepcopy(source_info)
    raw_info.update(
        total_episodes=len(episode_paths),
        total_frames=total_frames,
        total_videos=len(episode_paths),
        total_chunks=(len(episode_paths) + 999) // 1000,
        chunks_size=1000,
        fps=fps,
        splits={"train": f"0:{len(episode_paths)}"},
        script_config={
            "source": "physical_reward_gpu_ppo",
            "camera": camera,
            "checkpoint_sha256": rollout_summary["checkpoint"]["checkpoint_sha256"],
        },
    )
    raw_info["features"]["observation.images.ego_view"]["shape"] = [height, width, 3]
    _append_raw_features(raw_info)
    psi_info = copy.deepcopy(psi_info_template)
    psi_info.update(
        total_episodes=len(episode_paths),
        total_frames=total_frames,
        total_videos=len(episode_paths),
        total_chunks=(len(episode_paths) + 999) // 1000,
        chunks_size=1000,
        fps=fps,
    )
    psi_info["features"]["observation.images.egocentric"]["shape"] = [height, width, 3]

    task_row = {"task_index": 0, "task": instruction}
    psi_task_row = {**task_row, "category": "", "description": instruction}
    _write_json(groot_root / "meta" / "info.json", raw_info)
    _write_jsonl(groot_root / "meta" / "tasks.jsonl", [task_row])
    _write_jsonl(groot_root / "meta" / "episodes.jsonl", raw_episode_meta)
    _write_jsonl(groot_root / "meta" / "episodes_stats.jsonl", raw_episode_stats)
    shutil.copy2(
        source_dataset / "meta" / "modality.json", groot_root / "meta" / "modality.json"
    )

    _write_json(psi_root / "meta" / "info.json", psi_info)
    _write_jsonl(psi_root / "meta" / "tasks.jsonl", [psi_task_row])
    _write_jsonl(psi_root / "meta" / "episodes.jsonl", psi_episode_meta)
    _write_jsonl(psi_root / "meta" / "episodes_stats.jsonl", psi_episode_stats)
    shutil.copy2(
        psi0_template / "meta" / "modality.json", psi_root / "meta" / "modality.json"
    )
    _write_json(psi_root / "meta" / "lang_map.json", {instruction: 0})
    all_psi = {
        key: np.concatenate([row[key] for row in psi_tables], axis=0)
        for key in psi_tables[0]
    }
    global_stats = {
        key: _stats(all_psi[key], include_count=False)
        for key in (
            "states",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "next.done",
        )
    }
    for name in ("stats.json", "stats_psi0.json"):
        _write_json(psi_root / "meta" / name, global_stats)
    _write_json(psi_root / "meta" / "relative_stats.json", {})

    first_video = (
        groot_root
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / "episode_000000.mp4"
    )
    probe = _video_probe(first_video)
    first_rows = pq.read_table(
        groot_root / "data" / "chunk-000" / "episode_000000.parquet"
    ).num_rows
    first_psi_rows = pq.read_table(
        psi_root / "data" / "chunk-000" / "episode_000000.parquet"
    ).num_rows
    if int(probe["nb_frames"]) != first_rows or first_rows != first_psi_rows:
        raise ValueError("Video and Parquet frame counts do not match")
    if (
        probe["codec_name"] != "h264"
        or probe["pix_fmt"] != "yuv420p"
        or int(probe["width"]) != width
        or int(probe["height"]) != height
        or probe["r_frame_rate"] != f"{fps}/1"
    ):
        raise ValueError(f"Unexpected video encoding: {probe}")
    contact_sheet = audit_root / "episode_000000_contact_sheet.jpg"
    _contact_sheet(first_video, contact_sheet)
    audit = {
        "schema_version": 1,
        "complete": True,
        "episodes": len(episode_paths),
        "frames": total_frames,
        "camera": camera,
        "video": probe,
        "groot_root": str(groot_root),
        "psi0_root": str(psi_root),
        "contact_sheet": str(contact_sheet),
        "checkpoint": rollout_summary["checkpoint"],
        "all_rollouts_physical_success": True,
    }
    _write_json(audit_root / "summary.json", audit)
    return audit
