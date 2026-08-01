"""Render saved full-command evaluations through the real SIMPLE tracker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch
from tensordict import TensorDict

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.policy import add_optional_phase, load_actor
from simple.grasp_rl.reference import ReferenceLibrary, ReferenceTracker
from simple.grasp_rl.schema import REFERENCE_ACTOR_OBS_DIM, REFERENCE_ACTOR_OBS_V2_DIM
from simple.grasp_rl.task_spec import TaskSpec, TaskSpecV2, get_task_spec
from simple.grasp_rl.tracker import ActionTransform


def _select_camera(camera: str, available: tuple[str, ...]) -> str:
    """Resolve a portable default while keeping explicit names strict."""

    if not available:
        raise KeyError("The environment exposes no render cameras")
    if camera == "auto":
        for preferred in ("front_stereo_left", "head_stereo_left"):
            if preferred in available:
                return preferred
        return available[0]
    if camera not in available:
        raise KeyError(f"Unknown camera {camera}; available cameras: {available}")
    return camera


def render_saved_trajectory(
    trajectory_path: str | Path,
    action_transform_path: str | Path,
    dataset_dir: str | Path,
    output_path: str | Path,
    camera: str = "auto",
    fps: float = 50.0,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda:0",
    camera_fovy: float | None = 40.0,
    expert: bool = False,
    context_start: int | None = None,
    robot_position_xy: tuple[float, float] | None = None,
    task: str | TaskSpec | None = None,
) -> Path:
    """Run a closed-loop checkpoint (or replay commands) and encode an MP4."""
    task_spec = get_task_spec(task)
    is_v2 = isinstance(task_spec, TaskSpecV2)
    trajectory = Path(trajectory_path)
    rollout_episode = int(
        trajectory.stem.removeprefix("episode_").split("_repeat_", 1)[0]
    )
    saved_initial_qpos = None
    saved_initial_qvel = None
    saved_reference_episode = None
    saved_target_position = None
    saved_target_quaternion = None
    if not expert:
        with np.load(trajectory, allow_pickle=False) as saved:
            actions = saved["actions"].astype(np.float32)
            episode = (
                int(saved["base_episode"])
                if "base_episode" in saved
                else rollout_episode
            )
            saved_initial_qpos = (
                saved["initial_qpos"].copy()
                if "initial_qpos" in saved
                else None
            )
            saved_initial_qvel = (
                saved["initial_qvel"].copy()
                if "initial_qvel" in saved
                else None
            )
            if "target_position" in saved:
                saved_target_position = saved["target_position"].copy()
            if "target_quaternion" in saved:
                saved_target_quaternion = saved["target_quaternion"].copy()
            if "reference_episode" in saved and int(saved["reference_episode"]) >= 0:
                saved_reference_episode = int(saved["reference_episode"])
            if robot_position_xy is None and "robot_position_xy" in saved:
                saved_robot_xy = saved["robot_position_xy"].astype(np.float64)
                if saved_robot_xy.shape == (2,) and np.isfinite(saved_robot_xy).all():
                    robot_position_xy = tuple(map(float, saved_robot_xy))
    else:
        episode = rollout_episode
    rows = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (Path(dataset_dir) / "meta" / "episodes.jsonl")
            .read_text()
            .splitlines()
        )
    }
    if episode not in rows:
        raise KeyError(f"Episode {episode} is not present in {dataset_dir}")
    if expert:
        parquet = (
            Path(dataset_dir)
            / "data"
            / "chunk-000"
            / f"episode_{episode:06d}.parquet"
        )
        recorded = np.asarray(
            pq.read_table(parquet, columns=["action"])["action"].to_pylist(),
            dtype=np.float32,
        )
        actions = np.concatenate(
            [recorded, np.repeat(recorded[-1:], 40, axis=0)], axis=0
        )
    transform = ActionTransform.from_npz(action_transform_path)
    env = GraspRlEnv(
        transform,
        seed=1234,
        task=task_spec,
        enable_renderers=True,
    )
    actor = (
        load_actor(
            checkpoint_path,
            device,
            expected_task=task_spec,
            action_transform=action_transform_path,
        )
        if checkpoint_path
        else None
    )
    reference = None
    if actor is not None and int(
        getattr(actor, "grasp_observation_dim", 192)
    ) in (REFERENCE_ACTOR_OBS_DIM, REFERENCE_ACTOR_OBS_V2_DIM):
        reference = ReferenceTracker(
            ReferenceLibrary(Path(action_transform_path).parent)
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f"{output.stem}.mp4v{output.suffix}")
    writer = None
    completed = False
    try:
        actor_policy_step = 0

        def actor_step(observation: np.ndarray):
            nonlocal actor_policy_step
            assert actor is not None
            policy_observation = (
                reference.augment(observation)
                if reference is not None
                else observation
            )
            observation_tensor = add_optional_phase(
                actor,
                torch.as_tensor(policy_observation[None], device=device),
                actor_policy_step,
            )
            with torch.no_grad():
                raw_action = (
                    actor(
                        TensorDict(
                            {"actor": observation_tensor},
                            batch_size=[1],
                            device=device,
                        ),
                        stochastic_output=False,
                    )[0]
                    .cpu()
                    .numpy()
                )
            actor_policy_step += 1
            if reference is not None:
                assert env.reward is not None
                if hasattr(env.reward, "set_reference_contact"):
                    env.reward.set_reference_contact(
                        reference.post_step_contact_label(),
                        reference.post_step_contact_center_primary(),
                    )
            step = env.step_raw(raw_action)
            if reference is not None:
                reference.reward(
                    step.actor_observation,
                    transform.encode(env.previous_physical_action),
                )
            return step

        if actor is not None and context_start is not None:
            if not 0 <= context_start <= episode:
                raise ValueError("context_start must be between 0 and the episode")
            for context_episode in range(context_start, episode):
                actor_policy_step = 0
                context_observation, _ = env.reset(
                    state_dict=json.loads(
                        rows[context_episode]["environment_config"]
                    )
                )
                if reference is not None:
                    reference.reset(
                        context_observation,
                        exact_episode=context_episode,
                    )
                for _ in range(task_spec.max_episode_steps):
                    context_step = actor_step(context_observation)
                    context_observation = context_step.actor_observation
                    if context_step.done:
                        break
        state_dict = json.loads(rows[episode]["environment_config"])
        if robot_position_xy is not None:
            robot_uid = state_dict["robot_cfg"]["uid"]
            state_dict["dr_state_dict"]["spatial"][robot_uid]["position"][:2] = (
                robot_position_xy
            )
        observation, _ = env.reset(state_dict=state_dict)
        if (
            is_v2
            and saved_initial_qpos is not None
            and saved_initial_qvel is not None
        ):
            observation, _ = env.restore_v2_state(
                saved_initial_qpos, saved_initial_qvel
            )
        elif saved_initial_qpos is not None and not is_v2:
            env.capture_fast_reset_snapshot(randomize_target=False)
            _, current_quaternion = env.target_freejoint_pose()
            saved_position, saved_quaternion = env.target_pose_from_qpos(
                saved_initial_qpos
            )
            if np.linalg.norm(saved_quaternion) < 1e-8:
                saved_quaternion = current_quaternion
            observation, _ = env.reset_to_target_pose(
                saved_position, saved_quaternion
            )
        elif (
            is_v2
            and saved_target_position is not None
            and saved_target_quaternion is not None
            and np.linalg.norm(saved_target_quaternion) >= 1e-8
        ):
            observation, _ = env.set_primary_pose(
                saved_target_position, saved_target_quaternion
            )
        if reference is not None:
            reference.reset(
                observation,
                exact_episode=(
                    saved_reference_episode
                    if saved_reference_episode is not None
                    else episode
                ),
            )
        actor_policy_step = 0
        available = tuple(env.sim.renderers)
        camera = _select_camera(camera, available)
        if camera_fovy is not None:
            camera_id = mujoco.mj_name2id(
                env.sim.mjModel, mujoco.mjtObj.mjOBJ_CAMERA, camera
            )
            env.sim.mjModel.cam_fovy[camera_id] = camera_fovy
        horizon = task_spec.max_episode_steps if actor is not None else len(actions)
        success = False
        max_lift = -float("inf")
        for index in range(horizon):
            if actor is None:
                step = env.step_physical(actions[index])
            else:
                step = actor_step(observation)
            observation = step.actor_observation
            success |= step.terms.success
            max_lift = max(max_lift, step.terms.lift_height)
            renderer = env.sim.renderers[camera]
            renderer.update_scene(
                env.sim.mjData,
                scene_option=env.sim.render_option,
                camera=camera,
            )
            rgb = renderer.render()[..., :3].astype(np.uint8, copy=False).copy()
            height, width = rgb.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(staging),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output}")
            status = (
                f"episode {episode}  step {index + 1}/{horizon}  "
                f"stage {step.terms.stage_index}  "
                f"lift {100 * step.terms.lift_height:.1f} cm  grasp {int(step.terms.is_grasp)}"
            )
            color = (50, 220, 50) if step.terms.success else (255, 255, 255)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
            writer.write(bgr)
            if step.done:
                break
        metadata = {
            "episode": episode,
            "rollout_episode": rollout_episode,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "closed_loop": actor is not None,
            "expert": expert,
            "context_start": context_start,
            "robot_position_xy": robot_position_xy,
            "success": bool(success),
            "max_lift": float(max_lift),
            "frames": index + 1,
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        completed = True
    finally:
        if writer is not None:
            writer.release()
        env.close()
    if completed:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(staging),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(output),
            ],
            check=True,
        )
        staging.unlink()
    return output
