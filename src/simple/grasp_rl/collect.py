"""Fast successful-rollout collection for downstream simulation datasets."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.policy import add_optional_phase, load_actor
from simple.grasp_rl.reference import ReferenceLibrary, ReferenceTracker
from simple.grasp_rl.rewards import DEFAULT_TASK_REWARD_PROFILE
from simple.grasp_rl.schema import MAX_EPISODE_STEPS, REFERENCE_ACTOR_OBS_DIM
from simple.grasp_rl.tracker import ActionTransform


@torch.no_grad()
def collect_policy_dataset(
    checkpoint: str | Path,
    action_transform_path: str | Path,
    dataset_dir: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path,
    num_successes: int,
    *,
    max_attempts: int | None = None,
    scene_hold_attempts: int = 16,
    target_position_jitter_xy: tuple[float, float] | None = None,
    target_yaw_jitter: float = 0.15,
    task_reward_profile: str = DEFAULT_TASK_REWARD_PROFILE,
    reference_source: str = "bc",
    reference_splits: tuple[str, ...] = ("train", "val", "test"),
    reference_ranks: tuple[int, ...] = (0, 1),
    seed: int = 20260729,
    device: str = "cuda:0",
) -> dict:
    """Collect exactly ``num_successes`` random-target successful rollouts.

    Failed attempts are represented in the manifest and summary but are not
    silently mixed into the trajectory directory.  Each saved NPZ contains
    enough state/action tensors for policy training and deterministic MuJoCo
    state reconstruction against the same SIMPLE model.
    """
    if num_successes < 1:
        raise ValueError("num_successes must be positive")
    if scene_hold_attempts < 1:
        raise ValueError("scene_hold_attempts must be positive")
    if max_attempts is None:
        max_attempts = 3 * num_successes
    if max_attempts < num_successes:
        raise ValueError("max_attempts must be at least num_successes")
    if not reference_ranks or any(rank < 0 for rank in reference_ranks):
        raise ValueError("reference_ranks must contain non-negative values")
    if target_position_jitter_xy is not None and (
        len(target_position_jitter_xy) != 2
        or any(value < 0.0 for value in target_position_jitter_xy)
    ):
        raise ValueError(
            "target_position_jitter_xy must contain two non-negative values"
        )

    output = Path(output_dir)
    trajectory_dir = output / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (
            Path(dataset_dir) / "meta" / "episodes.jsonl"
        ).read_text().splitlines()
    ]
    rng = np.random.default_rng(seed)
    scene_order = rng.permutation(len(rows))
    transform = ActionTransform.from_npz(action_transform_path)
    actor = load_actor(checkpoint, device)
    observation_dim = int(getattr(actor, "grasp_observation_dim", 192))
    reference = (
        ReferenceTracker(
            ReferenceLibrary(
                processed_dir,
                source=reference_source,
                splits=reference_splits,
            )
        )
        if observation_dim == REFERENCE_ACTOR_OBS_DIM
        else None
    )
    env = GraspRlEnv(
        transform,
        seed=seed,
        task_reward_profile=task_reward_profile,
    )
    manifest: list[dict] = []
    successes = 0
    start_time = time.monotonic()
    current_scene_slot = -1

    def run_rollout(
        observation: np.ndarray,
        initial_frame: np.ndarray,
        reference_rank: int,
    ) -> dict:
        if hasattr(actor, "reset"):
            actor.reset()
        if reference is not None:
            reference.reset(observation, rank=reference_rank)
            reference_episode = reference.episode
        else:
            reference_episode = None
        arrays: dict[str, list] = {
            "observations": [],
            "policy_observations": [],
            "post_observations": [],
            "raw_actions": [],
            "actions": [],
            "motion_frames": [initial_frame],
            "target_rewards": [],
            "penalties": [],
        }
        final_terms = None
        for policy_step in range(MAX_EPISODE_STEPS):
            policy_observation = (
                reference.augment(observation)
                if reference is not None
                else observation
            )
            actor_observation = add_optional_phase(
                actor,
                torch.as_tensor(policy_observation[None], device=device),
                policy_step,
            )
            raw_action = actor(
                TensorDict(
                    {"actor": actor_observation},
                    batch_size=[1],
                    device=device,
                ),
                stochastic_output=False,
            )[0].cpu().numpy()
            if reference is not None:
                assert env.reward is not None
                env.reward.set_reference_contact(reference.post_step_contact_label())
            step = env.step_raw(raw_action)
            if reference is not None:
                reference.reward(
                    step.actor_observation,
                    transform.encode(env.previous_physical_action),
                )
            arrays["observations"].append(observation)
            arrays["policy_observations"].append(policy_observation)
            arrays["post_observations"].append(step.actor_observation)
            arrays["raw_actions"].append(raw_action)
            arrays["actions"].append(env.previous_physical_action.copy())
            arrays["motion_frames"].append(step.motion_frame)
            arrays["target_rewards"].append(step.terms.target_reward)
            arrays["penalties"].append(step.terms.penalty)
            observation = step.actor_observation
            final_terms = step.terms
            if step.done:
                break
        assert final_terms is not None
        return {
            "arrays": arrays,
            "terms": final_terms,
            "reference_rank": reference_rank,
            "reference_episode": reference_episode,
        }

    try:
        for attempt in range(max_attempts):
            scene_slot = attempt // scene_hold_attempts
            if scene_slot != current_scene_slot:
                current_scene_slot = scene_slot
                row = rows[int(scene_order[scene_slot % len(scene_order)])]
                base_episode = int(row["episode_index"])
                env.reset(state_dict=json.loads(row["environment_config"]))
                assert env.state is not None
                reference_target_position = env.state.initial_object_pos.copy()
                env.capture_fast_reset_snapshot(
                    randomize_target=True,
                    position_jitter_xy=target_position_jitter_xy,
                    yaw_jitter=target_yaw_jitter,
                )
            observation, initial_frame = env.reset()
            assert env.state is not None
            target_position = env.state.initial_object_pos.copy()
            _, target_quaternion = env.target_freejoint_pose()
            initial_qpos = env.sim.mjData.qpos.copy()
            initial_qvel = env.sim.mjData.qvel.copy()
            rollout_attempts = []
            rollout = None
            for rank_index, reference_rank in enumerate(reference_ranks):
                if rank_index:
                    observation, initial_frame = env.reset_to_target_pose(
                        target_position, target_quaternion
                    )
                rollout = run_rollout(
                    observation,
                    initial_frame,
                    reference_rank,
                )
                terms = rollout["terms"]
                rollout_attempts.append(
                    {
                        "reference_rank": reference_rank,
                        "reference_episode": rollout["reference_episode"],
                        "success": bool(terms.success),
                        "failure": bool(terms.failure),
                        "timeout": bool(terms.timeout),
                        "length": len(rollout["arrays"]["raw_actions"]),
                    }
                )
                if terms.success:
                    break
            assert rollout is not None
            final_terms = rollout["terms"]
            arrays = rollout["arrays"]
            reference_episode = rollout["reference_episode"]
            success = bool(final_terms.success)
            record = {
                "attempt": attempt,
                "base_episode": base_episode,
                "reference_episode": reference_episode,
                "success": success,
                "failure": bool(final_terms.failure),
                "timeout": bool(final_terms.timeout),
                "length": len(arrays["raw_actions"]),
                "reference_rank": rollout["reference_rank"],
                "plan_attempts": rollout_attempts,
                "target_position": target_position.tolist(),
                "reference_target_position": reference_target_position.tolist(),
                "target_offset": (
                    target_position - reference_target_position
                ).tolist(),
            }
            if success:
                filename = f"episode_{successes:06d}.npz"
                np.savez_compressed(
                    trajectory_dir / filename,
                    observations=np.asarray(
                        arrays["observations"], dtype=np.float32
                    ),
                    policy_observations=np.asarray(
                        arrays["policy_observations"], dtype=np.float32
                    ),
                    post_observations=np.asarray(
                        arrays["post_observations"], dtype=np.float32
                    ),
                    raw_actions=np.asarray(
                        arrays["raw_actions"], dtype=np.float32
                    ),
                    actions=np.asarray(arrays["actions"], dtype=np.float32),
                    motion_frames=np.asarray(
                        arrays["motion_frames"], dtype=np.float32
                    ),
                    target_rewards=np.asarray(
                        arrays["target_rewards"], dtype=np.float32
                    ),
                    penalties=np.asarray(arrays["penalties"], dtype=np.float32),
                    initial_qpos=initial_qpos,
                    initial_qvel=initial_qvel,
                    target_position=target_position.astype(np.float32),
                    target_quaternion=target_quaternion.astype(np.float32),
                    reference_target_position=reference_target_position.astype(
                        np.float32
                    ),
                    base_episode=np.asarray(base_episode, dtype=np.int64),
                    reference_episode=np.asarray(
                        -1 if reference_episode is None else reference_episode,
                        dtype=np.int64,
                    ),
                    reference_rank=np.asarray(
                        rollout["reference_rank"], dtype=np.int64
                    ),
                )
                record["trajectory"] = str(Path("trajectories") / filename)
                successes += 1
            manifest.append(record)
            if successes >= num_successes:
                break
    finally:
        env.close()

    elapsed = time.monotonic() - start_time
    attempts = len(manifest)
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "dataset": str(Path(dataset_dir).resolve()),
        "processed": str(Path(processed_dir).resolve()),
        "requested_successes": num_successes,
        "successes": successes,
        "attempts": attempts,
        "attempt_success_rate": successes / max(attempts, 1),
        "complete": successes == num_successes,
        "elapsed_seconds": elapsed,
        "successful_trajectories_per_second": successes / max(elapsed, 1e-6),
        "scene_hold_attempts": scene_hold_attempts,
        "reference_ranks": list(reference_ranks),
        "plan_rollouts": sum(
            len(row["plan_attempts"]) for row in manifest
        ),
        "target_position_jitter_xy": (
            None
            if target_position_jitter_xy is None
            else list(target_position_jitter_xy)
        ),
        "target_yaw_jitter": target_yaw_jitter,
        "seed": seed,
    }
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise RuntimeError(
            f"Collected only {successes}/{num_successes} successes in "
            f"{attempts} attempts; partial data is preserved at {output}"
        )
    return summary
