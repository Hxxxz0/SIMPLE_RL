"""Deterministic policy evaluation on saved SIMPLE environment configurations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tensordict import TensorDict

from simple.grasp_rl.diffusion import Guidance
from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.motion import BatchedMotionBuffer
from simple.grasp_rl.policy import add_optional_phase, load_actor
from simple.grasp_rl.reference import ReferenceLibrary, ReferenceTracker
from simple.grasp_rl.rewards import DEFAULT_TASK_REWARD_PROFILE, compose_reward
from simple.grasp_rl.schema import REFERENCE_ACTOR_OBS_DIM
from simple.grasp_rl.task_spec import GraspTaskSpec, get_task_spec
from simple.grasp_rl.tracker import ActionTransform


@torch.no_grad()
def evaluate_policy(
    checkpoint: str | Path,
    action_transform_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    diffusion_checkpoint: str | Path | None = None,
    num_episodes: int = 100,
    device: str = "cuda:0",
    reward_variant: str = "task_only",
    initialization_prefix: int | None = None,
    initialization_phase: float | None = None,
    episode_offset: int = 0,
    task_reward_weight: float = 0.02,
    smp_reward_weight: float = 0.01,
    task_reward_profile: str = DEFAULT_TASK_REWARD_PROFILE,
    reference_processed: str | Path | None = None,
    reference_source: str = "bc",
    reference_splits: tuple[str, ...] = ("train", "val", "test"),
    reference_reward_weight: float = 0.0,
    reference_action_override: str = "none",
    reference_rank: int = 0,
    reference_base_episode: bool = False,
    reach_extension_threshold: float | None = None,
    reach_extension_velocity: float = 0.0,
    randomize_target: bool = False,
    target_position_jitter_xy: tuple[float, float] | None = (0.025, 0.03),
    target_yaw_jitter: float = 0.15,
    task: str | GraspTaskSpec | None = None,
) -> dict:
    task_spec = get_task_spec(task)
    output = Path(output_dir)
    trajectory_dir = output / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [
        json.loads(line)
        for line in (Path(dataset_dir) / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    if episode_offset < 0:
        raise ValueError("episode_offset must be non-negative")
    if initialization_phase is not None and not 0.0 <= initialization_phase <= 1.0:
        raise ValueError("initialization_phase must be in [0, 1]")
    if initialization_prefix is not None and initialization_phase is not None:
        raise ValueError("initialization_prefix and initialization_phase are exclusive")
    if target_position_jitter_xy is not None and (
        len(target_position_jitter_xy) != 2
        or any(value < 0.0 for value in target_position_jitter_xy)
    ):
        raise ValueError(
            "target_position_jitter_xy must contain two non-negative values"
        )
    if target_yaw_jitter < 0.0:
        raise ValueError("target_yaw_jitter must be non-negative")
    if reference_rank < 0:
        raise ValueError("reference_rank must be non-negative")
    if reach_extension_threshold is not None and not 0.0 < reach_extension_threshold < 1.0:
        raise ValueError("reach_extension_threshold must be in (0, 1)")
    if not 0.0 <= reach_extension_velocity <= 1.0:
        raise ValueError("reach_extension_velocity must be in [0, 1]")
    if reference_action_override not in {
        "none",
        "right_hand",
        "right_arm_hand",
        "all",
    }:
        raise ValueError(
            f"Unknown reference_action_override {reference_action_override}"
        )
    rows = all_rows[episode_offset : episode_offset + num_episodes]
    transform = ActionTransform.from_npz(action_transform_path)
    actor = load_actor(
        checkpoint,
        device,
        expected_task=task_spec,
        action_transform=action_transform_path,
    )
    actor_observation_dim = int(getattr(actor, "grasp_observation_dim", 192))
    needs_reference = (
        actor_observation_dim == REFERENCE_ACTOR_OBS_DIM
        or reference_reward_weight > 0.0
    )
    if needs_reference and reference_processed is None:
        raise ValueError(
            "reference_processed is required for a reference-conditioned actor/reward"
        )
    reference = (
        ReferenceTracker(
            ReferenceLibrary(
                reference_processed,
                source=reference_source,
                splits=reference_splits,
            )
        )
        if needs_reference
        else None
    )
    guidance = Guidance(diffusion_checkpoint, device) if diffusion_checkpoint else None
    env = GraspRlEnv(
        transform,
        seed=1234,
        task_reward_profile=task_reward_profile,
        task=task_spec,
    )
    results = []
    try:
        for row in rows:
            if hasattr(actor, "reset"):
                actor.reset()
            episode = int(row["episode_index"])
            observation, frame = env.reset(state_dict=json.loads(row["environment_config"]))
            assert env.state is not None
            reference_target_position = env.state.initial_object_pos.copy()
            if randomize_target:
                env.capture_fast_reset_snapshot(
                    randomize_target=True,
                    position_jitter_xy=target_position_jitter_xy,
                    yaw_jitter=target_yaw_jitter,
                )
                observation, frame = env.reset()
                assert env.state is not None
            target_position = env.state.initial_object_pos.copy()
            _, target_quaternion = env.target_freejoint_pose()
            target_offset = target_position - reference_target_position
            if initialization_prefix is not None or initialization_phase is not None:
                # Mirror the training worker exactly: cache the configured scene,
                # restore it through the fast path, then replay the prefix.
                env.capture_fast_reset_snapshot(randomize_target=False)
                observation, frame = env.reset()
                parquet = (
                    Path(dataset_dir)
                    / "data"
                    / "chunk-000"
                    / f"episode_{episode:06d}.parquet"
                )
                demo_actions = np.asarray(
                    pq.read_table(parquet, columns=["action"])["action"].to_pylist(),
                    dtype=np.float32,
                )
                if initialization_phase is not None:
                    stop = int(round(initialization_phase * (len(demo_actions) - 1)))
                else:
                    assert initialization_prefix is not None
                    stop = min(initialization_prefix, len(demo_actions) - 1)
                for physical_action in demo_actions[: stop + 1]:
                    initialized = env.step_physical(physical_action)
                observation = initialized.actor_observation
                frame = initialized.motion_frame
                assert env.reward is not None
                env.reward.reset()
            # Preserve the exact randomized physical initial state so a saved
            # evaluation can be rerun closed-loop and rendered, rather than
            # silently falling back to the dataset's original object pose.
            initial_qpos = env.sim.mjData.qpos.copy()
            initial_qvel = env.sim.mjData.qvel.copy()
            policy_step = (
                0
                if initialization_prefix is None and initialization_phase is None
                else stop + 1
            )
            if reference is not None:
                reference.reset(
                    observation,
                    exact_episode=(
                        episode
                        if reference_base_episode or not randomize_target
                        else None
                    ),
                    start_index=policy_step,
                    rank=(
                        reference_rank
                        if randomize_target and not reference_base_episode
                        else 0
                    ),
                )
                selected_reference_episode = reference.episode
            else:
                selected_reference_episode = None
            buffer = BatchedMotionBuffer(1, device)
            buffer.reset(torch.tensor([0], device=device), torch.as_tensor(frame[None], device=device))
            actions, rewards, raw_errors, reference_rewards = [], [], [], []
            max_lift = -float("inf")
            max_pregrasp = 0.0
            max_grasp_quality = 0.0
            grasp_steps = 0
            used_reach_extension = False
            final_terms = None
            native_success_any = False
            for _ in range(task_spec.max_episode_steps):
                policy_observation = (
                    reference.augment(observation)
                    if reference is not None
                    else observation
                )
                obs_tensor = add_optional_phase(
                    actor,
                    torch.as_tensor(policy_observation[None], device=device),
                    policy_step,
                )
                raw = actor(
                    TensorDict({"actor": obs_tensor}, batch_size=[1], device=device),
                    stochastic_output=False,
                )[0].cpu().numpy()
                # The demonstrations never command locomotion (raw index 32
                # is identically zero), so arm-only PPO cannot discover how to
                # handle targets beyond the demonstrated reach envelope.  This
                # state-gated complete-command adapter uses SIMPLE's existing
                # tracker velocity input until the target re-enters that
                # envelope, then hands control back to the learned policy.
                reach_extension_active = bool(
                    reach_extension_threshold is not None
                    and observation[132] > reach_extension_threshold
                )
                if reach_extension_active:
                    raw[32] = reach_extension_velocity
                    used_reach_extension = True
                if reference_action_override != "none":
                    if reference is None:
                        raise ValueError(
                            "reference action override requires reference context"
                        )
                    reference_raw = policy_observation[192:228]
                    if reference_action_override == "all":
                        raw = reference_raw.copy()
                    else:
                        raw[7:14] = reference_raw[7:14]
                        if reference_action_override == "right_arm_hand":
                            raw[21:28] = reference_raw[21:28]
                policy_step += 1
                if reference is not None:
                    assert env.reward is not None
                    env.reward.set_reference_contact(
                        reference.post_step_contact_label()
                    )
                step = env.step_raw(raw)
                buffer.update(torch.as_tensor(step.motion_frame[None], device=device))
                if guidance is not None and bool(buffer.is_full[0]):
                    smp, error = guidance.score(buffer.features())
                    smp_value, error_value = float(smp[0]), float(error[0])
                else:
                    smp_value, error_value = 1.0, 0.0
                reward, _, _ = compose_reward(
                    step.terms.target_reward,
                    step.terms.penalty,
                    step.terms.terminal_adjustment,
                    smp_value,
                    bool(buffer.is_full[0]),
                    reward_variant,
                    task_reward_weight,
                    smp_reward_weight,
                )
                if reference is not None:
                    reference_terms = reference.reward(
                        step.actor_observation,
                        transform.encode(env.previous_physical_action),
                    )
                    reference_value = reference_terms.total
                else:
                    reference_value = 0.0
                reward = reward + reference_reward_weight * reference_value
                actions.append(env.previous_physical_action.copy())
                rewards.append(reward)
                raw_errors.append(error_value)
                reference_rewards.append(reference_value)
                max_lift = max(max_lift, step.terms.lift_height)
                max_pregrasp = max(max_pregrasp, step.terms.pregrasp)
                max_grasp_quality = max(max_grasp_quality, step.terms.grasp_quality)
                grasp_steps += int(step.terms.is_grasp)
                observation = step.actor_observation
                final_terms = step.terms
                native_success_any |= step.terms.native_success
                if step.done:
                    break
            assert final_terms is not None
            np.savez_compressed(
                trajectory_dir / f"episode_{episode:06d}.npz",
                actions=np.stack(actions),
                rewards=np.asarray(rewards, dtype=np.float32),
                raw_smp_mse=np.asarray(raw_errors, dtype=np.float32),
                raw_reference_reward=np.asarray(
                    reference_rewards, dtype=np.float32
                ),
                initial_qpos=initial_qpos,
                initial_qvel=initial_qvel,
                target_position=target_position.astype(np.float32),
                target_quaternion=target_quaternion.astype(np.float32),
                reference_target_position=reference_target_position.astype(np.float32),
                base_episode=np.asarray(episode, dtype=np.int64),
                reference_episode=np.asarray(
                    -1
                    if selected_reference_episode is None
                    else selected_reference_episode,
                    dtype=np.int64,
                ),
                reference_rank=np.asarray(reference_rank, dtype=np.int64),
            )
            results.append(
                {
                    "episode": episode,
                    "reference_episode": selected_reference_episode,
                    "used_reach_extension": used_reach_extension,
                    "success": bool(final_terms.success),
                    "native_success": bool(native_success_any),
                    "failure": bool(final_terms.failure),
                    "timeout": bool(final_terms.timeout),
                    "length": len(actions),
                    "return": float(np.sum(rewards)),
                    "max_lift": float(max_lift),
                    "max_pregrasp": float(max_pregrasp),
                    "max_grasp_quality": float(max_grasp_quality),
                    "grasp_steps": grasp_steps,
                    "target_position": target_position.tolist(),
                    "target_quaternion": target_quaternion.tolist(),
                    "reference_target_position": reference_target_position.tolist(),
                    "target_offset": target_offset.tolist(),
                    "mean_raw_smp_mse": float(np.mean(raw_errors[9:])) if len(raw_errors) > 9 else 0.0,
                    "mean_raw_reference_reward": float(
                        np.mean(reference_rewards)
                    ),
                }
            )
    finally:
        env.close()
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "task": task_spec.name,
        "task_metadata": task_spec.metadata(),
        "initialization_prefix": initialization_prefix,
        "initialization_phase": initialization_phase,
        "episode_offset": episode_offset,
        "task_reward_profile": task_reward_profile,
        "reference_processed": (
            str(Path(reference_processed).resolve())
            if reference_processed is not None
            else None
        ),
        "reference_reward_weight": reference_reward_weight,
        "reference_action_override": reference_action_override,
        "reference_rank": reference_rank,
        "reference_base_episode": reference_base_episode,
        "reach_extension_threshold": reach_extension_threshold,
        "reach_extension_velocity": reach_extension_velocity,
        "randomize_target": randomize_target,
        "target_position_jitter_xy": (
            None
            if target_position_jitter_xy is None
            else list(target_position_jitter_xy)
        ),
        "target_yaw_jitter": target_yaw_jitter,
        "episodes": len(results),
        "success_rate": float(np.mean([row["success"] for row in results])),
        "native_success_rate": float(
            np.mean([row["native_success"] for row in results])
        ),
        "failure_rate": float(np.mean([row["failure"] for row in results])),
        "timeout_rate": float(np.mean([row["timeout"] for row in results])),
        "contact_rate": float(
            np.mean([row["max_grasp_quality"] > 0.0 for row in results])
        ),
        "goal_lift_rate": float(
            np.mean(
                [
                    row["max_lift"] >= task_spec.reward.success_lift
                    for row in results
                ]
            )
        ),
        "mean_return": float(np.mean([row["return"] for row in results])),
        "mean_max_lift": float(np.mean([row["max_lift"] for row in results])),
        "mean_max_pregrasp": float(np.mean([row["max_pregrasp"] for row in results])),
        "mean_max_grasp_quality": float(
            np.mean([row["max_grasp_quality"] for row in results])
        ),
        "mean_grasp_steps": float(np.mean([row["grasp_steps"] for row in results])),
        "mean_raw_smp_mse": float(np.mean([row["mean_raw_smp_mse"] for row in results])),
        "mean_raw_reference_reward": float(
            np.mean([row["mean_raw_reference_reward"] for row in results])
        ),
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
