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
from simple.grasp_rl.schema import (
    ACTION_DIM,
    REFERENCE_ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_V2_DIM,
)
from simple.grasp_rl.task_spec import TaskSpec, TaskSpecV2, get_task_spec
from simple.grasp_rl.tracker import ActionTransform


def reference_action_from_observation(
    policy_observation: np.ndarray,
    base_observation_dim: int,
) -> np.ndarray:
    """Extract the current complete command from either reference schema."""

    start = int(base_observation_dim)
    stop = start + ACTION_DIM
    if policy_observation.ndim != 1 or start < 0 or stop > len(policy_observation):
        raise ValueError("Reference action lies outside the policy observation")
    return policy_observation[start:stop]


def _filter_evaluation_split(
    rows: list[dict],
    evaluation_split: str,
    reference_processed: str | Path | None,
) -> list[dict]:
    """Restrict evaluation to replay-gated IDs from a processed manifest."""

    if evaluation_split not in {"all", "train", "val", "test"}:
        raise ValueError(f"Unknown evaluation_split {evaluation_split!r}")
    if evaluation_split == "all":
        return rows
    if reference_processed is None:
        raise ValueError("evaluation_split requires reference_processed")
    manifest = json.loads(
        (Path(reference_processed) / "manifest.json").read_text()
    )
    allowed = {int(value) for value in manifest["splits"][evaluation_split]}
    return [row for row in rows if int(row["episode_index"]) in allowed]


def validate_final_protocol(
    *,
    num_episodes: int,
    initialization_prefix: int | None,
    initialization_phase: float | None,
    evaluation_split: str,
    randomize_target: bool,
    target_position_jitter_xy: tuple[float, float] | None,
    target_position_offset_center_xy: tuple[float, float],
    target_yaw_jitter: float,
    reference_rank: int,
    reference_splits: tuple[str, ...],
    fixed_base_episode: int | None,
) -> None:
    expected = {
        "num_episodes": num_episodes == 200,
        "full_start": initialization_prefix is None and initialization_phase is None,
        "test_split": evaluation_split == "test",
        "randomize_target": randomize_target,
        "standard_xy_jitter": target_position_jitter_xy == (0.025, 0.03),
        "zero_offset_center": target_position_offset_center_xy == (0.0, 0.0),
        "standard_yaw_jitter": np.isclose(target_yaw_jitter, 0.15),
        "reference_rank_zero": reference_rank == 0,
        "training_reference_library_only": reference_splits == ("train", "val"),
        "fixed_base_episode": fixed_base_episode is not None,
    }
    failed = [name for name, passed in expected.items() if not passed]
    if failed:
        raise ValueError("final protocol violation: " + ", ".join(failed))


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
    evaluation_split: str = "all",
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
    fixed_reference_episode: int | None = None,
    fixed_base_episode: int | None = None,
    reach_extension_threshold: float | None = None,
    reach_extension_velocity: float = 0.0,
    randomize_target: bool = False,
    target_position_jitter_xy: tuple[float, float] | None = (0.025, 0.03),
    target_position_offset_center_xy: tuple[float, float] = (0.0, 0.0),
    target_yaw_jitter: float = 0.15,
    target_position_xy: tuple[float, float] | None = None,
    robot_position_xy: tuple[float, float] | None = None,
    seed: int = 1234,
    final_protocol: bool = False,
    task: str | TaskSpec | None = None,
) -> dict:
    task_spec = get_task_spec(task)
    is_v2 = isinstance(task_spec, TaskSpecV2)
    output = Path(output_dir)
    if final_protocol:
        validate_final_protocol(
            num_episodes=num_episodes,
            initialization_prefix=initialization_prefix,
            initialization_phase=initialization_phase,
            evaluation_split=evaluation_split,
            randomize_target=randomize_target,
            target_position_jitter_xy=target_position_jitter_xy,
            target_position_offset_center_xy=target_position_offset_center_xy,
            target_yaw_jitter=target_yaw_jitter,
            reference_rank=reference_rank,
            reference_splits=reference_splits,
            fixed_base_episode=fixed_base_episode,
        )
    trajectory_dir = output / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [
        json.loads(line)
        for line in (Path(dataset_dir) / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    all_rows = _filter_evaluation_split(
        all_rows, evaluation_split, reference_processed
    )
    valid_source_uids = (
        set(task_spec.source_uids)
        if isinstance(task_spec, TaskSpecV2)
        else {task_spec.registry_uid}
    )
    all_rows = [
        row
        for row in all_rows
        if json.loads(row["environment_config"]).get("uid") in valid_source_uids
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
    if len(target_position_offset_center_xy) != 2:
        raise ValueError(
            "target_position_offset_center_xy must contain two values"
        )
    if target_position_xy is not None:
        if len(target_position_xy) != 2:
            raise ValueError("target_position_xy must contain two values")
        if not is_v2:
            raise ValueError("target_position_xy is only supported for v2 tasks")
        if randomize_target:
            raise ValueError(
                "target_position_xy and randomize_target are mutually exclusive"
            )
    if robot_position_xy is not None:
        if len(robot_position_xy) != 2:
            raise ValueError("robot_position_xy must contain two values")
        if not is_v2:
            raise ValueError("robot_position_xy is only supported for v2 tasks")
    if reference_rank < 0:
        raise ValueError("reference_rank must be non-negative")
    if fixed_reference_episode is not None and fixed_reference_episode < 0:
        raise ValueError("fixed_reference_episode must be non-negative")
    if fixed_base_episode is not None and fixed_base_episode < 0:
        raise ValueError("fixed_base_episode must be non-negative")
    if fixed_reference_episode is not None and reference_base_episode:
        raise ValueError(
            "fixed_reference_episode and reference_base_episode are mutually exclusive"
        )
    if fixed_base_episode is not None and episode_offset != 0:
        raise ValueError("fixed_base_episode requires episode_offset=0")
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
    if fixed_base_episode is not None:
        matching_rows = [
            row
            for row in all_rows
            if int(row["episode_index"]) == fixed_base_episode
        ]
        if not matching_rows:
            raise ValueError(
                f"Fixed base episode {fixed_base_episode} is not available in "
                f"evaluation split {evaluation_split!r}"
            )
        # Repeat one exact robot/scene initial state while the environment RNG
        # generates a new object-position/yaw perturbation on every reset.
        rows = [matching_rows[0]] * num_episodes
    else:
        rows = all_rows[episode_offset : episode_offset + num_episodes]
    if not rows:
        raise ValueError("Evaluation selected no compatible dataset episodes")
    target_relocated = (
        randomize_target
        or target_position_xy is not None
        or robot_position_xy is not None
    )
    transform = ActionTransform.from_npz(action_transform_path)
    actor = load_actor(
        checkpoint,
        device,
        expected_task=task_spec,
        action_transform=action_transform_path,
    )
    actor_observation_dim = int(getattr(actor, "grasp_observation_dim", 192))
    needs_reference = (
        actor_observation_dim in (REFERENCE_ACTOR_OBS_DIM, REFERENCE_ACTOR_OBS_V2_DIM)
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
    if fixed_reference_episode is not None and reference is None:
        raise ValueError(
            "fixed_reference_episode requires a reference-conditioned actor/reward"
        )
    guidance = Guidance(diffusion_checkpoint, device) if diffusion_checkpoint else None
    env = GraspRlEnv(
        transform,
        seed=seed,
        task_reward_profile=task_reward_profile,
        task=task_spec,
    )
    results = []
    try:
        for rollout_index, row in enumerate(rows):
            if hasattr(actor, "reset"):
                actor.reset()
            episode = int(row["episode_index"])
            state_dict = json.loads(row["environment_config"])
            if robot_position_xy is not None:
                robot_uid = state_dict["robot_cfg"]["uid"]
                robot_pose = state_dict["dr_state_dict"]["spatial"][robot_uid]
                robot_pose["position"][:2] = robot_position_xy
            observation, frame = env.reset(state_dict=state_dict)
            assert env.state is not None
            reference_target_position = (
                env.state.initial_primary_pos.copy()
                if is_v2 else env.state.initial_object_pos.copy()
            )
            if target_position_xy is not None:
                position, quaternion = env.primary_freejoint_pose()
                position[:2] = target_position_xy
                observation, frame = env.set_primary_pose(position, quaternion)
                assert env.state is not None
            elif randomize_target:
                if is_v2:
                    env.randomize_primary_pose(
                        target_position_jitter_xy or (0.025, 0.03),
                        target_yaw_jitter,
                        target_position_offset_center_xy,
                    )
                    observation, _ = env.state.actor_observation()
                    frame = env.motion.extract()
                else:
                    env.capture_fast_reset_snapshot(
                        randomize_target=True,
                        position_jitter_xy=target_position_jitter_xy,
                        yaw_jitter=target_yaw_jitter,
                    )
                    observation, frame = env.reset()
                assert env.state is not None
            target_position = (
                env.state.initial_primary_pos.copy()
                if is_v2 else env.state.initial_object_pos.copy()
            )
            target_quaternion = (
                env.primary_freejoint_pose()[1]
                if is_v2
                else env.target_freejoint_pose()[1]
            )
            target_offset = target_position - reference_target_position
            if initialization_prefix is not None or initialization_phase is not None:
                # Mirror the training worker exactly: cache the configured scene,
                # restore it through the fast path, then replay the prefix.
                if not is_v2:
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
                        fixed_reference_episode
                        if fixed_reference_episode is not None
                        else episode
                        if reference_base_episode or not target_relocated
                        else None
                    ),
                    start_index=policy_step,
                    rank=(
                        reference_rank
                        if target_relocated and not reference_base_episode
                        else 0
                    ),
                )
                selected_reference_episode = reference.episode
            else:
                selected_reference_episode = None
            buffer = BatchedMotionBuffer(1, device)
            buffer.reset(torch.tensor([0], device=device), torch.as_tensor(frame[None], device=device))
            actions, rewards, raw_errors, reference_rewards = [], [], [], []
            previous_policy_residual = np.zeros(ACTION_DIM, dtype=np.float32)
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
                    and not is_v2
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
                    reference_raw = reference_action_from_observation(
                        policy_observation, len(observation)
                    )
                    if reference_action_override == "all":
                        raw = reference_raw.copy()
                    else:
                        raw[7:14] = reference_raw[7:14]
                        if reference_action_override == "right_arm_hand":
                            raw[21:28] = reference_raw[21:28]
                policy_step += 1
                if reference is not None:
                    assert env.reward is not None
                    if hasattr(env.reward, "set_reference_contact"):
                        env.reward.set_reference_contact(
                            reference.post_step_contact_label(),
                            reference.post_step_contact_center_primary(),
                        )
                step = env.step_raw(raw)
                buffer.update(torch.as_tensor(step.motion_frame[None], device=device))
                if guidance is not None and bool(buffer.is_full[0]):
                    smp, error = guidance.score(buffer.features())
                    smp_value, error_value = float(smp[0]), float(error[0])
                else:
                    smp_value, error_value = 1.0, 0.0
                task_penalty = step.terms.penalty
                if reference is not None:
                    proposal = policy_observation[
                        len(observation) : len(observation) + ACTION_DIM
                    ]
                    residual = raw - proposal
                    residual_rate = float(
                        np.sum((residual - previous_policy_residual) ** 2)
                    )
                    task_penalty = (
                        task_penalty
                        - 0.1 * step.terms.action_rate_penalty
                        + 0.1 * residual_rate
                    )
                    previous_policy_residual = residual
                reward, _, _ = compose_reward(
                    step.terms.target_reward,
                    task_penalty,
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
                    reference_time_out = reference.is_complete and not (
                        step.terms.success or step.terms.failure
                    )
                    if reference_time_out:
                        step.terms.timeout = True
                else:
                    reference_value = 0.0
                    reference_time_out = False
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
                if step.done or reference_time_out:
                    break
            assert final_terms is not None
            trajectory_name = (
                f"episode_{episode:06d}_repeat_{rollout_index:06d}.npz"
                if fixed_base_episode is not None
                else f"episode_{episode:06d}.npz"
            )
            np.savez_compressed(
                trajectory_dir / trajectory_name,
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
                robot_position_xy=np.asarray(
                    [np.nan, np.nan]
                    if robot_position_xy is None
                    else robot_position_xy,
                    dtype=np.float32,
                ),
            )
            results.append(
                {
                    "rollout_index": rollout_index,
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
        "seed": seed,
        "final_test": final_protocol,
        "initialization_prefix": initialization_prefix,
        "initialization_phase": initialization_phase,
        "episode_offset": episode_offset,
        "evaluation_split": evaluation_split,
        "task_reward_profile": task_reward_profile,
        "reference_processed": (
            str(Path(reference_processed).resolve())
            if reference_processed is not None
            else None
        ),
        "reference_splits": list(reference_splits),
        "reference_reward_weight": reference_reward_weight,
        "reference_action_override": reference_action_override,
        "reference_rank": reference_rank,
        "reference_base_episode": reference_base_episode,
        "fixed_reference_episode": fixed_reference_episode,
        "fixed_base_episode": fixed_base_episode,
        "reach_extension_threshold": reach_extension_threshold,
        "reach_extension_velocity": reach_extension_velocity,
        "randomize_target": randomize_target,
        "target_position_jitter_xy": (
            None
            if target_position_jitter_xy is None
            else list(target_position_jitter_xy)
        ),
        "target_position_offset_center_xy": list(
            target_position_offset_center_xy
        ),
        "target_yaw_jitter": target_yaw_jitter,
        "target_position_xy": (
            None if target_position_xy is None else list(target_position_xy)
        ),
        "robot_position_xy": (
            None if robot_position_xy is None else list(robot_position_xy)
        ),
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
                    row["max_lift"] >= (
                        task_spec.lift_height
                        if isinstance(task_spec, TaskSpecV2)
                        else task_spec.reward.success_lift
                    )
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
