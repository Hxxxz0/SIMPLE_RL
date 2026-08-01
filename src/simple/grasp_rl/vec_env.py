"""Synchronous multi-process SIMPLE workers with batched GPU guidance."""

from __future__ import annotations

import multiprocessing as mp
import json
import os
import traceback
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from simple.grasp_rl.diffusion import Guidance
from simple.grasp_rl.collect import _filter_replay_gated_rows
from simple.grasp_rl.motion import BatchedMotionBuffer
from simple.grasp_rl.rewards import (
    DEFAULT_TASK_REWARD_PROFILE,
    REWARD_VARIANTS,
    TASK_REWARD_PROFILES,
    compose_reward,
)
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    ACTOR_OBS_V2_DIM,
    REFERENCE_FUTURE_OFFSETS,
)
from simple.grasp_rl.task_spec import DEFAULT_TASK, TaskSpecV2, get_task_spec
from simple.grasp_rl.tracker import ActionTransform


def apply_reference_action_bias(
    observation: torch.Tensor,
    bias: torch.Tensor,
    base_observation_dim: int,
    reference_frame_dim: int,
) -> torch.Tensor:
    """Add one coherent command bias to every future reference frame."""

    if observation.ndim != 2 or bias.shape != (observation.shape[0], ACTION_DIM):
        raise ValueError("Expected observations [N,D] and reference bias [N,36]")
    result = observation.clone()
    for frame_index in range(len(REFERENCE_FUTURE_OFFSETS)):
        start = base_observation_dim + frame_index * reference_frame_dim
        stop = start + ACTION_DIM
        if stop > result.shape[1]:
            raise ValueError("Reference frame extends beyond observation")
        result[:, start:stop] = torch.clamp(
            result[:, start:stop] + bias,
            -1.0,
            1.0,
        )
    return result


def reference_residual_action_rate(
    actions: torch.Tensor,
    proposal: torch.Tensor,
    previous_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GRAIL-style residual rate and the current policy residual."""

    if actions.shape != proposal.shape or actions.shape != previous_residual.shape:
        raise ValueError("Actions, proposals and residuals must have equal shape")
    residual = actions - proposal
    rate = (residual - previous_residual).square().sum(dim=-1)
    return rate, residual


def sample_target_randomization(
    rng: np.random.Generator,
    target_mix: dict[str, float],
    hard_target_offsets_xy: list[list[float]] | None,
    uniform_jitter_xy: tuple[float, float],
    uniform_yaw_jitter: float,
) -> tuple[str, tuple[float, float], tuple[float, float], float]:
    """Sample standard, failure-neighbourhood, or near-native target DR."""

    modes = ("uniform", "hard", "native")
    weights = np.asarray([target_mix.get(mode, 0.0) for mode in modes], dtype=float)
    if np.any(weights < 0.0) or not np.isfinite(weights).all() or weights.sum() <= 0.0:
        raise ValueError("target_mix must contain finite non-negative weights")
    weights /= weights.sum()
    mode = modes[int(rng.choice(len(modes), p=weights))]
    if mode == "hard":
        if not hard_target_offsets_xy:
            raise ValueError("hard target sampling requires a non-empty manifest")
        center = hard_target_offsets_xy[int(rng.integers(len(hard_target_offsets_xy)))]
        if len(center) != 2:
            raise ValueError("hard target offsets must have two values")
        jitter = (
            min(float(uniform_jitter_xy[0]), 0.0075),
            min(float(uniform_jitter_xy[1]), 0.0075),
        )
        return mode, jitter, (float(center[0]), float(center[1])), uniform_yaw_jitter
    if mode == "native":
        jitter = (
            min(float(uniform_jitter_xy[0]), 0.005),
            min(float(uniform_jitter_xy[1]), 0.005),
        )
        return mode, jitter, (0.0, 0.0), min(uniform_yaw_jitter, 0.03)
    return mode, uniform_jitter_xy, (0.0, 0.0), uniform_yaw_jitter


def rsi_stage_bounds(
    *,
    trajectory_length: int,
    task_family: str,
    first_grasp: int | None,
    first_lift: int | None,
    stage_entries: dict[int, int],
) -> dict[str, tuple[int, int]]:
    """Convert audited expert events into inclusive RSI start windows."""

    if trajectory_length < 1:
        raise ValueError("trajectory_length must be positive")
    last = trajectory_length - 1
    grasp = int(round(0.67 * last)) if first_grasp is None else first_grasp
    lift = int(round(0.90 * last)) if first_lift is None else first_lift
    if task_family != "place":
        return {
            "pregrasp": (max(grasp - 20, 0), grasp),
            "grasp_to_lift": (grasp, max(grasp, lift - 1)),
            "lift": (max(lift - 3, 0), min(lift + 3, last)),
        }
    boundaries = [
        stage_entries.get(1, int(round(0.20 * last))),
        stage_entries.get(2, int(round(0.45 * last))),
        stage_entries.get(3, int(round(0.65 * last))),
        stage_entries.get(4, int(round(0.85 * last))),
    ]
    boundaries = [max(0, min(int(value), last)) for value in boundaries]
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1])
    approach, grasp_end, lift_end, transport_end = boundaries
    return {
        "approach": (0, approach),
        "grasp": (approach, max(approach, grasp_end - 1)),
        "lift": (grasp_end, max(grasp_end, lift_end - 1)),
        "transport": (lift_end, max(lift_end, transport_end - 1)),
    }


def _worker(
    connection,
    action_transform_path: str,
    seed: int,
    cuda_device: int,
    rsi_dataset: str | None,
    rsi_processed: str | None,
    rsi_prefix: tuple[int, int],
    rsi_phase: tuple[float, float] | None,
    rsi_stage: str | None,
    rsi_episodes: tuple[int, ...] | None,
    rsi_probability: float,
    rsi_scene_hold_episodes: int,
    rsi_randomize_target: bool,
    target_position_jitter_xy: tuple[float, float] | None,
    target_position_offset_center_xy: tuple[float, float],
    target_yaw_jitter: float,
    task_reward_profile: str,
    reference_processed: str | None,
    reference_source: str,
    reference_splits: tuple[str, ...],
    reference_rank_max: int,
    reference_base_episode_probability: float,
    task_name: str,
    training_config: dict | None,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ.setdefault("MUJOCO_GL", "egl")
        torch.set_num_threads(1)
        from simple.grasp_rl.env import GraspRlEnv

        transform = ActionTransform.from_npz(action_transform_path)
        task_spec = get_task_spec(task_name)
        is_v2 = isinstance(task_spec, TaskSpecV2)
        env = GraspRlEnv(
            transform,
            seed=seed,
            task_reward_profile=task_reward_profile,
            task=task_name,
        )
        rsi_rows = None
        rsi_action_cache: dict[int, np.ndarray] = {}
        current_rsi_actions: np.ndarray | None = None
        current_rsi_episode: int | None = None
        current_rsi_state_dict: dict | None = None
        current_rsi_stage_bounds: dict[str, tuple[int, int]] | None = None
        rsi_scene_uses = 0
        rng = np.random.default_rng(seed + 7919)
        runtime = {
            "phase_name": "static",
            "rsi_probability": float(rsi_probability),
            "reference_rank_max": int(reference_rank_max),
            "reference_base_episode_probability": float(
                reference_base_episode_probability
            ),
            "domain_randomization": {
                "target_mass_scale": (1.0, 1.0),
                "friction_scale": (1.0, 1.0),
                "manipulation_action_noise_std": 0.0,
                "action_delay_max_steps": 0,
            },
        }
        if training_config is not None:
            runtime.update(training_config)
        current_dr = {
            "target_mass_scale": 1.0,
            "friction_scale": 1.0,
            "action_delay_steps": 0,
        }
        current_target_mode = "uniform"
        delayed_raw_action: np.ndarray | None = None
        reference = None
        current_reference_rank = 0
        if reference_processed is not None:
            from simple.grasp_rl.reference import ReferenceLibrary, ReferenceTracker

            reference = ReferenceTracker(
                ReferenceLibrary(
                    reference_processed,
                    source=reference_source,
                    splits=reference_splits,
                )
            )
        if rsi_dataset is not None:
            import pyarrow.parquet as pq

            dataset = Path(rsi_dataset)
            rsi_rows = [
                json.loads(line)
                for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
            ]
            valid_uids = (
                set(task_spec.source_uids)
                if is_v2
                else {task_spec.registry_uid}
            )
            rsi_rows = [
                row
                for row in rsi_rows
                if json.loads(row["environment_config"]).get("uid") in valid_uids
            ]
            if rsi_processed is not None:
                rsi_rows = _filter_replay_gated_rows(
                    rsi_rows, rsi_processed
                )
            if not rsi_rows:
                raise ValueError("rsi_dataset contains no compatible task episodes")
            if rsi_episodes is not None:
                allowed = set(rsi_episodes)
                rsi_rows = [
                    row
                    for row in rsi_rows
                    if int(row["episode_index"]) in allowed
                ]
                if not rsi_rows:
                    raise ValueError("rsi_episodes did not select any dataset rows")

            def load_rsi_actions(episode: int) -> np.ndarray:
                actions = rsi_action_cache.get(episode)
                if actions is None:
                    # V2 preparation may repair incomplete controller fields
                    # and append a simulator-feedback completion.  RSI must
                    # replay that audited physical trajectory, rather than the
                    # immutable but historically truncated source command.
                    prepared = (
                        Path(rsi_processed) / "bc" / f"episode_{episode:06d}.npz"
                        if rsi_processed is not None
                        else None
                    )
                    if prepared is not None and prepared.exists():
                        with np.load(prepared, allow_pickle=False) as saved:
                            actions = saved["physical_actions"].astype(np.float32)
                    else:
                        candidates = sorted(
                            (dataset / "data").glob(
                                f"chunk-*/episode_{episode:06d}.parquet"
                            )
                        )
                        if not candidates:
                            raise FileNotFoundError(f"Missing RSI episode {episode}")
                        actions = np.asarray(
                            pq.read_table(candidates[0], columns=["action"])[
                                "action"
                            ].to_pylist(),
                            dtype=np.float32,
                        )
                    rsi_action_cache[episode] = actions
                return actions

        def randomize_v2_training_target() -> None:
            nonlocal current_target_mode
            mode, jitter, center, yaw = sample_target_randomization(
                rng,
                runtime.get(
                    "target_mix",
                    {"uniform": 1.0, "hard": 0.0, "native": 0.0},
                ),
                runtime.get("hard_target_offsets_xy"),
                target_position_jitter_xy or (0.025, 0.03),
                target_yaw_jitter,
            )
            if mode == "uniform":
                center = target_position_offset_center_xy
            env.randomize_primary_pose(jitter, yaw, center)
            current_target_mode = mode

        def finalize_training_reset(observation, frame, is_rsi):
            nonlocal current_dr, delayed_raw_action
            dr = runtime["domain_randomization"]
            mass_low, mass_high = dr["target_mass_scale"]
            friction_low, friction_high = dr["friction_scale"]
            mass_scale = float(rng.uniform(mass_low, mass_high))
            friction_scale = float(rng.uniform(friction_low, friction_high))
            env.randomize_training_physics(mass_scale, friction_scale)
            delay_max = int(dr["action_delay_max_steps"])
            current_dr = {
                "target_mass_scale": mass_scale,
                "friction_scale": friction_scale,
                "action_delay_steps": int(rng.integers(delay_max + 1)),
            }
            delayed_raw_action = None
            return observation, frame, is_rsi

        def reset_episode(state_dict=None):
            # A newly sampled demonstration scene is rebuilt exactly once;
            # subsequent episodes restore its complete MuJoCo + tracker state.
            nonlocal current_rsi_actions, current_rsi_episode
            nonlocal current_rsi_state_dict
            nonlocal current_rsi_stage_bounds, rsi_scene_uses
            nonlocal current_reference_rank
            rsi_actions = None
            exact_reference_episode = None
            reference_rank = 0
            reference_start = 0
            if state_dict is not None:
                observation, frame = env.reset(state_dict=state_dict)
            elif rsi_rows is not None:
                if (
                    current_rsi_actions is None
                    or rsi_scene_uses >= rsi_scene_hold_episodes
                ):
                    # Rotate through the complete library, but amortize the
                    # expensive MuJoCo layout rebuild over several episodes.
                    row = rsi_rows[int(rng.integers(len(rsi_rows)))]
                    episode = int(row["episode_index"])
                    current_rsi_episode = episode
                    current_rsi_actions = load_rsi_actions(episode)
                    current_rsi_state_dict = json.loads(row["environment_config"])
                    observation, frame = env.reset(state_dict=current_rsi_state_dict)
                    rsi_scene_uses = 0
                    current_rsi_stage_bounds = None
                    if rsi_stage is not None or runtime.get("rsi_stage_weights"):
                        # Detect curriculum boundaries in exact expert replay.
                        # Moving the object first would intentionally break the
                        # expert grasp and make stage detection fall back to an
                        # arbitrary percentage of the trajectory.
                        first_grasp = None
                        first_lift = None
                        stage_entries: dict[int, int] = {}
                        for action_index, action in enumerate(current_rsi_actions):
                            scanned = env.step_physical(action)
                            if first_grasp is None and scanned.terms.is_grasp:
                                first_grasp = action_index
                            if first_lift is None and scanned.terms.lift_height >= 0.02:
                                first_lift = action_index
                            stage_entries.setdefault(
                                int(scanned.terms.stage_index), action_index
                            )
                        observation, frame = (
                            env.reset(state_dict=current_rsi_state_dict)
                            if is_v2
                            else env.reset()
                        )
                        current_rsi_stage_bounds = rsi_stage_bounds(
                            trajectory_length=len(current_rsi_actions),
                            task_family=(task_spec.family if is_v2 else "grasp"),
                            first_grasp=first_grasp,
                            first_lift=first_lift,
                            stage_entries=stage_entries,
                        )
                    env.capture_fast_reset_snapshot(
                        randomize_target=(rsi_randomize_target and not is_v2),
                        position_jitter_xy=target_position_jitter_xy,
                        yaw_jitter=target_yaw_jitter,
                    )
                    if rsi_randomize_target and is_v2:
                        randomize_v2_training_target()
                        assert env.state is not None and env.motion is not None
                        observation, _ = env.state.actor_observation()
                        frame = env.motion.extract()
                    elif rsi_randomize_target and not is_v2:
                        # The rebuild above establishes the exact recorded
                        # robot/reference origin.  Start the policy episode
                        # from an independently moved target immediately,
                        # including on the first use of a new scene.
                        observation, frame = env.reset()
                else:
                    if is_v2:
                        observation, frame = env.reset()
                        if rsi_randomize_target:
                            randomize_v2_training_target()
                            assert env.state is not None and env.motion is not None
                            observation, _ = env.state.actor_observation()
                            frame = env.motion.extract()
                    else:
                        observation, frame = env.reset()
                rsi_scene_uses += 1
                rsi_actions = current_rsi_actions
                # A moved target is a new task, so retrieve the geometrically
                # nearest complete plan just as production will.  Pinning the
                # original episode needlessly asks PPO to correct the entire
                # displacement even when a better plan already exists.
                exact_reference_episode = (
                    None if rsi_randomize_target else current_rsi_episode
                )
                if (
                    rsi_randomize_target
                    and current_rsi_episode is not None
                    and rng.random()
                    < runtime["reference_base_episode_probability"]
                ):
                    exact_reference_episode = current_rsi_episode
            else:
                observation, frame = env.reset()
            if reference is not None and exact_reference_episode is None:
                reference_rank = int(
                    rng.integers(int(runtime["reference_rank_max"]) + 1)
                )
            current_reference_rank = reference_rank
            if (
                rsi_actions is None
                or state_dict is not None
                or rng.random() >= runtime["rsi_probability"]
            ):
                if reference is not None:
                    reference.reset(
                        observation,
                        exact_episode=exact_reference_episode,
                        start_index=reference_start,
                        rank=reference_rank,
                    )
                    observation = reference.augment(observation)
                return finalize_training_reset(observation, frame, False)
            if current_rsi_stage_bounds is not None:
                stage = rsi_stage
                stage_weights = runtime.get("rsi_stage_weights")
                if stage_weights:
                    names = tuple(stage_weights)
                    weights = np.asarray(
                        [stage_weights[name] for name in names], dtype=float
                    )
                    weights /= weights.sum()
                    stage = names[int(rng.choice(len(names), p=weights))]
                if stage not in current_rsi_stage_bounds:
                    raise ValueError(
                        f"RSI stage {stage!r} is invalid for {task_spec.name}"
                    )
                low, high = current_rsi_stage_bounds[stage]
            elif rsi_phase is not None:
                phase_low, phase_high = rsi_phase
                last_index = len(rsi_actions) - 1
                low = int(np.ceil(phase_low * last_index))
                high = int(np.floor(phase_high * last_index))
                high = max(low, high)
            else:
                low, high = rsi_prefix
            if high < 0:
                if reference is not None:
                    reference.reset(
                        observation,
                        exact_episode=exact_reference_episode,
                        start_index=reference_start,
                        rank=reference_rank,
                    )
                    observation = reference.augment(observation)
                return finalize_training_reset(observation, frame, False)
            if low < 0:
                raise ValueError("rsi_prefix lower bound must be >= 0 when replay is enabled")
            stop = int(rng.integers(low, min(high, len(rsi_actions) - 1) + 1))
            last = None
            for action in rsi_actions[: stop + 1]:
                last = env.step_physical(action)
            assert last is not None
            assert env.reward is not None
            env.reward.reset()
            observation = last.actor_observation
            reference_start = stop + 1
            if reference is not None:
                reference.reset(
                    observation,
                    exact_episode=exact_reference_episode,
                    start_index=reference_start,
                    rank=reference_rank,
                )
                observation = reference.augment(observation)
            return finalize_training_reset(observation, last.motion_frame, True)

        observation, frame, is_rsi = reset_episode()
        connection.send(("ready", observation, frame, is_rsi))
        while True:
            command, payload = connection.recv()
            if command == "step":
                if reference is not None:
                    assert env.reward is not None
                    if hasattr(env.reward, "set_reference_contact"):
                        env.reward.set_reference_contact(
                            reference.post_step_contact_label(),
                            reference.post_step_contact_center_primary(),
                        )
                intended_raw = np.asarray(payload, dtype=np.float32)
                executed_raw = intended_raw.copy()
                action_noise_std = float(
                    runtime["domain_randomization"][
                        "manipulation_action_noise_std"
                    ]
                )
                if action_noise_std > 0.0:
                    noise = np.zeros(ACTION_DIM, dtype=np.float32)
                    noise[7:14] = rng.normal(0.0, action_noise_std, 7)
                    noise[21:28] = rng.normal(0.0, action_noise_std, 7)
                    executed_raw = np.clip(executed_raw + noise, -1.0, 1.0)
                if current_dr["action_delay_steps"]:
                    next_delayed = executed_raw
                    executed_raw = (
                        transform.encode(env.previous_physical_action)
                        if delayed_raw_action is None
                        else delayed_raw_action
                    )
                    delayed_raw_action = next_delayed
                step = env.step_raw(executed_raw)
                observation = step.actor_observation
                terms = step.terms.to_dict()
                terms["training_executed_raw_action"] = executed_raw.tolist()
                terms["dr_target_mass_scale"] = current_dr[
                    "target_mass_scale"
                ]
                terms["dr_friction_scale"] = current_dr["friction_scale"]
                terms["dr_action_delay_steps"] = current_dr[
                    "action_delay_steps"
                ]
                terms["curriculum_phase"] = runtime["phase_name"]
                terms["target_mode_uniform"] = float(
                    current_target_mode == "uniform"
                )
                terms["target_mode_hard"] = float(current_target_mode == "hard")
                terms["target_mode_native"] = float(
                    current_target_mode == "native"
                )
                if reference is not None:
                    executed_raw = transform.encode(env.previous_physical_action)
                    reference_terms = reference.reward(
                        step.actor_observation, executed_raw
                    )
                    terms.update(
                        {
                            f"reference_{name}": value
                            for name, value in reference_terms.to_dict().items()
                        }
                    )
                    terms["reference_rank"] = current_reference_rank
                    observation = reference.augment(step.actor_observation)
                    # GRAIL's tracking environment ends at motion_time_out.
                    # Continuing hundreds of steps after the plan ends creates
                    # an unrelated objective and lets PPO erase useful motion.
                    reference_time_out = reference.is_complete and not (
                        terms["success"] or terms["failure"]
                    )
                    if reference_time_out:
                        terms["timeout"] = True
                else:
                    terms["reference_total"] = 0.0
                    terms["reference_rank"] = 0
                    reference_time_out = False
                connection.send(
                    (
                        "step",
                        observation,
                        step.motion_frame,
                        terms,
                        step.done or reference_time_out,
                    )
                )
            elif command == "reset":
                observation, frame, is_rsi = reset_episode(state_dict=payload)
                connection.send(("reset", observation, frame, is_rsi))
            elif command == "configure":
                runtime.update(payload)
                connection.send(("configured", runtime["phase_name"]))
            elif command == "close":
                env.close()
                connection.close()
                return
            else:
                raise ValueError(f"Unknown worker command {command}")
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        connection.close()


class DistributedGraspVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        action_transform_path: str | Path,
        device: str = "cuda:0",
        worker_devices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
        diffusion_checkpoint: str | Path | None = None,
        reward_variant: str = "task_only",
        ws: float = 4.0,
        task_reward_weight: float = 0.02,
        smp_reward_weight: float = 0.01,
        task_reward_profile: str = DEFAULT_TASK_REWARD_PROFILE,
        seed: int = 42,
        observation_noise: bool = True,
        rsi_dataset: str | Path | None = None,
        rsi_processed: str | Path | None = None,
        rsi_prefix: tuple[int, int] = (75, 115),
        rsi_phase: tuple[float, float] | None = None,
        rsi_stage: str | None = None,
        rsi_episodes: tuple[int, ...] | None = None,
        rsi_probability: float = 0.0,
        rsi_scene_hold_episodes: int = 32,
        rsi_randomize_target: bool = False,
        target_position_jitter_xy: tuple[float, float] | None = (0.025, 0.03),
        target_position_offset_center_xy: tuple[float, float] = (0.0, 0.0),
        target_yaw_jitter: float = 0.15,
        reference_processed: str | Path | None = None,
        reference_source: str = "bc",
    reference_splits: tuple[str, ...] = ("train", "val", "test"),
    reference_reward_weight: float = 0.0,
    reference_rank_max: int = 0,
    reference_base_episode_probability: float = 0.0,
    reference_action_noise_std: float = 0.0,
    reference_action_noise_hold_steps: int = 25,
    task: str = DEFAULT_TASK,
    training_config: dict | None = None,
    ):
        task_spec = get_task_spec(task)
        self.num_envs = num_envs
        self.num_actions = ACTION_DIM
        self.max_episode_length = task_spec.max_episode_steps
        self.device = torch.device(device)
        self.cfg = {
            "num_envs": num_envs,
            "reward_variant": reward_variant,
            "ws": ws,
            "task_reward_weight": task_reward_weight,
            "smp_reward_weight": smp_reward_weight,
            "task_reward_profile": task_reward_profile,
            "reference_processed": (
                str(reference_processed) if reference_processed is not None else None
            ),
            "reference_source": reference_source,
            "reference_splits": reference_splits,
            "reference_reward_weight": reference_reward_weight,
            "reference_rank_max": reference_rank_max,
            "reference_base_episode_probability": (
                reference_base_episode_probability
            ),
            "reference_action_noise_std": reference_action_noise_std,
            "reference_action_noise_hold_steps": reference_action_noise_hold_steps,
            "rsi_randomize_target": rsi_randomize_target,
            "rsi_processed": (
                str(rsi_processed) if rsi_processed is not None else None
            ),
            "target_position_jitter_xy": target_position_jitter_xy,
            "target_position_offset_center_xy": target_position_offset_center_xy,
            "target_yaw_jitter": target_yaw_jitter,
            "seed": seed,
            "task": task_spec.name,
        }
        if not worker_devices:
            raise ValueError("worker_devices must contain at least one CUDA device")
        if not 0.0 <= rsi_probability <= 1.0:
            raise ValueError("rsi_probability must be in [0, 1]")
        if rsi_phase is not None and not (
            0.0 <= rsi_phase[0] <= rsi_phase[1] <= 1.0
        ):
            raise ValueError("rsi_phase must satisfy 0 <= low <= high <= 1")
        if rsi_stage not in {None, "pregrasp", "grasp_to_lift", "lift"}:
            raise ValueError(f"Unknown rsi_stage {rsi_stage}")
        if rsi_scene_hold_episodes < 1:
            raise ValueError("rsi_scene_hold_episodes must be at least 1")
        if target_position_jitter_xy is not None and (
            len(target_position_jitter_xy) != 2
            or any(value < 0.0 for value in target_position_jitter_xy)
        ):
            raise ValueError(
                "target_position_jitter_xy must contain two non-negative values"
            )
        if len(target_position_offset_center_xy) != 2:
            raise ValueError(
                "target_position_offset_center_xy must contain two values"
            )
        if target_yaw_jitter < 0.0:
            raise ValueError("target_yaw_jitter must be non-negative")
        if rsi_randomize_target and rsi_dataset is None:
            raise ValueError("rsi_randomize_target requires rsi_dataset")
        if rsi_processed is not None and rsi_dataset is None:
            raise ValueError("rsi_processed requires rsi_dataset")
        if reference_reward_weight < 0.0:
            raise ValueError("reference_reward_weight must be non-negative")
        if reference_reward_weight > 0.0 and reference_processed is None:
            raise ValueError(
                "reference_processed is required when reference reward is enabled"
            )
        if reference_rank_max < 0:
            raise ValueError("reference_rank_max must be non-negative")
        if not 0.0 <= reference_base_episode_probability <= 1.0:
            raise ValueError(
                "reference_base_episode_probability must be in [0, 1]"
            )
        if reference_action_noise_std < 0.0:
            raise ValueError("reference_action_noise_std must be non-negative")
        if reference_action_noise_hold_steps < 1:
            raise ValueError("reference_action_noise_hold_steps must be at least 1")
        if (
            reference_rank_max > 0
            or reference_base_episode_probability > 0.0
            or reference_action_noise_std > 0.0
        ) and reference_processed is None:
            raise ValueError("Reference perturbation requires reference_processed")
        if reward_variant not in REWARD_VARIANTS:
            raise ValueError(f"Unknown reward variant {reward_variant}")
        if task_reward_profile not in TASK_REWARD_PROFILES:
            raise ValueError(f"Unknown task reward profile {task_reward_profile}")
        if reward_variant != "task_only" and diffusion_checkpoint is None:
            raise ValueError(f"{reward_variant} requires a diffusion checkpoint")
        self.reward_variant = reward_variant
        self.task_reward_weight = task_reward_weight
        self.smp_reward_weight = smp_reward_weight
        self.observation_noise = observation_noise
        self.reference_reward_weight = reference_reward_weight
        self.reference_action_noise_std = reference_action_noise_std
        self.reference_action_noise_hold_steps = reference_action_noise_hold_steps
        self._training_config = {
            "phase_name": "static",
            "rsi_probability": float(rsi_probability),
            "reference_rank_max": int(reference_rank_max),
            "reference_base_episode_probability": float(
                reference_base_episode_probability
            ),
            "reference_action_noise_std": float(reference_action_noise_std),
            "domain_randomization": {
                "target_mass_scale": (1.0, 1.0),
                "friction_scale": (1.0, 1.0),
                "manipulation_action_noise_std": 0.0,
                "action_delay_max_steps": 0,
            },
        }
        self._training_config.update(training_config or {})
        self.reference_action_noise_std = float(
            self._training_config.get(
                "reference_action_noise_std", self.reference_action_noise_std
            )
        )
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._episode_return = torch.zeros(num_envs, device=self.device)
        self._episode_success = torch.zeros(num_envs, device=self.device)
        self._episode_native_success = torch.zeros(num_envs, device=self.device)
        self._episode_max_grasp_quality = torch.zeros(num_envs, device=self.device)
        self._episode_max_lift = torch.full(
            (num_envs,), -torch.inf, device=self.device
        )
        self._closed = False

        context = mp.get_context("spawn")
        self._connections = []
        self._processes = []
        for index in range(num_envs):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(
                    child,
                    str(Path(action_transform_path).resolve()),
                    seed + 1009 * index,
                    worker_devices[index % len(worker_devices)],
                    str(Path(rsi_dataset).resolve()) if rsi_dataset is not None else None,
                    (
                        str(Path(rsi_processed).resolve())
                        if rsi_processed is not None
                        else None
                    ),
                    rsi_prefix,
                    rsi_phase,
                    rsi_stage,
                    rsi_episodes,
                    rsi_probability,
                    rsi_scene_hold_episodes,
                    rsi_randomize_target,
                    target_position_jitter_xy,
                    target_position_offset_center_xy,
                    target_yaw_jitter,
                    task_reward_profile,
                    (
                        str(Path(reference_processed).resolve())
                        if reference_processed is not None
                        else None
                    ),
                    reference_source,
                    reference_splits,
                    reference_rank_max,
                    reference_base_episode_probability,
                    task_spec.name,
                    self._training_config,
                ),
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

        initial_obs, initial_frames, initial_is_rsi = [], [], []
        for connection in self._connections:
            message = connection.recv()
            self._raise_worker_error(message)
            _, observation, frame, is_rsi = message
            initial_obs.append(observation)
            initial_frames.append(frame)
            initial_is_rsi.append(is_rsi)
        self._clean_obs = torch.as_tensor(np.stack(initial_obs), device=self.device)
        if reference_processed is not None:
            if self._clean_obs.shape[1] == 842:
                self._reference_base_dim = ACTOR_OBS_V2_DIM
                self._reference_frame_dim = 51
            elif self._clean_obs.shape[1] == 593:
                self._reference_base_dim = ACTOR_OBS_DIM
                self._reference_frame_dim = 40
            else:
                raise ValueError(
                    "Reference-conditioned observation has unsupported dimension "
                    f"{self._clean_obs.shape[1]}"
                )
        else:
            self._reference_base_dim = None
            self._reference_frame_dim = None
        self._reference_action_bias = torch.zeros(
            (num_envs, ACTION_DIM), device=self.device
        )
        # GRAIL's meta_action_rate_l2 acts on the policy residual, not on the
        # generated reference motion.  In this implementation the RSL action
        # is the complete command, so reconstruct that residual explicitly.
        self._previous_policy_residual = torch.zeros(
            (num_envs, ACTION_DIM), device=self.device
        )
        self._reference_noise_age = 0
        self._refresh_reference_action_bias()
        self._critic_history = self._clean_obs[:, None, :].repeat(1, 10, 1)
        self._motion = BatchedMotionBuffer(num_envs, self.device)
        ids = torch.arange(num_envs, device=self.device)
        self._motion.reset(ids, torch.as_tensor(np.stack(initial_frames), device=self.device))
        self._episode_is_rsi = torch.as_tensor(
            initial_is_rsi, dtype=torch.bool, device=self.device
        )
        self._guidance = (
            Guidance(diffusion_checkpoint, self.device, ws=ws)
            if diffusion_checkpoint is not None
            else None
        )

    @staticmethod
    def _raise_worker_error(message: tuple) -> None:
        if message[0] == "error":
            raise RuntimeError(f"SIMPLE worker failed:\n{message[1]}")

    def configure_training(self, config: dict) -> None:
        """Broadcast one curriculum phase between synchronous env steps."""

        self._training_config = dict(config)
        for connection in self._connections:
            connection.send(("configure", self._training_config))
        for connection in self._connections:
            message = connection.recv()
            self._raise_worker_error(message)
            if message[0] != "configured":
                raise RuntimeError(f"Unexpected worker response {message[0]!r}")
        new_reference_noise = float(
            config.get(
                "reference_action_noise_std", self.reference_action_noise_std
            )
        )
        if new_reference_noise != self.reference_action_noise_std:
            self.reference_action_noise_std = new_reference_noise
            self._reference_action_bias.zero_()
            self._refresh_reference_action_bias()
            self._reference_noise_age = 0

    def _actor_obs(self) -> torch.Tensor:
        observation = self._clean_obs.clone()
        if self.observation_noise:
            def noise(start: int, stop: int, magnitude: float) -> None:
                observation[:, start:stop] += (
                    2 * torch.rand_like(observation[:, start:stop]) - 1
                ) * magnitude

            noise(0, 43, 0.01)
            noise(43, 86, 0.5)
            noise(86, 89, 0.05)
            noise(89, 92, 0.5)
            noise(92, 95, 0.2)
        if self.reference_action_noise_std > 0.0:
            assert self._reference_base_dim is not None
            assert self._reference_frame_dim is not None
            observation = apply_reference_action_bias(
                observation,
                self._reference_action_bias,
                self._reference_base_dim,
                self._reference_frame_dim,
            )
        return observation

    def _refresh_reference_action_bias(
        self, indices: torch.Tensor | None = None
    ) -> None:
        if self.reference_action_noise_std == 0.0:
            return
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        if indices.numel() == 0:
            return
        sample = torch.randn((len(indices), ACTION_DIM), device=self.device)
        sample.clamp_(-2.0, 2.0).mul_(self.reference_action_noise_std)
        manipulation_mask = torch.zeros(ACTION_DIM, device=self.device)
        manipulation_mask[7:14] = 1.0
        manipulation_mask[21:28] = 1.0
        self._reference_action_bias[indices] = sample * manipulation_mask

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {
                "actor": self._actor_obs(),
                "critic": self._critic_history.flatten(1),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor):
        actions = actions.detach()
        policy_action_rate: torch.Tensor | None = None
        proposal: torch.Tensor | None = None
        if self._reference_base_dim is not None:
            proposal = self._clean_obs[
                :,
                self._reference_base_dim : self._reference_base_dim + ACTION_DIM,
            ].clone()
            if self.reference_action_noise_std > 0.0:
                proposal = torch.clamp(
                    proposal + self._reference_action_bias, -1.0, 1.0
                )
        action_array = actions.cpu().numpy().astype(np.float32)
        for connection, action in zip(self._connections, action_array, strict=True):
            connection.send(("step", action))

        observations, frames, term_rows, done_rows = [], [], [], []
        for connection in self._connections:
            message = connection.recv()
            self._raise_worker_error(message)
            _, observation, frame, terms, done = message
            observations.append(observation)
            frames.append(frame)
            term_rows.append(terms)
            done_rows.append(done)

        if proposal is not None:
            executed_actions = torch.tensor(
                [row["training_executed_raw_action"] for row in term_rows],
                device=self.device,
            )
            policy_action_rate, residual = reference_residual_action_rate(
                executed_actions, proposal, self._previous_policy_residual
            )
            self._previous_policy_residual.copy_(residual)

        terminal_obs = torch.as_tensor(np.stack(observations), device=self.device)
        frame_tensor = torch.as_tensor(np.stack(frames), device=self.device)
        self._motion.update(frame_tensor)
        target = torch.tensor([row["target_reward"] for row in term_rows], device=self.device)
        penalty = torch.tensor([row["penalty"] for row in term_rows], device=self.device)
        if policy_action_rate is not None:
            complete_action_rate = torch.tensor(
                [row["action_rate_penalty"] for row in term_rows],
                device=self.device,
            )
            # GoalGraphReward includes 0.1 * complete-command rate. Replace it
            # with the exact pnp_table coefficient on the learned residual.
            penalty = penalty - 0.1 * complete_action_rate + 0.1 * policy_action_rate
            for row, value in zip(
                term_rows, policy_action_rate.detach().cpu().tolist(), strict=True
            ):
                row["action_rate_penalty"] = value
        adjustment = torch.tensor([row["terminal_adjustment"] for row in term_rows], device=self.device)
        if self.reward_variant != "task_only":
            assert self._guidance is not None
            full = self._motion.is_full
            smp = torch.ones(self.num_envs, device=self.device)
            raw_error = torch.zeros(self.num_envs, device=self.device)
            if full.any():
                full_score, full_error = self._guidance.score(self._motion.features()[full])
                smp[full] = full_score
                raw_error[full] = full_error
        else:
            smp = torch.ones(self.num_envs, device=self.device)
            raw_error = torch.zeros(self.num_envs, device=self.device)
        rewards, task_component, smp_contribution = compose_reward(
            target,
            penalty,
            adjustment,
            smp,
            self._motion.is_full.to(dtype=target.dtype),
            self.reward_variant,
            self.task_reward_weight,
            self.smp_reward_weight,
        )
        reference_raw = torch.tensor(
            [row.get("reference_total", 0.0) for row in term_rows],
            device=self.device,
        )
        reference_contribution = self.reference_reward_weight * reference_raw
        rewards = rewards + reference_contribution
        dones = torch.tensor(done_rows, dtype=torch.bool, device=self.device)
        timeouts = torch.tensor([row["timeout"] for row in term_rows], dtype=torch.float32, device=self.device)

        self.episode_length_buf += 1
        self._episode_return += rewards
        self._episode_success = torch.maximum(
            self._episode_success,
            torch.tensor(
                [row["success"] for row in term_rows],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        self._episode_native_success = torch.maximum(
            self._episode_native_success,
            torch.tensor(
                [row["native_success"] for row in term_rows],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        step_grasp_quality = torch.tensor(
            [row["grasp_quality"] for row in term_rows], device=self.device
        )
        step_lift = torch.tensor(
            [row["lift_height"] for row in term_rows], device=self.device
        )
        self._episode_max_grasp_quality = torch.maximum(
            self._episode_max_grasp_quality, step_grasp_quality
        )
        self._episode_max_lift = torch.maximum(self._episode_max_lift, step_lift)
        finished = dones.nonzero(as_tuple=False).flatten()
        log: dict[str, torch.Tensor] = {
            "reward/target": target.mean().reshape(1),
            "reward/smp": smp.mean().reshape(1),
            "reward/raw_smp_mse": raw_error.mean().reshape(1),
            "reward/penalty": penalty.mean().reshape(1),
            "reward/task_component": task_component.mean().reshape(1),
            "reward/smp_contribution": smp_contribution.mean().reshape(1),
            "reward/reference_raw": reference_raw.mean().reshape(1),
            "reward/reference_contribution": reference_contribution.mean().reshape(1),
            "reference/root": torch.tensor([row.get("reference_root", 0.0) for row in term_rows], device=self.device).mean().reshape(1),
            "reference/joint_pose": torch.tensor([row.get("reference_joint_pose", 0.0) for row in term_rows], device=self.device).mean().reshape(1),
            "reference/joint_velocity": torch.tensor([row.get("reference_joint_velocity", 0.0) for row in term_rows], device=self.device).mean().reshape(1),
            "reference/tracker_action": torch.tensor([row.get("reference_tracker_action", 0.0) for row in term_rows], device=self.device).mean().reshape(1),
            "reference/rank": torch.tensor(
                [row.get("reference_rank", 0.0) for row in term_rows],
                device=self.device,
                dtype=torch.float32,
            ).mean().reshape(1),
            "reference/action_bias_rms": torch.cat(
                (
                    self._reference_action_bias[:, 7:14],
                    self._reference_action_bias[:, 21:28],
                ),
                dim=-1,
            ).square().mean().sqrt().reshape(1),
            "domain_randomization/target_mass_scale": torch.tensor(
                [row["dr_target_mass_scale"] for row in term_rows],
                device=self.device,
            ).mean().reshape(1),
            "domain_randomization/friction_scale": torch.tensor(
                [row["dr_friction_scale"] for row in term_rows],
                device=self.device,
            ).mean().reshape(1),
            "domain_randomization/action_delay_steps": torch.tensor(
                [row["dr_action_delay_steps"] for row in term_rows],
                device=self.device,
                dtype=torch.float32,
            ).mean().reshape(1),
            "curriculum/rsi_probability": torch.tensor(
                [self._training_config.get("rsi_probability", 0.0)],
                device=self.device,
            ),
            "curriculum/target_uniform": torch.tensor(
                [row["target_mode_uniform"] for row in term_rows],
                device=self.device,
            ).mean().reshape(1),
            "curriculum/target_hard": torch.tensor(
                [row["target_mode_hard"] for row in term_rows],
                device=self.device,
            ).mean().reshape(1),
            "curriculum/target_native": torch.tensor(
                [row["target_mode_native"] for row in term_rows],
                device=self.device,
            ).mean().reshape(1),
            "task/pregrasp": torch.tensor([row["pregrasp"] for row in term_rows], device=self.device).mean().reshape(1),
            "task/grasp_quality": torch.tensor([row["grasp_quality"] for row in term_rows], device=self.device).mean().reshape(1),
            "task/grail_grasp": torch.tensor([row["grail_grasp"] for row in term_rows], device=self.device).mean().reshape(1),
            "task/grail_finger_direction": torch.tensor([row["grail_finger_direction"] for row in term_rows], device=self.device).mean().reshape(1),
            "task/lift": torch.tensor([row["lift_height"] for row in term_rows], device=self.device).mean().reshape(1),
            "task/stable": torch.tensor([row["stable"] for row in term_rows], device=self.device).mean().reshape(1),
            "task/progress": torch.tensor([row["progress"] for row in term_rows], device=self.device).mean().reshape(1),
            "task/progress_bonus": torch.tensor([row["progress_bonus"] for row in term_rows], device=self.device).mean().reshape(1),
            "penalty/approach": torch.tensor([row["approach_penalty"] for row in term_rows], device=self.device).mean().reshape(1),
            "penalty/table": torch.tensor([row["table_penalty"] for row in term_rows], device=self.device).mean().reshape(1),
            # Keep the keys present from the first step so RSL-RL's logger does
            # not silently omit episode metrics when the first transition is
            # non-terminal. Empty tensors contribute no samples.
            "success": torch.empty(0, device=self.device),
            "native_success": torch.empty(0, device=self.device),
            "success_rsi": torch.empty(0, device=self.device),
            "success_full_start": torch.empty(0, device=self.device),
            "return": torch.empty(0, device=self.device),
            "length": torch.empty(0, device=self.device),
            "max_grasp_quality": torch.empty(0, device=self.device),
            "max_lift": torch.empty(0, device=self.device),
        }
        if finished.numel():
            log["success"] = self._episode_success[finished].mean().reshape(1)
            log["native_success"] = (
                self._episode_native_success[finished].mean().reshape(1)
            )
            rsi_finished = finished[self._episode_is_rsi[finished]]
            full_finished = finished[~self._episode_is_rsi[finished]]
            if rsi_finished.numel():
                log["success_rsi"] = (
                    self._episode_success[rsi_finished].mean().reshape(1)
                )
            if full_finished.numel():
                log["success_full_start"] = (
                    self._episode_success[full_finished].mean().reshape(1)
                )
            log["return"] = self._episode_return[finished].mean().reshape(1)
            log["length"] = self.episode_length_buf[finished].float().mean().reshape(1)
            log["max_grasp_quality"] = (
                self._episode_max_grasp_quality[finished].mean().reshape(1)
            )
            log["max_lift"] = self._episode_max_lift[finished].mean().reshape(1)

        next_obs = terminal_obs.clone()
        for index in finished.cpu().tolist():
            self._connections[index].send(("reset", None))
        reset_frames, reset_is_rsi = [], []
        for index in finished.cpu().tolist():
            message = self._connections[index].recv()
            self._raise_worker_error(message)
            _, observation, frame, is_rsi = message
            next_obs[index] = torch.as_tensor(observation, device=self.device)
            reset_frames.append(frame)
            reset_is_rsi.append(is_rsi)
        if finished.numel():
            reset_frame_tensor = torch.as_tensor(np.stack(reset_frames), device=self.device)
            self._motion.reset(finished, reset_frame_tensor)
            self._episode_is_rsi[finished] = torch.as_tensor(
                reset_is_rsi, dtype=torch.bool, device=self.device
            )
            self._episode_return[finished] = 0
            self._episode_success[finished] = 0
            self._episode_native_success[finished] = 0
            self._episode_max_grasp_quality[finished] = 0
            self._episode_max_lift[finished] = -torch.inf
            self._previous_policy_residual[finished] = 0
            self.episode_length_buf[finished] = 0

        self._clean_obs = next_obs
        self._reference_noise_age += 1
        if self._reference_noise_age >= self.reference_action_noise_hold_steps:
            self._refresh_reference_action_bias()
            self._reference_noise_age = 0
        self._refresh_reference_action_bias(finished)
        self._critic_history = torch.roll(self._critic_history, shifts=-1, dims=1)
        self._critic_history[:, -1] = next_obs
        if finished.numel():
            self._critic_history[finished] = next_obs[finished, None, :]
        extras = {"time_outs": timeouts, "log": log}
        return self.get_observations(), rewards, dones, extras

    def close(self) -> None:
        if self._closed:
            return
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for connection in self._connections:
            connection.close()
        self._closed = True

    def __del__(self):
        self.close()
