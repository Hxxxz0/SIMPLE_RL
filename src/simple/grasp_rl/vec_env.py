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
from simple.grasp_rl.motion import BatchedMotionBuffer
from simple.grasp_rl.rewards import (
    DEFAULT_TASK_REWARD_PROFILE,
    REWARD_VARIANTS,
    TASK_REWARD_PROFILES,
    compose_reward,
)
from simple.grasp_rl.schema import ACTION_DIM, MAX_EPISODE_STEPS
from simple.grasp_rl.tracker import ActionTransform


def _worker(
    connection,
    action_transform_path: str,
    seed: int,
    cuda_device: int,
    rsi_dataset: str | None,
    rsi_prefix: tuple[int, int],
    rsi_phase: tuple[float, float] | None,
    rsi_stage: str | None,
    rsi_episodes: tuple[int, ...] | None,
    rsi_probability: float,
    rsi_scene_hold_episodes: int,
    rsi_randomize_target: bool,
    target_position_jitter_xy: tuple[float, float] | None,
    target_yaw_jitter: float,
    task_reward_profile: str,
    reference_processed: str | None,
    reference_source: str,
    reference_splits: tuple[str, ...],
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        os.environ.setdefault("MUJOCO_GL", "egl")
        torch.set_num_threads(1)
        from simple.grasp_rl.env import GraspRlEnv

        transform = ActionTransform.from_npz(action_transform_path)
        env = GraspRlEnv(
            transform,
            seed=seed,
            max_episode_steps=MAX_EPISODE_STEPS,
            task_reward_profile=task_reward_profile,
        )
        rsi_rows = None
        rsi_action_cache: dict[int, np.ndarray] = {}
        current_rsi_actions: np.ndarray | None = None
        current_rsi_episode: int | None = None
        current_rsi_stage_bounds: tuple[int, int] | None = None
        rsi_scene_uses = 0
        rng = np.random.default_rng(seed + 7919)
        reference = None
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
                    parquet = (
                        dataset
                        / "data"
                        / "chunk-000"
                        / f"episode_{episode:06d}.parquet"
                    )
                    actions = np.asarray(
                        pq.read_table(parquet, columns=["action"])[
                            "action"
                        ].to_pylist(),
                        dtype=np.float32,
                    )
                    rsi_action_cache[episode] = actions
                return actions

        def reset_episode(state_dict=None):
            # A newly sampled demonstration scene is rebuilt exactly once;
            # subsequent episodes restore its complete MuJoCo + tracker state.
            nonlocal current_rsi_actions, current_rsi_episode
            nonlocal current_rsi_stage_bounds, rsi_scene_uses
            rsi_actions = None
            exact_reference_episode = None
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
                    observation, frame = env.reset(
                        state_dict=json.loads(row["environment_config"])
                    )
                    env.capture_fast_reset_snapshot(
                        randomize_target=rsi_randomize_target,
                        position_jitter_xy=target_position_jitter_xy,
                        yaw_jitter=target_yaw_jitter,
                    )
                    rsi_scene_uses = 0
                    if rsi_stage is not None:
                        first_grasp = None
                        first_lift = None
                        for action_index, action in enumerate(current_rsi_actions):
                            scanned = env.step_physical(action)
                            if first_grasp is None and scanned.terms.is_grasp:
                                first_grasp = action_index
                            if first_lift is None and scanned.terms.lift_height >= 0.02:
                                first_lift = action_index
                        observation, frame = env.reset()
                        last_index = len(current_rsi_actions) - 1
                        if first_grasp is None:
                            first_grasp = int(round(0.67 * last_index))
                        if first_lift is None:
                            first_lift = int(round(0.90 * last_index))
                        if rsi_stage == "pregrasp":
                            current_rsi_stage_bounds = (
                                max(first_grasp - 20, 0),
                                first_grasp,
                            )
                        elif rsi_stage == "grasp_to_lift":
                            current_rsi_stage_bounds = (
                                first_grasp,
                                max(first_grasp, first_lift - 1),
                            )
                        else:
                            current_rsi_stage_bounds = (
                                max(first_lift - 3, 0),
                                min(first_lift + 3, last_index),
                            )
                    elif rsi_randomize_target:
                        # The rebuild above establishes the exact recorded
                        # robot/reference origin.  Start the policy episode
                        # from an independently moved target immediately,
                        # including on the first use of a new scene.
                        observation, frame = env.reset()
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
            else:
                observation, frame = env.reset()
            if (
                rsi_actions is None
                or state_dict is not None
                or rng.random() >= rsi_probability
            ):
                if reference is not None:
                    reference.reset(
                        observation,
                        exact_episode=exact_reference_episode,
                        start_index=reference_start,
                    )
                    observation = reference.augment(observation)
                return observation, frame, False
            if current_rsi_stage_bounds is not None:
                low, high = current_rsi_stage_bounds
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
                    )
                    observation = reference.augment(observation)
                return observation, frame, False
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
                )
                observation = reference.augment(observation)
            return observation, last.motion_frame, True

        observation, frame, is_rsi = reset_episode()
        connection.send(("ready", observation, frame, is_rsi))
        while True:
            command, payload = connection.recv()
            if command == "step":
                if reference is not None:
                    assert env.reward is not None
                    env.reward.set_reference_contact(
                        reference.post_step_contact_label()
                    )
                step = env.step_raw(payload)
                observation = step.actor_observation
                terms = step.terms.to_dict()
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
                    observation = reference.augment(step.actor_observation)
                else:
                    terms["reference_total"] = 0.0
                connection.send(
                    (
                        "step",
                        observation,
                        step.motion_frame,
                        terms,
                        step.done,
                    )
                )
            elif command == "reset":
                observation, frame, is_rsi = reset_episode(state_dict=payload)
                connection.send(("reset", observation, frame, is_rsi))
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
        rsi_prefix: tuple[int, int] = (75, 115),
        rsi_phase: tuple[float, float] | None = None,
        rsi_stage: str | None = None,
        rsi_episodes: tuple[int, ...] | None = None,
        rsi_probability: float = 0.0,
        rsi_scene_hold_episodes: int = 32,
        rsi_randomize_target: bool = False,
        target_position_jitter_xy: tuple[float, float] | None = (0.025, 0.03),
        target_yaw_jitter: float = 0.15,
        reference_processed: str | Path | None = None,
        reference_source: str = "bc",
        reference_splits: tuple[str, ...] = ("train", "val", "test"),
        reference_reward_weight: float = 0.0,
    ):
        self.num_envs = num_envs
        self.num_actions = ACTION_DIM
        self.max_episode_length = MAX_EPISODE_STEPS
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
            "rsi_randomize_target": rsi_randomize_target,
            "target_position_jitter_xy": target_position_jitter_xy,
            "target_yaw_jitter": target_yaw_jitter,
            "seed": seed,
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
        if target_yaw_jitter < 0.0:
            raise ValueError("target_yaw_jitter must be non-negative")
        if rsi_randomize_target and rsi_dataset is None:
            raise ValueError("rsi_randomize_target requires rsi_dataset")
        if reference_reward_weight < 0.0:
            raise ValueError("reference_reward_weight must be non-negative")
        if reference_reward_weight > 0.0 and reference_processed is None:
            raise ValueError(
                "reference_processed is required when reference reward is enabled"
            )
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
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._episode_return = torch.zeros(num_envs, device=self.device)
        self._episode_success = torch.zeros(num_envs, device=self.device)
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
                    rsi_prefix,
                    rsi_phase,
                    rsi_stage,
                    rsi_episodes,
                    rsi_probability,
                    rsi_scene_hold_episodes,
                    rsi_randomize_target,
                    target_position_jitter_xy,
                    target_yaw_jitter,
                    task_reward_profile,
                    (
                        str(Path(reference_processed).resolve())
                        if reference_processed is not None
                        else None
                    ),
                    reference_source,
                    reference_splits,
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

    def _actor_obs(self) -> torch.Tensor:
        observation = self._clean_obs.clone()
        if not self.observation_noise:
            return observation
        def noise(start: int, stop: int, magnitude: float) -> None:
            observation[:, start:stop] += (2 * torch.rand_like(observation[:, start:stop]) - 1) * magnitude

        noise(0, 43, 0.01)
        noise(43, 86, 0.5)
        noise(86, 89, 0.05)
        noise(89, 92, 0.5)
        noise(92, 95, 0.2)
        return observation

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
        action_array = actions.detach().cpu().numpy().astype(np.float32)
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

        terminal_obs = torch.as_tensor(np.stack(observations), device=self.device)
        frame_tensor = torch.as_tensor(np.stack(frames), device=self.device)
        self._motion.update(frame_tensor)
        target = torch.tensor([row["target_reward"] for row in term_rows], device=self.device)
        penalty = torch.tensor([row["penalty"] for row in term_rows], device=self.device)
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
            "success_rsi": torch.empty(0, device=self.device),
            "success_full_start": torch.empty(0, device=self.device),
            "return": torch.empty(0, device=self.device),
            "length": torch.empty(0, device=self.device),
            "max_grasp_quality": torch.empty(0, device=self.device),
            "max_lift": torch.empty(0, device=self.device),
        }
        if finished.numel():
            log["success"] = self._episode_success[finished].mean().reshape(1)
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
            self._episode_max_grasp_quality[finished] = 0
            self._episode_max_lift[finished] = -torch.inf
            self.episode_length_buf[finished] = 0

        self._clean_obs = next_obs
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
