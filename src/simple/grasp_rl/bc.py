"""Replay-derived behavior cloning for a full-command actor warm start."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.policy import load_actor, make_actor
from simple.grasp_rl.reference import augment_reference_observation
from simple.grasp_rl.rewards import DEFAULT_TASK_REWARD_PROFILE
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_DIM,
)
from simple.grasp_rl.tracker import ActionTransform


def _episode_file(dataset: Path, episode: int) -> Path:
    return dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"


def _replay_bc_chunk(
    dataset_string: str,
    output_string: str,
    rows: list[dict[str, Any]],
    episodes: list[int],
    transform_path: str,
    cuda_device: int,
    rollout_checkpoint: str | None = None,
    initialization_prefix: int | None = None,
    task_reward_profile: str = DEFAULT_TASK_REWARD_PROFILE,
    teacher_checkpoint: str | None = None,
    teacher_rollout_blend: float = 0.0,
    teacher_rollout_probability: float = 0.0,
) -> list[dict[str, Any]]:
    if not 0.0 <= teacher_rollout_blend <= 1.0:
        raise ValueError("teacher_rollout_blend must be in [0, 1]")
    if not 0.0 <= teacher_rollout_probability <= 1.0:
        raise ValueError("teacher_rollout_probability must be in [0, 1]")
    if teacher_rollout_blend > 0.0 and teacher_rollout_probability > 0.0:
        raise ValueError("teacher blend and probability are mutually exclusive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(1)
    dataset = Path(dataset_string)
    output = Path(output_string)
    transform = ActionTransform.from_npz(transform_path)
    env = GraspRlEnv(
        transform,
        seed=42 + episodes[0],
        task_reward_profile=task_reward_profile,
    )
    actor = None
    if rollout_checkpoint is not None:
        actor = load_actor(rollout_checkpoint, "cuda:0")
    teacher = None
    if teacher_checkpoint is not None:
        teacher = load_actor(teacher_checkpoint, "cuda:0")
    reports: list[dict[str, Any]] = []
    try:
        for episode in episodes:
            if actor is not None and hasattr(actor, "reset"):
                actor.reset()
            if teacher is not None and hasattr(teacher, "reset"):
                teacher.reset()
            recorded_actions = np.asarray(
                pq.read_table(_episode_file(dataset, episode), columns=["action"])[
                    "action"
                ].to_pylist(),
                dtype=np.float32,
            )
            # The demonstrations terminate immediately after lifting. Repeating
            # the last complete tracker command generates the missing stable
            # hold states through the real tracker and MuJoCo dynamics.
            actions = np.concatenate(
                [recorded_actions, np.repeat(recorded_actions[-1:], 40, axis=0)],
                axis=0,
            )
            observation, _ = env.reset(
                state_dict=json.loads(rows[episode]["environment_config"])
            )
            start = 0
            if initialization_prefix is not None:
                stop = min(initialization_prefix, len(recorded_actions) - 1)
                initialized = None
                for physical_action in recorded_actions[: stop + 1]:
                    initialized = env.step_physical(physical_action)
                assert initialized is not None
                observation = initialized.actor_observation
                start = stop + 1
                assert env.reward is not None
                env.reward.reset()
            observations: list[np.ndarray] = []
            raw_actions: list[np.ndarray] = []
            recorded_raw_actions: list[np.ndarray] = []
            executed_actions: list[np.ndarray] = []
            success = False
            max_lift = -float("inf")
            max_grasp_quality = 0.0
            target_reward_sum = 0.0
            penalty_sum = 0.0
            terminal_adjustment_sum = 0.0
            max_target_reward = -float("inf")
            reward_steps = 0
            episode_active = True
            audited_terms = (
                "reach",
                "pregrasp",
                "contact",
                "grasp_quality",
                "finger",
                "xy",
                "lift",
                "stable",
                "hold",
                "progress",
                "progress_bonus",
                "approach_penalty",
                "table_penalty",
                "action_rate_penalty",
                "joint_limit_penalty",
            )
            reward_term_sums = {name: 0.0 for name in audited_terms}
            rollout_rng = np.random.default_rng(10_000 + episode)
            teacher_block_remaining = 0
            use_teacher_block = False
            for physical_action in actions[start:]:
                recorded_raw_action = transform.encode(physical_action)
                observations.append(observation.copy())
                recorded_raw_actions.append(recorded_raw_action)
                if actor is not None or teacher is not None:
                    observation_tensor = torch.as_tensor(
                        observation[None], device="cuda:0"
                    )
                if teacher is None:
                    expert_raw_action = recorded_raw_action
                else:
                    with torch.no_grad():
                        expert_raw_action = (
                            teacher(
                                TensorDict(
                                    {"actor": observation_tensor},
                                    batch_size=[1],
                                    device="cuda:0",
                                ),
                                stochastic_output=False,
                            )[0]
                            .cpu()
                            .numpy()
                        )
                raw_actions.append(expert_raw_action)
                if actor is None:
                    rollout_action = expert_raw_action
                else:
                    with torch.no_grad():
                        rollout_action = (
                            actor(
                                TensorDict(
                                    {"actor": observation_tensor},
                                    batch_size=[1],
                                    device="cuda:0",
                                ),
                                stochastic_output=False,
                            )[0]
                            .cpu()
                            .numpy()
                        )
                    if teacher is not None and teacher_rollout_blend > 0.0:
                        rollout_action = (
                            (1.0 - teacher_rollout_blend) * rollout_action
                            + teacher_rollout_blend * expert_raw_action
                        )
                    elif teacher is not None and teacher_rollout_probability > 0.0:
                        if teacher_block_remaining == 0:
                            use_teacher_block = bool(
                                rollout_rng.random() < teacher_rollout_probability
                            )
                            teacher_block_remaining = 8
                        teacher_block_remaining -= 1
                        if use_teacher_block:
                            rollout_action = expert_raw_action
                step = env.step_raw(rollout_action)
                executed_actions.append(env.previous_physical_action.copy())
                observation = step.actor_observation
                success |= step.terms.success
                max_lift = max(max_lift, step.terms.lift_height)
                max_grasp_quality = max(
                    max_grasp_quality, step.terms.grasp_quality
                )
                # Keep collecting post-terminal hold examples for BC, but audit
                # the reward only through the first terminal transition.  A
                # standalone replay is not auto-reset like the vectorized PPO
                # environment, so accumulating beyond ``done`` would count the
                # success bonus repeatedly.
                if episode_active:
                    target_reward_sum += step.terms.target_reward
                    penalty_sum += step.terms.penalty
                    terminal_adjustment_sum += step.terms.terminal_adjustment
                    max_target_reward = max(
                        max_target_reward, step.terms.target_reward
                    )
                    for name in audited_terms:
                        reward_term_sums[name] += float(
                            getattr(step.terms, name)
                        )
                    reward_steps += 1
                    episode_active = not step.done

            observation_array = np.stack(observations).astype(np.float32)
            raw_array = np.stack(raw_actions).astype(np.float32)
            recorded_raw_array = np.stack(recorded_raw_actions).astype(np.float32)
            executed_array = np.stack(executed_actions).astype(np.float32)
            num_samples = len(actions) - start
            if observation_array.shape != (num_samples, ACTOR_OBS_DIM):
                raise RuntimeError(
                    f"Episode {episode}: invalid observation shape {observation_array.shape}"
                )
            if raw_array.shape != (num_samples, ACTION_DIM):
                raise RuntimeError(
                    f"Episode {episode}: invalid action shape {raw_array.shape}"
                )
            phase = np.arange(start, len(actions), dtype=np.float32) / max(
                len(actions) - 1, 1
            )
            sample_weights = 1.0 + 2.0 * phase**2
            np.savez_compressed(
                output / f"episode_{episode:06d}.npz",
                observations=observation_array,
                raw_actions=raw_array,
                sample_weights=sample_weights,
            )
            reports.append(
                {
                    "episode": episode,
                    "frames": num_samples,
                    "initialization_prefix": initialization_prefix,
                    "success": bool(success),
                    "max_lift": float(max_lift),
                    "max_grasp_quality": float(max_grasp_quality),
                    "target_reward_sum": float(target_reward_sum),
                    "penalty_sum": float(penalty_sum),
                    "terminal_adjustment_sum": float(terminal_adjustment_sum),
                    "task_only_return_w002": float(
                        0.02 * (target_reward_sum - penalty_sum)
                        + terminal_adjustment_sum
                    ),
                    "reward_steps": reward_steps,
                    "reward_term_sums": reward_term_sums,
                    "mean_target_reward": float(target_reward_sum / reward_steps),
                    "max_target_reward": float(max_target_reward),
                    "mean_slew_error": float(
                        np.mean(np.abs(executed_array - actions[start:]))
                    ),
                    "mean_teacher_l1_to_recorded": float(
                        np.mean(np.abs(raw_array - recorded_raw_array))
                    ),
                    "finite": bool(
                        np.isfinite(observation_array).all()
                        and np.isfinite(raw_array).all()
                    ),
                }
            )
    finally:
        env.close()
    return reports


def prepare_bc_dataset(
    dataset_dir: str | Path,
    processed_dir: str | Path,
    num_workers: int = 7,
) -> Path:
    dataset = Path(dataset_dir).resolve()
    processed = Path(processed_dir).resolve()
    output = processed / "bc"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((processed / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    episodes = sorted(
        episode for split in manifest["splits"].values() for episode in split
    )
    chunks = [
        list(map(int, chunk))
        for chunk in np.array_split(episodes, min(num_workers, len(episodes)))
        if len(chunk)
    ]
    args = [
        (
            str(dataset),
            str(output),
            rows,
            chunk,
            str(processed / "action_transform.npz"),
            worker % 7,
            None,
            None,
        )
        for worker, chunk in enumerate(chunks)
    ]
    if len(args) == 1:
        reports = _replay_bc_chunk(*args[0])
    else:
        with mp.get_context("spawn").Pool(len(args)) as pool:
            reports = [item for group in pool.starmap(_replay_bc_chunk, args) for item in group]
    reports.sort(key=lambda row: row["episode"])
    summary = {
        "dataset": str(dataset),
        "num_episodes": len(reports),
        "num_frames": int(sum(row["frames"] for row in reports)),
        "replay_success_rate": float(np.mean([row["success"] for row in reports])),
        "mean_task_only_return_w002": float(
            np.mean([row["task_only_return_w002"] for row in reports])
        ),
        "mean_target_reward": float(
            np.mean([row["mean_target_reward"] for row in reports])
        ),
        "mean_max_target_reward": float(
            np.mean([row["max_target_reward"] for row in reports])
        ),
        "mean_slew_error": float(np.mean([row["mean_slew_error"] for row in reports])),
        "all_finite": bool(all(row["finite"] for row in reports)),
        "reports": reports,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    return output


def collect_dagger_dataset(
    dataset_dir: str | Path,
    processed_dir: str | Path,
    checkpoint: str | Path,
    round_index: int,
    num_workers: int = 7,
    initialization_prefix: int | None = None,
    teacher_checkpoint: str | Path | None = None,
    teacher_rollout_blend: float = 0.0,
    teacher_rollout_probability: float = 0.0,
) -> Path:
    dataset = Path(dataset_dir).resolve()
    processed = Path(processed_dir).resolve()
    output = processed / f"bc_dagger_{round_index:03d}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((processed / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    episodes = sorted(
        episode for split in manifest["splits"].values() for episode in split
    )
    chunks = [
        list(map(int, chunk))
        for chunk in np.array_split(episodes, min(num_workers, len(episodes)))
        if len(chunk)
    ]
    args = [
        (
            str(dataset),
            str(output),
            rows,
            chunk,
            str(processed / "action_transform.npz"),
            worker % 7,
            str(Path(checkpoint).resolve()),
            initialization_prefix,
            DEFAULT_TASK_REWARD_PROFILE,
            (
                str(Path(teacher_checkpoint).resolve())
                if teacher_checkpoint is not None
                else None
            ),
            teacher_rollout_blend,
            teacher_rollout_probability,
        )
        for worker, chunk in enumerate(chunks)
    ]
    if len(args) == 1:
        reports = _replay_bc_chunk(*args[0])
    else:
        with mp.get_context("spawn").Pool(len(args)) as pool:
            reports = [
                item for group in pool.starmap(_replay_bc_chunk, args) for item in group
            ]
    reports.sort(key=lambda row: row["episode"])
    summary = {
        "dataset": str(dataset),
        "rollout_checkpoint": str(Path(checkpoint).resolve()),
        "teacher_checkpoint": (
            str(Path(teacher_checkpoint).resolve())
            if teacher_checkpoint is not None
            else None
        ),
        "teacher_rollout_blend": teacher_rollout_blend,
        "teacher_rollout_probability": teacher_rollout_probability,
        "round": round_index,
        "initialization_prefix": initialization_prefix,
        "num_episodes": len(reports),
        "num_frames": int(sum(row["frames"] for row in reports)),
        "rollout_success_rate": float(
            np.mean([row["success"] for row in reports])
        ),
        "mean_slew_error_to_expert": float(
            np.mean([row["mean_slew_error"] for row in reports])
        ),
        "mean_teacher_l1_to_recorded": float(
            np.mean([row["mean_teacher_l1_to_recorded"] for row in reports])
        ),
        "all_finite": bool(all(row["finite"] for row in reports)),
        "reports": reports,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    return output


class BcDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        sources: tuple[str, ...] = ("bc",),
        reference_conditioning: bool = False,
    ):
        root = Path(processed_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        observations, actions, weights = [], [], []
        for source_name in sources:
            source = root / source_name
            if not source.is_dir():
                raise FileNotFoundError(f"BC source does not exist: {source}")
            for episode in manifest["splits"][split]:
                with np.load(
                    source / f"episode_{episode:06d}.npz", allow_pickle=False
                ) as data:
                    episode_observations = data["observations"].astype(np.float32)
                    episode_actions = data["raw_actions"].astype(np.float32)
                    if reference_conditioning:
                        episode_observations = np.stack(
                            [
                                augment_reference_observation(
                                    observation,
                                    data["observations"],
                                    episode_actions,
                                    index,
                                )
                                for index, observation in enumerate(
                                    episode_observations
                                )
                            ]
                        )
                    observations.append(episode_observations)
                    actions.append(episode_actions)
                    weights.append(data["sample_weights"].astype(np.float32))
        self.observations = torch.from_numpy(np.concatenate(observations))
        self.actions = torch.from_numpy(np.concatenate(actions))
        original_weights = np.concatenate(weights)
        phase_squared = np.clip((original_weights - 1.0) / 2.0, 0.0, 1.0)
        # Contact and lift occupy only the final part of each demonstration.
        self.weights = torch.from_numpy(
            (1.0 + 9.0 * phase_squared**2).astype(np.float32)
        )

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index], self.weights[index]


class BcSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Episode-preserving version of the replay dataset for recurrent BC."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        sources: tuple[str, ...] = ("bc",),
        reference_conditioning: bool = False,
    ) -> None:
        root = Path(processed_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        self.episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for source_name in sources:
            source = root / source_name
            if not source.is_dir():
                raise FileNotFoundError(f"BC source does not exist: {source}")
            for episode in manifest["splits"][split]:
                with np.load(
                    source / f"episode_{episode:06d}.npz", allow_pickle=False
                ) as data:
                    observations = data["observations"].astype(np.float32)
                    actions = data["raw_actions"].astype(np.float32)
                    if reference_conditioning:
                        observations = np.stack(
                            [
                                augment_reference_observation(
                                    observation,
                                    data["observations"],
                                    actions,
                                    index,
                                )
                                for index, observation in enumerate(observations)
                            ]
                        )
                    original_weights = data["sample_weights"].astype(np.float32)
                    phase_squared = np.clip(
                        (original_weights - 1.0) / 2.0, 0.0, 1.0
                    )
                    weights = (1.0 + 9.0 * phase_squared**2).astype(np.float32)
                    self.episodes.append(
                        (
                            torch.from_numpy(observations),
                            torch.from_numpy(actions),
                            torch.from_numpy(weights),
                        )
                    )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.episodes[index]


def _pad_bc_sequences(batch):
    lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    observations = torch.nn.utils.rnn.pad_sequence(
        [item[0] for item in batch]
    )
    actions = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch])
    weights = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch])
    mask = torch.arange(len(observations))[:, None] < lengths[None, :]
    return observations, actions, weights, mask


@dataclass
class BcTrainConfig:
    epochs: int = 500
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    patience: int = 100
    seed: int = 42
    device: str = "cuda:0"
    initialize_checkpoint: str | None = None
    sources: tuple[str, ...] = ("bc",)
    reference_conditioning: bool = False
    recurrent: bool = False
    sequence_batch_size: int = 8
    rnn_hidden_dim: int = 256
    right_hand_weight: float = 3.0
    right_arm_weight: float = 2.0


def _action_dimension_weights(
    device: torch.device,
    right_hand_weight: float = 3.0,
    right_arm_weight: float = 2.0,
) -> torch.Tensor:
    weights = torch.ones(ACTION_DIM, device=device)
    weights[7:14] = right_hand_weight
    weights[21:28] = right_arm_weight
    return weights


@torch.no_grad()
def _validate(
    actor,
    loader: DataLoader,
    device: torch.device,
    right_hand_weight: float = 3.0,
    right_arm_weight: float = 2.0,
) -> float:
    actor.eval()
    total = 0.0
    count = 0
    for observations, actions, weights in loader:
        observations = observations.to(device)
        actions = actions.to(device)
        weights = weights.to(device)
        prediction = actor(
            TensorDict(
                {"actor": observations},
                batch_size=[len(observations)],
                device=device,
            ),
            stochastic_output=False,
        )
        dimension_weights = _action_dimension_weights(
            device, right_hand_weight, right_arm_weight
        )
        element_loss = F.smooth_l1_loss(prediction, actions, reduction="none")
        loss = (element_loss * dimension_weights).sum(-1) / dimension_weights.sum()
        total += float((loss * weights).sum())
        count += len(observations)
    return total / count


def _recurrent_predictions(actor, observations: torch.Tensor) -> torch.Tensor:
    """Efficient teacher-forced GRU pass over a padded ``[T,B,D]`` batch."""
    latent = actor.obs_normalizer(observations)
    latent, _ = actor.rnn.rnn(latent)
    output = actor.mlp(latent)
    return actor.distribution.deterministic_output(output)


@torch.no_grad()
def _validate_recurrent(
    actor,
    loader: DataLoader,
    device: torch.device,
    right_hand_weight: float = 3.0,
    right_arm_weight: float = 2.0,
) -> float:
    actor.eval()
    total = 0.0
    denominator = 0.0
    dimension_weights = _action_dimension_weights(
        device, right_hand_weight, right_arm_weight
    )
    for observations, actions, weights, mask in loader:
        observations = observations.to(device)
        actions = actions.to(device)
        weights = weights.to(device)
        mask = mask.to(device)
        prediction = _recurrent_predictions(actor, observations)
        element_loss = F.smooth_l1_loss(prediction, actions, reduction="none")
        per_step = (
            element_loss * dimension_weights
        ).sum(-1) / dimension_weights.sum()
        valid_weights = weights * mask
        total += float((per_step * valid_weights).sum())
        denominator += float(valid_weights.sum())
    return total / denominator


def train_bc_actor(
    processed_dir: str | Path,
    output_dir: str | Path,
    config: BcTrainConfig | None = None,
) -> Path:
    config = config or BcTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))

    train = BcDataset(
        processed_dir,
        "train",
        config.sources,
        reference_conditioning=config.reference_conditioning,
    )
    validation = BcDataset(
        processed_dir,
        "val",
        config.sources,
        reference_conditioning=config.reference_conditioning,
    )
    if config.recurrent:
        train_sequences = BcSequenceDataset(
            processed_dir,
            "train",
            config.sources,
            reference_conditioning=config.reference_conditioning,
        )
        validation_sequences = BcSequenceDataset(
            processed_dir,
            "val",
            config.sources,
            reference_conditioning=config.reference_conditioning,
        )
        train_loader = DataLoader(
            train_sequences,
            batch_size=config.sequence_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_pad_bc_sequences,
        )
        validation_loader = DataLoader(
            validation_sequences,
            batch_size=config.sequence_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_pad_bc_sequences,
        )
    else:
        train_loader = DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        validation_loader = DataLoader(
            validation,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
    observation_dim = (
        REFERENCE_ACTOR_OBS_DIM
        if config.reference_conditioning
        else ACTOR_OBS_DIM
    )
    actor = make_actor(
        device,
        observation_dim=observation_dim,
        recurrent=config.recurrent,
        rnn_hidden_dim=config.rnn_hidden_dim,
    )
    if config.initialize_checkpoint:
        checkpoint = torch.load(
            config.initialize_checkpoint, map_location=device, weights_only=False
        )
        actor.load_state_dict(checkpoint["actor_state_dict"])
    else:
        mean = train.observations.mean(0, keepdim=True).to(device)
        variance = train.observations.var(0, unbiased=False, keepdim=True).to(device)
        with torch.no_grad():
            actor.obs_normalizer._mean.copy_(mean)
            actor.obs_normalizer._var.copy_(variance)
            actor.obs_normalizer._std.copy_(torch.sqrt(variance))
            actor.obs_normalizer.count.fill_(len(train))
    # Fine-tuning must not silently change the policy just by replacing its
    # input statistics with an on-policy deviation dataset. Use fixed dataset
    # statistics from scratch and preserve checkpoint statistics on restart.
    if hasattr(actor.obs_normalizer, "until"):
        actor.obs_normalizer.until = int(actor.obs_normalizer.count.item())

    optimizer = torch.optim.AdamW(
        (parameter for parameter in actor.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best = float("inf")
    stale = 0
    history = []
    best_path = output / "best.pt"
    dimension_weights = _action_dimension_weights(
        device, config.right_hand_weight, config.right_arm_weight
    )
    for epoch in range(config.epochs):
        actor.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            if config.recurrent:
                observations, actions, weights, mask = batch
                observations = observations.to(device)
                actions = actions.to(device)
                weights = weights.to(device)
                mask = mask.to(device)
                prediction = _recurrent_predictions(actor, observations)
            else:
                observations, actions, weights = batch
                observations = observations.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                prediction = actor(
                    TensorDict(
                        {"actor": observations},
                        batch_size=[len(observations)],
                        device=device,
                    ),
                    stochastic_output=False,
                )
                mask = torch.ones_like(weights, dtype=torch.bool)
            element_loss = F.smooth_l1_loss(
                prediction, actions, reduction="none"
            )
            per_sample = (
                element_loss * dimension_weights
            ).sum(-1) / dimension_weights.sum()
            valid_weights = weights * mask
            loss = (per_sample * valid_weights).sum() / valid_weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            valid_count = int(mask.sum())
            total += float(loss.detach()) * valid_count
            count += valid_count
        val = (
            _validate_recurrent(
                actor,
                validation_loader,
                device,
                config.right_hand_weight,
                config.right_arm_weight,
            )
            if config.recurrent
            else _validate(
                actor,
                validation_loader,
                device,
                config.right_hand_weight,
                config.right_arm_weight,
            )
        )
        row = {"epoch": epoch, "train": total / count, "val": val}
        history.append(row)
        if epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        payload = {
            "actor_state_dict": actor.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "val_loss": val,
        }
        torch.save({**payload, "optimizer_state_dict": optimizer.state_dict()}, output / "latest.pt")
        if val < best:
            best = val
            stale = 0
            torch.save(payload, best_path)
        else:
            stale += 1
        if stale >= config.patience:
            break
    (output / "history.json").write_text(json.dumps(history, indent=2))
    return best_path
