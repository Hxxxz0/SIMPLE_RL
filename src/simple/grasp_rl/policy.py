"""Shared construction and checkpoint loading for the 36-D PPO actor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from simple.grasp_rl.models import (
    ClippedMLPModel,
    ClippedRNNModel,
    PlanConditionedMLPModel,
)
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_DIM,
)
from simple.grasp_rl.task_spec import (
    GraspTaskSpec,
    checkpoint_task_metadata,
    get_task_spec,
    task_from_manifest,
    validate_task_metadata,
)


def make_actor(
    device: str | torch.device = "cuda:0",
    observation_dim: int = ACTOR_OBS_DIM,
    recurrent: bool = False,
    rnn_hidden_dim: int = 256,
    plan_conditioned: bool = False,
) -> MLPModel:
    device = torch.device(device)
    dummy = TensorDict(
        {"actor": torch.zeros(1, observation_dim, device=device)},
        batch_size=[1],
        device=device,
    )
    if recurrent and plan_conditioned:
        raise ValueError("Plan-conditioned recurrent actor is not implemented")
    model_class = (
        ClippedRNNModel
        if recurrent
        else PlanConditionedMLPModel
        if plan_conditioned
        else ClippedMLPModel
    )
    model_kwargs = (
        {
            "rnn_type": "gru",
            "rnn_hidden_dim": rnn_hidden_dim,
            "rnn_num_layers": 1,
        }
        if recurrent
        else {}
    )
    return model_class(
        dummy,
        {"actor": ["actor"]},
        "actor",
        ACTION_DIM,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.05,
            "std_type": "scalar",
            "learn_std": False,
        },
        **model_kwargs,
    ).to(device)


class KnnBcActor(nn.Module):
    """Non-parametric state-feedback expert built from physical BC replay.

    This actor is a diagnostic/DAgger teacher, not the deployable PPO model.
    It lets us distinguish insufficient state-action supervision from reward
    or tracker failures without introducing an object reference trajectory.
    """

    def __init__(self, data: dict, device: str | torch.device):
        super().__init__()
        self.register_buffer("observations", data["observations"].to(device))
        self.register_buffer("actions", data["actions"].to(device))
        self.register_buffer("mean", data["mean"].to(device))
        self.register_buffer("std", data["std"].to(device))
        episode_ids = data.get("episode_ids")
        step_ids = data.get("step_ids")
        self.register_buffer(
            "episode_ids",
            episode_ids.to(device) if episode_ids is not None else None,
        )
        self.register_buffer(
            "step_ids", step_ids.to(device) if step_ids is not None else None
        )
        normalized = (self.observations - self.mean) / self.std
        self.register_buffer("normalized_observations", normalized)
        self.register_buffer(
            "observation_norm_squared", (normalized * normalized).sum(-1)
        )
        self.grasp_observation_dim = ACTOR_OBS_DIM
        self.last_indices: torch.Tensor | None = None

    def reset(self) -> None:
        self.last_indices = None

    def forward(
        self, observation: TensorDict, stochastic_output: bool = False
    ) -> torch.Tensor:
        del stochastic_output
        query = (observation["actor"] - self.mean) / self.std
        distance = (
            (query * query).sum(-1, keepdim=True)
            + self.observation_norm_squared.unsqueeze(0)
            - 2.0 * query @ self.normalized_observations.T
        )
        if self.last_indices is None and self.step_ids is not None:
            # At reset, choose among trajectory starts. Searching every frame
            # can lock a standing initial state onto a similar late hold state.
            distance[:, self.step_ids != 0] = torch.inf
        elif self.last_indices is not None and self.episode_ids is not None:
            if len(self.last_indices) != len(query):
                raise ValueError("KNN batch size changed without reset")
            for batch_index, previous in enumerate(self.last_indices):
                episode = self.episode_ids[previous]
                expected_step = self.step_ids[previous] + 1
                local = (self.episode_ids == episode) & (
                    (self.step_ids - expected_step).abs() <= 4
                )
                distance[batch_index, ~local] = torch.inf
        indices = distance.argmin(-1)
        self.last_indices = indices.detach()
        return self.actions[indices]


def build_knn_actor_checkpoint(
    processed_dir: str | Path,
    output_path: str | Path,
    sources: tuple[str, ...] = ("bc",),
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Path:
    """Package replayed state-action pairs as a nearest-neighbor expert."""
    processed = Path(processed_dir)
    manifest = json.loads((processed / "manifest.json").read_text())
    task_spec = task_from_manifest(processed)
    observations, actions, episode_ids, step_ids = [], [], [], []
    for source_name in sources:
        source = processed / source_name
        if not source.is_dir():
            raise FileNotFoundError(f"BC source does not exist: {source}")
        for split in splits:
            for episode in manifest["splits"][split]:
                with np.load(
                    source / f"episode_{episode:06d}.npz", allow_pickle=False
                ) as episode_data:
                    observations.append(
                        episode_data["observations"].astype(np.float32)
                    )
                    actions.append(episode_data["raw_actions"].astype(np.float32))
                    frames = len(episode_data["observations"])
                    episode_ids.append(np.full(frames, episode, dtype=np.int64))
                    step_ids.append(np.arange(frames, dtype=np.int64))
    observation_tensor = torch.from_numpy(np.concatenate(observations))
    action_tensor = torch.from_numpy(np.concatenate(actions))
    mean = observation_tensor.mean(0)
    std = observation_tensor.std(0).clamp_min(1e-3)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": "knn_bc_v1",
            "observations": observation_tensor,
            "actions": action_tensor,
            "mean": mean,
            "std": std,
            "episode_ids": torch.from_numpy(np.concatenate(episode_ids)),
            "step_ids": torch.from_numpy(np.concatenate(step_ids)),
            "sources": sources,
            "splits": splits,
            "task_metadata": checkpoint_task_metadata(
                task_spec, processed / "action_transform.npz"
            ),
        },
        output,
    )
    return output


def load_actor(
    checkpoint: str | Path,
    device: str | torch.device = "cuda:0",
    expected_task: str | GraspTaskSpec | None = None,
    action_transform: str | Path | None = None,
) -> MLPModel | KnnBcActor:
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    if expected_task is not None:
        validate_task_metadata(
            data,
            get_task_spec(expected_task),
            checkpoint=checkpoint,
            action_transform=action_transform,
        )
    if data.get("policy_type") == "knn_bc_v1":
        return KnnBcActor(data, device).eval()
    recurrent = "rnn.rnn.weight_ih_l0" in data["actor_state_dict"]
    plan_conditioned = (
        "_plan_conditioned_actor" in data["actor_state_dict"]
    )
    if recurrent:
        observation_dim = int(
            data["actor_state_dict"]["rnn.rnn.weight_ih_l0"].shape[1]
        )
        rnn_hidden_dim = int(
            data["actor_state_dict"]["rnn.rnn.weight_hh_l0"].shape[1]
        )
    else:
        observation_dim = int(data["actor_state_dict"]["mlp.0.weight"].shape[1])
        rnn_hidden_dim = 256
    if observation_dim not in {
        ACTOR_OBS_DIM,
        ACTOR_OBS_DIM + 1,
        REFERENCE_ACTOR_OBS_DIM,
    }:
        raise ValueError(
            f"Unsupported actor observation dimension {observation_dim}"
        )
    actor = make_actor(
        device,
        observation_dim,
        recurrent=recurrent,
        rnn_hidden_dim=rnn_hidden_dim,
        plan_conditioned=plan_conditioned,
    )
    actor.load_state_dict(data["actor_state_dict"])
    actor.grasp_observation_dim = observation_dim
    actor.eval()
    return actor


def add_optional_phase(
    actor: MLPModel | KnnBcActor, observation: torch.Tensor, policy_step: int
) -> torch.Tensor:
    """Append the legacy GRAIL-style motion phase for compatible 193-D actors."""
    observation_dim = int(
        getattr(actor, "grasp_observation_dim", observation.shape[-1])
    )
    if observation_dim == observation.shape[-1]:
        return observation
    if observation_dim != observation.shape[-1] + 1:
        raise ValueError(
            f"Actor expects {observation_dim} values, got {observation.shape[-1]}"
        )
    phase = torch.full(
        (*observation.shape[:-1], 1),
        min(max(policy_step / 128.0, 0.0), 1.0),
        dtype=observation.dtype,
        device=observation.device,
    )
    return torch.cat((observation, phase), dim=-1)
