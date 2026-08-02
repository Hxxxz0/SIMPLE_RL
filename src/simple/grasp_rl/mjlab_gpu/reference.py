"""Packed CUDA replay references with clean truth and noisy policy views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.config import ReferenceNoiseConfig
from simple.grasp_rl.mjlab_gpu.reference_noise import apply_reference_noise
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    ACTOR_OBS_V2_DIM,
    REFERENCE_CONTEXT_DIM,
    REFERENCE_CONTEXT_V2_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_FRAME_V2_DIM,
    REFERENCE_FUTURE_OFFSETS,
)

OBJECT_POS_BODY = slice(132, 135)
DESCRIPTOR_INDICES = tuple(range(132, 141)) + tuple(range(147, 165))
CONTACT_FORCE = slice(165, 189)
V2_HANDS = slice(132, 162)
V2_ENTITIES = slice(162, 219)
V2_PRIMARY_POS = slice(163, 166)
V2_AUXILIARY_POS = slice(201, 204)
V2_PREDICATES = slice(300, 308)
V2_ARTICULATION = slice(308, 316)
V2_STAGE = slice(322, 330)
V2_DESCRIPTOR_INDICES = (
    tuple(range(V2_HANDS.start, V2_HANDS.stop))
    + tuple(range(V2_ENTITIES.start, V2_ENTITIES.stop))
    + tuple(range(V2_ARTICULATION.start, V2_ARTICULATION.stop))
)


def _file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.name).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class GpuReferenceLibrary:
    """Pack a v1/v2 replay split once, then perform all tracking on CUDA."""

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        num_envs: int,
        device: str,
        source: str = "bc",
        splits: tuple[str, ...] = ("train", "val", "test"),
        target_x_arm_gains: tuple[float, float] = (0.0, 0.0),
    ):
        self.root = Path(processed_dir).resolve()
        self.device = device
        self.num_envs = int(num_envs)
        self.source = source
        self.target_x_arm_gains = tuple(float(value) for value in target_x_arm_gains)
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        episode_ids = sorted(
            {int(episode) for split in splits for episode in manifest["splits"][split]}
        )
        if not episode_ids:
            raise ValueError("Reference library selected no episodes")
        episode_paths = [
            self.root / source / f"episode_{episode:06d}.npz" for episode in episode_ids
        ]
        observations = []
        actions = []
        observation_dim: int | None = None
        for episode, path in zip(episode_ids, episode_paths, strict=True):
            with np.load(path, allow_pickle=False) as saved:
                episode_observations = saved["observations"].astype(np.float32)
                episode_actions = saved["raw_actions"].astype(np.float32)
            if episode_observations.ndim != 2 or episode_observations.shape[1] not in (
                ACTOR_OBS_DIM,
                ACTOR_OBS_V2_DIM,
            ):
                raise ValueError(f"Episode {episode} has invalid observations")
            if observation_dim is None:
                observation_dim = int(episode_observations.shape[1])
            elif episode_observations.shape[1] != observation_dim:
                raise ValueError("Reference episodes mix incompatible observation schemas")
            if episode_actions.shape != (len(episode_observations), ACTION_DIM):
                raise ValueError(f"Episode {episode} has invalid actions")
            observations.append(episode_observations)
            actions.append(episode_actions)

        assert observation_dim is not None
        self.observation_dim = observation_dim
        if any(self.target_x_arm_gains) and observation_dim != ACTOR_OBS_V2_DIM:
            raise ValueError("target-X reference retargeting requires v2 observations")
        self.context_dim = (
            REFERENCE_CONTEXT_V2_DIM
            if observation_dim == ACTOR_OBS_V2_DIM
            else REFERENCE_CONTEXT_DIM
        )
        lengths = np.asarray([len(item) for item in actions], dtype=np.int64)
        max_length = int(lengths.max())
        packed_observations = np.zeros(
            (len(episode_ids), max_length, observation_dim), dtype=np.float32
        )
        packed_actions = np.zeros(
            (len(episode_ids), max_length, ACTION_DIM), dtype=np.float32
        )
        for row, (episode_observations, episode_actions) in enumerate(
            zip(observations, actions, strict=True)
        ):
            length = len(episode_actions)
            packed_observations[row, :length] = episode_observations
            packed_actions[row, :length] = episode_actions
            packed_observations[row, length:] = episode_observations[-1]
            packed_actions[row, length:] = episode_actions[-1]

        descriptor_indices = np.asarray(
            V2_DESCRIPTOR_INDICES
            if observation_dim == ACTOR_OBS_V2_DIM
            else DESCRIPTOR_INDICES,
            dtype=np.int64,
        )
        descriptors = packed_observations[:, 0, descriptor_indices]
        descriptor_mean = descriptors.mean(axis=0)
        descriptor_std = descriptors.std(axis=0).clip(1e-3)
        self.episode_ids = torch.tensor(episode_ids, dtype=torch.long, device=device)
        self.lengths = torch.from_numpy(lengths).to(device)
        self.observations = torch.from_numpy(packed_observations).to(device)
        self.actions = torch.from_numpy(packed_actions).to(device)
        self.descriptor_indices = torch.from_numpy(descriptor_indices).to(device)
        self.descriptor_mean = torch.from_numpy(descriptor_mean).to(device)
        self.descriptor_std = torch.from_numpy(descriptor_std).to(device)
        self.normalized_descriptors = (
            self.observations[:, 0, self.descriptor_indices] - self.descriptor_mean
        ) / self.descriptor_std
        self.episode_rows = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.indices = torch.zeros_like(self.episode_rows)
        self.reference_object_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=device
        )
        self.future_offsets = torch.tensor(
            REFERENCE_FUTURE_OFFSETS, dtype=torch.long, device=device
        )
        self.data_sha256 = _file_digest([manifest_path, *episode_paths])
        self._generation = 0
        self._cached_generation = -1
        self._cached_policy_context: torch.Tensor | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "source": self.source,
            "episodes": self.episode_ids.tolist(),
            "data_sha256": self.data_sha256,
            "observation_dim": self.observation_dim,
            "target_x_arm_gains": list(self.target_x_arm_gains),
        }

    def _retarget_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Shift the proposal using the observed scene/reference X offset."""

        if actions.shape[0] != self.num_envs or actions.shape[-1] != ACTION_DIM:
            raise ValueError("Retarget actions have an incompatible shape")
        shoulder_gain, elbow_gain = self.target_x_arm_gains
        if shoulder_gain == 0.0 and elbow_gain == 0.0:
            return actions
        offset_x = self.reference_object_offset[:, 0]
        for _ in range(actions.ndim - 2):
            offset_x = offset_x.unsqueeze(-1)
        result = actions.clone()
        result[..., 21].add_(offset_x, alpha=shoulder_gain)
        result[..., 24].add_(offset_x, alpha=elbow_gain)
        return result.clamp(-1.0, 1.0)

    def rows_for_episode(self, episode: int, count: int) -> torch.Tensor:
        matches = (self.episode_ids == int(episode)).nonzero(as_tuple=False).flatten()
        if len(matches) != 1:
            raise ValueError(f"Reference episode {episode} is unavailable or duplicated")
        return matches[0].expand(count)

    def reset(
        self,
        observation: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        *,
        episode_rows: torch.Tensor | None = None,
        start_indices: torch.Tensor | None = None,
        rank: int = 0,
    ) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        if observation.shape != (len(ids), self.observation_dim):
            raise ValueError("Reference reset observation has the wrong shape")
        if episode_rows is None:
            query = (
                observation[:, self.descriptor_indices] - self.descriptor_mean
            ) / self.descriptor_std
            distance = (
                (self.normalized_descriptors[None] - query[:, None])
                .square()
                .sum(dim=-1)
            )
            if rank < 0 or rank >= len(self.episode_ids):
                raise ValueError("Reference rank is out of range")
            rows = torch.topk(
                distance, k=rank + 1, dim=-1, largest=False, sorted=True
            ).indices[:, rank]
        else:
            rows = episode_rows.to(device=self.device, dtype=torch.long)
            if rows.shape != (len(ids),):
                raise ValueError("episode_rows has the wrong shape")
        starts = (
            torch.zeros(len(ids), dtype=torch.long, device=self.device)
            if start_indices is None
            else start_indices.to(device=self.device, dtype=torch.long)
        )
        starts = torch.minimum(starts.clamp_min(0), self.lengths[rows] - 1)
        self.episode_rows[ids] = rows
        self.indices[ids] = starts
        position_slice = (
            V2_PRIMARY_POS
            if self.observation_dim == ACTOR_OBS_V2_DIM
            else OBJECT_POS_BODY
        )
        initial_reference = self.observations[rows, starts, position_slice]
        self.reference_object_offset[ids] = (
            observation[:, position_slice] - initial_reference
        )
        self._generation += 1
        self._cached_policy_context = None

    def _contact_label(self, observation: torch.Tensor) -> torch.Tensor:
        if self.observation_dim == ACTOR_OBS_V2_DIM:
            return (observation[..., V2_PREDICATES.start + 1] > 0.5).to(
                torch.float32
            )
        forces = observation[..., CONTACT_FORCE].reshape(*observation.shape[:-1], 8, 3)
        magnitude = forces.norm(dim=-1)
        thumb = magnitude[..., 1:4].amax(dim=-1) > 2.0
        support = magnitude[..., 4:8].amax(dim=-1) > 2.0
        return (thumb & support).to(torch.float32)

    def clean_context(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape != (self.num_envs, self.observation_dim):
            raise ValueError("Reference context observation has the wrong shape")
        rows = self.episode_rows
        future = self.indices[:, None] + self.future_offsets
        future = torch.minimum(future, self.lengths[rows, None] - 1)
        reference_observations = self.observations[rows[:, None], future]
        reference_actions = self.actions[rows[:, None], future]
        if self.observation_dim == ACTOR_OBS_V2_DIM:
            return self._clean_context_v2(
                observation, rows, future, reference_observations, reference_actions
            )
        reference_object = (
            reference_observations[..., OBJECT_POS_BODY]
            + self.reference_object_offset[:, None]
        )
        object_delta = reference_object - observation[:, None, OBJECT_POS_BODY]
        contact = self._contact_label(reference_observations)[..., None]
        frames = torch.cat((reference_actions, object_delta, contact), dim=-1)
        if frames.shape != (
            self.num_envs,
            len(REFERENCE_FUTURE_OFFSETS),
            REFERENCE_FRAME_DIM,
        ):
            raise RuntimeError("Reference frame layout mismatch")
        phase = self.indices.float() / (self.lengths[rows] - 1).clamp_min(1)
        context = torch.cat((frames.flatten(1), phase[:, None]), dim=-1)
        if context.shape != (self.num_envs, REFERENCE_CONTEXT_DIM):
            raise RuntimeError("Reference context layout mismatch")
        return context

    def _clean_context_v2(
        self,
        observation: torch.Tensor,
        rows: torch.Tensor,
        future: torch.Tensor,
        reference_observations: torch.Tensor,
        reference_actions: torch.Tensor,
    ) -> torch.Tensor:
        del future
        reference_actions = self._retarget_actions(reference_actions)
        frame_offsets = self.future_offsets[None].expand(self.num_envs, -1)
        remaining = self.lengths[rows, None] - 1 - self.indices[:, None]
        dt = torch.minimum(frame_offsets, remaining.clamp_min(0)).float() / 50.0
        root_velocity = reference_observations[..., 89:95]
        root_delta = torch.stack(
            (root_velocity[..., 0] * dt, root_velocity[..., 1] * dt,
             root_velocity[..., 5] * dt),
            dim=-1,
        )
        primary = (
            reference_observations[..., V2_PRIMARY_POS]
            + self.reference_object_offset[:, None]
        )
        primary_delta = primary - observation[:, None, V2_PRIMARY_POS]
        auxiliary_delta = (
            reference_observations[..., V2_AUXILIARY_POS]
            - observation[:, None, V2_AUXILIARY_POS]
        )
        interactions = reference_observations[..., V2_PREDICATES.start:V2_PREDICATES.start + 4]
        articulation = reference_observations[..., V2_ARTICULATION]
        present = articulation[..., (0, 4)] > 0.5
        target_delta = articulation[..., (3, 7)].abs()
        minimum = torch.where(
            present,
            target_delta,
            torch.ones_like(target_delta),
        ).amin(dim=-1)
        articulation_progress = (1.0 - minimum.clamp(max=1.0))[..., None]
        stage = reference_observations[..., V2_STAGE].argmax(dim=-1).float()[..., None] / 7.0
        frames = torch.cat(
            (
                reference_actions,
                root_delta,
                primary_delta,
                auxiliary_delta,
                interactions,
                articulation_progress,
                stage,
            ),
            dim=-1,
        )
        if frames.shape != (
            self.num_envs,
            len(REFERENCE_FUTURE_OFFSETS),
            REFERENCE_FRAME_V2_DIM,
        ):
            raise RuntimeError("V2 reference frame layout mismatch")
        phase = self.indices.float() / (self.lengths[rows] - 1).clamp_min(1)
        context = torch.cat((frames.flatten(1), phase[:, None]), dim=-1)
        if context.shape != (self.num_envs, REFERENCE_CONTEXT_V2_DIM):
            raise RuntimeError("V2 reference context layout mismatch")
        return context

    def policy_context(
        self,
        observation: torch.Tensor,
        noise: ReferenceNoiseConfig,
        *,
        training: bool,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_context(observation)
        if training and self._cached_generation == self._generation:
            assert self._cached_policy_context is not None
            return self._cached_policy_context, clean
        policy = apply_reference_noise(
            clean, noise, enabled=training, generator=generator
        )
        if training:
            self._cached_generation = self._generation
            self._cached_policy_context = policy
        return policy, clean

    def post_step_contact_label(self) -> torch.Tensor:
        rows = self.episode_rows
        following = torch.minimum(self.indices + 1, self.lengths[rows] - 1)
        return self._contact_label(self.observations[rows, following])

    def current_action(self) -> torch.Tensor:
        action = self.actions[self.episode_rows, self.indices]
        return self._retarget_actions(action)

    def advance(self) -> None:
        rows = self.episode_rows
        self.indices.copy_(torch.minimum(self.indices + 1, self.lengths[rows] - 1))
        self._generation += 1
        self._cached_policy_context = None
