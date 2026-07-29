"""GRAIL-style robot-reference conditioning and imitation reward.

The policy still emits a complete 36-D SIMPLE tracker command.  The standard
actor uses references only as training/conditioning signals.  The explicitly
named ``PlanConditionedMLPModel`` is a diagnostic upper-bound variant that
initializes this complete command from a generated plan and learns a feedback
correction.  Object tracking is intentionally absent from the reward, matching
the released GRAIL tabletop configuration where its weight is zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_DIM,
    REFERENCE_CONTEXT_DIM,
    REFERENCE_FUTURE_OFFSETS,
)


# Frozen slices from the 192-D base observation contract in state.py.
JOINT_POS = slice(0, 43)
JOINT_VEL = slice(43, 86)
ROOT_STATE = slice(86, 96)  # gravity, pelvis linear/angular velocity, height
OBJECT_POS_BODY = slice(132, 135)
OBJECT_ROT_BODY = slice(135, 141)
OBJECT_POS_HAND = slice(147, 150)
OBJECT_ROT_HAND = slice(150, 156)
TABLE_POSE_BODY = slice(156, 165)
CONTACT_FORCE = slice(165, 189)


def reference_contact_label(observation: np.ndarray) -> float:
    """Return the replay reference's thumb--support bilateral contact label."""
    forces = np.asarray(observation[CONTACT_FORCE], dtype=np.float32).reshape(8, 3)
    magnitudes = np.linalg.norm(forces, axis=-1)
    thumb = float(magnitudes[1:4].max(initial=0.0)) > 2.0
    support = float(magnitudes[4:8].max(initial=0.0)) > 2.0
    return float(thumb and support)


def build_reference_context(
    observations: np.ndarray,
    actions: np.ndarray,
    index: int,
    current_observation: np.ndarray,
) -> np.ndarray:
    """Build ten 0.1-s-spaced future commands and object/contact deltas."""
    if observations.ndim != 2 or observations.shape[1] != ACTOR_OBS_DIM:
        raise ValueError(f"Expected reference observations [T,{ACTOR_OBS_DIM}]")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected reference actions [T,{ACTION_DIM}]")
    if len(observations) != len(actions):
        raise ValueError("Reference observation/action lengths differ")
    index = int(np.clip(index, 0, len(actions) - 1))
    pieces: list[np.ndarray] = []
    current_object = np.asarray(current_observation[OBJECT_POS_BODY], dtype=np.float32)
    for offset in REFERENCE_FUTURE_OFFSETS:
        future = min(index + offset, len(actions) - 1)
        object_delta = observations[future, OBJECT_POS_BODY] - current_object
        pieces.extend(
            (
                actions[future],
                object_delta.astype(np.float32),
                np.asarray(
                    [reference_contact_label(observations[future])],
                    dtype=np.float32,
                ),
            )
        )
    phase = np.asarray([index / max(len(actions) - 1, 1)], dtype=np.float32)
    context = np.concatenate([*pieces, phase]).astype(np.float32)
    if context.shape != (REFERENCE_CONTEXT_DIM,):
        raise RuntimeError(f"Reference context has shape {context.shape}")
    return context


def augment_reference_observation(
    current_observation: np.ndarray,
    reference_observations: np.ndarray,
    reference_actions: np.ndarray,
    index: int,
) -> np.ndarray:
    augmented = np.concatenate(
        [
            np.asarray(current_observation, dtype=np.float32),
            build_reference_context(
                reference_observations,
                reference_actions,
                index,
                current_observation,
            ),
        ]
    ).astype(np.float32)
    if augmented.shape != (REFERENCE_ACTOR_OBS_DIM,):
        raise RuntimeError(f"Reference-conditioned observation has shape {augmented.shape}")
    return augmented


@dataclass(frozen=True)
class ReferenceRewardTerms:
    root: float
    joint_pose: float
    joint_velocity: float
    tracker_action: float
    total: float
    episode: int
    phase: float

    def to_dict(self) -> dict:
        return asdict(self)


class ReferenceLibrary:
    """Small replay library used for paired reference conditioning.

    Exact episode IDs are used for demonstration-scene RSI and evaluation.  A
    normal randomized reset selects the nearest replay by initial object/hand/
    table geometry, analogous to selecting a paired motion command in GRAIL.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        source: str = "bc",
        splits: tuple[str, ...] = ("train", "val", "test"),
    ) -> None:
        root = Path(processed_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        episode_ids = sorted(
            {
                int(episode)
                for split in splits
                for episode in manifest["splits"][split]
            }
        )
        if not episode_ids:
            raise ValueError("Reference library selected no episodes")
        directory = root / source
        self.observations: dict[int, np.ndarray] = {}
        self.actions: dict[int, np.ndarray] = {}
        for episode in episode_ids:
            with np.load(
                directory / f"episode_{episode:06d}.npz", allow_pickle=False
            ) as data:
                self.observations[episode] = data["observations"].astype(np.float32)
                self.actions[episode] = data["raw_actions"].astype(np.float32)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        descriptors = np.stack(
            [self._descriptor(self.observations[e][0]) for e in episode_ids]
        )
        self.descriptor_mean = descriptors.mean(axis=0)
        self.descriptor_std = descriptors.std(axis=0).clip(1e-3)
        self.normalized_descriptors = (
            descriptors - self.descriptor_mean
        ) / self.descriptor_std

    @staticmethod
    def _descriptor(observation: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                observation[OBJECT_POS_BODY],
                observation[OBJECT_ROT_BODY],
                observation[OBJECT_POS_HAND],
                observation[OBJECT_ROT_HAND],
                observation[TABLE_POSE_BODY],
            ]
        ).astype(np.float32)

    def select_episode(
        self,
        observation: np.ndarray,
        exact_episode: int | None = None,
        rank: int = 0,
    ) -> int:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if exact_episode is not None and exact_episode in self.observations:
            return exact_episode
        query = (self._descriptor(observation) - self.descriptor_mean) / self.descriptor_std
        distance = np.sum((self.normalized_descriptors - query[None]) ** 2, axis=-1)
        order = np.argsort(distance, kind="stable")
        return int(self.episode_ids[int(order[min(rank, len(order) - 1)])])


class ReferenceTracker:
    """Per-environment monotonic phase and imitation-reward state."""

    def __init__(self, library: ReferenceLibrary) -> None:
        self.library = library
        self.episode: int | None = None
        self.index = 0

    @property
    def is_ready(self) -> bool:
        return self.episode is not None

    def reset(
        self,
        observation: np.ndarray,
        exact_episode: int | None = None,
        start_index: int = 0,
        rank: int = 0,
    ) -> None:
        self.episode = self.library.select_episode(
            observation, exact_episode, rank=rank
        )
        length = len(self.library.actions[self.episode])
        self.index = int(np.clip(start_index, 0, length - 1))

    def augment(self, observation: np.ndarray) -> np.ndarray:
        if self.episode is None:
            raise RuntimeError("ReferenceTracker.reset must be called first")
        return augment_reference_observation(
            observation,
            self.library.observations[self.episode],
            self.library.actions[self.episode],
            self.index,
        )

    def post_step_contact_label(self) -> float:
        """Contact intent for the state produced by the next policy action."""
        if self.episode is None:
            raise RuntimeError("ReferenceTracker.reset must be called first")
        observations = self.library.observations[self.episode]
        following = min(self.index + 1, len(observations) - 1)
        return reference_contact_label(observations[following])

    def reward(
        self, post_observation: np.ndarray, executed_raw_action: np.ndarray
    ) -> ReferenceRewardTerms:
        """Score the executed transition against the advancing robot reference.

        Magnitudes mirror GRAIL's dominant anchor-orientation (2.5) and relative
        body-orientation (5.0) terms.  Joint pose is the available MuJoCo proxy
        for relative body orientation, and a small directly learnable complete-
        command term prevents sparse-contact exploration from erasing the motion.
        """
        if self.episode is None:
            raise RuntimeError("ReferenceTracker.reset must be called first")
        observations = self.library.observations[self.episode]
        actions = self.library.actions[self.episode]
        current = min(self.index, len(actions) - 1)
        following = min(current + 1, len(observations) - 1)
        target_observation = observations[following]

        joint_weights = np.ones(43, dtype=np.float32)
        joint_weights[22:29] = 2.0  # right arm
        joint_weights[36:43] = 3.0  # right hand
        action_weights = np.ones(ACTION_DIM, dtype=np.float32)
        action_weights[7:14] = 3.0
        action_weights[21:28] = 2.0

        pose_error = float(
            np.sum(
                joint_weights
                * (post_observation[JOINT_POS] - target_observation[JOINT_POS]) ** 2
            )
            / np.sum(joint_weights)
        )
        velocity_error = float(
            np.sum(
                joint_weights
                * (post_observation[JOINT_VEL] - target_observation[JOINT_VEL]) ** 2
            )
            / np.sum(joint_weights)
        )
        root_scale = np.asarray(
            [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1],
            dtype=np.float32,
        )
        root_error = float(
            np.mean(
                (
                    (post_observation[ROOT_STATE] - target_observation[ROOT_STATE])
                    / root_scale
                )
                ** 2
            )
        )
        action_error = float(
            np.sum(
                action_weights
                * (np.asarray(executed_raw_action) - actions[current]) ** 2
            )
            / np.sum(action_weights)
        )

        # Whole-body averages hide the exact failure observed in this task: a
        # plausible lift with a slightly mistimed right arm/hand.  GRAIL avoids
        # that dilution through relative body tracking plus a discrete hand
        # primitive.  Track the corresponding SIMPLE manipulation coordinates
        # separately and tightly while retaining the complete-command score.
        manipulation_joint_indices = np.r_[22:29, 36:43]
        manipulation_pose_error = float(
            np.mean(
                (
                    post_observation[JOINT_POS][manipulation_joint_indices]
                    - target_observation[JOINT_POS][manipulation_joint_indices]
                )
                ** 2
            )
        )
        manipulation_action = np.r_[7:14, 21:28]
        manipulation_action_error = float(
            np.mean(
                (
                    np.asarray(executed_raw_action)[manipulation_action]
                    - actions[current, manipulation_action]
                )
                ** 2
            )
        )

        root_reward = float(np.exp(-root_error))
        whole_body_pose_reward = float(np.exp(-pose_error / (0.35**2)))
        manipulation_pose_reward = float(
            np.exp(-manipulation_pose_error / (0.08**2))
        )
        pose_reward = 0.25 * whole_body_pose_reward + 0.75 * manipulation_pose_reward
        velocity_reward = float(np.exp(-velocity_error / (1.0**2)))
        whole_action_reward = float(np.exp(-action_error / (0.15**2)))
        manipulation_action_reward = float(
            np.exp(-manipulation_action_error / (0.03**2))
        )
        action_reward = 0.25 * whole_action_reward + 0.75 * manipulation_action_reward
        total = (
            2.5 * root_reward
            + 5.0 * pose_reward
            + velocity_reward
            + action_reward
        )
        phase = current / max(len(actions) - 1, 1)
        terms = ReferenceRewardTerms(
            root=root_reward,
            joint_pose=pose_reward,
            joint_velocity=velocity_reward,
            tracker_action=action_reward,
            total=float(total),
            episode=self.episode,
            phase=float(phase),
        )
        self.index = following
        return terms
