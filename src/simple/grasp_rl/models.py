"""Numerically robust RSL-RL models for contact-rich manipulation."""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories

from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_DIM,
)


class ClippedEmpiricalNormalization(EmpiricalNormalization):
    """Empirical normalization with a bounded out-of-distribution response.

    Sparse contact channels can have zero variance in a finite demonstration
    split and then become non-zero after the policy makes a small mistake.  The
    stock RSL-RL normalizer consequently maps a perfectly finite force to an
    input in the thousands.  Clipping normalized observations is standard PPO
    practice and, importantly, preserves the exact state-dict layout of the
    upstream normalizer.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        until: int | None = None,
        clip: float = 10.0,
    ) -> None:
        super().__init__(shape, eps=eps, until=until)
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = super().forward(x)
        return torch.clamp(normalized, -self.clip, self.clip)


class ClippedMLPModel(MLPModel):
    """Drop-in ``MLPModel`` using clipped empirical normalization."""

    def __init__(self, *args, observation_clip: float = 10.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.obs_normalization:
            self.obs_normalizer = ClippedEmpiricalNormalization(
                self.obs_dim, clip=observation_clip
            ).to(next(self.parameters()).device)


class ClippedRNNModel(RNNModel):
    """Drop-in ``RNNModel`` using the same bounded normalization."""

    def __init__(self, *args, observation_clip: float = 10.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.obs_normalization:
            self.obs_normalizer = ClippedEmpiricalNormalization(
                self.obs_dim, clip=observation_clip
            ).to(next(self.parameters()).device)


class PlanConditionedMLPModel(ClippedMLPModel):
    """Emit a complete tracker command initialized from the proposed plan.

    The first future-reference frame contains the generator's current 36-D
    tracker command.  A zero-initialized policy therefore starts exactly on
    the generated trajectory, while PPO can still change every output
    dimension using state and target feedback.  The distribution is defined
    over the *complete command* (not over a separately executed residual), so
    rollout actions and PPO log-probabilities remain consistent.

    During the current replay ablation the proposal comes from the recorded
    trajectory.  In the final system this slot must be populated by the SMP
    trajectory generator; it is not an object ground-truth state.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.obs_dim != REFERENCE_ACTOR_OBS_DIM:
            raise ValueError(
                "PlanConditionedMLPModel requires the 593-D plan-conditioned "
                f"observation, got {self.obs_dim}"
            )
        # Persistent marker makes standalone RSL checkpoints self-describing.
        self.register_buffer("_plan_conditioned_actor", torch.ones(()))

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        proposal = raw[
            ..., ACTOR_OBS_DIM : ACTOR_OBS_DIM + ACTION_DIM
        ]
        latent = self.get_latent(obs, None, hidden_state)
        complete_command = proposal + self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(complete_command)
                return self.distribution.sample()
            return self.distribution.deterministic_output(complete_command)
        return complete_command
