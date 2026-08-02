"""Numerically robust RSL-RL models for contact-rich manipulation."""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories

from simple.grasp_rl.schema import (
    ACTION_DIM,
    REFERENCE_ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_V2_DIM,
    base_observation_dim,
)

V2_RESIDUAL_LAST_ACTIVE_STAGE = 4
V2_HANDOVER_FAMILY_INDEX = 2


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
    the generated trajectory.  For v2 multi-stage tasks, PPO correction is
    restricted to the right arm/hand through placement; the audited plan owns
    locomotion and release/settle.  Keeping the grasp residual active during
    transport prevents a delayed policy stage from inheriting a prematurely
    opening replay command.  The distribution is defined over the *complete
    command* (not over a separately executed residual), so rollout actions and
    PPO log-probabilities remain consistent.

    During the current replay ablation the proposal comes from the recorded
    trajectory.  In the final system this slot must be populated by the SMP
    trajectory generator; it is not an object ground-truth state.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.obs_dim not in (REFERENCE_ACTOR_OBS_DIM, REFERENCE_ACTOR_OBS_V2_DIM):
            raise ValueError(
                "PlanConditionedMLPModel requires a supported plan-conditioned "
                f"observation, got {self.obs_dim}"
            )
        self.base_observation_dim = base_observation_dim(self.obs_dim)
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
        proposal = raw[..., self.base_observation_dim : self.base_observation_dim + ACTION_DIM]
        latent = self.get_latent(obs, None, hidden_state)
        correction = self.mlp(latent)
        if self.obs_dim == REFERENCE_ACTOR_OBS_V2_DIM:
            # V2 task-context stage one-hot is at base-state indices 322:330.
            # Most pick tasks use stages 0..4 for approach through place and
            # stage 5 for release/settle.  Handover is shorter: its stage 4 is
            # already release/settle, so exclude that family/stage pair too.
            # The environment applies the same mask after sampling, so
            # exploration noise cannot bypass it.
            active = raw[
                ..., 322 : 323 + V2_RESIDUAL_LAST_ACTIVE_STAGE
            ].sum(-1, keepdim=True).clamp(0.0, 1.0)
            handover_release = (
                raw[..., 316 + V2_HANDOVER_FAMILY_INDEX : 317 + V2_HANDOVER_FAMILY_INDEX]
                * raw[..., 326:327]
            )
            active = (active - handover_release).clamp(0.0, 1.0)
            correction_mask = torch.zeros_like(correction)
            correction_mask[..., 7:14] = active
            correction_mask[..., 21:28] = active
            correction = correction * correction_mask
        complete_command = proposal + correction
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(complete_command)
                return self.distribution.sample()
            return self.distribution.deterministic_output(complete_command)
        return complete_command
