"""Exploration distributions suited to complete tracker commands."""

from __future__ import annotations

import torch
from rsl_rl.modules.distribution import GaussianDistribution

from simple.grasp_rl.schema import ACTION_SLICES


class ActionGroupedGaussianDistribution(GaussianDistribution):
    """Gaussian exploration with opt-in standard deviations per action group."""

    def __init__(
        self,
        *args,
        action_group_stds: tuple[tuple[str, float], ...] = (),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        seen: set[str] = set()
        parameter = getattr(self, "std_param", None)
        log_parameter = getattr(self, "log_std_param", None)
        with torch.no_grad():
            for group, raw_std in action_group_stds:
                if group not in ACTION_SLICES:
                    raise ValueError(f"Unknown exploration action group: {group}")
                if group in seen:
                    raise ValueError(f"Duplicate exploration action group: {group}")
                std = float(raw_std)
                if not torch.isfinite(torch.tensor(std)) or std <= 0.0:
                    raise ValueError("Action-group exploration std must be positive")
                spec = ACTION_SLICES[group]
                if parameter is not None:
                    parameter[spec.start : spec.stop] = std
                elif log_parameter is not None:
                    log_parameter[spec.start : spec.stop] = torch.log(
                        torch.tensor(std, device=log_parameter.device)
                    )
                else:
                    raise TypeError("Gaussian distribution has no std parameter")
                seen.add(group)


class TemporallyCorrelatedGaussianDistribution(ActionGroupedGaussianDistribution):
    """Legacy non-PPO distribution that holds standardized Gaussian noise.

    The per-step marginals are Gaussian, but the conditional trajectory density
    is not the product of those marginals.  The GPU PPO configuration therefore
    rejects ``hold_steps > 1`` because RSL-RL computes independent per-step
    likelihood ratios.  The class remains importable for old standalone callers
    and checkpoint/config compatibility.
    """

    def __init__(self, *args, hold_steps: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if hold_steps < 1:
            raise ValueError("hold_steps must be at least 1")
        self.hold_steps = hold_steps
        self._held_noise: torch.Tensor | None = None
        self._sample_count = 0

    def sample(self) -> torch.Tensor:
        mean = self.mean
        must_refresh = (
            self._held_noise is None
            or self._held_noise.shape != mean.shape
            or self._held_noise.device != mean.device
            or self._sample_count % self.hold_steps == 0
        )
        if must_refresh:
            self._held_noise = torch.randn_like(mean)
        self._sample_count += 1
        return mean + self.std * self._held_noise
