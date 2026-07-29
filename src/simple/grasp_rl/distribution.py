"""Exploration distributions suited to complete tracker commands."""

from __future__ import annotations

import torch
from rsl_rl.modules.distribution import GaussianDistribution


class TemporallyCorrelatedGaussianDistribution(GaussianDistribution):
    """Hold standardized Gaussian noise for several consecutive policy calls.

    The marginal action distribution remains the configured diagonal Gaussian,
    while short-horizon correlation avoids injecting independent 50 Hz jitter
    into contact-sensitive 36D tracker commands.
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
