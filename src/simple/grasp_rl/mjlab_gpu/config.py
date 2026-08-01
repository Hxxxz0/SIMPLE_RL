"""Versioned configuration and GPU-only safety gates for mjlab PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

MJLAB_PPO_CONFIG_VERSION = 1
MIN_LONG_TRAIN_ENVS = 2048
GPU_SENSOR_SCHEMA_VERSION = 1


def _range_pair(value: tuple[float, float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    low, high = (float(item) for item in value)
    if low > high:
        raise ValueError(f"{name} must satisfy low <= high")
    return low, high


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReferenceNoiseConfig:
    """Small independent noise applied only to policy reference inputs.

    It never modifies simulator state, reward truth, success, termination, or
    locked evaluation inputs.  Scene pose randomization is handled separately
    and must be synchronously transformed into the reference before this noise.
    """

    schema_version: int = 1
    action_std: float = 0.002
    position_std: float = 0.0025
    phase_std: float = 0.01
    future_dropout_probability: float = 0.02

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("reference noise schema_version must be 1")
        for name in ("action_std", "position_std", "phase_std"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.future_dropout_probability <= 1.0:
            raise ValueError("future_dropout_probability must be in [0, 1]")

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_sha256"] = _canonical_hash(payload)
        return payload

    def scaled(self, strength: float) -> "ReferenceNoiseConfig":
        value = min(max(float(strength), 0.0), 1.0)
        return ReferenceNoiseConfig(
            schema_version=self.schema_version,
            action_std=self.action_std * value,
            position_std=self.position_std * value,
            phase_std=self.phase_std * value,
            future_dropout_probability=self.future_dropout_probability * value,
        )


@dataclass(frozen=True)
class DomainRandomizationConfig:
    enabled: bool = True
    target_position_jitter_xy: tuple[float, float] = (0.025, 0.03)
    target_yaw_jitter: float = 0.15
    target_mass_scale: tuple[float, float] = (0.8, 1.2)
    friction_scale: tuple[float, float] = (0.7, 1.3)
    joint_damping_scale: tuple[float, float] = (0.9, 1.1)
    actuator_strength_scale: tuple[float, float] = (0.9, 1.1)
    action_delay_max_steps: int = 1
    curriculum_initial_strength: float = 0.10
    curriculum_warmup_steps: int = 0
    curriculum_ramp_steps: int = 2_400
    sync_reference_scene_transform: bool = True
    reference_noise: ReferenceNoiseConfig = field(default_factory=ReferenceNoiseConfig)

    def __post_init__(self) -> None:
        for name in (
            "target_mass_scale",
            "friction_scale",
            "joint_damping_scale",
            "actuator_strength_scale",
        ):
            low, high = _range_pair(getattr(self, name), name)
            if low <= 0.0:
                raise ValueError(f"{name} values must be positive")
        if len(self.target_position_jitter_xy) != 2 or any(
            value < 0.0 for value in self.target_position_jitter_xy
        ):
            raise ValueError("target_position_jitter_xy values must be non-negative")
        if self.target_yaw_jitter < 0.0:
            raise ValueError("target_yaw_jitter must be non-negative")
        if self.action_delay_max_steps not in (0, 1):
            raise ValueError("action_delay_max_steps must be 0 or 1")
        if not 0.0 <= self.curriculum_initial_strength <= 1.0:
            raise ValueError("curriculum_initial_strength must be in [0, 1]")
        if self.curriculum_warmup_steps < 0 or self.curriculum_ramp_steps < 1:
            raise ValueError("DR curriculum steps must be non-negative/positive")
        if self.enabled and not self.sync_reference_scene_transform:
            raise ValueError(
                "domain randomization requires sync_reference_scene_transform"
            )

    def strength(self, vector_step: int) -> float:
        if not self.enabled:
            return 0.0
        progress = (int(vector_step) - self.curriculum_warmup_steps) / float(
            self.curriculum_ramp_steps
        )
        return min(max(progress, 0.0), 1.0)

    def training_strength(self, vector_step: int) -> float:
        """Keep reset worlds decorrelated before the main DR ramp starts."""

        if not self.enabled:
            return 0.0
        return max(self.curriculum_initial_strength, self.strength(vector_step))


@dataclass(frozen=True)
class MjlabPpoConfig:
    """Resolved training settings embedded in every GPU PPO checkpoint."""

    task: str
    asset_bundle: str
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 42
    smoke_mode: bool = False
    reference_processed: str | None = None
    reference_source: str = "bc"
    reference_reward_weight: float = 0.05
    max_reference_action_deviation: float = 0.35
    full_dr_reference_reward_scale: float = 0.2
    sensor_schema_version: int = GPU_SENSOR_SCHEMA_VERSION
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    schema_version: int = MJLAB_PPO_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MJLAB_PPO_CONFIG_VERSION:
            raise ValueError(
                f"mjlab PPO schema_version must be {MJLAB_PPO_CONFIG_VERSION}"
            )
        if self.sensor_schema_version != GPU_SENSOR_SCHEMA_VERSION:
            raise ValueError(
                f"sensor_schema_version must be {GPU_SENSOR_SCHEMA_VERSION}"
            )
        if not self.task:
            raise ValueError("task must be non-empty")
        if not self.asset_bundle:
            raise ValueError("asset_bundle must be non-empty")
        if not self.reference_source:
            raise ValueError("reference_source must be non-empty")
        if self.reference_reward_weight < 0.0:
            raise ValueError("reference_reward_weight must be non-negative")
        if not 0.0 < self.max_reference_action_deviation <= 2.0:
            raise ValueError("max_reference_action_deviation must be in (0, 2]")
        if not 0.0 <= self.full_dr_reference_reward_scale <= 1.0:
            raise ValueError("full_dr_reference_reward_scale must be in [0, 1]")
        if not self.device.startswith("cuda:"):
            raise ValueError("mjlab PPO requires an explicit cuda:<index> device")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.num_envs < MIN_LONG_TRAIN_ENVS and not self.smoke_mode:
            raise ValueError(
                f"long GPU training requires at least {MIN_LONG_TRAIN_ENVS} envs; "
                "set smoke_mode only for bounded validation"
            )

    def resolved(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["asset_bundle"] = str(Path(self.asset_bundle).resolve())
        if self.reference_processed is not None:
            payload["reference_processed"] = str(
                Path(self.reference_processed).resolve()
            )
        return payload

    def checkpoint_metadata(self) -> dict[str, Any]:
        resolved = self.resolved()
        return {
            "backend": "mjlab_mujoco_warp",
            "schema_version": MJLAB_PPO_CONFIG_VERSION,
            "resolved": resolved,
            "resolved_sha256": _canonical_hash(resolved),
            "reference_noise": self.domain_randomization.reference_noise.metadata(),
        }

    def assert_resume_compatible(self, metadata: dict[str, Any]) -> None:
        expected = self.checkpoint_metadata()
        if metadata.get("backend") != expected["backend"]:
            raise ValueError("checkpoint is not from the mjlab MuJoCo-Warp backend")
        if metadata.get("resolved_sha256") != expected["resolved_sha256"]:
            raise ValueError("checkpoint mjlab PPO configuration hash mismatch")
