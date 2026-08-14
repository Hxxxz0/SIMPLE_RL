"""Versioned configuration and GPU-only safety gates for mjlab PPO."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

MJLAB_PPO_CONFIG_VERSION = 1
MIN_LONG_TRAIN_ENVS = 1024
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

    @property
    def enabled(self) -> bool:
        return bool(
            self.action_std
            or self.position_std
            or self.phase_std
            or self.future_dropout_probability
        )

    def scaled(self, strength: float) -> ReferenceNoiseConfig:
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
    target_position_offset_center_xy: tuple[float, float] = (0.0, 0.0)
    target_position_focus_probability: float = 0.0
    target_position_focus_jitter_xy: tuple[float, float] = (0.0, 0.0)
    target_position_focus_offset_center_xy: tuple[float, float] = (0.0, 0.0)
    target_yaw_jitter: float = 0.15
    destination_position_jitter_xy: tuple[float, float] = (0.0, 0.0)
    destination_yaw_jitter: float = 0.0
    distractor_position_jitter_xy: tuple[float, float] = (0.0, 0.0)
    distractor_yaw_jitter: float = 0.0
    robot_base_position_jitter_xy: tuple[float, float] = (0.0, 0.0)
    robot_base_yaw_jitter: float = 0.0
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
            low, _high = _range_pair(getattr(self, name), name)
            if low <= 0.0:
                raise ValueError(f"{name} values must be positive")
        for name in (
            "target_position_jitter_xy",
            "target_position_focus_jitter_xy",
            "destination_position_jitter_xy",
            "distractor_position_jitter_xy",
            "robot_base_position_jitter_xy",
        ):
            value = getattr(self, name)
            if len(value) != 2 or any(item < 0.0 for item in value):
                raise ValueError(f"{name} values must be non-negative")
        for name in (
            "target_position_offset_center_xy",
            "target_position_focus_offset_center_xy",
        ):
            value = getattr(self, name)
            if len(value) != 2 or not all(
                math.isfinite(float(item)) for item in value
            ):
                raise ValueError(f"{name} values must be finite")
        if not 0.0 <= self.target_position_focus_probability <= 1.0:
            raise ValueError("target_position_focus_probability must be in [0, 1]")
        for name in (
            "target_yaw_jitter",
            "destination_yaw_jitter",
            "distractor_yaw_jitter",
            "robot_base_yaw_jitter",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
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

    def pose_only(self) -> DomainRandomizationConfig:
        """Retain target-pose DR while staging harder dynamics and input noise."""

        return replace(
            self,
            target_mass_scale=(1.0, 1.0),
            friction_scale=(1.0, 1.0),
            joint_damping_scale=(1.0, 1.0),
            actuator_strength_scale=(1.0, 1.0),
            action_delay_max_steps=0,
            reference_noise=ReferenceNoiseConfig(
                action_std=0.0,
                position_std=0.0,
                phase_std=0.0,
                future_dropout_probability=0.0,
            ),
        )

    def target_x_only(self) -> DomainRandomizationConfig:
        """Isolate the target X offset for a diagnosed staged curriculum."""

        pose = self.pose_only()
        return replace(
            pose,
            target_position_jitter_xy=(pose.target_position_jitter_xy[0], 0.0),
            target_position_offset_center_xy=(
                pose.target_position_offset_center_xy[0],
                0.0,
            ),
            target_position_focus_jitter_xy=(
                pose.target_position_focus_jitter_xy[0],
                0.0,
            ),
            target_position_focus_offset_center_xy=(
                pose.target_position_focus_offset_center_xy[0],
                0.0,
            ),
            target_yaw_jitter=0.0,
            destination_position_jitter_xy=(0.0, 0.0),
            destination_yaw_jitter=0.0,
            distractor_position_jitter_xy=(0.0, 0.0),
            distractor_yaw_jitter=0.0,
            robot_base_position_jitter_xy=(0.0, 0.0),
            robot_base_yaw_jitter=0.0,
        )

    def target_y_only(self) -> DomainRandomizationConfig:
        """Isolate the target Y offset for a diagnosed staged curriculum."""

        pose = self.pose_only()
        return replace(
            pose,
            target_position_jitter_xy=(0.0, pose.target_position_jitter_xy[1]),
            target_position_offset_center_xy=(
                0.0,
                pose.target_position_offset_center_xy[1],
            ),
            target_position_focus_jitter_xy=(
                0.0,
                pose.target_position_focus_jitter_xy[1],
            ),
            target_position_focus_offset_center_xy=(
                0.0,
                pose.target_position_focus_offset_center_xy[1],
            ),
            target_yaw_jitter=0.0,
            destination_position_jitter_xy=(0.0, 0.0),
            destination_yaw_jitter=0.0,
            distractor_position_jitter_xy=(0.0, 0.0),
            distractor_yaw_jitter=0.0,
            robot_base_position_jitter_xy=(0.0, 0.0),
            robot_base_yaw_jitter=0.0,
        )

    def target_yaw_only(self) -> DomainRandomizationConfig:
        """Isolate target yaw for a diagnosed staged curriculum."""

        pose = self.pose_only()
        return replace(
            pose,
            target_position_jitter_xy=(0.0, 0.0),
            target_position_offset_center_xy=(0.0, 0.0),
            target_position_focus_probability=0.0,
            target_position_focus_jitter_xy=(0.0, 0.0),
            target_position_focus_offset_center_xy=(0.0, 0.0),
            destination_position_jitter_xy=(0.0, 0.0),
            destination_yaw_jitter=0.0,
            distractor_position_jitter_xy=(0.0, 0.0),
            distractor_yaw_jitter=0.0,
            robot_base_position_jitter_xy=(0.0, 0.0),
            robot_base_yaw_jitter=0.0,
        )


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
    strict_reference_episode: int | None = None
    reference_selection: str = "asset"
    max_reference_initial_position_offset: float | None = None
    reference_reward_weight: float = 0.05
    reference_target_x_arm_gains: tuple[float, float] = (0.0, 0.0)
    reference_target_y_arm_gains: tuple[float, float] = (0.0, 0.0)
    reference_target_positive_y_arm_gains: tuple[float, float] | None = None
    reference_target_yaw_arm_gains: tuple[float, float] = (0.0, 0.0)
    max_reference_action_deviation: float = 0.35
    full_dr_reference_reward_scale: float = 0.2
    grasp_anything_lift_arm_residual_min_scale: float = 1.0
    grasp_anything_lift_arm_residual_decay_steps: int = 0
    grasp_anything_lift_arm_residual_grasp_steps: int = 3
    grasp_anything_goal_potential_scale: float = 5.0
    grasp_anything_goal_potential_negative_clip: float = 0.25
    grasp_anything_success_bonus: float = 40.0
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
        if self.strict_reference_episode is not None and self.strict_reference_episode < 0:
            raise ValueError("strict_reference_episode must be non-negative")
        if self.reference_selection not in ("asset", "nearest", "balanced"):
            raise ValueError(
                "reference_selection must be asset, nearest, or balanced"
            )
        if (
            self.strict_reference_episode is not None
            and self.reference_selection != "asset"
        ):
            raise ValueError(
                "strict_reference_episode requires asset reference selection"
            )
        if self.max_reference_initial_position_offset is not None and (
            not math.isfinite(self.max_reference_initial_position_offset)
            or self.max_reference_initial_position_offset <= 0.0
        ):
            raise ValueError(
                "max_reference_initial_position_offset must be positive and finite"
            )
        if (
            self.reference_selection != "asset"
            and self.max_reference_initial_position_offset is None
        ):
            raise ValueError(
                "non-asset reference selection requires an explicit initial-position "
                "alignment limit"
            )
        if self.reference_reward_weight < 0.0:
            raise ValueError("reference_reward_weight must be non-negative")
        if len(self.reference_target_x_arm_gains) != 2 or not all(
            math.isfinite(float(value)) for value in self.reference_target_x_arm_gains
        ):
            raise ValueError(
                "reference_target_x_arm_gains must contain two finite values"
            )
        if len(self.reference_target_y_arm_gains) != 2 or not all(
            math.isfinite(float(value)) for value in self.reference_target_y_arm_gains
        ):
            raise ValueError(
                "reference_target_y_arm_gains must contain two finite values"
            )
        if self.reference_target_positive_y_arm_gains is not None and (
            len(self.reference_target_positive_y_arm_gains) != 2
            or not all(
                math.isfinite(float(value))
                for value in self.reference_target_positive_y_arm_gains
            )
        ):
            raise ValueError(
                "reference_target_positive_y_arm_gains must contain two finite values"
            )
        if len(self.reference_target_yaw_arm_gains) != 2 or not all(
            math.isfinite(float(value)) for value in self.reference_target_yaw_arm_gains
        ):
            raise ValueError(
                "reference_target_yaw_arm_gains must contain two finite values"
            )
        if not 0.0 < self.max_reference_action_deviation <= 2.0:
            raise ValueError("max_reference_action_deviation must be in (0, 2]")
        if not 0.0 <= self.full_dr_reference_reward_scale <= 1.0:
            raise ValueError("full_dr_reference_reward_scale must be in [0, 1]")
        if not 0.0 < self.grasp_anything_lift_arm_residual_min_scale <= 1.0:
            raise ValueError(
                "grasp_anything_lift_arm_residual_min_scale must be in (0, 1]"
            )
        if self.grasp_anything_lift_arm_residual_decay_steps < 0:
            raise ValueError(
                "grasp_anything_lift_arm_residual_decay_steps must be non-negative"
            )
        if self.grasp_anything_lift_arm_residual_grasp_steps < 1:
            raise ValueError(
                "grasp_anything_lift_arm_residual_grasp_steps must be positive"
            )
        if (
            not math.isfinite(self.grasp_anything_goal_potential_scale)
            or self.grasp_anything_goal_potential_scale <= 0.0
        ):
            raise ValueError(
                "grasp_anything_goal_potential_scale must be positive and finite"
            )
        if not (
            math.isfinite(self.grasp_anything_goal_potential_negative_clip)
            and 0.0 < self.grasp_anything_goal_potential_negative_clip <= 1.0
        ):
            raise ValueError(
                "grasp_anything_goal_potential_negative_clip must be in (0, 1]"
            )
        if (
            not math.isfinite(self.grasp_anything_success_bonus)
            or self.grasp_anything_success_bonus <= 0.0
        ):
            raise ValueError("grasp_anything_success_bonus must be positive and finite")
        object_reward_changed = (
            self.grasp_anything_goal_potential_scale != 5.0
            or self.grasp_anything_goal_potential_negative_clip != 0.25
            or self.grasp_anything_success_bonus != 40.0
        )
        if object_reward_changed and self.task != "grasp_anything":
            raise ValueError(
                "grasp_anything reward shaping overrides require grasp_anything"
            )
        lift_arm_decay_enabled = (
            self.grasp_anything_lift_arm_residual_min_scale < 1.0
            or self.grasp_anything_lift_arm_residual_decay_steps > 0
        )
        if lift_arm_decay_enabled and (
            self.task != "grasp_anything"
            or self.grasp_anything_lift_arm_residual_min_scale >= 1.0
            or self.grasp_anything_lift_arm_residual_decay_steps < 1
        ):
            raise ValueError(
                "lift arm residual decay requires grasp_anything, a minimum scale "
                "below one, and positive decay steps"
            )
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
        if metadata.get("resolved_sha256") == expected["resolved_sha256"]:
            return
        # Checkpoints released before optional reference retargeting lack these
        # zero-default fields.  Preserve exact resume only when each missing
        # behavior is disabled and every other resolved value still matches.
        legacy = metadata.get("resolved")
        if isinstance(legacy, dict):
            normalized = dict(legacy)
            for name, gains in (
                ("reference_target_x_arm_gains", self.reference_target_x_arm_gains),
                ("reference_target_y_arm_gains", self.reference_target_y_arm_gains),
                (
                    "reference_target_positive_y_arm_gains",
                    self.reference_target_positive_y_arm_gains,
                ),
                (
                    "reference_target_yaw_arm_gains",
                    self.reference_target_yaw_arm_gains,
                ),
            ):
                if name not in normalized:
                    if gains == (0.0, 0.0):
                        normalized[name] = [0.0, 0.0]
                    elif gains is None:
                        normalized[name] = None
            if (
                "strict_reference_episode" not in normalized
                and self.strict_reference_episode is None
            ):
                normalized["strict_reference_episode"] = None
            if (
                "reference_selection" not in normalized
                and self.reference_selection == "asset"
            ):
                normalized["reference_selection"] = "asset"
            if (
                "max_reference_initial_position_offset" not in normalized
                and self.max_reference_initial_position_offset is None
            ):
                normalized["max_reference_initial_position_offset"] = None
            for name, default in (
                ("grasp_anything_lift_arm_residual_min_scale", 1.0),
                ("grasp_anything_lift_arm_residual_decay_steps", 0),
                ("grasp_anything_lift_arm_residual_grasp_steps", 3),
                ("grasp_anything_goal_potential_scale", 5.0),
                ("grasp_anything_goal_potential_negative_clip", 0.25),
                ("grasp_anything_success_bonus", 40.0),
            ):
                if name not in normalized and getattr(self, name) == default:
                    normalized[name] = default
            expected_dr = expected["resolved"]["domain_randomization"]
            legacy_dr = normalized.get("domain_randomization")
            if isinstance(legacy_dr, dict):
                legacy_dr = dict(legacy_dr)
                if (
                    "target_position_offset_center_xy" not in legacy_dr
                    and tuple(
                        expected_dr.get("target_position_offset_center_xy", ())
                    )
                    == (0.0, 0.0)
                ):
                    legacy_dr["target_position_offset_center_xy"] = [0.0, 0.0]
                for name, default in (
                    ("target_position_focus_probability", 0.0),
                    ("target_position_focus_jitter_xy", [0.0, 0.0]),
                    ("target_position_focus_offset_center_xy", [0.0, 0.0]),
                ):
                    if name not in legacy_dr and expected_dr.get(name) == default:
                        legacy_dr[name] = default
                normalized["domain_randomization"] = legacy_dr
            # A portable release may relocate frozen assets and reference data.
            # The runner separately verifies both content hashes, so filesystem
            # paths are not part of behavioral compatibility.
            normalized["asset_bundle"] = expected["resolved"]["asset_bundle"]
            normalized["reference_processed"] = expected["resolved"][
                "reference_processed"
            ]
            if _canonical_hash(normalized) == expected["resolved_sha256"]:
                return
        raise ValueError("checkpoint mjlab PPO configuration hash mismatch")
