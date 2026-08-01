"""Versioned, reproducible curriculum configuration for pick PPO training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


CURRICULUM_SCHEMA_VERSION = 1
TARGET_MIX_KEYS = ("uniform", "hard", "native")


def _range_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low, high = (float(item) for item in value)
    if low > high:
        raise ValueError(f"{name} must satisfy low <= high")
    return low, high


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _normalized_weights(
    value: Any,
    name: str,
    *,
    allowed: tuple[str, ...] | None = None,
) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    result = {str(key): float(weight) for key, weight in value.items()}
    if allowed is not None and set(result) != set(allowed):
        raise ValueError(f"{name} must contain exactly {', '.join(allowed)}")
    if any(weight < 0.0 for weight in result.values()):
        raise ValueError(f"{name} weights must be non-negative")
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return {key: weight / total for key, weight in result.items()}


@dataclass(frozen=True)
class DomainRandomization:
    target_mass_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)
    manipulation_action_noise_std: float = 0.0
    action_delay_max_steps: int = 0

    @classmethod
    def from_dict(cls, value: Any) -> "DomainRandomization":
        value = {} if value is None else value
        if not isinstance(value, dict):
            raise ValueError("domain_randomization must be an object")
        unknown = set(value) - {
            "target_mass_scale",
            "friction_scale",
            "manipulation_action_noise_std",
            "action_delay_max_steps",
        }
        if unknown:
            raise ValueError(
                "Unknown domain_randomization fields: " + ", ".join(sorted(unknown))
            )
        mass = _range_pair(value.get("target_mass_scale", (1.0, 1.0)), "target_mass_scale")
        friction = _range_pair(value.get("friction_scale", (1.0, 1.0)), "friction_scale")
        if mass[0] <= 0.0 or friction[0] <= 0.0:
            raise ValueError("mass and friction scales must be positive")
        noise = float(value.get("manipulation_action_noise_std", 0.0))
        delay = int(value.get("action_delay_max_steps", 0))
        if noise < 0.0:
            raise ValueError("manipulation_action_noise_std must be non-negative")
        if delay not in (0, 1):
            raise ValueError("action_delay_max_steps must be 0 or 1")
        return cls(mass, friction, noise, delay)


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    start_update: int
    rsi_probability: float
    target_mix: dict[str, float]
    rsi_stage_weights: dict[str, float]
    reference_rank_max: int
    reference_base_episode_probability: float
    reference_action_noise_std: float
    domain_randomization: DomainRandomization

    @classmethod
    def from_dict(cls, value: Any) -> "CurriculumPhase":
        if not isinstance(value, dict):
            raise ValueError("Every curriculum phase must be an object")
        required = {"name", "start_update", "rsi_probability"}
        missing = required - set(value)
        if missing:
            raise ValueError("Curriculum phase missing: " + ", ".join(sorted(missing)))
        allowed = required | {
            "target_mix",
            "rsi_stage_weights",
            "reference_rank_max",
            "reference_base_episode_probability",
            "reference_action_noise_std",
            "domain_randomization",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("Unknown curriculum phase fields: " + ", ".join(sorted(unknown)))
        start = int(value["start_update"])
        rank = int(value.get("reference_rank_max", 0))
        reference_noise = float(value.get("reference_action_noise_std", 0.0))
        if start < 0 or rank < 0 or reference_noise < 0.0:
            raise ValueError("start_update, reference rank, and noise must be non-negative")
        stage_weights = value.get("rsi_stage_weights", {"pregrasp": 0.6, "grasp_to_lift": 0.4})
        return cls(
            name=str(value["name"]),
            start_update=start,
            rsi_probability=_probability(value["rsi_probability"], "rsi_probability"),
            target_mix=_normalized_weights(
                value.get("target_mix", {"uniform": 1.0, "hard": 0.0, "native": 0.0}),
                "target_mix",
                allowed=TARGET_MIX_KEYS,
            ),
            rsi_stage_weights=_normalized_weights(stage_weights, "rsi_stage_weights"),
            reference_rank_max=rank,
            reference_base_episode_probability=_probability(
                value.get("reference_base_episode_probability", 0.0),
                "reference_base_episode_probability",
            ),
            reference_action_noise_std=reference_noise,
            domain_randomization=DomainRandomization.from_dict(
                value.get("domain_randomization")
            ),
        )

    def runtime_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["phase_name"] = result.pop("name")
        result["domain_randomization"] = asdict(self.domain_randomization)
        return result


@dataclass(frozen=True)
class TrainingCurriculum:
    phases: tuple[CurriculumPhase, ...]
    source_path: str
    source_sha256: str
    resolved_sha256: str

    def phase_for_update(self, update: int) -> CurriculumPhase:
        if update < 0:
            raise ValueError("update must be non-negative")
        selected = self.phases[0]
        for phase in self.phases[1:]:
            if phase.start_update > update:
                break
            selected = phase
        return selected

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CURRICULUM_SCHEMA_VERSION,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "resolved_sha256": self.resolved_sha256,
            "phases": [phase.runtime_dict() for phase in self.phases],
        }


def load_curriculum(path: str | Path) -> TrainingCurriculum:
    source = Path(path)
    raw = source.read_bytes()
    document = json.loads(raw)
    if document.get("schema_version") != CURRICULUM_SCHEMA_VERSION:
        raise ValueError(
            f"curriculum schema_version must be {CURRICULUM_SCHEMA_VERSION}"
        )
    unknown = set(document) - {"schema_version", "phases"}
    if unknown:
        raise ValueError("Unknown curriculum fields: " + ", ".join(sorted(unknown)))
    phases = tuple(CurriculumPhase.from_dict(item) for item in document.get("phases", []))
    if not phases:
        raise ValueError("curriculum must contain at least one phase")
    if phases[0].start_update != 0:
        raise ValueError("the first curriculum phase must start at update 0")
    starts = [phase.start_update for phase in phases]
    if starts != sorted(set(starts)):
        raise ValueError("curriculum phase start_update values must be unique and sorted")
    resolved = json.dumps(
        [phase.runtime_dict() for phase in phases],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return TrainingCurriculum(
        phases=phases,
        source_path=str(source.resolve()),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        resolved_sha256=hashlib.sha256(resolved).hexdigest(),
    )
