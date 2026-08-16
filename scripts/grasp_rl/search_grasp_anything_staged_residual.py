"""Find a contact-producing staged residual before grasp-anything PPO training."""

# ruff: noqa: B023

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.config import (
    DomainRandomizationConfig,
    MjlabPpoConfig,
    ReferenceNoiseConfig,
)
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv

HAND = slice(7, 14)
ARM = slice(21, 28)
BODY = slice(28, 32)
NAVIGATION = slice(32, 34)
PARAMETER_COUNT = 11
PARAMETER_NAMES = (
    "arm_0",
    "arm_1",
    "arm_2",
    "arm_3",
    "arm_4",
    "arm_5",
    "arm_6",
    "thumb_abduction",
    "thumb_curl",
    "index_open",
    "middle_open",
)
CAPTURE_ARM_PARAMETER_NAMES = tuple(f"capture_arm_{index}" for index in range(7))
BODY_PARAMETER_NAMES = ("torso_roll", "torso_pitch", "torso_yaw", "base_height")
NAVIGATION_PARAMETER_NAMES = ("torso_vx", "torso_vy")


def reference_episode_path(reference: Path, episode: int) -> Path:
    """Return the canonical strict-reference episode path."""

    if episode < 0:
        raise ValueError("reference episode must be non-negative")
    return reference / "bc" / f"episode_{episode:06d}.npz"


ARM_BOUNDS = (
    (-0.75, -0.25),
    (-0.15, 0.35),
    (0.10, 0.75),
    (0.20, 0.90),
    (-0.70, 0.20),
    (-0.10, 0.80),
    (-0.70, 0.50),
)
APERTURE_BOUNDS = (
    # The mapped public action remains within [-0.89, 0.81]. Wider thumb
    # abduction is needed to place the thumb around thin, concave rims.
    (0.10, 1.80),
    (0.00, 0.50),
    (0.00, 0.55),
    (0.00, 0.55),
)


def _arm_bounds(args: argparse.Namespace) -> tuple[tuple[float, float], ...]:
    values = tuple(float(value) for value in args.arm_bounds)
    if len(values) != 2 * len(ARM_BOUNDS):
        raise ValueError("arm bounds require one low/high pair per arm joint")
    bounds = tuple(zip(values[::2], values[1::2], strict=True))
    if any(low >= high for low, high in bounds):
        raise ValueError("each arm bound must be strictly ordered")
    return bounds


def _capture_arm_bounds(
    args: argparse.Namespace,
) -> tuple[tuple[float, float], ...]:
    if not args.search_capture_arm:
        return ()
    values = tuple(float(value) for value in args.capture_arm_bounds)
    if len(values) != 2 * len(ARM_BOUNDS):
        raise ValueError("capture arm bounds require one low/high pair per arm joint")
    bounds = tuple(zip(values[::2], values[1::2], strict=True))
    if any(low >= high for low, high in bounds):
        raise ValueError("each capture arm bound must be strictly ordered")
    return bounds


def _body_bounds(args: argparse.Namespace) -> tuple[tuple[float, float], ...]:
    if not args.search_body:
        return ()
    values = tuple(float(value) for value in args.body_bounds)
    if len(values) != 2 * len(BODY_PARAMETER_NAMES):
        raise ValueError("body bounds require one low/high pair per body action")
    bounds = tuple(zip(values[::2], values[1::2], strict=True))
    if any(low >= high for low, high in bounds):
        raise ValueError("each body bound must be strictly ordered")
    return bounds


def _navigation_bounds(
    args: argparse.Namespace,
) -> tuple[tuple[float, float], ...]:
    if not args.search_navigation:
        return ()
    values = tuple(float(value) for value in args.navigation_bounds)
    if len(values) != 2 * len(NAVIGATION_PARAMETER_NAMES):
        raise ValueError("navigation bounds require one low/high pair per XY command")
    bounds = tuple(zip(values[::2], values[1::2], strict=True))
    if any(low >= high for low, high in bounds):
        raise ValueError("each navigation bound must be strictly ordered")
    return bounds


def _parameter_names(
    *,
    search_capture_arm: bool,
    search_body: bool = False,
    search_navigation: bool = False,
) -> tuple[str, ...]:
    return (
        *PARAMETER_NAMES[:7],
        *(CAPTURE_ARM_PARAMETER_NAMES if search_capture_arm else ()),
        *(BODY_PARAMETER_NAMES if search_body else ()),
        *(NAVIGATION_PARAMETER_NAMES if search_navigation else ()),
        *PARAMETER_NAMES[7:],
    )


def _candidate_layout(
    candidates: torch.Tensor,
) -> tuple[slice | None, slice | None, slice | None, slice]:
    """Resolve optional staged blocks while retaining every released layout."""

    flags = {
        11: (False, False, False),
        13: (False, False, True),
        15: (False, True, False),
        17: (False, True, True),
        18: (True, False, False),
        20: (True, False, True),
        22: (True, True, False),
        24: (True, True, True),
    }.get(candidates.shape[-1])
    if flags is None:
        raise ValueError("candidate parameters use an unsupported staged layout")
    has_capture, has_body, has_navigation = flags
    cursor = 7
    capture = slice(cursor, cursor + 7) if has_capture else None
    cursor += 7 if has_capture else 0
    body = slice(cursor, cursor + 4) if has_body else None
    cursor += 4 if has_body else 0
    navigation = slice(cursor, cursor + 2) if has_navigation else None
    cursor += 2 if has_navigation else 0
    return capture, body, navigation, slice(cursor, cursor + 4)


def _candidate_capture_arm(candidates: torch.Tensor) -> torch.Tensor:
    capture, _, _, _ = _candidate_layout(candidates)
    return candidates[..., :7] if capture is None else candidates[..., capture]


def _candidate_body(candidates: torch.Tensor) -> torch.Tensor | None:
    _, body, _, _ = _candidate_layout(candidates)
    return None if body is None else candidates[..., body]


def _candidate_navigation(candidates: torch.Tensor) -> torch.Tensor | None:
    _, _, navigation, _ = _candidate_layout(candidates)
    return None if navigation is None else candidates[..., navigation]


def _candidate_aperture(candidates: torch.Tensor) -> torch.Tensor:
    _, _, _, aperture = _candidate_layout(candidates)
    return candidates[..., aperture]


def phase_blend(
    step: int, *, arm_start: int, capture_step: int, close_end: int
) -> tuple[float, float]:
    """Return arm-approach and hand-close interpolation weights."""

    if not 0 <= arm_start < capture_step < close_end:
        raise ValueError("staged residual boundaries must be strictly ordered")
    arm = min(max((step - arm_start) / (capture_step - arm_start), 0.0), 1.0)
    close = min(max((step - capture_step) / (close_end - capture_step), 0.0), 1.0)
    return arm, close


def arm_release_blend(
    step: int, *, release_start: int | None, release_end: int | None
) -> float:
    """Return an optional blend from the captured pose to the reference arm."""

    if release_start is None and release_end is None:
        return 0.0
    if release_start is None or release_end is None or release_end <= release_start:
        raise ValueError("arm release requires strictly ordered start and end steps")
    return min(max((step - release_start) / (release_end - release_start), 0.0), 1.0)


def body_phase_blend(
    step: int,
    *,
    body_start: int | None,
    body_full_step: int | None,
    fallback_weight: float,
) -> float:
    """Optionally reach the posture correction before the arm capture phase."""

    if body_start is None and body_full_step is None:
        return fallback_weight
    if (
        body_start is None
        or body_full_step is None
        or not 0 <= body_start < body_full_step
    ):
        raise ValueError("body phase requires strictly ordered non-negative steps")
    return min(max((step - body_start) / (body_full_step - body_start), 0.0), 1.0)


def _effective_arm_release_weight(
    args: argparse.Namespace,
    *,
    step: int,
    capture_active: bool,
    overhead_release_weight: float,
) -> float:
    if not capture_active:
        return overhead_release_weight
    if args.search_capture_arm and (
        args.capture_arm_release_start is not None
        or args.capture_arm_release_end is not None
    ):
        return arm_release_blend(
            step,
            release_start=args.capture_arm_release_start,
            release_end=args.capture_arm_release_end,
        )
    if (
        args.search_capture_arm
        and args.arm_release_end is not None
        and args.arm_release_end <= args.capture_step
    ):
        return 0.0
    return overhead_release_weight


def _initial_distribution(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    capture_mean = args.base_capture_arm if args.search_capture_arm else ()
    capture_std = args.capture_arm_std if args.search_capture_arm else ()
    body_mean = args.base_body if args.search_body else ()
    body_std = args.body_std if args.search_body else ()
    navigation_mean = args.base_navigation if args.search_navigation else ()
    navigation_std = args.navigation_std if args.search_navigation else ()
    mean = torch.tensor(
        [
            *args.base_arm,
            *capture_mean,
            *body_mean,
            *navigation_mean,
            *args.aperture_mean,
        ],
        dtype=torch.float32,
        device=args.device,
    )
    std = torch.tensor(
        [
            *args.arm_std,
            *capture_std,
            *body_std,
            *navigation_std,
            *args.aperture_std,
        ],
        dtype=torch.float32,
        device=args.device,
    )
    expected = (
        PARAMETER_COUNT
        + (7 if args.search_capture_arm else 0)
        + (4 if args.search_body else 0)
        + (2 if args.search_navigation else 0)
    )
    if mean.shape != (expected,) or std.shape != (expected,):
        raise ValueError("arm vectors need seven values and aperture vectors need four")
    if (std <= 0).any():
        raise ValueError("CEM standard deviations must be positive")
    return mean, std


def _sample_candidates(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
    arm_bounds: tuple[tuple[float, float], ...] = ARM_BOUNDS,
    capture_arm_bounds: tuple[tuple[float, float], ...] = (),
    body_bounds: tuple[tuple[float, float], ...] = (),
    navigation_bounds: tuple[tuple[float, float], ...] = (),
) -> torch.Tensor:
    if count < 32:
        raise ValueError("staged residual search requires at least 32 candidates")
    bounds = (
        *arm_bounds,
        *capture_arm_bounds,
        *body_bounds,
        *navigation_bounds,
        *APERTURE_BOUNDS,
    )
    if mean.shape != std.shape or mean.shape != (len(bounds),):
        raise ValueError("candidate distribution does not match parameter bounds")
    candidates = mean + std * torch.randn(
        count, len(bounds), device=mean.device, generator=generator
    )
    candidates[0] = mean
    for index, (low, high) in enumerate(bounds):
        candidates[:, index].clamp_(low, high)
    # Always probe each bound independently. The previous narrow Apple run
    # converged onto several bounds without establishing which direction still
    # improved geometry, and pure Gaussian samples need not isolate that signal.
    for index, (low, high) in enumerate(bounds):
        low_row = 1 + 2 * index
        high_row = low_row + 1
        if high_row >= count:
            break
        candidates[low_row] = mean
        candidates[low_row, index] = low
        candidates[high_row] = mean
        candidates[high_row, index] = high
    return candidates


def replay_candidate_across_worlds(
    candidates: torch.Tensor,
    candidate: torch.Tensor,
    *,
    replicas: int,
) -> None:
    """Replace a prefix with one candidate for opt-in robust-world evaluation."""

    if replicas < 0:
        raise ValueError("replay best candidate replicas must be non-negative")
    if replicas > len(candidates):
        raise ValueError("replay best candidate replicas exceed environment count")
    if candidate.shape != candidates.shape[1:]:
        raise ValueError("replayed candidate has the wrong parameter shape")
    if replicas:
        candidates[:replicas] = candidate


def _environment(args: argparse.Namespace) -> GpuGraspVecEnv:
    dr = DomainRandomizationConfig(
        enabled=True,
        target_position_jitter_xy=tuple(args.target_position_jitter_xy),
        target_position_offset_center_xy=tuple(args.target_position_offset_center_xy),
        target_yaw_jitter=args.target_yaw_jitter,
        destination_position_jitter_xy=(0.0, 0.0),
        destination_yaw_jitter=0.0,
        distractor_position_jitter_xy=(0.0, 0.0),
        distractor_yaw_jitter=0.0,
        robot_base_position_jitter_xy=tuple(args.robot_base_position_jitter_xy),
        robot_base_yaw_jitter=args.robot_base_yaw_jitter,
        target_mass_scale=(1.0, 1.0),
        friction_scale=(1.0, 1.0),
        joint_damping_scale=(1.0, 1.0),
        actuator_strength_scale=(1.0, 1.0),
        action_delay_max_steps=0,
        curriculum_initial_strength=1.0,
        curriculum_warmup_steps=0,
        curriculum_ramp_steps=1,
        reference_noise=ReferenceNoiseConfig(
            action_std=0.0,
            position_std=0.0,
            phase_std=0.0,
            future_dropout_probability=0.0,
        ),
    )
    config = MjlabPpoConfig(
        task="grasp_anything",
        asset_bundle=str(args.asset.resolve()),
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        smoke_mode=args.num_envs < 2048,
        reference_processed=str(args.reference.resolve()),
        strict_reference_episode=args.reference_episode,
        max_reference_initial_position_offset=0.12,
        reference_reward_weight=0.005,
        # The audited pick references close the hand to approximately +/-0.99
        # in one frame. Apple search must be able to reopen that proposal
        # completely. This is local to this diagnostic and does not change any
        # training default.
        max_reference_action_deviation=1.0,
        reference_target_x_arm_gains=tuple(args.reference_target_x_arm_gains),
        reference_target_y_arm_gains=tuple(args.reference_target_y_arm_gains),
        reference_target_positive_y_arm_gains=tuple(
            args.reference_target_positive_y_arm_gains
        ),
        domain_randomization=dr,
    )
    return GpuGraspVecEnv(config, training=True, randomization_enabled=True)


def aperture_hand_target(aperture: torch.Tensor) -> torch.Tensor:
    """Map four physical opening controls to the seven public hand targets."""

    if aperture.shape[-1] != 4:
        raise ValueError("aperture controls must contain four values")
    thumb_abduction, thumb_curl, index_open, middle_open = aperture.unbind(-1)
    return torch.stack(
        (
            -0.99 + thumb_abduction,
            -0.99 + thumb_curl,
            -0.99 + thumb_curl,
            0.99 - index_open,
            0.99 - index_open,
            0.99 - middle_open,
            0.99 - middle_open,
        ),
        dim=-1,
    )


def fingertip_shell_metrics(
    distal_pos_w: torch.Tensor,
    grasp_center_w: torch.Tensor,
    *,
    grip_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return center distances, shell gaps, and the worst opposing-side gap."""

    if distal_pos_w.ndim != 3 or distal_pos_w.shape[-2:] != (3, 3):
        raise ValueError("distal positions must have shape (N, 3, 3)")
    if grasp_center_w.shape != (distal_pos_w.shape[0], 3):
        raise ValueError("grasp centers must have shape (N, 3)")
    radius = max(0.5 * float(grip_width_m), 1e-3)
    center_distances = (distal_pos_w - grasp_center_w[:, None]).norm(dim=-1)
    shell_gaps = (center_distances - radius).abs()
    balanced_gap = torch.maximum(shell_gaps[:, 0], shell_gaps[:, 1:].amin(-1))
    return center_distances, shell_gaps, balanced_gap


def balanced_surface_gap(surface_distances: torch.Tensor) -> torch.Tensor:
    """Require the thumb and at least one support finger near the object surface."""

    if surface_distances.ndim != 2 or surface_distances.shape[-1] != 3:
        raise ValueError("surface distances must have shape (N, 3)")
    return torch.maximum(surface_distances[:, 0], surface_distances[:, 1:].amin(dim=-1))


def candidate_score(
    *,
    success: torch.Tensor,
    max_grasp_streak: torch.Tensor,
    max_bilateral_streak: torch.Tensor,
    min_opposing_force: torch.Tensor,
    balanced_shell_gap: torch.Tensor,
    finger_opposition: torch.Tensor,
    lift_height: torch.Tensor,
    grasp_streak_score_cap: int = 20,
) -> torch.Tensor:
    """Rank physical contact first and otherwise minimize both shell gaps."""

    if grasp_streak_score_cap < 1:
        raise ValueError("grasp streak score cap must be positive")
    physical_lift_gate = (max_grasp_streak >= 2) | (max_bilateral_streak >= 2)
    return (
        2500.0 * success.float()
        + 1000.0 * (max_grasp_streak >= 2).float()
        + 300.0 * (max_bilateral_streak >= 2).float()
        + 10.0 * max_grasp_streak.float().clamp_max(grasp_streak_score_cap)
        + 5.0 * max_bilateral_streak.float().clamp_max(grasp_streak_score_cap)
        + 20.0 * (min_opposing_force / 8.0).clamp(0.0, 1.0)
        - 100.0 * balanced_shell_gap
        + 2.0 * finger_opposition.clamp(0.0, 1.0)
        + 500.0 * physical_lift_gate.float() * (lift_height / 0.09).clamp(0.0, 1.0)
    )


def should_stop_search(
    record: dict[str, object],
    *,
    stop_grasp_candidates: int,
    stop_success_candidates: int | None,
) -> bool:
    """Apply the legacy grasp stop unless native-success stopping is requested."""

    if stop_grasp_candidates < 1:
        raise ValueError("stop grasp candidates must be positive")
    if stop_success_candidates is not None:
        if stop_success_candidates < 1:
            raise ValueError("stop success candidates must be positive")
        return int(record["success_candidates"]) >= stop_success_candidates
    return int(record["grasp_candidates"]) >= stop_grasp_candidates


def _parameter_bound_mask(
    candidates: torch.Tensor,
    *,
    arm_bounds: tuple[tuple[float, float], ...] = ARM_BOUNDS,
    capture_arm_bounds: tuple[tuple[float, float], ...] = (),
    body_bounds: tuple[tuple[float, float], ...] = (),
    navigation_bounds: tuple[tuple[float, float], ...] = (),
) -> torch.Tensor:
    bounds = torch.tensor(
        (
            *arm_bounds,
            *capture_arm_bounds,
            *body_bounds,
            *navigation_bounds,
            *APERTURE_BOUNDS,
        ),
        dtype=candidates.dtype,
        device=candidates.device,
    )
    return torch.isclose(candidates, bounds[:, 0], atol=1e-6, rtol=0.0) | torch.isclose(
        candidates, bounds[:, 1], atol=1e-6, rtol=0.0
    )


def _candidate_parameters(candidate: dict[str, object]) -> torch.Tensor:
    capture = candidate.get("capture_arm_correction") or ()
    body = candidate.get("body_correction") or ()
    navigation = candidate.get("navigation_correction") or ()
    return torch.tensor(
        [
            *candidate["right_arm_correction"],
            *capture,
            *body,
            *navigation,
            *candidate["aperture"],
        ],
        dtype=torch.float32,
    )


def rank_export_candidates(
    result: dict[str, object],
    *,
    limit: int,
    arm_bounds: tuple[tuple[float, float], ...] = ARM_BOUNDS,
    capture_arm_bounds: tuple[tuple[float, float], ...] = (),
    body_bounds: tuple[tuple[float, float], ...] = (),
    navigation_bounds: tuple[tuple[float, float], ...] = (),
) -> list[dict[str, object]]:
    """Rank distinct native successes by local density and physical margin."""

    if limit < 1:
        raise ValueError("export candidate replay limit must be positive")
    pool: list[dict[str, object]] = []
    for round_record in result.get("rounds", []):
        if not isinstance(round_record, dict):
            continue
        for candidate in round_record.get("successful_candidates", []):
            if isinstance(candidate, dict) and candidate.get("had_success"):
                pool.append(candidate)
    best = result.get("global_best")
    if isinstance(best, dict) and best.get("had_success"):
        pool.append(best)
    if not pool:
        return []

    unique: list[dict[str, object]] = []
    unique_parameters: list[torch.Tensor] = []
    for candidate in pool:
        parameters = _candidate_parameters(candidate)
        if any(
            torch.allclose(parameters, prior, atol=1e-7, rtol=0.0)
            for prior in unique_parameters
        ):
            continue
        unique.append(candidate)
        unique_parameters.append(parameters)

    bounds = torch.tensor(
        (
            *arm_bounds,
            *capture_arm_bounds,
            *body_bounds,
            *navigation_bounds,
            *APERTURE_BOUNDS,
        ),
        dtype=torch.float32,
    )
    widths = bounds[:, 1] - bounds[:, 0]
    normalized = torch.stack(unique_parameters) / widths
    pairwise = torch.cdist(normalized, normalized)
    neighbor_count = min(5, max(len(unique) - 1, 0))
    ranked: list[dict[str, object]] = []
    for index, candidate in enumerate(unique):
        if neighbor_count:
            distances = pairwise[index][torch.arange(len(unique)) != index]
            neighbor_radius = float(
                distances.topk(neighbor_count, largest=False).values.mean()
            )
        else:
            neighbor_radius = float("inf")
        enriched = dict(candidate)
        enriched["success_neighbor_radius"] = neighbor_radius
        ranked.append(enriched)

    def key(candidate: dict[str, object]) -> tuple[object, ...]:
        return (
            len(candidate.get("candidate_bound_parameters", [])) > 0,
            candidate["success_neighbor_radius"],
            -int(candidate.get("max_grasp_streak", 0)),
            -int(candidate.get("max_bilateral_contact_streak", 0)),
            -float(candidate.get("max_lift_m", 0.0)),
            -float(candidate.get("max_min_opposing_force_n", 0.0)),
        )

    return sorted(ranked, key=key)[:limit]


def replay_quality_key(
    replay: dict[str, object],
) -> tuple[float, int, int, float, float]:
    """Rank independent replays by repeatability, then physical margin."""

    return (
        float(replay["independent_replay_success_rate_at_first_terminal_step"]),
        int(replay["max_grasp_streak"]),
        int(replay["max_bilateral_contact_streak"]),
        float(replay["max_lift_m"]),
        float(replay["max_min_opposing_force_n"]),
    )


def search_rollout_path(
    output: Path, *, round_index: int, env_id: int
) -> Path:
    """Return the isolated trajectory path for one native search success."""

    if round_index < 0 or env_id < 0:
        raise ValueError("search rollout indices must be non-negative")
    return (
        output.parent
        / f"{output.stem}_rollouts"
        / f"round_{round_index:03d}_env_{env_id:04d}.npz"
    ).resolve()


def load_search_rollout(candidate: dict[str, object]) -> dict[str, object]:
    """Load and validate a trajectory captured in the successful search batch."""

    path_value = candidate.get("search_rollout_path")
    if not isinstance(path_value, str):
        raise TypeError("native-success candidate has no captured search rollout")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        observations = saved["observations"].astype(np.float32)
        raw_actions = saved["raw_actions"].astype(np.float32)
        terminal_success = bool(saved["terminal_success"].item())
        terminal_step = int(saved["terminal_step"].item())
        max_lift_m = float(saved["max_lift_m"].item())
        max_grasp_streak = int(saved["max_grasp_streak"].item())
        max_bilateral_streak = int(saved["max_bilateral_contact_streak"].item())
        max_min_force = float(saved["max_min_opposing_force_n"].item())
    if observations.ndim != 2 or observations.shape[1] != 331:
        raise RuntimeError("captured search rollout did not contain 331-D observations")
    if raw_actions.shape != (len(observations), 36):
        raise RuntimeError("captured search rollout did not contain aligned 36-D actions")
    if not terminal_success or terminal_step != len(raw_actions):
        raise RuntimeError("captured search rollout did not end at native success")
    if max_lift_m < 0.09 or max_grasp_streak < 13:
        raise RuntimeError("captured search rollout failed strict physical margins")
    return {
        "observations": observations,
        "raw_actions": raw_actions,
        "successful_world_id": int(candidate["env_id"]),
        "replay_envs": 1,
        "successes_at_first_terminal_step": 1,
        "independent_replay_success_rate_at_first_terminal_step": None,
        "terminal_step": terminal_step,
        "max_lift_m": max_lift_m,
        "max_grasp_streak": max_grasp_streak,
        "max_bilateral_contact_streak": max_bilateral_streak,
        "max_min_opposing_force_n": max_min_force,
        "trajectory_source": "captured_search_rollout",
        "search_rollout_path": str(path),
        "search_batch_context_preserved": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action(
    reference: torch.Tensor,
    candidates: torch.Tensor,
    *,
    capture_arm: torch.Tensor | None,
    arm_weight: float,
    close_weight: float,
    hand_active: bool,
    arm_release_weight: float = 0.0,
    body_weight: float | None = None,
    navigation_release_weight: float | None = None,
    capture_arm_release_residual_scale: float = 0.0,
) -> torch.Tensor:
    action = reference.clone()
    body = _candidate_body(candidates)
    if body is not None:
        posture_weight = arm_weight if body_weight is None else body_weight
        action[:, BODY] = reference[:, BODY] + (
            posture_weight * (1.0 - arm_release_weight) * body
        )
    navigation = _candidate_navigation(candidates)
    if navigation is not None:
        release = (
            arm_release_weight
            if navigation_release_weight is None
            else navigation_release_weight
        )
        action[:, NAVIGATION] = reference[:, NAVIGATION] + (
            arm_weight * (1.0 - release) * navigation
        )
    if capture_arm is None:
        # A release window may intentionally finish before capture: this makes
        # the sampled correction an overhead detour and returns to the known
        # grasp reference before the hand closes.  With the default zero
        # release weight this is identical to the original staged search.
        correction_weight = arm_weight * (1.0 - arm_release_weight)
        action[:, ARM] = reference[:, ARM] + correction_weight * candidates[:, :7]
    else:
        if not 0.0 <= capture_arm_release_residual_scale <= 1.0:
            raise ValueError("capture arm release residual scale must be in [0, 1]")
        release_target = reference[:, ARM] + (
            capture_arm_release_residual_scale * _candidate_capture_arm(candidates)
        )
        action[:, ARM] = torch.lerp(capture_arm, release_target, arm_release_weight)
    # The hand remains exactly on the reference before it starts closing. Once
    # the reference snaps shut, explicitly hold it open, then close to a
    # candidate aperture after the arm pose has been captured.
    if hand_active:
        open_hand = torch.zeros_like(reference[:, HAND])
        closed_hand = aperture_hand_target(_candidate_aperture(candidates))
        action[:, HAND] = torch.lerp(open_hand, closed_hand, close_weight)
    return action


@torch.inference_mode()
def replay_successful_candidate(
    args: argparse.Namespace, candidate: torch.Tensor
) -> dict[str, object]:
    """Broadcast one candidate and retain an actually successful world."""

    replay_args = argparse.Namespace(**vars(args))
    replay_envs = args.export_replay_envs or args.num_envs
    if replay_envs < 1:
        raise ValueError("export replay environments must be positive")
    replay_args.num_envs = replay_envs
    env = _environment(replay_args)
    raw_actions: list[torch.Tensor] = []
    observations: list[torch.Tensor] = []
    capture_arm: torch.Tensor | None = None
    max_lift = torch.full(
        (env.num_envs,), -torch.inf, dtype=torch.float32, device=args.device
    )
    max_grasp_streak = torch.zeros(env.num_envs, dtype=torch.long, device=args.device)
    grasp_streak = torch.zeros_like(max_grasp_streak)
    max_bilateral_streak = torch.zeros_like(max_grasp_streak)
    bilateral_streak = torch.zeros_like(max_grasp_streak)
    max_min_force = torch.zeros(env.num_envs, dtype=torch.float32, device=args.device)
    terminal_step: int | None = None
    successful_world_id: int | None = None
    successes_at_first_terminal_step = 0
    try:
        candidate = candidate.to(device=args.device, dtype=torch.float32)[None].expand(
            env.num_envs, -1
        )
        for step in range(args.max_steps):
            arm_weight, close_weight = phase_blend(
                step,
                arm_start=args.arm_start,
                capture_step=args.capture_step,
                close_end=args.close_end,
            )
            release_weight = arm_release_blend(
                step,
                release_start=args.arm_release_start,
                release_end=args.arm_release_end,
            )
            reference = env.reference.current_action()
            if step == args.capture_step:
                capture_arm = reference[:, ARM] + _candidate_capture_arm(candidate)
            effective_release_weight = _effective_arm_release_weight(
                args,
                step=step,
                capture_active=capture_arm is not None,
                overhead_release_weight=release_weight,
            )
            body_weight = body_phase_blend(
                step,
                body_start=args.body_start,
                body_full_step=args.body_full_step,
                fallback_weight=arm_weight,
            )
            observation, _ = env.state_reader.actor_observation()
            observations.append(observation.detach())
            action = _action(
                reference,
                candidate,
                capture_arm=capture_arm,
                arm_weight=arm_weight,
                close_weight=close_weight,
                hand_active=step >= args.open_hand_step,
                arm_release_weight=effective_release_weight,
                body_weight=body_weight,
                navigation_release_weight=release_weight,
                capture_arm_release_residual_scale=(
                    args.capture_arm_release_residual_scale
                ),
            )
            executed_raw = env._bounded_reference_action(action)
            raw_actions.append(executed_raw.detach())
            env.step(action)
            terms = env.last_terms
            assert terms is not None
            state = env.state_reader.actor_observation()[1]
            magnitudes = state.contact_forces_pelvis[:, 1].norm(dim=-1)
            thumb = magnitudes[:, 1:4].amax(dim=-1)
            support = magnitudes[:, 4:8].amax(dim=-1)
            min_force = torch.minimum(thumb, support)
            bilateral = (thumb > 2.0) & (support > 2.0)
            bilateral_streak = torch.where(bilateral, bilateral_streak + 1, 0)
            grasp_streak = torch.where(terms.is_grasp, grasp_streak + 1, 0)
            max_bilateral_streak = torch.maximum(max_bilateral_streak, bilateral_streak)
            max_grasp_streak = torch.maximum(max_grasp_streak, grasp_streak)
            max_min_force = torch.maximum(max_min_force, min_force)
            max_lift = torch.maximum(max_lift, terms.lift_height)
            successful_ids = terms.success.nonzero().flatten()
            if len(successful_ids):
                selection_score = (
                    max_grasp_streak[successful_ids].float()
                    + 0.1 * max_bilateral_streak[successful_ids].float()
                    + max_lift[successful_ids]
                )
                successful_world_id = int(
                    successful_ids[selection_score.argmax()].item()
                )
                successes_at_first_terminal_step = len(successful_ids)
                terminal_step = step + 1
                break
        if terminal_step is None or successful_world_id is None:
            raise RuntimeError(
                "candidate did not reproduce native 9 cm / 13-step success "
                f"in any of {env.num_envs} independent replay worlds"
            )
        world_id = successful_world_id
        selected_observations = (
            torch.stack([item[world_id] for item in observations]).cpu().numpy()
        )
        selected_raw_actions = (
            torch.stack([item[world_id] for item in raw_actions]).cpu().numpy()
        )
        if selected_observations.ndim != 2 or selected_observations.shape[1] != 331:
            raise RuntimeError(
                "staged reference replay did not produce 331-D observations"
            )
        if selected_raw_actions.ndim != 2 or selected_raw_actions.shape[1] != 36:
            raise RuntimeError(
                "staged reference replay did not produce 36-D raw actions"
            )
        return {
            "observations": selected_observations,
            "raw_actions": selected_raw_actions,
            "successful_world_id": world_id,
            "replay_envs": env.num_envs,
            "successes_at_first_terminal_step": successes_at_first_terminal_step,
            "independent_replay_success_rate_at_first_terminal_step": (
                successes_at_first_terminal_step / env.num_envs
            ),
            "terminal_step": terminal_step,
            "max_lift_m": float(max_lift[world_id].item()),
            "max_grasp_streak": int(max_grasp_streak[world_id].item()),
            "max_bilateral_contact_streak": int(max_bilateral_streak[world_id].item()),
            "max_min_opposing_force_n": float(max_min_force[world_id].item()),
        }
    finally:
        env.close()


def export_reference(
    args: argparse.Namespace,
    result: dict[str, object],
) -> dict[str, object]:
    """Write an isolated reference only after native-success reproduction."""

    output = args.export_reference.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite staged reference {output}")
    candidates = rank_export_candidates(
        result,
        limit=args.export_max_candidate_replays,
        arm_bounds=_arm_bounds(args),
        capture_arm_bounds=_capture_arm_bounds(args),
        body_bounds=_body_bounds(args),
        navigation_bounds=_navigation_bounds(args),
    )
    if not candidates:
        raise RuntimeError(
            "staged reference export requires a native-success candidate"
        )
    replay_attempts: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    replay: dict[str, object] | None = None
    selected_attempt_index: int | None = None
    for index, candidate_record in enumerate(candidates):
        try:
            if args.export_search_rollout:
                candidate_replay = load_search_rollout(candidate_record)
            else:
                candidate = _candidate_parameters(candidate_record).to(args.device)
                candidate_replay = replay_successful_candidate(args, candidate)
        except (FileNotFoundError, RuntimeError, TypeError) as error:
            replay_attempts.append(
                {
                    "rank": index,
                    "search_round": candidate_record.get("round"),
                    "search_env_id": candidate_record.get("env_id"),
                    "success_neighbor_radius": candidate_record[
                        "success_neighbor_radius"
                    ],
                    "reproduced": False,
                    "error": str(error),
                }
            )
            continue
        replay_attempts.append(
            {
                "rank": index,
                "search_round": candidate_record.get("round"),
                "search_env_id": candidate_record.get("env_id"),
                "success_neighbor_radius": candidate_record["success_neighbor_radius"],
                "reproduced": True,
                "successful_world_id": candidate_replay["successful_world_id"],
                "successes_at_first_terminal_step": candidate_replay[
                    "successes_at_first_terminal_step"
                ],
                "independent_replay_success_rate_at_first_terminal_step": candidate_replay[
                    "independent_replay_success_rate_at_first_terminal_step"
                ],
                "terminal_step": candidate_replay["terminal_step"],
                "max_lift_m": candidate_replay["max_lift_m"],
                "max_grasp_streak": candidate_replay["max_grasp_streak"],
                "max_bilateral_contact_streak": candidate_replay[
                    "max_bilateral_contact_streak"
                ],
                "max_min_opposing_force_n": candidate_replay[
                    "max_min_opposing_force_n"
                ],
            }
        )
        use_candidate = replay is None
        if replay is not None and not args.export_search_rollout:
            use_candidate = replay_quality_key(candidate_replay) > replay_quality_key(
                replay
            )
        if use_candidate:
            selected = candidate_record
            replay = candidate_replay
            selected_attempt_index = len(replay_attempts) - 1
    if selected is None or replay is None:
        raise RuntimeError(
            "refusing to export staged reference: none of "
            f"{len(candidates)} ranked native-success candidates reproduced"
        )
    assert selected_attempt_index is not None
    for index, attempt in enumerate(replay_attempts):
        attempt["selected"] = index == selected_attempt_index
    observations = replay.pop("observations")
    raw_actions = replay.pop("raw_actions")
    assert isinstance(observations, np.ndarray)
    assert isinstance(raw_actions, np.ndarray)

    source = args.reference.resolve()
    episode_path = reference_episode_path(Path("."), args.reference_episode)
    source_episode = reference_episode_path(source, args.reference_episode)
    output_episode = output / episode_path
    length = len(raw_actions)
    stage_indices = observations[:, 322:330].argmax(axis=-1)

    (output / "bc").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_episode,
        observations=observations,
        raw_actions=raw_actions,
        sample_weights=np.ones(length, dtype=np.float32),
        sample_sources=np.full(length, "staged_cem", dtype="<U10"),
        prepare_version=np.asarray(args.reference_derivation_version),
        max_stage=np.asarray(int(stage_indices.max())),
        max_lift=np.asarray(replay["max_lift_m"]),
        max_grasp_quality=np.asarray(1.0),
        terminal_success=np.asarray(True),
        terminal_failure=np.asarray(False),
        terminal_timeout=np.asarray(False),
    )
    shutil.copy2(source / "action_transform.npz", output / "action_transform.npz")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.update(
        {
            "reports": [
                {
                    "episode": args.reference_episode,
                    "success": True,
                    "failure": False,
                    "timeout": False,
                    "frames": length,
                    "terminal_step": replay["terminal_step"],
                    "max_lift": replay["max_lift_m"],
                    "max_grasp_streak": replay["max_grasp_streak"],
                    "native_success_criteria": {
                        "lift_height_m": 0.09,
                        "stable_hold_steps": 13,
                    },
                }
            ],
            "replay_gate_passed": not args.export_search_rollout,
            "search_rollout_gate_passed": True,
            # A captured search world proves native physics success, but it is
            # not an independent replay. Keep the scopes distinct so consumers
            # cannot mistake a numerically isolated success for a usable expert.
            "replay_success_rate": (
                None if args.export_search_rollout else 1.0
            ),
            "independent_replay_success_rate_at_first_terminal_step": replay[
                "independent_replay_success_rate_at_first_terminal_step"
            ],
            "independent_replay_gate_required": not args.export_search_rollout,
            "requires_independent_reference_validation": args.export_search_rollout,
            "strict_subset_source": str(source),
            "reference_derivation": {
                "version": args.reference_derivation_version,
                "search_result": str(args.output.resolve()),
                "source_reference_sha256": _sha256(source_episode),
                "right_arm_correction": selected["right_arm_correction"],
                "capture_arm_correction": selected.get("capture_arm_correction"),
                "body_correction": selected.get("body_correction"),
                "navigation_correction": selected.get("navigation_correction"),
                "aperture": selected["aperture"],
                "search_round": selected.get("round"),
                "search_env_id": selected.get("env_id"),
                "success_neighbor_radius": selected["success_neighbor_radius"],
                "phase": result["phase"],
                "environment_adaptation": result["environment_adaptation"],
                "arm_bounds": result["arm_bounds"],
                "reproduction": replay,
                "replay_attempts": replay_attempts,
                "source_immutable": True,
            },
        }
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "output": str(output),
        "reference_sha256": _sha256(output_episode),
        **replay,
    }


@torch.inference_mode()
def search(args: argparse.Namespace) -> dict[str, object]:
    env = _environment(args)
    mean, std = _initial_distribution(args)
    arm_bounds = _arm_bounds(args)
    capture_arm_bounds = _capture_arm_bounds(args)
    body_bounds = _body_bounds(args)
    navigation_bounds = _navigation_bounds(args)
    parameter_names = _parameter_names(
        search_capture_arm=args.search_capture_arm,
        search_body=args.search_body,
        search_navigation=args.search_navigation,
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 97)
    all_ids = torch.arange(args.num_envs, device=args.device)
    elite_count = max(8, round(args.num_envs * args.elite_fraction))
    global_best: dict[str, object] | None = None
    rounds: list[dict[str, object]] = []

    try:
        for round_index in range(args.rounds):
            if round_index:
                env._reset(all_ids)
            candidates = _sample_candidates(
                mean,
                std,
                count=args.num_envs,
                generator=generator,
                arm_bounds=arm_bounds,
                capture_arm_bounds=capture_arm_bounds,
                body_bounds=body_bounds,
                navigation_bounds=navigation_bounds,
            )
            if args.replay_best_candidate_replicas:
                replay_candidate = (
                    _candidate_parameters(global_best).to(args.device)
                    if global_best is not None
                    else mean
                )
                replay_candidate_across_worlds(
                    candidates,
                    replay_candidate,
                    replicas=args.replay_best_candidate_replicas,
                )
            max_score = torch.full((args.num_envs,), -torch.inf, device=args.device)
            max_geometry = torch.zeros_like(max_score)
            max_min_force = torch.zeros_like(max_score)
            max_lift = torch.full_like(max_score, -torch.inf)
            min_balanced_shell_gap = torch.full_like(max_score, torch.inf)
            min_thumb_shell_gap = torch.full_like(max_score, torch.inf)
            min_support_shell_gap = torch.full_like(max_score, torch.inf)
            min_thumb_center_distance = torch.full_like(max_score, torch.inf)
            min_support_center_distance = torch.full_like(max_score, torch.inf)
            best_thumb_shell_gap = torch.full_like(max_score, torch.inf)
            best_support_shell_gap = torch.full_like(max_score, torch.inf)
            best_thumb_center_distance = torch.full_like(max_score, torch.inf)
            best_support_center_distance = torch.full_like(max_score, torch.inf)
            best_opposition = torch.zeros_like(max_score)
            best_arm_action_saturation = torch.zeros_like(max_score)
            bilateral = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
            grasp = torch.zeros_like(bilateral)
            success = torch.zeros_like(bilateral)
            numerical_failure = torch.zeros_like(bilateral)
            bilateral_streak = torch.zeros(
                args.num_envs, dtype=torch.long, device=args.device
            )
            grasp_streak = torch.zeros_like(bilateral_streak)
            max_bilateral_streak = torch.zeros_like(bilateral_streak)
            max_grasp_streak = torch.zeros_like(bilateral_streak)
            first_success_step = torch.zeros_like(bilateral_streak)
            best_step = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
            capture_arm: torch.Tensor | None = None
            rollout_observations: list[torch.Tensor] = []
            rollout_raw_actions: list[torch.Tensor] = []

            for step in range(args.max_steps):
                arm_weight, close_weight = phase_blend(
                    step,
                    arm_start=args.arm_start,
                    capture_step=args.capture_step,
                    close_end=args.close_end,
                )
                arm_release_weight = arm_release_blend(
                    step,
                    release_start=args.arm_release_start,
                    release_end=args.arm_release_end,
                )
                reference = env.reference.current_action()
                if step == args.capture_step:
                    capture_arm = reference[:, ARM] + _candidate_capture_arm(candidates)
                effective_release_weight = _effective_arm_release_weight(
                    args,
                    step=step,
                    capture_active=capture_arm is not None,
                    overhead_release_weight=arm_release_weight,
                )
                body_weight = body_phase_blend(
                    step,
                    body_start=args.body_start,
                    body_full_step=args.body_full_step,
                    fallback_weight=arm_weight,
                )
                action = _action(
                    reference,
                    candidates,
                    capture_arm=capture_arm,
                    arm_weight=arm_weight,
                    close_weight=close_weight,
                    hand_active=step >= args.open_hand_step,
                    arm_release_weight=effective_release_weight,
                    body_weight=body_weight,
                    navigation_release_weight=arm_release_weight,
                    capture_arm_release_residual_scale=(
                        args.capture_arm_release_residual_scale
                    ),
                )
                if args.export_search_rollout:
                    observation, _ = env.state_reader.actor_observation()
                    rollout_observations.append(observation.detach().clone())
                    rollout_raw_actions.append(
                        env._bounded_reference_action(action).detach().clone()
                    )
                arm_action_saturation = (action[:, ARM].abs() >= 1.0).float().mean(-1)
                _, _, _, _ = env.step(action)
                terms = env.last_terms
                assert terms is not None
                state = env.state_reader.actor_observation()[1]
                magnitudes = state.contact_forces_pelvis[:, 1].norm(dim=-1)
                thumb = magnitudes[:, 1:4].amax(dim=-1)
                support = magnitudes[:, 4:8].amax(dim=-1)
                min_force = torch.minimum(thumb, support)
                physical_bilateral = (thumb > 2.0) & (support > 2.0)
                grasp_center = state.primary.pos_w + torch.einsum(
                    "bij,j->bi", state.primary.rot_w, env.reward.grasp_frame_position
                )
                distances, shell_gap, shell_balanced_gap = fingertip_shell_metrics(
                    state.distal_pos_w[:, 1],
                    grasp_center,
                    grip_width_m=env.reward.grip_width_m,
                )
                surface_distances = state.fingertip_distances[:, 1]
                balanced_gap = (
                    balanced_surface_gap(surface_distances)
                    if args.geometry_mode == "surface"
                    else shell_balanced_gap
                )
                thumb_shell_gap = shell_gap[:, 0]
                support_shell_gap = shell_gap[:, 1:].amin(-1)
                thumb_center_distance = distances[:, 0]
                support_center_distance = distances[:, 1:].amin(-1)
                balanced_shell = torch.exp(-40.0 * balanced_gap)
                geometry = balanced_shell * (
                    0.20 + 0.80 * env.reward.last_finger_opposition
                )
                bilateral_streak = torch.where(
                    physical_bilateral, bilateral_streak + 1, 0
                )
                grasp_streak = torch.where(terms.is_grasp, grasp_streak + 1, 0)
                max_bilateral_streak = torch.maximum(
                    max_bilateral_streak, bilateral_streak
                )
                max_grasp_streak = torch.maximum(max_grasp_streak, grasp_streak)
                score = candidate_score(
                    success=success | terms.success,
                    max_grasp_streak=max_grasp_streak,
                    max_bilateral_streak=max_bilateral_streak,
                    min_opposing_force=min_force,
                    balanced_shell_gap=balanced_gap,
                    finger_opposition=env.reward.last_finger_opposition,
                    lift_height=terms.lift_height,
                    grasp_streak_score_cap=args.grasp_streak_score_cap,
                )
                if step < args.capture_step:
                    score.fill_(-torch.inf)
                improved = score > max_score
                max_score = torch.maximum(max_score, score)
                best_step[improved] = step + 1
                best_thumb_shell_gap[improved] = thumb_shell_gap[improved]
                best_support_shell_gap[improved] = support_shell_gap[improved]
                best_thumb_center_distance[improved] = thumb_center_distance[improved]
                best_support_center_distance[improved] = support_center_distance[
                    improved
                ]
                best_opposition[improved] = env.reward.last_finger_opposition[improved]
                best_arm_action_saturation[improved] = arm_action_saturation[improved]
                max_geometry = torch.maximum(max_geometry, geometry)
                max_min_force = torch.maximum(max_min_force, min_force)
                max_lift = torch.maximum(max_lift, terms.lift_height)
                min_balanced_shell_gap = torch.minimum(
                    min_balanced_shell_gap, balanced_gap
                )
                min_thumb_shell_gap = torch.minimum(
                    min_thumb_shell_gap, thumb_shell_gap
                )
                min_support_shell_gap = torch.minimum(
                    min_support_shell_gap, support_shell_gap
                )
                min_thumb_center_distance = torch.minimum(
                    min_thumb_center_distance, thumb_center_distance
                )
                min_support_center_distance = torch.minimum(
                    min_support_center_distance, support_center_distance
                )
                bilateral.logical_or_(physical_bilateral)
                grasp.logical_or_(terms.is_grasp)
                success.logical_or_(terms.success)
                first_success_step = torch.where(
                    (first_success_step == 0) & terms.success,
                    torch.full_like(first_success_step, step + 1),
                    first_success_step,
                )
                numerical_failure.logical_or_(env.last_numerical_failure)

            elite_ids = torch.topk(max_score, elite_count).indices
            elite = candidates[elite_ids]
            elite_mean = elite.mean(dim=0)
            elite_std = elite.std(dim=0, unbiased=False).clamp_min(args.minimum_std)
            mean = args.cem_momentum * mean + (1.0 - args.cem_momentum) * elite_mean
            std = args.cem_momentum * std + (1.0 - args.cem_momentum) * elite_std

            best_id = int(max_score.argmax().item())
            parameter_bound_mask = _parameter_bound_mask(
                candidates,
                arm_bounds=arm_bounds,
                capture_arm_bounds=capture_arm_bounds,
                body_bounds=body_bounds,
                navigation_bounds=navigation_bounds,
            )

            # This helper is consumed synchronously before the next CEM round.
            def candidate_record(
                candidate_id: int,
            ) -> dict[str, object]:
                candidate = candidates[candidate_id]
                bound_parameters = [
                    name
                    for name, saturated in zip(
                        parameter_names,
                        parameter_bound_mask[candidate_id].cpu().tolist(),
                        strict=True,
                    )
                    if saturated
                ]
                return {
                    "round": round_index,
                    "env_id": candidate_id,
                    "score": float(max_score[candidate_id].item()),
                    "best_step": int(best_step[candidate_id].item()),
                    "had_grasp": bool(grasp[candidate_id].item()),
                    "had_bilateral_contact": bool(bilateral[candidate_id].item()),
                    "had_success": bool(success[candidate_id].item()),
                    "max_geometry": float(max_geometry[candidate_id].item()),
                    "minimum_balanced_shell_gap_m": float(
                        min_balanced_shell_gap[candidate_id].item()
                    ),
                    "minimum_thumb_shell_gap_m": float(
                        min_thumb_shell_gap[candidate_id].item()
                    ),
                    "minimum_support_shell_gap_m": float(
                        min_support_shell_gap[candidate_id].item()
                    ),
                    "minimum_thumb_center_distance_m": float(
                        min_thumb_center_distance[candidate_id].item()
                    ),
                    "minimum_support_center_distance_m": float(
                        min_support_center_distance[candidate_id].item()
                    ),
                    "best_step_thumb_shell_gap_m": float(
                        best_thumb_shell_gap[candidate_id].item()
                    ),
                    "best_step_support_shell_gap_m": float(
                        best_support_shell_gap[candidate_id].item()
                    ),
                    "best_step_thumb_center_distance_m": float(
                        best_thumb_center_distance[candidate_id].item()
                    ),
                    "best_step_support_center_distance_m": float(
                        best_support_center_distance[candidate_id].item()
                    ),
                    "best_step_finger_opposition": float(
                        best_opposition[candidate_id].item()
                    ),
                    "best_step_arm_action_saturation_fraction": float(
                        best_arm_action_saturation[candidate_id].item()
                    ),
                    "candidate_bound_parameters": bound_parameters,
                    "max_bilateral_contact_streak": int(
                        max_bilateral_streak[candidate_id].item()
                    ),
                    "max_grasp_streak": int(max_grasp_streak[candidate_id].item()),
                    "max_min_opposing_force_n": float(
                        max_min_force[candidate_id].item()
                    ),
                    "max_lift_m": float(max_lift[candidate_id].item()),
                    "terminal_step": int(first_success_step[candidate_id].item()),
                    "right_arm_correction": candidate[:7].cpu().tolist(),
                    "capture_arm_correction": (
                        _candidate_capture_arm(candidate).cpu().tolist()
                        if args.search_capture_arm
                        else None
                    ),
                    "body_correction": (
                        _candidate_body(candidate).cpu().tolist()
                        if args.search_body
                        else None
                    ),
                    "navigation_correction": (
                        _candidate_navigation(candidate).cpu().tolist()
                        if args.search_navigation
                        else None
                    ),
                    "aperture": _candidate_aperture(candidate).cpu().tolist(),
                    "closed_hand_target": aperture_hand_target(
                        _candidate_aperture(candidate)[None]
                    )[0]
                    .cpu()
                    .tolist(),
                }

            successful_ids = success.nonzero().flatten().cpu().tolist()
            successful_records = [candidate_record(index) for index in successful_ids]
            if successful_records:
                successful_parameters = torch.stack(
                    [_candidate_parameters(item) for item in successful_records]
                )
                widths = torch.tensor(
                    [
                        high - low
                        for low, high in (
                            *arm_bounds,
                            *capture_arm_bounds,
                            *body_bounds,
                            *navigation_bounds,
                            *APERTURE_BOUNDS,
                        )
                    ]
                )
                center = successful_parameters.median(dim=0).values
                for item, distance in zip(
                    successful_records,
                    ((successful_parameters - center) / widths).norm(dim=1).tolist(),
                    strict=True,
                ):
                    item["round_success_center_distance"] = distance
                successful_records.sort(
                    key=lambda item: (
                        len(item["candidate_bound_parameters"]) > 0,
                        item["round_success_center_distance"],
                        -item["max_grasp_streak"],
                        -item["max_lift_m"],
                    )
                )
                successful_records = successful_records[
                    : args.retained_success_candidates
                ]
                if args.export_search_rollout:
                    rollout_dir = search_rollout_path(
                        args.output, round_index=round_index, env_id=0
                    ).parent
                    rollout_dir.mkdir(parents=True, exist_ok=True)
                    for item in successful_records:
                        candidate_id = int(item["env_id"])
                        terminal_step = int(item["terminal_step"])
                        if terminal_step < 1:
                            raise RuntimeError(
                                "native success did not retain its terminal step"
                            )
                        path = search_rollout_path(
                            args.output,
                            round_index=round_index,
                            env_id=candidate_id,
                        )
                        np.savez_compressed(
                            path,
                            observations=torch.stack(
                                [
                                    frame[candidate_id]
                                    for frame in rollout_observations[:terminal_step]
                                ]
                            )
                            .cpu()
                            .numpy(),
                            raw_actions=torch.stack(
                                [
                                    frame[candidate_id]
                                    for frame in rollout_raw_actions[:terminal_step]
                                ]
                            )
                            .cpu()
                            .numpy(),
                            terminal_success=np.asarray(True),
                            terminal_step=np.asarray(terminal_step),
                            max_lift_m=np.asarray(item["max_lift_m"]),
                            max_grasp_streak=np.asarray(item["max_grasp_streak"]),
                            max_bilateral_contact_streak=np.asarray(
                                item["max_bilateral_contact_streak"]
                            ),
                            max_min_opposing_force_n=np.asarray(
                                item["max_min_opposing_force_n"]
                            ),
                        )
                        item["search_rollout_path"] = str(path)
            record = {
                "round": round_index,
                "grasp_candidates": int(grasp.sum().item()),
                "bilateral_contact_candidates": int(bilateral.sum().item()),
                "success_candidates": int(success.sum().item()),
                "numerical_failures": int(numerical_failure.sum().item()),
                "best": candidate_record(best_id),
                "successful_candidates": successful_records,
                "distribution": {
                    "mean": mean.cpu().tolist(),
                    "std": std.cpu().tolist(),
                    "parameter_bound_saturation_fraction": {
                        name: fraction
                        for name, fraction in zip(
                            parameter_names,
                            parameter_bound_mask.float().mean(0).cpu().tolist(),
                            strict=True,
                        )
                    },
                },
            }
            rounds.append(record)
            if global_best is None or record["best"]["score"] > global_best["score"]:
                global_best = record["best"]
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"distribution", "successful_candidates"}
                    }
                )
            )
            if should_stop_search(
                record,
                stop_grasp_candidates=args.stop_grasp_candidates,
                stop_success_candidates=args.stop_success_candidates,
            ):
                break
    finally:
        env.close()

    assert global_best is not None
    result = {
        "schema_version": 1,
        "asset": str(args.asset.resolve()),
        "reference_episode": args.reference_episode,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "rounds_requested": args.rounds,
        "rounds_completed": len(rounds),
        "elite_count": elite_count,
        "search_stopping": {
            "stop_grasp_candidates": args.stop_grasp_candidates,
            "stop_success_candidates": args.stop_success_candidates,
            "grasp_streak_score_cap": args.grasp_streak_score_cap,
            "replay_best_candidate_replicas": args.replay_best_candidate_replicas,
            "export_search_rollout": args.export_search_rollout,
        },
        "phase": {
            "arm_start": args.arm_start,
            "open_hand_step": args.open_hand_step,
            "capture_step": args.capture_step,
            "close_end": args.close_end,
            "arm_release_start": args.arm_release_start,
            "arm_release_end": args.arm_release_end,
            "capture_arm_release_start": args.capture_arm_release_start,
            "capture_arm_release_end": args.capture_arm_release_end,
            "capture_arm_release_residual_scale": (
                args.capture_arm_release_residual_scale
            ),
            "body_start": args.body_start,
            "body_full_step": args.body_full_step,
        },
        "geometry_mode": args.geometry_mode,
        "arm_bounds": [list(bounds) for bounds in arm_bounds],
        "capture_arm_bounds": [list(bounds) for bounds in capture_arm_bounds],
        "body_bounds": [list(bounds) for bounds in body_bounds],
        "navigation_bounds": [list(bounds) for bounds in navigation_bounds],
        "environment_adaptation": {
            "target_position_jitter_xy": list(args.target_position_jitter_xy),
            "target_position_offset_center_xy": list(
                args.target_position_offset_center_xy
            ),
            "target_yaw_jitter": args.target_yaw_jitter,
            "robot_base_position_jitter_xy": list(args.robot_base_position_jitter_xy),
            "robot_base_yaw_jitter": args.robot_base_yaw_jitter,
            "reference_target_x_arm_gains": list(args.reference_target_x_arm_gains),
            "reference_target_y_arm_gains": list(args.reference_target_y_arm_gains),
            "reference_target_positive_y_arm_gains": list(
                args.reference_target_positive_y_arm_gains
            ),
        },
        "global_best": global_best,
        "rounds": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.export_reference is not None:
        result["reference_export"] = export_reference(args, result)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--reference-episode",
        type=int,
        default=82,
        help="Strict reference episode to load and preserve on export",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-reference", type=Path)
    parser.add_argument("--export-replay-envs", type=int)
    parser.add_argument(
        "--export-search-rollout",
        action="store_true",
        help=(
            "Export the actual native-success trajectory captured in its original "
            "search batch instead of requiring a changed broadcast batch to match"
        ),
    )
    parser.add_argument("--export-max-candidate-replays", type=int, default=32)
    parser.add_argument("--retained-success-candidates", type=int, default=64)
    parser.add_argument(
        "--reference-derivation-version",
        default="apple_staged_native_success_v1",
        help="Explicit provenance label stored only when exporting a reference",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260860)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--elite-fraction", type=float, default=0.05)
    parser.add_argument("--cem-momentum", type=float, default=0.25)
    parser.add_argument("--minimum-std", type=float, default=0.008)
    parser.add_argument("--stop-grasp-candidates", type=int, default=8)
    parser.add_argument(
        "--stop-success-candidates",
        type=int,
        help=(
            "When set, replace the legacy grasp-candidate early stop with an "
            "early stop requiring this many native successes"
        ),
    )
    parser.add_argument(
        "--grasp-streak-score-cap",
        type=int,
        default=20,
        help=(
            "Maximum grasp/contact streak rewarded by CEM; increase for lift "
            "searches that must retain contact through a release window"
        ),
    )
    parser.add_argument(
        "--replay-best-candidate-replicas",
        type=int,
        default=0,
        help=(
            "Evaluate the current best candidate in this many worlds each CEM "
            "round; zero preserves legacy independent sampling"
        ),
    )
    parser.add_argument(
        "--geometry-mode",
        choices=("shell", "surface"),
        default="shell",
        help=(
            "Pre-contact CEM geometry: object-centered shell or actual "
            "finger-to-collision-surface distance"
        ),
    )
    parser.add_argument("--arm-start", type=int, default=120)
    parser.add_argument("--open-hand-step", type=int, default=164)
    parser.add_argument("--capture-step", type=int, default=193)
    parser.add_argument("--close-end", type=int, default=233)
    parser.add_argument("--arm-release-start", type=int)
    parser.add_argument("--arm-release-end", type=int)
    parser.add_argument("--capture-arm-release-start", type=int)
    parser.add_argument("--capture-arm-release-end", type=int)
    parser.add_argument(
        "--capture-arm-release-residual-scale",
        type=float,
        default=0.0,
        help=(
            "Fraction of the capture correction retained after release; zero "
            "preserves the legacy return to the unshifted reference arm"
        ),
    )
    parser.add_argument(
        "--target-position-offset-center-xy", type=float, nargs=2, default=(0.087, 0.0)
    )
    parser.add_argument(
        "--target-position-jitter-xy", type=float, nargs=2, default=(0.0, 0.0)
    )
    parser.add_argument("--target-yaw-jitter", type=float, default=0.0)
    parser.add_argument(
        "--robot-base-position-jitter-xy", type=float, nargs=2, default=(0.0, 0.0)
    )
    parser.add_argument("--robot-base-yaw-jitter", type=float, default=0.0)
    parser.add_argument(
        "--reference-target-x-arm-gains", type=float, nargs=2, default=(-5.3, 2.4)
    )
    parser.add_argument(
        "--reference-target-y-arm-gains", type=float, nargs=2, default=(12.0, 0.0)
    )
    parser.add_argument(
        "--reference-target-positive-y-arm-gains",
        type=float,
        nargs=2,
        default=(22.0, 0.0),
    )
    parser.add_argument("--base-arm", type=float, nargs=7, required=True)
    parser.add_argument(
        "--search-capture-arm",
        action="store_true",
        help=(
            "Search a second right-arm correction held after the overhead "
            "detour returns to the grasp reference"
        ),
    )
    parser.add_argument(
        "--base-capture-arm",
        type=float,
        nargs=7,
        default=(0.0,) * 7,
    )
    parser.add_argument(
        "--arm-bounds",
        type=float,
        nargs=14,
        default=tuple(value for bounds in ARM_BOUNDS for value in bounds),
        help=(
            "Optional per-joint residual low/high pairs; defaults preserve the "
            "audited episode-82 search domain"
        ),
    )
    parser.add_argument(
        "--arm-std",
        type=float,
        nargs=7,
        default=(0.06, 0.035, 0.06, 0.06, 0.10, 0.10, 0.12),
    )
    parser.add_argument(
        "--capture-arm-bounds",
        type=float,
        nargs=14,
        default=tuple(value for _ in range(7) for value in (-0.25, 0.25)),
    )
    parser.add_argument(
        "--capture-arm-std",
        type=float,
        nargs=7,
        default=(0.08,) * 7,
    )
    parser.add_argument(
        "--search-body",
        action="store_true",
        help=(
            "Search opt-in torso roll/pitch/yaw and base-height corrections; "
            "omitted preserves the legacy arm/hand parameter layout"
        ),
    )
    parser.add_argument(
        "--base-body",
        type=float,
        nargs=4,
        default=(0.0,) * 4,
    )
    parser.add_argument(
        "--body-bounds",
        type=float,
        nargs=8,
        default=tuple(value for _ in range(4) for value in (-1.0, 1.0)),
        help="Per-action low/high bounds for torso R/P/Y and base height",
    )
    parser.add_argument(
        "--body-std",
        type=float,
        nargs=4,
        default=(0.25,) * 4,
    )
    parser.add_argument(
        "--body-start",
        type=int,
        help="Optional start step for an independent early posture ramp",
    )
    parser.add_argument(
        "--body-full-step",
        type=int,
        help="Optional step where the independent posture correction is complete",
    )
    parser.add_argument(
        "--search-navigation",
        action="store_true",
        help=(
            "Search opt-in torso XY velocity corrections during the overhead "
            "approach; omitted preserves every legacy candidate layout"
        ),
    )
    parser.add_argument(
        "--base-navigation",
        type=float,
        nargs=2,
        default=(0.0,) * 2,
    )
    parser.add_argument(
        "--navigation-bounds",
        type=float,
        nargs=4,
        default=(-1.0, 1.0, -1.0, 1.0),
        help="Per-action low/high bounds for torso X/Y velocity",
    )
    parser.add_argument(
        "--navigation-std",
        type=float,
        nargs=2,
        default=(0.25,) * 2,
    )
    parser.add_argument(
        "--aperture-mean", type=float, nargs=4, default=(0.35, 0.25, 0.25, 0.25)
    )
    parser.add_argument(
        "--aperture-std", type=float, nargs=4, default=(0.15, 0.15, 0.16, 0.16)
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = search(args)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rounds"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
