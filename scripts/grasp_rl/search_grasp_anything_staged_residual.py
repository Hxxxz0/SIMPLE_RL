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


def _initial_distribution(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(
        [*args.base_arm, *args.aperture_mean],
        dtype=torch.float32,
        device=args.device,
    )
    std = torch.tensor(
        [*args.arm_std, *args.aperture_std],
        dtype=torch.float32,
        device=args.device,
    )
    if mean.shape != (PARAMETER_COUNT,) or std.shape != (PARAMETER_COUNT,):
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
) -> torch.Tensor:
    if count < 32:
        raise ValueError("staged residual search requires at least 32 candidates")
    candidates = mean + std * torch.randn(
        count, PARAMETER_COUNT, device=mean.device, generator=generator
    )
    candidates[0] = mean
    bounds = (*ARM_BOUNDS, *APERTURE_BOUNDS)
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


def _environment(args: argparse.Namespace) -> GpuGraspVecEnv:
    dr = DomainRandomizationConfig(
        enabled=True,
        target_position_jitter_xy=(0.0, 0.0),
        target_position_offset_center_xy=tuple(args.target_position_offset_center_xy),
        target_yaw_jitter=0.0,
        destination_position_jitter_xy=(0.0, 0.0),
        destination_yaw_jitter=0.0,
        distractor_position_jitter_xy=(0.0, 0.0),
        distractor_yaw_jitter=0.0,
        robot_base_position_jitter_xy=(0.0, 0.0),
        robot_base_yaw_jitter=0.0,
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
        strict_reference_episode=82,
        max_reference_initial_position_offset=0.12,
        reference_reward_weight=0.005,
        # Episode 82 closes the hand to approximately +/-0.99 in one frame.
        # Apple search must be able to reopen that proposal completely. This is
        # local to this diagnostic and does not change any training default.
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
    return torch.maximum(
        surface_distances[:, 0], surface_distances[:, 1:].amin(dim=-1)
    )


def candidate_score(
    *,
    success: torch.Tensor,
    max_grasp_streak: torch.Tensor,
    max_bilateral_streak: torch.Tensor,
    min_opposing_force: torch.Tensor,
    balanced_shell_gap: torch.Tensor,
    finger_opposition: torch.Tensor,
    lift_height: torch.Tensor,
) -> torch.Tensor:
    """Rank physical contact first and otherwise minimize both shell gaps."""

    physical_lift_gate = (max_grasp_streak >= 2) | (max_bilateral_streak >= 2)
    return (
        2500.0 * success.float()
        + 1000.0 * (max_grasp_streak >= 2).float()
        + 300.0 * (max_bilateral_streak >= 2).float()
        + 10.0 * max_grasp_streak.float().clamp_max(20)
        + 5.0 * max_bilateral_streak.float().clamp_max(20)
        + 20.0 * (min_opposing_force / 8.0).clamp(0.0, 1.0)
        - 100.0 * balanced_shell_gap
        + 2.0 * finger_opposition.clamp(0.0, 1.0)
        + 500.0
        * physical_lift_gate.float()
        * (lift_height / 0.09).clamp(0.0, 1.0)
    )


def _parameter_bound_mask(candidates: torch.Tensor) -> torch.Tensor:
    bounds = torch.tensor(
        (*ARM_BOUNDS, *APERTURE_BOUNDS),
        dtype=candidates.dtype,
        device=candidates.device,
    )
    return torch.isclose(
        candidates, bounds[:, 0], atol=1e-6, rtol=0.0
    ) | torch.isclose(candidates, bounds[:, 1], atol=1e-6, rtol=0.0)


def _candidate_parameters(candidate: dict[str, object]) -> torch.Tensor:
    return torch.tensor(
        [*candidate["right_arm_correction"], *candidate["aperture"]],
        dtype=torch.float32,
    )


def rank_export_candidates(
    result: dict[str, object], *, limit: int
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
        if any(torch.allclose(parameters, prior, atol=1e-7, rtol=0.0) for prior in unique_parameters):
            continue
        unique.append(candidate)
        unique_parameters.append(parameters)

    bounds = torch.tensor((*ARM_BOUNDS, *APERTURE_BOUNDS), dtype=torch.float32)
    widths = bounds[:, 1] - bounds[:, 0]
    normalized = torch.stack(unique_parameters) / widths
    pairwise = torch.cdist(normalized, normalized)
    neighbor_count = min(5, max(len(unique) - 1, 0))
    ranked: list[dict[str, object]] = []
    for index, candidate in enumerate(unique):
        if neighbor_count:
            distances = pairwise[index][torch.arange(len(unique)) != index]
            neighbor_radius = float(distances.topk(neighbor_count, largest=False).values.mean())
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


def replay_quality_key(replay: dict[str, object]) -> tuple[float, int, int, float, float]:
    """Rank independent replays by repeatability, then physical margin."""

    return (
        float(replay["independent_replay_success_rate_at_first_terminal_step"]),
        int(replay["max_grasp_streak"]),
        int(replay["max_bilateral_contact_streak"]),
        float(replay["max_lift_m"]),
        float(replay["max_min_opposing_force_n"]),
    )


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
) -> torch.Tensor:
    action = reference.clone()
    if capture_arm is None:
        action[:, ARM] = reference[:, ARM] + arm_weight * candidates[:, :7]
    else:
        action[:, ARM] = torch.lerp(
            capture_arm, reference[:, ARM], arm_release_weight
        )
    # The hand remains exactly on the reference before it starts closing. Once
    # episode 82 snaps shut, explicitly hold it open, then close to a candidate
    # aperture after the arm pose has been captured.
    if hand_active:
        open_hand = torch.zeros_like(reference[:, HAND])
        closed_hand = aperture_hand_target(candidates[:, 7:])
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
    max_grasp_streak = torch.zeros(
        env.num_envs, dtype=torch.long, device=args.device
    )
    grasp_streak = torch.zeros_like(max_grasp_streak)
    max_bilateral_streak = torch.zeros_like(max_grasp_streak)
    bilateral_streak = torch.zeros_like(max_grasp_streak)
    max_min_force = torch.zeros(
        env.num_envs, dtype=torch.float32, device=args.device
    )
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
                capture_arm = reference[:, ARM] + candidate[:, :7]
            observation, _ = env.state_reader.actor_observation()
            observations.append(observation.detach())
            action = _action(
                reference,
                candidate,
                capture_arm=capture_arm,
                arm_weight=arm_weight,
                close_weight=close_weight,
                hand_active=step >= args.open_hand_step,
                arm_release_weight=release_weight,
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
            max_bilateral_streak = torch.maximum(
                max_bilateral_streak, bilateral_streak
            )
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
        selected_observations = torch.stack(
            [item[world_id] for item in observations]
        ).cpu().numpy()
        selected_raw_actions = torch.stack(
            [item[world_id] for item in raw_actions]
        ).cpu().numpy()
        if selected_observations.ndim != 2 or selected_observations.shape[1] != 331:
            raise RuntimeError("staged reference replay did not produce 331-D observations")
        if selected_raw_actions.ndim != 2 or selected_raw_actions.shape[1] != 36:
            raise RuntimeError("staged reference replay did not produce 36-D raw actions")
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
            "max_bilateral_contact_streak": int(
                max_bilateral_streak[world_id].item()
            ),
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
        result, limit=args.export_max_candidate_replays
    )
    if not candidates:
        raise RuntimeError("staged reference export requires a native-success candidate")
    replay_attempts: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    replay: dict[str, object] | None = None
    selected_attempt_index: int | None = None
    for index, candidate_record in enumerate(candidates):
        candidate = _candidate_parameters(candidate_record).to(args.device)
        try:
            candidate_replay = replay_successful_candidate(args, candidate)
        except RuntimeError as error:
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
                "success_neighbor_radius": candidate_record[
                    "success_neighbor_radius"
                ],
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
        if replay is None or replay_quality_key(candidate_replay) > replay_quality_key(
            replay
        ):
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
    source_episode = source / "bc/episode_000082.npz"
    length = len(raw_actions)
    stage_indices = observations[:, 322:330].argmax(axis=-1)

    (output / "bc").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "bc/episode_000082.npz",
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
                    "episode": 82,
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
            "replay_gate_passed": True,
            # This rate describes the one exported episode in reports. The
            # independent batch reproduction rate is recorded separately.
            "replay_success_rate": 1.0,
            "independent_replay_success_rate_at_first_terminal_step": replay[
                "independent_replay_success_rate_at_first_terminal_step"
            ],
            "strict_subset_source": str(source),
            "reference_derivation": {
                "version": args.reference_derivation_version,
                "search_result": str(args.output.resolve()),
                "source_reference_sha256": _sha256(source_episode),
                "right_arm_correction": selected["right_arm_correction"],
                "aperture": selected["aperture"],
                "search_round": selected.get("round"),
                "search_env_id": selected.get("env_id"),
                "success_neighbor_radius": selected["success_neighbor_radius"],
                "phase": result["phase"],
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
        "reference_sha256": _sha256(output / "bc/episode_000082.npz"),
        **replay,
    }


@torch.inference_mode()
def search(args: argparse.Namespace) -> dict[str, object]:
    env = _environment(args)
    mean, std = _initial_distribution(args)
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
                mean, std, count=args.num_envs, generator=generator
            )
            max_score = torch.full(
                (args.num_envs,), -torch.inf, device=args.device
            )
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
            bilateral = torch.zeros(
                args.num_envs, dtype=torch.bool, device=args.device
            )
            grasp = torch.zeros_like(bilateral)
            success = torch.zeros_like(bilateral)
            numerical_failure = torch.zeros_like(bilateral)
            bilateral_streak = torch.zeros(
                args.num_envs, dtype=torch.long, device=args.device
            )
            grasp_streak = torch.zeros_like(bilateral_streak)
            max_bilateral_streak = torch.zeros_like(bilateral_streak)
            max_grasp_streak = torch.zeros_like(bilateral_streak)
            best_step = torch.zeros(
                args.num_envs, dtype=torch.long, device=args.device
            )
            capture_arm: torch.Tensor | None = None

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
                    capture_arm = reference[:, ARM] + candidates[:, :7]
                action = _action(
                    reference,
                    candidates,
                    capture_arm=capture_arm,
                    arm_weight=arm_weight,
                    close_weight=close_weight,
                    hand_active=step >= args.open_hand_step,
                    arm_release_weight=arm_release_weight,
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
                )
                if step < args.capture_step:
                    score.fill_(-torch.inf)
                improved = score > max_score
                max_score = torch.maximum(max_score, score)
                best_step[improved] = step + 1
                best_thumb_shell_gap[improved] = thumb_shell_gap[improved]
                best_support_shell_gap[improved] = support_shell_gap[improved]
                best_thumb_center_distance[improved] = thumb_center_distance[improved]
                best_support_center_distance[improved] = support_center_distance[improved]
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
                numerical_failure.logical_or_(env.last_numerical_failure)

            elite_ids = torch.topk(max_score, elite_count).indices
            elite = candidates[elite_ids]
            elite_mean = elite.mean(dim=0)
            elite_std = elite.std(dim=0, unbiased=False).clamp_min(args.minimum_std)
            mean = args.cem_momentum * mean + (1.0 - args.cem_momentum) * elite_mean
            std = args.cem_momentum * std + (1.0 - args.cem_momentum) * elite_std

            best_id = int(max_score.argmax().item())
            parameter_bound_mask = _parameter_bound_mask(candidates)

            # This helper is consumed synchronously before the next CEM round.
            def candidate_record(
                candidate_id: int,
            ) -> dict[str, object]:
                candidate = candidates[candidate_id]
                bound_parameters = [
                    name
                    for name, saturated in zip(
                        PARAMETER_NAMES,
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
                    "right_arm_correction": candidate[:7].cpu().tolist(),
                    "aperture": candidate[7:].cpu().tolist(),
                    "closed_hand_target": aperture_hand_target(
                        candidate[None, 7:]
                    )[0].cpu().tolist(),
                }

            successful_ids = success.nonzero().flatten().cpu().tolist()
            successful_records = [candidate_record(index) for index in successful_ids]
            if successful_records:
                successful_parameters = torch.stack(
                    [_candidate_parameters(item) for item in successful_records]
                )
                widths = torch.tensor(
                    [high - low for low, high in (*ARM_BOUNDS, *APERTURE_BOUNDS)]
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
                            PARAMETER_NAMES,
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
            if record["grasp_candidates"] >= args.stop_grasp_candidates:
                break
    finally:
        env.close()

    assert global_best is not None
    result = {
        "schema_version": 1,
        "asset": str(args.asset.resolve()),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "rounds_requested": args.rounds,
        "rounds_completed": len(rounds),
        "elite_count": elite_count,
        "phase": {
            "arm_start": args.arm_start,
            "open_hand_step": args.open_hand_step,
            "capture_step": args.capture_step,
            "close_end": args.close_end,
            "arm_release_start": args.arm_release_start,
            "arm_release_end": args.arm_release_end,
        },
        "geometry_mode": args.geometry_mode,
        "global_best": global_best,
        "rounds": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.export_reference is not None:
        result["reference_export"] = export_reference(args, result)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-reference", type=Path)
    parser.add_argument("--export-replay-envs", type=int)
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
    parser.add_argument("--target-position-offset-center-xy", type=float, nargs=2, default=(0.087, 0.0))
    parser.add_argument("--reference-target-x-arm-gains", type=float, nargs=2, default=(-5.3, 2.4))
    parser.add_argument("--reference-target-y-arm-gains", type=float, nargs=2, default=(12.0, 0.0))
    parser.add_argument("--reference-target-positive-y-arm-gains", type=float, nargs=2, default=(22.0, 0.0))
    parser.add_argument("--base-arm", type=float, nargs=7, required=True)
    parser.add_argument(
        "--arm-std",
        type=float,
        nargs=7,
        default=(0.06, 0.035, 0.06, 0.06, 0.10, 0.10, 0.12),
    )
    parser.add_argument(
        "--aperture-mean", type=float, nargs=4, default=(0.35, 0.25, 0.25, 0.25)
    )
    parser.add_argument(
        "--aperture-std", type=float, nargs=4, default=(0.15, 0.15, 0.16, 0.16)
    )
    args = parser.parse_args()
    result = search(args)
    print(json.dumps({key: value for key, value in result.items() if key != "rounds"}, indent=2))


if __name__ == "__main__":
    main()
