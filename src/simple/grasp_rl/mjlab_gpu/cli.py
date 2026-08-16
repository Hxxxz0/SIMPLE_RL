"""Minimal command line entrypoint for GPU-only mjlab PPO."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import torch

from simple.grasp_rl.mjlab_gpu.benchmark import compare_paired_results
from simple.grasp_rl.mjlab_gpu.collect import collect_successful_trajectories
from simple.grasp_rl.mjlab_gpu.config import MjlabPpoConfig
from simple.grasp_rl.mjlab_gpu.dataset_audit import audit_ppo_dataset
from simple.grasp_rl.mjlab_gpu.dataset_export import export_dual_dataset
from simple.grasp_rl.mjlab_gpu.recording import (
    MAX_PRECONTACT_TARGET_DISPLACEMENT_M,
    record_success_videos,
)
from simple.grasp_rl.mjlab_gpu.release import sha256_file, verify_release
from simple.grasp_rl.mjlab_gpu.robometer_reward import (
    ROBOMETER_REPLACE_TASKS,
    RobometerTaskRewardConfig,
)
from simple.grasp_rl.mjlab_gpu.runner import (
    GpuPpoRunner,
    SpatialCheckpointRouter,
    SpatialPolicyRouter,
    checkpoint_uses_plan_conditioned_actor,
    ppo_train_config,
)
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv
from simple.grasp_rl.schema import ACTION_DIM, ACTION_SLICES


def _common(parser: argparse.ArgumentParser, *, num_envs: int = 4096) -> None:
    parser.add_argument("--task", default="tabletop_grasp")
    parser.add_argument("--asset-bundle", type=Path, required=True)
    parser.add_argument("--reference-processed", type=Path, required=True)
    parser.add_argument("--reference-source", default="bc")
    parser.add_argument(
        "--strict-reference-episode",
        type=int,
        help=(
            "Fail closed unless processed data, frozen assets and every runtime "
            "reference use exactly this episode"
        ),
    )
    parser.add_argument(
        "--reference-selection",
        choices=("asset", "nearest", "balanced"),
        default="asset",
        help=(
            "Reference reset policy: asset preserves legacy pin/nearest behavior; "
            "balanced evenly covers the full processed library"
        ),
    )
    parser.add_argument(
        "--max-reference-initial-position-offset",
        type=float,
        help=(
            "Fail before simulation when any reference initial primary position is "
            "farther than this many metres from the frozen asset reset"
        ),
    )
    parser.add_argument(
        "--reference-reward-weight",
        type=float,
        default=0.05,
        help=(
            "Weight of the clean-reference action regularizer; the legacy "
            "default is 0.05"
        ),
    )
    parser.add_argument(
        "--disable-reference-contact-reward-gate",
        dest="reference_contact_reward_gate",
        action="store_false",
        default=True,
        help=(
            "Use physical grasp contact instead of reference contact timing for "
            "dense grasp rewards; omitted preserves legacy reward behavior"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=num_envs)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--max-reference-action-deviation",
        type=float,
        default=0.35,
        help=(
            "Maximum normalized residual action around the replay reference; "
            "the legacy default is 0.35"
        ),
    )
    parser.add_argument(
        "--policy-action-filter-alpha",
        type=float,
        help=(
            "Opt-in low-pass coefficient for complete policy commands; one "
            "preserves direct execution and values below one smooth exploration"
        ),
    )
    parser.add_argument(
        "--residual-action-groups",
        nargs="+",
        choices=tuple(ACTION_SLICES),
        help=(
            "Action groups PPO may correct around the reference; omitted uses "
            "checkpoint metadata or the legacy right-hand/right-arm mask"
        ),
    )
    parser.add_argument(
        "--grasp-anything-lift-arm-residual-min-scale",
        type=float,
        default=1.0,
        help=(
            "Opt-in minimum right-arm residual scale after a stable "
            "grasp_anything grasp; one preserves existing behavior"
        ),
    )
    parser.add_argument(
        "--grasp-anything-lift-arm-residual-decay-steps",
        type=int,
        default=0,
        help=(
            "Steps used to reach the opt-in lift-stage right-arm residual "
            "minimum; zero preserves existing behavior"
        ),
    )
    parser.add_argument(
        "--grasp-anything-lift-arm-residual-grasp-steps",
        type=int,
        default=3,
        help="Consecutive physical grasp steps required before arm decay starts",
    )
    parser.add_argument(
        "--grasp-anything-goal-potential-scale",
        type=float,
        default=5.0,
        help="Opt-in grasp_anything goal-progress scale; five preserves behavior",
    )
    parser.add_argument(
        "--grasp-anything-goal-potential-negative-clip",
        type=float,
        default=0.25,
        help=(
            "Opt-in negative potential-delta clip; 0.25 preserves behavior and "
            "one makes progress gains and losses symmetric"
        ),
    )
    parser.add_argument(
        "--grasp-anything-success-bonus",
        type=float,
        default=40.0,
        help="Opt-in terminal success bonus; forty preserves existing behavior",
    )
    parser.add_argument(
        "--grasp-anything-overhead-approach-clearance-m",
        type=float,
        default=0.0,
        help=(
            "Opt-in adaptive overhead approach waypoint clearance for wide "
            "workspaces; zero preserves existing reward behavior"
        ),
    )
    parser.add_argument(
        "--grasp-anything-overhead-final-descent-weight",
        type=float,
        default=0.25,
        help=(
            "Fraction of overhead approach potential reserved for descending to "
            "the object after reaching the waypoint; 0.25 preserves behavior"
        ),
    )
    parser.add_argument(
        "--reference-target-x-arm-gains",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("SHOULDER", "ELBOW"),
        help=(
            "Retarget normalized right shoulder-pitch/elbow reference actions "
            "per metre of observed target-X offset; default keeps legacy replay"
        ),
    )
    parser.add_argument(
        "--reference-target-positive-x-arm-gains",
        type=float,
        nargs=2,
        metavar=("SHOULDER", "ELBOW"),
        help=(
            "Opt-in gains used only for positive observed target-X offsets; "
            "omitted preserves symmetric X retargeting"
        ),
    )
    parser.add_argument(
        "--reference-target-x-arm-gain-y-bounds",
        type=float,
        nargs=2,
        metavar=("MIN_Y", "MAX_Y"),
        help=(
            "Optional observed target-Y offset interval where X arm retargeting "
            "is active; omitted preserves unbounded X retargeting"
        ),
    )
    parser.add_argument(
        "--reference-target-y-arm-gains",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("SHOULDER_YAW", "WRIST_YAW"),
        help=(
            "Retarget normalized right shoulder-yaw/wrist-yaw reference "
            "actions per metre of observed target-Y offset; default keeps "
            "legacy replay"
        ),
    )
    parser.add_argument(
        "--reference-target-positive-y-arm-gains",
        type=float,
        nargs=2,
        metavar=("SHOULDER_YAW", "WRIST_YAW"),
        help=(
            "Opt-in gains used only for positive observed target-Y offsets; "
            "omitted preserves the symmetric legacy Y retargeting path"
        ),
    )
    parser.add_argument(
        "--reference-target-yaw-arm-gains",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("SHOULDER_YAW", "WRIST_YAW"),
        help=(
            "Retarget normalized right shoulder-yaw/wrist-yaw reference "
            "actions per radian of sampled target yaw; default keeps legacy replay"
        ),
    )
    parser.add_argument("--dr-initial-strength", type=float)
    parser.add_argument("--dr-warmup-steps", type=int)
    parser.add_argument("--dr-ramp-steps", type=int)
    parser.add_argument(
        "--disable-domain-randomization",
        action="store_true",
        help="Disable DR explicitly for a bootstrap or diagnostic run",
    )
    parser.add_argument(
        "--target-position-jitter-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help=(
            "Override the task's symmetric target-position DR envelope in "
            "metres; omitted keeps the legacy 0.025,0.03 default"
        ),
    )
    parser.add_argument(
        "--target-position-offset-center-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Center of the staged target-position DR distribution in metres",
    )
    parser.add_argument(
        "--keep-target-position-center-during-curriculum",
        dest="target_position_scale_offset_center",
        action="store_false",
        default=True,
        help=(
            "Keep the target DR center fixed while curriculum strength expands "
            "only its jitter; omitted preserves legacy center scaling"
        ),
    )
    parser.add_argument(
        "--target-position-focus-probability",
        type=float,
        help=(
            "Opt-in probability of replacing a normal target-position sample "
            "with a focused sample"
        ),
    )
    parser.add_argument(
        "--target-position-focus-jitter-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Symmetric focused target-position jitter in metres",
    )
    parser.add_argument(
        "--target-position-focus-offset-center-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Center of the focused target-position distribution in metres",
    )
    parser.add_argument(
        "--target-position-focus-region",
        type=float,
        nargs=5,
        action="append",
        metavar=("CENTER_X", "CENTER_Y", "JITTER_X", "JITTER_Y", "PROBABILITY"),
        help=(
            "Repeatable target-position sampling region in metres; the five "
            "values are center X/Y, symmetric jitter X/Y and batch probability"
        ),
    )
    parser.add_argument(
        "--target-position-stratified-grid",
        type=int,
        nargs=2,
        metavar=("X_BINS", "Y_BINS"),
        help="Round-robin target-position sampling over a full workspace grid",
    )
    parser.add_argument(
        "--target-position-stratified-focus-cell",
        type=int,
        nargs=2,
        action="append",
        metavar=("X_INDEX", "Y_INDEX"),
        help=(
            "Repeatable cell to oversample after retaining one environment "
            "for every cell in the stratified grid"
        ),
    )
    parser.add_argument(
        "--target-yaw-jitter",
        type=float,
        help=(
            "Override the symmetric target-yaw DR envelope in radians; "
            "omitted keeps the legacy 0.15 default"
        ),
    )
    parser.add_argument(
        "--destination-position-jitter-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Symmetric destination-object position jitter in metres",
    )
    parser.add_argument(
        "--destination-yaw-jitter",
        type=float,
        help="Symmetric destination-object yaw jitter in radians",
    )
    parser.add_argument(
        "--distractor-position-jitter-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Symmetric position jitter for non-role free scene objects",
    )
    parser.add_argument(
        "--distractor-yaw-jitter",
        type=float,
        help="Symmetric yaw jitter for non-role free scene objects",
    )
    parser.add_argument(
        "--robot-base-position-jitter-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Symmetric floating-base reset-position jitter in metres",
    )
    parser.add_argument(
        "--robot-base-yaw-jitter",
        type=float,
        help="Symmetric floating-base reset-yaw jitter in radians",
    )
    parser.add_argument(
        "--target-mass-scale",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Override target mass randomization scale range",
    )
    parser.add_argument(
        "--friction-scale",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Override contact friction randomization scale range",
    )
    parser.add_argument(
        "--joint-damping-scale",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Override controlled-joint damping randomization scale range",
    )
    parser.add_argument(
        "--actuator-strength-scale",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Override actuator-strength randomization scale range",
    )
    parser.add_argument(
        "--action-delay-max-steps",
        type=int,
        choices=(0, 1),
        help="Override the maximum sampled action delay",
    )
    parser.add_argument("--reference-action-noise-std", type=float)
    parser.add_argument("--reference-position-noise-std", type=float)
    parser.add_argument("--reference-phase-noise-std", type=float)
    parser.add_argument(
        "--reference-future-dropout-probability",
        type=float,
    )
    parser.add_argument(
        "--dr-profile",
        choices=(
            "full",
            "pose_only",
            "target_x_only",
            "target_y_only",
            "target_yaw_only",
        ),
        default="full",
        help="Stage diagnosed target-pose adaptation before the full-DR profile",
    )


def _robometer_options(
    parser: argparse.ArgumentParser, *, include_source: bool
) -> None:
    if include_source:
        parser.add_argument(
            "--task-reward-source",
            choices=("physical", "robometer"),
            default="physical",
            help="Experimental task-reward source; physical preserves legacy behavior",
        )
    parser.add_argument("--robometer-server-url")
    parser.add_argument(
        "--robometer-instruction",
        help=(
            "Override the task instruction sent to Robometer; omitted uses the "
            "asset-specific validated instruction"
        ),
    )
    parser.add_argument("--robometer-inference-interval-steps", type=int, default=25)
    parser.add_argument("--robometer-progress-scale", type=float, default=1.0)


def _spatial_policy_router_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--spatial-router-teacher-checkpoint",
        type=Path,
        help="Frozen PPO teacher used outside the explicitly routed free cells",
    )
    parser.add_argument(
        "--spatial-router-free-cell",
        type=int,
        nargs=2,
        action="append",
        metavar=("X_INDEX", "Y_INDEX"),
        help="Repeatable grid cell routed to the learner checkpoint",
    )
    parser.add_argument(
        "--spatial-router-cell-checkpoint",
        nargs=3,
        action="append",
        metavar=("CHECKPOINT", "X_INDEX", "Y_INDEX"),
        help=(
            "Repeatable checkpoint expert and grid cell; the main checkpoint "
            "remains the default actor outside routed cells"
        ),
    )


def _robometer_config(
    args: argparse.Namespace, *, mode: str | None = None
) -> RobometerTaskRewardConfig | None:
    source = getattr(args, "task_reward_source", None)
    enabled = mode is not None or source == "robometer"
    if not enabled:
        return None
    if not args.robometer_server_url:
        raise ValueError("Robometer reward requires --robometer-server-url")
    values: dict[str, object] = {
        "server_url": args.robometer_server_url,
        "mode": mode or "replace",
        "task": args.task,
        "inference_interval_steps": args.robometer_inference_interval_steps,
        "progress_scale": args.robometer_progress_scale,
    }
    if args.robometer_instruction is not None:
        values["instruction"] = args.robometer_instruction
    return RobometerTaskRewardConfig(**values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simple-mjlab-ppo")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    _common(train)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--iterations", type=int, default=20_000)
    train.add_argument("--warm-start", type=Path)
    train.add_argument(
        "--plan-conditioned-actor",
        action="store_true",
        help=(
            "Use the proposal-residual actor architecture for a scratch run; "
            "checkpoint-based runs infer the architecture automatically"
        ),
    )
    train.add_argument(
        "--scratch-actor-output-scale",
        type=float,
        help=(
            "For a scratch plan-conditioned actor, multiply the randomly "
            "initialized correction head by this scale before training"
        ),
    )
    train.add_argument(
        "--scratch-right-hand-correction",
        type=float,
        nargs=7,
        help=(
            "Initialize a scratch grasp_anything policy with a reviewed "
            "right-hand correction around its reference"
        ),
    )
    train.add_argument(
        "--scratch-right-arm-correction",
        type=float,
        nargs=7,
        help=(
            "Initialize a scratch grasp_anything policy with a reviewed "
            "right-arm correction around its reference"
        ),
    )
    train.add_argument(
        "--warm-start-critic",
        action="store_true",
        help=(
            "With --warm-start, also restore the critic from an audited GPU "
            "checkpoint while starting a fresh optimizer"
        ),
    )
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--initial-vector-step",
        type=int,
        default=0,
        help="Start a warm-start run at this DR curriculum vector step",
    )
    train.add_argument(
        "--resume-vector-step",
        type=int,
        help=(
            "After an exact PPO resume, reset worlds at this DR curriculum "
            "vector step while preserving critic and Adam state"
        ),
    )
    train.add_argument(
        "--reset-resume-optimizer",
        action="store_true",
        help="Reset Adam moments after an exact resume while retaining actor/critic",
    )
    train.add_argument(
        "--learning-rate",
        type=float,
        help="Override the reviewed PPO learning rate for a continuation run",
    )
    train.add_argument(
        "--actor-learning-rate-scale",
        type=float,
        default=1.0,
        help="Scale the actor LR while retaining the critic base LR",
    )
    train.add_argument(
        "--actor-anchor-checkpoint",
        type=Path,
        help=(
            "Frozen teacher actor used to constrain a focused PPO run; "
            "disabled by default"
        ),
    )
    train.add_argument(
        "--actor-anchor-weight",
        type=float,
        default=0.0,
        help="Weight of the opt-in teacher action loss",
    )
    train.add_argument(
        "--actor-anchor-free-spatial-cell",
        type=int,
        nargs=2,
        action="append",
        metavar=("X_INDEX", "Y_INDEX"),
        help=(
            "Repeatable spatial cell where PPO is exempt from the actor anchor; "
            "requires --spatial-advantage-grid"
        ),
    )
    train.add_argument(
        "--schedule",
        choices=("adaptive", "fixed"),
        help="Override the PPO learning-rate schedule",
    )
    train.add_argument(
        "--exploration-std",
        type=float,
        help="Override the fixed Gaussian PPO action standard deviation",
    )
    train.add_argument(
        "--exploration-group-std",
        nargs=2,
        action="append",
        metavar=("ACTION_GROUP", "STD"),
        help=(
            "Repeatable per-action-group exploration std override; unspecified "
            "groups retain --exploration-std or the legacy default"
        ),
    )
    train.add_argument(
        "--exploration-hold-steps",
        type=int,
        default=1,
        help=(
            "PPO-safe exploration cadence; must be one because PPO uses "
            "independent per-step Gaussian likelihood ratios"
        ),
    )
    train.add_argument(
        "--learn-exploration-std",
        action="store_true",
        help="Allow PPO to learn its Gaussian exploration standard deviation",
    )
    train.add_argument(
        "--ppo-entropy-coef",
        type=float,
        default=0.0,
        help="PPO entropy coefficient; zero preserves legacy behavior",
    )
    train.add_argument(
        "--spatial-advantage-grid",
        type=int,
        nargs=2,
        metavar=("X_BINS", "Y_BINS"),
        help="Normalize PPO advantages independently per spatial cell",
    )
    train.add_argument(
        "--spatial-advantage-weighting",
        choices=("cell", "sample"),
        default="cell",
        help=(
            "Weight normalized spatial advantages equally per cell (legacy) or "
            "per sampled transition so focused-cell oversampling affects PPO"
        ),
    )
    train.add_argument(
        "--ppo-clip-param",
        type=float,
        help="Override the PPO probability-ratio clipping radius",
    )
    train.add_argument(
        "--ppo-learning-epochs",
        type=int,
        help="Override the number of fresh-rollout PPO optimization epochs",
    )
    train.add_argument(
        "--ppo-max-grad-norm",
        type=float,
        help="Override actor/critic gradient clipping norm",
    )
    train.add_argument(
        "--ppo-steps-per-env",
        type=int,
        help="Override fresh rollout length per environment and PPO update",
    )
    train.add_argument(
        "--save-interval",
        type=int,
        help="Override the checkpoint interval without changing legacy defaults",
    )
    train.add_argument(
        "--freeze-actor-normalizer",
        action="store_true",
        help="Keep warm-start actor observation statistics fixed during PPO",
    )
    train.add_argument(
        "--bootstrap-gate-episodes",
        type=int,
        default=0,
        help=(
            "Opt-in deterministic rollout count required before PPO updates; "
            "zero preserves legacy training behavior"
        ),
    )
    train.add_argument(
        "--bootstrap-gate-min-success-rate",
        type=float,
        default=0.0,
        help="Minimum success rate for the opt-in pre-PPO bootstrap gate",
    )
    train.add_argument(
        "--bootstrap-gate-mode",
        choices=("reference", "policy", "either"),
        default="either",
        help=(
            "Evaluate the replay reference, deterministic initialized policy, "
            "or accept either one for the opt-in bootstrap gate"
        ),
    )
    train.add_argument(
        "--bootstrap-gate-spatial-scope",
        choices=("all", "focus"),
        default="all",
        help=(
            "Gate on all bootstrap worlds (legacy) or on the minimum success "
            "rate across explicitly focused stratified target-position cells"
        ),
    )
    _robometer_options(train, include_source=True)

    shadow = commands.add_parser("shadow-reward")
    _common(shadow, num_envs=8)
    shadow.set_defaults(smoke=True)
    policy = shadow.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=Path)
    policy.add_argument(
        "--proposal-only",
        action="store_true",
        help="Execute the noisy reference proposal instead of a PPO checkpoint",
    )
    shadow.add_argument("--episodes", type=int, default=8)
    shadow.add_argument("--output", type=Path, required=True)
    _robometer_options(shadow, include_source=False)

    evaluate = commands.add_parser("evaluate")
    _common(evaluate)
    # Evaluation is already bounded by --episodes.  Mark it as a validation
    # run so deterministic checkpoint screening can use fewer than 1024 worlds
    # without weakening the minimum-environment guard on actual training.
    evaluate.set_defaults(smoke=True)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument(
        "--reference-only",
        action="store_true",
        help="Execute the clean replay reference as an expert upper bound",
    )
    evaluate.add_argument(
        "--proposal-only",
        action="store_true",
        help=(
            "Execute the noisy current reference proposal without a PPO "
            "correction; this is the noise-matched no-PPO baseline"
        ),
    )
    evaluate.add_argument("--episodes", type=int, default=200)
    evaluate.add_argument(
        "--minimum-success-rate",
        type=float,
        help="Fail the evaluation when success rate is below this threshold",
    )
    evaluate.add_argument(
        "--minimum-spatial-cell-success-rate",
        type=float,
        help=(
            "Fail when any sampled cell in the 8x8 target-XY diagnostic grid "
            "has a lower success rate"
        ),
    )
    evaluate.add_argument(
        "--minimum-focus-cell-success-rate",
        type=float,
        help=(
            "Fail when any explicitly focused stratified target-position cell "
            "has a lower exact-cell success rate"
        ),
    )
    evaluate.add_argument(
        "--stress-domain-randomization",
        action="store_true",
        help="Evaluate deterministic policy under full physics/reference DR",
    )
    evaluate.add_argument(
        "--evaluation-dr-strength",
        type=float,
        help="Evaluate at an exact staged DR strength in [0, 1]",
    )
    evaluate.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample the checkpoint's learned PPO Gaussian policy",
    )
    _spatial_policy_router_options(evaluate)

    benchmark = commands.add_parser("benchmark")
    _common(benchmark, num_envs=128)
    benchmark.set_defaults(smoke=True)
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--episodes", type=int, default=128)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument(
        "--stress-domain-randomization",
        action="store_true",
        help="Run all paired modes under full physics/reference DR",
    )
    benchmark.add_argument(
        "--evaluation-dr-strength",
        type=float,
        help="Run all paired modes at an exact staged DR strength in [0, 1]",
    )
    benchmark.set_defaults(
        stochastic_policy=False,
        minimum_success_rate=None,
        minimum_spatial_cell_success_rate=None,
        minimum_focus_cell_success_rate=None,
    )
    _spatial_policy_router_options(benchmark)

    record = commands.add_parser("record")
    _common(record, num_envs=32)
    record.set_defaults(smoke=True)
    record_policy = record.add_mutually_exclusive_group(required=True)
    record_policy.add_argument("--checkpoint", type=Path)
    record_policy.add_argument(
        "--reference-only",
        action="store_true",
        help="Record the clean replay reference without loading a PPO checkpoint",
    )
    record.add_argument("--output-dir", type=Path, required=True)
    record.add_argument("--videos", type=int, default=3)
    record.add_argument("--max-attempts", type=int, default=100)
    record.add_argument("--width", type=int, default=640)
    record.add_argument("--height", type=int, default=360)
    record.add_argument("--fps", type=int, default=50)
    record.add_argument(
        "--camera-view",
        choices=("full_robot", "grasp_closeup"),
        default="full_robot",
        help="External release camera; closeup makes finger/object contact auditable",
    )
    record.add_argument(
        "--stress-domain-randomization",
        action="store_true",
        help="Record deterministic policy under full physics/reference DR",
    )
    record.add_argument(
        "--evaluation-dr-strength",
        type=float,
        help="Record at an exact staged DR strength in [0, 1]",
    )
    record.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample the checkpoint's learned PPO Gaussian policy",
    )
    record.add_argument(
        "--allow-diagnostic-fallback",
        action="store_true",
        help=(
            "When complete success is unavailable, record the best failed "
            "episodes with success=false and diagnostic provenance"
        ),
    )
    record.add_argument(
        "--maximum-precontact-target-motion-m",
        type=float,
        default=MAX_PRECONTACT_TARGET_DISPLACEMENT_M,
        help=(
            "Reject recorded successes whose target moves farther than this "
            "before first hand contact; the legacy default is 0.003 m"
        ),
    )
    _spatial_policy_router_options(record)

    collect = commands.add_parser("collect")
    _common(collect, num_envs=64)
    collect.set_defaults(smoke=True)
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--successes", type=int, required=True)
    collect.add_argument("--max-attempts", type=int)
    collect.add_argument(
        "--stress-domain-randomization",
        action="store_true",
        help="Collect successful trajectories under full physics/reference DR",
    )
    collect.add_argument(
        "--evaluation-dr-strength",
        type=float,
        help="Collect at an exact staged DR strength in [0, 1]",
    )
    collect.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample the checkpoint's learned PPO Gaussian policy",
    )
    _spatial_policy_router_options(collect)

    collect_dataset = commands.add_parser("collect-dataset")
    _common(collect_dataset, num_envs=64)
    collect_dataset.set_defaults(smoke=True)
    collect_dataset.add_argument("--checkpoint", type=Path, required=True)
    collect_dataset.add_argument("--output-root", type=Path, required=True)
    collect_dataset.add_argument("--source-dataset", type=Path, required=True)
    collect_dataset.add_argument("--psi0-template", type=Path, required=True)
    collect_dataset.add_argument("--successes", type=int, required=True)
    collect_dataset.add_argument("--max-attempts", type=int)
    collect_dataset.add_argument("--camera", default="head_stereo_left")
    collect_dataset.add_argument("--width", type=int, default=640)
    collect_dataset.add_argument("--height", type=int, default=360)
    collect_dataset.add_argument("--fps", type=int, default=50)
    collect_dataset.add_argument(
        "--stress-domain-randomization",
        action="store_true",
        help="Collect successful trajectories under full physics/reference DR",
    )
    collect_dataset.add_argument(
        "--evaluation-dr-strength",
        type=float,
        help="Collect at an exact staged DR strength in [0, 1]",
    )
    collect_dataset.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample the checkpoint policy instead of using its deterministic mean",
    )
    _spatial_policy_router_options(collect_dataset)

    verify = commands.add_parser("verify-release")
    verify.add_argument("--release-dir", type=Path, required=True)

    audit_dataset = commands.add_parser("audit-dataset")
    audit_dataset.add_argument("--dataset-root", type=Path, required=True)
    audit_dataset.add_argument("--expected-successes", type=int)
    audit_dataset.add_argument("--expected-task")
    audit_dataset.add_argument("--expected-dr-strength", type=float)
    audit_dataset.add_argument("--require-full-dr-coverage", action="store_true")
    return parser


def _checkpoint_residual_action_groups(args: argparse.Namespace) -> tuple[str, ...]:
    explicit = getattr(args, "residual_action_groups", None)
    if explicit is not None:
        return tuple(explicit)
    checkpoint = next(
        (
            value
            for value in (
                getattr(args, "resume", None),
                getattr(args, "warm_start", None),
                getattr(args, "checkpoint", None),
            )
            if value is not None
        ),
        None,
    )
    if checkpoint is not None and Path(checkpoint).is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        resolved = (
            payload.get("mjlab_gpu_metadata", {}).get("config", {}).get("resolved", {})
        )
        groups = resolved.get("residual_action_groups")
        if groups is not None:
            return tuple(groups)
    return ("right_hand", "right_arm")


def _checkpoint_policy_action_filter_alpha(args: argparse.Namespace) -> float:
    explicit = getattr(args, "policy_action_filter_alpha", None)
    if explicit is not None:
        return float(explicit)
    checkpoint = next(
        (
            value
            for value in (
                getattr(args, "resume", None),
                getattr(args, "warm_start", None),
                getattr(args, "checkpoint", None),
            )
            if value is not None
        ),
        None,
    )
    if checkpoint is not None and Path(checkpoint).is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        resolved = (
            payload.get("mjlab_gpu_metadata", {}).get("config", {}).get("resolved", {})
        )
        value = resolved.get("policy_action_filter_alpha")
        if value is not None:
            return float(value)
    return 1.0


def _config(args: argparse.Namespace) -> MjlabPpoConfig:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES=<physical GPU>; the process uses logical cuda:0"
        )
    config = MjlabPpoConfig(
        task=args.task,
        asset_bundle=str(args.asset_bundle.resolve()),
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        smoke_mode=args.smoke,
        reference_processed=str(args.reference_processed.resolve()),
        reference_source=args.reference_source,
        strict_reference_episode=args.strict_reference_episode,
        reference_selection=args.reference_selection,
        max_reference_initial_position_offset=(
            args.max_reference_initial_position_offset
        ),
        reference_reward_weight=args.reference_reward_weight,
        reference_contact_reward_gate=args.reference_contact_reward_gate,
        max_reference_action_deviation=args.max_reference_action_deviation,
        policy_action_filter_alpha=_checkpoint_policy_action_filter_alpha(args),
        residual_action_groups=_checkpoint_residual_action_groups(args),
        grasp_anything_lift_arm_residual_min_scale=(
            args.grasp_anything_lift_arm_residual_min_scale
        ),
        grasp_anything_lift_arm_residual_decay_steps=(
            args.grasp_anything_lift_arm_residual_decay_steps
        ),
        grasp_anything_lift_arm_residual_grasp_steps=(
            args.grasp_anything_lift_arm_residual_grasp_steps
        ),
        grasp_anything_goal_potential_scale=(args.grasp_anything_goal_potential_scale),
        grasp_anything_goal_potential_negative_clip=(
            args.grasp_anything_goal_potential_negative_clip
        ),
        grasp_anything_success_bonus=args.grasp_anything_success_bonus,
        grasp_anything_overhead_approach_clearance_m=(
            args.grasp_anything_overhead_approach_clearance_m
        ),
        grasp_anything_overhead_final_descent_weight=(
            args.grasp_anything_overhead_final_descent_weight
        ),
        reference_target_x_arm_gains=tuple(args.reference_target_x_arm_gains),
        reference_target_positive_x_arm_gains=(
            None
            if args.reference_target_positive_x_arm_gains is None
            else tuple(args.reference_target_positive_x_arm_gains)
        ),
        reference_target_x_arm_gain_y_bounds=(
            None
            if args.reference_target_x_arm_gain_y_bounds is None
            else tuple(args.reference_target_x_arm_gain_y_bounds)
        ),
        reference_target_y_arm_gains=tuple(args.reference_target_y_arm_gains),
        reference_target_positive_y_arm_gains=(
            None
            if args.reference_target_positive_y_arm_gains is None
            else tuple(args.reference_target_positive_y_arm_gains)
        ),
        reference_target_yaw_arm_gains=tuple(args.reference_target_yaw_arm_gains),
    )
    stratified_focus_cell_ids: list[int] = []
    if args.target_position_stratified_focus_cell:
        if args.target_position_stratified_grid is None:
            raise ValueError("stratified focus cells require a stratified grid")
        x_bins, y_bins = args.target_position_stratified_grid
        for x_index, y_index in args.target_position_stratified_focus_cell:
            if not (0 <= x_index < x_bins and 0 <= y_index < y_bins):
                raise ValueError("stratified focus cell index is outside the grid")
            cell_id = x_index * y_bins + y_index
            if cell_id in stratified_focus_cell_ids:
                raise ValueError("duplicate stratified focus cell")
            stratified_focus_cell_ids.append(cell_id)
        if args.num_envs <= x_bins * y_bins:
            raise ValueError(
                "stratified focus cells require num-envs greater than the "
                "stratified grid cell count"
            )
    dr_overrides = {
        name: value
        for name, value in (
            ("curriculum_initial_strength", args.dr_initial_strength),
            ("curriculum_warmup_steps", args.dr_warmup_steps),
            ("curriculum_ramp_steps", args.dr_ramp_steps),
            (
                "target_position_jitter_xy",
                None
                if args.target_position_jitter_xy is None
                else tuple(args.target_position_jitter_xy),
            ),
            (
                "target_position_offset_center_xy",
                None
                if args.target_position_offset_center_xy is None
                else tuple(args.target_position_offset_center_xy),
            ),
            (
                "target_position_scale_offset_center",
                args.target_position_scale_offset_center,
            ),
            (
                "target_position_focus_probability",
                args.target_position_focus_probability,
            ),
            (
                "target_position_focus_jitter_xy",
                None
                if args.target_position_focus_jitter_xy is None
                else tuple(args.target_position_focus_jitter_xy),
            ),
            (
                "target_position_focus_offset_center_xy",
                None
                if args.target_position_focus_offset_center_xy is None
                else tuple(args.target_position_focus_offset_center_xy),
            ),
            (
                "target_position_focus_regions",
                None
                if args.target_position_focus_region is None
                else tuple(
                    tuple(region) for region in args.target_position_focus_region
                ),
            ),
            (
                "target_position_stratified_grid",
                None
                if args.target_position_stratified_grid is None
                else tuple(args.target_position_stratified_grid),
            ),
            (
                "target_position_stratified_focus_cell_ids",
                tuple(stratified_focus_cell_ids),
            ),
            ("target_yaw_jitter", args.target_yaw_jitter),
            (
                "destination_position_jitter_xy",
                None
                if args.destination_position_jitter_xy is None
                else tuple(args.destination_position_jitter_xy),
            ),
            ("destination_yaw_jitter", args.destination_yaw_jitter),
            (
                "distractor_position_jitter_xy",
                None
                if args.distractor_position_jitter_xy is None
                else tuple(args.distractor_position_jitter_xy),
            ),
            ("distractor_yaw_jitter", args.distractor_yaw_jitter),
            (
                "robot_base_position_jitter_xy",
                None
                if args.robot_base_position_jitter_xy is None
                else tuple(args.robot_base_position_jitter_xy),
            ),
            ("robot_base_yaw_jitter", args.robot_base_yaw_jitter),
            (
                "target_mass_scale",
                None
                if args.target_mass_scale is None
                else tuple(args.target_mass_scale),
            ),
            (
                "friction_scale",
                None if args.friction_scale is None else tuple(args.friction_scale),
            ),
            (
                "joint_damping_scale",
                None
                if args.joint_damping_scale is None
                else tuple(args.joint_damping_scale),
            ),
            (
                "actuator_strength_scale",
                None
                if args.actuator_strength_scale is None
                else tuple(args.actuator_strength_scale),
            ),
            ("action_delay_max_steps", args.action_delay_max_steps),
        )
        if value is not None
    }
    noise_overrides = {
        name: value
        for name, value in (
            ("action_std", args.reference_action_noise_std),
            ("position_std", args.reference_position_noise_std),
            ("phase_std", args.reference_phase_noise_std),
            (
                "future_dropout_probability",
                args.reference_future_dropout_probability,
            ),
        )
        if value is not None
    }
    if dr_overrides:
        config = replace(
            config,
            domain_randomization=replace(config.domain_randomization, **dr_overrides),
        )
    if noise_overrides:
        config = replace(
            config,
            domain_randomization=replace(
                config.domain_randomization,
                reference_noise=replace(
                    config.domain_randomization.reference_noise,
                    **noise_overrides,
                ),
            ),
        )
    if args.disable_domain_randomization:
        config = replace(
            config,
            domain_randomization=replace(config.domain_randomization, enabled=False),
        )
    if args.dr_profile == "pose_only":
        config = replace(
            config,
            domain_randomization=config.domain_randomization.pose_only(),
        )
    elif args.dr_profile == "target_x_only":
        config = replace(
            config,
            domain_randomization=config.domain_randomization.target_x_only(),
        )
    elif args.dr_profile == "target_y_only":
        config = replace(
            config,
            domain_randomization=config.domain_randomization.target_y_only(),
        )
    elif args.dr_profile == "target_yaw_only":
        config = replace(
            config,
            domain_randomization=config.domain_randomization.target_yaw_only(),
        )
    return config


def _seed_torch(seed: int) -> None:
    """Make runner construction reproducible on the selected CUDA device."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _initialize_scratch_correction_head(
    policy: torch.nn.Module,
    *,
    output_scale: float | None,
    correction_bias: torch.Tensor | None,
) -> None:
    """Initialize only the final residual head of a scratch policy."""

    linear_layers = [
        module for module in policy.mlp.modules() if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        raise RuntimeError("scratch actor has no linear correction head")
    head = linear_layers[-1]
    if head.out_features != ACTION_DIM:
        raise ValueError("scratch correction head has the wrong action dimension")
    with torch.no_grad():
        if output_scale is not None:
            head.weight.mul_(output_scale)
            if head.bias is not None:
                head.bias.mul_(output_scale)
        if correction_bias is not None:
            if correction_bias.shape != (ACTION_DIM,):
                raise ValueError("scratch correction bias must contain 36 values")
            if head.bias is None:
                raise ValueError("scratch correction head has no bias")
            head.bias.copy_(correction_bias.to(head.bias))


def _validate_robometer_run(
    config: MjlabPpoConfig,
    reward_config: RobometerTaskRewardConfig | None,
) -> None:
    if reward_config is None:
        return
    if config.task != reward_config.task:
        raise ValueError(f"Robometer reward is restricted to {reward_config.task}")
    if (
        reward_config.mode == "replace"
        and reward_config.task not in ROBOMETER_REPLACE_TASKS
    ):
        validated = ", ".join(sorted(ROBOMETER_REPLACE_TASKS))
        raise ValueError(
            "Robometer reward replacement is validation-gated to: " + validated
        )
    if not config.smoke_mode:
        raise ValueError("Robometer reward requires --smoke")
    if config.num_envs > 8:
        raise ValueError("Robometer reward supports at most eight environments")


@torch.no_grad()
def _bootstrap_gate_rollout(
    env: GpuGraspVecEnv,
    *,
    actor: torch.nn.Module | None,
    episodes: int,
    spatial_cell_ids: tuple[int, ...] = (),
) -> dict[str, object]:
    """Measure a deterministic reference or policy before allowing PPO updates."""

    if episodes < 1:
        raise ValueError("bootstrap rollout episodes must be positive")
    if episodes > env.num_envs:
        raise ValueError(
            "bootstrap rollout episodes must not exceed num_envs so every "
            "outcome comes from one unique initial world"
        )
    if len(set(spatial_cell_ids)) != len(spatial_cell_ids) or any(
        cell_id < 0 for cell_id in spatial_cell_ids
    ):
        raise ValueError("bootstrap spatial cell IDs must be unique and non-negative")
    observations = env.get_observations()
    evaluation_world = (
        torch.arange(env.num_envs, device=getattr(env, "device", None)) < episodes
    )
    initial_spatial_cells: torch.Tensor | None = None
    spatial_outcomes = {
        cell_id: {"successes": 0, "failures": 0, "timeouts": 0}
        for cell_id in spatial_cell_ids
    }
    if spatial_cell_ids:
        observed_cells = observations.get("target_position_cell_id")
        if observed_cells is None:
            raise RuntimeError("spatial bootstrap gate requires target-position cell IDs")
        initial_spatial_cells = observed_cells.flatten().to(torch.long).clone()
        requested = torch.tensor(
            spatial_cell_ids, dtype=torch.long, device=initial_spatial_cells.device
        )
        matching = torch.isin(initial_spatial_cells, requested)
        candidates_by_cell = {}
        for cell_id in spatial_cell_ids:
            candidates = (initial_spatial_cells == cell_id).nonzero(
                as_tuple=False
            ).flatten()
            if not len(candidates):
                raise ValueError(
                    f"bootstrap spatial cell {cell_id} has no assigned worlds"
                )
            candidates_by_cell[cell_id] = candidates
        available = int(matching.sum().item())
        if episodes > available:
            raise ValueError(
                "bootstrap rollout episodes exceed the number of worlds in "
                f"the requested spatial cells: {episodes} > {available}"
            )
        if episodes < len(spatial_cell_ids):
            raise ValueError(
                "bootstrap rollout episodes must include every requested spatial cell"
            )
        selected_indices: list[torch.Tensor] = []
        selected_mask = torch.zeros_like(matching)
        for cell_id in spatial_cell_ids:
            candidates = candidates_by_cell[cell_id]
            selected_indices.append(candidates[:1])
            selected_mask[candidates[0]] = True
        remaining = matching & ~selected_mask
        remaining_indices = remaining.nonzero(as_tuple=False).flatten()
        selected_indices.append(remaining_indices[: episodes - len(spatial_cell_ids)])
        evaluation_world = torch.zeros_like(matching)
        evaluation_world[torch.cat(selected_indices)] = True
    completed = 0
    successes = 0
    failures = 0
    timeouts = 0
    vector_steps = 0
    maximum_vector_steps = (env.max_episode_length + 1) * 2
    completed_world: torch.Tensor | None = None
    was_training = actor.training if actor is not None else None
    if actor is not None:
        actor.eval()
    try:
        while completed < episodes:
            if vector_steps >= maximum_vector_steps:
                raise RuntimeError("bootstrap rollout did not complete enough episodes")
            actions = (
                env.reference.current_action().clone()
                if actor is None
                else actor(observations, stochastic_output=False)
            )
            observations, _, dones, _ = env.step(actions)
            vector_steps += 1
            if completed_world is None:
                completed_world = torch.zeros_like(dones)
            finished = (
                dones & evaluation_world & ~completed_world
            ).nonzero(as_tuple=False).flatten()
            if not len(finished):
                continue
            assert env.last_terms is not None
            completed_world[finished] = True
            successes += int(env.last_terms.success[finished].sum().item())
            failures += int(env.last_terms.failure[finished].sum().item())
            timeouts += int(env.last_terms.timeout[finished].sum().item())
            if initial_spatial_cells is not None:
                for cell_id in spatial_cell_ids:
                    cell_finished = finished[
                        initial_spatial_cells[finished] == cell_id
                    ]
                    if not len(cell_finished):
                        continue
                    spatial_outcomes[cell_id]["successes"] += int(
                        env.last_terms.success[cell_finished].sum().item()
                    )
                    spatial_outcomes[cell_id]["failures"] += int(
                        env.last_terms.failure[cell_finished].sum().item()
                    )
                    spatial_outcomes[cell_id]["timeouts"] += int(
                        env.last_terms.timeout[cell_finished].sum().item()
                    )
            completed += len(finished)
    finally:
        if actor is not None:
            actor.train(bool(was_training))
    result: dict[str, object] = {
        "episodes": completed,
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "success_rate": successes / completed,
        "vector_steps": vector_steps,
    }
    if initial_spatial_cells is not None:
        spatial_cells = []
        for cell_id in spatial_cell_ids:
            cell_world = evaluation_world & (initial_spatial_cells == cell_id)
            cell_episodes = int(cell_world.sum().item())
            outcome = spatial_outcomes[cell_id]
            spatial_cells.append(
                {
                    "cell_id": cell_id,
                    "episodes": cell_episodes,
                    **outcome,
                    "success_rate": outcome["successes"] / cell_episodes,
                }
            )
        result["spatial_cell_ids"] = list(spatial_cell_ids)
        result["spatial_cells"] = spatial_cells
        result["minimum_spatial_cell_success_rate"] = min(
            cell["success_rate"] for cell in spatial_cells
        )
    return result


def _train(args: argparse.Namespace) -> dict[str, object]:
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.initial_vector_step < 0:
        raise ValueError("initial-vector-step must be non-negative")
    if args.resume_vector_step is not None and args.resume_vector_step < 0:
        raise ValueError("resume-vector-step must be non-negative")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.actor_learning_rate_scale <= 0.0:
        raise ValueError("actor-learning-rate-scale must be positive")
    if args.actor_anchor_weight < 0.0:
        raise ValueError("actor-anchor-weight must be non-negative")
    if (
        args.spatial_advantage_weighting != "cell"
        and args.spatial_advantage_grid is None
    ):
        raise ValueError(
            "non-default spatial-advantage-weighting requires "
            "--spatial-advantage-grid"
        )
    if (args.actor_anchor_checkpoint is None) != (args.actor_anchor_weight == 0.0):
        raise ValueError(
            "actor-anchor-checkpoint and positive actor-anchor-weight "
            "must be set together"
        )
    free_anchor_cell_ids: list[int] = []
    if args.actor_anchor_free_spatial_cell:
        if args.actor_anchor_checkpoint is None:
            raise ValueError("free actor-anchor cells require an actor anchor")
        if args.spatial_advantage_grid is None:
            raise ValueError(
                "free actor-anchor cells require --spatial-advantage-grid"
            )
        x_bins, y_bins = args.spatial_advantage_grid
        for x_index, y_index in args.actor_anchor_free_spatial_cell:
            if not (0 <= x_index < x_bins and 0 <= y_index < y_bins):
                raise ValueError("free actor-anchor cell index is outside the grid")
            cell_id = x_index * y_bins + y_index
            if cell_id in free_anchor_cell_ids:
                raise ValueError("duplicate free actor-anchor spatial cell")
            free_anchor_cell_ids.append(cell_id)
    if args.exploration_std is not None and args.exploration_std <= 0.0:
        raise ValueError("exploration-std must be positive")
    exploration_group_stds: list[tuple[str, float]] = []
    seen_exploration_groups: set[str] = set()
    for group, raw_std in args.exploration_group_std or ():
        if group not in ACTION_SLICES:
            raise ValueError(f"Unknown exploration action group: {group}")
        if group in seen_exploration_groups:
            raise ValueError(f"Duplicate exploration action group: {group}")
        try:
            std = float(raw_std)
        except ValueError as error:
            raise ValueError("Action-group exploration std must be numeric") from error
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("Action-group exploration std must be positive")
        exploration_group_stds.append((group, std))
        seen_exploration_groups.add(group)
    if args.exploration_hold_steps != 1:
        raise ValueError(
            "exploration-hold-steps must be 1 for valid PPO likelihood ratios"
        )
    if args.ppo_clip_param is not None and not 0.0 < args.ppo_clip_param <= 1.0:
        raise ValueError("ppo-clip-param must be in (0, 1]")
    if args.ppo_learning_epochs is not None and args.ppo_learning_epochs < 1:
        raise ValueError("ppo-learning-epochs must be positive")
    if args.ppo_max_grad_norm is not None and args.ppo_max_grad_norm <= 0.0:
        raise ValueError("ppo-max-grad-norm must be positive")
    if args.ppo_steps_per_env is not None and args.ppo_steps_per_env < 1:
        raise ValueError("ppo-steps-per-env must be positive")
    if args.save_interval is not None and args.save_interval < 1:
        raise ValueError("save-interval must be positive")
    if args.bootstrap_gate_episodes < 0:
        raise ValueError("bootstrap-gate-episodes must be non-negative")
    if not 0.0 <= args.bootstrap_gate_min_success_rate <= 1.0:
        raise ValueError("bootstrap-gate-min-success-rate must be in [0, 1]")
    if args.bootstrap_gate_episodes == 0 and args.bootstrap_gate_min_success_rate:
        raise ValueError(
            "positive bootstrap-gate-min-success-rate requires "
            "--bootstrap-gate-episodes"
        )
    if (
        args.scratch_actor_output_scale is not None
        and args.scratch_actor_output_scale <= 0.0
    ):
        raise ValueError("scratch-actor-output-scale must be positive")
    if args.resume is not None and args.warm_start is not None:
        raise ValueError("resume and warm-start are mutually exclusive")
    if args.resume is not None and args.initial_vector_step:
        raise ValueError("resume restores vector step; do not override it")
    if args.resume_vector_step is not None and args.resume is None:
        raise ValueError("resume-vector-step requires --resume")
    if args.reset_resume_optimizer and args.resume is None:
        raise ValueError("reset-resume-optimizer requires --resume")
    if args.warm_start_critic and args.warm_start is None:
        raise ValueError("warm-start-critic requires --warm-start")
    config = _config(args)
    bootstrap_spatial_cell_ids = (
        config.domain_randomization.target_position_stratified_focus_cell_ids
        if args.bootstrap_gate_spatial_scope == "focus"
        else ()
    )
    if (
        args.bootstrap_gate_episodes
        and args.bootstrap_gate_spatial_scope == "focus"
        and not bootstrap_spatial_cell_ids
    ):
        raise ValueError(
            "focus bootstrap gate requires explicitly focused stratified "
            "target-position cells"
        )
    hand_correction = args.scratch_right_hand_correction
    arm_correction = args.scratch_right_arm_correction
    if (hand_correction is None) != (arm_correction is None):
        raise ValueError(
            "scratch right-hand and right-arm corrections must be provided together"
        )
    if hand_correction is not None and (
        config.task != "grasp_anything"
        or args.resume is not None
        or args.warm_start is not None
        or not args.plan_conditioned_actor
    ):
        raise ValueError(
            "scratch grasp corrections require a scratch "
            "grasp_anything --plan-conditioned-actor run"
        )
    reward_config = _robometer_config(args)
    _validate_robometer_run(config, reward_config)
    if reward_config is not None:
        if args.resume is not None:
            raise ValueError(
                "Robometer reward does not support exact resume; use --warm-start"
            )
        if args.warm_start_critic:
            raise ValueError(
                "Robometer reward training starts a fresh critic and optimizer"
            )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed_torch(config.seed)
    env = GpuGraspVecEnv(
        config,
        training=True,
        robometer_reward_config=reward_config,
    )
    if args.initial_vector_step:
        env.common_step_counter = args.initial_vector_step
        env._reset(torch.arange(env.num_envs, device=env.device))
    architecture_checkpoint = args.resume or args.warm_start
    checkpoint_plan_conditioned_actor = bool(
        architecture_checkpoint
        and checkpoint_uses_plan_conditioned_actor(architecture_checkpoint)
    )
    if (
        args.plan_conditioned_actor
        and architecture_checkpoint is not None
        and not checkpoint_plan_conditioned_actor
    ):
        raise ValueError(
            "--plan-conditioned-actor conflicts with the checkpoint architecture"
        )
    plan_conditioned_actor = (
        args.plan_conditioned_actor or checkpoint_plan_conditioned_actor
    )
    if args.scratch_actor_output_scale is not None and (
        architecture_checkpoint is not None or not plan_conditioned_actor
    ):
        raise ValueError(
            "--scratch-actor-output-scale requires a scratch "
            "--plan-conditioned-actor run"
        )
    train_config = ppo_train_config(
        smoke=config.smoke_mode,
        plan_conditioned_actor=plan_conditioned_actor,
        exploration_std=args.exploration_std,
        exploration_hold_steps=args.exploration_hold_steps,
        learn_exploration_std=args.learn_exploration_std,
        entropy_coef=args.ppo_entropy_coef,
        residual_action_groups=config.residual_action_groups,
        exploration_group_stds=tuple(exploration_group_stds),
    )
    train_config["seed"] = config.seed
    if args.learning_rate is not None:
        train_config["algorithm"]["learning_rate"] = args.learning_rate
    if args.schedule is not None:
        train_config["algorithm"]["schedule"] = args.schedule
    if args.ppo_clip_param is not None:
        train_config["algorithm"]["clip_param"] = args.ppo_clip_param
    if args.ppo_learning_epochs is not None:
        train_config["algorithm"]["num_learning_epochs"] = args.ppo_learning_epochs
    if args.ppo_max_grad_norm is not None:
        train_config["algorithm"]["max_grad_norm"] = args.ppo_max_grad_norm
    if args.ppo_steps_per_env is not None:
        train_config["num_steps_per_env"] = args.ppo_steps_per_env
    if args.save_interval is not None:
        train_config["save_interval"] = args.save_interval
    correction_bias = None
    if hand_correction is not None:
        correction_bias = torch.zeros(ACTION_DIM, device=config.device)
        correction_bias[7:14] = torch.as_tensor(
            hand_correction, dtype=torch.float32, device=config.device
        )
        correction_bias[21:28] = torch.as_tensor(
            arm_correction, dtype=torch.float32, device=config.device
        )
    run_metadata: dict[str, object] = {
        "environment": config.resolved(),
        "ppo": train_config,
        "initial_vector_step": args.initial_vector_step,
        "resume_vector_step": args.resume_vector_step,
        "reset_resume_optimizer": args.reset_resume_optimizer,
        "warm_start_critic": args.warm_start_critic,
        "freeze_actor_normalizer": args.freeze_actor_normalizer,
        "actor_learning_rate_scale": args.actor_learning_rate_scale,
        "actor_anchor_checkpoint": (
            None
            if args.actor_anchor_checkpoint is None
            else str(args.actor_anchor_checkpoint.resolve())
        ),
        "actor_anchor_weight": args.actor_anchor_weight,
        "actor_anchor_free_spatial_cell_ids": free_anchor_cell_ids,
        "spatial_advantage_weighting": args.spatial_advantage_weighting,
        "plan_conditioned_actor": plan_conditioned_actor,
        "scratch_actor_output_scale": args.scratch_actor_output_scale,
        "scratch_right_hand_correction": hand_correction,
        "scratch_right_arm_correction": arm_correction,
        "reference_alignment": env.reference_alignment,
    }
    if reward_config is not None:
        run_metadata["task_reward_override"] = reward_config.metadata()
    (output / "config.json").write_text(
        json.dumps(run_metadata, indent=2, default=list)
    )
    # Environment construction may consume RNG differently when an external
    # reward renderer is enabled.  Re-seed immediately before runner creation
    # so scratch A/B runs receive identical actor and critic initialization.
    _seed_torch(config.seed)
    runner = GpuPpoRunner(
        env,
        train_config,
        log_dir=str(output),
        integrity_path=output / "ppo_integrity.jsonl",
        actor_learning_rate_scale=args.actor_learning_rate_scale,
        actor_anchor_checkpoint=args.actor_anchor_checkpoint,
        actor_anchor_weight=args.actor_anchor_weight,
        actor_anchor_free_spatial_cell_ids=tuple(free_anchor_cell_ids),
        spatial_advantage_grid=(
            None
            if args.spatial_advantage_grid is None
            else tuple(args.spatial_advantage_grid)
        ),
        spatial_advantage_weighting=args.spatial_advantage_weighting,
    )
    initial_checkpoint: Path | None = None
    if args.resume is None and args.warm_start is None:
        if args.scratch_actor_output_scale is not None or correction_bias is not None:
            _initialize_scratch_correction_head(
                runner.alg.get_policy(),
                output_scale=args.scratch_actor_output_scale,
                correction_bias=correction_bias,
            )
        run_metadata["random_initialization"] = {
            "seed": config.seed,
            "actor_sha256": _tensor_collection_sha256(
                dict(runner.alg.get_policy().state_dict())
            ),
            "critic_sha256": _tensor_collection_sha256(
                dict(runner.alg._raw_critic.state_dict())
            ),
        }
        (output / "config.json").write_text(
            json.dumps(run_metadata, indent=2, default=list)
        )
        initial_checkpoint = output / "model_initial.pt"
        runner.save(str(initial_checkpoint), infos={"stage": "random_initialization"})
    if args.resume is not None:
        runner.load(str(args.resume.resolve()))
        if args.reset_resume_optimizer:
            runner.alg.optimizer.state.clear()
        if args.learning_rate is not None:
            runner.set_learning_rate(args.learning_rate)
        if args.exploration_std is not None or exploration_group_stds:
            distribution = runner.alg.get_policy().distribution
            std_param = getattr(distribution, "std_param", None)
            if std_param is None:
                raise ValueError("exploration std overrides require scalar Gaussian std")
            with torch.no_grad():
                if args.exploration_std is not None:
                    std_param.fill_(args.exploration_std)
                for group, std in exploration_group_stds:
                    spec = ACTION_SLICES[group]
                    std_param[spec.start : spec.stop] = std
        if args.resume_vector_step is not None:
            env.common_step_counter = args.resume_vector_step
            env._reset(torch.arange(env.num_envs, device=env.device))
    elif args.warm_start is not None:
        runner.load_actor_warm_start(args.warm_start.resolve())
        if args.warm_start_critic:
            runner.load_critic_warm_start(args.warm_start.resolve())
    if args.freeze_actor_normalizer:
        runner.freeze_actor_normalizer()
    if args.bootstrap_gate_episodes:
        gate_modes = (
            ("reference", "policy")
            if args.bootstrap_gate_mode == "either"
            else (args.bootstrap_gate_mode,)
        )
        original_vector_step = env.common_step_counter
        gate_results = []
        for mode in gate_modes:
            result = _bootstrap_gate_rollout(
                env,
                actor=(None if mode == "reference" else runner.alg.get_policy()),
                episodes=args.bootstrap_gate_episodes,
                spatial_cell_ids=bootstrap_spatial_cell_ids,
            )
            gate_success_rate = (
                result["minimum_spatial_cell_success_rate"]
                if bootstrap_spatial_cell_ids
                else result["success_rate"]
            )
            gate_results.append(
                {"mode": mode, "gate_success_rate": gate_success_rate, **result}
            )
            env.common_step_counter = original_vector_step
            env._reset(torch.arange(env.num_envs, device=env.device))
            if gate_success_rate >= args.bootstrap_gate_min_success_rate:
                break
        run_metadata["bootstrap_gate"] = {
            "episodes": args.bootstrap_gate_episodes,
            "minimum_success_rate": args.bootstrap_gate_min_success_rate,
            "requested_mode": args.bootstrap_gate_mode,
            "spatial_scope": args.bootstrap_gate_spatial_scope,
            "spatial_cell_ids": list(bootstrap_spatial_cell_ids),
            "attempts": gate_results,
            "passed": bool(
                gate_results[-1]["gate_success_rate"]
                >= args.bootstrap_gate_min_success_rate
            ),
        }
        (output / "config.json").write_text(
            json.dumps(run_metadata, indent=2, default=list)
        )
        if not run_metadata["bootstrap_gate"]["passed"]:
            rates = ", ".join(
                f"{item['mode']}={item['successes']}/{item['episodes']}"
                for item in gate_results
            )
            raise RuntimeError(
                "PPO bootstrap gate rejected an infeasible initialization: "
                f"{rates}; required success rate "
                f">= {args.bootstrap_gate_min_success_rate:.4f}"
            )
    runner.assert_cuda_integrity(
        require_optimizer_state=bool(runner.alg.optimizer.state)
    )
    runner.learn(args.iterations)
    runner.assert_cuda_integrity(require_optimizer_state=True)
    checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
    runner.save(str(checkpoint), infos={"stage": "completed"})
    record = (
        runner.integrity_auditor.latest_record
        if runner.integrity_auditor is not None
        else None
    )
    return {
        "checkpoint": str(checkpoint),
        "initial_checkpoint": (
            None if initial_checkpoint is None else str(initial_checkpoint)
        ),
        "next_learning_iteration": runner.current_learning_iteration + 1,
        "num_envs": config.num_envs,
        "ppo_integrity": record,
        "optimizer_state_cuda": True,
    }


def _evaluation_dr_strength(args: argparse.Namespace) -> float:
    staged = args.evaluation_dr_strength
    if staged is not None and not 0.0 <= staged <= 1.0:
        raise ValueError("evaluation-dr-strength must be in [0, 1]")
    if args.stress_domain_randomization and staged is not None:
        raise ValueError(
            "stress-domain-randomization and evaluation-dr-strength are "
            "mutually exclusive"
        )
    return 1.0 if args.stress_domain_randomization else float(staged or 0.0)


def _set_evaluation_dr_strength(env: GpuGraspVecEnv, strength: float) -> None:
    curriculum = env.config.domain_randomization
    env.common_step_counter = round(
        curriculum.curriculum_warmup_steps + strength * curriculum.curriculum_ramp_steps
    )
    env._reset(torch.arange(env.num_envs, device=env.device))


def _checkpoint_training_dr_strength(checkpoint: Path) -> float | None:
    """Return the DR strength represented by a saved training checkpoint."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env_state = payload.get("env_state")
    metadata = payload.get("mjlab_gpu_metadata")
    if not isinstance(env_state, dict) or not isinstance(metadata, dict):
        return None
    resolved = metadata.get("config", {}).get("resolved", {})
    randomization = resolved.get("domain_randomization")
    if not isinstance(randomization, dict):
        return None
    if not randomization.get("enabled", True):
        return 0.0
    step = int(env_state.get("common_step_counter", 0))
    warmup = int(randomization.get("curriculum_warmup_steps", 0))
    ramp = int(randomization.get("curriculum_ramp_steps", 1))
    if ramp < 1:
        return None
    scheduled = min(max((step - warmup) / float(ramp), 0.0), 1.0)
    initial = float(randomization.get("curriculum_initial_strength", 0.0))
    return max(initial, scheduled)


def _spatial_router_cell_ids(args: argparse.Namespace) -> tuple[int, ...]:
    teacher = getattr(args, "spatial_router_teacher_checkpoint", None)
    cells = getattr(args, "spatial_router_free_cell", None)
    if (teacher is None) != (not cells):
        raise ValueError(
            "spatial router teacher checkpoint and free cells must be set together"
        )
    if teacher is None:
        return ()
    grid = args.target_position_stratified_grid
    if grid is None:
        raise ValueError("spatial policy router requires a stratified target grid")
    x_bins, y_bins = grid
    cell_ids: list[int] = []
    for x_index, y_index in cells:
        if not (0 <= x_index < x_bins and 0 <= y_index < y_bins):
            raise ValueError("spatial router cell index is outside the target grid")
        cell_id = x_index * y_bins + y_index
        if cell_id in cell_ids:
            raise ValueError("duplicate spatial router cell")
        cell_ids.append(cell_id)
    return tuple(cell_ids)


def _spatial_checkpoint_routes(
    args: argparse.Namespace,
) -> tuple[tuple[Path, tuple[int, ...]], ...]:
    raw_routes = getattr(args, "spatial_router_cell_checkpoint", None)
    if not raw_routes:
        return ()
    if getattr(args, "spatial_router_teacher_checkpoint", None) is not None or getattr(
        args, "spatial_router_free_cell", None
    ):
        raise ValueError(
            "cell-checkpoint routes and teacher/learner spatial routing are "
            "mutually exclusive"
        )
    grid = args.target_position_stratified_grid
    if grid is None:
        raise ValueError("spatial checkpoint routes require a stratified target grid")
    x_bins, y_bins = grid
    routes: dict[Path, list[int]] = {}
    routed_cells: set[int] = set()
    for raw_checkpoint, raw_x_index, raw_y_index in raw_routes:
        try:
            x_index = int(raw_x_index)
            y_index = int(raw_y_index)
        except ValueError as error:
            raise ValueError("spatial checkpoint cell indices must be integers") from error
        if not (0 <= x_index < x_bins and 0 <= y_index < y_bins):
            raise ValueError("spatial checkpoint cell index is outside the target grid")
        cell_id = x_index * y_bins + y_index
        if cell_id in routed_cells:
            raise ValueError("duplicate spatial checkpoint cell")
        routed_cells.add(cell_id)
        checkpoint = Path(raw_checkpoint).expanduser().resolve()
        routes.setdefault(checkpoint, []).append(cell_id)
    return tuple((path, tuple(cell_ids)) for path, cell_ids in routes.items())


def _load_evaluation_actor(
    env: GpuGraspVecEnv,
    config: MjlabPpoConfig,
    checkpoint: Path,
    args: argparse.Namespace,
) -> torch.nn.Module:
    train_config = ppo_train_config(
        smoke=True,
        plan_conditioned_actor=checkpoint_uses_plan_conditioned_actor(checkpoint),
        residual_action_groups=getattr(
            config, "residual_action_groups", ("right_hand", "right_arm")
        ),
    )
    runner = GpuPpoRunner(env, train_config, log_dir=None)
    runner.load_actor_warm_start(checkpoint)
    learner = runner.alg.get_policy().eval()
    checkpoint_routes = _spatial_checkpoint_routes(args)
    if checkpoint_routes:
        routes = tuple(
            (path, runner.frozen_actor_copy(path), cell_ids)
            for path, cell_ids in checkpoint_routes
        )
        return SpatialCheckpointRouter(learner, routes).to(config.device).eval()
    free_cell_ids = _spatial_router_cell_ids(args)
    if not free_cell_ids:
        return learner
    teacher_checkpoint = args.spatial_router_teacher_checkpoint.resolve()
    teacher = runner.frozen_actor_copy(teacher_checkpoint)
    return SpatialPolicyRouter(
        learner,
        teacher,
        free_spatial_cell_ids=free_cell_ids,
        teacher_checkpoint=teacher_checkpoint,
    ).to(config.device).eval()


def _tensor_collection_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _evaluation_world_sha256(
    env: GpuGraspVecEnv, observations: object, episodes: int
) -> tuple[str, str, str]:
    """Fingerprint physical worlds and noisy policy inputs independently."""

    policy = observations["policy"]
    model = env.gpu.sim.model
    randomizer = env.randomizer
    geom_ids = torch.cat(
        (randomizer.target_geom_ids, randomizer.contact_geom_ids)
    ).unique()
    arm_ids = randomizer.arm_actuator_ids
    physical_tensors = {
        "qpos": env.gpu.sim.data.qpos[:episodes],
        "qvel": env.gpu.sim.data.qvel[:episodes],
        "target_mass": model.body_mass[:episodes, randomizer.target_body_id],
        "target_inertia": model.body_inertia[:episodes, randomizer.target_body_id],
        "contact_friction": model.geom_friction[:episodes, geom_ids],
        "controlled_damping": model.dof_damping[:episodes, randomizer.dof_ids],
        "arm_gainprm": model.actuator_gainprm[:episodes, arm_ids],
        "arm_biasprm": model.actuator_biasprm[:episodes, arm_ids],
        "arm_forcerange": model.actuator_forcerange[:episodes, arm_ids],
        "actuator_strength": env.controller.actuator_strength_scale[:episodes],
        "action_delay": env.randomizer.action_delay_steps[:episodes],
        "target_translation_xy": env.randomizer.target_translation_xy[:episodes],
        "target_yaw": env.randomizer.target_yaw[:episodes],
        "destination_translation_xy": (
            env.randomizer.destination_translation_xy[:episodes]
        ),
        "destination_yaw": env.randomizer.destination_yaw[:episodes],
        "distractor_translation_xy": (
            env.randomizer.distractor_translation_xy[:episodes]
        ),
        "distractor_yaw": env.randomizer.distractor_yaw[:episodes],
        "robot_base_translation_xy": (
            env.randomizer.robot_base_translation_xy[:episodes]
        ),
        "robot_base_yaw": env.randomizer.robot_base_yaw[:episodes],
        "reference_rows": env.reference.episode_rows[:episodes],
        "reference_indices": env.reference.indices[:episodes],
        "reference_object_offset": env.reference.reference_object_offset[:episodes],
        "reference_object_yaw_offset": (
            env.reference.reference_object_yaw_offset[:episodes]
        ),
    }
    return (
        _tensor_collection_sha256(physical_tensors),
        _tensor_collection_sha256(
            {"policy_state": policy[:episodes, : env.reference.observation_dim]}
        ),
        _tensor_collection_sha256(
            {"proposal_context": policy[:episodes, env.reference.observation_dim :]}
        ),
    )


def _pose_distribution_summary(
    values: torch.Tensor, mask: torch.Tensor
) -> dict[str, int | float | None]:
    selected = values[mask].to(dtype=torch.float64)
    if not selected.numel():
        return {
            "samples": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
        }
    return {
        "samples": int(selected.numel()),
        "minimum": float(selected.min().item()),
        "maximum": float(selected.max().item()),
        "mean": float(selected.mean().item()),
        "standard_deviation": float(selected.std(unbiased=False).item()),
    }


def _pose_axis_diagnostics(
    values: torch.Tensor,
    outcome_codes: torch.Tensor,
    *,
    bin_count: int = 8,
) -> dict[str, object]:
    """Summarize one initial-pose axis against terminal episode outcomes."""

    values = values.detach().flatten().to(dtype=torch.float64, device="cpu")
    outcome_codes = outcome_codes.detach().flatten().to(device="cpu")
    if values.shape != outcome_codes.shape or not values.numel():
        raise ValueError("pose values and outcomes must be non-empty matching vectors")
    if bin_count < 1:
        raise ValueError("pose diagnostic bin count must be positive")
    if not torch.isfinite(values).all():
        raise ValueError("initial pose diagnostics require finite values")
    if not torch.isin(outcome_codes, torch.tensor([0, 1, 2])).all():
        raise ValueError("outcome codes must be 0=failure, 1=success or 2=timeout")

    masks = {
        "all": torch.ones_like(outcome_codes, dtype=torch.bool),
        "success": outcome_codes == 1,
        "physical_failure": outcome_codes == 0,
        "timeout": outcome_codes == 2,
    }
    summaries = {
        name: _pose_distribution_summary(values, mask) for name, mask in masks.items()
    }

    lower = float(values.min().item())
    upper = float(values.max().item())
    if lower == upper:
        edges = torch.tensor([lower, upper], dtype=torch.float64)
        bin_ids = torch.zeros_like(outcome_codes, dtype=torch.long)
    else:
        edges = torch.linspace(lower, upper, bin_count + 1, dtype=torch.float64)
        bin_ids = torch.bucketize(values, edges[1:-1])
    bins = []
    for index in range(len(edges) - 1):
        mask = bin_ids == index
        samples = int(mask.sum().item())
        successes = int(((outcome_codes == 1) & mask).sum().item())
        physical_failures = int(((outcome_codes == 0) & mask).sum().item())
        timeouts = int(((outcome_codes == 2) & mask).sum().item())
        bins.append(
            {
                "lower": float(edges[index].item()),
                "upper": float(edges[index + 1].item()),
                "samples": samples,
                "successes": successes,
                "physical_failures": physical_failures,
                "timeouts": timeouts,
                "success_rate": successes / samples if samples else None,
            }
        )
    return {"summaries": summaries, "bins": bins}


def _pose_xy_diagnostics(
    values: torch.Tensor,
    outcome_codes: torch.Tensor,
    *,
    bin_count: int | tuple[int, int] = 8,
) -> dict[str, object]:
    """Summarize spatial success coverage on an initial target XY grid."""

    values = values.detach().to(dtype=torch.float64, device="cpu")
    outcome_codes = outcome_codes.detach().flatten().to(device="cpu")
    if tuple(values.shape) != (outcome_codes.numel(), 2) or not values.numel():
        raise ValueError("pose XY values must be a non-empty Nx2 tensor")
    if isinstance(bin_count, int):
        bin_counts = (bin_count, bin_count)
    else:
        bin_counts = tuple(bin_count)
        if len(bin_counts) != 2:
            raise ValueError("pose diagnostic bin count must contain X/Y counts")
    if any(not isinstance(count, int) or count < 1 for count in bin_counts):
        raise ValueError("pose diagnostic bin counts must be positive integers")
    if not torch.isfinite(values).all():
        raise ValueError("initial pose diagnostics require finite values")
    if not torch.isin(outcome_codes, torch.tensor([0, 1, 2])).all():
        raise ValueError("outcome codes must be 0=failure, 1=success or 2=timeout")

    edges = []
    bin_ids = []
    for axis in range(2):
        axis_values = values[:, axis].contiguous()
        lower = float(axis_values.min().item())
        upper = float(axis_values.max().item())
        if lower == upper:
            axis_edges = torch.tensor([lower, upper], dtype=torch.float64)
            axis_bin_ids = torch.zeros_like(outcome_codes, dtype=torch.long)
        else:
            axis_edges = torch.linspace(
                lower, upper, bin_counts[axis] + 1, dtype=torch.float64
            )
            axis_bin_ids = torch.bucketize(axis_values, axis_edges[1:-1])
        edges.append(axis_edges)
        bin_ids.append(axis_bin_ids)

    cells = []
    sampled_cells = 0
    successful_cells = 0
    for y_index in range(len(edges[1]) - 1):
        for x_index in range(len(edges[0]) - 1):
            mask = (bin_ids[0] == x_index) & (bin_ids[1] == y_index)
            samples = int(mask.sum().item())
            successes = int(((outcome_codes == 1) & mask).sum().item())
            physical_failures = int(((outcome_codes == 0) & mask).sum().item())
            timeouts = int(((outcome_codes == 2) & mask).sum().item())
            sampled_cells += int(samples > 0)
            successful_cells += int(successes > 0)
            cells.append(
                {
                    "x_index": x_index,
                    "y_index": y_index,
                    "x_lower": float(edges[0][x_index].item()),
                    "x_upper": float(edges[0][x_index + 1].item()),
                    "y_lower": float(edges[1][y_index].item()),
                    "y_upper": float(edges[1][y_index + 1].item()),
                    "samples": samples,
                    "successes": successes,
                    "physical_failures": physical_failures,
                    "timeouts": timeouts,
                    "success_rate": successes / samples if samples else None,
                }
            )
    return {
        "shape_yx": [len(edges[1]) - 1, len(edges[0]) - 1],
        "x_edges": [float(value.item()) for value in edges[0]],
        "y_edges": [float(value.item()) for value in edges[1]],
        "sampled_cells": sampled_cells,
        "successful_cells": successful_cells,
        "successful_cell_coverage": (
            successful_cells / sampled_cells if sampled_cells else None
        ),
        "cells": cells,
    }


def _stratified_cell_diagnostics(
    cell_ids: torch.Tensor,
    outcome_codes: torch.Tensor,
    *,
    grid: tuple[int, int],
) -> dict[str, object]:
    """Summarize outcomes using the randomizer's exact stratified cell IDs."""

    cell_ids = cell_ids.detach().flatten().to(dtype=torch.long, device="cpu")
    outcome_codes = outcome_codes.detach().flatten().to(device="cpu")
    if cell_ids.shape != outcome_codes.shape or not cell_ids.numel():
        raise ValueError("stratified cell IDs and outcomes must be non-empty peers")
    x_bins, y_bins = grid
    if any(not isinstance(count, int) or count < 1 for count in grid):
        raise ValueError("stratified diagnostic grid counts must be positive integers")
    cell_count = x_bins * y_bins
    if torch.any((cell_ids < 0) | (cell_ids >= cell_count)):
        raise ValueError("stratified cell IDs are outside the diagnostic grid")
    if not torch.isin(outcome_codes, torch.tensor([0, 1, 2])).all():
        raise ValueError("outcome codes must be 0=failure, 1=success or 2=timeout")

    cells = []
    for x_index in range(x_bins):
        for y_index in range(y_bins):
            cell_id = x_index * y_bins + y_index
            selected = cell_ids == cell_id
            samples = int(selected.sum().item())
            successes = int(((outcome_codes == 1) & selected).sum().item())
            physical_failures = int(
                ((outcome_codes == 0) & selected).sum().item()
            )
            timeouts = int(((outcome_codes == 2) & selected).sum().item())
            cells.append(
                {
                    "cell_id": cell_id,
                    "x_index": x_index,
                    "y_index": y_index,
                    "samples": samples,
                    "successes": successes,
                    "physical_failures": physical_failures,
                    "timeouts": timeouts,
                    "success_rate": successes / samples if samples else None,
                }
            )
    return {
        "shape_yx": [y_bins, x_bins],
        "sampled_cells": sum(cell["samples"] > 0 for cell in cells),
        "successful_cells": sum(cell["successes"] > 0 for cell in cells),
        "cells": cells,
    }


def _initial_pose_diagnostics(
    initial_poses: dict[str, torch.Tensor],
    outcome_codes: torch.Tensor,
    *,
    target_xy_bin_count: int | tuple[int, int] = 8,
) -> dict[str, object]:
    required = {
        "target_translation_xy": 2,
        "target_yaw": 1,
        "robot_base_translation_xy": 2,
        "robot_base_yaw": 1,
    }
    missing = required.keys() - initial_poses.keys()
    if missing:
        raise ValueError(f"initial pose diagnostics are missing {sorted(missing)}")
    episodes = int(outcome_codes.numel())
    values: dict[str, torch.Tensor] = {}
    for name, width in required.items():
        tensor = initial_poses[name]
        expected = (episodes,) if width == 1 else (episodes, width)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if width == 1:
            values[name] = tensor
        else:
            values[f"{name}_x"] = tensor[:, 0]
            values[f"{name}_y"] = tensor[:, 1]

    return {
        "sample_count": episodes,
        "outcome_counts": {
            "success": int((outcome_codes == 1).sum().item()),
            "physical_failure": int((outcome_codes == 0).sum().item()),
            "timeout": int((outcome_codes == 2).sum().item()),
        },
        "target_translation_xy": {
            "x": _pose_axis_diagnostics(
                values["target_translation_xy_x"], outcome_codes
            ),
            "y": _pose_axis_diagnostics(
                values["target_translation_xy_y"], outcome_codes
            ),
            "grid": _pose_xy_diagnostics(
                initial_poses["target_translation_xy"],
                outcome_codes,
                bin_count=target_xy_bin_count,
            ),
        },
        "target_yaw": _pose_axis_diagnostics(values["target_yaw"], outcome_codes),
        "robot_base_translation_xy": {
            "x": _pose_axis_diagnostics(
                values["robot_base_translation_xy_x"], outcome_codes
            ),
            "y": _pose_axis_diagnostics(
                values["robot_base_translation_xy_y"], outcome_codes
            ),
        },
        "robot_base_yaw": _pose_axis_diagnostics(
            values["robot_base_yaw"], outcome_codes
        ),
    }


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.minimum_success_rate is not None and not (
        0.0 <= args.minimum_success_rate <= 1.0
    ):
        raise ValueError("minimum-success-rate must be in [0, 1]")
    if args.minimum_spatial_cell_success_rate is not None and not (
        0.0 <= args.minimum_spatial_cell_success_rate <= 1.0
    ):
        raise ValueError("minimum-spatial-cell-success-rate must be in [0, 1]")
    if args.minimum_focus_cell_success_rate is not None and not (
        0.0 <= args.minimum_focus_cell_success_rate <= 1.0
    ):
        raise ValueError("minimum-focus-cell-success-rate must be in [0, 1]")
    if args.reference_only and args.proposal_only:
        raise ValueError("reference-only and proposal-only are mutually exclusive")
    baseline_only = args.reference_only or args.proposal_only
    if baseline_only and args.stochastic_policy:
        raise ValueError("stochastic-policy is only valid for a PPO checkpoint")
    if not baseline_only and args.checkpoint is None:
        raise ValueError("policy evaluation requires --checkpoint")
    config = _config(args)
    focus_cell_ids = (
        config.domain_randomization.target_position_stratified_focus_cell_ids
    )
    if args.minimum_focus_cell_success_rate is not None and not focus_cell_ids:
        raise ValueError(
            "minimum-focus-cell-success-rate requires explicitly focused "
            "stratified target-position cells"
        )
    _seed_torch(config.seed)
    if args.episodes > config.num_envs:
        raise ValueError(
            "paired evaluation requires episodes <= num-envs so each result "
            "comes from one unique initial world"
        )
    dr_strength = _evaluation_dr_strength(args)
    env = GpuGraspVecEnv(
        config,
        training=False,
        randomization_enabled=dr_strength > 0.0,
    )
    actor = None
    if not baseline_only:
        assert args.checkpoint is not None
        actor = _load_evaluation_actor(
            env, config, args.checkpoint.resolve(), args
        )
    elif baseline_only:
        # Runner construction queries observations once before PPO evaluation.
        # Consume the same state extraction/noise draw so both baselines and
        # PPO start from identical policy state.  Proposal-only additionally
        # uses the resulting noise-matched reference command.
        env.get_observations()
    # Construct every routed expert before the evaluation reset.  Actor loading
    # must not perturb the physical worlds used for same-seed comparisons.
    if dr_strength > 0.0:
        _set_evaluation_dr_strength(env, dr_strength)
    observations = env.get_observations()
    (
        initial_world_sha256,
        initial_policy_state_sha256,
        initial_proposal_context_sha256,
    ) = _evaluation_world_sha256(env, observations, args.episodes)
    initial_poses = {
        "target_translation_xy": env.randomizer.target_translation_xy[: args.episodes]
        .detach()
        .clone(),
        "target_yaw": env.randomizer.target_yaw[: args.episodes].detach().clone(),
        "robot_base_translation_xy": env.randomizer.robot_base_translation_xy[
            : args.episodes
        ]
        .detach()
        .clone(),
        "robot_base_yaw": env.randomizer.robot_base_yaw[: args.episodes]
        .detach()
        .clone(),
    }
    initial_target_position_cell_ids = env.randomizer.target_position_cell_ids[
        : args.episodes
    ].detach().clone()
    checkpoint_training_dr_strength = (
        None
        if args.checkpoint is None
        else _checkpoint_training_dr_strength(args.checkpoint.resolve())
    )
    successes = 0
    failures = 0
    timeouts = 0
    native_successes = 0
    grasp_episodes = 0
    episodes = 0
    episode_max_lift = torch.full((env.num_envs,), -torch.inf, device=env.device)
    episode_max_grasp_quality = torch.zeros(env.num_envs, device=env.device)
    episode_native_success = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    episode_had_grasp = torch.zeros_like(episode_native_success)
    episode_max_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episode_return = torch.zeros(env.num_envs, device=env.device)
    episode_task_return = torch.zeros_like(episode_return)
    episode_reference_return = torch.zeros_like(episode_return)
    completed_max_lift: list[float] = []
    completed_max_grasp_quality: list[float] = []
    completed_max_stage: list[int] = []
    completed_return: list[float] = []
    completed_task_return: list[float] = []
    completed_reference_return: list[float] = []
    completed_success: list[bool] = []
    completed_world = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    evaluation_world = torch.arange(env.num_envs, device=env.device) < args.episodes
    outcomes = torch.full((env.num_envs,), -1, dtype=torch.int8, device=env.device)
    terminal_outcomes = torch.full_like(outcomes, -1)
    action_delta_sum = 0.0
    action_delta_count = 0
    max_action_delta = 0.0
    effective_action_delta_sum = 0.0
    effective_action_delta_count = 0
    max_effective_action_delta = 0.0
    while episodes < args.episodes:
        stage_index = getattr(env.reward, "stage_index", None)
        if stage_index is not None:
            episode_max_stage.copy_(torch.maximum(episode_max_stage, stage_index))
        if args.reference_only:
            actions = env.reference.current_action().clone()
        elif args.proposal_only:
            base_dim = env.reference.observation_dim
            actions = observations["policy"][:, base_dim : base_dim + ACTION_DIM]
        else:
            assert actor is not None
            actions = actor(observations, stochastic_output=args.stochastic_policy)
        active_world = evaluation_world & ~completed_world
        action_delta = (actions - env.reference.current_action()).abs()[active_world]
        action_delta_sum += float(action_delta.sum().item())
        action_delta_count += int(action_delta.numel())
        if action_delta.numel():
            max_action_delta = max(max_action_delta, float(action_delta.max().item()))
        effective_action = env._bounded_reference_action(actions)
        effective_delta = (effective_action - env.reference.current_action()).abs()[
            active_world
        ]
        effective_action_delta_sum += float(effective_delta.sum().item())
        effective_action_delta_count += int(effective_delta.numel())
        if effective_delta.numel():
            max_effective_action_delta = max(
                max_effective_action_delta, float(effective_delta.max().item())
            )
        observations, rewards, dones, _ = env.step(actions)
        assert env.last_terms is not None
        terms = env.last_terms
        task_rewards = terms.task_reward()
        episode_return.add_(rewards)
        episode_task_return.add_(task_rewards)
        episode_reference_return.add_(rewards - task_rewards)
        episode_max_lift.copy_(torch.maximum(episode_max_lift, terms.lift_height))
        episode_max_grasp_quality.copy_(
            torch.maximum(episode_max_grasp_quality, terms.grasp_quality)
        )
        episode_native_success.logical_or_(terms.native_success)
        episode_had_grasp.logical_or_(terms.is_grasp)
        finished = (
            (dones & evaluation_world & ~completed_world)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(finished):
            selected = finished
            successes += int(terms.success[selected].sum().item())
            failures += int(terms.failure[selected].sum().item())
            timeouts += int(terms.timeout[selected].sum().item())
            native_successes += int(episode_native_success[selected].sum().item())
            grasp_episodes += int(episode_had_grasp[selected].sum().item())
            completed_max_lift.extend(
                episode_max_lift[selected].detach().cpu().tolist()
            )
            completed_max_grasp_quality.extend(
                episode_max_grasp_quality[selected].detach().cpu().tolist()
            )
            completed_max_stage.extend(
                episode_max_stage[selected].detach().cpu().tolist()
            )
            completed_return.extend(episode_return[selected].detach().cpu().tolist())
            completed_task_return.extend(
                episode_task_return[selected].detach().cpu().tolist()
            )
            completed_reference_return.extend(
                episode_reference_return[selected].detach().cpu().tolist()
            )
            completed_success.extend(terms.success[selected].detach().cpu().tolist())
            episodes += len(selected)
            completed_world[selected] = True
            outcomes[selected] = torch.where(
                terms.success[selected],
                torch.ones_like(outcomes[selected]),
                torch.zeros_like(outcomes[selected]),
            )
            terminal_outcomes[selected] = torch.where(
                terms.success[selected],
                torch.ones_like(terminal_outcomes[selected]),
                torch.where(
                    terms.failure[selected],
                    torch.zeros_like(terminal_outcomes[selected]),
                    torch.full_like(terminal_outcomes[selected], 2),
                ),
            )
            episode_max_lift[finished] = -torch.inf
            episode_max_grasp_quality[finished] = 0.0
            episode_native_success[finished] = False
            episode_had_grasp[finished] = False
            episode_max_stage[finished] = 0
            episode_return[finished] = 0.0
            episode_task_return[finished] = 0.0
            episode_reference_return[finished] = 0.0
    stage_names = getattr(getattr(env.state_reader, "spec", None), "stages", ())
    max_stage_counts = {
        (
            stage_names[index].name if index < len(stage_names) else str(index)
        ): completed_max_stage.count(index)
        for index in sorted(set(completed_max_stage))
    }
    successful_task_returns = [
        value
        for value, success in zip(completed_task_return, completed_success, strict=True)
        if success
    ]
    failed_task_returns = [
        value
        for value, success in zip(completed_task_return, completed_success, strict=True)
        if not success
    ]
    initial_pose_diagnostics = _initial_pose_diagnostics(
        initial_poses,
        terminal_outcomes[: args.episodes],
        target_xy_bin_count=(
            config.domain_randomization.target_position_stratified_grid or 8
        ),
    )
    stratified_spatial_grid = (
        None
        if (
            dr_strength <= 0.0
            or config.domain_randomization.target_position_stratified_grid is None
        )
        else _stratified_cell_diagnostics(
            initial_target_position_cell_ids,
            terminal_outcomes[: args.episodes],
            grid=config.domain_randomization.target_position_stratified_grid,
        )
    )
    spatial_grid = (
        stratified_spatial_grid
        if stratified_spatial_grid is not None
        else initial_pose_diagnostics["target_translation_xy"]["grid"]
    )
    sampled_cell_rates = [
        cell["success_rate"]
        for cell in spatial_grid["cells"]
        if cell["samples"] > 0
    ]
    minimum_spatial_cell_success_rate = (
        min(sampled_cell_rates) if sampled_cell_rates else None
    )
    focus_cell_rates: list[float] = []
    if stratified_spatial_grid is not None and focus_cell_ids:
        cells_by_id = {
            cell["cell_id"]: cell for cell in stratified_spatial_grid["cells"]
        }
        if all(cells_by_id[cell_id]["samples"] > 0 for cell_id in focus_cell_ids):
            focus_cell_rates = [
                cells_by_id[cell_id]["success_rate"] for cell_id in focus_cell_ids
            ]
    minimum_focus_cell_success_rate = (
        min(focus_cell_rates) if focus_cell_rates else None
    )
    result = {
        "mode": (
            "reference_only"
            if args.reference_only
            else "proposal_only"
            if args.proposal_only
            else "ppo"
        ),
        "checkpoint": (None if args.checkpoint is None else str(args.checkpoint)),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "failures": failures,
        "failure_rate": failures / episodes,
        "timeouts": timeouts,
        "timeout_rate": timeouts / episodes,
        "native_success_rate": native_successes / episodes,
        "grasp_episode_rate": grasp_episodes / episodes,
        "mean_episode_return": sum(completed_return) / episodes,
        "mean_task_return": sum(completed_task_return) / episodes,
        "mean_reference_return": sum(completed_reference_return) / episodes,
        "mean_success_task_return": (
            sum(successful_task_returns) / len(successful_task_returns)
            if successful_task_returns
            else None
        ),
        "mean_failed_task_return": (
            sum(failed_task_returns) / len(failed_task_returns)
            if failed_task_returns
            else None
        ),
        "mean_max_lift": sum(completed_max_lift) / episodes,
        "max_lift": max(completed_max_lift),
        "mean_max_grasp_quality": sum(completed_max_grasp_quality) / episodes,
        "max_stage_counts": max_stage_counts,
        "domain_randomization": dr_strength > 0.0,
        "evaluation_dr_strength": dr_strength,
        "checkpoint_training_dr_strength": checkpoint_training_dr_strength,
        "evaluation_matches_checkpoint_dr": (
            None
            if checkpoint_training_dr_strength is None
            else abs(checkpoint_training_dr_strength - dr_strength) < 1e-9
        ),
        "initial_world_sha256": initial_world_sha256,
        "initial_policy_state_sha256": initial_policy_state_sha256,
        "initial_proposal_context_sha256": initial_proposal_context_sha256,
        "reference_noise": bool(
            dr_strength > 0.0 and config.domain_randomization.reference_noise.enabled
        ),
        "dr_profile": args.dr_profile,
        "device": config.device,
        "deterministic_actor": not args.stochastic_policy,
        "policy_sampling": "ppo_gaussian" if args.stochastic_policy else "mean",
        "policy_seed": config.seed,
        "spatial_policy_router": (
            actor.metadata()
            if isinstance(actor, (SpatialPolicyRouter, SpatialCheckpointRouter))
            else None
        ),
        "mean_absolute_action_delta": action_delta_sum / action_delta_count,
        "max_absolute_action_delta": max_action_delta,
        "mean_absolute_effective_action_delta": (
            effective_action_delta_sum / effective_action_delta_count
        ),
        "max_absolute_effective_action_delta": max_effective_action_delta,
        "success_world_ids": (
            (outcomes == 1).nonzero(as_tuple=False).flatten().cpu().tolist()
        ),
        "failed_world_ids": (
            (outcomes == 0).nonzero(as_tuple=False).flatten().cpu().tolist()
        ),
        "minimum_spatial_cell_success_rate": minimum_spatial_cell_success_rate,
        "minimum_focus_cell_success_rate": minimum_focus_cell_success_rate,
        "stratified_target_position_grid": stratified_spatial_grid,
        "initial_pose_diagnostics": initial_pose_diagnostics,
    }
    if (
        args.minimum_success_rate is not None
        and result["success_rate"] < args.minimum_success_rate
    ):
        raise RuntimeError(
            "Evaluation success-rate gate failed: "
            f"{result['success_rate']:.6f} < {args.minimum_success_rate:.6f}"
        )
    if (
        args.minimum_spatial_cell_success_rate is not None
        and (
            minimum_spatial_cell_success_rate is None
            or minimum_spatial_cell_success_rate
            < args.minimum_spatial_cell_success_rate
        )
    ):
        raise RuntimeError(
            "Evaluation spatial-cell gate failed: "
            f"{minimum_spatial_cell_success_rate} < "
            f"{args.minimum_spatial_cell_success_rate:.6f}"
        )
    if (
        args.minimum_focus_cell_success_rate is not None
        and (
            minimum_focus_cell_success_rate is None
            or minimum_focus_cell_success_rate
            < args.minimum_focus_cell_success_rate
        )
    ):
        raise RuntimeError(
            "Evaluation focus-cell gate failed: "
            f"{minimum_focus_cell_success_rate} < "
            f"{args.minimum_focus_cell_success_rate:.6f}"
        )
    return result


def _finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


@torch.inference_mode()
def _shadow_reward(args: argparse.Namespace) -> dict[str, object]:
    """Measure Robometer predictions without changing executed task rewards."""

    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    config = _config(args)
    reward_config = _robometer_config(args, mode="shadow")
    assert reward_config is not None
    _validate_robometer_run(config, reward_config)
    _seed_torch(config.seed)

    env = GpuGraspVecEnv(
        config,
        training=False,
        randomization_enabled=True,
        capture_step_data=True,
        robometer_reward_config=reward_config,
    )
    try:
        _set_evaluation_dr_strength(env, 1.0)
        actor = None
        if args.checkpoint is not None:
            train_config = ppo_train_config(
                smoke=True,
                plan_conditioned_actor=checkpoint_uses_plan_conditioned_actor(
                    args.checkpoint
                ),
                residual_action_groups=getattr(
                    config, "residual_action_groups", ("right_hand", "right_arm")
                ),
            )
            runner = GpuPpoRunner(env, train_config, log_dir=None)
            runner.load_actor_warm_start(args.checkpoint.resolve())
            actor = runner.alg.get_policy().eval()

        observations = env.get_observations()
        episode_length = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        episode_return = torch.zeros(env.num_envs, device=env.device)
        physical_task_return = torch.zeros_like(episode_return)
        max_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        max_progress = torch.full_like(episode_return, -torch.inf)
        cumulative_progress_delta = torch.zeros_like(episode_return)
        final_progress = torch.full_like(episode_return, float("nan"))
        final_success_probability = torch.full_like(episode_return, float("nan"))
        max_success_probability = torch.full_like(episode_return, -torch.inf)
        inference_count = torch.zeros_like(episode_length)
        latency_sum_ms = torch.zeros_like(episode_return)
        records: list[dict[str, object]] = []

        while len(records) < args.episodes:
            if args.proposal_only:
                base_dim = env.reference.observation_dim
                actions = observations["policy"][:, base_dim : base_dim + ACTION_DIM]
            else:
                assert actor is not None
                actions = actor(observations, stochastic_output=False)

            observations, rewards, dones, extras = env.step(actions)
            terms = env.last_terms
            if terms is None:
                raise RuntimeError("GPU environment did not expose reward terms")
            step_data = extras.get("step_data")
            robometer = extras.get("robometer")
            if not isinstance(step_data, dict) or not isinstance(robometer, dict):
                raise TypeError("Shadow run did not expose reward diagnostics")

            stages = step_data["stage_index"]
            physical_rewards = step_data["physical_task_reward"]
            inferred = robometer["inferred"]
            progress = robometer["progress"]
            progress_delta = robometer["progress_delta"]
            success_probability = robometer["success_probability"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    stages,
                    physical_rewards,
                    inferred,
                    progress,
                    progress_delta,
                    success_probability,
                )
            ):
                raise RuntimeError("Shadow diagnostics contain non-tensor values")

            episode_length.add_(1)
            episode_return.add_(rewards)
            physical_task_return.add_(physical_rewards)
            max_stage.copy_(torch.maximum(max_stage, stages))
            if inferred.any():
                max_progress[inferred] = torch.maximum(
                    max_progress[inferred], progress[inferred]
                )
                cumulative_progress_delta[inferred] += progress_delta[inferred]
                final_progress[inferred] = progress[inferred]
                final_success_probability[inferred] = success_probability[inferred]
                max_success_probability[inferred] = torch.maximum(
                    max_success_probability[inferred],
                    success_probability[inferred],
                )
                inference_count[inferred] += 1
                latency_sum_ms[inferred] += float(robometer["latency_ms"])

            finished = dones.nonzero(as_tuple=False).flatten()
            for env_id in finished.detach().cpu().tolist():
                if len(records) >= args.episodes:
                    break
                env_id = int(env_id)
                success = bool(terms.success[env_id].item())
                failure = bool(terms.failure[env_id].item())
                timeout = bool(terms.timeout[env_id].item())
                outcome = (
                    "success"
                    if success
                    else "failure"
                    if failure
                    else "timeout"
                    if timeout
                    else "unknown"
                )
                count = int(inference_count[env_id].item())
                records.append(
                    {
                        "episode": len(records),
                        "env_id": env_id,
                        "physical_outcome": outcome,
                        "physical_success": success,
                        "physical_failure": failure,
                        "physical_timeout": timeout,
                        "length": int(episode_length[env_id].item()),
                        "max_stage": int(max_stage[env_id].item()),
                        "final_progress": float(final_progress[env_id].item()),
                        "max_progress": float(max_progress[env_id].item()),
                        "cumulative_progress_delta": float(
                            cumulative_progress_delta[env_id].item()
                        ),
                        "final_success_probability": float(
                            final_success_probability[env_id].item()
                        ),
                        "max_success_probability": float(
                            max_success_probability[env_id].item()
                        ),
                        "robometer_inferences": count,
                        "robometer_latency_ms_sum": float(
                            latency_sum_ms[env_id].item()
                        ),
                        "robometer_latency_ms_mean": (
                            float(latency_sum_ms[env_id].item()) / count
                            if count
                            else None
                        ),
                        "physical_task_return": float(
                            physical_task_return[env_id].item()
                        ),
                        "episode_return": float(episode_return[env_id].item()),
                    }
                )

            if len(finished):
                episode_length[finished] = 0
                episode_return[finished] = 0.0
                physical_task_return[finished] = 0.0
                max_stage[finished] = 0
                max_progress[finished] = -torch.inf
                cumulative_progress_delta[finished] = 0.0
                final_progress[finished] = float("nan")
                final_success_probability[finished] = float("nan")
                max_success_probability[finished] = -torch.inf
                inference_count[finished] = 0
                latency_sum_ms[finished] = 0.0

        stage_names = getattr(getattr(env.state_reader, "spec", None), "stages", ())
        for record in records:
            stage_index = int(record["max_stage"])
            record["max_stage_name"] = (
                stage_names[stage_index].name
                if stage_index < len(stage_names)
                else str(stage_index)
            )

        successes = [record for record in records if record["physical_success"]]
        non_successes = [record for record in records if not record["physical_success"]]
        result: dict[str, object] = {
            "mode": "shadow",
            "policy": "proposal_only" if args.proposal_only else "ppo",
            "checkpoint": (
                None if args.checkpoint is None else str(args.checkpoint.resolve())
            ),
            "checkpoint_sha256": (
                None
                if args.checkpoint is None
                else sha256_file(args.checkpoint.resolve())
            ),
            "episodes": len(records),
            "successes": len(successes),
            "success_rate": len(successes) / len(records),
            "full_domain_randomization": True,
            "physical_task_reward_active": True,
            "task_reward_override": reward_config.metadata(),
            "summary": {
                "mean_final_progress": _finite_mean(
                    [float(record["final_progress"]) for record in records]
                ),
                "mean_success_final_progress": _finite_mean(
                    [float(record["final_progress"]) for record in successes]
                ),
                "mean_non_success_final_progress": _finite_mean(
                    [float(record["final_progress"]) for record in non_successes]
                ),
                "mean_max_progress": _finite_mean(
                    [float(record["max_progress"]) for record in records]
                ),
                "mean_cumulative_progress_delta": _finite_mean(
                    [float(record["cumulative_progress_delta"]) for record in records]
                ),
                "mean_final_success_probability": _finite_mean(
                    [float(record["final_success_probability"]) for record in records]
                ),
                "mean_inference_latency_ms": _finite_mean(
                    [
                        float(record["robometer_latency_ms_mean"])
                        for record in records
                        if record["robometer_latency_ms_mean"] is not None
                    ]
                ),
            },
            "episode_results": records,
        }
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
        return {**result, "output": str(output)}
    finally:
        env.close()


def _paired_mode_args(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "command": "evaluate",
            "reference_only": mode == "reference_only",
            "proposal_only": mode == "proposal_only",
            "checkpoint": None if mode != "ppo" else args.checkpoint,
            "stochastic_policy": False,
        }
    )
    return argparse.Namespace(**values)


@torch.inference_mode()
def _benchmark(args: argparse.Namespace) -> dict[str, object]:
    results: dict[str, dict[str, object]] = {}
    for mode in ("reference_only", "proposal_only", "ppo"):
        results[mode] = _evaluate(_paired_mode_args(args, mode))
        gc.collect()
        torch.cuda.empty_cache()
    paired = compare_paired_results(
        results["reference_only"],
        results["proposal_only"],
        results["ppo"],
    )
    paired["checkpoint"] = str(args.checkpoint)
    paired["checkpoint_sha256"] = sha256_file(args.checkpoint.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(paired, indent=2))
    return {**paired, "output": str(args.output)}


@torch.inference_mode()
def _collect(args: argparse.Namespace) -> dict[str, object]:
    if args.successes < 1:
        raise ValueError("successes must be positive")
    max_attempts = args.max_attempts or 3 * args.successes
    config = _config(args)
    _seed_torch(config.seed)
    dr_strength = _evaluation_dr_strength(args)
    env = GpuGraspVecEnv(
        config,
        training=False,
        randomization_enabled=dr_strength > 0.0,
        capture_terminal_qpos=True,
        capture_step_data=True,
    )
    try:
        actor = _load_evaluation_actor(
            env, config, args.checkpoint.resolve(), args
        )
        if dr_strength > 0.0:
            _set_evaluation_dr_strength(env, dr_strength)
        return collect_successful_trajectories(
            env,
            actor,
            args.checkpoint.resolve(),
            args.output_dir.resolve(),
            successes=args.successes,
            max_attempts=max_attempts,
            domain_randomization=dr_strength > 0.0,
            stochastic_policy=args.stochastic_policy,
        )
    finally:
        env.close()


def _collect_dataset(args: argparse.Namespace) -> dict[str, object]:
    for name in ("width", "height", "fps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty dataset root: {output_root}"
        )
    rollouts = output_root / "rollouts"
    values = vars(args).copy()
    values.update(command="collect", output_dir=rollouts)
    rollout_result = _collect(argparse.Namespace(**values))
    gc.collect()
    torch.cuda.empty_cache()
    dataset_result = export_dual_dataset(
        rollouts,
        output_root,
        args.asset_bundle.resolve(),
        args.source_dataset.resolve(),
        args.psi0_template.resolve(),
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    return {"rollout": rollout_result, "dataset": dataset_result}


@torch.inference_mode()
def _record(args: argparse.Namespace) -> dict[str, object]:
    for name in ("videos", "max_attempts", "width", "height", "fps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.reference_only and args.stochastic_policy:
        raise ValueError("stochastic-policy is only valid for a PPO checkpoint")
    if (
        not math.isfinite(args.maximum_precontact_target_motion_m)
        or args.maximum_precontact_target_motion_m < 0.0
    ):
        raise ValueError("maximum-precontact-target-motion-m must be non-negative")
    config = _config(args)
    _seed_torch(config.seed)
    dr_strength = _evaluation_dr_strength(args)
    env = GpuGraspVecEnv(
        config,
        training=False,
        randomization_enabled=dr_strength > 0.0,
        capture_terminal_qpos=True,
    )
    actor = None
    checkpoint = None
    if not args.reference_only:
        assert args.checkpoint is not None
        checkpoint = args.checkpoint.resolve()
        actor = _load_evaluation_actor(env, config, checkpoint, args)
    if dr_strength > 0.0:
        _set_evaluation_dr_strength(env, dr_strength)
    return record_success_videos(
        env,
        actor,
        checkpoint,
        args.output_dir.resolve(),
        videos=args.videos,
        max_attempts=args.max_attempts,
        width=args.width,
        height=args.height,
        fps=args.fps,
        domain_randomization=dr_strength > 0.0,
        allow_diagnostic_fallback=args.allow_diagnostic_fallback,
        camera_view=args.camera_view,
        stochastic_policy=args.stochastic_policy,
        reference_only=args.reference_only,
        policy_provenance=(
            {"spatial_policy_router": actor.metadata()}
            if isinstance(actor, (SpatialPolicyRouter, SpatialCheckpointRouter))
            else None
        ),
        maximum_precontact_target_displacement_m=(
            args.maximum_precontact_target_motion_m
        ),
    )


def main() -> None:
    args = _parser().parse_args()
    handlers = {
        "train": _train,
        "shadow-reward": _shadow_reward,
        "evaluate": _evaluate,
        "benchmark": _benchmark,
        "collect": _collect,
        "collect-dataset": _collect_dataset,
        "record": _record,
        "verify-release": lambda value: verify_release(value.release_dir),
        "audit-dataset": lambda value: audit_ppo_dataset(
            value.dataset_root,
            expected_successes=value.expected_successes,
            expected_task=value.expected_task,
            expected_dr_strength=value.expected_dr_strength,
            require_full_dr_coverage=value.require_full_dr_coverage,
        ),
    }
    result = handlers[args.command](args)
    print(json.dumps({"result": result}, indent=2))


if __name__ == "__main__":
    main()
