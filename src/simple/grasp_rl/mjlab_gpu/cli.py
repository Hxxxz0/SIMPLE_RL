"""Minimal command line entrypoint for GPU-only mjlab PPO."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from simple.grasp_rl.mjlab_gpu.benchmark import compare_paired_results
from simple.grasp_rl.mjlab_gpu.collect import collect_successful_trajectories
from simple.grasp_rl.mjlab_gpu.config import MjlabPpoConfig
from simple.grasp_rl.mjlab_gpu.recording import record_success_videos
from simple.grasp_rl.mjlab_gpu.release import sha256_file, verify_release
from simple.grasp_rl.mjlab_gpu.runner import (
    GpuPpoRunner,
    checkpoint_uses_plan_conditioned_actor,
    ppo_train_config,
)
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv
from simple.grasp_rl.schema import ACTION_DIM


def _common(parser: argparse.ArgumentParser, *, num_envs: int = 4096) -> None:
    parser.add_argument("--task", default="tabletop_grasp")
    parser.add_argument("--asset-bundle", type=Path, required=True)
    parser.add_argument("--reference-processed", type=Path, required=True)
    parser.add_argument("--reference-source", default="bc")
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
        "--target-yaw-jitter",
        type=float,
        help=(
            "Override the symmetric target-yaw DR envelope in radians; "
            "omitted keeps the legacy 0.15 default"
        ),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simple-mjlab-ppo")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    _common(train)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--iterations", type=int, default=20_000)
    train.add_argument("--warm-start", type=Path)
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
        "--freeze-actor-normalizer",
        action="store_true",
        help="Keep warm-start actor observation statistics fixed during PPO",
    )

    evaluate = commands.add_parser("evaluate")
    _common(evaluate)
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

    benchmark = commands.add_parser("benchmark")
    _common(benchmark, num_envs=128)
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
    benchmark.set_defaults(stochastic_policy=False)

    record = commands.add_parser("record")
    _common(record, num_envs=32)
    record.set_defaults(smoke=True)
    record.add_argument("--checkpoint", type=Path, required=True)
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

    verify = commands.add_parser("verify-release")
    verify.add_argument("--release-dir", type=Path, required=True)
    return parser


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
        max_reference_action_deviation=args.max_reference_action_deviation,
        reference_target_x_arm_gains=tuple(args.reference_target_x_arm_gains),
        reference_target_y_arm_gains=tuple(args.reference_target_y_arm_gains),
        reference_target_yaw_arm_gains=tuple(args.reference_target_yaw_arm_gains),
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
            ("target_yaw_jitter", args.target_yaw_jitter),
        )
        if value is not None
    }
    if dr_overrides:
        config = replace(
            config,
            domain_randomization=replace(config.domain_randomization, **dr_overrides),
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
    if args.exploration_std is not None and args.exploration_std <= 0.0:
        raise ValueError("exploration-std must be positive")
    if args.ppo_clip_param is not None and not 0.0 < args.ppo_clip_param <= 1.0:
        raise ValueError("ppo-clip-param must be in (0, 1]")
    if args.ppo_learning_epochs is not None and args.ppo_learning_epochs < 1:
        raise ValueError("ppo-learning-epochs must be positive")
    if args.ppo_max_grad_norm is not None and args.ppo_max_grad_norm <= 0.0:
        raise ValueError("ppo-max-grad-norm must be positive")
    if args.ppo_steps_per_env is not None and args.ppo_steps_per_env < 1:
        raise ValueError("ppo-steps-per-env must be positive")
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
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed_torch(config.seed)
    env = GpuGraspVecEnv(config, training=True)
    if args.initial_vector_step:
        env.common_step_counter = args.initial_vector_step
        env._reset(torch.arange(env.num_envs, device=env.device))
    architecture_checkpoint = args.resume or args.warm_start
    plan_conditioned_actor = bool(
        architecture_checkpoint
        and checkpoint_uses_plan_conditioned_actor(architecture_checkpoint)
    )
    train_config = ppo_train_config(
        smoke=config.smoke_mode,
        plan_conditioned_actor=plan_conditioned_actor,
        exploration_std=args.exploration_std,
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
    (output / "config.json").write_text(
        json.dumps(
            {
                "environment": config.resolved(),
                "ppo": train_config,
                "initial_vector_step": args.initial_vector_step,
                "resume_vector_step": args.resume_vector_step,
                "reset_resume_optimizer": args.reset_resume_optimizer,
                "warm_start_critic": args.warm_start_critic,
                "freeze_actor_normalizer": args.freeze_actor_normalizer,
                "actor_learning_rate_scale": args.actor_learning_rate_scale,
            },
            indent=2,
            default=list,
        )
    )
    runner = GpuPpoRunner(
        env,
        train_config,
        log_dir=str(output),
        integrity_path=output / "ppo_integrity.jsonl",
        actor_learning_rate_scale=args.actor_learning_rate_scale,
    )
    if args.resume is not None:
        runner.load(str(args.resume.resolve()))
        if args.reset_resume_optimizer:
            runner.alg.optimizer.state.clear()
        if args.learning_rate is not None:
            runner.set_learning_rate(args.learning_rate)
        if args.exploration_std is not None:
            distribution = runner.alg.get_policy().distribution
            std_param = getattr(distribution, "std_param", None)
            if std_param is None:
                raise ValueError("exploration-std requires scalar Gaussian std")
            with torch.no_grad():
                std_param.fill_(args.exploration_std)
        if args.resume_vector_step is not None:
            env.common_step_counter = args.resume_vector_step
            env._reset(torch.arange(env.num_envs, device=env.device))
    elif args.warm_start is not None:
        runner.load_actor_warm_start(args.warm_start.resolve())
        if args.warm_start_critic:
            runner.load_critic_warm_start(args.warm_start.resolve())
    if args.freeze_actor_normalizer:
        runner.freeze_actor_normalizer()
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


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.reference_only and args.proposal_only:
        raise ValueError("reference-only and proposal-only are mutually exclusive")
    baseline_only = args.reference_only or args.proposal_only
    if baseline_only and args.stochastic_policy:
        raise ValueError("stochastic-policy is only valid for a PPO checkpoint")
    if not baseline_only and args.checkpoint is None:
        raise ValueError("policy evaluation requires --checkpoint")
    config = _config(args)
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
    if dr_strength > 0.0:
        _set_evaluation_dr_strength(env, dr_strength)
    actor = None
    if not baseline_only:
        assert args.checkpoint is not None
        train_config = ppo_train_config(
            smoke=True,
            plan_conditioned_actor=checkpoint_uses_plan_conditioned_actor(
                args.checkpoint
            ),
        )
        runner = GpuPpoRunner(env, train_config, log_dir=None)
        runner.load_actor_warm_start(args.checkpoint.resolve())
        actor = runner.alg.get_policy().eval()
    elif baseline_only:
        # Runner construction queries observations once before PPO evaluation.
        # Consume the same state extraction/noise draw so both baselines and
        # PPO start from identical policy state.  Proposal-only additionally
        # uses the resulting noise-matched reference command.
        env.get_observations()
    observations = env.get_observations()
    (
        initial_world_sha256,
        initial_policy_state_sha256,
        initial_proposal_context_sha256,
    ) = _evaluation_world_sha256(env, observations, args.episodes)
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
    return {
        "mode": (
            "reference_only"
            if args.reference_only
            else "proposal_only"
            if args.proposal_only
            else "ppo"
        ),
        "checkpoint": (
            None if args.checkpoint is None else str(args.checkpoint)
        ),
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
    }


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
    if dr_strength > 0.0:
        _set_evaluation_dr_strength(env, dr_strength)
    train_config = ppo_train_config(
        smoke=True,
        plan_conditioned_actor=checkpoint_uses_plan_conditioned_actor(args.checkpoint),
    )
    runner = GpuPpoRunner(env, train_config, log_dir=None)
    runner.load_actor_warm_start(args.checkpoint.resolve())
    return collect_successful_trajectories(
        env,
        runner.alg.get_policy().eval(),
        args.checkpoint.resolve(),
        args.output_dir.resolve(),
        successes=args.successes,
        max_attempts=max_attempts,
        domain_randomization=dr_strength > 0.0,
        stochastic_policy=args.stochastic_policy,
    )


@torch.inference_mode()
def _record(args: argparse.Namespace) -> dict[str, object]:
    for name in ("videos", "max_attempts", "width", "height", "fps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    config = _config(args)
    _seed_torch(config.seed)
    dr_strength = _evaluation_dr_strength(args)
    env = GpuGraspVecEnv(
        config,
        training=False,
        randomization_enabled=dr_strength > 0.0,
        capture_terminal_qpos=True,
    )
    if dr_strength > 0.0:
        _set_evaluation_dr_strength(env, dr_strength)
    train_config = ppo_train_config(
        smoke=True,
        plan_conditioned_actor=checkpoint_uses_plan_conditioned_actor(args.checkpoint),
    )
    runner = GpuPpoRunner(env, train_config, log_dir=None)
    runner.load_actor_warm_start(args.checkpoint.resolve())
    return record_success_videos(
        env,
        runner.alg.get_policy().eval(),
        args.checkpoint.resolve(),
        args.output_dir.resolve(),
        videos=args.videos,
        max_attempts=args.max_attempts,
        width=args.width,
        height=args.height,
        fps=args.fps,
        domain_randomization=args.stress_domain_randomization,
        allow_diagnostic_fallback=args.allow_diagnostic_fallback,
        camera_view=args.camera_view,
        stochastic_policy=args.stochastic_policy,
    )


def main() -> None:
    args = _parser().parse_args()
    handlers = {
        "train": _train,
        "evaluate": _evaluate,
        "benchmark": _benchmark,
        "collect": _collect,
        "record": _record,
        "verify-release": lambda value: verify_release(value.release_dir),
    }
    result = handlers[args.command](args)
    print(json.dumps({"result": result}, indent=2))


if __name__ == "__main__":
    main()
