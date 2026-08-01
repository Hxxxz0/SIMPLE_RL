"""Minimal command line entrypoint for GPU-only mjlab PPO."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import torch

from simple.grasp_rl.mjlab_gpu.config import MjlabPpoConfig
from simple.grasp_rl.mjlab_gpu.runner import (
    GpuPpoRunner,
    checkpoint_uses_plan_conditioned_actor,
    ppo_train_config,
)
from simple.grasp_rl.mjlab_gpu.recording import record_success_videos
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv


def _common(parser: argparse.ArgumentParser, *, num_envs: int = 4096) -> None:
    parser.add_argument("--task", default="tabletop_grasp")
    parser.add_argument("--asset-bundle", type=Path, required=True)
    parser.add_argument("--reference-processed", type=Path, required=True)
    parser.add_argument("--reference-source", default="bc")
    parser.add_argument("--num-envs", type=int, default=num_envs)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dr-initial-strength", type=float)
    parser.add_argument("--dr-warmup-steps", type=int)
    parser.add_argument("--dr-ramp-steps", type=int)
    parser.add_argument(
        "--dr-profile",
        choices=("full", "pose_only"),
        default="full",
        help="Stage target-pose adaptation before the unchanged full-DR profile",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simple-mjlab-ppo")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    _common(train)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--iterations", type=int, default=10_000)
    train.add_argument("--warm-start", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--initial-vector-step",
        type=int,
        default=0,
        help="Start a warm-start run at this DR curriculum vector step",
    )
    train.add_argument(
        "--learning-rate",
        type=float,
        help="Override the reviewed PPO learning rate for a continuation run",
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

    evaluate = commands.add_parser("evaluate")
    _common(evaluate)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument(
        "--reference-only",
        action="store_true",
        help="Execute the replay reference directly without constructing a PPO actor",
    )
    evaluate.add_argument("--episodes", type=int, default=200)
    evaluate.add_argument(
        "--stress-domain-randomization",
        action="store_true",
        help="Evaluate deterministic policy under full physics/reference DR",
    )

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
        "--stress-domain-randomization",
        action="store_true",
        help="Record deterministic policy under full physics/reference DR",
    )
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
    )
    dr_overrides = {
        name: value
        for name, value in (
            ("curriculum_initial_strength", args.dr_initial_strength),
            ("curriculum_warmup_steps", args.dr_warmup_steps),
            ("curriculum_ramp_steps", args.dr_ramp_steps),
        )
        if value is not None
    }
    if dr_overrides:
        config = replace(
            config,
            domain_randomization=replace(
                config.domain_randomization, **dr_overrides
            ),
        )
    if args.dr_profile == "pose_only":
        config = replace(
            config,
            domain_randomization=config.domain_randomization.pose_only(),
        )
    return config


def _train(args: argparse.Namespace) -> dict[str, object]:
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.initial_vector_step < 0:
        raise ValueError("initial-vector-step must be non-negative")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.exploration_std is not None and args.exploration_std <= 0.0:
        raise ValueError("exploration-std must be positive")
    if args.resume is not None and args.warm_start is not None:
        raise ValueError("resume and warm-start are mutually exclusive")
    if args.resume is not None and args.initial_vector_step:
        raise ValueError("resume restores vector step; do not override it")
    config = _config(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
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
    (output / "config.json").write_text(
        json.dumps(
            {
                "environment": config.resolved(),
                "ppo": train_config,
                "initial_vector_step": args.initial_vector_step,
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
    )
    if args.resume is not None:
        runner.load(str(args.resume.resolve()))
    elif args.warm_start is not None:
        runner.load_actor_warm_start(args.warm_start.resolve())
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


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if not args.reference_only and args.checkpoint is None:
        raise ValueError("policy evaluation requires --checkpoint")
    config = _config(args)
    if args.episodes > config.num_envs:
        raise ValueError(
            "paired evaluation requires episodes <= num-envs so each result "
            "comes from one unique initial world"
        )
    env = GpuGraspVecEnv(
        config,
        training=False,
        randomization_enabled=args.stress_domain_randomization,
    )
    if args.stress_domain_randomization:
        curriculum = config.domain_randomization
        env.common_step_counter = (
            curriculum.curriculum_warmup_steps + curriculum.curriculum_ramp_steps
        )
        env._reset(torch.arange(env.num_envs, device=env.device))
    actor = None
    if not args.reference_only:
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
    observations = env.get_observations()
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
    completed_max_lift: list[float] = []
    completed_max_grasp_quality: list[float] = []
    completed_world = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    evaluation_world = torch.arange(env.num_envs, device=env.device) < args.episodes
    outcomes = torch.full(
        (env.num_envs,), -1, dtype=torch.int8, device=env.device
    )
    while episodes < args.episodes:
        actions = (
            env.reference.current_action().clone()
            if actor is None
            else actor(observations, stochastic_output=False)
        )
        observations, _, dones, _ = env.step(actions)
        assert env.last_terms is not None
        terms = env.last_terms
        episode_max_lift.copy_(torch.maximum(episode_max_lift, terms.lift_height))
        episode_max_grasp_quality.copy_(
            torch.maximum(episode_max_grasp_quality, terms.grasp_quality)
        )
        episode_native_success.logical_or_(terms.native_success)
        episode_had_grasp.logical_or_(terms.is_grasp)
        finished = (
            dones & evaluation_world & ~completed_world
        ).nonzero(as_tuple=False).flatten()
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
    return {
        "mode": "reference_only" if args.reference_only else "ppo",
        "checkpoint": (
            None if args.checkpoint is None else str(args.checkpoint.resolve())
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
        "mean_max_lift": sum(completed_max_lift) / episodes,
        "max_lift": max(completed_max_lift),
        "mean_max_grasp_quality": sum(completed_max_grasp_quality) / episodes,
        "domain_randomization": bool(args.stress_domain_randomization),
        "reference_noise": bool(
            args.stress_domain_randomization
            and config.domain_randomization.reference_noise.enabled
        ),
        "dr_profile": args.dr_profile,
        "device": config.device,
        "success_world_ids": (
            (outcomes == 1).nonzero(as_tuple=False).flatten().cpu().tolist()
        ),
        "failed_world_ids": (
            (outcomes == 0).nonzero(as_tuple=False).flatten().cpu().tolist()
        ),
    }


@torch.inference_mode()
def _record(args: argparse.Namespace) -> dict[str, object]:
    for name in ("videos", "max_attempts", "width", "height", "fps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    config = _config(args)
    env = GpuGraspVecEnv(
        config,
        training=False,
        randomization_enabled=args.stress_domain_randomization,
        capture_terminal_qpos=True,
    )
    if args.stress_domain_randomization:
        curriculum = config.domain_randomization
        env.common_step_counter = (
            curriculum.curriculum_warmup_steps + curriculum.curriculum_ramp_steps
        )
        env._reset(torch.arange(env.num_envs, device=env.device))
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
    )


def main() -> None:
    args = _parser().parse_args()
    handlers = {"train": _train, "evaluate": _evaluate, "record": _record}
    result = handlers[args.command](args)
    print(json.dumps({"result": result}, indent=2))


if __name__ == "__main__":
    main()
