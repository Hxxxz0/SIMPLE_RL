"""Command-line entry point for the complete grasp-RL pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple.grasp_rl.bc import (
    BcTrainConfig,
    collect_dagger_dataset,
    prepare_bc_dataset,
    train_bc_actor,
)
from simple.grasp_rl.collect import collect_policy_dataset
from simple.grasp_rl.diffusion import DiffusionTrainConfig, train_diffusion
from simple.grasp_rl.evaluate import evaluate_policy
from simple.grasp_rl.hard_targets import mine_hard_targets
from simple.grasp_rl.mjlab_assets import (
    export_mjlab_render_scene,
    export_mjlab_scene,
    validate_asset_bundle,
)
from simple.grasp_rl.policy import build_knn_actor_checkpoint
from simple.grasp_rl.paired import compare_paired_evaluations
from simple.grasp_rl.rewards import (
    DEFAULT_TASK_REWARD_PROFILE,
    REWARD_VARIANTS,
    TASK_REWARD_PROFILES,
)
from simple.grasp_rl.render import render_saved_trajectory
from simple.grasp_rl.prepare import prepare_dataset
from simple.grasp_rl.data_v2 import prepare_v2_dataset, write_repaired_actions
from simple.grasp_rl.audit_v2 import audit_v2_reward
from simple.grasp_rl.reward_audit import AUDIT_SCENARIOS, audit_reward
from simple.grasp_rl.train import PpoTrainConfig, train_ppo
from simple.grasp_rl.task_spec import (
    DEFAULT_TASK,
    get_task_spec,
    task_from_manifest,
    task_names,
    TaskSpecV2,
)

DEFAULT_DATASET = Path("data/simple/G1WholebodyTabletopGraspMP-v0")
DEFAULT_PROCESSED = Path("data/grasp_rl/G1WholebodyTabletopGraspMP-v0")
DEFAULT_OUTPUT = Path("outputs/grasp_rl")


def _parse_target_position_jitter(value: str) -> tuple[float, float] | None:
    if value.strip().lower() == "task":
        return None
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if len(result) != 2:
        raise ValueError(
            "target position jitter must be 'task' or two comma-separated values"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grasp-rl")
    commands = parser.add_subparsers(dest="command", required=True)

    list_tasks = commands.add_parser("list-tasks")
    list_tasks.set_defaults(task=DEFAULT_TASK)

    export_mjlab = commands.add_parser("export-mjlab-assets")
    export_mjlab.add_argument("--task", default=DEFAULT_TASK)
    export_mjlab.add_argument("--output", type=Path, required=True)
    export_mjlab.add_argument("--seed", type=int, default=42)
    export_mjlab.add_argument("--target-object")
    export_mjlab.add_argument("--warmup-steps", type=int, default=60)
    export_mjlab.add_argument(
        "--base-episode",
        type=int,
        help="Freeze the exact recorded scene used by this reference episode",
    )

    validate_mjlab = commands.add_parser("validate-mjlab-assets")
    validate_mjlab.set_defaults(task=DEFAULT_TASK)
    validate_mjlab.add_argument("--bundle", type=Path, required=True)

    export_mjlab_render = commands.add_parser("export-mjlab-render-assets")
    export_mjlab_render.set_defaults(task=DEFAULT_TASK)
    export_mjlab_render.add_argument("--bundle", type=Path, required=True)

    compare_paired = commands.add_parser("compare-paired")
    compare_paired.set_defaults(task=DEFAULT_TASK)
    compare_paired.add_argument("--policy-evaluation", type=Path, required=True)
    compare_paired.add_argument("--reference-evaluation", type=Path, required=True)
    compare_paired.add_argument("--output", type=Path, required=True)

    mine_hard = commands.add_parser("mine-hard-targets")
    mine_hard.set_defaults(task=DEFAULT_TASK)
    mine_hard.add_argument("--evaluation-summary", type=Path, required=True)
    mine_hard.add_argument("--output", type=Path, required=True)
    mine_hard.add_argument("--limit", type=int, default=256)

    def task_argument(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--task",
            default=DEFAULT_TASK,
            help=f"Task adapter ({', '.join(task_names())})",
        )

    prepare = commands.add_parser("prepare")
    task_argument(prepare)
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--output", type=Path, default=DEFAULT_PROCESSED)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--episodes", type=int)
    prepare.add_argument("--warmup-steps", type=int, default=60)

    repair = commands.add_parser("repair-data")
    task_argument(repair)
    repair.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    repair.add_argument("--output", type=Path, required=True)

    prepare_bc = commands.add_parser("prepare-bc")
    task_argument(prepare_bc)
    prepare_bc.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare_bc.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    prepare_bc.add_argument("--workers", type=int, default=7)

    audit = commands.add_parser("audit-reward")
    task_argument(audit)
    audit.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    audit.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "reward_audit")
    audit.add_argument("--episodes", type=int, default=100)
    audit.add_argument("--episode-offset", type=int, default=0)
    audit.add_argument("--workers", type=int, default=7)
    audit.add_argument(
        "--profiles",
        default=",".join(TASK_REWARD_PROFILES),
    )
    audit.add_argument(
        "--scenarios",
        default=",".join(AUDIT_SCENARIOS),
    )
    audit.add_argument("--task-reward-weight", type=float, default=0.02)

    pretrain_actor = commands.add_parser("pretrain-actor")
    task_argument(pretrain_actor)
    pretrain_actor.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    pretrain_actor.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT / "bc_actor"
    )
    pretrain_actor.add_argument("--epochs", type=int, default=500)
    pretrain_actor.add_argument("--batch-size", type=int, default=1024)
    pretrain_actor.add_argument("--learning-rate", type=float, default=3e-4)
    pretrain_actor.add_argument("--device", default="cuda:0")
    pretrain_actor.add_argument("--initialize", type=Path)
    pretrain_actor.add_argument(
        "--sources",
        default="bc",
        help="Comma-separated replay directories below --processed",
    )
    pretrain_actor.add_argument(
        "--reference-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition the full-command actor on ten GRAIL-style future references",
    )
    pretrain_actor.add_argument(
        "--plan-conditioned",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict a residual around the first complete plan command",
    )
    pretrain_actor.add_argument(
        "--recurrent", action=argparse.BooleanOptionalAction, default=False
    )
    pretrain_actor.add_argument("--sequence-batch-size", type=int, default=8)
    pretrain_actor.add_argument("--rnn-hidden-dim", type=int, default=256)
    pretrain_actor.add_argument("--right-hand-weight", type=float, default=3.0)
    pretrain_actor.add_argument("--right-arm-weight", type=float, default=2.0)

    build_knn = commands.add_parser("build-knn")
    task_argument(build_knn)
    build_knn.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    build_knn.add_argument("--output", type=Path, required=True)
    build_knn.add_argument("--sources", default="bc")
    build_knn.add_argument("--splits", default="train,val,test")

    dagger = commands.add_parser("collect-dagger")
    task_argument(dagger)
    dagger.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    dagger.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    dagger.add_argument("--checkpoint", type=Path, required=True)
    dagger.add_argument("--teacher-checkpoint", type=Path)
    dagger.add_argument("--teacher-rollout-blend", type=float, default=0.0)
    dagger.add_argument("--teacher-rollout-probability", type=float, default=0.0)
    dagger.add_argument("--round", type=int, required=True)
    dagger.add_argument("--workers", type=int, default=7)
    dagger.add_argument("--initialization-prefix", type=int)

    pretrain = commands.add_parser("pretrain")
    task_argument(pretrain)
    pretrain.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    pretrain.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "diffusion")
    pretrain.add_argument("--epochs", type=int, default=5000)
    pretrain.add_argument("--min-epochs", type=int, default=1000)
    pretrain.add_argument("--device", default="cuda:0")
    pretrain.add_argument("--resume", type=Path)

    train = commands.add_parser("train")
    task_argument(train)
    train.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--diffusion", type=Path)
    train.add_argument(
        "--variant",
        choices=REWARD_VARIANTS,
        default="task_only",
    )
    train.add_argument("--num-envs", type=int, default=56)
    train.add_argument("--iterations", type=int, default=1500)
    train.add_argument("--save-interval", type=int, default=500)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--ws", type=float, default=4.0)
    train.add_argument("--task-reward-weight", type=float, default=0.02)
    train.add_argument("--smp-reward-weight", type=float, default=0.01)
    train.add_argument(
        "--task-reward-profile",
        choices=TASK_REWARD_PROFILES,
        default=DEFAULT_TASK_REWARD_PROFILE,
    )
    checkpoint_mode = train.add_mutually_exclusive_group()
    checkpoint_mode.add_argument("--resume", type=Path)
    checkpoint_mode.add_argument("--warm-start", type=Path)
    checkpoint_mode.add_argument("--actor-warm-start", type=Path)
    train.add_argument("--worker-devices", default="1,2,3,4,5,6,7")
    train.add_argument(
        "--curriculum-config",
        type=Path,
        help="Versioned JSON curriculum; hashes are embedded in every checkpoint",
    )
    train.add_argument(
        "--hard-target-manifest",
        type=Path,
        help="Train/val failure targets; final-test manifests are rejected",
    )
    train.add_argument("--rsi-dataset", type=Path)
    train.add_argument(
        "--rsi-processed",
        type=Path,
        help="Replay-gated prepared actions for RSI without enabling reference input",
    )
    train.add_argument("--rsi-prefix", default="75,115")
    train.add_argument(
        "--rsi-phase",
        help="Optional normalized low,high demonstration phase; overrides --rsi-prefix",
    )
    train.add_argument(
        "--rsi-stage",
        choices=("pregrasp", "grasp_to_lift", "lift"),
        help="Detect task stage from replay; overrides phase and absolute prefix",
    )
    train.add_argument(
        "--rsi-episodes",
        help="Optional comma-separated episode indices for hard-scene mining",
    )
    train.add_argument("--rsi-probability", type=float, default=0.0)
    train.add_argument("--rsi-scene-hold-episodes", type=int, default=32)
    train.add_argument(
        "--rsi-randomize-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move the object while retaining the recorded robot/reference plan",
    )
    train.add_argument(
        "--target-position-jitter-xy",
        default="0.025,0.03",
        help="Per-axis target position half-ranges in metres",
    )
    train.add_argument(
        "--target-position-offset-center-xy",
        default="0,0",
        help=(
            "Centre of the target-offset distribution in metres; use a "
            "non-zero x/y centre for hard-failure mining"
        ),
    )
    train.add_argument("--target-yaw-jitter", type=float, default=0.15)
    train.add_argument("--action-std", type=float, default=0.30)
    train.add_argument(
        "--manipulation-action-std",
        type=float,
        help="Optional lower exploration std for right arm and right hand",
    )
    train.add_argument(
        "--observation-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--actor-lr-scale", type=float, default=1.0)
    train.add_argument(
        "--learning-schedule", choices=("adaptive", "fixed"), default="adaptive"
    )
    train.add_argument("--num-steps-per-env", type=int, default=24)
    train.add_argument("--ppo-epochs", type=int, default=5)
    train.add_argument("--num-mini-batches", type=int, default=4)
    train.add_argument("--exploration-hold-steps", type=int, default=1)
    train.add_argument(
        "--freeze-actor-normalizer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train.add_argument("--bc-anchor-weight", type=float, default=0.0)
    train.add_argument("--bc-anchor-sources", default="bc")
    train.add_argument("--bc-anchor-batch-size", type=int, default=1024)
    train.add_argument(
        "--bc-anchor-manipulation-weight",
        type=float,
        default=10.0,
        help="Relative BC weight for the right-arm and right-hand action dimensions",
    )
    train.add_argument("--teacher-anchor-checkpoint", type=Path)
    train.add_argument("--teacher-anchor-weight", type=float, default=0.0)
    train.add_argument(
        "--reference-processed",
        type=Path,
        help="Processed replay root used for future-reference input and reward",
    )
    train.add_argument("--reference-source", default="bc")
    train.add_argument("--reference-splits", default="train,val,test")
    train.add_argument("--reference-reward-weight", type=float, default=0.0)
    train.add_argument(
        "--reference-rank-max",
        type=int,
        default=0,
        help="Uniformly sample reference retrieval ranks from zero through this value",
    )
    train.add_argument(
        "--reference-base-episode-probability",
        type=float,
        default=0.0,
        help="Probability of retaining the base scene plan after moving its target",
    )
    train.add_argument(
        "--reference-action-noise-std",
        type=float,
        default=0.0,
        help="Correlated normalized right-arm/hand bias applied only to actor plans",
    )
    train.add_argument(
        "--reference-action-noise-hold-steps",
        type=int,
        default=25,
        help="Control steps to retain each sampled reference command bias",
    )
    train.add_argument(
        "--recurrent-actor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a GRU actor that still emits the complete 36-D command",
    )
    train.add_argument(
        "--plan-conditioned-actor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize the complete 36-D output from the generated plan command",
    )
    train.add_argument("--rnn-hidden-dim", type=int, default=256)
    train.add_argument("--max-grad-norm", type=float, default=0.1)
    train.add_argument(
        "--reward-audit",
        type=Path,
        help="Passing expert/counterfactual audit required for new tasks",
    )

    evaluate = commands.add_parser("evaluate")
    task_argument(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    evaluate.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--diffusion", type=Path)
    evaluate.add_argument(
        "--variant",
        choices=REWARD_VARIANTS,
        default="task_only",
    )
    evaluate.add_argument("--episodes", type=int, default=100)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--seed", type=int, default=1234)
    evaluate.add_argument(
        "--final-protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enforce the locked 200-target unseen-test protocol",
    )
    evaluate.add_argument("--task-reward-weight", type=float, default=0.02)
    evaluate.add_argument("--smp-reward-weight", type=float, default=0.01)
    evaluate.add_argument(
        "--task-reward-profile",
        choices=TASK_REWARD_PROFILES,
        default=DEFAULT_TASK_REWARD_PROFILE,
    )
    evaluate.add_argument("--episode-offset", type=int, default=0)
    evaluate.add_argument(
        "--evaluation-split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Restrict evaluation to replay-gated processed episode IDs",
    )
    evaluate.add_argument("--reference-source", default="bc")
    evaluate.add_argument("--reference-splits", default="train,val,test")
    evaluate.add_argument("--reference-reward-weight", type=float, default=0.0)
    evaluate.add_argument(
        "--reference-rank",
        type=int,
        default=0,
        help="Use the Nth nearest complete plan for randomized-target evaluation",
    )
    evaluate.add_argument(
        "--reference-base-episode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the current dataset scene's plan after moving its target",
    )
    evaluate.add_argument(
        "--fixed-reference-episode",
        type=int,
        help="Use one exact reference plan for every evaluation rollout",
    )
    evaluate.add_argument(
        "--fixed-base-episode",
        type=int,
        help="Repeat one exact robot/scene state for all randomized targets",
    )
    evaluate.add_argument("--reach-extension-threshold", type=float)
    evaluate.add_argument(
        "--reach-extension-velocity",
        type=float,
        default=0.0,
        help="Normalized forward tracker velocity while target is out of reach",
    )
    evaluate.add_argument(
        "--randomize-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate with the target moved relative to each recorded scene",
    )
    evaluate.add_argument(
        "--target-position-jitter-xy",
        default="0.025,0.03",
    )
    evaluate.add_argument(
        "--target-position-offset-center-xy",
        default="0,0",
        help="Centre of the randomized target-offset distribution in metres",
    )
    evaluate.add_argument("--target-yaw-jitter", type=float, default=0.15)
    evaluate.add_argument(
        "--target-position-xy",
        help="Set an exact v2 target world XY, for explicit out-of-distribution tests",
    )
    evaluate.add_argument(
        "--robot-position-xy",
        help="Set an exact v2 robot world XY before stabilization",
    )
    evaluate.add_argument(
        "--reference-action-override",
        choices=("none", "right_hand", "right_arm_hand", "all"),
        default="none",
        help="Diagnostic: replace selected outputs with the current reference command",
    )
    evaluate_initialization = evaluate.add_mutually_exclusive_group()
    evaluate_initialization.add_argument("--initialization-prefix", type=int)
    evaluate_initialization.add_argument("--initialization-phase", type=float)

    render = commands.add_parser("render")
    task_argument(render)
    render.add_argument("--trajectory", type=Path, required=True)
    render.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    render.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument(
        "--camera",
        default="auto",
        help="Camera name, or auto for a task-compatible left camera",
    )
    render.add_argument("--fps", type=float, default=50.0)
    render.add_argument("--checkpoint", type=Path)
    render.add_argument("--device", default="cuda:0")
    render.add_argument("--camera-fovy", type=float, default=40.0)
    render.add_argument("--expert", action="store_true")
    render.add_argument("--context-start", type=int)
    render.add_argument(
        "--robot-position-xy",
        help="Restore an exact v2 robot world XY for an OOD rollout",
    )

    collect = commands.add_parser("collect-policy")
    task_argument(collect)
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    collect.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--successes", type=int, required=True)
    collect.add_argument("--max-attempts", type=int)
    collect.add_argument("--scene-hold-attempts", type=int, default=16)
    collect.add_argument("--target-position-jitter-xy", default="task")
    collect.add_argument("--target-yaw-jitter", type=float, default=0.15)
    collect.add_argument("--reference-ranks", default="0,1")
    collect.add_argument(
        "--base-reference-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    collect.add_argument(
        "--base-reference-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try the current scene's complete plan before retrieved ranks",
    )
    collect.add_argument("--seed", type=int, default=20260729)
    collect.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = _parser().parse_args()
    task_spec = get_task_spec(args.task)
    # Preserve old commands while making `--task bend_pick` useful without
    # repeating the conventional dataset/processed paths on every command.
    if task_spec.name != DEFAULT_TASK:
        if hasattr(args, "dataset") and args.dataset == DEFAULT_DATASET:
            args.dataset = task_spec.dataset_path()
        if hasattr(args, "processed") and args.processed == DEFAULT_PROCESSED:
            args.processed = task_spec.processed_path()
        if args.command == "prepare" and args.output == DEFAULT_PROCESSED:
            args.output = task_spec.processed_path()
        task_output = DEFAULT_OUTPUT / task_spec.name
        conventional_outputs = {
            "audit-reward": (
                DEFAULT_OUTPUT / "reward_audit",
                task_output / "reward_audit",
            ),
            "pretrain-actor": (DEFAULT_OUTPUT / "bc_actor", task_output / "bc_actor"),
            "pretrain": (DEFAULT_OUTPUT / "diffusion", task_output / "diffusion"),
        }
        if args.command in conventional_outputs:
            old_default, task_default = conventional_outputs[args.command]
            if args.output == old_default:
                args.output = task_default
    if hasattr(args, "processed") and (args.processed / "manifest.json").exists():
        processed_task = task_from_manifest(args.processed)
        if processed_task.name != task_spec.name:
            raise ValueError(
                f"Processed data is for {processed_task.name}, not {task_spec.name}"
            )
    if args.command == "list-tasks":
        result = {name: get_task_spec(name).metadata() for name in task_names()}
    elif args.command == "export-mjlab-assets":
        result = export_mjlab_scene(
            task_spec.name,
            args.output,
            seed=args.seed,
            target_object=args.target_object,
            warmup_steps=args.warmup_steps,
            base_episode=args.base_episode,
        )
    elif args.command == "validate-mjlab-assets":
        result = validate_asset_bundle(args.bundle)
    elif args.command == "export-mjlab-render-assets":
        result = export_mjlab_render_scene(args.bundle)
    elif args.command == "compare-paired":
        result = compare_paired_evaluations(
            args.policy_evaluation,
            args.reference_evaluation,
            args.output,
        )
    elif args.command == "mine-hard-targets":
        result = mine_hard_targets(
            args.evaluation_summary,
            args.output,
            limit=args.limit,
        )
    elif args.command == "prepare":
        result = (
            prepare_v2_dataset(
                args.dataset,
                args.output,
                num_workers=args.workers,
                task=task_spec,
                episodes=args.episodes,
                warmup_steps=args.warmup_steps,
            )
            if isinstance(task_spec, TaskSpecV2)
            else prepare_dataset(
                args.dataset,
                args.output,
                num_workers=args.workers,
                task=task_spec,
            )
        )
    elif args.command == "repair-data":
        result = write_repaired_actions(args.dataset, args.output, task_spec)
    elif args.command == "prepare-bc":
        if isinstance(task_spec, TaskSpecV2):
            raise ValueError(
                "v2 prepare already writes the BC replay; use the prepare command"
            )
        result = prepare_bc_dataset(
            args.dataset,
            args.processed,
            num_workers=args.workers,
            task=task_spec,
        )
    elif args.command == "audit-reward":
        scenarios = tuple(
            value.strip() for value in args.scenarios.split(",") if value.strip()
        )
        result = (
            audit_v2_reward(
                args.dataset,
                args.processed,
                args.output,
                episodes=args.episodes,
                episode_offset=args.episode_offset,
                scenarios=scenarios,
                task=task_spec,
                workers=args.workers,
            )
            if isinstance(task_spec, TaskSpecV2)
            else audit_reward(
                args.dataset,
                args.processed / "action_transform.npz",
                args.output,
                episodes=args.episodes,
                episode_offset=args.episode_offset,
                workers=args.workers,
                profiles=tuple(
                    value.strip() for value in args.profiles.split(",") if value.strip()
                ),
                scenarios=scenarios,
                task_weight=args.task_reward_weight,
                task=task_spec,
            )
        )
    elif args.command == "pretrain-actor":
        result = train_bc_actor(
            args.processed,
            args.output,
            BcTrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                initialize_checkpoint=(
                    str(args.initialize) if args.initialize else None
                ),
                sources=tuple(
                    value.strip() for value in args.sources.split(",") if value.strip()
                ),
                reference_conditioning=args.reference_conditioning,
                plan_conditioned=args.plan_conditioned,
                recurrent=args.recurrent,
                sequence_batch_size=args.sequence_batch_size,
                rnn_hidden_dim=args.rnn_hidden_dim,
                right_hand_weight=args.right_hand_weight,
                right_arm_weight=args.right_arm_weight,
            ),
        )
    elif args.command == "build-knn":
        result = build_knn_actor_checkpoint(
            args.processed,
            args.output,
            sources=tuple(
                value.strip() for value in args.sources.split(",") if value.strip()
            ),
            splits=tuple(
                value.strip() for value in args.splits.split(",") if value.strip()
            ),
        )
    elif args.command == "collect-dagger":
        result = collect_dagger_dataset(
            args.dataset,
            args.processed,
            args.checkpoint,
            args.round,
            num_workers=args.workers,
            initialization_prefix=args.initialization_prefix,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_rollout_blend=args.teacher_rollout_blend,
            teacher_rollout_probability=args.teacher_rollout_probability,
            task=task_spec,
        )
    elif args.command == "pretrain":
        result = train_diffusion(
            args.processed,
            args.output,
            DiffusionTrainConfig(
                max_epochs=args.epochs,
                min_epochs=min(args.min_epochs, args.epochs),
                device=args.device,
                resume=str(args.resume) if args.resume else None,
            ),
        )
    elif args.command == "train":
        result = train_ppo(
            args.processed / "action_transform.npz",
            args.output,
            args.diffusion,
            PpoTrainConfig(
                task=task_spec.name,
                reward_audit=(str(args.reward_audit) if args.reward_audit else None),
                num_envs=args.num_envs,
                iterations=args.iterations,
                seed=args.seed,
                device=args.device,
                save_interval=args.save_interval,
                reward_variant=args.variant,
                ws=args.ws,
                task_reward_weight=args.task_reward_weight,
                smp_reward_weight=args.smp_reward_weight,
                task_reward_profile=args.task_reward_profile,
                resume=str(args.resume) if args.resume else None,
                warm_start=str(args.warm_start) if args.warm_start else None,
                actor_warm_start=(
                    str(args.actor_warm_start) if args.actor_warm_start else None
                ),
                worker_devices=tuple(
                    int(value)
                    for value in args.worker_devices.split(",")
                    if value.strip()
                ),
                rsi_dataset=str(args.rsi_dataset) if args.rsi_dataset else None,
                rsi_processed=(str(args.rsi_processed) if args.rsi_processed else None),
                rsi_prefix=tuple(int(value) for value in args.rsi_prefix.split(",")),
                rsi_phase=(
                    tuple(float(value) for value in args.rsi_phase.split(","))
                    if args.rsi_phase
                    else None
                ),
                rsi_stage=args.rsi_stage,
                rsi_episodes=(
                    tuple(int(value) for value in args.rsi_episodes.split(","))
                    if args.rsi_episodes
                    else None
                ),
                rsi_probability=args.rsi_probability,
                rsi_scene_hold_episodes=args.rsi_scene_hold_episodes,
                rsi_randomize_target=args.rsi_randomize_target,
                target_position_jitter_xy=_parse_target_position_jitter(
                    args.target_position_jitter_xy
                ),
                target_position_offset_center_xy=(
                    _parse_target_position_jitter(args.target_position_offset_center_xy)
                    or (0.0, 0.0)
                ),
                target_yaw_jitter=args.target_yaw_jitter,
                action_std=args.action_std,
                manipulation_action_std=args.manipulation_action_std,
                observation_noise=args.observation_noise,
                learning_rate=args.learning_rate,
                num_steps_per_env=args.num_steps_per_env,
                num_learning_epochs=args.ppo_epochs,
                num_mini_batches=args.num_mini_batches,
                exploration_hold_steps=args.exploration_hold_steps,
                actor_learning_rate_scale=args.actor_lr_scale,
                learning_schedule=args.learning_schedule,
                freeze_actor_normalizer=args.freeze_actor_normalizer,
                bc_anchor_weight=args.bc_anchor_weight,
                bc_anchor_processed=(
                    str(args.processed) if args.bc_anchor_weight > 0.0 else None
                ),
                bc_anchor_sources=tuple(
                    value.strip()
                    for value in args.bc_anchor_sources.split(",")
                    if value.strip()
                ),
                bc_anchor_batch_size=args.bc_anchor_batch_size,
                bc_anchor_manipulation_weight=(args.bc_anchor_manipulation_weight),
                teacher_anchor_checkpoint=(
                    str(args.teacher_anchor_checkpoint)
                    if args.teacher_anchor_checkpoint
                    else None
                ),
                teacher_anchor_weight=args.teacher_anchor_weight,
                reference_processed=(
                    str(args.reference_processed) if args.reference_processed else None
                ),
                reference_source=args.reference_source,
                reference_splits=tuple(
                    value.strip()
                    for value in args.reference_splits.split(",")
                    if value.strip()
                ),
                reference_reward_weight=args.reference_reward_weight,
                reference_rank_max=args.reference_rank_max,
                reference_base_episode_probability=(
                    args.reference_base_episode_probability
                ),
                reference_action_noise_std=args.reference_action_noise_std,
                reference_action_noise_hold_steps=(
                    args.reference_action_noise_hold_steps
                ),
                recurrent_actor=args.recurrent_actor,
                plan_conditioned_actor=args.plan_conditioned_actor,
                rnn_hidden_dim=args.rnn_hidden_dim,
                max_grad_norm=args.max_grad_norm,
                curriculum_config=(
                    str(args.curriculum_config) if args.curriculum_config else None
                ),
                hard_target_manifest=(
                    str(args.hard_target_manifest)
                    if args.hard_target_manifest
                    else None
                ),
            ),
        )
    elif args.command == "evaluate":
        result = evaluate_policy(
            args.checkpoint,
            args.processed / "action_transform.npz",
            args.dataset,
            args.output,
            diffusion_checkpoint=args.diffusion,
            num_episodes=args.episodes,
            reward_variant=args.variant,
            initialization_prefix=args.initialization_prefix,
            initialization_phase=args.initialization_phase,
            episode_offset=args.episode_offset,
            evaluation_split=args.evaluation_split,
            task_reward_weight=args.task_reward_weight,
            smp_reward_weight=args.smp_reward_weight,
            task_reward_profile=args.task_reward_profile,
            device=args.device,
            reference_processed=args.processed,
            reference_source=args.reference_source,
            reference_splits=tuple(
                value.strip()
                for value in args.reference_splits.split(",")
                if value.strip()
            ),
            reference_reward_weight=args.reference_reward_weight,
            reference_action_override=args.reference_action_override,
            reference_rank=args.reference_rank,
            reference_base_episode=args.reference_base_episode,
            fixed_reference_episode=args.fixed_reference_episode,
            fixed_base_episode=args.fixed_base_episode,
            reach_extension_threshold=args.reach_extension_threshold,
            reach_extension_velocity=args.reach_extension_velocity,
            randomize_target=args.randomize_target,
            target_position_jitter_xy=_parse_target_position_jitter(
                args.target_position_jitter_xy
            ),
            target_position_offset_center_xy=(
                _parse_target_position_jitter(args.target_position_offset_center_xy)
                or (0.0, 0.0)
            ),
            target_yaw_jitter=args.target_yaw_jitter,
            target_position_xy=(
                _parse_target_position_jitter(args.target_position_xy)
                if args.target_position_xy is not None
                else None
            ),
            robot_position_xy=(
                _parse_target_position_jitter(args.robot_position_xy)
                if args.robot_position_xy is not None
                else None
            ),
            seed=args.seed,
            final_protocol=args.final_protocol,
            task=task_spec,
        )
    elif args.command == "render":
        result = render_saved_trajectory(
            args.trajectory,
            args.processed / "action_transform.npz",
            args.dataset,
            args.output,
            camera=args.camera,
            fps=args.fps,
            checkpoint_path=args.checkpoint,
            device=args.device,
            camera_fovy=args.camera_fovy,
            expert=args.expert,
            context_start=args.context_start,
            robot_position_xy=(
                _parse_target_position_jitter(args.robot_position_xy)
                if args.robot_position_xy is not None
                else None
            ),
            task=task_spec,
        )
    elif args.command == "collect-policy":
        result = collect_policy_dataset(
            args.checkpoint,
            args.processed / "action_transform.npz",
            args.dataset,
            args.processed,
            args.output,
            args.successes,
            max_attempts=args.max_attempts,
            scene_hold_attempts=args.scene_hold_attempts,
            target_position_jitter_xy=_parse_target_position_jitter(
                args.target_position_jitter_xy
            ),
            target_yaw_jitter=args.target_yaw_jitter,
            reference_ranks=tuple(
                int(value) for value in args.reference_ranks.split(",") if value.strip()
            ),
            base_reference_fallback=args.base_reference_fallback,
            base_reference_first=args.base_reference_first,
            seed=args.seed,
            device=args.device,
            task=task_spec,
        )
    else:
        raise AssertionError(args.command)
    if isinstance(result, dict):
        # Full per-episode/audit details are already persisted by each command.
        # Keep stdout valid and concise so long evaluations do not dump hundreds
        # of kilobytes of a Python-dict string into experiment logs.
        omitted = {"results", "episodes_detail", "task_metadata"}
        result_payload = {
            key: value for key, value in result.items() if key not in omitted
        }
        if hasattr(args, "output"):
            result_payload["output"] = str(args.output)
    else:
        result_payload = str(result)
    print(json.dumps({"result": result_payload}, indent=2, default=str))


if __name__ == "__main__":
    main()
