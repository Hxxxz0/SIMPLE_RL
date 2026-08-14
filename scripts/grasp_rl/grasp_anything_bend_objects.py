#!/usr/bin/env python3
"""Opt-in episode-11 bend route for low grasp-anything objects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOLMO_ROOT = Path(
    "/mnt/workspace/Jensen/.cache/molmo-spaces-resources/objects/thor/20251117"
)
BASE_ASSET = (
    REPO_ROOT
    / "outputs/grasp_rl/other/assets/mjlab_assets/xmove_bend_pick/episode11"
)
REFERENCE_EPISODE = 11
ROUTE_VERSION = "xmove_bend_ep11_v1"


@dataclass(frozen=True)
class BendObjectSpec:
    object_id: str
    source_relative: str
    scale: float
    grip_width_m: float
    mass_kg: float
    grasp_frame_position_m: tuple[float, float, float]
    maximum_grip_force_newtons: float


OBJECTS = {
    spec.object_id: spec
    for spec in (
        BendObjectSpec(
            object_id="Apple_1",
            source_relative="Kitchen Objects/Apple/Prefabs/Apple_1/Apple_1.xml",
            scale=0.6,
            grip_width_m=0.064,
            mass_kg=0.14,
            grasp_frame_position_m=(0.0, 0.0, 0.0),
            maximum_grip_force_newtons=60.0,
        ),
        BendObjectSpec(
            object_id="Bowl_1",
            source_relative="Kitchen Objects/Bowl/Prefabs/Bowl_1/Bowl_1.xml",
            scale=0.45,
            grip_width_m=0.085,
            mass_kg=0.18,
            grasp_frame_position_m=(0.0, 0.012, -0.024),
            maximum_grip_force_newtons=70.0,
        ),
        BendObjectSpec(
            object_id="Tomato_1",
            source_relative="Kitchen Objects/Tomato/Prefabs/Tomato_1/Tomato_1.xml",
            scale=0.7,
            grip_width_m=0.052,
            mass_kg=0.12,
            grasp_frame_position_m=(0.0, 0.0, 0.0),
            maximum_grip_force_newtons=55.0,
        ),
        BendObjectSpec(
            object_id="Potato_1",
            source_relative="Kitchen Objects/Potato/Prefabs/Potato_1/Potato_1.xml",
            scale=0.8,
            grip_width_m=0.053,
            mass_kg=0.15,
            grasp_frame_position_m=(0.0, 0.0, 0.0),
            maximum_grip_force_newtons=55.0,
        ),
    )
}


@dataclass(frozen=True)
class PoseProfile:
    target_jitter_xy_m: tuple[float, float]
    robot_base_jitter_xy_m: tuple[float, float] = (0.0, 0.0)
    target_yaw_jitter_rad: float = 0.0
    robot_base_yaw_jitter_rad: float = 0.0


PROFILES = {
    "fixed": PoseProfile((0.0, 0.0)),
    "target_xy_2p5mm": PoseProfile((0.0025, 0.0025)),
    "target_xy_5mm": PoseProfile((0.005, 0.005)),
    "target_xy_10mm": PoseProfile((0.010, 0.010)),
    "target_base_xy_2p5mm": PoseProfile((0.0025, 0.0025), (0.0025, 0.0025)),
}
TARGET_CENTER_XY_M = (0.0, -0.09)


def asset_path(object_id: str) -> Path:
    return (
        REPO_ROOT
        / "outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything"
        / f"{object_id}_object_reward_v4_xmove_bend_ep11"
    )


def strict_reference_path(object_id: str) -> Path:
    return (
        REPO_ROOT
        / "outputs/grasp_rl/other/references/grasp_anything"
        / f"{object_id}_xmove_bend_ep11_strict_v1"
    )


def staged_reference_path(object_id: str) -> Path:
    return (
        REPO_ROOT
        / "outputs/grasp_rl/other/references/grasp_anything"
        / f"{object_id}_xmove_bend_ep11_staged_native_v1"
    )


def run_root(object_id: str) -> Path:
    return (
        REPO_ROOT
        / "outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything"
        / object_id
        / ROUTE_VERSION
    )


def _reference_output_suffix(reference_processed: Path | None) -> str:
    if reference_processed is None:
        return ""
    return f"_reference-{reference_processed.name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_checkpoint(output: Path) -> Path:
    checkpoints = [
        path
        for path in output.glob("model_*.pt")
        if path.stem.removeprefix("model_").isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"training produced no checkpoint in {output}")
    return max(
        checkpoints,
        key=lambda path: int(path.stem.removeprefix("model_")),
    )


def derive(spec: BendObjectSpec, molmo_root: Path) -> dict[str, object]:
    source = molmo_root / spec.source_relative
    output = asset_path(spec.object_id)
    if (output / "manifest.json").is_file():
        return json.loads((output / "manifest.json").read_text())
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from simple.grasp_rl.grasp_anything import derive_grasp_anything_bundle

    return derive_grasp_anything_bundle(
        BASE_ASSET,
        source,
        output,
        object_id=spec.object_id,
        grip_width_m=spec.grip_width_m,
        mass_kg=spec.mass_kg,
        scale=spec.scale,
        upright_quaternion_wxyz=(2**-0.5, 2**-0.5, 0.0, 0.0),
        grasp_frame_position_m=spec.grasp_frame_position_m,
        maximum_grip_force_newtons=spec.maximum_grip_force_newtons,
        table_clearance_m=0.002,
    )


def verify(spec: BendObjectSpec, molmo_root: Path) -> dict[str, object]:
    source = molmo_root / spec.source_relative
    bundle = asset_path(spec.object_id)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from simple.grasp_rl.grasp_anything import validate_grasp_anything_bundle
    from simple.grasp_rl.mjlab_gpu.reference import validate_strict_reference_manifest

    validation = validate_grasp_anything_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    contract = manifest["object_contract"]
    if manifest["source_task"] != "xmove_bend_pick":
        raise ValueError("bend object asset must preserve xmove_bend_pick provenance")
    if contract["reference_task"] != "xmove_bend_pick":
        raise ValueError("bend object contract has the wrong reference task")
    if contract["source_mjcf_sha256"] != _sha256(source):
        raise ValueError("source object changed after bend asset derivation")
    strict = strict_reference_path(spec.object_id)
    strict_manifest = validate_strict_reference_manifest(
        json.loads((strict / "manifest.json").read_text()), REFERENCE_EPISODE
    )
    return {
        "object_id": spec.object_id,
        "asset": str(bundle),
        "strict_reference": str(strict),
        "staged_reference": str(staged_reference_path(spec.object_id)),
        "validation": validation,
        "strict_reference_manifest": strict_manifest,
    }


def environment_args(
    spec: BendObjectSpec,
    profile: str,
    seed: int,
    *,
    reference_processed: Path | None = None,
    dr_initial_strength: float = 1.0,
    dr_ramp_steps: int = 1,
    goal_potential_scale: float = 5.0,
    goal_potential_negative_clip: float = 0.25,
    success_bonus: float = 40.0,
) -> list[str]:
    pose = PROFILES[profile]
    reference = reference_processed or staged_reference_path(spec.object_id)
    return [
        "--task",
        "grasp_anything",
        "--asset-bundle",
        str(asset_path(spec.object_id)),
        "--reference-processed",
        str(reference),
        "--strict-reference-episode",
        str(REFERENCE_EPISODE),
        "--device",
        "cuda:0",
        "--seed",
        str(seed),
        "--max-reference-initial-position-offset",
        "0.12",
        "--reference-reward-weight",
        "0.005",
        "--max-reference-action-deviation",
        "0.7",
        "--grasp-anything-goal-potential-scale",
        str(goal_potential_scale),
        "--grasp-anything-goal-potential-negative-clip",
        str(goal_potential_negative_clip),
        "--grasp-anything-success-bonus",
        str(success_bonus),
        "--reference-target-x-arm-gains",
        "-10",
        "4",
        "--reference-target-y-arm-gains",
        "12",
        "0",
        "--reference-target-positive-y-arm-gains",
        "22",
        "0",
        "--reference-target-yaw-arm-gains",
        "0",
        "0",
        "--dr-initial-strength",
        str(dr_initial_strength),
        "--dr-warmup-steps",
        "0",
        "--dr-ramp-steps",
        str(dr_ramp_steps),
        "--target-position-jitter-xy",
        *(str(value) for value in pose.target_jitter_xy_m),
        "--target-position-offset-center-xy",
        *(str(value) for value in TARGET_CENTER_XY_M),
        "--target-yaw-jitter",
        str(pose.target_yaw_jitter_rad),
        "--destination-position-jitter-xy",
        "0",
        "0",
        "--destination-yaw-jitter",
        "0",
        "--distractor-position-jitter-xy",
        "0",
        "0",
        "--distractor-yaw-jitter",
        "0",
        "--robot-base-position-jitter-xy",
        *(str(value) for value in pose.robot_base_jitter_xy_m),
        "--robot-base-yaw-jitter",
        str(pose.robot_base_yaw_jitter_rad),
        "--target-mass-scale",
        "1",
        "1",
        "--friction-scale",
        "1",
        "1",
        "--joint-damping-scale",
        "1",
        "1",
        "--actuator-strength-scale",
        "1",
        "1",
        "--action-delay-max-steps",
        "0",
        "--reference-action-noise-std",
        "0",
        "--reference-position-noise-std",
        "0",
        "--reference-phase-noise-std",
        "0",
        "--reference-future-dropout-probability",
        "0",
    ]


def _gpu_cli(gpu: int) -> tuple[list[str], dict[str, str]]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MUJOCO_GL": "egl",
        }
    )
    return (
        [
            "uv",
            "run",
            "--project",
            "mjlab_gpu",
            "--no-sync",
            "python",
            "-m",
            "simple.grasp_rl.mjlab_gpu.cli",
        ],
        environment,
    )


def _run_gpu(arguments: list[str], *, gpu: int, log: Path | None = None) -> None:
    command, environment = _gpu_cli(gpu)
    process = subprocess.run(
        [*command, *arguments],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE if log else None,
        stderr=subprocess.STDOUT if log else None,
        check=False,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        assert process.stdout is not None
        log.write_text(process.stdout)
        print(process.stdout, end="")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def evaluate(
    spec: BendObjectSpec,
    *,
    profile: str,
    gpu: int,
    seed: int,
    episodes: int,
    checkpoint: Path | None,
    reference_processed: Path | None = None,
) -> Path:
    mode = "reference" if checkpoint is None else "ppo"
    reference_suffix = _reference_output_suffix(reference_processed)
    output = run_root(spec.object_id) / "acceptance" / (
        f"{mode}{reference_suffix}_{profile}_seed{seed}_{episodes}.log"
    )
    policy = ["--reference-only"] if checkpoint is None else ["--checkpoint", str(checkpoint)]
    _run_gpu(
        [
            "evaluate",
            *environment_args(
                spec, profile, seed, reference_processed=reference_processed
            ),
            *policy,
            "--num-envs",
            str(episodes),
            "--episodes",
            str(episodes),
            "--evaluation-dr-strength",
            "1",
            "--minimum-success-rate",
            "0",
            "--smoke",
        ],
        gpu=gpu,
        log=output,
    )
    return output


def train(
    spec: BendObjectSpec,
    *,
    profile: str,
    gpu: int,
    seed: int,
    num_envs: int,
    iterations: int,
    exploration_std: float,
    run_name: str,
    reference_processed: Path | None = None,
    dr_initial_strength: float = 0.1,
    dr_ramp_steps: int = 480,
    resume: Path | None = None,
    warm_start: Path | None = None,
    actor_anchor_checkpoint: Path | None = None,
    actor_anchor_weight: float = 0.0,
    goal_potential_scale: float = 5.0,
    goal_potential_negative_clip: float = 0.25,
    success_bonus: float = 40.0,
) -> Path:
    output = run_root(spec.object_id) / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing PPO run: {output}")
    reference = reference_processed or staged_reference_path(spec.object_id)
    if not reference.is_dir():
        raise FileNotFoundError(reference)
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")
    for checkpoint in (resume, warm_start, actor_anchor_checkpoint):
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if (actor_anchor_checkpoint is None) != (actor_anchor_weight == 0.0):
        raise ValueError(
            "actor anchor checkpoint and positive weight are required together"
        )
    if resume is not None:
        initialization = ["--resume", str(resume)]
    elif warm_start is not None:
        initialization = ["--warm-start", str(warm_start)]
    else:
        initialization = [
            "--plan-conditioned-actor",
            "--scratch-actor-output-scale",
            "0.000001",
            "--scratch-right-hand-correction",
            *("0" for _ in range(7)),
            "--scratch-right-arm-correction",
            *("0" for _ in range(7)),
        ]
    anchor = (
        []
        if actor_anchor_checkpoint is None
        else [
            "--actor-anchor-checkpoint",
            str(actor_anchor_checkpoint),
            "--actor-anchor-weight",
            str(actor_anchor_weight),
        ]
    )
    _run_gpu(
        [
            "train",
            *environment_args(
                spec,
                profile,
                seed,
                reference_processed=reference,
                dr_initial_strength=dr_initial_strength,
                dr_ramp_steps=dr_ramp_steps,
                goal_potential_scale=goal_potential_scale,
                goal_potential_negative_clip=goal_potential_negative_clip,
                success_bonus=success_bonus,
            ),
            "--num-envs",
            str(num_envs),
            "--iterations",
            str(iterations),
            "--output",
            str(output),
            *initialization,
            *anchor,
            "--exploration-std",
            str(exploration_std),
            "--learning-rate",
            "0.00005",
            "--schedule",
            "fixed",
            "--ppo-steps-per-env",
            "24",
            "--save-interval",
            "5",
        ],
        gpu=gpu,
    )
    return _latest_checkpoint(output)


def record(
    spec: BendObjectSpec,
    *,
    profile: str,
    gpu: int,
    seed: int,
    videos: int,
    camera_view: str,
    checkpoint: Path | None,
    reference_processed: Path | None = None,
) -> Path:
    mode = "reference" if checkpoint is None else "ppo"
    reference_suffix = _reference_output_suffix(reference_processed)
    output = run_root(spec.object_id) / "videos" / (
        f"{mode}{reference_suffix}_{profile}_{camera_view}_seed{seed}"
    )
    policy = ["--reference-only"] if checkpoint is None else ["--checkpoint", str(checkpoint)]
    _run_gpu(
        [
            "record",
            *environment_args(
                spec, profile, seed, reference_processed=reference_processed
            ),
            *policy,
            "--output-dir",
            str(output),
            "--num-envs",
            "512",
            "--videos",
            str(videos),
            "--max-attempts",
            "512",
            "--camera-view",
            camera_view,
            "--evaluation-dr-strength",
            "1",
            "--smoke",
        ],
        gpu=gpu,
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molmo-root",
        type=Path,
        default=Path(os.environ.get("MOLMO_OBJECT_ROOT", DEFAULT_MOLMO_ROOT)),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for name in ("derive", "verify"):
        child = commands.add_parser(name)
        child.add_argument("object", choices=OBJECTS)
    child = commands.add_parser("evaluate")
    child.add_argument("object", choices=OBJECTS)
    child.add_argument("--profile", choices=PROFILES, default="target_xy_2p5mm")
    child.add_argument("--gpu", type=int, default=0)
    child.add_argument("--seed", type=int, default=20260920)
    child.add_argument("--episodes", type=int, default=512)
    child.add_argument("--checkpoint", type=Path)
    child.add_argument("--reference-processed", type=Path)
    child = commands.add_parser("train")
    child.add_argument("object", choices=OBJECTS)
    child.add_argument("--profile", choices=PROFILES, default="target_xy_2p5mm")
    child.add_argument("--gpu", type=int, default=0)
    child.add_argument("--seed", type=int, default=20260920)
    child.add_argument("--num-envs", type=int, default=8192)
    child.add_argument("--iterations", type=int, default=40)
    child.add_argument("--exploration-std", type=float, default=0.01)
    child.add_argument("--run-name", required=True)
    child.add_argument("--reference-processed", type=Path)
    child.add_argument("--dr-initial-strength", type=float, default=0.1)
    child.add_argument("--dr-ramp-steps", type=int, default=480)
    child.add_argument("--resume", type=Path)
    child.add_argument("--warm-start", type=Path)
    child.add_argument("--actor-anchor-checkpoint", type=Path)
    child.add_argument("--actor-anchor-weight", type=float, default=0.0)
    child.add_argument("--goal-potential-scale", type=float, default=5.0)
    child.add_argument("--goal-potential-negative-clip", type=float, default=0.25)
    child.add_argument("--success-bonus", type=float, default=40.0)
    child = commands.add_parser("record")
    child.add_argument("object", choices=OBJECTS)
    child.add_argument("--profile", choices=PROFILES, default="target_xy_2p5mm")
    child.add_argument("--gpu", type=int, default=0)
    child.add_argument("--seed", type=int, default=20260920)
    child.add_argument("--videos", type=int, default=1)
    child.add_argument(
        "--camera-view", choices=("grasp_closeup", "full_robot"), default="full_robot"
    )
    child.add_argument("--checkpoint", type=Path)
    child.add_argument("--reference-processed", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        print(
            json.dumps(
                {
                    "route": ROUTE_VERSION,
                    "objects": {name: asdict(spec) for name, spec in OBJECTS.items()},
                    "profiles": {name: asdict(profile) for name, profile in PROFILES.items()},
                },
                indent=2,
            )
        )
        return 0
    spec = OBJECTS[args.object]
    if args.command == "derive":
        result = derive(spec, args.molmo_root)
    elif args.command == "verify":
        result = verify(spec, args.molmo_root)
    elif args.command == "evaluate":
        verify(spec, args.molmo_root)
        result = {
            "output": str(
                evaluate(
                    spec,
                    profile=args.profile,
                    gpu=args.gpu,
                    seed=args.seed,
                    episodes=args.episodes,
                    checkpoint=args.checkpoint,
                    reference_processed=args.reference_processed,
                )
            )
        }
    elif args.command == "train":
        verify(spec, args.molmo_root)
        result = {
            "output": str(
                train(
                    spec,
                    profile=args.profile,
                    gpu=args.gpu,
                    seed=args.seed,
                    num_envs=args.num_envs,
                    iterations=args.iterations,
                    exploration_std=args.exploration_std,
                    run_name=args.run_name,
                    reference_processed=args.reference_processed,
                    dr_initial_strength=args.dr_initial_strength,
                    dr_ramp_steps=args.dr_ramp_steps,
                    resume=args.resume,
                    warm_start=args.warm_start,
                    actor_anchor_checkpoint=args.actor_anchor_checkpoint,
                    actor_anchor_weight=args.actor_anchor_weight,
                    goal_potential_scale=args.goal_potential_scale,
                    goal_potential_negative_clip=args.goal_potential_negative_clip,
                    success_bonus=args.success_bonus,
                )
            )
        }
    else:
        verify(spec, args.molmo_root)
        result = {
            "output": str(
                record(
                    spec,
                    profile=args.profile,
                    gpu=args.gpu,
                    seed=args.seed,
                    videos=args.videos,
                    camera_view=args.camera_view,
                    checkpoint=args.checkpoint,
                    reference_processed=args.reference_processed,
                )
            )
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
