#!/usr/bin/env python3
"""Isolated per-object grasp-anything PPO catalog and runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOLMO_ROOT = Path(
    "/mnt/workspace/Jensen/.cache/molmo-spaces-resources/objects/thor/20251117"
)
BASE_ASSET = REPO_ROOT / "outputs/grasp_rl/other/assets/mjlab_assets/xmove_pick/episode82"
REFERENCE = (
    REPO_ROOT
    / "data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2_single_ref_ep82_shared_transform"
)
APPLE_STAGED_REFERENCE = (
    REPO_ROOT
    / "outputs/grasp_rl/other/references/grasp_anything/Apple_1_staged_native_success_v2"
)
BOWL_STAGED_REFERENCE = (
    REPO_ROOT
    / "outputs/grasp_rl/other/references/grasp_anything/Bowl_1_staged_native_success_v1"
)


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    category: str
    source_relative: str
    asset_version: str
    policy_version: str
    scale: float
    grip_width_m: float
    mass_kg: float
    grasp_frame_position_m: tuple[float, float, float]
    maximum_grip_force_newtons: float
    scratch_right_hand_correction: tuple[float, ...]
    scratch_right_arm_correction: tuple[float, ...]
    exploration_std: float = 0.025

    def validate(self) -> None:
        if not self.object_id or "/" in self.object_id:
            raise ValueError("object_id must be a portable name")
        if not self.source_relative.endswith(f"/{self.object_id}.xml"):
            raise ValueError(f"source path does not match {self.object_id}")
        if not 0.0 < self.scale <= 2.0:
            raise ValueError(f"invalid scale for {self.object_id}")
        if not 0.035 <= self.grip_width_m <= 0.09:
            raise ValueError(f"invalid grip width for {self.object_id}")
        if not 0.05 <= self.mass_kg <= 1.5:
            raise ValueError(f"invalid mass for {self.object_id}")
        if len(self.grasp_frame_position_m) != 3:
            raise ValueError(f"invalid grasp frame for {self.object_id}")
        for name in ("scratch_right_hand_correction", "scratch_right_arm_correction"):
            values = getattr(self, name)
            if len(values) != 7 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"invalid {name} for {self.object_id}")
        if not 0.0 < self.exploration_std <= 0.2:
            raise ValueError(f"invalid exploration std for {self.object_id}")


BASE_HAND_CORRECTION = (
    0.00324760,
    -0.09997928,
    0.00110806,
    -0.02096313,
    -0.05184034,
    0.00053336,
    0.12066098,
)
BASE_ARM_CORRECTION = (
    -0.63216424,
    -0.05362973,
    0.14593603,
    0.29428789,
    -0.10058338,
    0.43392509,
    -0.24306557,
)


OBJECTS = {
    spec.object_id: spec
    for spec in (
        ObjectSpec(
            object_id="Bottle_1",
            category="bottle",
            source_relative="Kitchen Objects/Bottle/Prefabs/Bottle_1/Bottle_1.xml",
            asset_version="object_reward_v2",
            policy_version="single_ref_ep82_per_object_v1",
            scale=0.75,
            grip_width_m=0.078,
            mass_kg=0.25,
            grasp_frame_position_m=(0.0, -0.025, 0.0),
            maximum_grip_force_newtons=80.0,
            scratch_right_hand_correction=BASE_HAND_CORRECTION,
            scratch_right_arm_correction=BASE_ARM_CORRECTION,
        ),
        ObjectSpec(
            object_id="Soap_Bottle_1",
            category="bottle",
            source_relative=(
                "Bathroom Objects/SoapBottle/Prefabs/Soap_Bottle_1/"
                "Soap_Bottle_1.xml"
            ),
            asset_version="object_reward_v2",
            policy_version="single_ref_ep82_per_object_v1",
            scale=1.0,
            grip_width_m=0.065,
            mass_kg=0.20,
            grasp_frame_position_m=(0.0, -0.015, 0.0),
            maximum_grip_force_newtons=70.0,
            scratch_right_hand_correction=BASE_HAND_CORRECTION,
            scratch_right_arm_correction=BASE_ARM_CORRECTION,
        ),
        ObjectSpec(
            object_id="Apple_1",
            category="round",
            source_relative="Kitchen Objects/Apple/Prefabs/Apple_1/Apple_1.xml",
            # The v2 apple was 8.5 cm wide, larger than both graspable bottle
            # bodies and the reference hand aperture. Keep that failed asset
            # immutable and derive a realistically sized 6.4 cm variant.
            asset_version="object_reward_v3_small",
            policy_version="single_ref_ep82_small_v4_multireplay",
            scale=0.6,
            grip_width_m=0.064,
            mass_kg=0.14,
            grasp_frame_position_m=(0.0, 0.0, 0.0),
            maximum_grip_force_newtons=60.0,
            # The isolated staged reference already contains Apple's complete
            # grasp trajectory. Start PPO exactly on it with a zero residual.
            scratch_right_hand_correction=(0.0,) * 7,
            scratch_right_arm_correction=(0.0,) * 7,
            exploration_std=0.005,
        ),
        ObjectSpec(
            object_id="Bowl_1",
            category="bowl",
            source_relative="Kitchen Objects/Bowl/Prefabs/Bowl_1/Bowl_1.xml",
            asset_version="object_reward_v3_rim",
            policy_version="single_ref_ep82_rim_v4_staged",
            scale=0.45,
            grip_width_m=0.085,
            mass_kg=0.18,
            # Source +Y is vertical after the catalog's upright rotation.  Aim
            # at the near rim instead of the root-centred point near the table.
            grasp_frame_position_m=(0.0, 0.012, -0.024),
            maximum_grip_force_newtons=70.0,
            # The isolated staged reference already contains the rim grasp.
            scratch_right_hand_correction=(0.0,) * 7,
            scratch_right_arm_correction=(0.0,) * 7,
            exploration_std=0.005,
        ),
    )
}


@dataclass(frozen=True)
class StageSpec:
    center_xy_m: tuple[float, float]
    jitter_xy_m: tuple[float, float]
    yaw_jitter_rad: float
    default_iterations: int


STAGES = {
    # Bootstrap a newly exported reference at its exact verified object pose.
    # Position/yaw DR remains gated on success here.
    "fixed": StageSpec((0.087, 0.0), (0.0, 0.0), 0.0, 80),
    # The original reference is near the edge. Both stages shift the object
    # inward far enough to retain a conservative 3 cm table margin even for
    # the largest yaw-randomized footprint in this catalog.
    "narrow": StageSpec((0.087, 0.0), (0.005, 0.005), 0.015, 200),
    "workspace": StageSpec((0.105, 0.0), (0.020, 0.025), 0.080, 40),
}

LIFT_ARM_DECAY_MIN_SCALE = 0.1
LIFT_ARM_DECAY_STEPS = 10
LIFT_ARM_DECAY_GRASP_STEPS = 3
LIFT_ARM_DECAY_VARIANT = "lift_arm_decay_v1"


def asset_path(object_id: str) -> Path:
    spec = OBJECTS[object_id]
    return (
        REPO_ROOT
        / "outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything"
        / f"{object_id}_{spec.asset_version}"
    )


def reference_path(object_id: str) -> Path:
    return {
        "Apple_1": APPLE_STAGED_REFERENCE,
        "Bowl_1": BOWL_STAGED_REFERENCE,
    }.get(object_id, REFERENCE)


def run_root(object_id: str) -> Path:
    return (
        REPO_ROOT
        / "outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything"
        / object_id
        / OBJECTS[object_id].policy_version
    )


def stage_output(object_id: str, stage: str) -> Path:
    return run_root(object_id) / stage


def default_checkpoint(object_id: str, stage: str, iterations: int | None = None) -> Path:
    count = iterations or STAGES[stage].default_iterations
    return stage_output(object_id, stage) / f"model_{count - 1}.pt"


def _source(spec: ObjectSpec, molmo_root: Path) -> Path:
    return molmo_root / spec.source_relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive(spec: ObjectSpec, molmo_root: Path) -> dict[str, object]:
    spec.validate()
    source = _source(spec, molmo_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    output = asset_path(spec.object_id)
    if (output / "manifest.json").is_file():
        manifest = json.loads((output / "manifest.json").read_text())
        contract = manifest.get("object_contract", {})
        if (
            manifest.get("object_id") != spec.object_id
            or contract.get("source_mjcf_sha256") != _sha256(source)
        ):
            raise RuntimeError(f"existing bundle does not match catalog: {output}")
        return manifest

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


def _rotation_matrix(quaternion_wxyz: Iterable[float]) -> list[list[float]]:
    w, x, y, z = quaternion_wxyz
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _world_half_extents(contract: dict[str, object]) -> tuple[float, float, float]:
    rotation = _rotation_matrix(contract["upright_quaternion_wxyz"])
    local = contract["half_extents_m"]
    return tuple(
        sum(abs(rotation[row][column]) * float(local[column]) for column in range(3))
        for row in range(3)
    )


def _workspace_audit(manifest: dict[str, object], stage: str) -> dict[str, object]:
    import xml.etree.ElementTree as ET

    profile = STAGES[stage]
    root = ET.parse(asset_path(str(manifest["object_id"])) / "scene.xml").getroot()
    table_body = root.find(".//body[@name='table']")
    if table_body is None:
        raise ValueError("scene is missing the table body")
    table_geom = table_body.find("geom[@name='table_geom']")
    if table_geom is None:
        raise ValueError("scene is missing table_geom")
    table_center = tuple(float(value) for value in table_body.attrib["pos"].split()[:2])
    table_half = tuple(float(value) for value in table_geom.attrib["size"].split()[:2])
    initial = tuple(float(value) for value in manifest["reset"]["initial_object_pos"][:2])
    object_half = _world_half_extents(manifest["object_contract"])
    # A yaw-randomized rectangular footprint always fits inside this radius.
    footprint_radius = math.hypot(object_half[0], object_half[1])
    bounds = tuple(
        (
            initial[axis] + profile.center_xy_m[axis] - profile.jitter_xy_m[axis],
            initial[axis] + profile.center_xy_m[axis] + profile.jitter_xy_m[axis],
        )
        for axis in range(2)
    )
    table_bounds = tuple(
        (table_center[axis] - table_half[axis], table_center[axis] + table_half[axis])
        for axis in range(2)
    )
    margins = {
        "x_min": bounds[0][0] - footprint_radius - table_bounds[0][0],
        "x_max": table_bounds[0][1] - bounds[0][1] - footprint_radius,
        "y_min": bounds[1][0] - footprint_radius - table_bounds[1][0],
        "y_max": table_bounds[1][1] - bounds[1][1] - footprint_radius,
    }
    minimum_margin = min(margins.values())
    if minimum_margin < 0.03:
        raise ValueError(
            f"{manifest['object_id']} {stage} workspace has only "
            f"{minimum_margin:.4f} m conservative table margin"
        )
    return {
        "stage": stage,
        "target_center_bounds_xy_m": bounds,
        "world_half_extents_m": object_half,
        "yaw_safe_footprint_radius_m": footprint_radius,
        "table_edge_margins_m": margins,
        "minimum_table_margin_m": minimum_margin,
    }


def verify(spec: ObjectSpec, molmo_root: Path) -> dict[str, object]:
    source = _source(spec, molmo_root)
    bundle = asset_path(spec.object_id)
    if not (bundle / "manifest.json").is_file():
        raise FileNotFoundError(f"derive the object bundle first: {bundle}")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from simple.grasp_rl.grasp_anything import validate_grasp_anything_bundle

    validation = validate_grasp_anything_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    contract = manifest["object_contract"]
    if manifest["object_id"] != spec.object_id or contract["object_id"] != spec.object_id:
        raise ValueError("object identity is not isolated")
    if contract["source_mjcf_sha256"] != _sha256(source):
        raise ValueError("source object hash changed")
    if manifest["source_task"] != "xmove_pick" or contract["reference_task"] != "xmove_pick":
        raise ValueError("reference task contract changed")
    reference = reference_path(spec.object_id)
    expected_reference = "bc/episode_000082.npz"
    if not (reference / expected_reference).is_file():
        raise FileNotFoundError("strict episode-82 reference is missing")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(bundle / manifest["scene_file"]))
    primary = model.body(manifest["roles"]["primary"]).id
    free_joints = [
        joint
        for joint in range(model.njnt)
        if int(model.jnt_bodyid[joint]) == primary
        and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joints) != 1:
        raise ValueError("target object must remain a single free body")
    if model.neq:
        raise ValueError("target scene unexpectedly contains equality constraints")

    return {
        "object_id": spec.object_id,
        "category": spec.category,
        "asset": str(bundle),
        "validation": validation,
        "physics": {
            "target_is_free_body": True,
            "equality_constraint_count": int(model.neq),
            "mass_kg": contract["mass_kg"],
            "grip_width_m": contract["grip_width_m"],
            "half_extents_m": contract["half_extents_m"],
        },
        "workspaces": [
            _workspace_audit(manifest, "fixed"),
            _workspace_audit(manifest, "narrow"),
            _workspace_audit(manifest, "workspace"),
        ],
    }


def _environment_args(
    spec: ObjectSpec, stage: str, seed: int, *, lift_arm_decay: bool = False
) -> list[str]:
    profile = STAGES[stage]
    arguments = [
        "--task",
        "grasp_anything",
        "--asset-bundle",
        str(asset_path(spec.object_id)),
        "--reference-processed",
        str(reference_path(spec.object_id)),
        "--strict-reference-episode",
        "82",
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
        "--dr-initial-strength",
        "1",
        "--dr-warmup-steps",
        "0",
        "--dr-ramp-steps",
        "1",
        "--dr-profile",
        "pose_only",
        "--target-position-jitter-xy",
        *(str(value) for value in profile.jitter_xy_m),
        "--target-position-offset-center-xy",
        *(str(value) for value in profile.center_xy_m),
        "--target-yaw-jitter",
        str(profile.yaw_jitter_rad),
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
        "0",
        "0",
        "--robot-base-yaw-jitter",
        "0",
    ]
    if spec.object_id != "Apple_1":
        arguments.extend(
            [
                "--reference-target-x-arm-gains",
                "-5.3",
                "2.4",
                "--reference-target-y-arm-gains",
                "12",
                "0",
                "--reference-target-positive-y-arm-gains",
                "22",
                "0",
            ]
        )
    if lift_arm_decay:
        arguments.extend(
            [
                "--grasp-anything-lift-arm-residual-min-scale",
                str(LIFT_ARM_DECAY_MIN_SCALE),
                "--grasp-anything-lift-arm-residual-decay-steps",
                str(LIFT_ARM_DECAY_STEPS),
                "--grasp-anything-lift-arm-residual-grasp-steps",
                str(LIFT_ARM_DECAY_GRASP_STEPS),
            ]
        )
    return arguments


def _gpu_cli(gpu: int) -> tuple[list[str], dict[str, str]]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MUJOCO_GL": "egl",
        }
    )
    command = [
        "uv",
        "run",
        "--project",
        "mjlab_gpu",
        "--no-sync",
        "python",
        "-m",
        "simple.grasp_rl.mjlab_gpu.cli",
    ]
    return command, environment


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
        decoder = json.JSONDecoder()
        payload = None
        for offset, character in enumerate(process.stdout):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(process.stdout[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "result" in candidate:
                payload = candidate
                break
        if payload is None:
            log.write_text(process.stdout)
        else:
            log.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(process.stdout, end="")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _validate_checkpoint_object(spec: ObjectSpec, checkpoint: Path) -> None:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("mjlab_gpu_metadata")
    object_id = (
        None
        if not isinstance(metadata, dict)
        else metadata.get("reward", {}).get("object_id")
    )
    if object_id != spec.object_id:
        raise ValueError(
            f"checkpoint object {object_id!r} does not match {spec.object_id!r}"
        )


def train(
    spec: ObjectSpec,
    *,
    stage: str,
    gpu: int,
    seed: int,
    num_envs: int,
    iterations: int,
    warm_start: Path | None,
    smoke: bool = False,
    lift_arm_decay: bool = False,
) -> Path:
    output_stage = f"{stage}_smoke" if smoke else stage
    if lift_arm_decay:
        output_stage = f"{output_stage}_{LIFT_ARM_DECAY_VARIANT}"
    output = stage_output(spec.object_id, output_stage)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing PPO run: {output}")
    if warm_start is not None and not warm_start.is_file():
        raise FileNotFoundError(warm_start)
    if warm_start is not None:
        _validate_checkpoint_object(spec, warm_start)
    initialization = (
        [
            "--plan-conditioned-actor",
            "--scratch-actor-output-scale",
            "0.000001",
            "--scratch-right-hand-correction",
            *(str(value) for value in spec.scratch_right_hand_correction),
            "--scratch-right-arm-correction",
            *(str(value) for value in spec.scratch_right_arm_correction),
        ]
        if warm_start is None
        else ["--warm-start", str(warm_start), "--warm-start-critic"]
    )
    _run_gpu(
        [
            "train",
            *_environment_args(
                spec, stage, seed, lift_arm_decay=lift_arm_decay
            ),
            "--learning-rate",
            "0.0003",
            "--actor-learning-rate-scale",
            "0.05",
            "--schedule",
            "fixed",
            "--exploration-std",
            str(spec.exploration_std),
            "--exploration-hold-steps",
            "8",
            "--ppo-clip-param",
            "0.05",
            "--ppo-learning-epochs",
            "2",
            "--ppo-max-grad-norm",
            "0.2",
            "--ppo-steps-per-env",
            "24",
            "--save-interval",
            "4",
            "--output",
            str(output),
            *initialization,
            "--num-envs",
            str(num_envs),
            "--iterations",
            str(iterations),
            *(["--smoke"] if smoke else []),
        ],
        gpu=gpu,
    )
    return output / f"model_{iterations - 1}.pt"


def evaluate(
    spec: ObjectSpec,
    *,
    stage: str,
    gpu: int,
    seed: int,
    episodes: int,
    checkpoint: Path,
    label: str,
    lift_arm_decay: bool = False,
) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _validate_checkpoint_object(spec, checkpoint)
    log = run_root(spec.object_id) / "acceptance" / f"{label}_seed{seed}_{episodes}.json"
    _run_gpu(
        [
            "evaluate",
            *_environment_args(
                spec, stage, seed, lift_arm_decay=lift_arm_decay
            ),
            "--checkpoint",
            str(checkpoint),
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
        log=log,
    )
    return log


def record(
    spec: ObjectSpec,
    *,
    stage: str,
    gpu: int,
    seed: int,
    checkpoint: Path,
    videos: int,
    lift_arm_decay: bool = False,
) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _validate_checkpoint_object(spec, checkpoint)
    suffix = f"_{LIFT_ARM_DECAY_VARIANT}" if lift_arm_decay else ""
    output = run_root(spec.object_id) / f"videos_{stage}_dr1{suffix}"
    _run_gpu(
        [
            "record",
            *_environment_args(
                spec, stage, seed, lift_arm_decay=lift_arm_decay
            ),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--num-envs",
            "32",
            "--videos",
            str(videos),
            "--max-attempts",
            "300",
            "--camera-view",
            "grasp_closeup",
            "--evaluation-dr-strength",
            "1",
            "--smoke",
        ],
        gpu=gpu,
    )
    return output


def _selected_specs(name: str) -> list[ObjectSpec]:
    if name == "all":
        return list(OBJECTS.values())
    try:
        return [OBJECTS[name]]
    except KeyError as error:
        raise ValueError(f"unknown object {name!r}; choose from {sorted(OBJECTS)}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molmo-root",
        type=Path,
        default=Path(os.environ.get("MOLMO_OBJECT_ROOT", DEFAULT_MOLMO_ROOT)),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    for command in ("derive", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("object", choices=[*OBJECTS, "all"])
    for command in ("evaluate", "record", "train"):
        child = subparsers.add_parser(command)
        child.add_argument("object", choices=OBJECTS)
        child.add_argument("--stage", choices=STAGES, default="narrow")
        child.add_argument("--gpu", type=int, default=0)
        child.add_argument("--seed", type=int, default=20260830)
        child.add_argument("--checkpoint", type=Path)
        child.add_argument("--lift-arm-decay", action="store_true")
        if command == "evaluate":
            child.add_argument("--episodes", type=int, default=512)
        elif command == "record":
            child.add_argument("--videos", type=int, default=3)
        elif command == "train":
            child.add_argument("--num-envs", type=int, default=8192)
            child.add_argument("--iterations", type=int)
            child.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        print(json.dumps({name: asdict(spec) for name, spec in OBJECTS.items()}, indent=2))
        return 0
    specs = _selected_specs(args.object)
    if args.command == "derive":
        print(json.dumps([derive(spec, args.molmo_root) for spec in specs], indent=2))
        return 0
    if args.command == "verify":
        print(json.dumps([verify(spec, args.molmo_root) for spec in specs], indent=2))
        return 0

    spec = specs[0]
    verify(spec, args.molmo_root)
    if args.command == "train":
        iterations = args.iterations or STAGES[args.stage].default_iterations
        warm_start = args.checkpoint
        if warm_start is None and args.stage == "workspace":
            warm_start = default_checkpoint(spec.object_id, "narrow")
        output = train(
            spec,
            stage=args.stage,
            gpu=args.gpu,
            seed=args.seed,
            num_envs=args.num_envs,
            iterations=iterations,
            warm_start=warm_start,
            smoke=args.smoke,
            lift_arm_decay=args.lift_arm_decay,
        )
    elif args.command == "evaluate":
        checkpoint = args.checkpoint or default_checkpoint(spec.object_id, args.stage)
        output = evaluate(
            spec,
            stage=args.stage,
            gpu=args.gpu,
            seed=args.seed,
            episodes=args.episodes,
            checkpoint=checkpoint,
            label=(
                f"policy_{args.stage}_{LIFT_ARM_DECAY_VARIANT}"
                if args.lift_arm_decay
                else f"policy_{args.stage}"
            ),
            lift_arm_decay=args.lift_arm_decay,
        )
    else:
        checkpoint = args.checkpoint or default_checkpoint(spec.object_id, args.stage)
        output = record(
            spec,
            stage=args.stage,
            gpu=args.gpu,
            seed=args.seed,
            checkpoint=checkpoint,
            videos=args.videos,
            lift_arm_decay=args.lift_arm_decay,
        )
    print(json.dumps({"object_id": spec.object_id, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
