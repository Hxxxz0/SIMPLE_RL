"""Export frozen SIMPLE scenes for the MuJoCo-Warp PPO backend.

The exporter intentionally runs in SIMPLE's legacy Python 3.10 environment.
It asks the existing task implementation to build and stabilize a scene, then
serializes the exact MuJoCo topology, reset state, role mapping, and asset
hashes.  The GPU environment only consumes this frozen bundle; it never tries
to import Isaac Sim, Gear-Sonic, or the CPU task builder.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.schema import (
    JOINT_NAMES,
    LEFT_CONTACT_LINK_NAMES,
    LEFT_DISTAL_LINK_NAMES,
    RIGHT_CONTACT_LINK_NAMES,
    RIGHT_DISTAL_LINK_NAMES,
)
from simple.grasp_rl.task_spec import TaskSpecV2, get_task_spec
from simple.grasp_rl.tracker import ActionTransform

ASSET_BUNDLE_VERSION = 1
AMO_CONTROLLER_STATE_VERSION = 1
RENDER_BUNDLE_VERSION = 1


def _enum_label(value: int, choices: dict[int, str], name: str) -> str:
    try:
        return choices[int(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported GPU {name} enum {int(value)}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _repo_root() -> Path:
    # .../src/simple/grasp_rl/mjlab_assets.py -> repository root.
    return Path(__file__).resolve().parents[3]


def _resolve_asset_path(
    value: str,
    *,
    repo_root: Path,
    robot_asset_dir: Path,
) -> Path:
    candidate = Path(value)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
        parts = candidate.parts
        if "data" in parts:
            candidates.append(repo_root / Path(*parts[parts.index("data") :]))
    else:
        candidates.extend(
            (
                robot_asset_dir / "meshes" / candidate,
                robot_asset_dir / candidate,
                repo_root / candidate,
                repo_root / "data" / candidate,
            )
        )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Cannot resolve MuJoCo asset {value!r}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def _portable_scene_xml(
    xml: str,
    output_dir: Path,
    *,
    repo_root: Path,
    controller: str,
    physics_only: bool = True,
) -> tuple[str, list[dict[str, str | int]]]:
    """Copy referenced files, optionally stripping rendering-only elements."""

    root = ET.fromstring(xml)
    # MuJoCo 3.3's MjSpec serializer can emit an anonymous <default> nested
    # inside another anonymous <default>.  MuJoCo 3.9 rejects that as an empty
    # class name.  Moving only anonymous children up preserves the same global
    # defaults while leaving every named class hierarchy untouched.
    for defaults in root.findall(".//default"):
        for child in list(defaults):
            if child.tag != "default" or child.get("class"):
                continue
            insertion = list(defaults).index(child)
            defaults.remove(child)
            for nested in list(child):
                defaults.insert(insertion, nested)
                insertion += 1
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", ".")
    compiler.set("texturedir", ".")

    option = root.find("option")
    if option is not None:
        # MuJoCo-Warp does not implement the CPU noslip solver pass.
        option.set("noslip_iterations", "0")

    for parent in root.iter():
        for child in list(parent):
            if physics_only and child.tag in {"camera", "light"}:
                parent.remove(child)
                continue
            if physics_only and child.tag == "geom":
                contype = child.get("contype")
                conaffinity = child.get("conaffinity")
                density = child.get("density")
                group = child.get("group")
                visual_only = (
                    contype == "0"
                    and conaffinity == "0"
                    and (density in {None, "0", "0.0"})
                    and group == "1"
                )
                if visual_only:
                    parent.remove(child)

    robot_asset_dir = (
        repo_root / "data/robots/g1_sonic"
        if controller == "sonic_wbc"
        else repo_root / "data/robots/g1"
    )
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, str] = {}
    records: list[dict[str, str | int]] = []
    for element in root.iter():
        value = element.get("file")
        if not value:
            continue
        source = _resolve_asset_path(
            value,
            repo_root=repo_root,
            robot_asset_dir=robot_asset_dir,
        )
        relative = copied.get(source)
        if relative is None:
            digest = _sha256(source)
            relative = f"assets/{digest[:16]}_{source.name}"
            destination = output_dir / relative
            shutil.copy2(source, destination)
            copied[source] = relative
            try:
                source_record = str(source.relative_to(repo_root))
            except ValueError:
                source_record = str(source)
            records.append(
                {
                    "source": source_record,
                    "bundle_path": relative,
                    "sha256": digest,
                    "bytes": source.stat().st_size,
                }
            )
        element.set("file", relative)

    return ET.tostring(root, encoding="unicode"), sorted(
        records, key=lambda item: str(item["bundle_path"])
    )


def _joint_layout(model: mujoco.MjModel) -> list[tuple[str, int, int, int]]:
    return [
        (
            model.joint(index).name,
            int(model.jnt_type[index]),
            int(model.jnt_qposadr[index]),
            int(model.jnt_dofadr[index]),
        )
        for index in range(model.njnt)
    ]


def _pin_render_distractors(env: GraspRlEnv, physics_model: mujoco.MjModel) -> None:
    """Recreate the exact GraspNet distractors stored in a frozen bundle."""

    randomizer = env.task.dr.get_randomizer("distractors")
    if randomizer is None or randomizer.cfg.number_of_distractors == 0:
        return
    if randomizer.cfg.res_id != "graspnet1b":
        return

    from simple.assets.graspnet import GraspNet_1B_Object_Names

    label_to_id = {
        "_".join(name.split()).replace("-", "_"): str(asset_id)
        for asset_id, name in GraspNet_1B_Object_Names.items()
    }
    excluded = {str(value) for value in (randomizer.cfg.exclude or ())}
    distractor_ids = []
    for index in range(physics_model.njnt):
        joint_name = physics_model.joint(index).name
        if not joint_name.endswith("_joint"):
            continue
        asset_id = label_to_id.get(joint_name.removesuffix("_joint"))
        if asset_id is not None and asset_id not in excluded:
            distractor_ids.append(asset_id)
    expected = int(randomizer.cfg.number_of_distractors)
    if len(distractor_ids) != expected:
        raise ValueError(
            "Cannot recover frozen GraspNet distractors for render sidecar: "
            f"expected {expected}, found {distractor_ids}"
        )
    randomizer.cfg.include = distractor_ids
    randomizer._inner_state = None


def _body_name(model, body_id: int | None) -> str | None:
    return None if body_id is None else model.body(int(body_id)).name


def _model_topology(model) -> dict[str, int]:
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nsensor": int(model.nsensor),
    }


def _export_amo_controller(
    env: GraspRlEnv,
    output: Path,
    repo_root: Path,
) -> dict:
    runtime = env.task.robot.get_runtime_state()
    if "amo_policy" not in runtime:
        raise ValueError("AMO robot did not expose amo_policy runtime state")
    amo = runtime["amo_policy"]
    policy = env.task.robot.amo_policy
    parity_action = np.asarray(env.previous_physical_action, dtype=np.float32)
    command = np.asarray(
        [
            parity_action[32],
            parity_action[35],
            parity_action[33],
            parity_action[31] - 0.75,
            parity_action[30],
            parity_action[29],
            parity_action[28],
            parity_action[34],
        ],
        dtype=np.float32,
    )
    try:
        parity_pd_target, _ = policy.get_action(
            env.task.robot.joints,
            env.task.robot.actuators,
            env.sim.mjData,
            command,
        )
        parity = {
            "parity_physical_action": parity_action,
            "parity_pd_target": np.asarray(parity_pd_target, dtype=np.float32),
            "parity_obs_prop": np.asarray(policy.obs_prop, dtype=np.float32),
            "parity_policy_obs": np.asarray(policy.obs, dtype=np.float32),
            "parity_adapter_input": policy.adapter_input.detach()
            .cpu()
            .numpy()
            .squeeze(0),
            "parity_adapter_output": policy.adapter_output.detach()
            .cpu()
            .numpy()
            .squeeze(0),
            "parity_extra_history": np.asarray(
                policy.extra_history_buf, dtype=np.float32
            ),
            "parity_last_action_for_policy": np.asarray(
                policy.last_action_for_policy, dtype=np.float32
            ),
            "parity_gait_cycle": np.asarray(policy.gait_cycle, dtype=np.float32),
        }
    finally:
        policy.set_runtime_state(amo)
    required = {
        "initial_quat": np.asarray(amo["_initial_quat"], dtype=np.float32),
        "gait_cycle": np.asarray(amo["gait_cycle"], dtype=np.float32),
        "last_action_for_policy": np.asarray(
            amo["last_action_for_policy"], dtype=np.float32
        ),
        "proprio_history": np.asarray(amo["proprio_history_buf"], dtype=np.float32),
        "extra_history": np.asarray(amo["extra_history_buf"], dtype=np.float32),
        "last_commands": np.asarray(amo["_last_commands"], dtype=np.float32),
        "in_place_stand_flag": np.asarray(
            [amo["_in_place_stand_flag"]], dtype=np.bool_
        ),
        "last_target_yaw": np.asarray([amo["_last_target_yaw"]], dtype=np.float32),
        "target_yaw": np.asarray([amo["target_yaw"]], dtype=np.float32),
        **parity,
    }
    expected_shapes = {
        "initial_quat": (4,),
        "gait_cycle": (2,),
        "last_action_for_policy": (23,),
        "proprio_history": (10, 93),
        "extra_history": (25, 93),
        "last_commands": (8,),
        "in_place_stand_flag": (1,),
        "last_target_yaw": (1,),
        "target_yaw": (1,),
        "parity_physical_action": (36,),
        "parity_pd_target": (15,),
        "parity_obs_prop": (93,),
        "parity_policy_obs": (1043,),
        "parity_adapter_input": (12,),
        "parity_adapter_output": (15,),
        "parity_extra_history": (25, 93),
        "parity_last_action_for_policy": (23,),
        "parity_gait_cycle": (2,),
    }
    actual_shapes = {name: value.shape for name, value in required.items()}
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"Unexpected warm AMO state shapes: {actual_shapes} != {expected_shapes}"
        )
    for name, value in required.items():
        expected_dtype = np.bool_ if name == "in_place_stand_flag" else np.float32
        if value.dtype != expected_dtype:
            raise ValueError(
                f"Unexpected warm AMO state dtype for {name}: {value.dtype}"
            )
        if value.dtype != np.bool_ and not np.isfinite(value).all():
            raise ValueError(f"Warm AMO state {name} contains non-finite values")

    controller_dir = output / "controller"
    controller_dir.mkdir(parents=True, exist_ok=True)
    state_path = controller_dir / "state.npz"
    temporary_state_path = controller_dir / ".state.tmp.npz"
    np.savez(temporary_state_path, **required)
    temporary_state_path.replace(state_path)
    policy_dir = repo_root / "src/simple/robots/policy"
    artifacts = []
    for name in ("amo_jit.pt", "adapter_jit.pt", "adapter_norm_stats.pt"):
        source = policy_dir / name
        destination = controller_dir / name
        shutil.copy2(source, destination)
        artifacts.append(
            {
                "bundle_path": str(destination.relative_to(output)),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    robot_names = list(env.task.robot.joints_names)
    logical_indices = [robot_names.index(name) for name in JOINT_NAMES]
    return {
        "format_version": AMO_CONTROLLER_STATE_VERSION,
        "state_file": str(state_path.relative_to(output)),
        "state_sha256": _sha256(state_path),
        "state_shapes": {name: list(shape) for name, shape in expected_shapes.items()},
        "artifacts": artifacts,
        "joint_parameters": {
            "names": list(JOINT_NAMES),
            "stiffness": np.asarray(env.task.robot.stiffness)[logical_indices].tolist(),
            "damping": np.asarray(env.task.robot.damping)[logical_indices].tolist(),
            "torque_limits": np.asarray(env.task.robot.torque_limits)[
                logical_indices
            ].tolist(),
        },
        "action_scale": float(policy.action_scale),
        "gait_frequency": float(policy.gait_freq),
    }


def _role_names(env: GraspRlEnv) -> dict[str, str | None]:
    assert env.state is not None
    if isinstance(env.task_spec, TaskSpecV2):
        return {
            role: _body_name(env.sim.mjModel, body_id)
            for role, body_id in env.state.role_body_ids.items()
        }
    return {
        "primary": _body_name(env.sim.mjModel, env.state.target_id),
        "destination": _body_name(env.sim.mjModel, env.state.table_id),
        "auxiliary": None,
    }


def _robot_xml_path(task_spec, repo_root: Path) -> Path:
    relative = (
        "data/robots/g1_sonic/g1_29dof_with_hand.xml"
        if isinstance(task_spec, TaskSpecV2)
        and task_spec.controller_backend == "sonic_wbc"
        else "data/robots/g1/g1_29dof_wholebody_dex3.xml"
    )
    return repo_root / relative


def _default_action_transform(task_spec, repo_root: Path) -> Path:
    candidate = task_spec.processed_path(repo_root / "data/grasp_rl")
    candidate = candidate / "action_transform.npz"
    if candidate.is_file():
        return candidate
    fallback = (
        repo_root / "data/grasp_rl/G1WholebodyTabletopGraspMP-v0/action_transform.npz"
    )
    if not fallback.is_file():
        raise FileNotFoundError("No action transform is available for scene export")
    return fallback


def export_mjlab_scene(
    task: str,
    output_dir: str | Path,
    *,
    seed: int = 42,
    target_object: str | None = None,
    warmup_steps: int = 60,
) -> dict:
    """Export one stabilized task/asset variant for GPU training."""

    task_spec = get_task_spec(task)
    repo_root = _repo_root()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transform_path = _default_action_transform(task_spec, repo_root)
    transform = ActionTransform.from_npz(transform_path)
    frozen_transform_path = output / "controller" / "action_transform.npz"
    frozen_transform_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(transform_path, frozen_transform_path)
    env = GraspRlEnv(
        transform,
        seed=seed,
        target_object=target_object,
        warmup_steps=warmup_steps,
        task=task_spec,
        enable_renderers=False,
    )
    try:
        env.reset(target_object=target_object)
        assert env.state is not None
        model = env.sim.mjModel
        data = env.sim.mjData
        scene_xml, assets = _portable_scene_xml(
            env.sim.mjSpec.to_xml(),
            output,
            repo_root=repo_root,
            controller=(
                task_spec.controller_backend
                if isinstance(task_spec, TaskSpecV2)
                else "amo"
            ),
        )
        scene_path = output / "scene.xml"
        scene_path.write_text(scene_xml)
        exported_model = mujoco.MjModel.from_xml_path(str(scene_path))
        robot_xml = _robot_xml_path(task_spec, repo_root)
        actor_observation, _ = env.state.actor_observation()
        initial_object_pos = np.asarray(env.state.initial_object_pos).tolist()
        goal_pos = np.asarray(env.state.goal_pos).tolist()
        reset_arrays = {
            "qpos": np.asarray(data.qpos),
            "qvel": np.asarray(data.qvel),
            "act": np.asarray(data.act),
            "ctrl": np.asarray(data.ctrl),
            "qacc_warmstart": np.asarray(data.qacc_warmstart),
            "sensordata": np.asarray(data.sensordata),
            "mocap_pos": np.asarray(data.mocap_pos),
            "mocap_quat": np.asarray(data.mocap_quat),
            "previous_physical_action": np.asarray(env.previous_physical_action),
            "actor_observation": np.asarray(actor_observation),
        }
        for name, value in reset_arrays.items():
            if value.dtype.kind != "f":
                raise ValueError(f"Reset {name} must use a floating dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"Reset {name} contains non-finite values")
        if not np.isfinite(float(data.time)):
            raise ValueError("Reset time is non-finite")
        if not np.isfinite(initial_object_pos).all() or not np.isfinite(goal_pos).all():
            raise ValueError("Reset object/goal position contains non-finite values")
        controller = (
            task_spec.controller_backend if isinstance(task_spec, TaskSpecV2) else "amo"
        )
        controller_bundle = (
            _export_amo_controller(env, output, repo_root)
            if controller == "amo"
            else None
        )
        manifest = {
            "format_version": ASSET_BUNDLE_VERSION,
            "task": task_spec.name,
            "task_metadata": task_spec.metadata(),
            "task_spec_hash": _json_hash(task_spec.metadata()),
            "controller": controller,
            "controller_bundle": controller_bundle,
            "seed": int(seed),
            "target_object": target_object or task_spec.target_object,
            "warmup_steps": int(warmup_steps),
            "scene_file": "scene.xml",
            "scene_sha256": _sha256(scene_path),
            "assets": assets,
            "robot_xml": str(robot_xml.relative_to(repo_root)),
            "robot_xml_sha256": _sha256(robot_xml),
            "action_transform": str(frozen_transform_path.relative_to(output)),
            "action_transform_sha256": _sha256(frozen_transform_path),
            "source_model": _model_topology(model),
            "model": _model_topology(exported_model),
            "reset": {
                "qpos": reset_arrays["qpos"].tolist(),
                "qvel": reset_arrays["qvel"].tolist(),
                "act": reset_arrays["act"].tolist(),
                "ctrl": reset_arrays["ctrl"].tolist(),
                "qacc_warmstart": reset_arrays["qacc_warmstart"].tolist(),
                "sensordata": reset_arrays["sensordata"].tolist(),
                "time": float(data.time),
                "mocap_pos": reset_arrays["mocap_pos"].tolist(),
                "mocap_quat": reset_arrays["mocap_quat"].tolist(),
                "previous_physical_action": reset_arrays[
                    "previous_physical_action"
                ].tolist(),
                "actor_observation": reset_arrays["actor_observation"].tolist(),
                "initial_object_pos": initial_object_pos,
                "goal_pos": goal_pos,
            },
            "roles": _role_names(env),
            "schema": {
                "joint_names": list(JOINT_NAMES),
                "right_contact_links": list(RIGHT_CONTACT_LINK_NAMES),
                "left_contact_links": list(LEFT_CONTACT_LINK_NAMES),
                "right_distal_links": list(RIGHT_DISTAL_LINK_NAMES),
                "left_distal_links": list(LEFT_DISTAL_LINK_NAMES),
            },
            "physics": {
                "timestep": float(exported_model.opt.timestep),
                "gravity": np.asarray(exported_model.opt.gravity).tolist(),
                "impratio": float(exported_model.opt.impratio),
                "integrator": _enum_label(
                    exported_model.opt.integrator,
                    {
                        int(mujoco.mjtIntegrator.mjINT_EULER): "euler",
                        int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast",
                    },
                    "integrator",
                ),
                "cone": _enum_label(
                    exported_model.opt.cone,
                    {
                        int(mujoco.mjtCone.mjCONE_PYRAMIDAL): "pyramidal",
                        int(mujoco.mjtCone.mjCONE_ELLIPTIC): "elliptic",
                    },
                    "cone",
                ),
                "jacobian": _enum_label(
                    exported_model.opt.jacobian,
                    {
                        int(mujoco.mjtJacobian.mjJAC_AUTO): "auto",
                        int(mujoco.mjtJacobian.mjJAC_DENSE): "dense",
                        int(mujoco.mjtJacobian.mjJAC_SPARSE): "sparse",
                    },
                    "jacobian",
                ),
                "solver": _enum_label(
                    exported_model.opt.solver,
                    {
                        int(mujoco.mjtSolver.mjSOL_NEWTON): "newton",
                        int(mujoco.mjtSolver.mjSOL_CG): "cg",
                        int(mujoco.mjtSolver.mjSOL_PGS): "pgs",
                    },
                    "solver",
                ),
                "iterations": int(exported_model.opt.iterations),
                "tolerance": float(exported_model.opt.tolerance),
                "ls_iterations": int(exported_model.opt.ls_iterations),
                "ls_tolerance": float(exported_model.opt.ls_tolerance),
                "ccd_iterations": int(exported_model.opt.ccd_iterations),
                "noslip_iterations_cpu": int(model.opt.noslip_iterations),
                "noslip_iterations_gpu": int(exported_model.opt.noslip_iterations),
            },
            "reward_spec": (
                asdict(task_spec.reward)
                if not isinstance(task_spec, TaskSpecV2)
                else task_spec.metadata()["spec"]
            ),
        }
        manifest["reward_hash"] = _json_hash(manifest["reward_spec"])
        manifest["manifest_hash"] = _json_hash(manifest)
        manifest_path = output / "manifest.json"
        temporary_manifest_path = output / ".manifest.tmp.json"
        temporary_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        temporary_manifest_path.replace(manifest_path)
        return manifest
    finally:
        env.close()


def export_mjlab_render_scene(path: str | Path) -> dict:
    """Add a full visual sidecar without changing the frozen physics bundle."""

    output = Path(path).resolve()
    manifest = validate_asset_bundle(output)
    task_spec = get_task_spec(manifest["task"])
    repo_root = _repo_root()
    transform = ActionTransform.from_npz(output / manifest["action_transform"])
    env = GraspRlEnv(
        transform,
        seed=int(manifest["seed"]),
        target_object=manifest["target_object"],
        warmup_steps=int(manifest["warmup_steps"]),
        task=task_spec,
        enable_renderers=False,
    )
    temporary_scene = output / ".render_scene.tmp.xml"
    try:
        physics_model = mujoco.MjModel.from_xml_path(
            str(output / manifest["scene_file"])
        )
        _pin_render_distractors(env, physics_model)
        env.reset(target_object=manifest["target_object"])
        render_xml, assets = _portable_scene_xml(
            env.sim.mjSpec.to_xml(),
            output,
            repo_root=repo_root,
            controller=manifest["controller"],
            physics_only=False,
        )
        temporary_scene.write_text(render_xml)
        render_model = mujoco.MjModel.from_xml_path(str(temporary_scene))
        render_topology = _model_topology(render_model)
        state_fields = ("nq", "nv", "nu", "nbody", "njnt", "nsensor")
        if any(
            render_topology[field] != manifest["source_model"][field]
            for field in state_fields
        ):
            raise ValueError(
                "Render sidecar does not match frozen state topology: "
                f"{render_topology} != {manifest['source_model']}"
            )
        render_joints = _joint_layout(render_model)
        physics_joints = _joint_layout(physics_model)
        if render_joints != physics_joints:
            difference = next(
                (
                    pair
                    for pair in zip(render_joints, physics_joints)
                    if pair[0] != pair[1]
                ),
                (len(render_joints), len(physics_joints)),
            )
            raise ValueError(
                f"Render sidecar joint layout does not match GPU physics: {difference}"
            )
        required_bodies = {
            "pelvis",
            "right_wrist_yaw_link",
            manifest["roles"]["primary"],
            manifest["roles"]["destination"],
        }
        render_bodies = {render_model.body(i).name for i in range(render_model.nbody)}
        if not required_bodies.issubset(render_bodies):
            raise ValueError("Render sidecar is missing required task bodies")
        scene = output / "render_scene.xml"
        temporary_scene.replace(scene)
        sidecar = {
            "format_version": RENDER_BUNDLE_VERSION,
            "base_manifest_hash": manifest["manifest_hash"],
            "scene_file": scene.name,
            "scene_sha256": _sha256(scene),
            "assets": assets,
            "model": render_topology,
            "ncam": int(render_model.ncam),
            "nlight": int(render_model.nlight),
        }
        sidecar["manifest_hash"] = _json_hash(sidecar)
        temporary_manifest = output / ".render_manifest.tmp.json"
        temporary_manifest.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
        temporary_manifest.replace(output / "render_manifest.json")
        return sidecar
    finally:
        temporary_scene.unlink(missing_ok=True)
        env.close()


def validate_asset_bundle(path: str | Path) -> dict:
    """Validate topology and every copied byte without importing mjlab."""

    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    expected_manifest_hash = manifest.pop("manifest_hash")
    actual_manifest_hash = _json_hash(manifest)
    manifest["manifest_hash"] = expected_manifest_hash
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError("Asset manifest hash mismatch")
    scene = root / manifest["scene_file"]
    if _sha256(scene) != manifest["scene_sha256"]:
        raise ValueError("Exported scene hash mismatch")
    action_transform = root / manifest["action_transform"]
    if _sha256(action_transform) != manifest["action_transform_sha256"]:
        raise ValueError("Action transform hash mismatch")
    for asset in manifest["assets"]:
        asset_path = root / asset["bundle_path"]
        if not asset_path.is_file() or _sha256(asset_path) != asset["sha256"]:
            raise ValueError(f"Asset hash mismatch: {asset_path}")
    controller = manifest.get("controller_bundle")
    if controller is not None:
        state_path = root / controller["state_file"]
        if (
            not state_path.is_file()
            or _sha256(state_path) != controller["state_sha256"]
        ):
            raise ValueError(f"Controller state hash mismatch: {state_path}")
        for artifact in controller["artifacts"]:
            artifact_path = root / artifact["bundle_path"]
            if (
                not artifact_path.is_file()
                or _sha256(artifact_path) != artifact["sha256"]
            ):
                raise ValueError(f"Controller artifact hash mismatch: {artifact_path}")
    return manifest
