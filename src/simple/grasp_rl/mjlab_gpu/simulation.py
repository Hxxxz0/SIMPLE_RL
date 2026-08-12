"""Strict frozen-asset loader for the MuJoCo-Warp simulation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import torch
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from simple.grasp_rl.grasp_anything import (
    GRASP_ANYTHING_TASK,
    GraspAnythingObjectContract,
)
from simple.grasp_rl.mjlab_gpu.config import MjlabPpoConfig
from simple.grasp_rl.mjlab_gpu.reference import validate_strict_reference_manifest
from simple.grasp_rl.schema import (
    LEFT_CONTACT_LINK_NAMES,
    LEFT_DISTAL_LINK_NAMES,
    RIGHT_CONTACT_LINK_NAMES,
    RIGHT_DISTAL_LINK_NAMES,
)

CUDA_DATA_FIELDS = (
    "qpos",
    "qvel",
    "ctrl",
    "xpos",
    "xquat",
    "cvel",
    "sensordata",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _topology(model: mujoco.MjModel) -> dict[str, int]:
    return {
        name: int(getattr(model, name))
        for name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsensor")
    }


def _bundle_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset path escapes bundle: {relative}")
    return path


@dataclass(frozen=True)
class FrozenAssetBundle:
    root: Path
    manifest: dict[str, Any]
    model: mujoco.MjModel

    @classmethod
    def load(cls, path: str | Path, *, expected_task: str) -> "FrozenAssetBundle":
        root = Path(path).resolve()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version") != 1:
            raise ValueError("Unsupported frozen asset bundle version")
        if manifest.get("task") != expected_task:
            raise ValueError(
                f"Asset task {manifest.get('task')!r} does not match {expected_task!r}"
            )
        unhashed = deepcopy(manifest)
        expected_hash = unhashed.pop("manifest_hash", None)
        if expected_hash is None or _json_hash(unhashed) != expected_hash:
            raise ValueError("Asset manifest hash mismatch")
        if _json_hash(manifest.get("reward_spec")) != manifest.get("reward_hash"):
            raise ValueError("Frozen reward specification hash mismatch")
        scene = _bundle_file(root, manifest["scene_file"])
        if _sha256(scene) != manifest["scene_sha256"]:
            raise ValueError("Exported scene hash mismatch")
        action_transform = _bundle_file(root, manifest["action_transform"])
        if _sha256(action_transform) != manifest["action_transform_sha256"]:
            raise ValueError("Action transform hash mismatch")
        for asset in manifest["assets"]:
            asset_path = _bundle_file(root, asset["bundle_path"])
            if not asset_path.is_file() or _sha256(asset_path) != asset["sha256"]:
                raise ValueError(f"Asset hash mismatch: {asset_path}")
        controller = manifest.get("controller_bundle")
        if controller is not None:
            state_path = _bundle_file(root, controller["state_file"])
            if _sha256(state_path) != controller["state_sha256"]:
                raise ValueError("Controller state hash mismatch")
            for artifact in controller["artifacts"]:
                artifact_path = _bundle_file(root, artifact["bundle_path"])
                if _sha256(artifact_path) != artifact["sha256"]:
                    raise ValueError(
                        f"Controller artifact hash mismatch: {artifact_path}"
                    )
        model = mujoco.MjModel.from_xml_path(str(scene))
        if _topology(model) != manifest["model"]:
            raise ValueError(
                f"GPU scene topology mismatch: {_topology(model)} != {manifest['model']}"
            )
        if expected_task == GRASP_ANYTHING_TASK:
            contract = GraspAnythingObjectContract.from_metadata(
                manifest.get("object_contract", {})
            )
            if manifest.get("object_id") != contract.object_id:
                raise ValueError("grasp_anything object_id/contract mismatch")
        return cls(root=root, manifest=manifest, model=model)


@dataclass(frozen=True)
class GpuSimulation:
    bundle: FrozenAssetBundle
    sim: Simulation
    sensors: "GpuSensorLayout | GpuV2SensorLayout | None"

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restore the frozen warm state for all or a CUDA subset of worlds."""

        _restore_reset_state(self.sim, self.bundle.manifest["reset"], env_ids)
        _assert_cuda_residency(self.sim, self.sim.device)


@dataclass(frozen=True)
class GpuSensorLayout:
    object_force: tuple[slice, ...]
    object_position: tuple[slice, ...]
    table_force: tuple[slice, ...]
    fingertip_distance: tuple[tuple[slice, ...], ...]
    target_linear_velocity: slice
    target_angular_velocity: slice
    wrist_linear_velocity: slice
    distal_linear_velocity: tuple[slice, ...]
    pelvis_linear_velocity: slice
    pelvis_angular_velocity: slice


@dataclass(frozen=True)
class GpuV2SensorLayout:
    primary_force: tuple[tuple[slice, ...], tuple[slice, ...]]
    auxiliary_force: tuple[tuple[slice, ...], tuple[slice, ...]] | None
    hand_support_force: tuple[
        tuple[slice, ...], tuple[slice, ...]
    ] | None
    primary_destination_force: slice | None
    primary_auxiliary_force: slice | None
    auxiliary_destination_force: slice | None
    fingertip_distance: tuple[
        tuple[tuple[slice, ...], ...], tuple[tuple[slice, ...], ...]
    ]
    linear_velocity: dict[str, slice]
    angular_velocity: dict[str, slice]


def _sensor_slice(model: mujoco.MjModel, name: str) -> slice:
    sensor_id = model.sensor(name).id
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + int(model.sensor_dim[sensor_id]))


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    while current > 0 and current != ancestor_id:
        current = int(model.body_parentid[current])
    return current == ancestor_id


def _subtree_geom_names(model: mujoco.MjModel, body_name: str) -> list[str]:
    body_id = model.body(body_name).id
    return [
        model.geom(index).name
        for index in range(model.ngeom)
        if _is_descendant(model, int(model.geom_bodyid[index]), body_id)
    ]


def _augment_legacy_sensors(
    bundle: FrozenAssetBundle,
) -> tuple[mujoco.MjModel, GpuSensorLayout]:
    """Append deterministic GPU-native contact and geom-distance sensors."""

    scene = _bundle_file(bundle.root, bundle.manifest["scene_file"])
    spec = mujoco.MjSpec.from_file(str(scene))
    # MjSpec permits anonymous geoms, but GEOMDIST sensors require names.
    for index, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"gpu_geom_{index:04d}_{geom.parent.name}"
    named_model = spec.compile()
    target_name = bundle.manifest["roles"]["primary"]
    table_name = bundle.manifest["roles"]["destination"]
    if not target_name or not table_name:
        raise ValueError("Legacy GPU sensors require primary and destination bodies")

    object_force_names = []
    object_position_names = []
    table_force_names = []
    for index, link_name in enumerate(RIGHT_CONTACT_LINK_NAMES):
        force_name = f"gpu_object_force_{index:02d}"
        position_name = f"gpu_object_position_{index:02d}"
        support_name = f"gpu_table_force_{index:02d}"
        spec.add_sensor(
            name=force_name,
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname=link_name,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=target_name,
            intprm=[1 << 1, 3, 1],  # force, netforce, one slot
        )
        spec.add_sensor(
            name=position_name,
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname=link_name,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=target_name,
            intprm=[1 << 4, 2, 1],  # position, maxforce, one slot
        )
        spec.add_sensor(
            name=support_name,
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname=link_name,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=table_name,
            intprm=[1 << 1, 3, 1],
        )
        object_force_names.append(force_name)
        object_position_names.append(position_name)
        table_force_names.append(support_name)

    target_geoms = _subtree_geom_names(named_model, target_name)
    distance_names: list[list[str]] = []
    for finger_index, link_name in enumerate(RIGHT_DISTAL_LINK_NAMES):
        finger_names = []
        for pair_index, (finger_geom, target_geom) in enumerate(
            (finger, target)
            for finger in _subtree_geom_names(named_model, link_name)
            for target in target_geoms
        ):
            name = f"gpu_fingertip_distance_{finger_index:02d}_{pair_index:03d}"
            spec.add_sensor(
                name=name,
                type=mujoco.mjtSensor.mjSENS_GEOMDIST,
                objtype=mujoco.mjtObj.mjOBJ_GEOM,
                objname=finger_geom,
                reftype=mujoco.mjtObj.mjOBJ_GEOM,
                refname=target_geom,
                cutoff=2.0,
            )
            finger_names.append(name)
        if not finger_names:
            raise ValueError(f"No geom-distance pairs for {link_name}")
        distance_names.append(finger_names)

    velocity_sensors = {
        "gpu_target_linear_velocity": (
            mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            target_name,
        ),
        "gpu_target_angular_velocity": (
            mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
            target_name,
        ),
        "gpu_wrist_linear_velocity": (
            mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            "right_wrist_yaw_link",
        ),
        "gpu_pelvis_linear_velocity": (
            mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            "pelvis",
        ),
        "gpu_pelvis_angular_velocity": (
            mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
            "pelvis",
        ),
    }
    distal_velocity_names = []
    for index, link_name in enumerate(RIGHT_DISTAL_LINK_NAMES):
        name = f"gpu_distal_linear_velocity_{index:02d}"
        velocity_sensors[name] = (mujoco.mjtSensor.mjSENS_FRAMELINVEL, link_name)
        distal_velocity_names.append(name)
    for name, (sensor_type, body_name) in velocity_sensors.items():
        spec.add_sensor(
            name=name,
            type=sensor_type,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname=body_name,
        )

    model = spec.compile()
    layout = GpuSensorLayout(
        object_force=tuple(_sensor_slice(model, name) for name in object_force_names),
        object_position=tuple(
            _sensor_slice(model, name) for name in object_position_names
        ),
        table_force=tuple(_sensor_slice(model, name) for name in table_force_names),
        fingertip_distance=tuple(
            tuple(_sensor_slice(model, name) for name in names)
            for names in distance_names
        ),
        target_linear_velocity=_sensor_slice(model, "gpu_target_linear_velocity"),
        target_angular_velocity=_sensor_slice(model, "gpu_target_angular_velocity"),
        wrist_linear_velocity=_sensor_slice(model, "gpu_wrist_linear_velocity"),
        distal_linear_velocity=tuple(
            _sensor_slice(model, name) for name in distal_velocity_names
        ),
        pelvis_linear_velocity=_sensor_slice(model, "gpu_pelvis_linear_velocity"),
        pelvis_angular_velocity=_sensor_slice(model, "gpu_pelvis_angular_velocity"),
    )
    return model, layout


def _augment_v2_sensors(
    bundle: FrozenAssetBundle,
) -> tuple[mujoco.MjModel, GpuV2SensorLayout]:
    """Append the contact, distance and velocity sensors required by 331-D v2."""

    scene = _bundle_file(bundle.root, bundle.manifest["scene_file"])
    spec = mujoco.MjSpec.from_file(str(scene))
    for index, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"gpu_geom_{index:04d}_{geom.parent.name}"
    named_model = spec.compile()
    roles = bundle.manifest["roles"]
    primary = roles.get("primary")
    destination = roles.get("destination")
    auxiliary = roles.get("auxiliary")
    if not primary:
        raise ValueError("V2 GPU sensors require a primary role")

    def add_contact(name: str, first: str, second: str) -> str:
        spec.add_sensor(
            name=name,
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname=first,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=second,
            intprm=[1 << 1, 3, 1],
        )
        return name

    primary_force_names: list[list[str]] = [[], []]
    for hand_index, links in enumerate(
        (LEFT_CONTACT_LINK_NAMES, RIGHT_CONTACT_LINK_NAMES)
    ):
        for link_index, link in enumerate(links):
            primary_force_names[hand_index].append(
                add_contact(
                    f"gpu_v2_primary_force_{hand_index}_{link_index}", link, primary
                )
            )

    auxiliary_force_names: list[list[str]] | None = None
    if auxiliary:
        auxiliary_force_names = [[], []]
        for hand_index, links in enumerate(
            (LEFT_CONTACT_LINK_NAMES, RIGHT_CONTACT_LINK_NAMES)
        ):
            for link_index, link in enumerate(links):
                auxiliary_force_names[hand_index].append(
                    add_contact(
                        f"gpu_v2_auxiliary_force_{hand_index}_{link_index}",
                        link,
                        auxiliary,
                    )
                )

    support_name = roles.get("support") or destination
    hand_support_names: list[list[str]] | None = None
    if support_name:
        hand_support_names = [[], []]
        for hand_index, links in enumerate(
            (LEFT_CONTACT_LINK_NAMES, RIGHT_CONTACT_LINK_NAMES)
        ):
            for link_index, link in enumerate(links):
                hand_support_names[hand_index].append(
                    add_contact(
                        f"gpu_v2_support_{hand_index}_{link_index}",
                        link,
                        support_name,
                    )
                )
    primary_destination_name = (
        add_contact("gpu_v2_primary_destination", primary, destination)
        if destination else None
    )
    primary_auxiliary_name = (
        add_contact("gpu_v2_primary_auxiliary", primary, auxiliary)
        if auxiliary else None
    )
    auxiliary_destination_name = (
        add_contact("gpu_v2_auxiliary_destination", auxiliary, destination)
        if auxiliary and destination else None
    )

    target_geoms = _subtree_geom_names(named_model, primary)
    distance_names: list[list[list[str]]] = [[], []]
    for hand_index, links in enumerate((LEFT_DISTAL_LINK_NAMES, RIGHT_DISTAL_LINK_NAMES)):
        for finger_index, link in enumerate(links):
            names = []
            for pair_index, (finger_geom, target_geom) in enumerate(
                (finger, target)
                for finger in _subtree_geom_names(named_model, link)
                for target in target_geoms
            ):
                name = f"gpu_v2_distance_{hand_index}_{finger_index}_{pair_index}"
                spec.add_sensor(
                    name=name,
                    type=mujoco.mjtSensor.mjSENS_GEOMDIST,
                    objtype=mujoco.mjtObj.mjOBJ_GEOM,
                    objname=finger_geom,
                    reftype=mujoco.mjtObj.mjOBJ_GEOM,
                    refname=target_geom,
                    cutoff=2.0,
                )
                names.append(name)
            if not names:
                raise ValueError(f"No v2 geom-distance pairs for {link}")
            distance_names[hand_index].append(names)

    velocity_bodies = {
        "pelvis": "pelvis",
        "left_hand": "left_wrist_yaw_link",
        "right_hand": "right_wrist_yaw_link",
        "primary": primary,
    }
    if destination:
        velocity_bodies["destination"] = destination
    if auxiliary:
        velocity_bodies["auxiliary"] = auxiliary
    linear_names, angular_names = {}, {}
    for role, body in velocity_bodies.items():
        linear = f"gpu_v2_{role}_linear_velocity"
        angular = f"gpu_v2_{role}_angular_velocity"
        for name, sensor_type in (
            (linear, mujoco.mjtSensor.mjSENS_FRAMELINVEL),
            (angular, mujoco.mjtSensor.mjSENS_FRAMEANGVEL),
        ):
            spec.add_sensor(
                name=name,
                type=sensor_type,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                objname=body,
            )
        linear_names[role] = linear
        angular_names[role] = angular

    model = spec.compile()
    def contact_slices(names: list[str]) -> tuple[slice, ...]:
        return tuple(_sensor_slice(model, name) for name in names)
    layout = GpuV2SensorLayout(
        primary_force=tuple(
            contact_slices(names) for names in primary_force_names
        ),
        auxiliary_force=(
            tuple(contact_slices(names) for names in auxiliary_force_names)
            if auxiliary_force_names is not None else None
        ),
        hand_support_force=(
            tuple(contact_slices(names) for names in hand_support_names)
            if hand_support_names is not None else None
        ),
        primary_destination_force=(
            _sensor_slice(model, primary_destination_name)
            if primary_destination_name is not None else None
        ),
        primary_auxiliary_force=(
            _sensor_slice(model, primary_auxiliary_name)
            if primary_auxiliary_name is not None else None
        ),
        auxiliary_destination_force=(
            _sensor_slice(model, auxiliary_destination_name)
            if auxiliary_destination_name is not None else None
        ),
        fingertip_distance=tuple(
            tuple(contact_slices(names) for names in hand)
            for hand in distance_names
        ),
        linear_velocity={
            role: _sensor_slice(model, name) for role, name in linear_names.items()
        },
        angular_velocity={
            role: _sensor_slice(model, name) for role, name in angular_names.items()
        },
    )
    return model, layout


def _physics_config(manifest: dict[str, Any]) -> MujocoCfg:
    physics = manifest["physics"]
    return MujocoCfg(
        timestep=float(physics["timestep"]),
        integrator=physics["integrator"],
        impratio=float(physics["impratio"]),
        cone=physics["cone"],
        jacobian=physics["jacobian"],
        solver=physics["solver"],
        iterations=int(physics["iterations"]),
        tolerance=float(physics["tolerance"]),
        ls_iterations=int(physics["ls_iterations"]),
        ls_tolerance=float(physics["ls_tolerance"]),
        ccd_iterations=int(physics["ccd_iterations"]),
        gravity=tuple(float(item) for item in physics["gravity"]),
    )


def _restore_reset_state(
    sim: Simulation,
    reset: dict[str, Any],
    env_ids: torch.Tensor | None = None,
) -> None:
    if env_ids is not None:
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise ValueError("env_ids must be a one-dimensional torch.long tensor")
        if str(env_ids.device) != sim.device:
            raise ValueError("env_ids must be on the simulation CUDA device")
    sim.reset(env_ids)
    indices = slice(None) if env_ids is None else env_ids
    for name in (
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "qacc_warmstart",
        "mocap_pos",
        "mocap_quat",
    ):
        values = reset[name]
        target = getattr(sim.data, name)
        if target.numel() == 0 and not values:
            continue
        source = torch.as_tensor(values, dtype=target.dtype, device=target.device)
        if source.shape != target.shape[1:]:
            raise ValueError(
                f"Reset {name} shape {tuple(source.shape)} != {tuple(target.shape[1:])}"
            )
        target[indices] = source
    sim.data.time[indices] = float(reset["time"])
    sim.forward()
    sensor_target = sim.data.sensordata
    sensor_source = torch.as_tensor(
        reset["sensordata"],
        dtype=sensor_target.dtype,
        device=sensor_target.device,
    )
    if sensor_source.ndim != 1 or sensor_source.shape[0] > sensor_target.shape[1]:
        raise ValueError("Reset sensordata shape mismatch")
    # Preserve the CPU fast-reset controller input for the first action.  The
    # next MuJoCo-Warp step recomputes sensors from GPU state as usual.
    sensor_target[indices, : sensor_source.shape[0]] = sensor_source


def _assert_cuda_residency(sim: Simulation, expected_device: str) -> None:
    if sim.device != expected_device or not sim.wp_device.is_cuda:
        raise RuntimeError("MuJoCo-Warp simulation is not on the requested CUDA device")
    for name in CUDA_DATA_FIELDS:
        value = getattr(sim.data, name)
        # mjlab exposes a zero-copy TorchArray proxy rather than subclassing
        # torch.Tensor; its tensor properties delegate to the CUDA view.
        if not bool(getattr(value, "is_cuda", False)):
            raise RuntimeError(f"Simulation data {name} is not a CUDA tensor")
        if str(value.device) != expected_device:
            raise RuntimeError(
                f"Simulation data {name} is on {value.device}, not {expected_device}"
            )


def build_gpu_simulation(
    config: MjlabPpoConfig,
    *,
    nconmax: int = 256,
    njmax: int = 1024,
) -> GpuSimulation:
    """Load, hash-check and instantiate a genuinely GPU-resident simulation."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    bundle = FrozenAssetBundle.load(config.asset_bundle, expected_task=config.task)
    object_contract = bundle.manifest.get("object_contract")
    if object_contract is not None:
        if config.reference_processed is None:
            raise ValueError("grasp_anything requires reference_processed")
        processed_root = Path(config.reference_processed).resolve()
        processed_manifest = json.loads(
            (processed_root / "manifest.json").read_text()
        )
        expected_reference_task = object_contract["reference_task"]
        if processed_manifest.get("task") != expected_reference_task:
            raise ValueError(
                "Object contract requires reference task "
                f"{expected_reference_task!r}, got "
                f"{processed_manifest.get('task')!r}"
            )
        processed_transform = processed_root / "action_transform.npz"
        if _sha256(processed_transform) != bundle.manifest.get(
            "action_transform_sha256"
        ):
            raise ValueError(
                "grasp_anything reference/action-transform SHA mismatch"
            )
    strict_episode = config.strict_reference_episode
    if strict_episode is not None:
        if bundle.manifest.get("base_episode") != strict_episode:
            raise ValueError(
                "Strict reference episode does not match the frozen asset "
                "base_episode"
            )
        if config.reference_processed is None:
            raise ValueError("Strict reference mode requires reference_processed")
        processed_root = Path(config.reference_processed).resolve()
        processed_manifest = json.loads(
            (processed_root / "manifest.json").read_text()
        )
        expected_processed_task = (
            object_contract["reference_task"]
            if object_contract is not None
            else config.task
        )
        if processed_manifest.get("task") != expected_processed_task:
            raise ValueError("Strict reference processed task mismatch")
        validate_strict_reference_manifest(processed_manifest, strict_episode)
        processed_transform = processed_root / "action_transform.npz"
        processed_sha256 = _sha256(processed_transform)
        if processed_manifest.get("action_transform_sha256") != processed_sha256:
            raise ValueError(
                "Strict reference processed action-transform SHA mismatch"
            )
        if bundle.manifest.get("action_transform_sha256") != processed_sha256:
            raise ValueError(
                "Strict reference processed/asset action-transform SHA mismatch"
            )
    model = bundle.model
    sensors = None
    if bundle.manifest["task_metadata"].get("task_schema_version") == 2:
        model, sensors = _augment_v2_sensors(bundle)
    elif bundle.manifest.get("controller") == "amo":
        model, sensors = _augment_legacy_sensors(bundle)
    sim = Simulation(
        num_envs=config.num_envs,
        cfg=SimulationCfg(
            nconmax=nconmax,
            njmax=njmax,
            ls_parallel=True,
            contact_sensor_maxmatch=64,
            mujoco=_physics_config(bundle.manifest),
        ),
        model=model,
        device=config.device,
    )
    _assert_cuda_residency(sim, config.device)
    _restore_reset_state(sim, bundle.manifest["reset"])
    _assert_cuda_residency(sim, config.device)
    return GpuSimulation(bundle=bundle, sim=sim, sensors=sensors)
