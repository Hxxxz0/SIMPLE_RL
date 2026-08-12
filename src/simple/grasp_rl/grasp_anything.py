"""One-object frozen-asset contracts for GPU grasp-anything PPO.

This module intentionally derives a new bundle from an audited xmove bundle.
It never mutates the source bundle or the legacy CPU task, which keeps all
released tasks and checkpoints byte-for-byte compatible.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from simple.grasp_rl.state import _rotation_6d
from simple.grasp_rl.task_spec import get_task_spec

GRASP_ANYTHING_TASK = "grasp_anything"
OBJECT_CONTRACT_VERSION = 1
_OBJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float_vector(value: str | None, size: int, *, default: float = 0.0) -> np.ndarray:
    result = np.full(size, default, dtype=np.float64)
    if value:
        parsed = np.fromstring(value, sep=" ", dtype=np.float64)
        if parsed.size > size:
            raise ValueError(f"Expected at most {size} values, got {parsed.size}")
        result[: parsed.size] = parsed
    return result


def _set_float_vector(element: ET.Element, name: str, value: np.ndarray) -> None:
    element.set(name, " ".join(f"{float(item):.10g}" for item in value))


def _quat(value: tuple[float, float, float, float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("upright quaternion must contain four finite values")
    norm = float(np.linalg.norm(result))
    if norm < 1e-8:
        raise ValueError("upright quaternion must be non-zero")
    return result / norm


def _topology(model: mujoco.MjModel) -> dict[str, int]:
    return {
        name: int(getattr(model, name))
        for name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsensor")
    }


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    while current > 0 and current != ancestor_id:
        current = int(model.body_parentid[current])
    return current == ancestor_id


def _subtree_geoms(model: mujoco.MjModel, body_id: int) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(model.ngeom)
        if _is_descendant(model, int(model.geom_bodyid[index]), body_id)
    )


def _geom_world_corners(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> np.ndarray:
    center = np.asarray(model.geom_aabb[geom_id, :3], dtype=np.float64)
    half = np.asarray(model.geom_aabb[geom_id, 3:], dtype=np.float64)
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    local = center[None] + signs * half[None]
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    return local @ rotation.T + np.asarray(data.geom_xpos[geom_id])


def _collision_geoms(model: mujoco.MjModel, body_id: int) -> tuple[int, ...]:
    return tuple(
        geom_id
        for geom_id in _subtree_geoms(model, body_id)
        if int(model.geom_contype[geom_id]) != 0
        or int(model.geom_conaffinity[geom_id]) != 0
    )


def _world_bounds(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    if not geom_ids:
        raise ValueError("Object contains no collision geometry")
    points = np.concatenate(
        [_geom_world_corners(model, data, geom_id) for geom_id in geom_ids], axis=0
    )
    return points.min(axis=0), points.max(axis=0)


def _local_half_extents(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    geom_ids: tuple[int, ...],
) -> np.ndarray:
    points = np.concatenate(
        [_geom_world_corners(model, data, geom_id) for geom_id in geom_ids], axis=0
    )
    body_pos = np.asarray(data.xpos[body_id])
    body_rot = np.asarray(data.xmat[body_id]).reshape(3, 3)
    local = (points - body_pos[None]) @ body_rot
    return 0.5 * (local.max(axis=0) - local.min(axis=0))


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"Body {name!r} is absent from scene")


def _copy_source_assets(
    source_root: ET.Element,
    source_dir: Path,
    destination_root: ET.Element,
    destination_dir: Path,
    *,
    prefix: str,
    scale: float,
) -> tuple[dict[tuple[str, str], str], list[dict[str, str | int]]]:
    source_assets = source_root.find("asset")
    if source_assets is None:
        return {}, []
    destination_assets = destination_root.find("asset")
    if destination_assets is None:
        destination_assets = ET.SubElement(destination_root, "asset")
    name_map: dict[tuple[str, str], str] = {
        (source.tag, old_name): f"{prefix}_{old_name}"
        for source in source_assets
        if (old_name := source.get("name"))
    }
    records: list[dict[str, str | int]] = []
    for source in source_assets:
        copied = deepcopy(source)
        copied.attrib.pop("class", None)
        old_name = source.get("name")
        if old_name:
            copied.set("name", name_map[(source.tag, old_name)])
        texture = source.get("texture")
        if texture:
            try:
                copied.set("texture", name_map[("texture", texture)])
            except KeyError as error:
                raise ValueError(
                    f"Material references missing texture {texture!r}"
                ) from error
        file_value = source.get("file")
        if file_value:
            source_path = (source_dir / file_value).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"Object asset is missing: {source_path}")
            digest = _sha256(source_path)
            relative = f"assets/{digest[:16]}_{source_path.name}"
            destination_path = destination_dir / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if not destination_path.exists():
                shutil.copy2(source_path, destination_path)
            copied.set("file", relative)
            records.append(
                {
                    "source": str(source_path),
                    "bundle_path": relative,
                    "sha256": digest,
                    "bytes": source_path.stat().st_size,
                }
            )
        if source.tag == "mesh" and not math.isclose(scale, 1.0):
            mesh_scale = _float_vector(source.get("scale"), 3, default=1.0)
            _set_float_vector(copied, "scale", mesh_scale * scale)
        destination_assets.append(copied)
    unique = {str(record["bundle_path"]): record for record in records}
    return name_map, [unique[name] for name in sorted(unique)]


def _materialize_object_subtree(
    source_body: ET.Element,
    *,
    name_map: dict[tuple[str, str], str],
    prefix: str,
    scale: float,
    target_mass: float,
) -> list[ET.Element]:
    children = [
        deepcopy(child)
        for child in source_body
        if child.tag not in ("joint", "site", "inertial")
    ]
    # AI2-THOR receptacle/debug sites are bright red visualization helpers,
    # not physical object geometry.  They would otherwise pollute PPO videos.
    for node in children:
        for parent in node.iter():
            for child in list(parent):
                if child.tag in ("site", "inertial"):
                    parent.remove(child)
    if any(child.tag == "joint" for node in children for child in node.iter()):
        raise ValueError("V1 grasp-anything objects must have fixed child topology")
    collision_geoms = [
        geom
        for node in children
        for geom in node.iter("geom")
        if "VISUAL" not in geom.get("class", "").upper()
        and not (
            geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        )
    ]
    if not collision_geoms:
        raise ValueError("Object source has no physical collision geoms")
    mass_per_geom = target_mass / len(collision_geoms)
    for node in children:
        for element in node.iter():
            source_class = element.get("class", "").upper()
            old_name = element.get("name")
            if old_name:
                element.set("name", f"{prefix}_{old_name}")
            element.attrib.pop("class", None)
            for attribute in ("pos",):
                if attribute in element.attrib:
                    _set_float_vector(
                        element,
                        attribute,
                        _float_vector(element.get(attribute), 3) * scale,
                    )
            if "fromto" in element.attrib:
                values = np.fromstring(element.get("fromto", ""), sep=" ")
                if values.shape != (6,):
                    raise ValueError("Object geom fromto must contain six values")
                _set_float_vector(element, "fromto", values * scale)
            if element.tag in ("geom", "site") and "size" in element.attrib:
                values = np.fromstring(element.get("size", ""), sep=" ")
                _set_float_vector(element, "size", values * scale)
            if element.tag != "geom":
                continue
            mesh = element.get("mesh")
            material = element.get("material")
            if mesh:
                element.set("mesh", name_map[("mesh", mesh)])
            if material:
                element.set("material", name_map[("material", material)])
            visual = "VISUAL" in source_class or (
                element.get("contype") == "0"
                and element.get("conaffinity") == "0"
            )
            if visual:
                element.set("contype", "0")
                element.set("conaffinity", "0")
                element.set("group", "1")
                element.set("density", "0")
                element.attrib.pop("mass", None)
            else:
                element.set("contype", "1")
                element.set("conaffinity", "1")
                element.set("group", "4")
                element.set("condim", "4")
                element.set("friction", "0.8 0.05 0.005")
                element.set("solref", "0.005 2")
                element.set("solimp", "0.9 0.99 0.001")
                element.set("mass", f"{mass_per_geom:.10g}")
                element.attrib.pop("density", None)
    return children


def _replace_primary_object(
    scene_path: Path,
    source_root: ET.Element,
    source_body: ET.Element,
    source_dir: Path,
    output: Path,
    *,
    primary_name: str,
    prefix: str,
    scale: float,
    mass_kg: float,
    upright: np.ndarray,
) -> list[dict[str, str | int]]:
    """Replace one free body's geometry while preserving all state addresses."""

    scene_tree = ET.parse(scene_path)
    scene_root = scene_tree.getroot()
    worldbody = scene_root.find("worldbody")
    if worldbody is not None:
        for child in list(worldbody):
            if child.tag == "site" and child.get("name") == "com_marker":
                # The inherited xmove scene contains a bright red world-space
                # debug marker beside the robot's feet.  It is not referenced
                # by sensors or physics and should not pollute object-policy
                # review videos.  Source/legacy bundles remain untouched.
                worldbody.remove(child)
    target_body = _find_body(scene_root, primary_name)
    target_joints = [child for child in target_body if child.tag == "joint"]
    if len(target_joints) != 1 or target_joints[0].get("type") != "free":
        raise ValueError("Base primary body must contain one free joint")
    for child in list(target_body):
        if child is not target_joints[0]:
            target_body.remove(child)

    name_map, asset_records = _copy_source_assets(
        source_root,
        source_dir,
        scene_root,
        output,
        prefix=prefix,
        scale=scale,
    )
    object_children = _materialize_object_subtree(
        source_body,
        name_map=name_map,
        prefix=prefix,
        scale=scale,
        target_mass=mass_kg,
    )
    for child in object_children:
        target_body.append(child)
    _set_float_vector(target_body, "quat", upright)
    scene_tree.write(scene_path, encoding="unicode")
    return asset_records


def _derive_render_sidecar(
    base: Path,
    output: Path,
    *,
    new_manifest_hash: str,
    physics_topology: dict[str, int],
    source_root: ET.Element,
    source_body: ET.Element,
    source_dir: Path,
    primary_name: str,
    prefix: str,
    scale: float,
    mass_kg: float,
    upright: np.ndarray,
) -> dict[str, object] | None:
    """Keep an existing visual sidecar in lock-step with derived physics."""

    base_scene = base / "render_scene.xml"
    base_manifest_path = base / "render_manifest.json"
    if not base_scene.exists() and not base_manifest_path.exists():
        (output / "render_scene.xml").unlink(missing_ok=True)
        (output / "render_manifest.json").unlink(missing_ok=True)
        return None
    if not base_scene.is_file() or not base_manifest_path.is_file():
        raise ValueError("Base render sidecar is incomplete")

    base_render_manifest = json.loads(base_manifest_path.read_text())
    unhashed = deepcopy(base_render_manifest)
    expected_hash = unhashed.pop("manifest_hash", None)
    if expected_hash is None or _json_hash(unhashed) != expected_hash:
        raise ValueError("Base render sidecar manifest hash mismatch")
    if _sha256(base_scene) != base_render_manifest.get("scene_sha256"):
        raise ValueError("Base render sidecar scene hash mismatch")

    render_scene = output / "render_scene.xml"
    asset_records = _replace_primary_object(
        render_scene,
        source_root,
        source_body,
        source_dir,
        output,
        primary_name=primary_name,
        prefix=prefix,
        scale=scale,
        mass_kg=mass_kg,
        upright=upright,
    )
    model = mujoco.MjModel.from_xml_path(str(render_scene))
    topology = _topology(model)
    for field in ("nq", "nv", "nu", "njnt", "nsensor"):
        if topology[field] != int(base_render_manifest["model"][field]):
            raise ValueError(
                f"Derived render scene changed {field}: "
                f"{topology[field]} != {base_render_manifest['model'][field]}"
            )
    if topology["nbody"] != physics_topology["nbody"]:
        raise ValueError(
            "Derived render scene body layout does not match derived physics: "
            f"{topology['nbody']} != {physics_topology['nbody']}"
        )

    assets_by_path = {
        str(item["bundle_path"]): item
        for item in [*base_render_manifest["assets"], *asset_records]
    }
    sidecar = deepcopy(base_render_manifest)
    sidecar.pop("manifest_hash", None)
    sidecar.update(
        {
            "base_manifest_hash": new_manifest_hash,
            "scene_file": render_scene.name,
            "scene_sha256": _sha256(render_scene),
            "assets": [assets_by_path[name] for name in sorted(assets_by_path)],
            "model": topology,
            "ncam": int(model.ncam),
            "nlight": int(model.nlight),
        }
    )
    sidecar["manifest_hash"] = _json_hash(sidecar)
    temporary = output / ".render_manifest.tmp.json"
    temporary.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    temporary.replace(output / "render_manifest.json")
    return sidecar


def _copy_reset_state(
    base_manifest: dict,
    model: mujoco.MjModel,
    *,
    primary_name: str,
    destination_name: str,
    upright_quaternion: np.ndarray,
    table_clearance: float,
) -> tuple[dict, np.ndarray]:
    reset = base_manifest["reset"]
    data = mujoco.MjData(model)
    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
        destination = getattr(data, name)
        source = np.asarray(reset[name], dtype=np.float64)
        if destination.shape != source.shape:
            raise ValueError(
                f"Derived object changed {name} shape {destination.shape} != {source.shape}"
            )
        destination[:] = source
    if model.nmocap:
        data.mocap_pos[:] = np.asarray(reset["mocap_pos"], dtype=np.float64)
        data.mocap_quat[:] = np.asarray(reset["mocap_quat"], dtype=np.float64)
    data.time = float(reset["time"])

    primary_body = model.body(primary_name).id
    primary_joint = next(
        (
            joint_id
            for joint_id in range(model.njnt)
            if int(model.jnt_bodyid[joint_id]) == primary_body
            and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ),
        None,
    )
    if primary_joint is None:
        raise ValueError("Primary object must have one free joint")
    qpos_adr = int(model.jnt_qposadr[primary_joint])
    previous_position = np.asarray(reset["initial_object_pos"], dtype=np.float64)
    data.qpos[qpos_adr : qpos_adr + 3] = (previous_position[0], previous_position[1], 1.0)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = upright_quaternion
    qvel_adr = int(model.jnt_dofadr[primary_joint])
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    destination_body = model.body(destination_name).id
    table_geoms = _collision_geoms(model, destination_body)
    _, table_high = _world_bounds(model, data, table_geoms)
    object_geoms = _collision_geoms(model, primary_body)
    object_low, _ = _world_bounds(model, data, object_geoms)
    data.qpos[qpos_adr + 2] += float(table_high[2] + table_clearance - object_low[2])
    mujoco.mj_forward(model, data)

    extents = _local_half_extents(model, data, primary_body, object_geoms)
    pelvis = model.body("pelvis").id
    pelvis_rotation = np.asarray(data.xmat[pelvis]).reshape(3, 3)
    object_rotation = np.asarray(data.xmat[primary_body]).reshape(3, 3)
    object_position = np.asarray(data.xpos[primary_body]).copy()
    actor_observation = np.asarray(reset["actor_observation"], dtype=np.float64).copy()
    actor_observation[163:166] = pelvis_rotation.T @ (
        object_position - np.asarray(data.xpos[pelvis])
    )
    actor_observation[166:172] = _rotation_6d(pelvis_rotation.T @ object_rotation)
    actor_observation[178:181] = extents

    derived = {
        "qpos": np.asarray(data.qpos).tolist(),
        "qvel": np.asarray(data.qvel).tolist(),
        "act": np.asarray(data.act).tolist(),
        "ctrl": np.asarray(data.ctrl).tolist(),
        "qacc_warmstart": np.asarray(data.qacc_warmstart).tolist(),
        "sensordata": np.asarray(data.sensordata).tolist(),
        "time": float(data.time),
        "mocap_pos": np.asarray(data.mocap_pos).tolist(),
        "mocap_quat": np.asarray(data.mocap_quat).tolist(),
        "previous_physical_action": reset["previous_physical_action"],
        "actor_observation": actor_observation.tolist(),
        "initial_object_pos": object_position.tolist(),
        "goal_pos": reset["goal_pos"],
    }
    return derived, extents


@dataclass(frozen=True)
class GraspAnythingObjectContract:
    object_id: str
    source_mjcf_sha256: str
    reference_task: str
    half_extents_m: tuple[float, float, float]
    grip_width_m: float
    mass_kg: float
    upright_quaternion_wxyz: tuple[float, float, float, float]
    grasp_frame_position_m: tuple[float, float, float]
    grasp_frame_quaternion_wxyz: tuple[float, float, float, float]
    maximum_grip_force_newtons: float
    base_manifest_hash: str
    schema_version: int = OBJECT_CONTRACT_VERSION

    def validate(self) -> None:
        if self.schema_version != OBJECT_CONTRACT_VERSION:
            raise ValueError("Unsupported grasp-anything object contract version")
        if not _OBJECT_ID.fullmatch(self.object_id):
            raise ValueError("object_id must be a portable filesystem identifier")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_mjcf_sha256) is None:
            raise ValueError("source MJCF SHA256 is malformed")
        if re.fullmatch(r"[0-9a-f]{64}", self.base_manifest_hash) is None:
            raise ValueError("base manifest hash is malformed")
        if not self.reference_task:
            raise ValueError("reference_task must be non-empty")
        extents = np.asarray(self.half_extents_m, dtype=np.float64)
        if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents <= 0):
            raise ValueError("object half extents must be positive and finite")
        if not 0.035 <= self.grip_width_m <= 0.09:
            raise ValueError("grip_width_m must be in the reviewed 0.035..0.09 m range")
        if not 0.05 <= self.mass_kg <= 1.5:
            raise ValueError("mass_kg must be in the reviewed 0.05..1.5 kg range")
        if not 5.0 <= self.maximum_grip_force_newtons <= 250.0:
            raise ValueError("maximum grip force must be in 5..250 N")
        for name, value in (
            ("upright quaternion", self.upright_quaternion_wxyz),
            ("grasp frame quaternion", self.grasp_frame_quaternion_wxyz),
        ):
            normalized = _quat(value)
            if not math.isclose(
                float(np.linalg.norm(value)), 1.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError(f"{name} must be unit length")
            if not np.isfinite(normalized).all():
                raise ValueError(f"{name} is malformed")
        position = np.asarray(self.grasp_frame_position_m, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("grasp frame position must contain three finite values")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "object_id": self.object_id,
            "source_mjcf_sha256": self.source_mjcf_sha256,
            "reference_task": self.reference_task,
            "half_extents_m": list(self.half_extents_m),
            "grip_width_m": self.grip_width_m,
            "mass_kg": self.mass_kg,
            "upright_quaternion_wxyz": list(self.upright_quaternion_wxyz),
            "grasp_frame_position_m": list(self.grasp_frame_position_m),
            "grasp_frame_quaternion_wxyz": list(
                self.grasp_frame_quaternion_wxyz
            ),
            "maximum_grip_force_newtons": self.maximum_grip_force_newtons,
            "base_manifest_hash": self.base_manifest_hash,
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, object]) -> GraspAnythingObjectContract:
        try:
            contract = cls(
                object_id=str(payload["object_id"]),
                source_mjcf_sha256=str(payload["source_mjcf_sha256"]),
                reference_task=str(payload["reference_task"]),
                half_extents_m=tuple(float(x) for x in payload["half_extents_m"]),
                grip_width_m=float(payload["grip_width_m"]),
                mass_kg=float(payload["mass_kg"]),
                upright_quaternion_wxyz=tuple(
                    float(x) for x in payload["upright_quaternion_wxyz"]
                ),
                grasp_frame_position_m=tuple(
                    float(x) for x in payload["grasp_frame_position_m"]
                ),
                grasp_frame_quaternion_wxyz=tuple(
                    float(x) for x in payload["grasp_frame_quaternion_wxyz"]
                ),
                maximum_grip_force_newtons=float(
                    payload["maximum_grip_force_newtons"]
                ),
                base_manifest_hash=str(payload["base_manifest_hash"]),
                schema_version=int(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("grasp-anything object contract is malformed") from error
        contract.validate()
        return contract


def validate_grasp_anything_bundle(path: str | Path) -> dict[str, object]:
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("task") != GRASP_ANYTHING_TASK:
        raise ValueError("Bundle is not a grasp_anything bundle")
    contract = GraspAnythingObjectContract.from_metadata(
        manifest["object_contract"]
    )
    if manifest.get("object_id") != contract.object_id:
        raise ValueError("Bundle object_id does not match its object contract")
    scene = root / manifest["scene_file"]
    if _sha256(scene) != manifest["scene_sha256"]:
        raise ValueError("Derived scene hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if _topology(model) != manifest["model"]:
        raise ValueError("Derived scene topology does not match manifest")
    primary_name = manifest["roles"]["primary"]
    primary_body = model.body(primary_name).id
    free_joints = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_bodyid[joint_id]) == primary_body
        and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joints) != 1:
        raise ValueError("Object must have exactly one root free joint")
    actual_mass = float(model.body_subtreemass[primary_body])
    if not math.isclose(actual_mass, contract.mass_kg, rel_tol=2e-3, abs_tol=2e-4):
        raise ValueError(
            f"Object mass {actual_mass:.6g} does not match {contract.mass_kg:.6g}"
        )
    xml_root = ET.parse(scene).getroot()
    for equality in xml_root.findall(".//equality/*"):
        if primary_name in equality.attrib.values():
            raise ValueError("Object equality/weld attachment is forbidden")
    for geom_id in _collision_geoms(model, primary_body):
        if float(model.geom_friction[geom_id, 0]) > 2.0:
            raise ValueError("Object sliding friction exceeds the anti-adhesion limit")
    render_manifest_path = root / "render_manifest.json"
    render_scene_path = root / "render_scene.xml"
    render_manifest_hash = None
    if render_manifest_path.exists() != render_scene_path.exists():
        raise ValueError("Render sidecar is incomplete")
    if render_manifest_path.is_file():
        render_manifest = json.loads(render_manifest_path.read_text())
        render_unhashed = deepcopy(render_manifest)
        render_manifest_hash = render_unhashed.pop("manifest_hash", None)
        if (
            render_manifest_hash is None
            or _json_hash(render_unhashed) != render_manifest_hash
        ):
            raise ValueError("Render sidecar manifest hash mismatch")
        if render_manifest.get("base_manifest_hash") != manifest["manifest_hash"]:
            raise ValueError("Render sidecar belongs to a different physics bundle")
        declared_scene = (root / render_manifest["scene_file"]).resolve()
        if (
            not declared_scene.is_relative_to(root)
            or declared_scene != render_scene_path
            or _sha256(declared_scene) != render_manifest["scene_sha256"]
        ):
            raise ValueError("Render sidecar scene hash mismatch")
        for asset in render_manifest["assets"]:
            asset_path = (root / asset["bundle_path"]).resolve()
            if (
                not asset_path.is_relative_to(root)
                or not asset_path.is_file()
                or _sha256(asset_path) != asset["sha256"]
            ):
                raise ValueError(f"Render sidecar asset hash mismatch: {asset_path}")
        render_model = mujoco.MjModel.from_xml_path(str(declared_scene))
        if _topology(render_model) != render_manifest["model"]:
            raise ValueError("Render sidecar topology does not match its manifest")
        for field in ("nq", "nv", "nu", "nbody", "njnt", "nsensor"):
            if int(getattr(render_model, field)) != int(getattr(model, field)):
                raise ValueError(
                    f"Render sidecar {field} does not match derived physics"
                )
        if (
            int(render_model.ncam) != int(render_manifest["ncam"])
            or int(render_model.nlight) != int(render_manifest["nlight"])
        ):
            raise ValueError("Render sidecar camera/light count mismatch")
    return {
        "task": GRASP_ANYTHING_TASK,
        "object_id": contract.object_id,
        "mass_kg": actual_mass,
        "half_extents_m": list(contract.half_extents_m),
        "grip_width_m": contract.grip_width_m,
        "reference_task": contract.reference_task,
        "manifest_hash": manifest["manifest_hash"],
        "render_manifest_hash": render_manifest_hash,
    }


def derive_grasp_anything_bundle(
    base_bundle: str | Path,
    source_mjcf: str | Path,
    output_dir: str | Path,
    *,
    object_id: str,
    grip_width_m: float,
    mass_kg: float = 0.25,
    scale: float = 1.0,
    upright_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    grasp_frame_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    grasp_frame_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    maximum_grip_force_newtons: float = 80.0,
    table_clearance_m: float = 0.002,
) -> dict[str, object]:
    """Derive a portable, immutable one-object bundle from an xmove bundle."""

    if not _OBJECT_ID.fullmatch(object_id):
        raise ValueError("object_id must be a portable filesystem identifier")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    if not 0.0 <= table_clearance_m <= 0.01:
        raise ValueError("table clearance must be in 0..0.01 m")
    source = Path(source_mjcf).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    base = Path(base_bundle).resolve()
    base_manifest = json.loads((base / "manifest.json").read_text())
    if base_manifest.get("task") != "xmove_pick":
        raise ValueError("V1 requires an audited xmove_pick base bundle")
    unhashed = deepcopy(base_manifest)
    expected_base_hash = unhashed.pop("manifest_hash", None)
    if expected_base_hash is None or _json_hash(unhashed) != expected_base_hash:
        raise ValueError("Base bundle manifest hash mismatch")

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        shutil.copytree(base, output)
    else:
        shutil.copytree(base, output, dirs_exist_ok=True)

    scene_path = output / base_manifest["scene_file"]
    source_tree = ET.parse(source)
    source_root = source_tree.getroot()
    source_world = source_root.find("worldbody")
    if source_world is None:
        raise ValueError("Object MJCF has no worldbody")
    source_bodies = source_world.findall("body")
    if len(source_bodies) != 1:
        raise ValueError("Object MJCF must expose exactly one root body")
    source_body = source_bodies[0]
    source_free = [
        child
        for child in source_body
        if child.tag == "joint" and child.get("type", "hinge") == "free"
    ]
    if len(source_free) != 1:
        raise ValueError("Object MJCF root must contain exactly one free joint")

    primary_name = base_manifest["roles"]["primary"]
    prefix = f"ga_{object_id}"
    upright = _quat(upright_quaternion_wxyz)
    asset_records = _replace_primary_object(
        scene_path,
        source_root,
        source_body,
        source.parent,
        output,
        primary_name=primary_name,
        prefix=prefix,
        scale=scale,
        mass_kg=mass_kg,
        upright=upright,
    )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if (model.nq, model.nv, model.nu, model.nsensor) != (
        base_manifest["model"]["nq"],
        base_manifest["model"]["nv"],
        base_manifest["model"]["nu"],
        base_manifest["model"]["nsensor"],
    ):
        raise ValueError("Derived object changed controller/state topology")
    reset, extents = _copy_reset_state(
        base_manifest,
        model,
        primary_name=primary_name,
        destination_name=base_manifest["roles"]["destination"],
        upright_quaternion=upright,
        table_clearance=table_clearance_m,
    )

    task_spec = get_task_spec(GRASP_ANYTHING_TASK)
    contract = GraspAnythingObjectContract(
        object_id=object_id,
        source_mjcf_sha256=_sha256(source),
        reference_task=str(base_manifest["task"]),
        half_extents_m=tuple(float(value) for value in extents),
        grip_width_m=float(grip_width_m),
        mass_kg=float(mass_kg),
        upright_quaternion_wxyz=tuple(float(value) for value in upright),
        grasp_frame_position_m=tuple(float(value) for value in grasp_frame_position_m),
        grasp_frame_quaternion_wxyz=tuple(
            float(value) for value in _quat(grasp_frame_quaternion_wxyz)
        ),
        maximum_grip_force_newtons=float(maximum_grip_force_newtons),
        base_manifest_hash=str(base_manifest["manifest_hash"]),
    )
    contract.validate()
    manifest = deepcopy(base_manifest)
    manifest.pop("manifest_hash", None)
    manifest.pop("render_bundle", None)
    manifest.update(
        {
            "task": GRASP_ANYTHING_TASK,
            "task_metadata": task_spec.metadata(),
            "task_spec_hash": _json_hash(task_spec.metadata()),
            "source_task": base_manifest["task"],
            "object_id": object_id,
            "object_contract": contract.metadata(),
            "target_object": f"molmospaces:{object_id}",
            "scene_sha256": _sha256(scene_path),
            "assets": sorted(
                [*base_manifest["assets"], *asset_records],
                key=lambda item: str(item["bundle_path"]),
            ),
            "model": _topology(model),
            "reset": reset,
            "reward_spec": task_spec.metadata()["spec"],
        }
    )
    manifest["reward_hash"] = _json_hash(manifest["reward_spec"])
    manifest["manifest_hash"] = _json_hash(manifest)
    temporary = output / ".manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(output / "manifest.json")
    _derive_render_sidecar(
        base,
        output,
        new_manifest_hash=manifest["manifest_hash"],
        physics_topology=manifest["model"],
        source_root=source_root,
        source_body=source_body,
        source_dir=source.parent,
        primary_name=primary_name,
        prefix=prefix,
        scale=scale,
        mass_kg=mass_kg,
        upright=upright,
    )
    validate_grasp_anything_bundle(output)
    return manifest
