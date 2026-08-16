from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest

from simple.grasp_rl.grasp_anything import (
    SUPPORTED_REFERENCE_TASKS,
    GraspAnythingObjectContract,
    _add_target_only_support_extension,
    _extend_support_box,
    _replace_support_collision_with_thin_top,
    _source_free_joint_damping,
    audit_target_workspace_support,
    derive_grasp_anything_bundle,
)


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _contract() -> GraspAnythingObjectContract:
    return GraspAnythingObjectContract(
        object_id="Cup_6",
        source_mjcf_sha256="1" * 64,
        reference_task="xmove_pick",
        half_extents_m=(0.052, 0.048, 0.052),
        grip_width_m=0.083,
        mass_kg=0.25,
        upright_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        grasp_frame_position_m=(0.0, 0.0, 0.0),
        grasp_frame_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        maximum_grip_force_newtons=80.0,
        base_manifest_hash="2" * 64,
    )


def test_object_contract_round_trips_portable_metadata() -> None:
    contract = _contract()
    assert GraspAnythingObjectContract.from_metadata(contract.metadata()) == contract


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("object_id", "../cup", "portable"),
        ("grip_width_m", 0.1, "grip_width"),
        ("mass_kg", 0.01, "mass_kg"),
        ("upright_quaternion_wxyz", (0.0, 0.0, 0.0, 0.0), "non-zero"),
        ("grasp_frame_quaternion_wxyz", (2.0, 0.0, 0.0, 0.0), "unit"),
        ("source_mjcf_sha256", "z" * 64, "SHA256"),
        ("reference_task", "bend_pick", "reviewed V2 tasks"),
    ],
)
def test_object_contract_rejects_unreviewed_physics(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_contract(), **{field: value}).validate()


def test_converter_refuses_to_overwrite_nonempty_bundle(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    manifest = {"task": "xmove_pick"}
    manifest["manifest_hash"] = _json_hash(manifest)
    (base / "manifest.json").write_text(json.dumps(manifest))
    source = tmp_path / "object.xml"
    source.write_text("<mujoco/>")
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        derive_grasp_anything_bundle(
            base,
            source,
            output,
            object_id="Cup_6",
            grip_width_m=0.083,
        )
    assert marker.read_text() == "keep"


def test_empty_object_contract_reports_a_validation_error() -> None:
    with pytest.raises(ValueError, match="contract is malformed"):
        GraspAnythingObjectContract.from_metadata({})


def test_reference_task_allowlist_preserves_legacy_and_adds_bend_variant() -> None:
    assert SUPPORTED_REFERENCE_TASKS == {
        "xmove_pick",
        "xmove_bend_pick",
        "bend_pick_teleop",
    }
    assert replace(_contract(), reference_task="xmove_bend_pick").metadata()[
        "reference_task"
    ] == "xmove_bend_pick"


def test_source_free_joint_damping_is_preserved_and_validated() -> None:
    body = ET.fromstring('<body><joint type="free" damping="0.1"/></body>')
    assert _source_free_joint_damping(body) == pytest.approx(0.1)

    invalid = ET.fromstring('<body><joint type="free" damping="-0.1"/></body>')
    with pytest.raises(ValueError, match="damping"):
        _source_free_joint_damping(invalid)


def test_workspace_support_audit_rejects_table_edge_and_accepts_extension(
    tmp_path,
) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco><worldbody>
        <body name="table" pos="0.3 0 0.4">
          <geom name="table_geom" type="box" size="0.625 0.395 0.05"/>
        </body>
        <body name="object" pos="-0.2708 -0.0548 0.49">
          <joint type="free"/><geom type="sphere" size="0.03"/>
        </body>
        </worldbody></mujoco>"""
    )
    manifest = {
        "roles": {"primary": "object", "support": "table"},
        "object_contract": {"half_extents_m": [0.032, 0.034, 0.032]},
        "reset": {
            "qpos": [-0.2708, -0.0548, 0.49, 1.0, 0.0, 0.0, 0.0],
            "initial_object_pos": [-0.2708, -0.0548, 0.49],
        },
    }
    workspace = ((-0.2, 0.2), (-0.29, 0.11))

    with pytest.raises(ValueError, match="leaves its support"):
        audit_target_workspace_support(
            scene,
            manifest,
            translation_bounds_xy_m=workspace,
            required_margin_m=0.03,
        )

    _extend_support_box(
        scene,
        support_name="table",
        extensions_m=(0.25, 0.0, 0.05, 0.0),
    )
    audit = audit_target_workspace_support(
        scene,
        manifest,
        translation_bounds_xy_m=workspace,
        required_margin_m=0.03,
    )
    assert audit["minimum_edge_margin_m"] >= 0.03


def test_workspace_thin_top_preserves_surface_without_thick_robot_barrier(
    tmp_path,
) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco><worldbody>
        <body name="table" pos="0.3 0 0.4">
          <geom name="table_geom" type="box" size="0.625 0.395 0.05"/>
        </body>
        </worldbody></mujoco>"""
    )

    _replace_support_collision_with_thin_top(
        scene,
        support_name="table",
        extensions_m=(0.25, 0.0, 0.05, 0.0),
        half_height_m=0.005,
    )

    table = ET.parse(scene).getroot().find(".//body[@name='table']")
    assert table is not None
    visual, collision = table.findall("geom")
    assert visual.attrib["contype"] == "0"
    assert visual.attrib["conaffinity"] == "0"
    assert [float(value) for value in collision.attrib["size"].split()] == pytest.approx(
        [0.75, 0.42, 0.005]
    )
    assert [float(value) for value in collision.attrib["pos"].split()] == pytest.approx(
        [-0.125, -0.025, 0.045]
    )


def test_workspace_target_only_extension_preserves_robot_table_collision(
    tmp_path,
) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco><worldbody>
        <body name="table" pos="0.3 0 0.4">
          <geom name="table_geom" type="box" size="0.625 0.395 0.05"/>
        </body>
        <body name="object"><joint type="free"/>
          <geom name="visual" type="sphere" size="0.03" contype="0" conaffinity="0"/>
          <geom name="collision" type="sphere" size="0.03"/>
        </body>
        </worldbody></mujoco>"""
    )

    _add_target_only_support_extension(
        scene,
        support_name="table",
        target_name="object",
        extensions_m=(0.25, 0.0, 0.05, 0.0),
        half_height_m=0.005,
    )

    root = ET.parse(scene).getroot()
    table = root.find(".//body[@name='table']")
    target = root.find(".//body[@name='object']")
    assert table is not None and target is not None
    original, extension = table.findall("geom")
    assert original.get("contype", "1") == "1"
    assert original.get("conaffinity", "1") == "1"
    assert extension.attrib["contype"] == "2"
    assert extension.attrib["conaffinity"] == "0"
    assert target.findall("geom")[0].attrib["conaffinity"] == "0"
    assert target.findall("geom")[1].attrib["conaffinity"] == "3"
