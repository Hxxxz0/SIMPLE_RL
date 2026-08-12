from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from simple.grasp_rl.grasp_anything import (
    GraspAnythingObjectContract,
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
