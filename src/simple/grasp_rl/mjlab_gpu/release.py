"""Portable release-manifest verification for mjlab GPU PPO artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_artifact(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise TypeError("release artifact path must be a string")
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts:
        raise ValueError(f"release artifact path is not portable: {value}")
    path = (root / Path(*logical.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"release artifact escapes its root: {value}")
    return path


def verify_ppo_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("mjlab_gpu_metadata")
    integrity = payload.get("ppo_integrity")
    if not isinstance(metadata, dict) or not isinstance(integrity, dict):
        raise TypeError(f"{path.name} is not an audited mjlab GPU checkpoint")
    if metadata.get("config", {}).get("backend") != "mjlab_mujoco_warp":
        raise ValueError(f"{path.name} does not use the mjlab MuJoCo-Warp backend")
    latest = integrity.get("latest_record")
    if not isinstance(latest, dict):
        raise TypeError(f"{path.name} has no PPO integrity record")
    if latest.get("algorithm") != "rsl_rl.algorithms.ppo.PPO":
        raise ValueError(f"{path.name} was not trained by RSL-RL PPO")
    if not latest.get("on_policy") or latest.get("rollout_reused"):
        raise ValueError(f"{path.name} failed the fresh on-policy audit")
    for name in (
        "transitions",
        "optimizer_steps",
        "actor_parameter_delta_l2",
        "critic_parameter_delta_l2",
    ):
        if float(latest.get(name, 0.0)) <= 0.0:
            raise ValueError(f"{path.name} has invalid PPO integrity field {name}")
    return {
        "sha256": sha256_file(path),
        "iteration": int(payload.get("iter", -1)),
        "total_transitions": int(integrity.get("total_transitions", 0)),
        "latest_record": latest,
    }


def verify_release(release_dir: str | Path) -> dict[str, Any]:
    root = Path(release_dir).resolve()
    manifest_path = root / "release.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported release manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release manifest contains no artifacts")
    checked = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise TypeError("release artifact entry must be an object")
        path = _relative_artifact(root, artifact.get("path"))
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(artifact.get("bytes", -1)):
            raise ValueError(f"release artifact size mismatch: {path.name}")
        digest = sha256_file(path)
        if digest != artifact.get("sha256"):
            raise ValueError(f"release artifact SHA256 mismatch: {path.name}")
        checked.append(str(path.relative_to(root)))
    checkpoint = _relative_artifact(root, manifest.get("checkpoint"))
    checkpoint_audit = verify_ppo_checkpoint(checkpoint)
    if checkpoint_audit["sha256"] != manifest.get("checkpoint_sha256"):
        raise ValueError("release checkpoint SHA256 disagrees with manifest")
    return {
        "release": str(root),
        "release_id": manifest.get("release_id"),
        "artifacts_verified": len(checked),
        "checkpoint": checkpoint_audit,
    }
