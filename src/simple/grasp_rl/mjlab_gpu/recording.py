"""Successful offscreen videos from the real mjlab GPU policy loop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_provenance(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    gpu = payload.get("mjlab_gpu_metadata", {})
    integrity = payload.get("ppo_integrity", {})
    latest = integrity.get("latest_record", {})
    if gpu.get("config", {}).get("backend") != "mjlab_mujoco_warp":
        raise ValueError("Video recording requires an mjlab GPU checkpoint")
    if (
        not latest.get("on_policy")
        or latest.get("algorithm") != "rsl_rl.algorithms.ppo.PPO"
    ):
        raise ValueError("Video recording requires audited on-policy PPO")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "backend": "mjlab_mujoco_warp",
        "ppo_integrity": integrity,
    }


def _episode_randomization(env: GpuGraspVecEnv) -> list[dict[str, Any]]:
    randomizer = env.randomizer
    translations = randomizer.target_translation_xy.detach().cpu().tolist()
    yaws = randomizer.target_yaw.detach().cpu().tolist()
    delays = randomizer.action_delay_steps.detach().cpu().tolist()
    rows = env.reference.episode_rows.detach().cpu().tolist()
    return [
        {
            "target_translation_xy": translations[index],
            "target_yaw": yaws[index],
            "action_delay_steps": delays[index],
            "reference_episode_row": rows[index],
        }
        for index in range(env.num_envs)
    ]


def _render_model(env: GpuGraspVecEnv) -> tuple[mujoco.MjModel, str]:
    root = env.gpu.bundle.root
    manifest_path = root / "render_manifest.json"
    if not manifest_path.is_file():
        return env.gpu.sim.mj_model, "gpu_physics_fallback"
    manifest = json.loads(manifest_path.read_text())
    unhashed = dict(manifest)
    expected_hash = unhashed.pop("manifest_hash", None)
    actual_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("Render sidecar manifest hash mismatch")
    if manifest.get("base_manifest_hash") != env.gpu.bundle.manifest["manifest_hash"]:
        raise ValueError("Render sidecar belongs to a different physics bundle")
    scene = (root / manifest["scene_file"]).resolve()
    if not scene.is_relative_to(root) or _sha256(scene) != manifest["scene_sha256"]:
        raise ValueError("Render sidecar scene hash mismatch")
    for asset in manifest["assets"]:
        asset_path = (root / asset["bundle_path"]).resolve()
        if (
            not asset_path.is_relative_to(root)
            or _sha256(asset_path) != asset["sha256"]
        ):
            raise ValueError(f"Render sidecar asset hash mismatch: {asset_path}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    physics = env.gpu.sim.mj_model
    render_layout = [
        (
            model.joint(i).name,
            int(model.jnt_type[i]),
            int(model.jnt_qposadr[i]),
            int(model.jnt_dofadr[i]),
        )
        for i in range(model.njnt)
    ]
    physics_layout = [
        (
            physics.joint(i).name,
            int(physics.jnt_type[i]),
            int(physics.jnt_qposadr[i]),
            int(physics.jnt_dofadr[i]),
        )
        for i in range(physics.njnt)
    ]
    if model.nq != physics.nq or render_layout != physics_layout:
        raise ValueError("Render sidecar state layout does not match GPU physics")
    return model, "full_visual_sidecar"


def _write_video(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    output: Path,
    *,
    task: str,
    target_body: str,
    width: int,
    height: int,
    fps: int,
) -> str:
    import imageio.v2 as imageio

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera_names = tuple(model.camera(index).name for index in range(model.ncam))
    if "front_stereo_left" in camera_names:
        camera: str | mujoco.MjvCamera = "front_stereo_left"
        model.cam_fovy[model.camera(camera).id] = 40.0
        camera_label = camera
    else:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth = 135.0 if task == "tabletop_grasp" else 165.0
        camera.elevation = -18.0
        camera.distance = 1.8 if task == "tabletop_grasp" else 2.0
        target_id = model.body(target_body).id
        pelvis_id = model.body("pelvis").id
        data.qpos[:] = qpos[0]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = 0.5 * data.xpos[target_id] + 0.5 * data.xpos[pelvis_id]
        camera.lookat[2] += 0.15
        camera_label = "free_full_robot"
    writer = imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for state in qpos:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    return camera_label


@torch.inference_mode()
def record_success_videos(
    env: GpuGraspVecEnv,
    actor: torch.nn.Module,
    checkpoint: Path,
    output_dir: Path,
    *,
    videos: int,
    max_attempts: int,
    width: int,
    height: int,
    fps: int,
    domain_randomization: bool,
) -> dict[str, Any]:
    """Run deterministic CUDA rollouts and render only verified successes."""

    provenance = _checkpoint_provenance(checkpoint)
    render_model, render_source = _render_model(env)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = env.get_observations()
    traces: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]
    episode_dr = _episode_randomization(env)
    max_lift = torch.full((env.num_envs,), -torch.inf, device=env.device)
    max_quality = torch.zeros(env.num_envs, device=env.device)
    records: list[tuple[np.ndarray, dict[str, Any]]] = []
    attempts = 0
    while len(records) < videos and attempts < max_attempts:
        current = env.gpu.sim.data.qpos.detach().cpu().numpy()
        for index in range(env.num_envs):
            traces[index].append(current[index].copy())
        actions = actor(observations, stochastic_output=False)
        observations, _, dones, extras = env.step(actions)
        assert env.last_terms is not None
        terms = env.last_terms
        max_lift.copy_(torch.maximum(max_lift, terms.lift_height))
        max_quality.copy_(torch.maximum(max_quality, terms.grasp_quality))
        finished = dones.nonzero(as_tuple=False).flatten()
        if not len(finished):
            continue
        terminal_ids = extras["terminal_env_ids"].detach().cpu().tolist()
        terminal_qpos = extras["terminal_qpos"].detach().cpu().numpy()
        terminal = dict(zip(terminal_ids, terminal_qpos, strict=True))
        for env_id in finished.detach().cpu().tolist():
            attempts += 1
            traces[env_id].append(terminal[env_id].copy())
            if bool(terms.success[env_id]) and len(records) < videos:
                records.append(
                    (
                        np.stack(traces[env_id]),
                        {
                            "attempt": attempts,
                            "world_id": env_id,
                            "steps": len(traces[env_id]) - 1,
                            "max_lift": float(max_lift[env_id]),
                            "max_grasp_quality": float(max_quality[env_id]),
                            "randomization": episode_dr[env_id],
                        },
                    )
                )
            traces[env_id] = []
            max_lift[env_id] = -torch.inf
            max_quality[env_id] = 0.0
        reset_dr = _episode_randomization(env)
        for env_id in finished.detach().cpu().tolist():
            episode_dr[env_id] = reset_dr[env_id]
    if len(records) < videos:
        raise RuntimeError(
            f"Only found {len(records)} successful episodes in {attempts} attempts"
        )

    mode = "full_dr" if domain_randomization else "clean"
    outputs = []
    target_body = env.gpu.bundle.manifest["roles"]["primary"]
    for index, (qpos, episode) in enumerate(records, 1):
        stem = f"{env.config.task}_{mode}_seed{env.config.seed}_{index:02d}"
        video = output_dir / f"{stem}.mp4"
        metadata = output_dir / f"{stem}.json"
        camera = _write_video(
            render_model,
            qpos,
            video,
            task=env.config.task,
            target_body=target_body,
            width=width,
            height=height,
            fps=fps,
        )
        record = {
            **episode,
            **provenance,
            "video": str(video.resolve()),
            "task": env.config.task,
            "success": True,
            "deterministic_actor": True,
            "domain_randomization": domain_randomization,
            "reference_noise": domain_randomization,
            "dr_strength": (
                env.config.domain_randomization.strength(env.common_step_counter)
                if domain_randomization
                else 0.0
            ),
            "fps": fps,
            "resolution": [width, height],
            "render_source": render_source,
            "camera": camera,
        }
        metadata.write_text(json.dumps(record, indent=2))
        outputs.append(record)
    return {"videos": outputs, "attempts": attempts, "successes": len(outputs)}
