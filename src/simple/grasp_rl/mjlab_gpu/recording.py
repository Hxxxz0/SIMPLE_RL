"""Audited offscreen videos from the real mjlab GPU policy loop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.reward import (
    FINGER_DISTAL_CLOSURE_RAD,
    FINGER_MEAN_CLOSURE_RAD,
    finger_closure_score,
)
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv
from simple.grasp_rl.schema import (
    JOINT_NAMES,
    LEFT_CONTACT_LINK_NAMES,
    RIGHT_CONTACT_LINK_NAMES,
)

FINGER_AUDIT_SCHEMA_VERSION = 1
FINGER_CONTACT_FORCE = 2.0
FINGER_CONTACT_HOLD_STEPS = 5


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
    latest = integrity.get("latest_record") or {}
    if gpu.get("config", {}).get("backend") != "mjlab_mujoco_warp":
        raise ValueError("Video recording requires an mjlab GPU checkpoint")
    if (
        not latest.get("on_policy")
        or latest.get("algorithm") != "rsl_rl.algorithms.ppo.PPO"
    ):
        raise ValueError("Video recording requires audited on-policy PPO")
    portable_integrity = dict(integrity)
    audit_path = portable_integrity.get("audit_path")
    if isinstance(audit_path, str):
        portable_integrity["audit_path"] = Path(audit_path).name
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "backend": "mjlab_mujoco_warp",
        "ppo_integrity": portable_integrity,
    }


def _reference_provenance(env: GpuGraspVecEnv) -> dict[str, Any]:
    root = env.reference.root.resolve()
    manifest = root / "manifest.json"
    metadata = env.reference.metadata()
    return {
        "checkpoint": None,
        "checkpoint_sha256": None,
        "backend": "mjlab_mujoco_warp",
        "ppo_integrity": None,
        "reference": {
            "directory_absolute": str(root),
            "manifest_sha256": _sha256(manifest),
            "data_sha256": metadata["data_sha256"],
            "action_transform_sha256": metadata["action_transform_sha256"],
            "episodes": metadata["episodes"],
            "strict_episode": metadata["strict_episode"],
        },
    }


def _episode_randomization(env: GpuGraspVecEnv) -> list[dict[str, Any]]:
    randomizer = env.randomizer
    translations = randomizer.target_translation_xy.detach().cpu().tolist()
    yaws = randomizer.target_yaw.detach().cpu().tolist()
    destination_translations = (
        randomizer.destination_translation_xy.detach().cpu().tolist()
    )
    destination_yaws = randomizer.destination_yaw.detach().cpu().tolist()
    distractor_translations = (
        randomizer.distractor_translation_xy.detach().cpu().tolist()
    )
    distractor_yaws = randomizer.distractor_yaw.detach().cpu().tolist()
    distractor_names = [
        randomizer.sim.mj_model.body(body_id).name
        for body_id in randomizer.distractor_body_ids
    ]
    base_translations = randomizer.robot_base_translation_xy.detach().cpu().tolist()
    base_yaws = randomizer.robot_base_yaw.detach().cpu().tolist()
    mass_scales = randomizer.target_mass_scale.detach().cpu().tolist()
    friction_scales = randomizer.friction_scale.detach().cpu().tolist()
    damping_scales = randomizer.joint_damping_scale.detach().cpu().tolist()
    actuator_scales = randomizer.actuator_strength_scale.detach().cpu().tolist()
    delays = randomizer.action_delay_steps.detach().cpu().tolist()
    rows = env.reference.episode_rows.detach().cpu().tolist()
    return [
        {
            "target_translation_xy": translations[index],
            "target_yaw": yaws[index],
            "destination_translation_xy": destination_translations[index],
            "destination_yaw": destination_yaws[index],
            "distractor_poses": {
                name: {
                    "translation_xy": distractor_translations[index][slot],
                    "yaw": distractor_yaws[index][slot],
                }
                for slot, name in enumerate(distractor_names)
            },
            "robot_base_translation_xy": base_translations[index],
            "robot_base_yaw": base_yaws[index],
            "target_mass_scale": mass_scales[index],
            "friction_scale": friction_scales[index],
            "joint_damping_scale": damping_scales[index],
            "actuator_strength_scale": actuator_scales[index],
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
    camera_view: str,
) -> str:
    import imageio.v2 as imageio

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    # Robot-mounted stereo cameras cannot show whether the full-body motion is
    # physically valid.  Release videos always use an external MuJoCo camera.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    target_id = model.body(target_body).id
    pelvis_id = model.body("pelvis").id
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    if camera_view == "full_robot":
        camera.azimuth = 135.0
        camera.elevation = -14.0
        camera.distance = 1.8 if task == "tabletop_grasp" else 2.5
        camera.lookat[:] = 0.5 * data.xpos[target_id] + 0.5 * data.xpos[pelvis_id]
        camera.lookat[2] += 0.05
        camera_label = "free_full_robot"
    elif camera_view == "grasp_closeup":
        wrist_ids = [
            model.body(name).id
            for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")
        ]
        wrist_id = min(
            wrist_ids,
            key=lambda index: np.linalg.norm(data.xpos[index] - data.xpos[target_id]),
        )
        camera.azimuth = 45.0
        camera.elevation = -10.0
        camera.distance = 0.75
        camera.lookat[:] = 0.6 * data.xpos[target_id] + 0.4 * data.xpos[wrist_id]
        camera.lookat[2] += 0.03
        camera_label = "free_grasp_closeup"
    else:
        raise ValueError(f"Unknown camera view: {camera_view}")
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


def _target_physics_audit(model: mujoco.MjModel, target_body: str) -> dict[str, Any]:
    """Prove that a recorded target is a free body without attachment constraints."""

    body_id = model.body(target_body).id
    joint_start = int(model.body_jntadr[body_id])
    joint_count = int(model.body_jntnum[body_id])
    joint_ids = range(joint_start, joint_start + joint_count)
    joint_types = [
        mujoco.mjtJoint(int(model.jnt_type[index])).name for index in joint_ids
    ]
    attachment_types = {
        int(mujoco.mjtEq.mjEQ_CONNECT),
        int(mujoco.mjtEq.mjEQ_WELD),
    }
    target_equalities = [
        index
        for index in range(model.neq)
        if int(model.eq_type[index]) in attachment_types
        and body_id in (int(model.eq_obj1id[index]), int(model.eq_obj2id[index]))
    ]
    target_is_free_body = joint_types == [mujoco.mjtJoint.mjJNT_FREE.name]
    return {
        "target_body": target_body,
        "target_joint_types": joint_types,
        "target_is_free_body": target_is_free_body,
        "target_equality_constraint_count": len(target_equalities),
        "physically_unattached": target_is_free_body and not target_equalities,
    }


def _diagnostic_rank(episode: dict[str, Any]) -> tuple[int, float, float]:
    """Rank failed episodes by task progress before grasp quality and lift."""

    return (
        int(episode["max_stage"]),
        float(episode["max_grasp_quality"]),
        float(episode["max_lift"]),
    )


def _finger_grasp_truth(
    hand_qpos: torch.Tensor,
    initial_hand_qpos: torch.Tensor,
    contact_forces: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return visual/physical grasp truth for two seven-DoF hands."""

    if hand_qpos.shape != initial_hand_qpos.shape or hand_qpos.shape[-2:] != (2, 7):
        raise ValueError("hand qpos must have shape (N, 2, 7)")
    if contact_forces.shape != (*hand_qpos.shape[:-1], 8, 3):
        raise ValueError("contact forces must have shape (N, 2, 8, 3)")
    closure = (hand_qpos - initial_hand_qpos).abs()
    magnitudes = contact_forces.norm(dim=-1)
    thumb_contact = magnitudes[..., 1:4].amax(dim=-1) > FINGER_CONTACT_FORCE
    index_contact = magnitudes[..., 4:6].amax(dim=-1) > FINGER_CONTACT_FORCE
    middle_contact = magnitudes[..., 6:8].amax(dim=-1) > FINGER_CONTACT_FORCE
    distal_closure = torch.stack(
        (closure[..., 2], closure[..., 4], closure[..., 6]), dim=-1
    )
    closure_gate = finger_closure_score(hand_qpos, initial_hand_qpos) >= 1.0
    diverse_contact = thumb_contact & (index_contact | middle_contact)
    return {
        "closure_delta": closure,
        "contact_magnitudes": magnitudes,
        "distal_closure": distal_closure,
        "closure_gate": closure_gate,
        "thumb_contact": thumb_contact,
        "index_contact": index_contact,
        "middle_contact": middle_contact,
        "diverse_contact": diverse_contact,
        "valid_grasp": closure_gate & diverse_contact,
    }


def _required_grasp_hands(env: GpuGraspVecEnv) -> tuple[int, ...]:
    spec = getattr(env.state_reader, "spec", None)
    if spec is None:
        return (1,)
    required: set[int] = set()
    for stage in spec.stages:
        if stage.primitive not in ("grasp", "bimanual", "handover"):
            continue
        if stage.hand in ("left", "both") or stage.primitive == "handover":
            required.add(0)
        if stage.hand in ("right", "both") or stage.primitive == "handover":
            required.add(1)
    return tuple(sorted(required))


def _contact_forces(env: GpuGraspVecEnv) -> torch.Tensor:
    _, state = env.state_reader.actor_observation()
    forces = getattr(state, "contact_forces_pelvis", None)
    if forces is not None:
        return forces
    result = torch.zeros(env.num_envs, 2, 8, 3, dtype=torch.float32, device=env.device)
    result[:, 1] = state.contact.link_forces_pelvis
    return result


@torch.inference_mode()
def record_success_videos(
    env: GpuGraspVecEnv,
    actor: torch.nn.Module | None,
    checkpoint: Path | None,
    output_dir: Path,
    *,
    videos: int,
    max_attempts: int,
    width: int,
    height: int,
    fps: int,
    domain_randomization: bool,
    allow_diagnostic_fallback: bool = False,
    camera_view: str = "full_robot",
    stochastic_policy: bool = False,
    reference_only: bool = False,
) -> dict[str, Any]:
    """Render verified successes, optionally falling back to best failed episodes."""

    if reference_only:
        if actor is not None or checkpoint is not None:
            raise ValueError("Reference-only recording cannot use a PPO checkpoint")
        if stochastic_policy:
            raise ValueError("Reference-only recording cannot be stochastic")
        provenance = _reference_provenance(env)
    else:
        if actor is None or checkpoint is None:
            raise ValueError("PPO video recording requires a checkpoint and actor")
        provenance = _checkpoint_provenance(checkpoint)
    render_model, render_source = _render_model(env)
    recorded_dr_strength = (
        env.config.domain_randomization.strength(env.common_step_counter)
        if domain_randomization
        else 0.0
    )
    physics_audit = _target_physics_audit(
        env.gpu.sim.mj_model, env.gpu.bundle.manifest["roles"]["primary"]
    )
    if not physics_audit["physically_unattached"]:
        raise ValueError("Video recording requires a free, unattached target body")
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = env.get_observations()
    traces: list[list[np.ndarray]] = [[] for _ in range(env.num_envs)]
    episode_dr = _episode_randomization(env)
    max_lift = torch.full((env.num_envs,), -torch.inf, device=env.device)
    max_quality = torch.zeros(env.num_envs, device=env.device)
    max_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    model = env.gpu.sim.mj_model
    hand_qpos_indices = torch.tensor(
        [int(model.jnt_qposadr[model.joint(name).id]) for name in JOINT_NAMES[29:43]],
        dtype=torch.long,
        device=env.device,
    ).reshape(2, 7)
    initial_hand_qpos = env.gpu.sim.data.qpos[:, hand_qpos_indices].clone()
    max_hand_closure = torch.zeros(
        env.num_envs, 2, 7, dtype=torch.float32, device=env.device
    )
    max_hand_contact = torch.zeros(
        env.num_envs, 2, 8, dtype=torch.float32, device=env.device
    )
    grasp_hold = torch.zeros(env.num_envs, 2, dtype=torch.long, device=env.device)
    max_grasp_hold = torch.zeros_like(grasp_hold)
    required_hands = _required_grasp_hands(env)
    hand_names = ("left", "right")
    contact_names = (LEFT_CONTACT_LINK_NAMES, RIGHT_CONTACT_LINK_NAMES)
    records: list[tuple[np.ndarray, dict[str, Any]]] = []
    diagnostic_records: list[tuple[np.ndarray, dict[str, Any]]] = []
    stage_names = getattr(getattr(env.state_reader, "spec", None), "stages", ())
    attempts = 0
    while len(records) < videos and attempts < max_attempts:
        current = env.gpu.sim.data.qpos.detach().cpu().numpy()
        for index in range(env.num_envs):
            traces[index].append(current[index].copy())
        stage_index = getattr(env.reward, "stage_index", None)
        if stage_index is not None:
            max_stage.copy_(torch.maximum(max_stage, stage_index))
        current_hand_qpos = env.gpu.sim.data.qpos[:, hand_qpos_indices]
        audit = _finger_grasp_truth(
            current_hand_qpos, initial_hand_qpos, _contact_forces(env)
        )
        max_hand_closure.copy_(torch.maximum(max_hand_closure, audit["closure_delta"]))
        max_hand_contact.copy_(
            torch.maximum(max_hand_contact, audit["contact_magnitudes"])
        )
        grasp_hold.copy_(torch.where(audit["valid_grasp"], grasp_hold + 1, 0))
        max_grasp_hold.copy_(torch.maximum(max_grasp_hold, grasp_hold))
        actions = (
            env.reference.current_action().clone()
            if reference_only
            else actor(observations, stochastic_output=stochastic_policy)
        )
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
            simulator_success = bool(terms.success[env_id])
            hand_audits = {}
            for hand_id, hand_name in enumerate(hand_names):
                peak_closure = max_hand_closure[env_id, hand_id]
                peak_contact = max_hand_contact[env_id, hand_id]
                hand_audits[hand_name] = {
                    "joint_names": list(
                        JOINT_NAMES[29 + 7 * hand_id : 36 + 7 * hand_id]
                    ),
                    "peak_closure_delta_rad": peak_closure.detach().cpu().tolist(),
                    "distal_closure_delta_rad": peak_closure[[2, 4, 6]]
                    .detach()
                    .cpu()
                    .tolist(),
                    "contact_link_names": list(contact_names[hand_id]),
                    "peak_contact_force": peak_contact.detach().cpu().tolist(),
                    "max_consecutive_valid_grasp_steps": int(
                        max_grasp_hold[env_id, hand_id]
                    ),
                    "passed": int(max_grasp_hold[env_id, hand_id])
                    >= FINGER_CONTACT_HOLD_STEPS,
                }
            finger_audit_passed = all(
                hand_audits[hand_names[hand_id]]["passed"] for hand_id in required_hands
            )
            success = simulator_success and finger_audit_passed
            stage = int(max_stage[env_id])
            episode = {
                "attempt": attempts,
                "world_id": env_id,
                "steps": len(traces[env_id]) - 1,
                "success": success,
                "simulator_success": simulator_success,
                "max_lift": float(max_lift[env_id]),
                "max_grasp_quality": float(max_quality[env_id]),
                "terminal_lift": float(terms.lift_height[env_id]),
                "terminal_grasp_quality": float(terms.grasp_quality[env_id]),
                "terminal_is_grasp": bool(terms.is_grasp[env_id]),
                "max_stage": stage,
                "max_stage_name": (
                    stage_names[stage].name if stage < len(stage_names) else str(stage)
                ),
                "randomization": episode_dr[env_id],
                "finger_grasp_audit": {
                    "schema_version": FINGER_AUDIT_SCHEMA_VERSION,
                    "required_hands": [hand_names[index] for index in required_hands],
                    "distal_closure_threshold_rad": FINGER_DISTAL_CLOSURE_RAD,
                    "mean_closure_threshold_rad": FINGER_MEAN_CLOSURE_RAD,
                    "contact_force_threshold": FINGER_CONTACT_FORCE,
                    "required_consecutive_steps": FINGER_CONTACT_HOLD_STEPS,
                    "hands": hand_audits,
                    "passed": finger_audit_passed,
                },
                "success_rejection_reason": (
                    None
                    if success or not simulator_success
                    else "insufficient_finger_closure_or_diverse_contact"
                ),
            }
            candidate = (np.stack(traces[env_id]), episode)
            if success and len(records) < videos:
                records.append(candidate)
            elif allow_diagnostic_fallback:
                diagnostic_records.append(candidate)
                diagnostic_records.sort(
                    key=lambda item: _diagnostic_rank(item[1]), reverse=True
                )
                del diagnostic_records[videos:]
            traces[env_id] = []
            max_lift[env_id] = -torch.inf
            max_quality[env_id] = 0.0
            max_stage[env_id] = 0
            max_hand_closure[env_id] = 0.0
            max_hand_contact[env_id] = 0.0
            grasp_hold[env_id] = 0
            max_grasp_hold[env_id] = 0
        reset_dr = _episode_randomization(env)
        for env_id in finished.detach().cpu().tolist():
            episode_dr[env_id] = reset_dr[env_id]
            initial_hand_qpos[env_id] = env.gpu.sim.data.qpos[env_id, hand_qpos_indices]
    if len(records) < videos and not allow_diagnostic_fallback:
        raise RuntimeError(
            f"Only found {len(records)} successful episodes in {attempts} attempts"
        )
    if len(records) < videos:
        records.extend(diagnostic_records[: videos - len(records)])
    if len(records) < videos:
        raise RuntimeError(
            f"Only completed {len(records)} episodes in {attempts} attempts"
        )

    mode = "full_dr" if domain_randomization else "clean"
    outputs = []
    target_body = env.gpu.bundle.manifest["roles"]["primary"]
    for index, (qpos, episode) in enumerate(records, 1):
        outcome = "success" if episode["success"] else "diagnostic"
        policy_name = "reference" if reference_only else "ppo"
        stem = f"gpu_{policy_name}_{outcome}_{mode}_{index:02d}"
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
            camera_view=camera_view,
        )
        record = {
            **episode,
            **provenance,
            "video": video.name,
            "task": env.config.task,
            "diagnostic": not episode["success"],
            "selection": (
                "simulator_truth_and_finger_audit_success"
                if episode["success"]
                else "best_available_failed_episode"
            ),
            "policy": "reference_only" if reference_only else "ppo",
            "deterministic_actor": not stochastic_policy,
            "policy_sampling": (
                "reference"
                if reference_only
                else "ppo_gaussian"
                if stochastic_policy
                else "mean"
            ),
            "policy_seed": env.config.seed,
            "domain_randomization": domain_randomization,
            "reference_noise": bool(
                domain_randomization
                and env.config.domain_randomization.reference_noise.enabled
            ),
            "dr_strength": recorded_dr_strength,
            "fps": fps,
            "resolution": [width, height],
            "render_source": render_source,
            "camera": camera,
            "physical_target_audit": physics_audit,
        }
        metadata.write_text(json.dumps(record, indent=2))
        outputs.append(record)
    successes = sum(bool(output["success"]) for output in outputs)
    return {
        "videos": outputs,
        "attempts": attempts,
        "successes": successes,
        "diagnostics": len(outputs) - successes,
    }
