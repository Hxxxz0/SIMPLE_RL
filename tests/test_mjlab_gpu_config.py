import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from simple.grasp_rl.mjlab_gpu.cli import _evaluation_dr_strength, _parser, _seed_torch
from simple.grasp_rl.mjlab_gpu.config import (
    DomainRandomizationConfig,
    MjlabPpoConfig,
    ReferenceNoiseConfig,
)
from simple.grasp_rl.mjlab_gpu.reference_noise import (
    apply_reference_noise,
    transform_reference_positions,
)
from simple.grasp_rl.schema import REFERENCE_CONTEXT_DIM, REFERENCE_FRAME_DIM


def test_gpu_ppo_config_rejects_cpu_and_small_long_runs(tmp_path) -> None:
    with pytest.raises(ValueError, match="cuda"):
        MjlabPpoConfig("tabletop_grasp", str(tmp_path), device="cpu")
    with pytest.raises(ValueError, match="2048"):
        MjlabPpoConfig("tabletop_grasp", str(tmp_path), num_envs=1024)
    smoke = MjlabPpoConfig(
        "tabletop_grasp", str(tmp_path), num_envs=16, smoke_mode=True
    )
    assert smoke.num_envs == 16


def test_train_cli_accepts_full_trajectory_rollout_override() -> None:
    args = _parser().parse_args(
        [
            "train",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "reference",
            "--output",
            "output",
            "--ppo-steps-per-env",
            "240",
        ]
    )
    assert args.ppo_steps_per_env == 240


def test_train_cli_accepts_reference_target_pose_retarget_gains() -> None:
    args = _parser().parse_args(
        [
            "train",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "reference",
            "--output",
            "output",
            "--reference-target-x-arm-gains",
            "-10",
            "2",
            "--reference-target-y-arm-gains",
            "5",
            "-3.5",
        ]
    )
    assert args.reference_target_x_arm_gains == [-10.0, 2.0]
    assert args.reference_target_y_arm_gains == [5.0, -3.5]


def test_gpu_entrypoints_seed_torch_and_cuda(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("cpu", seed)))
    monkeypatch.setattr(
        torch.cuda, "manual_seed", lambda seed: calls.append(("cuda", seed))
    )

    _seed_torch(17)

    assert calls == [("cpu", 17), ("cuda", 17)]


def test_reference_noise_is_checkpoint_versioned(tmp_path) -> None:
    config = MjlabPpoConfig("tabletop_grasp", str(tmp_path))
    changed = replace(
        config,
        domain_randomization=replace(
            config.domain_randomization,
            reference_noise=replace(
                config.domain_randomization.reference_noise, phase_std=0.02
            ),
        ),
    )
    first = config.checkpoint_metadata()
    assert first == config.checkpoint_metadata()
    assert first["resolved_sha256"] != changed.checkpoint_metadata()["resolved_sha256"]
    config.assert_resume_compatible(first)
    with pytest.raises(ValueError, match="hash mismatch"):
        changed.assert_resume_compatible(first)


def test_zero_retarget_accepts_legacy_checkpoint_metadata(tmp_path) -> None:
    config = MjlabPpoConfig("tabletop_grasp", str(tmp_path))
    metadata = config.checkpoint_metadata()
    legacy_resolved = dict(metadata["resolved"])
    legacy_resolved.pop("reference_target_x_arm_gains")
    legacy_resolved.pop("reference_target_y_arm_gains")
    metadata["resolved"] = legacy_resolved
    metadata["resolved_sha256"] = hashlib.sha256(
        json.dumps(legacy_resolved, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    config.assert_resume_compatible(metadata)
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(
            config, reference_target_x_arm_gains=(-10.0, 2.0)
        ).assert_resume_compatible(metadata)
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(
            config, reference_target_y_arm_gains=(5.0, -3.5)
        ).assert_resume_compatible(metadata)


def test_reference_noise_only_changes_intended_policy_inputs() -> None:
    context = torch.zeros(2, REFERENCE_CONTEXT_DIM, device="cpu")
    frames = context[:, :-1].reshape(2, -1, REFERENCE_FRAME_DIM)
    frames[..., -1] = 1.0  # contact truth inside the policy reference context
    context[:, -1] = 0.5
    original = context.clone()
    untouched = apply_reference_noise(
        context,
        ReferenceNoiseConfig(),
        enabled=False,
    )
    assert untouched.data_ptr() == context.data_ptr()

    generator = torch.Generator().manual_seed(7)
    noisy = apply_reference_noise(
        context,
        ReferenceNoiseConfig(future_dropout_probability=0.0),
        enabled=True,
        generator=generator,
    )
    noisy_frames = noisy[:, :-1].reshape(2, -1, REFERENCE_FRAME_DIM)
    assert torch.count_nonzero(noisy_frames[..., :39]) > 0
    assert torch.equal(noisy_frames[..., -1], frames[..., -1])
    assert torch.all((0.0 <= noisy[:, -1]) & (noisy[:, -1] <= 1.0))
    assert torch.equal(context, original)


def test_reference_scene_transform_matches_sampled_scene_pose() -> None:
    positions = torch.tensor([[[1.0, 0.0, 0.2], [0.0, 1.0, 0.3]]])
    transformed = transform_reference_positions(
        positions,
        translation_xy=torch.tensor([[0.1, -0.2]]),
        yaw=torch.tensor([torch.pi / 2]),
        origin_xy=torch.zeros(1, 2),
    )
    expected = torch.tensor([[[0.1, 0.8, 0.2], [-0.9, -0.2, 0.3]]])
    torch.testing.assert_close(transformed, expected)


def test_enabled_dr_requires_synchronized_reference_transform() -> None:
    with pytest.raises(ValueError, match="sync_reference"):
        DomainRandomizationConfig(sync_reference_scene_transform=False)


def test_domain_randomization_curriculum_reaches_full_strength() -> None:
    config = DomainRandomizationConfig(
        curriculum_warmup_steps=10, curriculum_ramp_steps=20
    )
    assert config.strength(10) == 0.0
    assert config.training_strength(10) == 0.10
    assert config.strength(20) == 0.5
    assert config.training_strength(20) == 0.5
    assert config.strength(30) == 1.0
    assert config.reference_noise.scaled(0.5).action_std == pytest.approx(
        0.5 * config.reference_noise.action_std
    )
    assert config.reference_noise.action_std == pytest.approx(0.002)


def test_pose_only_dr_stages_dynamics_and_reference_noise() -> None:
    full = DomainRandomizationConfig(
        curriculum_initial_strength=0.25, curriculum_ramp_steps=19_200
    )
    pose = full.pose_only()
    target_x = full.target_x_only()
    target_y = full.target_y_only()
    target_yaw = full.target_yaw_only()

    assert pose.target_position_jitter_xy == full.target_position_jitter_xy
    assert pose.target_yaw_jitter == full.target_yaw_jitter
    assert pose.curriculum_initial_strength == 0.25
    assert pose.curriculum_ramp_steps == 19_200
    assert pose.target_mass_scale == (1.0, 1.0)
    assert pose.friction_scale == (1.0, 1.0)
    assert pose.joint_damping_scale == (1.0, 1.0)
    assert pose.actuator_strength_scale == (1.0, 1.0)
    assert pose.action_delay_max_steps == 0
    assert pose.reference_noise == ReferenceNoiseConfig(
        action_std=0.0,
        position_std=0.0,
        phase_std=0.0,
        future_dropout_probability=0.0,
    )
    assert full.reference_noise.enabled
    assert not pose.reference_noise.enabled
    assert target_x.target_position_jitter_xy == (
        full.target_position_jitter_xy[0],
        0.0,
    )
    assert target_x.target_yaw_jitter == 0.0
    assert not target_x.reference_noise.enabled
    assert target_y.target_position_jitter_xy == (
        0.0,
        full.target_position_jitter_xy[1],
    )
    assert target_y.target_yaw_jitter == 0.0
    assert not target_y.reference_noise.enabled
    assert target_yaw.target_position_jitter_xy == (0.0, 0.0)
    assert target_yaw.target_yaw_jitter == full.target_yaw_jitter
    assert not target_yaw.reference_noise.enabled


def test_evaluation_dr_strength_is_explicit_and_backward_compatible() -> None:
    staged = SimpleNamespace(
        evaluation_dr_strength=0.375, stress_domain_randomization=False
    )
    full = SimpleNamespace(
        evaluation_dr_strength=None, stress_domain_randomization=True
    )
    assert _evaluation_dr_strength(staged) == 0.375
    assert _evaluation_dr_strength(full) == 1.0

    staged.stress_domain_randomization = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        _evaluation_dr_strength(staged)
    staged.stress_domain_randomization = False
    staged.evaluation_dr_strength = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _evaluation_dr_strength(staged)
