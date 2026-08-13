import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from simple.grasp_rl.mjlab_gpu.cli import (
    _checkpoint_training_dr_strength,
    _config,
    _evaluate,
    _evaluation_dr_strength,
    _parser,
    _record,
    _seed_torch,
)
from simple.grasp_rl.mjlab_gpu.config import (
    DomainRandomizationConfig,
    MjlabPpoConfig,
    ReferenceNoiseConfig,
)
from simple.grasp_rl.mjlab_gpu.reference_noise import (
    apply_reference_noise,
    transform_reference_positions,
)
from simple.grasp_rl.mjlab_gpu.vec_env import (
    GpuGraspVecEnv,
    _lift_arm_residual_scale,
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


def test_lift_arm_residual_decay_is_opt_in_and_grasp_anything_only(tmp_path) -> None:
    default = MjlabPpoConfig("grasp_anything", str(tmp_path))
    assert default.grasp_anything_lift_arm_residual_min_scale == 1.0
    assert default.grasp_anything_lift_arm_residual_decay_steps == 0
    with pytest.raises(ValueError, match="requires grasp_anything"):
        replace(
            default,
            task="tabletop_grasp",
            grasp_anything_lift_arm_residual_min_scale=0.2,
            grasp_anything_lift_arm_residual_decay_steps=20,
        )
    with pytest.raises(ValueError, match="requires grasp_anything"):
        replace(default, grasp_anything_lift_arm_residual_min_scale=0.2)


def test_lift_arm_residual_scale_decays_linearly_to_floor() -> None:
    steps = torch.tensor([0, 5, 10, 20])
    scale = _lift_arm_residual_scale(
        steps, minimum_scale=0.2, decay_steps=10
    )
    assert scale.tolist() == pytest.approx([1.0, 0.6, 0.2, 0.2])


def test_lift_arm_residual_decay_keeps_right_hand_correction() -> None:
    env = object.__new__(GpuGraspVecEnv)
    env.config = SimpleNamespace(
        max_reference_action_deviation=0.7,
        grasp_anything_lift_arm_residual_min_scale=0.2,
        grasp_anything_lift_arm_residual_decay_steps=10,
    )
    env.reference = SimpleNamespace(
        observation_dim=0,
        current_action=lambda: torch.zeros(2, 36),
    )
    env._lift_arm_decay_step = torch.tensor([0, 10])
    action = torch.ones(2, 36)

    bounded = env._bounded_reference_action(action)

    assert torch.allclose(bounded[:, 7:14], torch.full((2, 7), 0.7))
    assert torch.allclose(bounded[0, 21:28], torch.full((7,), 0.7))
    assert torch.allclose(bounded[1, 21:28], torch.full((7,), 0.14))


def test_lift_arm_residual_decay_requires_stable_grasp() -> None:
    env = object.__new__(GpuGraspVecEnv)
    env.config = SimpleNamespace(
        grasp_anything_lift_arm_residual_min_scale=0.2,
        grasp_anything_lift_arm_residual_decay_steps=10,
        grasp_anything_lift_arm_residual_grasp_steps=3,
    )
    env._lift_arm_grasp_streak = torch.zeros(2, dtype=torch.long)
    env._lift_arm_decay_step = torch.zeros(2, dtype=torch.long)
    env._lift_arm_decay_triggered = torch.zeros(2, dtype=torch.bool)

    env._update_lift_arm_residual_decay(torch.tensor([True, True]))
    env._update_lift_arm_residual_decay(torch.tensor([False, True]))
    env._update_lift_arm_residual_decay(torch.tensor([True, True]))

    assert env._lift_arm_grasp_streak.tolist() == [1, 3]
    assert env._lift_arm_decay_triggered.tolist() == [False, True]
    assert env._lift_arm_decay_step.tolist() == [0, 1]


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


def test_train_cli_defaults_to_legacy_independent_exploration() -> None:
    args = _parser().parse_args(
        [
            "train",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "reference",
            "--output",
            "output",
        ]
    )
    assert args.exploration_hold_steps == 1


def test_train_cli_can_explicitly_disable_bootstrap_randomization(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    args = _parser().parse_args(
        [
            "train",
            "--asset-bundle",
            str(tmp_path / "assets"),
            "--reference-processed",
            str(tmp_path / "reference"),
            "--output",
            str(tmp_path / "output"),
            "--disable-domain-randomization",
            "--smoke",
        ]
    )

    assert not _config(args).domain_randomization.enabled


def test_reproduction_cli_exposes_paired_benchmark_and_collection() -> None:
    common = [
        "--task",
        "xmove_pick",
        "--asset-bundle",
        "assets",
        "--reference-processed",
        "reference",
        "--checkpoint",
        "model.pt",
    ]
    benchmark = _parser().parse_args(["benchmark", *common, "--output", "paired.json"])
    assert benchmark.episodes == 128
    assert benchmark.output.name == "paired.json"
    assert benchmark.minimum_success_rate is None

    collect = _parser().parse_args(
        [
            "collect",
            *common,
            "--output-dir",
            "dataset",
            "--successes",
            "3",
        ]
    )
    assert collect.smoke
    assert collect.num_envs == 64
    assert collect.successes == 3

    audit = _parser().parse_args(
        [
            "audit-dataset",
            "--dataset-root",
            "dataset",
            "--expected-successes",
            "500",
            "--expected-task",
            "bend_pick_teleop",
            "--expected-dr-strength",
            "1",
            "--require-full-dr-coverage",
        ]
    )
    assert audit.expected_successes == 500
    assert audit.require_full_dr_coverage


def test_record_cli_diagnostic_fallback_is_explicit() -> None:
    common = [
        "record",
        "--asset-bundle",
        "assets",
        "--reference-processed",
        "reference",
        "--checkpoint",
        "model.pt",
        "--output-dir",
        "videos",
    ]
    assert not _parser().parse_args(common).allow_diagnostic_fallback
    assert not _parser().parse_args(common).stochastic_policy
    assert (
        _parser()
        .parse_args([*common, "--allow-diagnostic-fallback"])
        .allow_diagnostic_fallback
    )
    assert _parser().parse_args(common).camera_view == "full_robot"
    assert (
        _parser().parse_args([*common, "--camera-view", "grasp_closeup"]).camera_view
        == "grasp_closeup"
    )
    assert _parser().parse_args([*common, "--stochastic-policy"]).stochastic_policy
    reference_only = [item for item in common if item not in ("--checkpoint", "model.pt")]
    reference_args = _parser().parse_args([*reference_only, "--reference-only"])
    assert reference_args.reference_only
    assert reference_args.checkpoint is None
    with pytest.raises(SystemExit):
        _parser().parse_args([*common, "--reference-only"])

    evaluate = [
        "evaluate",
        "--asset-bundle",
        "assets",
        "--reference-processed",
        "reference",
        "--checkpoint",
        "model.pt",
    ]
    assert not _parser().parse_args(evaluate).stochastic_policy
    assert not _parser().parse_args(evaluate).proposal_only
    assert _parser().parse_args([*evaluate, "--stochastic-policy"]).stochastic_policy
    assert _parser().parse_args([*evaluate, "--proposal-only"]).proposal_only


def test_record_marks_explicit_staged_dr_as_randomized(monkeypatch, tmp_path) -> None:
    args = _parser().parse_args(
        [
            "record",
            "--asset-bundle",
            str(tmp_path / "assets"),
            "--reference-processed",
            str(tmp_path / "reference"),
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--output-dir",
            str(tmp_path / "videos"),
            "--evaluation-dr-strength",
            "0.2",
        ]
    )
    env = SimpleNamespace(
        common_step_counter=0,
        device="cuda:0",
        close=lambda: None,
    )
    runner = SimpleNamespace(
        alg=SimpleNamespace(get_policy=lambda: SimpleNamespace(eval=lambda: "actor")),
        load_actor_warm_start=lambda _path: None,
    )
    captured = {}

    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli._config",
        lambda _args: SimpleNamespace(seed=42),
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.GpuGraspVecEnv", lambda *args, **kwargs: env
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli._set_evaluation_dr_strength",
        lambda _env, strength: captured.update(strength=strength),
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.checkpoint_uses_plan_conditioned_actor",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.GpuPpoRunner", lambda *args, **kwargs: runner
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.record_success_videos",
        lambda *args, **kwargs: captured.update(kwargs) or {"videos": []},
    )

    _record(args)

    assert captured["strength"] == pytest.approx(0.2)
    assert captured["domain_randomization"] is True


def test_record_reference_only_does_not_construct_ppo_runner(
    monkeypatch, tmp_path
) -> None:
    args = _parser().parse_args(
        [
            "record",
            "--asset-bundle",
            str(tmp_path / "assets"),
            "--reference-processed",
            str(tmp_path / "reference"),
            "--reference-only",
            "--output-dir",
            str(tmp_path / "videos"),
        ]
    )
    env = SimpleNamespace(common_step_counter=0, device="cuda:0")
    captured = {}

    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli._config",
        lambda _args: SimpleNamespace(seed=42),
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.GpuGraspVecEnv", lambda *args, **kwargs: env
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.GpuPpoRunner",
        lambda *args, **kwargs: pytest.fail("reference-only constructed PPO runner"),
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.record_success_videos",
        lambda *args, **kwargs: (
            captured.update(actor=args[1], checkpoint=args[2], **kwargs)
            or {"videos": []}
        ),
    )

    _record(args)

    assert captured["actor"] is None
    assert captured["checkpoint"] is None
    assert captured["reference_only"] is True

    reference_only = _parser().parse_args(
        [
            "evaluate",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "reference",
            "--reference-only",
            "--stochastic-policy",
        ]
    )
    with pytest.raises(ValueError, match="PPO checkpoint"):
        _evaluate(reference_only)

    incompatible_baselines = _parser().parse_args(
        [
            "evaluate",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "reference",
            "--reference-only",
            "--proposal-only",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _evaluate(incompatible_baselines)


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
            "--reference-target-positive-y-arm-gains",
            "8",
            "-1.5",
            "--reference-target-yaw-arm-gains",
            "1",
            "-2",
        ]
    )
    assert args.reference_target_x_arm_gains == [-10.0, 2.0]
    assert args.reference_target_y_arm_gains == [5.0, -3.5]
    assert args.reference_target_positive_y_arm_gains == [8.0, -1.5]
    assert args.reference_target_yaw_arm_gains == [1.0, -2.0]


def test_train_cli_accepts_scratch_plan_conditioned_actor() -> None:
    args = _parser().parse_args(
        [
            "train",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "reference",
            "--output",
            "output",
            "--plan-conditioned-actor",
            "--scratch-actor-output-scale",
            "0.001",
            "--scratch-right-hand-correction",
            *(["0"] * 7),
            "--scratch-right-arm-correction",
            *(["0"] * 7),
            "--save-interval",
            "10",
        ]
    )
    assert args.plan_conditioned_actor
    assert args.scratch_actor_output_scale == 0.001
    assert args.scratch_right_hand_correction == [0.0] * 7
    assert args.scratch_right_arm_correction == [0.0] * 7
    assert args.save_interval == 10


def test_cli_accepts_guarded_balanced_reference_selection(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    args = _parser().parse_args(
        [
            "train",
            "--asset-bundle",
            str(tmp_path / "assets"),
            "--reference-processed",
            str(tmp_path / "reference"),
            "--output",
            str(tmp_path / "output"),
            "--reference-selection",
            "balanced",
            "--max-reference-initial-position-offset",
            "0.05",
        ]
    )
    config = _config(args)
    assert config.reference_selection == "balanced"
    assert config.max_reference_initial_position_offset == 0.05
    with pytest.raises(ValueError, match="alignment limit"):
        replace(config, max_reference_initial_position_offset=None)


def test_strict_reference_rejects_non_asset_selection(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires asset reference selection"):
        MjlabPpoConfig(
            "tabletop_grasp",
            str(tmp_path),
            strict_reference_episode=20,
            reference_selection="balanced",
            max_reference_initial_position_offset=0.05,
        )


def test_cli_accepts_task_specific_dr_envelope(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    args = _parser().parse_args(
        [
            "evaluate",
            "--task",
            "bend_pick",
            "--asset-bundle",
            str(tmp_path / "assets"),
            "--reference-processed",
            str(tmp_path / "reference"),
            "--reference-only",
            "--target-position-jitter-xy",
            "0.015",
            "0.02",
            "--target-position-offset-center-xy",
            "0.01",
            "-0.005",
            "--target-position-focus-probability",
            "0.5",
            "--target-position-focus-jitter-xy",
            "0.01",
            "0.02",
            "--target-position-focus-offset-center-xy",
            "0.04",
            "0.03",
            "--target-yaw-jitter",
            "0.1",
            "--destination-position-jitter-xy",
            "0.02",
            "0.025",
            "--destination-yaw-jitter",
            "0.12",
            "--distractor-position-jitter-xy",
            "0.03",
            "0.04",
            "--distractor-yaw-jitter",
            "0.2",
            "--robot-base-position-jitter-xy",
            "0.01",
            "0.015",
            "--robot-base-yaw-jitter",
            "0.03",
        ]
    )
    config = _config(args)
    assert config.domain_randomization.target_position_jitter_xy == (0.015, 0.02)
    assert config.domain_randomization.target_position_offset_center_xy == (
        0.01,
        -0.005,
    )
    assert config.domain_randomization.target_position_focus_probability == 0.5
    assert config.domain_randomization.target_position_focus_jitter_xy == (
        0.01,
        0.02,
    )
    assert config.domain_randomization.target_position_focus_offset_center_xy == (
        0.04,
        0.03,
    )
    assert config.domain_randomization.target_yaw_jitter == 0.1
    assert config.domain_randomization.destination_position_jitter_xy == (
        0.02,
        0.025,
    )
    assert config.domain_randomization.destination_yaw_jitter == 0.12
    assert config.domain_randomization.distractor_position_jitter_xy == (0.03, 0.04)
    assert config.domain_randomization.distractor_yaw_jitter == 0.2
    assert config.domain_randomization.robot_base_position_jitter_xy == (0.01, 0.015)
    assert config.domain_randomization.robot_base_yaw_jitter == 0.03


def test_cli_accepts_strict_reference_and_wide_physics_noise_dr(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    args = _parser().parse_args(
        [
            "train",
            "--task",
            "bend_pick_teleop",
            "--asset-bundle",
            str(tmp_path / "assets"),
            "--reference-processed",
            str(tmp_path / "reference"),
            "--output",
            str(tmp_path / "output"),
            "--strict-reference-episode",
            "20",
            "--target-mass-scale",
            "0.6",
            "1.4",
            "--friction-scale",
            "0.5",
            "1.5",
            "--joint-damping-scale",
            "0.8",
            "1.2",
            "--actuator-strength-scale",
            "0.85",
            "1.15",
            "--action-delay-max-steps",
            "1",
            "--reference-action-noise-std",
            "0.004",
            "--reference-position-noise-std",
            "0.005",
            "--reference-phase-noise-std",
            "0.02",
            "--reference-future-dropout-probability",
            "0.05",
        ]
    )
    config = _config(args)
    dr = config.domain_randomization
    assert config.strict_reference_episode == 20
    assert dr.target_mass_scale == (0.6, 1.4)
    assert dr.friction_scale == (0.5, 1.5)
    assert dr.joint_damping_scale == (0.8, 1.2)
    assert dr.actuator_strength_scale == (0.85, 1.15)
    assert dr.action_delay_max_steps == 1
    assert dr.reference_noise == ReferenceNoiseConfig(
        action_std=0.004,
        position_std=0.005,
        phase_std=0.02,
        future_dropout_probability=0.05,
    )


def test_cli_exposes_backward_compatible_reference_residual_limit(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    args = _parser().parse_args(
        [
            "train",
            "--asset-bundle",
            str(tmp_path / "assets"),
            "--reference-processed",
            str(tmp_path / "reference"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert args.max_reference_action_deviation == 0.35
    assert args.reference_reward_weight == 0.05
    default_config = _config(args)
    assert default_config.max_reference_action_deviation == 0.35
    assert default_config.reference_reward_weight == 0.05
    assert default_config.domain_randomization.target_position_jitter_xy == (
        0.025,
        0.03,
    )
    assert default_config.domain_randomization.target_yaw_jitter == 0.15

    args.max_reference_action_deviation = 0.7
    assert _config(args).max_reference_action_deviation == 0.7
    args.reference_reward_weight = 0.5
    assert _config(args).reference_reward_weight == 0.5


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
    legacy_resolved.pop("reference_target_positive_y_arm_gains")
    legacy_resolved.pop("reference_target_yaw_arm_gains")
    legacy_resolved.pop("strict_reference_episode")
    legacy_resolved.pop("reference_selection")
    legacy_resolved.pop("max_reference_initial_position_offset")
    legacy_resolved.pop("grasp_anything_lift_arm_residual_min_scale")
    legacy_resolved.pop("grasp_anything_lift_arm_residual_decay_steps")
    legacy_resolved.pop("grasp_anything_lift_arm_residual_grasp_steps")
    legacy_resolved["domain_randomization"] = dict(
        legacy_resolved["domain_randomization"]
    )
    legacy_resolved["domain_randomization"].pop(
        "target_position_offset_center_xy"
    )
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
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(
            config, reference_target_positive_y_arm_gains=(8.0, -1.5)
        ).assert_resume_compatible(metadata)
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(
            config, reference_target_yaw_arm_gains=(1.0, -2.0)
        ).assert_resume_compatible(metadata)


def test_exact_resume_allows_verified_release_relocation(tmp_path) -> None:
    original = MjlabPpoConfig(
        "tabletop_grasp",
        str(tmp_path / "old_assets"),
        reference_processed=str(tmp_path / "old_reference"),
    )
    relocated = replace(
        original,
        asset_bundle=str(tmp_path / "other" / "assets"),
        reference_processed=str(tmp_path / "other" / "reference"),
    )

    relocated.assert_resume_compatible(original.checkpoint_metadata())
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(relocated, seed=original.seed + 1).assert_resume_compatible(
            original.checkpoint_metadata()
        )


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
        curriculum_initial_strength=0.25,
        curriculum_ramp_steps=19_200,
        target_position_offset_center_xy=(0.01, -0.02),
        target_position_focus_probability=0.5,
        target_position_focus_jitter_xy=(0.03, 0.04),
        target_position_focus_offset_center_xy=(0.02, -0.01),
        destination_position_jitter_xy=(0.02, 0.025),
        destination_yaw_jitter=0.1,
        distractor_position_jitter_xy=(0.03, 0.03),
        distractor_yaw_jitter=0.2,
        robot_base_position_jitter_xy=(0.01, 0.01),
        robot_base_yaw_jitter=0.03,
    )
    pose = full.pose_only()
    target_x = full.target_x_only()
    target_y = full.target_y_only()
    target_yaw = full.target_yaw_only()

    assert pose.target_position_jitter_xy == full.target_position_jitter_xy
    assert (
        pose.target_position_offset_center_xy
        == full.target_position_offset_center_xy
    )
    assert pose.target_position_focus_probability == 0.5
    assert pose.target_position_focus_jitter_xy == (0.03, 0.04)
    assert pose.target_position_focus_offset_center_xy == (0.02, -0.01)
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
    assert target_x.target_position_offset_center_xy == (0.01, 0.0)
    assert target_x.target_position_focus_jitter_xy == (0.03, 0.0)
    assert target_x.target_position_focus_offset_center_xy == (0.02, 0.0)
    assert target_x.target_yaw_jitter == 0.0
    assert target_x.destination_position_jitter_xy == (0.0, 0.0)
    assert target_x.distractor_position_jitter_xy == (0.0, 0.0)
    assert target_x.robot_base_position_jitter_xy == (0.0, 0.0)
    assert not target_x.reference_noise.enabled
    assert target_y.target_position_jitter_xy == (
        0.0,
        full.target_position_jitter_xy[1],
    )
    assert target_y.target_position_offset_center_xy == (0.0, -0.02)
    assert target_y.target_position_focus_jitter_xy == (0.0, 0.04)
    assert target_y.target_position_focus_offset_center_xy == (0.0, -0.01)
    assert target_y.target_yaw_jitter == 0.0
    assert target_y.destination_position_jitter_xy == (0.0, 0.0)
    assert not target_y.reference_noise.enabled
    assert target_yaw.target_position_jitter_xy == (0.0, 0.0)
    assert target_yaw.target_position_offset_center_xy == (0.0, 0.0)
    assert target_yaw.target_position_focus_probability == 0.0
    assert target_yaw.target_yaw_jitter == full.target_yaw_jitter
    assert target_yaw.destination_yaw_jitter == 0.0
    assert target_yaw.distractor_yaw_jitter == 0.0
    assert target_yaw.robot_base_yaw_jitter == 0.0
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


def test_checkpoint_reports_training_dr_strength(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "env_state": {"common_step_counter": 2_400},
            "mjlab_gpu_metadata": {
                "config": {
                    "resolved": {
                        "domain_randomization": {
                            "enabled": True,
                            "curriculum_initial_strength": 0.3,
                            "curriculum_warmup_steps": 0,
                            "curriculum_ramp_steps": 240_000,
                        }
                    }
                }
            },
        },
        checkpoint,
    )

    assert _checkpoint_training_dr_strength(checkpoint) == pytest.approx(0.3)
