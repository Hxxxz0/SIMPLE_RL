from __future__ import annotations

import sys

import numpy as np
import torch
from tensordict import TensorDict

from simple.grasp_rl.diffusion import DDPMScheduler, DiffusionDenoiser, _loss
from simple.grasp_rl.distribution import TemporallyCorrelatedGaussianDistribution
from simple.grasp_rl.motion import frames_to_features
from simple.grasp_rl.policy import KnnBcActor, make_actor
from simple.grasp_rl.reference import (
    ReferenceLibrary,
    ReferenceTracker,
    augment_reference_observation,
    reference_contact_label,
)
from simple.grasp_rl.rewards import GraspReward, compose_reward
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    JOINT_NAMES,
    MOTION_FEATURE_DIM,
    MOTION_FRAME_DIM,
    MOTION_WINDOW,
    MAX_EPISODE_STEPS,
    REFERENCE_ACTOR_OBS_DIM,
    REFERENCE_CONTEXT_DIM,
)
from simple.grasp_rl.train import PpoTrainConfig, rsl_config
from simple.grasp_rl.tracker import ActionTransform, upper_joints_from_tracker


def test_schema_dimensions_are_frozen() -> None:
    assert ACTION_DIM == 36
    assert ACTOR_OBS_DIM == 192
    assert REFERENCE_CONTEXT_DIM == 401
    assert REFERENCE_ACTOR_OBS_DIM == 593
    assert len(JOINT_NAMES) == 43
    assert (MOTION_WINDOW, MOTION_FRAME_DIM, MOTION_FEATURE_DIM) == (10, 80, 82)
    assert MAX_EPISODE_STEPS == 192


def test_tracker_mapping_matches_simple_act_g1_order() -> None:
    action = np.arange(ACTION_DIM, dtype=np.float32)
    expected = np.concatenate(
        [action[14:28], action[0:3], action[5:7], action[3:5], action[7:14]]
    )
    np.testing.assert_array_equal(upper_joints_from_tracker(action), expected)
    assert len(expected) == 28


def test_piecewise_action_decoder_and_slew_limit() -> None:
    transform = ActionTransform(
        center=np.zeros(ACTION_DIM),
        low=-np.ones(ACTION_DIM),
        high=2 * np.ones(ACTION_DIM),
        max_delta=np.full(ACTION_DIM, 0.25),
    )
    raw = np.concatenate([np.ones(18), -np.ones(18)]).astype(np.float32)
    decoded = transform.decode(raw)
    np.testing.assert_array_equal(decoded[:18], np.full(18, 2.0))
    np.testing.assert_array_equal(decoded[18:], np.full(18, -1.0))
    limited = transform.decode(raw, previous=np.zeros(ACTION_DIM))
    np.testing.assert_array_equal(limited[:18], np.full(18, 0.25))
    np.testing.assert_array_equal(limited[18:], np.full(18, -0.25))
    np.testing.assert_allclose(transform.decode(transform.encode(decoded)), decoded)


def test_motion_features_are_finite_and_final_xy_is_anchored() -> None:
    frames = torch.zeros(3, MOTION_WINDOW, MOTION_FRAME_DIM)
    frames[..., 3] = 1.0  # identity quaternion in MuJoCo wxyz order
    frames[..., 0] = torch.linspace(0.0, 0.1, MOTION_WINDOW)
    frames[..., 1] = torch.linspace(0.0, -0.2, MOTION_WINDOW)
    frames[..., 2] = 0.75
    features = frames_to_features(frames)
    assert features.shape == (3, MOTION_WINDOW, MOTION_FEATURE_DIM)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[:, -1, :2], torch.zeros(3, 2))
    torch.testing.assert_close(features[:, :, 2], torch.full((3, MOTION_WINDOW), 0.75))


def test_unconditional_denoiser_signature_and_training_loss() -> None:
    model = DiffusionDenoiser(MOTION_FEATURE_DIM, MOTION_WINDOW, d_model=32, nhead=4, num_layers=1)
    scheduler = DDPMScheduler(num_timesteps=8)
    clean = torch.randn(4, MOTION_WINDOW, MOTION_FEATURE_DIM)
    timestep = torch.tensor([0, 1, 2, 3])
    prediction = model(clean, timestep)
    assert prediction.shape == clean.shape
    loss = _loss(model, scheduler, clean, samples=2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_success_terminal_bonus_outweighs_failure_penalties() -> None:
    assert GraspReward._terminal_adjustment(True, True, True) == 5.0
    assert GraspReward._terminal_adjustment(False, True, False) == -1.0
    assert GraspReward._terminal_adjustment(False, False, True) == -0.5
    assert GraspReward._terminal_adjustment(False, False, False) == 0.0


def test_additive_reward_preserves_task_signal_and_uses_prior_only_when_ready() -> None:
    reward, task, prior = compose_reward(
        target=2.0,
        penalty=0.5,
        terminal_adjustment=5.0,
        smp=0.25,
        smp_active=True,
        variant="smp_additive",
        task_weight=0.02,
        smp_weight=0.01,
    )
    assert task == 0.03
    assert np.isclose(prior, 0.0025)
    assert np.isclose(reward, 5.0325)
    inactive, _, inactive_prior = compose_reward(
        2.0, 0.5, 0.0, 1.0, False, "smp_additive", 0.02, 0.01
    )
    assert np.isclose(inactive, 0.03)
    assert np.isclose(inactive_prior, 0.0)


def test_cli_forwards_actor_warm_start(monkeypatch, tmp_path) -> None:
    from simple.grasp_rl import cli

    captured = {}

    def fake_train(action_transform, output, diffusion, config):
        captured.update(
            action_transform=action_transform,
            output=output,
            diffusion=diffusion,
            config=config,
        )
        return tmp_path / "model.pt"

    monkeypatch.setattr(cli, "train_ppo", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grasp-rl",
            "train",
            "--output",
            str(tmp_path),
            "--actor-warm-start",
            "actor.pt",
            "--variant",
            "smp_additive",
            "--actor-lr-scale",
            "0.01",
            "--learning-schedule",
            "fixed",
            "--task-reward-profile",
            "progress_v2",
            "--rsi-scene-hold-episodes",
            "17",
            "--rsi-randomize-target",
            "--target-position-jitter-xy",
            "0.01,0.02",
            "--target-yaw-jitter",
            "0.05",
            "--rsi-phase",
            "0.64,0.75",
            "--rsi-stage",
            "grasp_to_lift",
            "--bc-anchor-weight",
            "0.5",
            "--teacher-anchor-checkpoint",
            "teacher.pt",
            "--teacher-anchor-weight",
            "10",
            "--plan-conditioned-actor",
        ],
    )
    cli.main()
    assert captured["config"].actor_warm_start == "actor.pt"
    assert captured["config"].reward_variant == "smp_additive"
    assert captured["config"].actor_learning_rate_scale == 0.01
    assert captured["config"].learning_schedule == "fixed"
    assert captured["config"].task_reward_profile == "progress_v2"
    assert captured["config"].rsi_scene_hold_episodes == 17
    assert captured["config"].rsi_randomize_target is True
    assert captured["config"].target_position_jitter_xy == (0.01, 0.02)
    assert captured["config"].target_yaw_jitter == 0.05
    assert captured["config"].rsi_phase == (0.64, 0.75)
    assert captured["config"].rsi_stage == "grasp_to_lift"
    assert captured["config"].bc_anchor_weight == 0.5
    assert captured["config"].bc_anchor_sources == ("bc",)
    assert captured["config"].teacher_anchor_checkpoint == "teacher.pt"
    assert captured["config"].teacher_anchor_weight == 10.0
    assert captured["config"].plan_conditioned_actor is True


def test_plan_conditioned_actor_emits_complete_proposed_command() -> None:
    actor = make_actor(
        "cpu",
        REFERENCE_ACTOR_OBS_DIM,
        plan_conditioned=True,
    )
    for parameter in actor.mlp.parameters():
        torch.nn.init.zeros_(parameter)
    observation = torch.randn(2, REFERENCE_ACTOR_OBS_DIM)
    proposal = observation[:, ACTOR_OBS_DIM : ACTOR_OBS_DIM + ACTION_DIM]
    output = actor(
        TensorDict({"actor": observation}, batch_size=[2]),
        stochastic_output=False,
    )
    torch.testing.assert_close(output, proposal)
    assert "_plan_conditioned_actor" in actor.state_dict()


def test_temporal_knn_starts_at_trajectory_zero_then_advances() -> None:
    observations = torch.stack(
        (
            torch.zeros(ACTOR_OBS_DIM),
            torch.ones(ACTOR_OBS_DIM),
            torch.full((ACTOR_OBS_DIM,), 2.0),
        )
    )
    actions = torch.stack(
        (
            torch.zeros(ACTION_DIM),
            torch.ones(ACTION_DIM),
            torch.full((ACTION_DIM,), 2.0),
        )
    )
    actor = KnnBcActor(
        {
            "observations": observations,
            "actions": actions,
            "mean": torch.zeros(ACTOR_OBS_DIM),
            "std": torch.ones(ACTOR_OBS_DIM),
            "episode_ids": torch.tensor([0, 0, 1]),
            "step_ids": torch.tensor([0, 1, 0]),
        },
        "cpu",
    )
    query = TensorDict(
        {"actor": torch.ones(1, ACTOR_OBS_DIM)}, batch_size=[1]
    )
    # The exact all-frame neighbor is step 1, but a reset may only select a
    # trajectory start. The next query can then advance within that episode.
    torch.testing.assert_close(actor(query), torch.zeros(1, ACTION_DIM))
    torch.testing.assert_close(actor(query), torch.ones(1, ACTION_DIM))
    actor.reset()
    torch.testing.assert_close(actor(query), torch.zeros(1, ACTION_DIM))


def test_progress_reward_cannot_farm_an_unchanged_pregrasp_state() -> None:
    first, best, bonus = GraspReward._progress_target(
        progress=2.0, best_progress=0.0, hold=0.0, stable=0.0
    )
    repeated, repeated_best, repeated_bonus = GraspReward._progress_target(
        progress=2.0, best_progress=best, hold=0.0, stable=0.0
    )
    assert (first, bonus, best) == (100.0, 100.0, 2.0)
    assert (repeated, repeated_bonus, repeated_best) == (0.0, 0.0, 2.0)
    assert GraspReward._terminal_adjustment(
        True, False, False, profile="progress_v2"
    ) == 10.0
    assert GraspReward._terminal_adjustment(
        False, False, False, timeout=True, profile="progress_v2"
    ) == -2.0


def test_grail_release_reward_preserves_published_grasp_weights() -> None:
    target = GraspReward._grail_release_target(
        pregrasp=0.75,
        grasp=0.5,
        finger_direction=0.25,
        lift=0.4,
        stable=0.2,
    )
    assert np.isclose(
        target,
        0.75 + 5.0 * 0.5 + 10.0 * 0.25 + 5.0 * 0.4 + 2.0 * 0.2,
    )
    assert GraspReward._terminal_adjustment(
        True, False, False, profile="grail_release_v1"
    ) == 20.0
    assert GraspReward._terminal_adjustment(
        False, False, False, timeout=True, profile="grail_release_v1"
    ) == -5.0


def test_temporally_correlated_exploration_holds_standardized_noise() -> None:
    torch.manual_seed(7)
    distribution = TemporallyCorrelatedGaussianDistribution(
        output_dim=3,
        init_std=0.2,
        learn_std=False,
        hold_steps=3,
    )
    normalized = []
    for value in range(4):
        mean = torch.full((2, 3), float(value))
        distribution.update(mean)
        normalized.append((distribution.sample() - mean) / distribution.std)
    torch.testing.assert_close(normalized[0], normalized[1])
    torch.testing.assert_close(normalized[1], normalized[2])
    assert not torch.equal(normalized[2], normalized[3])


def test_ppo_learning_schedule_is_configurable() -> None:
    config = PpoTrainConfig(
        learning_schedule="fixed",
        num_learning_epochs=1,
        num_mini_batches=2,
    )
    algorithm = rsl_config(config)["algorithm"]
    assert algorithm["schedule"] == "fixed"
    assert algorithm["num_learning_epochs"] == 1
    assert algorithm["num_mini_batches"] == 2


def test_cli_forwards_normalized_evaluation_phase(monkeypatch, tmp_path) -> None:
    from simple.grasp_rl import cli

    captured = {}

    def fake_evaluate(*args, **kwargs):
        captured.update(kwargs)
        return {"success_rate": 0.0}

    monkeypatch.setattr(cli, "evaluate_policy", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grasp-rl",
            "evaluate",
            "--checkpoint",
            "actor.pt",
            "--output",
            str(tmp_path),
            "--initialization-phase",
            "0.9",
            "--randomize-target",
            "--target-position-jitter-xy",
            "0.03,0.04",
            "--target-yaw-jitter",
            "0.2",
        ],
    )
    cli.main()
    assert captured["initialization_prefix"] is None
    assert captured["initialization_phase"] == 0.9
    assert captured["randomize_target"] is True
    assert captured["target_position_jitter_xy"] == (0.03, 0.04)
    assert captured["target_yaw_jitter"] == 0.2


def test_reference_context_contains_future_full_commands_and_contact() -> None:
    observations = np.zeros((60, ACTOR_OBS_DIM), dtype=np.float32)
    actions = np.stack(
        [np.full(ACTION_DIM, index / 100.0, dtype=np.float32) for index in range(60)]
    )
    observations[:, 132] = np.arange(60, dtype=np.float32) / 100.0
    # Thumb link 1 and index link 4 form the desired bilateral reference label.
    observations[5, 165 + 3 : 165 + 6] = 3.0
    observations[5, 165 + 12 : 165 + 15] = 3.0
    assert reference_contact_label(observations[5]) == 1.0
    augmented = augment_reference_observation(
        observations[0], observations, actions, index=0
    )
    assert augmented.shape == (REFERENCE_ACTOR_OBS_DIM,)
    context = augmented[ACTOR_OBS_DIM:]
    np.testing.assert_allclose(context[:ACTION_DIM], actions[0])
    # The second future frame is five control steps ahead.
    second = 40
    np.testing.assert_allclose(context[second : second + ACTION_DIM], actions[5])
    assert context[second + ACTION_DIM] == np.float32(0.05)
    assert context[second + ACTION_DIM + 3] == 1.0


def test_reference_reward_prefers_exact_transition(tmp_path) -> None:
    root = tmp_path / "processed"
    source = root / "bc"
    source.mkdir(parents=True)
    manifest = {
        "splits": {"train": [0], "val": [], "test": []},
    }
    (root / "manifest.json").write_text(__import__("json").dumps(manifest))
    observations = np.zeros((3, ACTOR_OBS_DIM), dtype=np.float32)
    observations[1, :43] = 0.2
    observations[1, 43:86] = 0.1
    observations[1, 86:89] = [0.0, 0.0, -1.0]
    actions = np.zeros((3, ACTION_DIM), dtype=np.float32)
    actions[0] = 0.25
    np.savez(
        source / "episode_000000.npz",
        observations=observations,
        raw_actions=actions,
        sample_weights=np.ones(3, dtype=np.float32),
    )
    library = ReferenceLibrary(root, splits=("train",))
    exact = ReferenceTracker(library)
    exact.reset(observations[0], exact_episode=0)
    exact_terms = exact.reward(observations[1], actions[0])
    bad = ReferenceTracker(library)
    bad.reset(observations[0], exact_episode=0)
    bad_terms = bad.reward(
        np.zeros(ACTOR_OBS_DIM, dtype=np.float32),
        -np.ones(ACTION_DIM, dtype=np.float32),
    )
    assert exact_terms.total > bad_terms.total
    assert np.isclose(exact_terms.joint_pose, 1.0)
    assert np.isclose(exact_terms.tracker_action, 1.0)
