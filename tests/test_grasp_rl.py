from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from simple.grasp_rl.collect import _filter_replay_gated_rows
from simple.grasp_rl.data_v2 import (
    _successful_replay_transform,
    _usable_replay_episode_ids,
    repair_cross_table_actions,
)
from simple.grasp_rl.diffusion import DDPMScheduler, DiffusionDenoiser, _loss
from simple.grasp_rl.distribution import TemporallyCorrelatedGaussianDistribution
from simple.grasp_rl.evaluate import (
    _filter_evaluation_split,
    reference_action_from_observation,
)
from simple.grasp_rl.goal_reward import GoalGraphReward
from simple.grasp_rl.motion import frames_to_features
from simple.grasp_rl.paired import exact_mcnemar_p_value
from simple.grasp_rl.policy import KnnBcActor, make_actor
from simple.grasp_rl.reference import (
    ReferenceLibrary,
    ReferenceTracker,
    augment_reference_observation,
    reference_contact_label,
)
from simple.grasp_rl.render import _select_camera
from simple.grasp_rl.rewards import GraspReward, compose_reward
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    ACTOR_OBS_V2_DIM,
    JOINT_NAMES,
    MAX_EPISODE_STEPS,
    MOTION_FEATURE_DIM,
    MOTION_FRAME_DIM,
    MOTION_WINDOW,
    REFERENCE_ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_V2_DIM,
    REFERENCE_CONTEXT_DIM,
    REFERENCE_CONTEXT_V2_DIM,
)
from simple.grasp_rl.state_v2 import V2_SLICES
from simple.grasp_rl.task_spec import (
    TaskSpecV2,
    checkpoint_task_metadata,
    get_task_spec,
    task_names,
    validate_task_metadata,
)
from simple.grasp_rl.tracker import (
    ActionTransform,
    compute_action_transform,
    upper_joints_from_tracker,
)
from simple.grasp_rl.train import PpoTrainConfig, rsl_config
from simple.grasp_rl.vec_env import (
    apply_reference_action_bias,
    reference_residual_action_rate,
)


def test_schema_dimensions_are_frozen() -> None:
    assert ACTION_DIM == 36
    assert ACTOR_OBS_DIM == 192
    assert REFERENCE_CONTEXT_DIM == 401
    assert REFERENCE_ACTOR_OBS_DIM == 593
    assert len(JOINT_NAMES) == 43
    assert (MOTION_WINDOW, MOTION_FRAME_DIM, MOTION_FEATURE_DIM) == (10, 80, 82)
    assert MAX_EPISODE_STEPS == 192


@pytest.mark.parametrize("base_dim", [ACTOR_OBS_DIM, ACTOR_OBS_V2_DIM])
def test_reference_override_reads_action_after_active_base_schema(base_dim: int) -> None:
    policy_observation = np.zeros(base_dim + ACTION_DIM + 7, dtype=np.float32)
    expected = np.linspace(-1.0, 1.0, ACTION_DIM, dtype=np.float32)
    policy_observation[base_dim : base_dim + ACTION_DIM] = expected
    np.testing.assert_array_equal(
        reference_action_from_observation(policy_observation, base_dim), expected
    )


def test_exact_mcnemar_uses_only_discordant_pairs() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(2, 0) == 0.5
    assert exact_mcnemar_p_value(3, 3) == 1.0
    assert exact_mcnemar_p_value(10, 0) == pytest.approx(2.0 / 1024.0)


def test_v2_schema_and_catalog_cover_merged_tasks() -> None:
    assert ACTOR_OBS_V2_DIM == 331
    assert REFERENCE_CONTEXT_V2_DIM == 511
    assert REFERENCE_ACTOR_OBS_V2_DIM == 842
    assert len(task_names()) == 15
    flagship = get_task_spec("G1WholebodyLocomotionPickBetweenTablesMixed-v0")
    assert isinstance(flagship, TaskSpecV2)
    assert flagship.controller_backend == "amo"
    assert flagship.family == "place"
    assert flagship.source_uids[0] == "g1_wholebody_locomotion_pick_between_tables_variant5"
    assert get_task_spec("open_oven").controller_backend == "sonic_wbc"
    grasp_anything = get_task_spec("grasp_anything")
    assert grasp_anything.dataset_name == "G1WholebodyGraspAnythingPhysicalPPO-v0"
    assert grasp_anything.registry_uid == get_task_spec("xmove_pick").registry_uid
    assert (
        get_task_spec("g1_wholebody_xmove_pick_teleop").name == "xmove_pick"
    )
    assert list(V2_SLICES.values())[0].start == 0
    assert list(V2_SLICES.values())[-1].stop == ACTOR_OBS_V2_DIM


def test_cross_table_repair_changes_only_missing_turn_fields() -> None:
    actions = np.zeros((260, ACTION_DIM), dtype=np.float32)
    actions[20:220, 32] = .1
    actions[220:, 32] = .35
    repaired, report = repair_cross_table_actions(actions)
    assert report["applied"] is True
    assert report["changed_dimensions"] == [34, 35]
    np.testing.assert_array_equal(repaired[:, :32], actions[:, :32])
    np.testing.assert_array_equal(repaired[:, 32:34], actions[:, 32:34])
    assert np.isclose(repaired[119, 35], np.pi / 2)
    assert np.isclose(repaired[219, 35], np.pi)
    assert np.all(repaired[20:220, 34] == 1)


def test_goal_graph_does_not_charge_valid_stationary_stage() -> None:
    assert GoalGraphReward.gamma == 1.0


def test_goal_graph_requires_completion_after_entering_final_stage() -> None:
    reward = GoalGraphReward.__new__(GoalGraphReward)
    reward.spec = SimpleNamespace(stages=(object(), object(), object()))
    reward.stage_index = 1
    reward.stage_hold = 3

    advanced, success = reward._advance_stage(completed=True)
    assert advanced is True
    assert success is False
    assert reward.stage_index == 2
    assert reward.stage_hold == 0

    _, success = reward._advance_stage(completed=True)
    assert success is True


def test_grail_v2_grasp_terms_use_reference_contact_intent_and_geometry() -> None:
    reward = GoalGraphReward.__new__(GoalGraphReward)
    reward.reference_should_contact = 1.0
    reward.reference_contact_center_primary = np.zeros(3)
    forces = np.zeros((2, 8, 3), dtype=np.float64)
    forces[1, :, 0] = 0.2
    state = SimpleNamespace(
        contact_forces_pelvis=forces,
        left_primary_contact=False,
        right_primary_contact=True,
        primary=SimpleNamespace(pos_w=np.zeros(3), rot_w=np.eye(3)),
        distal_pos_w=np.asarray(
            [
                np.zeros((3, 3)),
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            ]
        ),
    )
    grasp, direction = reward._grail_grasp_terms(state)
    assert np.isclose(grasp, 1.0)
    assert np.isclose(direction, 1.0)

    reward.set_reference_contact(0.0, np.zeros(3, dtype=np.float32))
    grasp, direction = reward._grail_grasp_terms(state)
    assert grasp == 0.0
    assert direction == 0.0


def test_v2_plan_correction_preserves_non_manipulation_and_release() -> None:
    actor = make_actor(
        "cpu", REFERENCE_ACTOR_OBS_V2_DIM, plan_conditioned=True
    ).eval()
    with torch.no_grad():
        for parameter in actor.mlp.parameters():
            parameter.zero_()
        linear = [
            module
            for module in actor.mlp.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        linear[-1].bias.fill_(1.0)

    def command(stage: int, family: int = 1) -> np.ndarray:
        observation = torch.zeros(1, REFERENCE_ACTOR_OBS_V2_DIM)
        observation[:, ACTOR_OBS_V2_DIM:ACTOR_OBS_V2_DIM + ACTION_DIM] = .25
        observation[:, 316 + family] = 1.0
        observation[:, 322 + stage] = 1.0
        result = actor(
            TensorDict({"actor": observation}, batch_size=[1]),
            stochastic_output=False,
        )
        return result.detach().numpy()[0]

    approach = command(0)
    transport = command(3)
    place = command(4)
    release = command(5)
    np.testing.assert_allclose(approach[7:14], 1.25)
    np.testing.assert_allclose(approach[21:28], 1.25)
    np.testing.assert_allclose(approach[:7], .25)
    np.testing.assert_allclose(approach[28:], .25)
    np.testing.assert_allclose(transport[7:14], 1.25)
    np.testing.assert_allclose(transport[21:28], 1.25)
    np.testing.assert_allclose(place[7:14], 1.25)
    np.testing.assert_allclose(place[21:28], 1.25)
    np.testing.assert_allclose(place[:7], .25)
    np.testing.assert_allclose(place[28:], .25)
    np.testing.assert_allclose(release, .25)
    handover_place = command(3, family=2)
    np.testing.assert_allclose(handover_place[7:14], 1.25)
    np.testing.assert_allclose(handover_place[21:28], 1.25)
    np.testing.assert_allclose(command(4, family=2), .25)


def test_bend_pick_adapter_preserves_policy_contract_and_native_goal() -> None:
    task = get_task_spec("bend_pick")
    metadata = task.metadata()
    assert task.registry_uid == "g1_wholebody_bend_pick_mp"
    assert task.reward.success_lift == 0.09
    assert task.reward.min_pelvis_height == 0.50
    assert task.reward.stalled_grasp_steps is None
    assert task.max_episode_steps == 300
    assert metadata["actor_observation_dim"] == ACTOR_OBS_DIM
    assert metadata["reference_actor_observation_dim"] == REFERENCE_ACTOR_OBS_DIM
    assert metadata["action_dim"] == ACTION_DIM


def test_checkpoint_metadata_rejects_cross_task_and_transform(tmp_path) -> None:
    transform_a = tmp_path / "a.npz"
    transform_b = tmp_path / "b.npz"
    transform_a.write_bytes(b"a")
    transform_b.write_bytes(b"b")
    payload = {
        "task_metadata": checkpoint_task_metadata(
            "bend_pick", transform_a
        )
    }
    validate_task_metadata(
        payload, "bend_pick", action_transform=transform_a
    )
    with pytest.raises(ValueError, match="task mismatch"):
        validate_task_metadata(payload, "tabletop_grasp")
    with pytest.raises(ValueError, match="action-transform"):
        validate_task_metadata(
            payload, "bend_pick", action_transform=transform_b
        )
    with pytest.raises(ValueError, match="no task metadata"):
        validate_task_metadata({}, "bend_pick")


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


def test_non_tabletop_transform_preserves_bending_command_range() -> None:
    episode = np.zeros((31, ACTION_DIM), dtype=np.float32)
    episode[:, 31] = np.linspace(0.45, 0.75, len(episode))
    transform = compute_action_transform(
        [episode, episode.copy()],
        legacy_tabletop_locomotion_bounds=False,
    )
    assert transform.low[31] <= np.float32(0.45)
    assert transform.high[31] >= np.float32(0.75)
    np.testing.assert_allclose(
        transform.decode(transform.encode(episode[-1])),
        episode[-1],
        atol=1e-6,
    )


def test_successful_replay_transform_round_trips_every_command() -> None:
    rng = np.random.default_rng(11)
    episodes = []
    for frames in (17, 29, 41):
        commands = rng.uniform(-1.5, 1.5, size=(frames, ACTION_DIM)).astype(
            np.float32
        )
        commands[0] = np.linspace(-0.2, 0.2, ACTION_DIM, dtype=np.float32)
        episodes.append(commands)
    transform = _successful_replay_transform(episodes)
    for commands in episodes:
        previous = transform.center
        decoded = []
        for command in commands:
            physical = transform.decode(transform.encode(command), previous)
            decoded.append(physical)
            previous = physical
        np.testing.assert_allclose(decoded, commands, atol=1e-6, rtol=0.0)


def test_replay_gate_selects_only_verified_successes() -> None:
    reports = [
        {"episode": 8, "success": True},
        {"episode": 9, "success": False},
        {"episode": 10, "success": True},
        {"episode": 11, "success": True},
    ]
    assert _usable_replay_episode_ids(reports) == [8, 10, 11]
    with pytest.raises(RuntimeError, match="Fewer than 3"):
        _usable_replay_episode_ids(reports[:3])


def test_evaluation_split_filters_to_manifest_episode_ids(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": [2, 5],
                    "val": [7],
                    "test": [11, 13],
                }
            }
        )
    )
    rows = [{"episode_index": value} for value in (2, 7, 11, 13, 17)]
    assert _filter_evaluation_split(rows, "test", tmp_path) == rows[2:4]
    assert _filter_evaluation_split(rows, "all", None) is rows
    with pytest.raises(ValueError, match="requires reference_processed"):
        _filter_evaluation_split(rows, "val", None)


def test_collection_excludes_known_replay_failures(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "unique_episodes": [1, 4, 8],
                "splits": {"train": [1], "val": [4], "test": [8]},
            }
        )
    )
    rows = [{"episode_index": value} for value in range(10)]
    assert _filter_replay_gated_rows(rows, tmp_path) == [
        rows[1],
        rows[4],
        rows[8],
    ]


def test_render_auto_camera_supports_legacy_and_sonic_tasks() -> None:
    assert _select_camera("auto", ("front_stereo_left",)) == "front_stereo_left"
    assert _select_camera(
        "auto", ("head_stereo_left", "head_stereo_right")
    ) == "head_stereo_left"
    with pytest.raises(KeyError, match="Unknown camera"):
        _select_camera("missing", ("head_stereo_left",))


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
            "--rsi-processed",
            "processed-v2",
            "--rsi-randomize-target",
            "--target-position-jitter-xy",
            "0.01,0.02",
            "--target-position-offset-center-xy=-0.025,0.01",
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
            "--reference-rank-max",
            "4",
            "--reference-base-episode-probability",
            "0.75",
            "--reference-action-noise-std",
            "0.03",
            "--reference-action-noise-hold-steps",
            "20",
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
    assert captured["config"].rsi_processed == "processed-v2"
    assert captured["config"].rsi_randomize_target is True
    assert captured["config"].target_position_jitter_xy == (0.01, 0.02)
    assert captured["config"].target_position_offset_center_xy == (-0.025, 0.01)
    assert captured["config"].target_yaw_jitter == 0.05
    assert captured["config"].rsi_phase == (0.64, 0.75)
    assert captured["config"].rsi_stage == "grasp_to_lift"
    assert captured["config"].bc_anchor_weight == 0.5
    assert captured["config"].bc_anchor_sources == ("bc",)
    assert captured["config"].teacher_anchor_checkpoint == "teacher.pt"
    assert captured["config"].teacher_anchor_weight == 10.0
    assert captured["config"].reference_rank_max == 4
    assert captured["config"].reference_base_episode_probability == 0.75
    assert captured["config"].reference_action_noise_std == 0.03
    assert captured["config"].reference_action_noise_hold_steps == 20
    assert captured["config"].plan_conditioned_actor is True


def test_reference_action_bias_is_coherent_and_manipulation_only() -> None:
    observation = torch.zeros(2, REFERENCE_ACTOR_OBS_V2_DIM)
    bias = torch.zeros(2, ACTION_DIM)
    bias[:, 7:14] = 0.1
    bias[:, 21:28] = -0.2
    perturbed = apply_reference_action_bias(
        observation,
        bias,
        ACTOR_OBS_V2_DIM,
        51,
    )
    torch.testing.assert_close(
        perturbed[:, :ACTOR_OBS_V2_DIM],
        observation[:, :ACTOR_OBS_V2_DIM],
    )
    for frame_index in range(10):
        start = ACTOR_OBS_V2_DIM + frame_index * 51
        torch.testing.assert_close(perturbed[:, start:start + ACTION_DIM], bias)
        torch.testing.assert_close(
            perturbed[:, start + ACTION_DIM:start + 51],
            observation[:, start + ACTION_DIM:start + 51],
        )


def test_reference_action_rate_does_not_penalize_generated_motion() -> None:
    proposal = torch.tensor([[0.2, -0.4], [0.6, 0.1]])
    previous = torch.zeros_like(proposal)
    rate, residual = reference_residual_action_rate(
        proposal.clone(), proposal, previous
    )
    torch.testing.assert_close(rate, torch.zeros(2))
    torch.testing.assert_close(residual, torch.zeros_like(proposal))

    actions = proposal + torch.tensor([[0.1, -0.2], [0.0, 0.3]])
    rate, residual = reference_residual_action_rate(actions, proposal, previous)
    torch.testing.assert_close(rate, torch.tensor([0.05, 0.09]))
    torch.testing.assert_close(residual, actions - proposal)


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


def test_v2_plan_conditioned_actor_uses_complete_command_slot() -> None:
    actor = make_actor("cpu", REFERENCE_ACTOR_OBS_V2_DIM, plan_conditioned=True)
    for parameter in actor.mlp.parameters():
        torch.nn.init.zeros_(parameter)
    observation = torch.randn(2, REFERENCE_ACTOR_OBS_V2_DIM)
    proposal = observation[:, ACTOR_OBS_V2_DIM:ACTOR_OBS_V2_DIM + ACTION_DIM]
    output = actor(TensorDict({"actor": observation}, batch_size=[2]), stochastic_output=False)
    torch.testing.assert_close(output, proposal)


def test_v2_plan_conditioned_actor_keeps_residual_through_place_only() -> None:
    actor = make_actor("cpu", REFERENCE_ACTOR_OBS_V2_DIM, plan_conditioned=True)
    for parameter in actor.mlp.parameters():
        torch.nn.init.zeros_(parameter)
    linear_layers = [
        module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)
    ]
    torch.nn.init.ones_(linear_layers[-1].bias)
    observation = torch.zeros(2, REFERENCE_ACTOR_OBS_V2_DIM)
    observation[0, 322 + 4] = 1.0  # place
    observation[1, 322 + 5] = 1.0  # release/settle

    output = actor(
        TensorDict({"actor": observation}, batch_size=[2]),
        stochastic_output=False,
    )
    expected = torch.zeros_like(output)
    expected[0, 7:14] = 1.0
    expected[0, 21:28] = 1.0
    torch.testing.assert_close(output, expected)


def test_collection_can_try_exact_base_plan_before_neighbor_ranks() -> None:
    from simple.grasp_rl.collect import reference_plan_requests

    assert reference_plan_requests(
        (6, 4, 2),
        17,
        base_reference_fallback=True,
        base_reference_first=True,
    ) == [(-1, 17), (6, None), (4, None), (2, None)]
    assert reference_plan_requests(
        (6, 4),
        17,
        base_reference_fallback=True,
        base_reference_first=False,
    ) == [(6, None), (4, None), (-1, 17)]


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
            "--evaluation-split",
            "test",
            "--randomize-target",
            "--target-position-jitter-xy",
            "0.03,0.04",
            "--target-position-offset-center-xy",
            "0.01,-0.02",
            "--target-yaw-jitter",
            "0.2",
            "--fixed-reference-episode",
            "82",
            "--fixed-base-episode",
            "82",
        ],
    )
    cli.main()
    assert captured["initialization_prefix"] is None
    assert captured["initialization_phase"] == 0.9
    assert captured["evaluation_split"] == "test"
    assert captured["randomize_target"] is True
    assert captured["target_position_jitter_xy"] == (0.03, 0.04)
    assert captured["target_position_offset_center_xy"] == (0.01, -0.02)
    assert captured["target_yaw_jitter"] == 0.2
    assert captured["fixed_reference_episode"] == 82
    assert captured["fixed_base_episode"] == 82


def test_cli_forwards_exact_v2_target_position(monkeypatch, tmp_path) -> None:
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
            "--task",
            "xmove_pick",
            "--checkpoint",
            "actor.pt",
            "--output",
            str(tmp_path),
            "--target-position-xy=-0.28,-0.06",
            "--robot-position-xy=-0.82,0.01",
        ],
    )
    cli.main()
    assert captured["target_position_xy"] == (-0.28, -0.06)
    assert captured["robot_position_xy"] == (-0.82, 0.01)
    assert captured["randomize_target"] is False


def test_cli_selects_task_scoped_dataset_defaults(monkeypatch) -> None:
    from simple.grasp_rl import cli

    captured = {}

    def fake_prepare(dataset, output, **kwargs):
        captured.update(dataset=dataset, output=output, **kwargs)
        return output

    monkeypatch.setattr(cli, "prepare_dataset", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        ["grasp-rl", "prepare", "--task", "bend_pick", "--workers", "1"],
    )
    cli.main()
    assert captured["dataset"] == get_task_spec("bend_pick").dataset_path()
    assert captured["output"] == get_task_spec("bend_pick").processed_path()
    assert captured["task"].name == "bend_pick"


def test_cli_forwards_exact_single_reference_episode(monkeypatch, tmp_path) -> None:
    from simple.grasp_rl import cli

    captured = {}

    def fake_prepare(dataset, output, **kwargs):
        captured.update(dataset=dataset, output=output, **kwargs)
        return output

    monkeypatch.setattr(cli, "prepare_v2_dataset", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grasp-rl",
            "prepare",
            "--task",
            "bend_pick_teleop",
            "--output",
            str(tmp_path),
            "--episode-id",
            "20",
            "--workers",
            "1",
        ],
    )
    cli.main()
    assert captured["episode_id"] == 20
    assert captured["episodes"] is None


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
    assert exact.is_complete is False
    exact_terms = exact.reward(observations[1], actions[0])
    assert exact.is_complete is False
    exact.reward(observations[2], actions[1])
    assert exact.is_complete is False
    exact.reward(observations[2], actions[2])
    assert exact.is_complete is True
    bad = ReferenceTracker(library)
    bad.reset(observations[0], exact_episode=0)
    bad_terms = bad.reward(
        np.zeros(ACTOR_OBS_DIM, dtype=np.float32),
        -np.ones(ACTION_DIM, dtype=np.float32),
    )
    assert exact_terms.total > bad_terms.total
    assert np.isclose(exact_terms.joint_pose, 1.0)
    assert np.isclose(exact_terms.tracker_action, 1.0)
