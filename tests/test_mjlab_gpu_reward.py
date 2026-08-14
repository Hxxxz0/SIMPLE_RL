from types import SimpleNamespace

import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.goal_reward import (
    _approach_progress,
    _lift_grasp_progress,
    _place_progress,
    _supported_grasp_progress,
    _terminal_adjustment,
)
from simple.grasp_rl.mjlab_gpu.object_grasp_reward import (
    object_grasp_geometry,
    object_grasp_progress,
)
from simple.grasp_rl.mjlab_gpu.reward import GpuGraspReward, finger_closure_score
from simple.grasp_rl.schema import JOINT_NAMES
from simple.grasp_rl.task_spec import GraspLiftRewardSpec


class _Model:
    jnt_range = np.tile(np.asarray([-1.0, 1.0]), (len(JOINT_NAMES), 1))

    @staticmethod
    def joint(name: str) -> SimpleNamespace:
        return SimpleNamespace(id=JOINT_NAMES.index(name))


def _reward(num_envs: int = 2) -> GpuGraspReward:
    simulation = SimpleNamespace(
        num_envs=num_envs,
        mj_model=_Model(),
        data=SimpleNamespace(qpos=torch.zeros(num_envs, len(JOINT_NAMES))),
    )
    reader = SimpleNamespace(
        sim=simulation,
        device="cpu",
        qpos_indices=torch.arange(len(JOINT_NAMES)),
        initial_object_pos=torch.zeros(num_envs, 3),
    )
    spec = GraspLiftRewardSpec(
        goal_lift=0.025,
        success_lift=0.02,
        success_hold_steps=2,
        stalled_grasp_steps=None,
    )
    return GpuGraspReward(reader, reward_spec=spec, max_episode_steps=2)


def _state() -> SimpleNamespace:
    num_envs = 2
    identity = torch.eye(3).expand(num_envs, -1, -1).clone()
    object_position = torch.tensor([[0.0, 0.0, 0.03], [0.0, 0.0, 0.0]])
    distal = object_position[:, None, :] + torch.tensor(
        [[[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]]
    )
    link_magnitudes = torch.zeros(num_envs, 8)
    link_magnitudes[0] = 1.0
    contact = SimpleNamespace(
        link_forces_pelvis=torch.zeros(num_envs, 8, 3),
        link_force_magnitudes=link_magnitudes,
        group_forces=torch.tensor([[2.1, 2.1, 0.0], [0.0, 0.0, 0.0]]),
        object_contact_center_w=torch.zeros(num_envs, 3),
        has_object_contact_center=torch.zeros(num_envs, dtype=torch.bool),
        hand_table_force=torch.zeros(num_envs),
    )
    zeros = torch.zeros(num_envs, 3)
    contact.group_contacts = contact.group_forces > 2.0
    contact.is_grasp = contact.group_contacts[:, 0] & (
        contact.group_contacts[:, 1] | contact.group_contacts[:, 2]
    )
    return SimpleNamespace(
        object_pos_w=object_position,
        object_rot_w=identity,
        object_lin_vel_w=zeros,
        object_ang_vel_w=zeros,
        hand_pos_w=object_position,
        hand_rot_w=identity,
        hand_lin_vel_w=zeros,
        wrist_pos_w=object_position,
        wrist_lin_vel_w=zeros,
        distal_pos_w=distal,
        fingertip_surface_distances=torch.zeros(num_envs, 3),
        table_pos_w=zeros,
        table_rot_w=identity,
        pelvis_pos_w=zeros,
        pelvis_rot_w=identity,
        pelvis_height=torch.ones(num_envs),
        contact=contact,
    )


def test_gpu_reward_success_timeout_and_subset_reset_are_batched() -> None:
    reward = _reward()
    actions = torch.zeros(2, 36)
    first = reward.compute(_state(), actions, actions, torch.ones(36))
    assert not first.done.any()

    second = reward.compute(_state(), actions, actions, torch.ones(36))
    assert second.success.tolist() == [True, False]
    assert second.timeout.tolist() == [False, True]
    torch.testing.assert_close(second.terminal_adjustment, torch.tensor([20.0, -5.0]))

    reward.reset(torch.tensor([0]))
    assert reward.step_count.tolist() == [0, 2]
    assert reward.hold_count.tolist() == [0, 0]


def test_clean_reference_contact_does_not_control_success_truth() -> None:
    reward = _reward()
    reward.set_reference_contact(torch.zeros(2))
    actions = torch.zeros(2, 36)
    reward.compute(_state(), actions, actions, torch.ones(36))
    terms = reward.compute(_state(), actions, actions, torch.ones(36))
    assert terms.grail_grasp.tolist() == [0.0, 0.0]
    assert terms.success.tolist() == [True, False]


def test_goal_graph_timeout_penalty_applies_to_grasp_tasks() -> None:
    terminal = _terminal_adjustment(
        torch.tensor([True, False, False, False]),
        torch.tensor([False, True, False, False]),
        torch.tensor([False, False, True, False]),
    )
    torch.testing.assert_close(terminal, torch.tensor([40.0, -10.0, -5.0, 0.0]))


def test_goal_graph_success_bonus_is_explicitly_overridable() -> None:
    terminal = _terminal_adjustment(
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        torch.tensor([False, True]),
        success_bonus=80.0,
    )

    torch.testing.assert_close(terminal, torch.tensor([80.0, -5.0]))


def test_grasp_progress_rewards_supported_multi_finger_reach() -> None:
    distances = torch.tensor([[0.02, 0.03, 0.04]]).repeat(3, 1)
    lift = torch.tensor([0.0, -0.03, 0.0])
    quality = torch.tensor([0.0, 0.0, 0.8])
    progress = _supported_grasp_progress(distances, lift, quality)

    assert progress[0] > progress[1]
    torch.testing.assert_close(progress[2], torch.tensor(0.8))


def test_approach_progress_requires_multi_finger_alignment() -> None:
    progress, nearest = _approach_progress(
        torch.tensor([[0.01, 0.01, 0.01], [0.01, 0.10, 0.10]])
    )

    torch.testing.assert_close(nearest, torch.tensor([0.01, 0.01]))
    assert progress[0] > progress[1]


def test_lift_progress_requires_retained_force_grasp() -> None:
    progress = _lift_grasp_progress(
        torch.tensor([0.8, 0.8, 0.2]),
        torch.tensor([1.0, 0.0, 0.5]),
    )

    torch.testing.assert_close(progress, torch.tensor([0.8, 0.0, 0.1]))


def test_place_progress_requires_xy_alignment_and_lowering() -> None:
    progress = _place_progress(
        torch.tensor([0.10, 0.10, 0.30]),
        torch.tensor([0.25, 0.45, 0.25]),
        0.12,
    )

    torch.testing.assert_close(progress[0], torch.tensor(1.0))
    assert progress[0] > progress[1]
    assert progress[0] > progress[2]


def test_finger_closure_score_requires_thumb_and_opposing_distal_motion() -> None:
    initial = torch.zeros(3, 7)
    qpos = torch.tensor(
        [
            [0.5, 0.7, 0.7, 1.0, 1.0, 0.6, 1.0],
            [0.5, 0.7, 0.0, 1.0, 1.0, 0.6, 1.0],
            [0.5, 0.7, 0.7, 1.0, 0.0, 0.6, 0.0],
        ]
    )

    torch.testing.assert_close(
        finger_closure_score(qpos, initial), torch.tensor([1.0, 0.0, 0.0])
    )


def test_object_grasp_geometry_prefers_a_centered_opposing_grip() -> None:
    center = torch.zeros(2, 3)
    distal = torch.tensor(
        [
            [[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]],
            [[0.12, 0.0, 0.0], [0.12, 0.0, 0.0], [0.12, 0.0, 0.0]],
        ]
    )

    reach, opposition, nearest = object_grasp_geometry(
        distal, center, grip_width_m=0.08
    )

    assert reach[0] > reach[1]
    assert opposition[0] > opposition[1]
    torch.testing.assert_close(nearest, torch.tensor([0.04, 0.12]))


def test_object_grasp_progress_requires_reach_before_geometry() -> None:
    progress = object_grasp_progress(
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    )

    assert progress[0] > progress[1]
    assert progress[2] > progress[0]
