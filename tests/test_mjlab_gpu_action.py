import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.action import (
    GpuActionTransform,
    sonic_upper_mappings,
    tracker_hand_targets,
)
from simple.grasp_rl.schema import JOINT_NAMES
from simple.grasp_rl.tracker import ActionTransform


def _transforms(num_envs: int = 4) -> tuple[ActionTransform, GpuActionTransform]:
    center = np.linspace(-0.2, 0.2, 36, dtype=np.float32)
    low = center - 0.4
    high = center + 0.6
    max_delta = np.linspace(0.01, 0.08, 36, dtype=np.float32)
    cpu = ActionTransform(center, low, high, max_delta)
    gpu = GpuActionTransform(
        center=torch.from_numpy(center),
        low=torch.from_numpy(low),
        high=torch.from_numpy(high),
        max_delta=torch.from_numpy(max_delta),
        initial_action=torch.from_numpy(center),
        num_envs=num_envs,
        device="cpu",
    )
    return cpu, gpu


def test_gpu_action_decode_and_encode_match_legacy_transform() -> None:
    cpu, gpu = _transforms()
    generator = np.random.default_rng(7)
    previous = np.repeat(cpu.center[None], 4, axis=0)
    for _ in range(5):
        raw = generator.uniform(-1.5, 1.5, size=(4, 36)).astype(np.float32)
        expected = np.stack(
            [cpu.decode(item, previous[index]) for index, item in enumerate(raw)]
        )
        actual = gpu.decode(torch.from_numpy(raw))
        torch.testing.assert_close(actual, torch.from_numpy(expected))
        torch.testing.assert_close(
            gpu.encode(actual), torch.from_numpy(cpu.encode(expected))
        )
        previous = expected


def test_gpu_action_delay_and_subset_reset_are_per_environment() -> None:
    _, transform = _transforms(num_envs=2)
    raw = torch.ones(2, 36)
    initial = transform.previous_action.clone()
    first = transform.decode(raw, torch.tensor([1, 0]))
    torch.testing.assert_close(first[0], initial[0])
    assert not torch.equal(first[1], initial[1])

    delayed = transform.pending_action[0].clone()
    second = transform.decode(raw, torch.tensor([1, 0]))
    torch.testing.assert_close(second[0], delayed)
    untouched = transform.previous_action[1].clone()
    transform.reset(torch.tensor([0]))
    torch.testing.assert_close(transform.previous_action[0], initial[0])
    torch.testing.assert_close(transform.previous_action[1], untouched)


def test_tracker_hand_targets_reorders_only_left_index_and_middle() -> None:
    marker = torch.arange(36, dtype=torch.float32).reshape(1, -1)
    expected = torch.tensor(
        [[0, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 13]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(tracker_hand_targets(marker), expected)


def test_sonic_upper_mapping_routes_every_named_finger_to_itself() -> None:
    upper_names = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        *JOINT_NAMES[15:29],
        *JOINT_NAMES[39:43],
        *JOINT_NAMES[36:39],
        *JOINT_NAMES[32:36],
        *JOINT_NAMES[29:32],
    ]
    action_indices, output_mapping = sonic_upper_mappings(upper_names)

    assert [upper_names[index] for index in output_mapping] == list(JOINT_NAMES[15:])
    by_name = dict(zip(upper_names, action_indices, strict=True))
    assert [by_name[name] for name in JOINT_NAMES[36:43]] == list(range(7, 14))
