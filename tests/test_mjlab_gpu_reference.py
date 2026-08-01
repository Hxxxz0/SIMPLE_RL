import json

import numpy as np
import torch

from simple.grasp_rl.mjlab_gpu.config import ReferenceNoiseConfig
from simple.grasp_rl.mjlab_gpu.reference import GpuReferenceLibrary
from simple.grasp_rl.reference import build_reference_context
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_FUTURE_OFFSETS,
)


def _reference_data(tmp_path):
    root = tmp_path / "processed"
    source = root / "bc"
    source.mkdir(parents=True)
    length = 60
    observations = np.zeros((length, ACTOR_OBS_DIM), dtype=np.float32)
    actions = np.linspace(-0.5, 0.5, length * ACTION_DIM, dtype=np.float32).reshape(
        length, ACTION_DIM
    )
    observations[:, 132] = np.linspace(0.0, 0.2, length)
    observations[:, 147:165] = 0.1
    observations[1:, 165 + 3 : 165 + 6] = 3.0
    observations[1:, 165 + 12 : 165 + 15] = 3.0
    (root / "manifest.json").write_text(
        json.dumps({"splits": {"train": [0], "val": [], "test": []}})
    )
    np.savez(
        source / "episode_000000.npz",
        observations=observations,
        raw_actions=actions,
    )
    return root, observations, actions


def test_gpu_clean_reference_context_matches_legacy_builder(tmp_path) -> None:
    root, observations, actions = _reference_data(tmp_path)
    library = GpuReferenceLibrary(root, num_envs=1, device="cpu")
    current = torch.from_numpy(observations[:1])
    library.reset(current, episode_rows=torch.tensor([0]))
    torch.testing.assert_close(
        library.current_action()[0], torch.from_numpy(actions[0])
    )
    clean = library.clean_context(current)
    expected = build_reference_context(observations, actions, 0, observations[0])
    torch.testing.assert_close(clean[0], torch.from_numpy(expected))


def test_reference_noise_changes_policy_view_but_not_clean_contact_truth(
    tmp_path,
) -> None:
    root, observations, _ = _reference_data(tmp_path)
    library = GpuReferenceLibrary(root, num_envs=1, device="cpu")
    current = torch.from_numpy(observations[:1]).clone()
    current[:, 132:134] += torch.tensor([0.1, -0.05])
    library.reset(current, episode_rows=torch.tensor([0]))
    noisy, clean = library.policy_context(
        current,
        ReferenceNoiseConfig(future_dropout_probability=0.0),
        training=True,
        generator=torch.Generator().manual_seed(3),
    )
    frames = clean[:, :-1].reshape(
        1, len(REFERENCE_FUTURE_OFFSETS), REFERENCE_FRAME_DIM
    )
    noisy_frames = noisy[:, :-1].reshape_as(frames)
    torch.testing.assert_close(
        frames[:, 0, ACTION_DIM : ACTION_DIM + 3], torch.zeros(1, 3)
    )
    assert not torch.equal(noisy, clean)
    torch.testing.assert_close(noisy_frames[..., -1], frames[..., -1])
    assert library.post_step_contact_label().item() == 1.0


def test_locked_eval_reference_is_exactly_clean(tmp_path) -> None:
    root, observations, _ = _reference_data(tmp_path)
    library = GpuReferenceLibrary(root, num_envs=1, device="cpu")
    current = torch.from_numpy(observations[:1])
    library.reset(current, episode_rows=torch.tensor([0]))
    policy, clean = library.policy_context(
        current, ReferenceNoiseConfig(), training=False
    )
    assert policy.data_ptr() == clean.data_ptr()


def test_training_reference_noise_is_stable_within_one_simulation_step(
    tmp_path,
) -> None:
    root, observations, _ = _reference_data(tmp_path)
    library = GpuReferenceLibrary(root, num_envs=1, device="cpu")
    current = torch.from_numpy(observations[:1])
    library.reset(current, episode_rows=torch.tensor([0]))
    generator = torch.Generator().manual_seed(5)
    first, _ = library.policy_context(
        current, ReferenceNoiseConfig(), training=True, generator=generator
    )
    second, _ = library.policy_context(
        current, ReferenceNoiseConfig(), training=True, generator=generator
    )
    assert first.data_ptr() == second.data_ptr()
    library.advance()
    third, _ = library.policy_context(
        current, ReferenceNoiseConfig(), training=True, generator=generator
    )
    assert not torch.equal(first, third)
