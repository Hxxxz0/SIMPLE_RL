import json

import numpy as np
import pytest
import torch

from simple.grasp_rl.mjlab_gpu.config import ReferenceNoiseConfig
from simple.grasp_rl.mjlab_gpu.reference import (
    GpuReferenceLibrary,
    derive_strict_reference_subset,
    validate_strict_reference_manifest,
)
from simple.grasp_rl.reference import build_reference_context
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    ACTOR_OBS_V2_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_FRAME_V2_DIM,
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


def test_strict_reference_manifest_rejects_trajectory_leakage() -> None:
    manifest = {
        "unique_episodes": [20],
        "source_episode_ids": [20],
        "requested_episode_id": 20,
        "splits": {"train": [20], "val": [], "test": []},
        "reports": [{"episode": 20, "success": True}],
    }
    validate_strict_reference_manifest(manifest, 20)
    manifest["unique_episodes"] = [20, 21]
    with pytest.raises(ValueError, match="only episode 20"):
        validate_strict_reference_manifest(manifest, 20)


def test_strict_reference_subset_preserves_source_transform(tmp_path) -> None:
    root = tmp_path / "multi"
    (root / "bc").mkdir(parents=True)
    transform = b"same-action-transform"
    (root / "action_transform.npz").write_bytes(transform)
    for episode in (7, 82):
        np.savez(
            root / "bc" / f"episode_{episode:06d}.npz",
            observations=np.zeros((2, ACTOR_OBS_DIM), dtype=np.float32),
            raw_actions=np.zeros((2, ACTION_DIM), dtype=np.float32),
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "unique_episodes": [7, 82],
                "source_episode_ids": [7, 82],
                "splits": {"train": [7, 82], "val": [], "test": []},
                "reports": [
                    {"episode": 7, "success": True},
                    {"episode": 82, "success": True},
                ],
            }
        )
    )

    output = tmp_path / "single"
    result = derive_strict_reference_subset(root, output, 82)
    manifest = json.loads((output / "manifest.json").read_text())

    validate_strict_reference_manifest(manifest, 82)
    assert (output / "action_transform.npz").read_bytes() == transform
    assert result["action_transform_sha256"] == manifest[
        "action_transform_sha256"
    ]
    assert not (output / "bc" / "episode_000007.npz").exists()


def test_balanced_rows_cover_library_and_alignment_gate_rejects_bad_scene(
    tmp_path,
) -> None:
    root = tmp_path / "multi"
    source = root / "bc"
    source.mkdir(parents=True)
    episodes = [3, 7, 11]
    (root / "manifest.json").write_text(
        json.dumps({"splits": {"train": episodes, "val": [], "test": []}})
    )
    for row, episode in enumerate(episodes):
        observations = np.zeros((8, ACTOR_OBS_DIM), dtype=np.float32)
        observations[:, 132] = 0.01 * row
        np.savez(
            source / f"episode_{episode:06d}.npz",
            observations=observations,
            raw_actions=np.zeros((8, ACTION_DIM), dtype=np.float32),
        )
    library = GpuReferenceLibrary(root, num_envs=8, device="cpu")
    rows = library.balanced_rows(
        8, generator=torch.Generator().manual_seed(17)
    )
    counts = torch.bincount(rows, minlength=3)
    assert sorted(counts.tolist()) == [2, 3, 3]
    one_at_a_time = [
        int(
            library.balanced_rows(
                1, generator=torch.Generator().manual_seed(seed)
            ).item()
        )
        for seed in range(3)
    ]
    assert one_at_a_time == [2, 0, 1]

    observation = torch.zeros(ACTOR_OBS_DIM)
    alignment = library.validate_initial_position_alignment(observation, 0.021)
    assert alignment["maximum_episode"] == 11
    assert alignment["maximum_metres"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="initial-position mismatch"):
        library.validate_initial_position_alignment(observation, 0.019)


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


def test_gpu_clean_reference_context_matches_v2_builder(tmp_path) -> None:
    root = tmp_path / "processed_v2"
    source = root / "bc"
    source.mkdir(parents=True)
    length = 60
    observations = np.zeros((length, ACTOR_OBS_V2_DIM), dtype=np.float32)
    actions = np.linspace(-0.25, 0.25, length * ACTION_DIM, dtype=np.float32).reshape(
        length, ACTION_DIM
    )
    observations[:, 89] = 0.1
    observations[:, 94] = 0.2
    observations[:, 163] = np.linspace(0.0, 0.3, length)
    observations[:, 201] = 0.4
    observations[:, 301] = 1.0
    observations[:, 308] = 1.0
    observations[:, 311] = np.linspace(1.0, 0.0, length)
    observations[:, 322] = 1.0
    (root / "manifest.json").write_text(
        json.dumps({"splits": {"train": [0], "val": [], "test": []}})
    )
    np.savez(
        source / "episode_000000.npz",
        observations=observations,
        raw_actions=actions,
    )
    library = GpuReferenceLibrary(root, num_envs=1, device="cpu")
    current = torch.from_numpy(observations[:1]).clone()
    library.reset(current, episode_rows=torch.tensor([0]))
    expected = build_reference_context(observations, actions, 0, current[0].numpy())
    torch.testing.assert_close(
        library.clean_context(current)[0], torch.from_numpy(expected)
    )
    assert library.metadata()["observation_dim"] == ACTOR_OBS_V2_DIM


def test_v2_reference_retargets_proposal_from_observed_target_pose(tmp_path) -> None:
    root = tmp_path / "retarget_v2"
    source = root / "bc"
    source.mkdir(parents=True)
    observations = np.zeros((60, ACTOR_OBS_V2_DIM), dtype=np.float32)
    actions = np.zeros((60, ACTION_DIM), dtype=np.float32)
    (root / "manifest.json").write_text(
        json.dumps({"splits": {"train": [0], "val": [], "test": []}})
    )
    np.savez(
        source / "episode_000000.npz",
        observations=observations,
        raw_actions=actions,
    )
    library = GpuReferenceLibrary(
        root,
        num_envs=1,
        device="cpu",
        target_x_arm_gains=(-10.0, 2.0),
        target_y_arm_gains=(5.0, -3.5),
        target_yaw_arm_gains=(1.0, -2.0),
    )
    current = torch.from_numpy(observations[:1]).clone()
    current[:, 163] = 0.02
    current[:, 164] = -0.03
    library.reset(
        current,
        episode_rows=torch.tensor([0]),
        target_yaw_offset=torch.tensor([0.1]),
    )

    torch.testing.assert_close(library.current_action()[0, 21], torch.tensor(-0.2))
    torch.testing.assert_close(library.current_action()[0, 24], torch.tensor(0.04))
    torch.testing.assert_close(library.current_action()[0, 23], torch.tensor(-0.05))
    torch.testing.assert_close(library.current_action()[0, 27], torch.tensor(-0.095))
    frames = library.clean_context(current)[:, :-1].reshape(
        1, len(REFERENCE_FUTURE_OFFSETS), REFERENCE_FRAME_V2_DIM
    )
    torch.testing.assert_close(frames[0, 0, 21], torch.tensor(-0.2))
    torch.testing.assert_close(frames[0, 0, 24], torch.tensor(0.04))
    torch.testing.assert_close(frames[0, 0, 23], torch.tensor(-0.05))
    torch.testing.assert_close(frames[0, 0, 27], torch.tensor(-0.095))


def test_v2_reference_can_retarget_positive_y_with_separate_gains(tmp_path) -> None:
    root = tmp_path / "retarget_positive_y_v2"
    source = root / "bc"
    source.mkdir(parents=True)
    observations = np.zeros((60, ACTOR_OBS_V2_DIM), dtype=np.float32)
    actions = np.zeros((60, ACTION_DIM), dtype=np.float32)
    (root / "manifest.json").write_text(
        json.dumps({"splits": {"train": [0], "val": [], "test": []}})
    )
    np.savez(
        source / "episode_000000.npz",
        observations=observations,
        raw_actions=actions,
    )
    library = GpuReferenceLibrary(
        root,
        num_envs=2,
        device="cpu",
        target_y_arm_gains=(12.0, -2.0),
        target_positive_y_arm_gains=(22.0, 3.0),
    )
    current = torch.from_numpy(observations[:2]).clone()
    current[:, 164] = torch.tensor([-0.01, 0.01])
    library.reset(current, episode_rows=torch.tensor([0, 0]))

    actions = library.current_action()
    torch.testing.assert_close(actions[:, 23], torch.tensor([-0.12, 0.22]))
    torch.testing.assert_close(actions[:, 27], torch.tensor([0.02, 0.03]))
    assert library.metadata()["target_positive_y_arm_gains"] == [22.0, 3.0]


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
