import json

import pytest
import torch

from simple.grasp_rl.mjlab_gpu.release import sha256_file, verify_release


def _checkpoint(path):
    latest = {
        "algorithm": "rsl_rl.algorithms.ppo.PPO",
        "on_policy": True,
        "rollout_reused": False,
        "transitions": 128,
        "optimizer_steps": 20,
        "actor_parameter_delta_l2": 0.1,
        "critic_parameter_delta_l2": 0.2,
    }
    torch.save(
        {
            "iter": 10,
            "mjlab_gpu_metadata": {
                "config": {"backend": "mjlab_mujoco_warp"}
            },
            "ppo_integrity": {
                "total_transitions": 1280,
                "latest_record": latest,
            },
        },
        path,
    )


def test_release_verifier_checks_hashes_and_real_ppo(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    (tmp_path / "release.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "test",
                "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": sha256_file(checkpoint),
                "artifacts": [
                    {
                        "path": "checkpoint.pt",
                        "bytes": checkpoint.stat().st_size,
                        "sha256": sha256_file(checkpoint),
                    }
                ],
            }
        )
    )

    result = verify_release(tmp_path)

    assert result["artifacts_verified"] == 1
    assert result["checkpoint"]["total_transitions"] == 1280


def test_release_verifier_rejects_corrupt_artifact(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    (tmp_path / "release.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": "bad",
                "artifacts": [
                    {
                        "path": "checkpoint.pt",
                        "bytes": checkpoint.stat().st_size,
                        "sha256": "bad",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_release(tmp_path)
