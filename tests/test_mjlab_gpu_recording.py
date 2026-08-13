from types import SimpleNamespace

import mujoco
import pytest
import torch

from simple.grasp_rl.mjlab_gpu.recording import (
    _checkpoint_provenance,
    _diagnostic_rank,
    _episode_randomization,
    _finger_grasp_truth,
    _reference_provenance,
    _target_physics_audit,
)


def test_episode_randomization_records_pose_and_dynamics() -> None:
    class _Model:
        @staticmethod
        def body(body_id):
            return SimpleNamespace(name=f"distractor_{body_id}")

    randomizer = SimpleNamespace(
        target_translation_xy=torch.tensor([[0.01, -0.02]]),
        target_yaw=torch.tensor([0.1]),
        destination_translation_xy=torch.tensor([[-0.01, 0.02]]),
        destination_yaw=torch.tensor([-0.1]),
        distractor_translation_xy=torch.tensor([[[0.02, 0.03]]]),
        distractor_yaw=torch.tensor([[0.2]]),
        distractor_body_ids=[7],
        robot_base_translation_xy=torch.tensor([[0.005, -0.006]]),
        robot_base_yaw=torch.tensor([0.02]),
        target_mass_scale=torch.tensor([1.1]),
        friction_scale=torch.tensor([0.8]),
        joint_damping_scale=torch.tensor([[0.9, 1.05]]),
        actuator_strength_scale=torch.tensor([[1.02, 0.97]]),
        action_delay_steps=torch.tensor([1]),
        sim=SimpleNamespace(mj_model=_Model()),
    )
    env = SimpleNamespace(
        num_envs=1,
        randomizer=randomizer,
        reference=SimpleNamespace(episode_rows=torch.tensor([11])),
    )

    row = _episode_randomization(env)[0]

    assert row["target_translation_xy"] == pytest.approx([0.01, -0.02])
    assert row["destination_translation_xy"] == pytest.approx([-0.01, 0.02])
    assert row["distractor_poses"]["distractor_7"]["yaw"] == pytest.approx(0.2)
    assert row["robot_base_translation_xy"] == pytest.approx([0.005, -0.006])
    assert row["target_mass_scale"] == pytest.approx(1.1)
    assert row["friction_scale"] == pytest.approx(0.8)
    assert row["joint_damping_scale"] == pytest.approx([0.9, 1.05])
    assert row["actuator_strength_scale"] == pytest.approx([1.02, 0.97])
    assert row["action_delay_steps"] == 1
    assert row["reference_episode_row"] == 11


def test_checkpoint_provenance_uses_portable_paths(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "mjlab_gpu_metadata": {"config": {"backend": "mjlab_mujoco_warp"}},
            "ppo_integrity": {
                "audit_path": "/old/machine/ppo_integrity.jsonl",
                "latest_record": {
                    "on_policy": True,
                    "algorithm": "rsl_rl.algorithms.ppo.PPO",
                },
            },
        },
        checkpoint,
    )

    result = _checkpoint_provenance(checkpoint.resolve())

    assert result["checkpoint"] == "model.pt"
    assert result["ppo_integrity"]["audit_path"] == "ppo_integrity.jsonl"


def test_checkpoint_provenance_cleanly_rejects_zero_update_checkpoint(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "model_initial.pt"
    torch.save(
        {
            "mjlab_gpu_metadata": {"config": {"backend": "mjlab_mujoco_warp"}},
            "ppo_integrity": {
                "algorithm": "rsl_rl.algorithms.ppo.PPO",
                "latest_record": None,
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="audited on-policy PPO"):
        _checkpoint_provenance(checkpoint.resolve())


def test_reference_provenance_has_absolute_auditable_source(tmp_path) -> None:
    root = tmp_path / "reference"
    root.mkdir()
    (root / "manifest.json").write_text('{"schema_version": 1}')
    reference = SimpleNamespace(
        root=root,
        metadata=lambda: {
            "data_sha256": "data-hash",
            "action_transform_sha256": "transform-hash",
            "episodes": [11],
            "strict_episode": 11,
        },
    )

    result = _reference_provenance(SimpleNamespace(reference=reference))

    assert result["checkpoint"] is None
    assert result["ppo_integrity"] is None
    assert result["reference"]["directory_absolute"] == str(root.resolve())
    assert result["reference"]["episodes"] == [11]
    assert len(result["reference"]["manifest_sha256"]) == 64


def test_diagnostic_rank_prioritizes_task_stage() -> None:
    early_high_lift = {
        "max_stage": 1,
        "max_grasp_quality": 1.0,
        "max_lift": 1.0,
    }
    later_low_lift = {
        "max_stage": 2,
        "max_grasp_quality": 0.1,
        "max_lift": 0.0,
    }

    assert _diagnostic_rank(later_low_lift) > _diagnostic_rank(early_high_lift)


def test_target_physics_audit_rejects_welded_target() -> None:
    free = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body name="target"><freejoint/>
          <geom type="sphere" size="0.01" mass="0.1"/>
        </body></worldbody></mujoco>
        """
    )
    assert _target_physics_audit(free, "target") == {
        "target_body": "target",
        "target_joint_types": ["mjJNT_FREE"],
        "target_is_free_body": True,
        "target_equality_constraint_count": 0,
        "physically_unattached": True,
    }

    welded = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody><body name="target"><freejoint/>
            <geom type="sphere" size="0.01" mass="0.1"/>
          </body></worldbody>
          <equality><weld body1="target"/></equality>
        </mujoco>
        """
    )
    audit = _target_physics_audit(welded, "target")
    assert audit["target_equality_constraint_count"] == 1
    assert not audit["physically_unattached"]


def test_finger_grasp_truth_requires_distal_closure_and_diverse_contact() -> None:
    initial = torch.zeros(3, 2, 7)
    qpos = initial.clone()
    qpos[:, 1] = torch.tensor([0.5, 0.7, 0.7, 1.0, 1.0, 0.6, 1.0])
    qpos[1, 1, 4] = 0.0
    qpos[1, 1, 6] = 0.0
    forces = torch.zeros(3, 2, 8, 3)
    forces[:, 1, 3, 0] = 3.0
    forces[:2, 1, 5, 0] = 3.0
    truth = _finger_grasp_truth(qpos, initial, forces)

    assert truth["valid_grasp"][:, 1].tolist() == [True, False, False]
