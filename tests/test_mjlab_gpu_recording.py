import mujoco
import torch

from simple.grasp_rl.mjlab_gpu.recording import (
    _checkpoint_provenance,
    _diagnostic_rank,
    _finger_grasp_truth,
    _target_physics_audit,
)


def test_checkpoint_provenance_uses_portable_paths(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "mjlab_gpu_metadata": {
                "config": {"backend": "mjlab_mujoco_warp"}
            },
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
