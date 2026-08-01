from types import SimpleNamespace

import numpy as np

from simple.grasp_rl.env import GraspRlEnv
from simple.grasp_rl.task_spec import get_task_spec


def test_training_physics_dr_restores_baseline_before_resampling(monkeypatch) -> None:
    model = SimpleNamespace(
        nbody=3,
        body_mass=np.array([0.0, 2.0, 4.0]),
        geom_friction=np.array([[1.0, 0.1, 0.01], [2.0, 0.2, 0.02]]),
    )
    state = SimpleNamespace(
        target_id=1,
        table_id=2,
        target_geom_ids=(0,),
        _is_descendant=lambda body, ancestor: body == ancestor,
        _subtree_geom_ids=lambda body: (1,) if body == 2 else (),
    )
    env = GraspRlEnv.__new__(GraspRlEnv)
    env.state = state
    env.task_spec = get_task_spec("tabletop_grasp")
    env.sim = SimpleNamespace(mjModel=model, mjData=object())
    env._training_dr_model = None
    env._training_dr_body_mass = None
    env._training_dr_geom_friction = None
    monkeypatch.setattr("simple.grasp_rl.env.mujoco.mj_forward", lambda *_: None)

    env.randomize_training_physics(1.5, 0.5)
    np.testing.assert_allclose(model.body_mass, [0.0, 3.0, 4.0])
    np.testing.assert_allclose(
        model.geom_friction,
        [[0.5, 0.05, 0.005], [1.0, 0.1, 0.01]],
    )

    env.randomize_training_physics(1.0, 1.0)
    np.testing.assert_allclose(model.body_mass, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(
        model.geom_friction,
        [[1.0, 0.1, 0.01], [2.0, 0.2, 0.02]],
    )
