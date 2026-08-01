import torch

from simple.grasp_rl.mjlab_gpu.runner import (
    _load_policy_warm_start,
    checkpoint_uses_plan_conditioned_actor,
    ppo_train_config,
)


class _TestPolicy(torch.nn.Module):
    def __init__(self, *, policy_value: float, exploration_std: float):
        super().__init__()
        self.mlp = torch.nn.Linear(2, 2, bias=False)
        self.distribution = torch.nn.Module()
        self.distribution.register_parameter(
            "std_param",
            torch.nn.Parameter(torch.full((2,), exploration_std), requires_grad=False),
        )
        with torch.no_grad():
            self.mlp.weight.fill_(policy_value)


def test_bc_warm_start_preserves_new_ppo_exploration_distribution() -> None:
    source = _TestPolicy(policy_value=3.0, exploration_std=0.05)
    target = _TestPolicy(policy_value=0.0, exploration_std=0.01)
    legacy_state = dict(source.state_dict())
    legacy_state["_plan_conditioned_actor"] = torch.tensor(True)

    _load_policy_warm_start(target, legacy_state)

    torch.testing.assert_close(target.mlp.weight, source.mlp.weight)
    torch.testing.assert_close(target.distribution.std_param, torch.full((2,), 0.01))


def test_legacy_plan_actor_marker_selects_compatible_model(tmp_path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"actor_state_dict": {"_plan_conditioned_actor": torch.tensor(True)}},
        checkpoint,
    )

    assert checkpoint_uses_plan_conditioned_actor(checkpoint)
    config = ppo_train_config(plan_conditioned_actor=True)
    assert config["actor"]["class_name"].endswith("PlanConditionedMLPModel")
