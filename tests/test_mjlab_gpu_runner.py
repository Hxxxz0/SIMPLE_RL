from types import SimpleNamespace

import torch
from tensordict import TensorDict

from simple.grasp_rl.mjlab_gpu.cli import (
    _initialize_scratch_correction_head,
    _parser,
)
from simple.grasp_rl.mjlab_gpu.runner import (
    GpuPpoRunner,
    _load_policy_warm_start,
    _reference_metadata_matches,
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


class _AnchorPolicy(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2, 36, bias=False))
        self.distribution = torch.nn.Module()
        self.distribution.register_parameter(
            "std_param",
            torch.nn.Parameter(torch.ones(36), requires_grad=False),
        )
        with torch.no_grad():
            self.mlp[0].weight.fill_(value)

    def forward(self, observations, *, stochastic_output: bool = False):
        del stochastic_output
        return self.mlp(observations["actor"])


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


def test_train_config_allows_explicit_exploration_std() -> None:
    config = ppo_train_config(exploration_std=0.05)
    assert config["actor"]["distribution_cfg"]["init_std"] == 0.05


def test_train_config_preserves_legacy_independent_exploration_by_default() -> None:
    distribution = ppo_train_config()["actor"]["distribution_cfg"]
    assert distribution["class_name"] == "GaussianDistribution"
    assert "hold_steps" not in distribution


def test_train_config_can_enable_temporally_correlated_exploration() -> None:
    distribution = ppo_train_config(exploration_hold_steps=8)["actor"][
        "distribution_cfg"
    ]
    assert distribution["class_name"].endswith(
        "TemporallyCorrelatedGaussianDistribution"
    )
    assert distribution["hold_steps"] == 8


def test_actor_anchor_is_disabled_by_default() -> None:
    args = _parser().parse_args(
        [
            "train",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "references",
            "--output",
            "run",
        ]
    )
    assert args.actor_anchor_checkpoint is None
    assert args.actor_anchor_weight == 0.0


def test_actor_anchor_adds_teacher_loss_to_existing_ppo_step(tmp_path) -> None:
    teacher = _AnchorPolicy(0.0)
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {
            "actor_state_dict": teacher.state_dict(),
            "mjlab_gpu_metadata": {
                "config": {"resolved": {"task": "grasp_anything"}},
                "asset_manifest_hash": "asset-hash",
            },
        },
        checkpoint,
    )

    actor = _AnchorPolicy(0.0)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
    observations = TensorDict(
        {"actor": torch.ones(8, 2)}, batch_size=[8], device="cpu"
    )

    def update():
        optimizer.zero_grad()
        prediction = actor(observations, stochastic_output=True)
        prediction.mean().backward()
        optimizer.step()
        return {"ppo": 1.0}

    algorithm = SimpleNamespace(
        optimizer=optimizer,
        update=update,
        get_policy=lambda: actor,
        max_grad_norm=1.0,
    )
    runner = object.__new__(GpuPpoRunner)
    runner.device = "cpu"
    runner.alg = algorithm
    runner.env = SimpleNamespace(
        config=SimpleNamespace(task="grasp_anything"),
        gpu=SimpleNamespace(
            bundle=SimpleNamespace(
                root=tmp_path,
                manifest={
                    "manifest_hash": "asset-hash",
                    "action_transform": "transform.json",
                },
            )
        ),
        reference=SimpleNamespace(observation_dim=2, context_dim=0),
    )
    runner._install_actor_anchor(checkpoint, weight=10.0)
    with torch.no_grad():
        actor.mlp[0].weight.fill_(0.1)

    metrics = algorithm.update()

    assert metrics["ppo"] == 1.0
    assert metrics["actor_anchor"] > 0.0
    assert runner.actor_anchor_metadata["weight"] == 10.0
    assert all(
        parameter.grad is None
        for parameter in runner._actor_anchor_teacher.parameters()
    )


def test_scratch_correction_initialization_changes_only_final_head() -> None:
    policy = torch.nn.Module()
    policy.mlp = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.ELU(), torch.nn.Linear(4, 36)
    )
    first_weight = policy.mlp[0].weight.detach().clone()
    final_weight = policy.mlp[2].weight.detach().clone()
    bias = torch.linspace(-0.3, 0.3, 36)

    _initialize_scratch_correction_head(
        policy, output_scale=0.01, correction_bias=bias
    )

    torch.testing.assert_close(policy.mlp[0].weight, first_weight)
    torch.testing.assert_close(policy.mlp[2].weight, final_weight * 0.01)
    torch.testing.assert_close(policy.mlp[2].bias, bias)


def test_zero_retarget_accepts_legacy_reference_metadata() -> None:
    legacy = {"source": "bc", "observation_dim": 331}
    expected = {
        **legacy,
        "target_x_arm_gains": [0.0, 0.0],
        "target_y_arm_gains": [0.0, 0.0],
        "target_positive_y_arm_gains": None,
        "target_yaw_arm_gains": [0.0, 0.0],
        "strict_episode": None,
        "action_transform_sha256": "abc",
    }
    assert _reference_metadata_matches(legacy, expected)
    assert not _reference_metadata_matches(
        legacy, {**legacy, "target_x_arm_gains": [-10.0, 2.0]}
    )
    assert not _reference_metadata_matches(
        legacy, {**legacy, "target_y_arm_gains": [5.0, -3.5]}
    )
    assert not _reference_metadata_matches(
        legacy, {**legacy, "target_positive_y_arm_gains": [8.0, -1.5]}
    )
    assert not _reference_metadata_matches(
        legacy, {**legacy, "target_yaw_arm_gains": [1.0, -2.0]}
    )
