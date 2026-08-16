from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from simple.grasp_rl.mjlab_gpu.cli import (
    _bootstrap_gate_rollout,
    _initialize_scratch_correction_head,
    _load_evaluation_actor,
    _parser,
    _spatial_checkpoint_routes,
)
from simple.grasp_rl.mjlab_gpu.runner import (
    GpuPpoRunner,
    SpatialCheckpointRouter,
    SpatialPolicyRouter,
    _asset_allows_policy_warm_start,
    _load_policy_warm_start,
    _reference_metadata_matches,
    _spatial_advantage_metadata_matches,
    _spatially_balance_advantages,
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


class _ResidualPolicy(_AnchorPolicy):
    def __init__(self, mask: torch.Tensor):
        super().__init__(1.0)
        self.register_buffer("_residual_action_mask", mask)


class _GateReference:
    @staticmethod
    def current_action() -> torch.Tensor:
        return torch.zeros(2, 36)


class _GateEnv:
    num_envs = 2
    max_episode_length = 3
    device = "cpu"

    def __init__(self) -> None:
        self.reference = _GateReference()
        self.last_terms = None
        self.steps = 0
        self.controller_state = torch.zeros(1)

    @staticmethod
    def get_observations() -> TensorDict:
        return TensorDict(
            {
                "actor": torch.zeros(2, 2),
                "target_position_cell_id": torch.tensor([7, 9]),
            },
            batch_size=[2],
        )

    def step(self, actions: torch.Tensor):
        assert actions.shape == (2, 36)
        self.steps += 1
        # The real controller also replaces persistent tensors during env steps.
        self.controller_state = torch.full((1,), float(self.steps))
        # World zero auto-resets and finishes again before slow world one.
        done = torch.tensor([self.steps in (1, 2), self.steps == 3])
        self.last_terms = SimpleNamespace(
            success=torch.tensor([self.steps in (1, 2), False]),
            failure=torch.tensor([False, self.steps == 3]),
            timeout=torch.zeros(2, dtype=torch.bool),
        )
        return self.get_observations(), torch.zeros(2), done, {}

    def reset_controller(self) -> None:
        self.controller_state.zero_()


def test_bc_warm_start_preserves_new_ppo_exploration_distribution() -> None:
    source = _TestPolicy(policy_value=3.0, exploration_std=0.05)
    target = _TestPolicy(policy_value=0.0, exploration_std=0.01)
    legacy_state = dict(source.state_dict())
    legacy_state["_plan_conditioned_actor"] = torch.tensor(True)

    _load_policy_warm_start(target, legacy_state)

    torch.testing.assert_close(target.mlp.weight, source.mlp.weight)
    torch.testing.assert_close(target.distribution.std_param, torch.full((2,), 0.01))


def test_warm_start_zeroes_only_newly_enabled_residual_head_rows() -> None:
    legacy_mask = torch.zeros(36, dtype=torch.bool)
    legacy_mask[7:14] = True
    legacy_mask[21:28] = True
    expanded_mask = legacy_mask.clone()
    expanded_mask[28:34] = True
    source = _ResidualPolicy(legacy_mask)
    target = _ResidualPolicy(expanded_mask)

    _load_policy_warm_start(target, source.state_dict())

    torch.testing.assert_close(
        target.mlp[0].weight[legacy_mask], source.mlp[0].weight[legacy_mask]
    )
    assert torch.count_nonzero(target.mlp[0].weight[28:34]) == 0
    assert target.mlp[0].bias is None


def test_workspace_asset_allows_only_declared_parent_warm_start() -> None:
    manifest = {
        "manifest_hash": "workspace",
        "warm_start_compatible_manifest_hashes": ["stable-parent"],
    }
    assert _asset_allows_policy_warm_start(manifest, "workspace")
    assert _asset_allows_policy_warm_start(manifest, "stable-parent")
    assert not _asset_allows_policy_warm_start(manifest, "unrelated")


def test_workspace_siblings_allow_only_identical_policy_contracts() -> None:
    invariant = {
        "format_version": 1,
        "task": "grasp_anything",
        "controller": "sonic_wbc",
        "controller_bundle": {"state_sha256": "controller"},
        "model": {"nq": 78, "nv": 73},
        "roles": {"primary": "target", "support": "table"},
        "reset": {"qpos": [0.0]},
        "action_transform_sha256": "actions",
        "reward_hash": "reward",
        "object_contract": {"object_id": "Apple_1"},
        "task_metadata": {"observation_schema": "task_privileged_v2"},
    }
    source = {
        **invariant,
        "manifest_hash": "workspace-05cm",
        "workspace_support_contract": {"source_manifest_hash": "stable-root"},
    }
    target = {
        **invariant,
        "manifest_hash": "workspace-10cm",
        "workspace_support_contract": {
            "root_source_manifest_hash": "stable-root"
        },
    }

    assert _asset_allows_policy_warm_start(target, "workspace-05cm", source)
    assert not _asset_allows_policy_warm_start(
        target,
        "workspace-05cm",
        {**source, "object_contract": {"object_id": "Tomato_1"}},
    )
    assert not _asset_allows_policy_warm_start(
        target,
        "workspace-05cm",
        {**source, "model": {"nq": 79, "nv": 73}},
    )


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


def test_train_config_rejects_temporally_correlated_ppo_exploration() -> None:
    with pytest.raises(ValueError, match="per-step likelihood ratios"):
        ppo_train_config(exploration_hold_steps=8)


def test_train_config_can_override_exploration_by_action_group() -> None:
    distribution = ppo_train_config(
        exploration_std=0.01,
        exploration_group_stds=(("right_hand", 0.08), ("right_arm", 0.12)),
    )["actor"]["distribution_cfg"]

    assert distribution["class_name"].endswith("ActionGroupedGaussianDistribution")
    assert distribution["init_std"] == pytest.approx(0.01)
    assert distribution["action_group_stds"] == (
        ("right_hand", 0.08),
        ("right_arm", 0.12),
    )


def test_train_config_can_learn_exploration_and_enable_entropy() -> None:
    config = ppo_train_config(
        plan_conditioned_actor=True,
        learn_exploration_std=True,
        entropy_coef=0.01,
        residual_action_groups=("right_hand", "right_arm", "torso_rpy"),
    )
    assert config["actor"]["distribution_cfg"]["learn_std"]
    assert config["actor"]["residual_action_groups"][-1] == "torso_rpy"
    assert config["algorithm"]["entropy_coef"] == pytest.approx(0.01)


def test_spatial_advantages_are_normalized_per_cell_and_count_balanced() -> None:
    advantages = torch.tensor([[[1.0], [3.0], [10.0], [20.0], [30.0]]])
    cells = torch.tensor([[0, 0, 1, 1, 1]])

    balanced, counts = _spatially_balance_advantages(advantages, cells, num_cells=2)

    assert counts.tolist() == [2, 3]
    for cell in range(2):
        selected = balanced.reshape(-1)[cells.reshape(-1) == cell]
        assert selected.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert balanced.abs().sum().item() > 0.0


def test_spatial_advantages_can_preserve_focused_sample_weight() -> None:
    advantages = torch.tensor([[[1.0], [3.0], [10.0], [20.0], [30.0]]])
    cells = torch.tensor([[0, 0, 1, 1, 1]])

    balanced, counts = _spatially_balance_advantages(
        advantages, cells, num_cells=2, weighting="sample"
    )

    assert counts.tolist() == [2, 3]
    for cell in range(2):
        selected = balanced.reshape(-1)[cells.reshape(-1) == cell]
        assert selected.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert selected.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)


def test_spatial_advantages_fail_when_a_required_cell_is_missing() -> None:
    with pytest.raises(RuntimeError, match="missed required workspace cells"):
        _spatially_balance_advantages(
            torch.ones(1, 2, 1), torch.zeros(1, 2, dtype=torch.long), num_cells=2
        )


def test_legacy_spatial_advantage_metadata_defaults_to_cell_weighting() -> None:
    assert _spatial_advantage_metadata_matches(
        {"grid": [8, 8]}, {"grid": [8, 8], "weighting": "cell"}
    )
    assert not _spatial_advantage_metadata_matches(
        {"grid": [8, 8]}, {"grid": [8, 8], "weighting": "sample"}
    )


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
    assert args.bootstrap_gate_episodes == 0
    assert args.bootstrap_gate_spatial_scope == "all"


def test_bootstrap_gate_counts_deterministic_reference_outcomes() -> None:
    env = _GateEnv()
    result = _bootstrap_gate_rollout(env, actor=None, episodes=2)

    assert result == {
        "episodes": 2,
        "successes": 1,
        "failures": 1,
        "timeouts": 0,
        "success_rate": 0.5,
        "vector_steps": 3,
    }
    assert not torch.is_inference(env.controller_state)
    env.reset_controller()


def test_bootstrap_gate_rejects_more_episodes_than_unique_worlds() -> None:
    with pytest.raises(ValueError, match="must not exceed num_envs"):
        _bootstrap_gate_rollout(_GateEnv(), actor=None, episodes=3)


def test_bootstrap_gate_reports_exact_requested_spatial_cell() -> None:
    result = _bootstrap_gate_rollout(
        _GateEnv(), actor=None, episodes=1, spatial_cell_ids=(9,)
    )

    assert result["success_rate"] == 0.0
    assert result["minimum_spatial_cell_success_rate"] == 0.0
    assert result["spatial_cells"] == [
        {
            "cell_id": 9,
            "episodes": 1,
            "successes": 0,
            "failures": 1,
            "timeouts": 0,
            "success_rate": 0.0,
        }
    ]


def test_bootstrap_gate_rejects_unassigned_spatial_cell() -> None:
    with pytest.raises(ValueError, match="has no assigned worlds"):
        _bootstrap_gate_rollout(
            _GateEnv(), actor=None, episodes=1, spatial_cell_ids=(10,)
        )


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
    observations = TensorDict({"actor": torch.ones(8, 2)}, batch_size=[8], device="cpu")

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


def test_actor_anchor_exempts_selected_spatial_cells(tmp_path) -> None:
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
        {
            "actor": torch.ones(8, 2),
            "target_position_cell_id": torch.zeros(8, dtype=torch.long),
        },
        batch_size=[8],
        device="cpu",
    )

    def update():
        optimizer.zero_grad()
        actor(observations, stochastic_output=True).mean().backward()
        optimizer.step()
        return {"ppo": 1.0}

    runner = object.__new__(GpuPpoRunner)
    runner.device = "cpu"
    runner.alg = SimpleNamespace(
        optimizer=optimizer,
        update=update,
        get_policy=lambda: actor,
        max_grad_norm=1.0,
    )
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
    runner._install_actor_anchor(
        checkpoint, weight=10.0, free_spatial_cell_ids=(0,)
    )
    with torch.no_grad():
        actor.mlp[0].weight.fill_(0.1)

    metrics = runner.alg.update()

    assert metrics["ppo"] == 1.0
    assert metrics["actor_anchor"] == 0.0
    assert runner.actor_anchor_metadata["free_spatial_cell_ids"] == [0]


def test_spatial_policy_router_uses_learner_only_in_free_cells(tmp_path) -> None:
    learner = _AnchorPolicy(1.0)
    teacher = _AnchorPolicy(0.0)
    observations = TensorDict(
        {
            "actor": torch.ones(2, 2),
            "target_position_cell_id": torch.tensor([3, 4]),
        },
        batch_size=[2],
    )
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    router = SpatialPolicyRouter(
        learner,
        teacher,
        free_spatial_cell_ids=(3,),
        teacher_checkpoint=checkpoint,
    )

    actions = router(observations, stochastic_output=False)

    torch.testing.assert_close(actions[0], torch.full((36,), 2.0))
    torch.testing.assert_close(actions[1], torch.zeros(36))
    assert router.metadata()["free_spatial_cell_ids"] == [3]


def test_evaluation_loader_moves_spatial_router_buffers_to_device(
    monkeypatch, tmp_path
) -> None:
    learner = _AnchorPolicy(1.0)
    teacher = _AnchorPolicy(0.0)
    runner = SimpleNamespace(
        alg=SimpleNamespace(get_policy=lambda: learner),
        load_actor_warm_start=lambda _checkpoint: None,
        frozen_actor_copy=lambda _checkpoint: teacher,
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.GpuPpoRunner",
        lambda *_args, **_kwargs: runner,
    )
    monkeypatch.setattr(
        "simple.grasp_rl.mjlab_gpu.cli.checkpoint_uses_plan_conditioned_actor",
        lambda _checkpoint: False,
    )
    checkpoint = tmp_path / "learner.pt"
    teacher_checkpoint = tmp_path / "teacher.pt"
    args = SimpleNamespace(
        spatial_router_teacher_checkpoint=teacher_checkpoint,
        spatial_router_free_cell=[(3, 0)],
        target_position_stratified_grid=(8, 8),
    )
    config = SimpleNamespace(
        device="meta", residual_action_groups=("right_hand", "right_arm")
    )

    actor = _load_evaluation_actor(None, config, checkpoint, args)

    assert actor.free_spatial_cell_ids.device.type == "meta"
    assert next(actor.learner.parameters()).device.type == "meta"
    assert next(actor.teacher.parameters()).device.type == "meta"


def test_spatial_checkpoint_router_uses_disjoint_experts(tmp_path) -> None:
    default = _AnchorPolicy(0.0)
    first = _AnchorPolicy(1.0)
    second = _AnchorPolicy(2.0)
    first_checkpoint = tmp_path / "first.pt"
    second_checkpoint = tmp_path / "second.pt"
    first_checkpoint.write_bytes(b"first")
    second_checkpoint.write_bytes(b"second")
    router = SpatialCheckpointRouter(
        default,
        (
            (first_checkpoint, first, (3,)),
            (second_checkpoint, second, (4,)),
        ),
    )
    observations = TensorDict(
        {
            "actor": torch.ones(3, 2),
            "target_position_cell_id": torch.tensor([3, 4, 5]),
        },
        batch_size=[3],
    )

    actions = router(observations, stochastic_output=False)

    torch.testing.assert_close(actions[0], torch.full((36,), 2.0))
    torch.testing.assert_close(actions[1], torch.full((36,), 4.0))
    torch.testing.assert_close(actions[2], torch.zeros(36))
    assert router.metadata()["experts"][0]["spatial_cell_ids"] == [3]


def test_spatial_checkpoint_routes_group_checkpoints_and_reject_cell_overlap(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "expert.pt"
    args = SimpleNamespace(
        spatial_router_cell_checkpoint=[
            [str(checkpoint), "0", "3"],
            [str(checkpoint), "1", "3"],
        ],
        spatial_router_teacher_checkpoint=None,
        spatial_router_free_cell=None,
        target_position_stratified_grid=(8, 8),
    )

    assert _spatial_checkpoint_routes(args) == ((checkpoint.resolve(), (3, 11)),)

    args.spatial_router_cell_checkpoint.append([str(tmp_path / "other.pt"), "0", "3"])
    with pytest.raises(ValueError, match="duplicate spatial checkpoint cell"):
        _spatial_checkpoint_routes(args)


def test_scratch_correction_initialization_changes_only_final_head() -> None:
    policy = torch.nn.Module()
    policy.mlp = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.ELU(), torch.nn.Linear(4, 36)
    )
    first_weight = policy.mlp[0].weight.detach().clone()
    final_weight = policy.mlp[2].weight.detach().clone()
    bias = torch.linspace(-0.3, 0.3, 36)

    _initialize_scratch_correction_head(policy, output_scale=0.01, correction_bias=bias)

    torch.testing.assert_close(policy.mlp[0].weight, first_weight)
    torch.testing.assert_close(policy.mlp[2].weight, final_weight * 0.01)
    torch.testing.assert_close(policy.mlp[2].bias, bias)


def test_zero_retarget_accepts_legacy_reference_metadata() -> None:
    legacy = {"source": "bc", "observation_dim": 331}
    expected = {
        **legacy,
        "target_x_arm_gains": [0.0, 0.0],
        "target_positive_x_arm_gains": None,
        "target_x_arm_gain_y_bounds": None,
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
    assert _reference_metadata_matches(
        legacy, {**expected, "target_positive_x_arm_gains": [0.0, 0.0]}
    )
    assert not _reference_metadata_matches(
        legacy, {**expected, "target_positive_x_arm_gains": [1.0, 0.0]}
    )
    assert not _reference_metadata_matches(
        legacy, {**expected, "target_x_arm_gain_y_bounds": [-0.05, 0.025]}
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
