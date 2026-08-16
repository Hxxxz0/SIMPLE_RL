"""Real RSL-RL PPO runner with CUDA and checkpoint integrity enforcement."""

from __future__ import annotations

import hashlib
import math
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDictBase
from torch import nn

from simple.grasp_rl.mjlab_gpu.simulation import FrozenAssetBundle
from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv
from simple.grasp_rl.ppo_integrity import PpoIntegrityAuditor
from simple.grasp_rl.schema import ACTION_SLICES, action_group_mask
from simple.grasp_rl.task_spec import validate_task_metadata


class SpatialPolicyRouter(nn.Module):
    """Execute a learned specialist only in explicitly selected spatial cells."""

    def __init__(
        self,
        learner: nn.Module,
        teacher: nn.Module,
        *,
        free_spatial_cell_ids: tuple[int, ...],
        teacher_checkpoint: str | Path,
    ) -> None:
        super().__init__()
        if not free_spatial_cell_ids:
            raise ValueError("spatial policy router requires at least one free cell")
        self.learner = learner
        self.teacher = teacher
        self.register_buffer(
            "free_spatial_cell_ids",
            torch.tensor(free_spatial_cell_ids, dtype=torch.long),
        )
        self.teacher_checkpoint = Path(teacher_checkpoint).resolve()

    def forward(self, observations, *, stochastic_output: bool = False):
        cell_ids = observations.get("target_position_cell_id")
        if cell_ids is None:
            raise RuntimeError(
                "spatial policy router requires target-position cell IDs"
            )
        learner_actions = self.learner(
            observations, stochastic_output=stochastic_output
        )
        teacher_actions = self.teacher(observations, stochastic_output=False)
        use_learner = torch.isin(
            cell_ids.flatten(), self.free_spatial_cell_ids
        ).unsqueeze(-1)
        return torch.where(use_learner, learner_actions, teacher_actions)

    def metadata(self) -> dict[str, object]:
        return {
            "teacher_checkpoint": str(self.teacher_checkpoint),
            "teacher_checkpoint_sha256": _sha256_file(self.teacher_checkpoint),
            "free_spatial_cell_ids": self.free_spatial_cell_ids.tolist(),
        }


class SpatialCheckpointRouter(nn.Module):
    """Route disjoint spatial cells to frozen checkpoint experts."""

    def __init__(
        self,
        default_actor: nn.Module,
        routes: tuple[tuple[str | Path, nn.Module, tuple[int, ...]], ...],
    ) -> None:
        super().__init__()
        if not routes:
            raise ValueError("spatial checkpoint router requires at least one route")
        self.default_actor = default_actor
        self.experts = nn.ModuleList()
        self.expert_checkpoints: list[Path] = []
        seen_cells: set[int] = set()
        for index, (checkpoint, expert, cell_ids) in enumerate(routes):
            if not cell_ids:
                raise ValueError("spatial checkpoint expert requires at least one cell")
            overlap = seen_cells.intersection(cell_ids)
            if overlap:
                raise ValueError(f"spatial checkpoint routes overlap: {sorted(overlap)}")
            seen_cells.update(cell_ids)
            self.experts.append(expert)
            self.expert_checkpoints.append(Path(checkpoint).resolve())
            self.register_buffer(
                f"expert_cell_ids_{index}", torch.tensor(cell_ids, dtype=torch.long)
            )

    def _expert_cell_ids(self, index: int) -> torch.Tensor:
        return getattr(self, f"expert_cell_ids_{index}")

    def forward(self, observations, *, stochastic_output: bool = False):
        cell_ids = observations.get("target_position_cell_id")
        if cell_ids is None:
            raise RuntimeError(
                "spatial checkpoint router requires target-position cell IDs"
            )
        actions = self.default_actor(
            observations, stochastic_output=stochastic_output
        )
        for index, expert in enumerate(self.experts):
            expert_actions = expert(
                observations, stochastic_output=stochastic_output
            )
            selected = torch.isin(
                cell_ids.flatten(), self._expert_cell_ids(index)
            ).unsqueeze(-1)
            actions = torch.where(selected, expert_actions, actions)
        return actions

    def metadata(self) -> dict[str, object]:
        return {
            "type": "cell_checkpoint_experts",
            "experts": [
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "spatial_cell_ids": self._expert_cell_ids(index).tolist(),
                }
                for index, checkpoint in enumerate(self.expert_checkpoints)
            ],
        }


def ppo_train_config(
    *,
    smoke: bool = False,
    plan_conditioned_actor: bool = False,
    exploration_std: float | None = None,
    exploration_hold_steps: int = 1,
    learn_exploration_std: bool = False,
    entropy_coef: float = 0.0,
    residual_action_groups: tuple[str, ...] = ("right_hand", "right_arm"),
    exploration_group_stds: tuple[tuple[str, float], ...] = (),
) -> dict:
    """Return the reviewed PPO hyperparameters for the reference actor."""

    if exploration_hold_steps != 1:
        raise ValueError(
            "PPO exploration_hold_steps must be 1: reusing Gaussian noise "
            "across actions does not match the per-step likelihood ratios "
            "used by PPO"
        )
    if exploration_std is not None and exploration_std <= 0.0:
        raise ValueError("exploration_std must be positive")
    if entropy_coef < 0.0:
        raise ValueError("entropy_coef must be non-negative")
    grouped_stds: list[tuple[str, float]] = []
    seen_groups: set[str] = set()
    for group, raw_std in exploration_group_stds:
        if group not in ACTION_SLICES:
            raise ValueError(f"Unknown exploration action group: {group}")
        if group in seen_groups:
            raise ValueError(f"Duplicate exploration action group: {group}")
        std = float(raw_std)
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("Action-group exploration std must be positive")
        grouped_stds.append((group, std))
        seen_groups.add(group)
    distribution_cfg: dict[str, object] = {
        "class_name": (
            "simple.grasp_rl.distribution.ActionGroupedGaussianDistribution"
            if grouped_stds
            else "GaussianDistribution"
        ),
        "init_std": (
            exploration_std
            if exploration_std is not None
            else (0.05 if smoke else 0.02)
        ),
        "std_type": "scalar",
        "learn_std": bool(learn_exploration_std),
    }
    if grouped_stds:
        distribution_cfg["action_group_stds"] = tuple(grouped_stds)

    return {
        "seed": 42,
        "num_steps_per_env": 4 if smoke else 24,
        # At 8192 envs, 100 updates already represent 19.7M fresh transitions.
        # Keep smoke runs dense while preventing GRAIL-scale runs from creating
        # thousands of redundant checkpoints.
        "save_interval": 1 if smoke else 100,
        "logger": "tensorboard",
        "experiment_name": "tabletop_grasp_mjlab_gpu",
        "run_name": "",
        "upload_model": False,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": (
                "simple.grasp_rl.models.PlanConditionedMLPModel"
                if plan_conditioned_actor
                else "MLPModel"
            ),
            "hidden_dims": (512, 256, 128),
            "activation": "elu",
            "obs_normalization": True,
            **(
                {"residual_action_groups": residual_action_groups}
                if plan_conditioned_actor
                else {}
            ),
            "distribution_cfg": distribution_cfg,
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": (512, 256, 128),
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2 if smoke else 5,
            "num_mini_batches": 2 if smoke else 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": float(entropy_coef),
            "learning_rate": 3.0e-4 if smoke else 1.0e-4,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "multi_gpu": None,
        "check_for_nan": True,
    }


def _spatially_balance_advantages(
    advantages: torch.Tensor,
    cell_ids: torch.Tensor,
    *,
    num_cells: int,
    require_all_cells: bool = True,
    weighting: str = "cell",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize advantages per workspace cell with explicit sample weighting."""

    if advantages.shape[:-1] != cell_ids.shape or advantages.shape[-1] != 1:
        raise ValueError("advantages and spatial cell IDs have incompatible shapes")
    if weighting not in ("cell", "sample"):
        raise ValueError("spatial advantage weighting must be 'cell' or 'sample'")
    flat_advantages = advantages.reshape(-1)
    flat_cells = cell_ids.reshape(-1).to(torch.long)
    if torch.any((flat_cells < 0) | (flat_cells >= num_cells)):
        raise ValueError("spatial cell IDs are outside the configured grid")
    counts = torch.bincount(flat_cells, minlength=num_cells)
    missing = (counts == 0).nonzero(as_tuple=False).flatten()
    if require_all_cells and len(missing):
        raise RuntimeError(
            "PPO rollout missed required workspace cells: "
            + ", ".join(str(int(value)) for value in missing)
        )

    balanced = torch.zeros_like(flat_advantages)
    target_count = flat_advantages.numel() / float(num_cells)
    for cell in range(num_cells):
        mask = flat_cells == cell
        count = int(counts[cell])
        if count == 0:
            continue
        values = flat_advantages[mask]
        normalized = (values - values.mean()) / (values.std(unbiased=False) + 1e-8)
        scale = target_count / count if weighting == "cell" else 1.0
        balanced[mask] = normalized * scale
    return balanced.reshape_as(advantages), counts


def _make_optimizer_capturable(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group["capturable"] = True


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_allows_policy_warm_start(
    bundle_manifest: dict[str, object],
    source_manifest_hash: object,
    source_bundle_manifest: dict[str, object] | None = None,
) -> bool:
    """Accept exact, declared-parent or audited workspace-family assets."""

    if source_manifest_hash == bundle_manifest.get("manifest_hash"):
        return True
    compatible = bundle_manifest.get(
        "warm_start_compatible_manifest_hashes", []
    )
    if isinstance(source_manifest_hash, str) and source_manifest_hash in compatible:
        return True
    if (
        not isinstance(source_manifest_hash, str)
        or not isinstance(source_bundle_manifest, dict)
        or source_bundle_manifest.get("manifest_hash") != source_manifest_hash
    ):
        return False

    target_contract = bundle_manifest.get("workspace_support_contract")
    source_contract = source_bundle_manifest.get("workspace_support_contract")
    if not isinstance(target_contract, dict) or not isinstance(source_contract, dict):
        return False

    def root_source_hash(contract: dict[str, object]) -> object:
        return contract.get(
            "root_source_manifest_hash", contract.get("source_manifest_hash")
        )

    family_hash = root_source_hash(target_contract)
    if not isinstance(family_hash, str) or family_hash != root_source_hash(
        source_contract
    ):
        return False

    # Scene geometry and the audited support envelope may differ. Everything
    # consumed by the policy/controller contract must remain byte-for-byte equal.
    invariant_fields = (
        "format_version",
        "task",
        "controller",
        "controller_bundle",
        "model",
        "roles",
        "reset",
        "action_transform_sha256",
        "reward_hash",
        "object_contract",
        "task_metadata",
    )
    return all(
        bundle_manifest.get(field) == source_bundle_manifest.get(field)
        for field in invariant_fields
    )


def _checkpoint_asset_allows_policy_warm_start(
    bundle_manifest: dict[str, object], metadata: dict[str, object]
) -> bool:
    """Resolve and verify a sibling workspace asset only when needed."""

    source_manifest_hash = metadata.get("asset_manifest_hash")
    if _asset_allows_policy_warm_start(bundle_manifest, source_manifest_hash):
        return True
    config = metadata.get("config")
    resolved = config.get("resolved") if isinstance(config, dict) else None
    asset_bundle = resolved.get("asset_bundle") if isinstance(resolved, dict) else None
    task = resolved.get("task") if isinstance(resolved, dict) else None
    if not isinstance(asset_bundle, str) or not isinstance(task, str):
        return False
    try:
        source_bundle = FrozenAssetBundle.load(asset_bundle, expected_task=task)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    return _asset_allows_policy_warm_start(
        bundle_manifest, source_manifest_hash, source_bundle.manifest
    )


def _load_policy_warm_start(
    policy: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    """Load BC policy state without replacing the new PPO distribution."""

    distribution = policy.distribution
    distribution_state = deepcopy(distribution.state_dict())
    # Legacy SIMPLE actors registered this architecture tag as a persistent
    # buffer.  RSL-RL 5.2 no longer owns it; it is metadata, not a weight.
    policy_keys = policy.state_dict().keys()
    compatible_state = dict(state)
    target_residual_mask = policy.state_dict().get("_residual_action_mask")
    source_residual_mask = state.get("_residual_action_mask")
    if target_residual_mask is not None and source_residual_mask is None:
        source_residual_mask = torch.tensor(
            action_group_mask(("right_hand", "right_arm"))
        )
    if "_plan_conditioned_actor" not in policy_keys:
        compatible_state.pop("_plan_conditioned_actor", None)
    if "_residual_action_mask" in policy_keys:
        compatible_state["_residual_action_mask"] = policy.state_dict()[
            "_residual_action_mask"
        ].clone()
    else:
        compatible_state.pop("_residual_action_mask", None)
    policy.load_state_dict(compatible_state, strict=True)
    distribution.load_state_dict(distribution_state, strict=True)
    if target_residual_mask is not None and source_residual_mask is not None:
        newly_enabled = target_residual_mask.bool() & ~source_residual_mask.bool().to(
            target_residual_mask.device
        )
        if torch.any(newly_enabled):
            head = policy.mlp[-1]
            if not isinstance(head, nn.Linear):
                raise TypeError("plan-conditioned actor output head must be linear")
            with torch.no_grad():
                head.weight[newly_enabled] = 0.0
                if head.bias is not None:
                    head.bias[newly_enabled] = 0.0


def checkpoint_uses_plan_conditioned_actor(checkpoint: str | Path) -> bool:
    """Read the persistent legacy architecture tag before runner creation."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("actor_state_dict")
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain an actor_state_dict")
    return "_plan_conditioned_actor" in state


def _reference_metadata_matches(actual: object, expected: dict[str, object]) -> bool:
    """Accept pre-retarget metadata only for the unchanged zero-gain path."""

    if actual == expected:
        return True
    if not isinstance(actual, dict):
        return False
    normalized = dict(actual)
    for name in (
        "target_x_arm_gains",
        "target_positive_x_arm_gains",
        "target_x_arm_gain_y_bounds",
        "target_y_arm_gains",
        "target_positive_y_arm_gains",
        "target_yaw_arm_gains",
    ):
        if name not in normalized:
            if expected.get(name) == [0.0, 0.0]:
                normalized[name] = [0.0, 0.0]
            elif expected.get(name) is None:
                normalized[name] = None
    if "strict_episode" not in normalized and expected.get("strict_episode") is None:
        normalized["strict_episode"] = None
    if "action_transform_sha256" not in normalized:
        normalized["action_transform_sha256"] = expected.get("action_transform_sha256")
    return normalized == expected


def _controller_metadata_matches(actual: object, expected: dict[str, object]) -> bool:
    """Keep unchanged AMO resumes compatible, but reject pre-fix Sonic resumes."""

    if actual == expected:
        return True
    return actual is None and expected.get("backend") == "amo"


def _spatial_advantage_metadata_matches(
    actual: object, expected: dict[str, object] | None
) -> bool:
    """Treat checkpoints without a weighting field as legacy cell weighting."""

    if actual == expected:
        return True
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    normalized = dict(actual)
    normalized.setdefault("weighting", "cell")
    return normalized == expected


class GpuPpoRunner(OnPolicyRunner):
    """On-policy runner that refuses fake PPO and CPU optimizer state."""

    env: GpuGraspVecEnv

    def __init__(
        self,
        env: GpuGraspVecEnv,
        train_cfg: dict,
        *,
        log_dir: str | None,
        integrity_path: str | Path | None = None,
        actor_learning_rate_scale: float = 1.0,
        actor_anchor_checkpoint: str | Path | None = None,
        actor_anchor_weight: float = 0.0,
        actor_anchor_free_spatial_cell_ids: tuple[int, ...] = (),
        spatial_advantage_grid: tuple[int, int] | None = None,
        spatial_advantage_weighting: str = "cell",
    ):
        super().__init__(env, deepcopy(train_cfg), log_dir, env.device)
        self.actor_learning_rate_scale = float(actor_learning_rate_scale)
        if self.actor_learning_rate_scale <= 0.0:
            raise ValueError("actor_learning_rate_scale must be positive")
        if self.actor_learning_rate_scale != 1.0:
            if self.alg.schedule != "fixed":
                raise ValueError(
                    "actor_learning_rate_scale requires a fixed PPO schedule"
                )
            base_lr = float(self.alg.learning_rate)
            self.alg.optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.alg.get_policy().parameters(),
                        "lr": base_lr * self.actor_learning_rate_scale,
                    },
                    {"params": self.alg._raw_critic.parameters(), "lr": base_lr},
                ]
            )
        _make_optimizer_capturable(self.alg.optimizer)
        self.actor_anchor_metadata: dict[str, object] | None = None
        if actor_anchor_weight < 0.0:
            raise ValueError("actor_anchor_weight must be non-negative")
        if (actor_anchor_checkpoint is None) != (actor_anchor_weight == 0.0):
            raise ValueError(
                "actor anchor checkpoint and positive weight must be set together"
            )
        if actor_anchor_free_spatial_cell_ids:
            if actor_anchor_checkpoint is None or spatial_advantage_grid is None:
                raise ValueError(
                    "free actor-anchor cells require an anchor and spatial grid"
                )
            cell_count = int(spatial_advantage_grid[0] * spatial_advantage_grid[1])
            if len(set(actor_anchor_free_spatial_cell_ids)) != len(
                actor_anchor_free_spatial_cell_ids
            ) or any(
                cell_id < 0 or cell_id >= cell_count
                for cell_id in actor_anchor_free_spatial_cell_ids
            ):
                raise ValueError("free actor-anchor cell IDs must be unique and in-grid")
        if actor_anchor_checkpoint is not None:
            self._install_actor_anchor(
                actor_anchor_checkpoint,
                weight=float(actor_anchor_weight),
                free_spatial_cell_ids=actor_anchor_free_spatial_cell_ids,
            )
        self.spatial_advantage_grid = spatial_advantage_grid
        self.spatial_advantage_weighting = spatial_advantage_weighting
        self.spatial_advantage_metrics: dict[str, float] = {}
        if spatial_advantage_grid is not None:
            self._install_spatial_advantage_balancing(
                spatial_advantage_grid, weighting=spatial_advantage_weighting
            )
        elif spatial_advantage_weighting != "cell":
            raise ValueError(
                "spatial advantage weighting requires a spatial advantage grid"
            )
        self.integrity_auditor = (
            PpoIntegrityAuditor(self, integrity_path)
            if integrity_path is not None
            else None
        )

    def _install_spatial_advantage_balancing(
        self, grid: tuple[int, int], *, weighting: str
    ) -> None:
        if len(grid) != 2 or any(
            int(value) != value or int(value) < 1 for value in grid
        ):
            raise ValueError("spatial_advantage_grid must contain positive integers")
        grid = (int(grid[0]), int(grid[1]))
        if weighting not in ("cell", "sample"):
            raise ValueError("spatial advantage weighting must be 'cell' or 'sample'")
        self.spatial_advantage_grid = grid
        self.spatial_advantage_weighting = weighting
        if self.env.config.domain_randomization.target_position_stratified_grid != grid:
            raise ValueError(
                "spatial advantage balancing requires the same stratified DR grid"
            )
        storage = self.alg.storage
        if storage.num_envs < grid[0] * grid[1]:
            raise ValueError(
                "spatial advantage balancing requires at least one env per cell"
            )
        rollout_cells = torch.empty(
            storage.num_transitions_per_env,
            storage.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        base_act = self.alg.act
        base_compute_returns = self.alg.compute_returns
        base_update = self.alg.update
        num_cells = grid[0] * grid[1]

        def spatial_act(observations):
            step = int(storage.step)
            if not 0 <= step < storage.num_transitions_per_env:
                raise RuntimeError("spatial rollout capture is out of bounds")
            rollout_cells[step].copy_(self.env.randomizer.target_position_cell_ids)
            return base_act(observations)

        def spatial_compute_returns(observations):
            result = base_compute_returns(observations)
            balanced, counts = _spatially_balance_advantages(
                storage.advantages,
                rollout_cells,
                num_cells=num_cells,
                weighting=weighting,
            )
            storage.advantages.copy_(balanced)
            self.spatial_advantage_metrics = {
                "spatial_coverage": float((counts > 0).float().mean()),
                "spatial_min_samples": float(counts.min()),
                "spatial_max_samples": float(counts.max()),
            }
            return result

        def spatial_update():
            metrics = base_update()
            return {**metrics, **self.spatial_advantage_metrics}

        self.alg.act = spatial_act
        self.alg.compute_returns = spatial_compute_returns
        self.alg.update = spatial_update

    def _install_actor_anchor(
        self,
        checkpoint: str | Path,
        *,
        weight: float,
        free_spatial_cell_ids: tuple[int, ...] = (),
    ) -> None:
        """Keep focused PPO close to a frozen, validated teacher actor."""

        checkpoint = Path(checkpoint).resolve()
        actor = self.alg.get_policy()
        teacher = self.frozen_actor_copy(checkpoint)

        optimizer = self.alg.optimizer
        base_step = optimizer.step
        base_update = self.alg.update
        base_forward = actor.forward
        captured_observations: TensorDictBase | None = None
        anchor_losses: list[float] = []
        dimension_weights = torch.ones(
            int(actor.mlp[-1].out_features), device=self.device
        )
        # Preserve the demonstrated grasp while still allowing small, targeted
        # corrections in the manipulation joints.
        dimension_weights[7:14] = 10.0
        dimension_weights[21:28] = 5.0
        free_cell_ids = torch.tensor(
            free_spatial_cell_ids, dtype=torch.long, device=self.device
        )

        def capturing_forward(observations, *args, **kwargs):
            nonlocal captured_observations
            result = base_forward(observations, *args, **kwargs)
            if kwargs.get("stochastic_output", False):
                captured_observations = observations.detach()
            return result

        actor.forward = capturing_forward

        def anchored_step(*args, **kwargs):
            if captured_observations is None:
                raise RuntimeError("PPO actor observations were not captured")
            with torch.no_grad():
                targets = teacher(captured_observations, stochastic_output=False)
            prediction = actor(captured_observations, stochastic_output=False)
            element_loss = F.smooth_l1_loss(prediction, targets, reduction="none")
            sample_loss = (element_loss * dimension_weights).sum(
                -1
            ) / dimension_weights.sum()
            if free_cell_ids.numel():
                cell_ids = captured_observations.get("target_position_cell_id")
                if cell_ids is None:
                    raise RuntimeError(
                        "selective actor anchor requires target-position cell IDs"
                    )
                anchored = ~torch.isin(cell_ids.flatten(), free_cell_ids)
                anchor_loss = (
                    sample_loss[anchored].mean()
                    if torch.any(anchored)
                    else prediction.sum() * 0.0
                )
            else:
                anchor_loss = sample_loss.mean()
            (weight * anchor_loss).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), self.alg.max_grad_norm)
            anchor_losses.append(float(anchor_loss.detach()))
            return base_step(*args, **kwargs)

        optimizer.step = anchored_step

        def anchored_update():
            anchor_losses.clear()
            metrics = base_update()
            if not anchor_losses:
                raise RuntimeError("PPO actor anchor did not run an optimizer step")
            metrics["actor_anchor"] = float(sum(anchor_losses) / len(anchor_losses))
            return metrics

        self.alg.update = anchored_update
        self._actor_anchor_teacher = teacher
        self.actor_anchor_metadata = {
            "weight": weight,
            "checkpoint_sha256": _sha256_file(checkpoint),
            "free_spatial_cell_ids": list(free_spatial_cell_ids),
        }

    def frozen_actor_copy(self, checkpoint: str | Path) -> nn.Module:
        """Load a validated frozen actor with the current architecture."""

        checkpoint = Path(checkpoint).resolve()
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = self._validated_actor_state(payload, checkpoint=checkpoint)
        actor = deepcopy(self.alg.get_policy())
        _load_policy_warm_start(actor, state)
        actor.eval()
        actor.requires_grad_(False)
        return actor

    def _validated_actor_state(
        self,
        payload: dict,
        *,
        checkpoint: str | Path,
    ) -> dict[str, torch.Tensor]:
        bundle = self.env.gpu.bundle
        gpu_metadata = payload.get("mjlab_gpu_metadata")
        if gpu_metadata is not None:
            resolved = gpu_metadata.get("config", {}).get("resolved", {})
            if resolved.get("task") != self.env.config.task:
                raise ValueError("GPU checkpoint task does not match this environment")
            if not _checkpoint_asset_allows_policy_warm_start(
                bundle.manifest, gpu_metadata
            ):
                raise ValueError("GPU checkpoint asset bundle does not match")
        else:
            validate_task_metadata(
                payload,
                self.env.config.task,
                checkpoint=checkpoint,
                action_transform=bundle.root / bundle.manifest["action_transform"],
            )
        state = payload.get("actor_state_dict")
        if not isinstance(state, dict):
            raise TypeError("Checkpoint does not contain an actor_state_dict")
        expected_observation_dim = (
            self.env.reference.observation_dim + self.env.reference.context_dim
        )
        first_weight = state.get("mlp.0.weight")
        if first_weight is None or first_weight.shape[1] != expected_observation_dim:
            raise ValueError(
                "Actor checkpoint observation dimension does not match the task: "
                f"{None if first_weight is None else first_weight.shape[1]} != "
                f"{expected_observation_dim}"
            )
        return state

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the critic base rate while retaining the actor rate scale."""

        self.alg.learning_rate = float(learning_rate)
        if self.actor_learning_rate_scale == 1.0:
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self.alg.learning_rate
            return
        if len(self.alg.optimizer.param_groups) != 2:
            raise RuntimeError("decoupled PPO optimizer must have two groups")
        self.alg.optimizer.param_groups[0]["lr"] = (
            self.alg.learning_rate * self.actor_learning_rate_scale
        )
        self.alg.optimizer.param_groups[1]["lr"] = self.alg.learning_rate

    def load_actor_warm_start(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = self._validated_actor_state(payload, checkpoint=checkpoint)
        # A BC checkpoint contains the old policy's sampling distribution too.
        # Warm-start the network and observation normalizer, but keep the PPO
        # run's reviewed exploration settings (notably init_std) authoritative.
        _load_policy_warm_start(self.alg.get_policy(), state)

    def load_critic_warm_start(self, checkpoint: str | Path) -> None:
        """Restore an audited GPU critic without importing optimizer state."""

        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        metadata = payload.get("mjlab_gpu_metadata")
        if metadata is None:
            raise ValueError("critic warm start requires an audited GPU checkpoint")
        bundle = self.env.gpu.bundle
        resolved = metadata.get("config", {}).get("resolved", {})
        if resolved.get("task") != self.env.config.task:
            raise ValueError("GPU checkpoint task does not match this environment")
        if not _checkpoint_asset_allows_policy_warm_start(
            bundle.manifest, metadata
        ):
            raise ValueError("GPU checkpoint asset bundle does not match")
        state = payload.get("critic_state_dict")
        if not isinstance(state, dict):
            raise TypeError("Checkpoint does not contain a critic_state_dict")
        expected_observation_dim = (
            self.env.reference.observation_dim + self.env.reference.context_dim
        )
        first_weight = state.get("mlp.0.weight")
        if first_weight is None or first_weight.shape[1] != expected_observation_dim:
            raise ValueError("Critic warm start observation dimension does not match")
        self.alg._raw_critic.load_state_dict(state, strict=True)

    def freeze_actor_normalizer(self) -> None:
        """Freeze loaded actor statistics while the critic keeps adapting."""

        normalizer = getattr(self.alg.get_policy(), "obs_normalizer", None)
        count = getattr(normalizer, "count", None)
        if normalizer is None or count is None or not hasattr(normalizer, "until"):
            raise ValueError(
                "Actor does not expose a freezeable observation normalizer"
            )
        normalizer.until = int(count.item())

    def assert_cuda_integrity(self, *, require_optimizer_state: bool) -> None:
        expected = self.device
        for name, module in (
            ("actor", self.alg.get_policy()),
            ("critic", self.alg._raw_critic),
        ):
            for parameter in module.parameters():
                if not parameter.is_cuda or str(parameter.device) != expected:
                    raise RuntimeError(f"PPO {name} parameter escaped CUDA")
        tensor_count = 0
        for state in self.alg.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    tensor_count += 1
                    if not value.is_cuda or str(value.device) != expected:
                        raise RuntimeError("PPO optimizer state escaped CUDA")
        if require_optimizer_state and tensor_count == 0:
            raise RuntimeError("PPO optimizer did not create Adam state")
        if not all(
            group.get("capturable") for group in self.alg.optimizer.param_groups
        ):
            raise RuntimeError("PPO Adam optimizer is not CUDA-capturable")

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "config": self.env.config.checkpoint_metadata(),
            "asset_manifest_hash": self.env.gpu.bundle.manifest["manifest_hash"],
            "controller": self.env.controller.metadata(),
            "reward": self.env.reward.metadata(),
            "reference": self.env.reference.metadata(),
            "optimizer": {
                "actor_learning_rate_scale": self.actor_learning_rate_scale,
            },
            "actor_anchor": getattr(self, "actor_anchor_metadata", None),
            "spatial_advantages": (
                None
                if self.spatial_advantage_grid is None
                else {
                    "grid": list(self.spatial_advantage_grid),
                    "weighting": self.spatial_advantage_weighting,
                }
            ),
        }
        reward_override = self.env.robometer_reward_metadata()
        if reward_override is not None:
            metadata["task_reward_override"] = reward_override
        return metadata

    def save(self, path: str, infos: dict | None = None) -> None:
        payload = self.alg.save()
        payload.update(
            {
                "iter": self.current_learning_iteration,
                "next_learning_iteration": self.current_learning_iteration + 1,
                "infos": infos or {},
                "mjlab_gpu_metadata": self.checkpoint_metadata(),
                "env_state": self.env.checkpoint_state(),
                "adaptive_learning_rate": self.alg.learning_rate,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(self.device),
                "task_metadata": {
                    **deepcopy(self.env.gpu.bundle.manifest["task_metadata"]),
                    "action_transform_sha256": self.env.gpu.bundle.manifest[
                        "action_transform_sha256"
                    ],
                },
            }
        )
        if self.integrity_auditor is not None:
            payload["ppo_integrity"] = self.integrity_auditor.metadata()
        torch.save(payload, path)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        payload = torch.load(
            path,
            map_location=map_location or self.device,
            weights_only=False,
        )
        metadata = payload.get("mjlab_gpu_metadata")
        if metadata is None:
            warnings.warn(
                "Loading a legacy CPU/RSL checkpoint without GPU metadata; "
                "use as a compatibility warm start, not an exact resume.",
                stacklevel=2,
            )
        else:
            self.env.config.assert_resume_compatible(metadata["config"])
            expected = self.checkpoint_metadata()
            for key in (
                "asset_manifest_hash",
                "controller",
                "reward",
                "reference",
                "optimizer",
                "actor_anchor",
                "spatial_advantages",
            ):
                actual = metadata.get(key)
                if key == "optimizer" and actual is None:
                    actual = {"actor_learning_rate_scale": 1.0}
                matches = (
                    _reference_metadata_matches(actual, expected[key])
                    if key == "reference"
                    else _controller_metadata_matches(actual, expected[key])
                    if key == "controller"
                    else _spatial_advantage_metadata_matches(actual, expected[key])
                    if key == "spatial_advantages"
                    else actual == expected[key]
                )
                if not matches:
                    raise ValueError(f"Checkpoint {key} metadata mismatch")
            expected_override = expected.get("task_reward_override")
            actual_override = metadata.get("task_reward_override")
            if (
                expected_override is not None or actual_override is not None
            ) and actual_override != expected_override:
                raise ValueError("Checkpoint task reward override metadata mismatch")
        actor_state = payload.get("actor_state_dict")
        actor = self.alg.get_policy()
        if (
            isinstance(actor_state, dict)
            and "_residual_action_mask" in actor.state_dict()
            and "_residual_action_mask" not in actor_state
        ):
            actor_state["_residual_action_mask"] = actor.state_dict()[
                "_residual_action_mask"
            ].clone()
        load_iteration = self.alg.load(payload, load_cfg, strict)
        _move_optimizer_state(self.alg.optimizer, self.device)
        _make_optimizer_capturable(self.alg.optimizer)
        self.set_learning_rate(
            float(payload.get("adaptive_learning_rate", self.alg.learning_rate))
        )
        if load_iteration:
            self.current_learning_iteration = int(
                payload.get("next_learning_iteration", payload.get("iter", 0) + 1)
            )
            if self.integrity_auditor is not None:
                self.integrity_auditor.update = self.current_learning_iteration
                previous_audit = payload.get("ppo_integrity", {})
                self.integrity_auditor.total_transitions = int(
                    previous_audit.get("total_transitions", 0)
                )
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if "cuda_rng_state" in payload:
            torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), self.device)
        env_state = payload.get("env_state")
        if env_state is not None:
            self.env.common_step_counter = int(env_state["common_step_counter"])
            self.env.randomizer.load_state_dict(env_state["domain_randomization"])
            if "reference_generator_state" in env_state:
                self.env.reference_generator.set_state(
                    env_state["reference_generator_state"].cpu()
                )
            self.env._reset(torch.arange(self.env.num_envs, device=self.device))
            completed = env_state.get("completed_episode_stats")
            if isinstance(completed, dict):
                self.env._completed_episode_count.copy_(
                    torch.as_tensor(completed["episodes"], device=self.device)
                )
                self.env._completed_success_count.copy_(
                    torch.as_tensor(completed["successes"], device=self.device)
                )
            repairs = env_state.get("reset_forward_repairs")
            if isinstance(repairs, dict):
                for key, target in (
                    ("events", self.env._reset_forward_repair_events),
                    ("worlds", self.env._reset_forward_repair_worlds),
                    ("values", self.env._reset_forward_repair_values),
                    (
                        "outside_requested",
                        self.env._reset_forward_repair_outside_requested,
                    ),
                ):
                    target.copy_(torch.as_tensor(repairs[key], device=self.device))
        self.assert_cuda_integrity(
            require_optimizer_state=bool(self.alg.optimizer.state)
        )
        return payload.get("infos", {})
