"""Real RSL-RL PPO runner with CUDA and checkpoint integrity enforcement."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import warnings

import torch
from rsl_rl.runners import OnPolicyRunner

from simple.grasp_rl.mjlab_gpu.vec_env import GpuGraspVecEnv
from simple.grasp_rl.ppo_integrity import PpoIntegrityAuditor
from simple.grasp_rl.task_spec import validate_task_metadata


def ppo_train_config(
    *, smoke: bool = False, plan_conditioned_actor: bool = False
) -> dict:
    """Return the reviewed PPO hyperparameters for the 593-D reference actor."""

    return {
        "seed": 42,
        "num_steps_per_env": 4 if smoke else 24,
        "save_interval": 100 if smoke else 25,
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
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.05 if smoke else 0.01,
                "std_type": "scalar",
                "learn_std": False,
            },
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
            "entropy_coef": 0.0,
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


def _make_optimizer_capturable(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group["capturable"] = True


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


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
    if "_plan_conditioned_actor" not in policy_keys:
        compatible_state.pop("_plan_conditioned_actor", None)
    policy.load_state_dict(compatible_state, strict=True)
    distribution.load_state_dict(distribution_state, strict=True)


def checkpoint_uses_plan_conditioned_actor(checkpoint: str | Path) -> bool:
    """Read the persistent legacy architecture tag before runner creation."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain an actor_state_dict")
    return "_plan_conditioned_actor" in state


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
    ):
        super().__init__(env, deepcopy(train_cfg), log_dir, env.device)
        _make_optimizer_capturable(self.alg.optimizer)
        self.integrity_auditor = (
            PpoIntegrityAuditor(self, integrity_path)
            if integrity_path is not None
            else None
        )

    def load_actor_warm_start(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        bundle = self.env.gpu.bundle
        gpu_metadata = payload.get("mjlab_gpu_metadata")
        if gpu_metadata is not None:
            resolved = gpu_metadata.get("config", {}).get("resolved", {})
            if resolved.get("task") != self.env.config.task:
                raise ValueError("GPU checkpoint task does not match this environment")
            if (
                gpu_metadata.get("asset_manifest_hash")
                != bundle.manifest["manifest_hash"]
            ):
                raise ValueError("GPU checkpoint asset bundle does not match")
        else:
            validate_task_metadata(
                payload,
                self.env.config.task,
                checkpoint=checkpoint,
                action_transform=bundle.root / bundle.manifest["action_transform"],
            )
        state = payload["actor_state_dict"]
        first_weight = state.get("mlp.0.weight")
        if first_weight is None or first_weight.shape[1] != 593:
            raise ValueError("Actor warm start is not a 593-D reference policy")
        # A BC checkpoint contains the old policy's sampling distribution too.
        # Warm-start the network and observation normalizer, but keep the PPO
        # run's reviewed exploration settings (notably init_std) authoritative.
        _load_policy_warm_start(self.alg.get_policy(), state)

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
        return {
            "config": self.env.config.checkpoint_metadata(),
            "asset_manifest_hash": self.env.gpu.bundle.manifest["manifest_hash"],
            "reward": self.env.reward.metadata(),
            "reference": self.env.reference.metadata(),
        }

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
            for key in ("asset_manifest_hash", "reward", "reference"):
                if metadata.get(key) != expected[key]:
                    raise ValueError(f"Checkpoint {key} metadata mismatch")
        load_iteration = self.alg.load(payload, load_cfg, strict)
        _move_optimizer_state(self.alg.optimizer, self.device)
        _make_optimizer_capturable(self.alg.optimizer)
        self.alg.learning_rate = float(
            payload.get("adaptive_learning_rate", self.alg.learning_rate)
        )
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
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
        self.assert_cuda_integrity(
            require_optimizer_state=bool(self.alg.optimizer.state)
        )
        return payload.get("infos", {})
