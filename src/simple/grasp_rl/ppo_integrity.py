"""Runtime evidence that training uses genuine, fresh on-policy PPO rollouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from rsl_rl.algorithms import PPO


def _parameter_snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in module.parameters())


def _parameter_delta_l2(
    before: tuple[torch.Tensor, ...], module: torch.nn.Module
) -> float:
    parameters = tuple(module.parameters())
    if len(before) != len(parameters):
        raise RuntimeError("Parameter structure changed during PPO update")
    squared = 0.0
    for old, new in zip(before, parameters, strict=True):
        difference = new.detach().cpu().to(torch.float64) - old.to(torch.float64)
        squared += float(torch.sum(difference.square()))
    return math.sqrt(squared)


def _parameters_equal(
    before: tuple[torch.Tensor, ...], module: torch.nn.Module
) -> bool:
    parameters = tuple(module.parameters())
    return len(before) == len(parameters) and all(
        torch.equal(old, new.detach().cpu())
        for old, new in zip(before, parameters, strict=True)
    )


class PpoIntegrityAuditor:
    """Wrap RSL-RL PPO and emit one machine-checkable record per update."""

    def __init__(
        self,
        runner: Any,
        output_path: str | Path,
        *,
        start_update: int = 0,
        start_total_transitions: int = 0,
        require_actor_update: bool = True,
    ) -> None:
        if not isinstance(runner.alg, PPO):
            algorithm = f"{type(runner.alg).__module__}.{type(runner.alg).__qualname__}"
            raise TypeError(f"PPO integrity audit requires rsl_rl PPO, got {algorithm}")
        self.runner = runner
        self.algorithm = runner.alg
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.update = int(start_update)
        if start_total_transitions < 0:
            raise ValueError("start_total_transitions must be non-negative")
        self.start_total_transitions = int(start_total_transitions)
        self.require_actor_update = require_actor_update
        self.expected_steps = int(runner.cfg["num_steps_per_env"])
        self.expected_optimizer_steps = int(
            self.algorithm.num_learning_epochs * self.algorithm.num_mini_batches
        )
        self.total_transitions = self.start_total_transitions
        self._rollout_actor: tuple[torch.Tensor, ...] | None = None
        self._act_calls = 0
        self._processed_calls = 0
        self._optimizer_steps = 0
        self._noise_squared = 0.0
        self._noise_count = 0
        self.latest_record: dict[str, Any] | None = None
        self._install()

    def _install(self) -> None:
        algorithm = self.algorithm
        base_act = algorithm.act
        base_process = algorithm.process_env_step
        base_update = algorithm.update
        base_optimizer_step = algorithm.optimizer.step

        def audited_act(observations):
            if self._act_calls == 0:
                if int(algorithm.storage.step) != 0:
                    raise RuntimeError("PPO rollout started with a non-empty buffer")
                self._rollout_actor = _parameter_snapshot(algorithm.get_policy())
            elif self._rollout_actor is None or not _parameters_equal(
                self._rollout_actor, algorithm.get_policy()
            ):
                raise RuntimeError("Policy parameters changed during rollout collection")
            actions = base_act(observations)
            transition = algorithm.transition
            if transition.actions is None or transition.actions_log_prob is None:
                raise RuntimeError("PPO act did not store sampled actions and old log-prob")
            if transition.observations is not observations:
                raise RuntimeError("PPO transition did not retain its on-policy observations")
            if not torch.equal(actions, transition.actions):
                raise RuntimeError("Environment actions differ from PPO transition actions")
            if not torch.isfinite(transition.actions_log_prob).all():
                raise RuntimeError("PPO sampled non-finite action log-probabilities")
            params = transition.distribution_params
            if params and params[0].shape == actions.shape:
                noise = actions - params[0]
                self._noise_squared += float(noise.square().sum())
                self._noise_count += int(noise.numel())
            self._act_calls += 1
            return actions

        def audited_process(*args, **kwargs):
            before = int(algorithm.storage.step)
            result = base_process(*args, **kwargs)
            if int(algorithm.storage.step) != before + 1:
                raise RuntimeError("PPO did not append exactly one transition per env step")
            self._processed_calls += 1
            return result

        def audited_optimizer_step(*args, **kwargs):
            self._optimizer_steps += 1
            return base_optimizer_step(*args, **kwargs)

        def audited_update():
            storage = algorithm.storage
            if self._act_calls != self.expected_steps:
                raise RuntimeError(
                    f"PPO collected {self._act_calls} steps, expected {self.expected_steps}"
                )
            if self._processed_calls != self.expected_steps:
                raise RuntimeError("PPO did not process every sampled environment step")
            if int(storage.step) != self.expected_steps:
                raise RuntimeError("PPO rollout buffer is not full at update time")
            if self._rollout_actor is None or not _parameters_equal(
                self._rollout_actor, algorithm.get_policy()
            ):
                raise RuntimeError("PPO update is not using one unchanged behavior policy")
            if storage.distribution_params is None:
                raise RuntimeError("PPO rollout has no saved behavior distribution")
            if not torch.isfinite(storage.actions_log_prob).all():
                raise RuntimeError("PPO rollout contains invalid old log-probabilities")
            if not torch.isfinite(storage.advantages).all():
                raise RuntimeError("PPO rollout contains invalid advantages")

            actor_before = _parameter_snapshot(algorithm.get_policy())
            critic_before = _parameter_snapshot(algorithm._raw_critic)
            transitions = int(storage.num_envs * storage.num_transitions_per_env)
            metrics = base_update()
            if int(storage.step) != 0:
                raise RuntimeError("PPO reused rollout data after the optimizer update")
            if self._optimizer_steps != self.expected_optimizer_steps:
                raise RuntimeError(
                    "PPO optimizer step count mismatch: "
                    f"{self._optimizer_steps} != {self.expected_optimizer_steps}"
                )
            actor_delta = _parameter_delta_l2(actor_before, algorithm.get_policy())
            critic_delta = _parameter_delta_l2(critic_before, algorithm._raw_critic)
            if self.require_actor_update and actor_delta <= 0.0:
                raise RuntimeError("PPO optimizer did not change the actor parameters")

            self.total_transitions += transitions
            record = {
                "update": self.update,
                "policy_version_collected": self.update,
                "algorithm": f"{type(algorithm).__module__}.{type(algorithm).__qualname__}",
                "on_policy": True,
                "rollout_reused": False,
                "stochastic_action_noise_rms": math.sqrt(
                    self._noise_squared / max(self._noise_count, 1)
                ),
                "transitions": transitions,
                "total_transitions": self.total_transitions,
                "num_learning_epochs": int(algorithm.num_learning_epochs),
                "num_mini_batches": int(algorithm.num_mini_batches),
                "optimizer_steps": self._optimizer_steps,
                "actor_parameter_delta_l2": actor_delta,
                "critic_parameter_delta_l2": critic_delta,
                "losses": {key: float(value) for key, value in metrics.items()},
            }
            with self.output_path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            self.latest_record = record
            self.update += 1
            self._rollout_actor = None
            self._act_calls = 0
            self._processed_calls = 0
            self._optimizer_steps = 0
            self._noise_squared = 0.0
            self._noise_count = 0
            return metrics

        algorithm.act = audited_act
        algorithm.process_env_step = audited_process
        algorithm.optimizer.step = audited_optimizer_step
        algorithm.update = audited_update

    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": f"{type(self.algorithm).__module__}.{type(self.algorithm).__qualname__}",
            "audit_path": str(self.output_path.resolve()),
            "expected_steps_per_rollout": self.expected_steps,
            "expected_optimizer_steps_per_update": self.expected_optimizer_steps,
            "starting_total_transitions": self.start_total_transitions,
            "total_transitions": self.total_transitions,
            "latest_record": self.latest_record,
        }
