"""RSL-RL VecEnv joining the real CUDA simulation, controller and reward."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from simple.grasp_rl.mjlab_gpu.action import GpuActionTransform
from simple.grasp_rl.mjlab_gpu.amo import BatchedAmoController
from simple.grasp_rl.mjlab_gpu.config import MjlabPpoConfig
from simple.grasp_rl.mjlab_gpu.domain_randomization import GpuDomainRandomizer
from simple.grasp_rl.mjlab_gpu.reference import GpuReferenceLibrary
from simple.grasp_rl.mjlab_gpu.reward import GpuGraspReward
from simple.grasp_rl.mjlab_gpu.simulation import build_gpu_simulation
from simple.grasp_rl.mjlab_gpu.state import GpuLegacyState
from simple.grasp_rl.schema import ACTION_DIM


class GpuGraspVecEnv(VecEnv):
    """GPU-only tabletop grasp environment with noisy actor/clean critic views."""

    def __init__(
        self,
        config: MjlabPpoConfig,
        *,
        training: bool = True,
        randomization_enabled: bool | None = None,
    ):
        if config.reference_processed is None:
            raise ValueError("GPU PPO requires reference_processed")
        self.config = config
        self.cfg = config.resolved()
        self.training = bool(training)
        self.randomization_enabled = (
            self.training
            if randomization_enabled is None
            else bool(randomization_enabled)
        )
        self.device = config.device
        self.num_envs = config.num_envs
        self.num_actions = ACTION_DIM
        self.gpu = build_gpu_simulation(config)
        self.controller = BatchedAmoController(self.gpu)
        self.state_reader = GpuLegacyState(self.gpu)
        self.reward = GpuGraspReward.from_frozen_bundle(self.state_reader)
        self.action_transform = GpuActionTransform.from_frozen_bundle(self.gpu)
        self.randomizer = GpuDomainRandomizer(
            self.gpu,
            self.controller,
            config.domain_randomization,
            seed=config.seed,
        )
        self.reference_generator = torch.Generator(device=self.device).manual_seed(
            config.seed + 1
        )
        self.reference = GpuReferenceLibrary(
            config.reference_processed,
            num_envs=self.num_envs,
            device=self.device,
            source=config.reference_source,
            splits=("train",) if training else ("val", "test"),
        )
        self.max_episode_length = self.reward.max_episode_steps
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.common_step_counter = 0
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, device=self.device)
        self._last_base_observation: torch.Tensor | None = None
        self._last_clean_context: torch.Tensor | None = None
        self.last_numerical_failure = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_terms = None
        self._reset(torch.arange(self.num_envs, device=self.device))

    @property
    def unwrapped(self) -> "GpuGraspVecEnv":
        return self

    def _reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.dtype != torch.long:
            env_ids = env_ids.to(torch.long)
        self.gpu.reset(env_ids)
        self.controller.reset(env_ids)
        self.state_reader.reset(env_ids)
        self.reward.reset(env_ids)
        self.action_transform.reset(env_ids)
        self.randomizer.reset(
            env_ids,
            training=self.randomization_enabled,
            strength=self.config.domain_randomization.strength(
                self.common_step_counter
            ),
        )
        self.state_reader.sync_episode_origin(env_ids)
        self.state_reader.previous_action[env_ids] = (
            self.action_transform.previous_action[env_ids]
        )
        base_observation, _ = self.state_reader.actor_observation()
        self.reference.reset(base_observation[env_ids], env_ids)
        self.episode_length_buf[env_ids] = 0
        self._episode_return[env_ids] = 0.0
        self._episode_success[env_ids] = 0.0
        self._last_base_observation = None
        self._last_clean_context = None

    def get_observations(self) -> TensorDict:
        base_observation, _ = self.state_reader.actor_observation()
        dr_strength = self.config.domain_randomization.strength(
            self.common_step_counter
        )
        policy_context, clean_context = self.reference.policy_context(
            base_observation,
            self.config.domain_randomization.reference_noise.scaled(dr_strength),
            training=(
                self.randomization_enabled and self.config.domain_randomization.enabled
            ),
            generator=self.reference_generator,
        )
        actor = torch.cat((base_observation, policy_context), dim=-1)
        critic = torch.cat((base_observation, clean_context), dim=-1)
        self._last_base_observation = base_observation
        self._last_clean_context = clean_context
        return TensorDict(
            {"policy": actor, "critic": critic},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _numerically_invalid_worlds(self) -> torch.Tensor:
        finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for value in (
            self.gpu.sim.data.qpos,
            self.gpu.sim.data.qvel,
            self.gpu.sim.data.ctrl,
            self.gpu.sim.data.sensordata,
        ):
            finite.logical_and_(torch.isfinite(value).all(dim=1))
        return ~finite

    @torch.no_grad()
    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        previous_action = self.action_transform.previous_action.clone()
        clean_reference_action = self.reference.current_action()
        physical_action = self.action_transform.decode(
            actions, self.randomizer.action_delay_steps
        )
        # This is the clean replay label for the post-action state.  Policy
        # reference noise never enters reward or termination truth.
        self.reward.set_reference_contact(self.reference.post_step_contact_label())
        self.controller.apply_physical_action(physical_action)
        self.state_reader.set_previous_action(physical_action)
        numerical_failure = self._numerically_invalid_worlds()
        invalid_ids = numerical_failure.nonzero(as_tuple=False).flatten()
        if len(invalid_ids):
            # Recover a finite state before evaluating the batched reward.  The
            # affected worlds are still marked as terminal failures below and
            # receive the normal subset reset, rather than hiding the event.
            self.gpu.reset(invalid_ids)
            self.controller.reset(invalid_ids)
            self.state_reader.reset(invalid_ids)
            self.action_transform.reset(invalid_ids)
            self.state_reader.sync_episode_origin(invalid_ids)
        _, state = self.state_reader.actor_observation()
        terms = self.reward.compute(
            state,
            physical_action,
            previous_action,
            self.action_transform.action_span,
        )
        if len(invalid_ids):
            terms.success[invalid_ids] = False
            terms.native_success[invalid_ids] = False
            terms.failure[invalid_ids] = True
            terms.timeout[invalid_ids] = False
            terms.target_reward[invalid_ids] = 0.0
            terms.penalty[invalid_ids] = 0.0
            terms.terminal_adjustment[invalid_ids] = -10.0
        self.last_numerical_failure.copy_(numerical_failure)
        reference_action_mse = (
            (actions.clamp(-1.0, 1.0) - clean_reference_action).square().mean(dim=-1)
        )
        reference_reward = torch.exp(-25.0 * reference_action_mse)
        dr_strength = self.config.domain_randomization.strength(
            self.common_step_counter
        )
        reference_weight = self.config.reference_reward_weight * (
            1.0 - dr_strength * (1.0 - self.config.full_dr_reference_reward_scale)
        )
        rewards = terms.task_reward() + reference_weight * reference_reward
        self.last_terms = terms
        dones = terms.done
        time_outs = terms.timeout
        self.reference.advance()
        self.episode_length_buf.add_(1)
        self.common_step_counter += 1
        self._episode_return.add_(rewards)
        self._episode_success.copy_(
            torch.maximum(self._episode_success, terms.success.to(torch.float32))
        )

        extras: dict[str, object] = {
            "time_outs": time_outs,
            "log": {
                "/reward/target": terms.target_reward.mean(),
                "/reward/penalty": terms.penalty.mean(),
                "/reward/terminal": terms.terminal_adjustment.mean(),
                "/reward/reference": (reference_weight * reference_reward).mean(),
                "/reference/action_mse": reference_action_mse.mean(),
                "/task/native_success": terms.native_success.float().mean(),
                "/task/numerical_failure": numerical_failure.float().mean(),
                "/domain_randomization/strength": torch.tensor(
                    self.config.domain_randomization.strength(self.common_step_counter),
                    device=self.device,
                ),
            },
        }
        finished = dones.nonzero(as_tuple=False).flatten()
        if len(finished):
            log = extras["log"]
            assert isinstance(log, dict)
            log["/episode/return"] = self._episode_return[finished].mean()
            log["/episode/length"] = self.episode_length_buf[finished].float().mean()
            log["/episode/success"] = self._episode_success[finished].mean()
            log["/episode/count"] = torch.tensor(
                len(finished), device=self.device, dtype=torch.float32
            )
            self._reset(finished)
        observations = self.get_observations()
        return observations, rewards, dones, extras

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "common_step_counter": self.common_step_counter,
            "episode_length_buf": self.episode_length_buf.clone(),
            "domain_randomization": self.randomizer.state_dict(),
            "reference_generator_state": self.reference_generator.get_state(),
            "reference": {
                "episode_rows": self.reference.episode_rows.clone(),
                "indices": self.reference.indices.clone(),
                "object_offset": self.reference.reference_object_offset.clone(),
            },
            "reward_metadata": self.reward.metadata(),
            "reference_metadata": self.reference.metadata(),
        }
