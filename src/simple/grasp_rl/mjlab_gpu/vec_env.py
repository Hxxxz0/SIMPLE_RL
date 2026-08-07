"""RSL-RL VecEnv joining the real CUDA simulation, controller and reward."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from simple.grasp_rl.mjlab_gpu.action import GpuActionTransform
from simple.grasp_rl.mjlab_gpu.amo import BatchedAmoController
from simple.grasp_rl.mjlab_gpu.config import MjlabPpoConfig
from simple.grasp_rl.mjlab_gpu.domain_randomization import GpuDomainRandomizer
from simple.grasp_rl.mjlab_gpu.goal_reward import GpuGoalGraphReward
from simple.grasp_rl.mjlab_gpu.reference import GpuReferenceLibrary
from simple.grasp_rl.mjlab_gpu.reward import GpuGraspReward
from simple.grasp_rl.mjlab_gpu.robometer_reward import (
    ROBOMETER_TASKS,
    RobometerTaskReward,
    RobometerTaskRewardConfig,
)
from simple.grasp_rl.mjlab_gpu.simulation import build_gpu_simulation
from simple.grasp_rl.mjlab_gpu.sonic import BatchedSonicController
from simple.grasp_rl.mjlab_gpu.state import GpuLegacyState
from simple.grasp_rl.mjlab_gpu.state_v2 import GpuTaskStateExtractorV2
from simple.grasp_rl.models import V2_RESIDUAL_LAST_ACTIVE_STAGE
from simple.grasp_rl.schema import ACTION_DIM, ACTOR_OBS_V2_DIM


def _nonfinite_output_rows(
    outputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """Return the union of damaged rows and the exact damaged-value count."""

    first = next(iter(outputs.values()))
    rows = torch.zeros(first.shape[0], dtype=torch.bool, device=first.device)
    value_count = 0
    for value in outputs.values():
        nonfinite = ~torch.isfinite(value)
        rows.logical_or_(nonfinite.flatten(1).any(dim=1))
        value_count += int(nonfinite.sum().item())
    return rows, value_count


class GpuGraspVecEnv(VecEnv):
    """GPU-only tabletop grasp environment with noisy actor/clean critic views."""

    def __init__(
        self,
        config: MjlabPpoConfig,
        *,
        training: bool = True,
        randomization_enabled: bool | None = None,
        capture_terminal_qpos: bool = False,
        capture_step_data: bool = False,
        robometer_reward_config: RobometerTaskRewardConfig | None = None,
    ):
        if config.reference_processed is None:
            raise ValueError("GPU PPO requires reference_processed")
        self.config = config
        self.cfg = config.resolved()
        self.training = bool(training)
        self.capture_terminal_qpos = bool(capture_terminal_qpos)
        self.capture_step_data = bool(capture_step_data)
        self.robometer_reward_config = robometer_reward_config
        self.randomization_enabled = (
            self.training
            if randomization_enabled is None
            else bool(randomization_enabled)
        )
        self.device = config.device
        self.num_envs = config.num_envs
        self.num_actions = ACTION_DIM
        self.gpu = build_gpu_simulation(config)
        controller = self.gpu.bundle.manifest["controller"]
        if controller == "amo":
            self.controller = BatchedAmoController(self.gpu)
        elif controller == "sonic_wbc":
            self.controller = BatchedSonicController(self.gpu)
        else:
            raise ValueError(f"Unsupported GPU controller {controller!r}")
        if self.gpu.bundle.manifest["task_metadata"]["task_schema_version"] == 2:
            self.state_reader = GpuTaskStateExtractorV2(self.gpu)
            self.reward = GpuGoalGraphReward.from_frozen_bundle(self.state_reader)
        else:
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
            splits=("train", "val", "test"),
            target_x_arm_gains=config.reference_target_x_arm_gains,
            target_y_arm_gains=config.reference_target_y_arm_gains,
            target_yaw_arm_gains=config.reference_target_yaw_arm_gains,
        )
        self.robometer_reward: RobometerTaskReward | None = None
        if robometer_reward_config is not None:
            if (
                config.task not in ROBOMETER_TASKS
                or config.task != robometer_reward_config.task
            ):
                raise ValueError(
                    "Robometer reward task must be supported and match the environment"
                )
            if not config.smoke_mode or config.num_envs > 8:
                raise ValueError(
                    "Robometer reward requires --smoke and at most eight environments"
                )
            self.robometer_reward = RobometerTaskReward(
                self.gpu,
                config=robometer_reward_config,
                num_envs=self.num_envs,
                device=self.device,
            )
        self.max_episode_length = self.reward.max_episode_steps
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.common_step_counter = 0
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._episode_success = torch.zeros(self.num_envs, device=self.device)
        self._completed_episode_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._completed_success_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._last_base_observation: torch.Tensor | None = None
        self._last_clean_context: torch.Tensor | None = None
        self.last_numerical_failure = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reset_forward_repair_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._reset_forward_repair_worlds = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._reset_forward_repair_values = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._reset_forward_repair_outside_requested = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._last_reset_forward_repair_worlds = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._last_reset_forward_repair_values = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._last_reset_forward_repair_outside_requested = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.last_terms = None
        self.last_task_reward: torch.Tensor | None = None
        self._reset(torch.arange(self.num_envs, device=self.device))

    @property
    def unwrapped(self) -> GpuGraspVecEnv:
        return self

    def _reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.dtype != torch.long:
            env_ids = env_ids.to(torch.long)
        self._reset_dynamics(env_ids)
        self.state_reader.sync_episode_origin(env_ids)
        if self.robometer_reward is not None:
            self.robometer_reward.reset(env_ids, self.gpu.sim.data.qpos)
        self.state_reader.previous_action[env_ids] = (
            self.action_transform.previous_action[env_ids]
        )
        base_observation, _ = self.state_reader.actor_observation()
        base_episode = self.gpu.bundle.manifest.get("base_episode")
        episode_rows = (
            None
            if base_episode is None
            else self.reference.rows_for_episode(base_episode, len(env_ids))
        )
        self.reference.reset(
            base_observation[env_ids],
            env_ids,
            episode_rows=episode_rows,
            target_yaw_offset=self.randomizer.target_yaw[env_ids],
        )
        self.episode_length_buf[env_ids] = 0
        self._episode_return[env_ids] = 0.0
        self._episode_success[env_ids] = 0.0
        self._last_base_observation = None
        self._last_clean_context = None

    def _reset_dynamics(self, env_ids: torch.Tensor) -> None:
        """Reset and validate physical state without resetting reference/accounting."""

        pending = env_ids
        invalid_details: list[str] = []
        for _ in range(3):
            preserved_forward = self._preserve_nonreset_forward_outputs(pending)
            self.gpu.reset(pending)
            self.controller.reset(pending)
            self.state_reader.reset(pending)
            self.reward.reset(pending)
            self.action_transform.reset(pending)
            self.randomizer.reset(
                pending,
                training=self.randomization_enabled,
                strength=self._domain_randomization_strength(),
            )
            self._restore_nonreset_forward_outputs(preserved_forward)
            pending, value_count, invalid_details = self._invalid_reset_outputs(pending)
            if len(pending) == 0:
                break
            self._record_reset_forward_repair(
                world_count=len(pending),
                value_count=value_count,
                outside_count=0,
            )
        else:
            raise FloatingPointError(
                "Reset remained non-finite after three GPU resamples: "
                + "; ".join(invalid_details)
            )

    def _preserve_nonreset_forward_outputs(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
        """Snapshot derived outputs for physical worlds not being reset."""

        if len(env_ids) == self.num_envs:
            return None
        requested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        requested[env_ids] = True
        preserved_ids = (~requested).nonzero(as_tuple=False).flatten()
        return preserved_ids, {
            name: getattr(self.gpu.sim.data, name)[preserved_ids].clone()
            for name in ("qacc", "qacc_warmstart", "sensordata")
        }

    def _record_reset_forward_repair(
        self, *, world_count: int, value_count: int, outside_count: int
    ) -> None:
        if world_count == 0:
            return
        self._reset_forward_repair_events.add_(1)
        self._reset_forward_repair_worlds.add_(world_count)
        self._reset_forward_repair_values.add_(value_count)
        self._reset_forward_repair_outside_requested.add_(outside_count)
        self._last_reset_forward_repair_worlds.add_(world_count)
        self._last_reset_forward_repair_values.add_(value_count)
        self._last_reset_forward_repair_outside_requested.add_(outside_count)

    def _restore_nonreset_forward_outputs(
        self,
        preserved: tuple[torch.Tensor, dict[str, torch.Tensor]] | None,
    ) -> None:
        """Undo full-batch forward writes to worlds outside a subset reset.

        MuJoCo-Warp cannot forward a world subset.  The non-reset worlds have
        unchanged state and model parameters, so their pre-reset acceleration
        and sensor outputs are the correct values.  Restoring them avoids a
        rare non-finite solve/sensor reduction leaking across world boundaries.
        """

        if preserved is None:
            return
        preserved_ids, fields = preserved
        affected = torch.zeros(len(preserved_ids), dtype=torch.bool, device=self.device)
        value_count = 0
        for name, values in fields.items():
            current = getattr(self.gpu.sim.data, name)[preserved_ids]
            newly_nonfinite = ~torch.isfinite(current) & torch.isfinite(values)
            affected.logical_or_(newly_nonfinite.any(dim=1))
            value_count += int(newly_nonfinite.sum().item())
            getattr(self.gpu.sim.data, name)[preserved_ids] = values
        world_count = int(affected.sum().item())
        self._record_reset_forward_repair(
            world_count=world_count,
            value_count=value_count,
            outside_count=world_count,
        )

    def _invalid_reset_outputs(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, int, list[str]]:
        """Reject non-finite reset samples while failing on cross-world damage."""

        outputs = {
            "qpos": self.gpu.sim.data.qpos,
            "qvel": self.gpu.sim.data.qvel,
            "ctrl": self.gpu.sim.data.ctrl,
            "qacc": self.gpu.sim.data.qacc,
            "qacc_warmstart": self.gpu.sim.data.qacc_warmstart,
            "sensordata": self.gpu.sim.data.sensordata,
        }
        requested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        requested[env_ids] = True
        invalid, _ = _nonfinite_output_rows(outputs)
        outside = invalid & ~requested
        if outside.any():
            rows = outside.nonzero(as_tuple=False).flatten()
            raise FloatingPointError(
                "Subset reset damaged an unrequested world after restoration; "
                f"rows={rows[:32].tolist()}"
            )
        invalid = invalid & requested
        details = []
        value_count = 0
        for name, value in outputs.items():
            nonfinite = ~torch.isfinite(value[invalid])
            count = int(nonfinite.sum().item())
            if count:
                details.append(f"{name}={count}")
                value_count += count
        return invalid.nonzero(as_tuple=False).flatten(), value_count, details

    def _domain_randomization_strength(self) -> float:
        curriculum = self.config.domain_randomization
        if self.training:
            return curriculum.training_strength(self.common_step_counter)
        return curriculum.strength(self.common_step_counter)

    def _bounded_reference_action(self, action: torch.Tensor) -> torch.Tensor:
        """Execute only the correction dimensions owned by the v2 residual actor."""

        reference = self.reference.current_action()
        correction = action - reference
        if self.reference.observation_dim == ACTOR_OBS_V2_DIM:
            stage = self.reward.stage_index
            mask = torch.zeros_like(correction)
            active = stage <= V2_RESIDUAL_LAST_ACTIVE_STAGE
            if self.state_reader.spec.family == "handover":
                active = active & (stage < 4)
            mask[:, 7:14] = active[:, None]
            mask[:, 21:28] = active[:, None]
            correction = correction * mask
        limit = self.config.max_reference_action_deviation
        return reference + correction.clamp(-limit, limit)

    def get_observations(self) -> TensorDict:
        base_observation, _ = self.state_reader.actor_observation()
        dr_strength = self._domain_randomization_strength()
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
        if not torch.isfinite(actor).all() or not torch.isfinite(critic).all():
            components = {
                "base_observation": base_observation,
                "policy_context": policy_context,
                "clean_context": clean_context,
            }
            details = []
            for name, value in components.items():
                mask = ~torch.isfinite(value)
                if mask.any():
                    rows = mask.any(dim=1).nonzero(as_tuple=False).flatten()
                    columns = mask.any(dim=0).nonzero(as_tuple=False).flatten()
                    details.append(
                        f"{name}: count={int(mask.sum())}, "
                        f"rows={rows[:16].tolist()}, cols={columns[:32].tolist()}"
                    )
            raise ValueError("Non-finite policy components: " + "; ".join(details))
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
            self.gpu.sim.data.qacc,
            self.gpu.sim.data.qacc_warmstart,
            self.gpu.sim.data.sensordata,
        ):
            finite.logical_and_(torch.isfinite(value).all(dim=1))
        return ~finite

    @torch.no_grad()
    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self._last_reset_forward_repair_worlds.zero_()
        self._last_reset_forward_repair_values.zero_()
        self._last_reset_forward_repair_outside_requested.zero_()
        previous_action = self.action_transform.previous_action.clone()
        clean_reference_action = self.reference.current_action()
        executed_action = self._bounded_reference_action(actions)
        physical_action = self.action_transform.decode(
            executed_action, self.randomizer.action_delay_steps
        )
        # This is the clean replay label for the post-action state.  Policy
        # reference noise never enters reward or termination truth.
        self.reward.set_reference_contact(self.reference.post_step_contact_label())
        joint_target = self.controller.apply_physical_action(physical_action)
        self.state_reader.set_previous_action(physical_action)
        numerical_failure = self._numerically_invalid_worlds()
        invalid_ids = numerical_failure.nonzero(as_tuple=False).flatten()
        if len(invalid_ids):
            # Recover a finite state before evaluating the batched reward.  The
            # affected worlds are still marked as terminal failures below and
            # receive the normal subset reset, rather than hiding the event.
            self._reset_dynamics(invalid_ids)
            self.state_reader.sync_episode_origin(invalid_ids)
            self.state_reader.previous_action[invalid_ids] = (
                self.action_transform.previous_action[invalid_ids]
            )
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
        dr_strength = self._domain_randomization_strength()
        reference_weight = self.config.reference_reward_weight * (
            1.0 - dr_strength * (1.0 - self.config.full_dr_reference_reward_scale)
        )
        physical_task_reward = terms.task_reward()
        robometer_snapshot: dict[str, object] | None = None
        if self.robometer_reward is None:
            task_reward = physical_task_reward
        else:
            robometer_dense = self.robometer_reward.compute(
                qpos=self.gpu.sim.data.qpos,
                terminal=terms.done,
                vector_step=self.common_step_counter,
            )
            robometer_task_reward = robometer_dense + terms.terminal_adjustment
            task_reward = (
                robometer_task_reward
                if self.robometer_reward_config is not None
                and self.robometer_reward_config.mode == "replace"
                else physical_task_reward
            )
            robometer_snapshot = self.robometer_reward.snapshot()
        rewards = task_reward + reference_weight * reference_reward
        self.last_terms = terms
        self.last_task_reward = task_reward
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
                "/reward/task_actual": task_reward.mean(),
                "/reward/task_physical": physical_task_reward.mean(),
                "/reference/action_mse": reference_action_mse.mean(),
                "/reference/executed_action_mse": (
                    (executed_action - clean_reference_action).square().mean()
                ),
                "/task/native_success": terms.native_success.float().mean(),
                "/task/grasp": terms.is_grasp.float().mean(),
                "/task/lift_height": terms.lift_height.mean(),
                "/task/numerical_failure": numerical_failure.float().mean(),
                "/domain_randomization/strength": torch.tensor(
                    self._domain_randomization_strength(),
                    device=self.device,
                ),
            },
        }
        if robometer_snapshot is not None:
            inferred = robometer_snapshot["inferred"]
            progress = robometer_snapshot["progress"]
            success_probability = robometer_snapshot["success_probability"]
            dense_reward = robometer_snapshot["dense_reward"]
            assert isinstance(inferred, torch.Tensor)
            assert isinstance(progress, torch.Tensor)
            assert isinstance(success_probability, torch.Tensor)
            assert isinstance(dense_reward, torch.Tensor)
            extras["robometer"] = robometer_snapshot
            log = extras["log"]
            assert isinstance(log, dict)
            log["/reward/robometer_dense"] = dense_reward.mean()
            log["/robometer/inference_fraction"] = inferred.float().mean()
            if inferred.any():
                log["/robometer/progress"] = progress[inferred].mean()
                log["/robometer/success_probability"] = success_probability[
                    inferred
                ].mean()
            log["/robometer/latency_ms"] = torch.tensor(
                float(robometer_snapshot["latency_ms"]), device=self.device
            )
        if self.capture_step_data:
            stage_index = getattr(self.reward, "stage_index", None)
            extras["step_data"] = {
                "executed_action": executed_action.clone(),
                "physical_action": physical_action.clone(),
                "joint_target": joint_target.clone(),
                "task_reward": task_reward.clone(),
                "physical_task_reward": physical_task_reward.clone(),
                "reference_reward": (reference_weight * reference_reward).clone(),
                "stage_index": (
                    torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                    if stage_index is None
                    else stage_index.clone()
                ),
            }
        finished = dones.nonzero(as_tuple=False).flatten()
        if len(finished):
            finished_successes = self._episode_success[finished].sum()
            self._completed_episode_count.add_(len(finished))
            self._completed_success_count.add_(finished_successes.to(torch.long))
            if self.capture_terminal_qpos:
                extras["terminal_env_ids"] = finished.clone()
                extras["terminal_qpos"] = self.gpu.sim.data.qpos[finished].clone()
                extras["terminal_qvel"] = self.gpu.sim.data.qvel[finished].clone()
            log = extras["log"]
            assert isinstance(log, dict)
            log["/episode/return"] = self._episode_return[finished].mean()
            log["/episode/length"] = self.episode_length_buf[finished].float().mean()
            # A short rollout can contain only an easy/synchronized completion
            # cohort and falsely print 100%.  Keep that batch statistic explicit,
            # while the primary metric is the exact rate over all completed
            # episodes seen by this environment.
            log["/episode/success_finished_batch"] = self._episode_success[
                finished
            ].mean()
            log["/episode/success"] = (
                self._completed_success_count.float()
                / self._completed_episode_count.clamp_min(1).float()
            )
            log["/episode/successes_finished"] = finished_successes
            log["/episode/count"] = torch.tensor(
                len(finished), device=self.device, dtype=torch.float32
            )
            log["/episode/completed_cumulative"] = self._completed_episode_count.float()
            self._reset(finished)
        log = extras["log"]
        assert isinstance(log, dict)
        log["/task/reset_forward_repair_worlds"] = (
            self._last_reset_forward_repair_worlds.float()
        )
        log["/task/reset_forward_repair_values"] = (
            self._last_reset_forward_repair_values.float()
        )
        log["/task/reset_forward_repair_outside_requested"] = (
            self._last_reset_forward_repair_outside_requested.float()
        )
        log["/task/reset_forward_repair_events_cumulative"] = (
            self._reset_forward_repair_events.float()
        )
        observations = self.get_observations()
        return observations, rewards, dones, extras

    def robometer_reward_metadata(self) -> dict[str, object] | None:
        if self.robometer_reward is None:
            return None
        return self.robometer_reward.metadata()

    def close(self) -> None:
        if self.robometer_reward is not None:
            self.robometer_reward.close()

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "common_step_counter": self.common_step_counter,
            "completed_episode_stats": {
                "episodes": self._completed_episode_count.clone(),
                "successes": self._completed_success_count.clone(),
            },
            "reset_forward_repairs": {
                "events": self._reset_forward_repair_events.clone(),
                "worlds": self._reset_forward_repair_worlds.clone(),
                "values": self._reset_forward_repair_values.clone(),
                "outside_requested": (
                    self._reset_forward_repair_outside_requested.clone()
                ),
            },
            "episode_length_buf": self.episode_length_buf.clone(),
            "domain_randomization": self.randomizer.state_dict(),
            "reference_generator_state": self.reference_generator.get_state(),
            "reference": {
                "episode_rows": self.reference.episode_rows.clone(),
                "indices": self.reference.indices.clone(),
                "object_offset": self.reference.reference_object_offset.clone(),
                "object_yaw_offset": (
                    self.reference.reference_object_yaw_offset.clone()
                ),
            },
            "reward_metadata": self.reward.metadata(),
            "reference_metadata": self.reference.metadata(),
        }
