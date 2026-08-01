"""CUDA-batched ordered goal-graph reward for v2 SIMPLE tasks."""

from __future__ import annotations

import hashlib
import json

import torch

from simple.grasp_rl.mjlab_gpu.reward import GpuRewardTerms
from simple.grasp_rl.mjlab_gpu.state_v2 import (
    GpuTaskStateExtractorV2,
    GpuTaskStateV2,
)
from simple.grasp_rl.schema import JOINT_NAMES
from simple.grasp_rl.task_spec import GoalStageSpec, TaskSpecV2


GPU_GOAL_REWARD_SCHEMA_VERSION = 1


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class GpuGoalGraphReward:
    """Vectorized equivalent of :class:`GoalGraphReward` on simulator CUDA."""

    profile = "grail_release_v1"

    def __init__(
        self,
        state_reader: GpuTaskStateExtractorV2,
        *,
        frozen_reward_hash: str | None = None,
    ):
        self.state_reader = state_reader
        self.sim = state_reader.sim
        self.spec: TaskSpecV2 = state_reader.spec
        self.device = state_reader.device
        self.num_envs = state_reader.num_envs
        self.max_episode_steps = int(self.spec.max_episode_steps)
        self.frozen_reward_hash = _json_hash(self.spec.metadata()["spec"])
        if (
            frozen_reward_hash is not None
            and frozen_reward_hash != self.frozen_reward_hash
        ):
            raise ValueError("Runtime goal reward does not match frozen assets")

        model = self.sim.mj_model
        ranges = torch.as_tensor(
            model.jnt_range[
                [model.joint(name).id for name in JOINT_NAMES]
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.joint_low = ranges[:, 0]
        self.joint_high = ranges[:, 1]
        self.joint_span = self.joint_high - self.joint_low
        self.joint_range_valid = self.joint_span > 1e-5

        self.step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.stage_index = torch.zeros_like(self.step_count)
        self.stage_hold = torch.zeros_like(self.step_count)
        self.stage_progress = torch.zeros(self.num_envs, device=self.device)
        self.previous_potential = torch.zeros(self.num_envs, device=self.device)
        self.fall_count = torch.zeros_like(self.step_count)
        self.danger_count = torch.zeros_like(self.step_count)
        self.drop_count = torch.zeros_like(self.step_count)
        self.below_support_count = torch.zeros_like(self.step_count)
        self.ever_lifted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.initial_transport_distance = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )
        self.reference_contact = torch.zeros(self.num_envs, device=self.device)
        self.reference_contact_valid = torch.zeros_like(self.ever_lifted)

    @classmethod
    def from_frozen_bundle(
        cls, state_reader: GpuTaskStateExtractorV2
    ) -> "GpuGoalGraphReward":
        return cls(
            state_reader,
            frozen_reward_hash=state_reader.gpu.bundle.manifest["reward_hash"],
        )

    def metadata(self) -> dict[str, object]:
        resolved = {
            "schema_version": GPU_GOAL_REWARD_SCHEMA_VERSION,
            "profile": self.profile,
            "max_episode_steps": self.max_episode_steps,
            "task": self.spec.name,
            "frozen_reward_hash": self.frozen_reward_hash,
        }
        return {**resolved, "resolved_sha256": _json_hash(resolved)}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        for value in (
            self.step_count,
            self.stage_index,
            self.stage_hold,
            self.stage_progress,
            self.previous_potential,
            self.fall_count,
            self.danger_count,
            self.drop_count,
            self.below_support_count,
            self.ever_lifted,
            self.reference_contact,
            self.reference_contact_valid,
        ):
            value[indices] = 0
        self.initial_transport_distance[indices] = float("nan")
        self.state_reader.set_stage(self.stage_index, self.stage_progress)

    def set_reference_contact(
        self,
        should_contact: torch.Tensor | None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        indices = slice(None) if env_ids is None else env_ids
        if should_contact is None:
            self.reference_contact_valid[indices] = False
            return
        values = torch.as_tensor(
            should_contact, dtype=torch.float32, device=self.device
        ).reshape(-1)
        expected = self.num_envs if env_ids is None else int(env_ids.numel())
        if values.numel() == 1:
            values = values.expand(expected)
        if values.shape != (expected,):
            raise ValueError("Reference contact truth has the wrong batch size")
        self.reference_contact[indices] = values.clamp(0.0, 1.0)
        self.reference_contact_valid[indices] = True

    @staticmethod
    def _hand_grasp(
        state: GpuTaskStateV2, hand_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitudes = state.contact_forces_pelvis[:, hand_index].norm(dim=-1)
        thumb = magnitudes[:, 1:4].amax(dim=-1)
        support = magnitudes[:, 4:8].amax(dim=-1)
        quality = (torch.minimum(thumb, support) / 8.0).clamp(0.0, 1.0)
        contact = state.predicates[:, hand_index].bool()
        return contact & (thumb > 2.0) & (support > 2.0), quality

    def _articulation_progress(
        self, state: GpuTaskStateV2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        progress = torch.zeros(self.num_envs, device=self.device)
        complete = torch.zeros_like(progress, dtype=torch.bool)
        goal = self.spec.articulation
        if goal is None:
            return progress, complete
        candidates = []
        predicates = []
        for slot, target in enumerate(goal.targets[:2]):
            present = state.articulation_present[:, slot]
            value = state.articulation_raw[:, slot]
            if goal.comparison == "le":
                candidates.append((value / min(target, -1e-3)).clamp(0.0, 1.0))
                predicates.append(present & (value <= target))
            elif goal.comparison == "abs_ge":
                candidates.append((value.abs() / max(abs(target), 1e-3)).clamp(0.0, 1.0))
                predicates.append(present & (value.abs() >= abs(target)))
            else:
                candidates.append((value / max(abs(target), 1e-3)).clamp(0.0, 1.0))
                predicates.append(present & (value >= target))
        if not candidates:
            return progress, complete
        values = torch.stack(candidates, dim=-1)
        truth = torch.stack(predicates, dim=-1)
        if goal.comparison in ("abs_ge", "any_ge"):
            return values.amax(dim=-1), truth.any(dim=-1)
        return values.amin(dim=-1), truth.all(dim=-1)

    def _stage(
        self,
        stage: GoalStageSpec,
        state: GpuTaskStateV2,
        left_grasp: torch.Tensor,
        left_quality: torch.Tensor,
        right_grasp: torch.Tensor,
        right_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distance = state.fingertip_distances.amin(dim=(1, 2))
        lift = state.primary.pos_w[:, 2] - state.initial_primary_pos[:, 2]
        xy = (state.primary.pos_w[:, :2] - state.destination.pos_w[:, :2]).norm(
            dim=-1
        )
        auxiliary_xy = (
            state.auxiliary.pos_w[:, :2] - state.destination.pos_w[:, :2]
        ).norm(dim=-1)
        primary_auxiliary_xy = (
            state.primary.pos_w[:, :2] - state.auxiliary.pos_w[:, :2]
        ).norm(dim=-1)
        if stage.primitive == "approach":
            return torch.exp(-20.0 * distance.clamp_min(0.0)), distance <= stage.threshold
        if stage.primitive == "contact":
            if stage.hand == "left":
                active = state.predicates[:, 0].bool()
            elif stage.hand == "both":
                active = state.predicates[:, 0].bool() & state.predicates[:, 1].bool()
            else:
                active = state.predicates[:, 1].bool()
            return active.float(), active
        if stage.primitive == "grasp":
            return (
                (left_quality, left_grasp)
                if stage.hand == "left"
                else (right_quality, right_grasp)
            )
        if stage.primitive == "bimanual":
            active = state.predicates[:, 0].bool() & state.predicates[:, 1].bool()
            return torch.minimum(left_quality, right_quality), active
        if stage.primitive == "handover":
            active = left_grasp & ~state.predicates[:, 1].bool()
            return active.float(), active
        if stage.primitive == "lift":
            return (lift / max(stage.threshold, 1e-3)).clamp(0.0, 1.0), lift >= stage.threshold
        if stage.primitive == "transport":
            active = self.stage_index == self.spec.stages.index(stage)
            initialize = active & torch.isnan(self.initial_transport_distance)
            start = torch.maximum(
                xy, torch.full_like(xy, stage.threshold + 1e-3)
            )
            self.initial_transport_distance[initialize] = start[initialize]
            start = torch.nan_to_num(self.initial_transport_distance, nan=1.0)
            progress = ((start - xy) / (start - stage.threshold).clamp_min(1e-3)).clamp(0.0, 1.0)
            grasp = left_grasp | right_grasp
            return progress * grasp.float(), (xy <= stage.threshold) & grasp
        if stage.primitive == "place":
            progress = torch.exp(-8.0 * (xy - stage.threshold).clamp_min(0.0))
            return progress, (xy <= stage.threshold) & state.predicates[:, 4].bool()
        if stage.primitive == "release_settle":
            no_hand = ~state.predicates[:, 0].bool() & ~state.predicates[:, 1].bool()
            stable = (
                state.primary.lin_vel_w.norm(dim=-1) < self.spec.stable_linear_speed
            ) & (state.primary.ang_vel_w.norm(dim=-1) < self.spec.stable_angular_speed)
            complete = (
                no_hand
                & state.predicates[:, 4].bool()
                & stable
                & (xy <= self.spec.placement_xy_tolerance)
            )
            return complete.float(), complete
        if stage.primitive == "articulation":
            return self._articulation_progress(state)
        if stage.primitive == "push":
            direction = state.destination.pos_w[:, :2] - state.initial_primary_pos[:, :2]
            fallback = torch.zeros_like(direction)
            fallback[:, 0] = 1.0
            direction = torch.where(
                (direction.norm(dim=-1) < 1e-6)[:, None],
                fallback,
                torch.nn.functional.normalize(direction, dim=-1, eps=1e-6),
            )
            delta = state.primary.pos_w[:, :2] - state.initial_primary_pos[:, :2]
            projected = (delta * direction).sum(dim=-1)
            lateral = (delta[:, 0] * direction[:, 1] - delta[:, 1] * direction[:, 0]).abs()
            return (projected / max(stage.threshold, 1e-3)).clamp(0.0, 1.0), (projected >= stage.threshold) & (lateral <= 0.30)
        if stage.primitive == "place_aux":
            progress = torch.exp(
                -8.0
                * (primary_auxiliary_xy - self.spec.placement_xy_tolerance).clamp_min(0.0)
            )
            complete = state.predicates[:, 5].bool() & (
                primary_auxiliary_xy <= self.spec.placement_xy_tolerance
            )
            return progress, complete
        if stage.primitive == "aux_bimanual":
            active = state.predicates[:, 2].bool() & state.predicates[:, 3].bool()
            return active.float(), active
        if stage.primitive == "aux_transport":
            progress = torch.exp(-4.0 * (auxiliary_xy - stage.threshold).clamp_min(0.0))
            complete = (
                (auxiliary_xy <= stage.threshold)
                & state.predicates[:, 2].bool()
                & state.predicates[:, 3].bool()
            )
            return progress, complete
        if stage.primitive == "aux_place":
            progress = torch.exp(-8.0 * (auxiliary_xy - stage.threshold).clamp_min(0.0))
            complete = (auxiliary_xy <= stage.threshold) & state.predicates[:, 6].bool()
            return progress, complete
        if stage.primitive == "aux_release_settle":
            no_hand = ~state.predicates[:, 2].bool() & ~state.predicates[:, 3].bool()
            stable = (
                state.auxiliary.lin_vel_w.norm(dim=-1) < self.spec.stable_linear_speed
            ) & (state.auxiliary.ang_vel_w.norm(dim=-1) < self.spec.stable_angular_speed)
            complete = no_hand & state.predicates[:, 6].bool() & stable
            return complete.float(), complete
        raise ValueError(f"Unknown goal primitive {stage.primitive!r}")

    def _joint_limit_penalty(self) -> torch.Tensor:
        position = self.sim.data.qpos[:, self.state_reader.qpos_indices]
        margin = 0.03 * self.joint_span.clamp_min(1e-5)
        near = (position < self.joint_low + margin) | (
            position > self.joint_high - margin
        )
        return 0.01 * (near & self.joint_range_valid).float().sum(dim=-1)

    def compute(
        self,
        state: GpuTaskStateV2,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        action_span: torch.Tensor,
    ) -> GpuRewardTerms:
        self.step_count.add_(1)
        left_grasp, left_quality = self._hand_grasp(state, 0)
        right_grasp, right_quality = self._hand_grasp(state, 1)
        progress = torch.zeros(self.num_envs, device=self.device)
        predicate = torch.zeros_like(progress, dtype=torch.bool)
        for index, stage in enumerate(self.spec.stages):
            stage_progress, stage_predicate = self._stage(
                stage,
                state,
                left_grasp,
                left_quality,
                right_grasp,
                right_quality,
            )
            active = self.stage_index == index
            progress = torch.where(active, stage_progress, progress)
            predicate = torch.where(active, stage_predicate, predicate)

        self.stage_hold.copy_(torch.where(predicate, self.stage_hold + 1, 0))
        hold_steps = torch.tensor(
            [stage.hold_steps for stage in self.spec.stages],
            dtype=torch.long,
            device=self.device,
        )[self.stage_index]
        completed = self.stage_hold >= hold_steps
        final_before_advance = self.stage_index == len(self.spec.stages) - 1
        success = completed & final_before_advance
        advance = completed & ~final_before_advance
        self.stage_index.add_(advance.to(torch.long))
        self.stage_hold[advance] = 0
        progress = torch.where(advance, torch.zeros_like(progress), progress)
        potential = self.stage_index.float() + progress.clamp(0.0, 1.0)
        potential_delta = 0.99 * potential - self.previous_potential
        graph_target = 5.0 * potential_delta.clamp(-0.25, 1.0)
        graph_target = graph_target + 2.0 * completed.float()
        self.previous_potential.copy_(potential)

        lift_height = state.primary.pos_w[:, 2] - state.initial_primary_pos[:, 2]
        xy = (state.primary.pos_w[:, :2] - state.destination.pos_w[:, :2]).norm(dim=-1)
        stable = (
            state.primary.lin_vel_w.norm(dim=-1) < self.spec.stable_linear_speed
        ) & (state.primary.ang_vel_w.norm(dim=-1) < self.spec.stable_angular_speed)
        self.ever_lifted.logical_or_(
            lift_height >= min(self.spec.lift_height, 0.05)
        )
        fallen_now = state.pelvis_height < 0.50
        self.fall_count.copy_(torch.where(fallen_now, self.fall_count + 1, 0))
        dangerous = (
            state.predicates[:, 7].bool()
            & (self.spec.family in ("articulation", "push"))
            & (self.stage_index != len(self.spec.stages) - 1)
        )
        self.danger_count.copy_(torch.where(dangerous, self.danger_count + 1, 0))
        dropped = (
            self.ever_lifted
            & ~state.predicates[:, 0].bool()
            & ~state.predicates[:, 1].bool()
            & ~state.predicates[:, 4].bool()
            & (lift_height < 0.01)
            & (xy > 1.5 * self.spec.placement_xy_tolerance)
        )
        self.drop_count.copy_(torch.where(dropped, self.drop_count + 1, 0))
        below_support = lift_height < -0.03
        self.below_support_count.copy_(
            torch.where(below_support, self.below_support_count + 1, 0)
        )
        failure = (
            (self.fall_count >= 3)
            | (self.danger_count >= 3)
            | (self.drop_count >= 5)
            | (self.below_support_count >= 3)
        )
        timeout = (self.step_count >= self.max_episode_steps) & ~(success | failure)

        magnitudes = state.contact_forces_pelvis[:, 1].norm(dim=-1)
        contact_intent = torch.where(
            self.reference_contact_valid,
            self.reference_contact,
            state.predicates[:, 1],
        )
        grail_grasp = (magnitudes > 0.1).float().mean(dim=-1) * contact_intent
        vectors = torch.nn.functional.normalize(
            state.distal_pos_w[:, 1] - state.primary.pos_w[:, None],
            dim=-1,
            eps=1e-6,
        )
        support = torch.nn.functional.normalize(
            vectors[:, 1:].mean(dim=1), dim=-1, eps=1e-6
        )
        grail_finger = (
            -(vectors[:, 0] * support).sum(dim=-1)
        ).clamp(0.0, 1.0) * contact_intent
        use_grail = self.spec.family == "grasp"
        wrist_distance = (state.hands[1].pos_w - state.primary.pos_w).norm(dim=-1)
        approach = state.hands[1].lin_vel_w.square().sum(dim=-1) * torch.exp(
            -wrist_distance.square() / (0.25**2)
        )
        # Once contact has advanced the ordered graph, upward wrist motion is
        # the desired lift action rather than a hazardous approach velocity.
        approach = approach * (self.stage_index == 0).float()
        normalized_delta = (action - previous_action) / action_span.clamp_min(1e-3)
        raw_action_rate = normalized_delta.square().mean(dim=-1)
        joint_limit = self._joint_limit_penalty()
        if use_grail:
            lift_progress = (
                lift_height / max(self.spec.lift_height, 1e-3)
            ).clamp(0.0, 1.0)
            target = (
                5.0 * grail_grasp
                + 10.0 * grail_finger
                + 5.0 * lift_progress
                + graph_target
            )
            penalty = (
                state.predicates[:, 7]
                + 0.01 * raw_action_rate
            )
        else:
            target = graph_target
            penalty = (
                0.001
                + 0.002 * raw_action_rate
                + 0.5 * dangerous.float()
                + joint_limit
            )
        terminal = torch.where(
            success,
            torch.full_like(progress, 20.0),
            torch.where(
                failure,
                torch.full_like(progress, -10.0),
                torch.where(
                    timeout & (not use_grail),
                    torch.full_like(progress, -5.0),
                    torch.zeros_like(progress),
                ),
            ),
        )
        self.stage_progress.copy_(progress)
        self.state_reader.set_stage(self.stage_index, self.stage_progress)
        quality = torch.maximum(left_quality, right_quality)
        grasp = left_grasp | right_grasp
        reach = torch.exp(-20.0 * state.fingertip_distances.amin(dim=(1, 2)).clamp_min(0.0))
        return GpuRewardTerms(
            target,
            penalty,
            terminal,
            success,
            success,
            failure,
            timeout,
            reach,
            torch.where(self.stage_index == 0, progress, torch.zeros_like(progress)),
            (state.predicates[:, 0].bool() | state.predicates[:, 1].bool()).float(),
            quality,
            quality,
            (lift_height / max(self.spec.lift_height, 1e-3)).clamp(0.0, 1.0),
            stable.float(),
            grail_grasp,
            grail_finger,
            approach,
            state.predicates[:, 7],
            raw_action_rate,
            joint_limit,
            lift_height,
            grasp,
        )
