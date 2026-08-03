"""CUDA-batched legacy grasp reward with truth-only termination state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import TYPE_CHECKING

import torch

from simple.grasp_rl.schema import JOINT_NAMES
from simple.grasp_rl.task_spec import GraspLiftRewardSpec

if TYPE_CHECKING:
    from simple.grasp_rl.mjlab_gpu.state import GpuLegacyState, GpuTargetState

GPU_REWARD_SCHEMA_VERSION = 1
FINGER_DISTAL_CLOSURE_RAD = 0.20
FINGER_MEAN_CLOSURE_RAD = 0.25


def finger_closure_score(
    hand_qpos: torch.Tensor, initial_hand_qpos: torch.Tensor
) -> torch.Tensor:
    """Score visible thumb/opposing-finger closure from physical joint state."""

    if hand_qpos.shape != initial_hand_qpos.shape or hand_qpos.shape[-1] != 7:
        raise ValueError("hand qpos must have a final dimension of seven")
    delta = (hand_qpos - initial_hand_qpos).abs()
    distal = torch.stack((delta[..., 2], delta[..., 4], delta[..., 6]), dim=-1)
    thumb = distal[..., 0] / FINGER_DISTAL_CLOSURE_RAD
    opposition = distal[..., 1:].amax(dim=-1) / FINGER_DISTAL_CLOSURE_RAD
    mean = delta.mean(dim=-1) / FINGER_MEAN_CLOSURE_RAD
    return torch.minimum(torch.minimum(thumb, opposition), mean).clamp(0.0, 1.0)


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GpuRewardTerms:
    target_reward: torch.Tensor
    penalty: torch.Tensor
    terminal_adjustment: torch.Tensor
    success: torch.Tensor
    native_success: torch.Tensor
    failure: torch.Tensor
    timeout: torch.Tensor
    reach: torch.Tensor
    pregrasp: torch.Tensor
    contact: torch.Tensor
    grasp_quality: torch.Tensor
    finger: torch.Tensor
    lift: torch.Tensor
    stable: torch.Tensor
    grail_grasp: torch.Tensor
    grail_finger_direction: torch.Tensor
    approach_penalty: torch.Tensor
    table_penalty: torch.Tensor
    action_rate_penalty: torch.Tensor
    joint_limit_penalty: torch.Tensor
    lift_height: torch.Tensor
    is_grasp: torch.Tensor

    @property
    def done(self) -> torch.Tensor:
        return self.success | self.failure | self.timeout

    def task_reward(self, weight: float = 0.02) -> torch.Tensor:
        return weight * (self.target_reward - self.penalty) + self.terminal_adjustment


class GpuGraspReward:
    """Exact ``grail_release_v1`` task reward over a batch of environments.

    Reference noise is deliberately absent from this class.  An optional clean
    reference contact label can gate GRAIL contact intent, while all success,
    failure, and timeout decisions use simulator truth from ``GpuTargetState``.
    """

    profile = "grail_release_v1"

    def __init__(
        self,
        state_reader: GpuLegacyState,
        *,
        reward_spec: GraspLiftRewardSpec,
        max_episode_steps: int,
        frozen_reward_hash: str | None = None,
    ):
        self.state_reader = state_reader
        self.sim = state_reader.sim
        self.device = state_reader.device
        self.num_envs = self.sim.num_envs
        self.reward_spec = reward_spec
        self.max_episode_steps = int(max_episode_steps)
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        spec_payload = asdict(reward_spec)
        self.frozen_reward_hash = _json_hash(spec_payload)
        if (
            frozen_reward_hash is not None
            and frozen_reward_hash != self.frozen_reward_hash
        ):
            raise ValueError(
                "Runtime reward specification does not match frozen assets"
            )

        model = self.sim.mj_model
        joint_ids = [model.joint(name).id for name in JOINT_NAMES]
        # The controller joint list and state observation have the same logical
        # order.  Resolve ranges by name so XML reordering cannot change reward.
        ranges = torch.as_tensor(
            model.jnt_range[joint_ids], dtype=torch.float32, device=self.device
        )
        if ranges.shape != (len(state_reader.qpos_indices), 2):
            raise ValueError("Joint range and legacy observation schemas disagree")
        self.joint_low = ranges[:, 0]
        self.joint_high = ranges[:, 1]
        self.joint_span = self.joint_high - self.joint_low
        self.joint_range_valid = self.joint_span > 1e-5

        self.step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.hold_count = torch.zeros_like(self.step_count)
        self.fall_count = torch.zeros_like(self.step_count)
        self.danger_count = torch.zeros_like(self.step_count)
        self.grasp_without_lift_count = torch.zeros_like(self.step_count)
        self.grasp_attempt_started = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.ever_lifted = torch.zeros_like(self.grasp_attempt_started)
        self.reference_contact = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.reference_contact_valid = torch.zeros_like(self.grasp_attempt_started)

    @classmethod
    def from_frozen_bundle(cls, state_reader: GpuLegacyState) -> "GpuGraspReward":
        manifest = state_reader.gpu.bundle.manifest
        return cls(
            state_reader,
            reward_spec=GraspLiftRewardSpec(**manifest["reward_spec"]),
            max_episode_steps=int(
                manifest["task_metadata"]["spec"]["max_episode_steps"]
            ),
            frozen_reward_hash=manifest["reward_hash"],
        )

    def metadata(self) -> dict[str, object]:
        resolved = {
            "schema_version": GPU_REWARD_SCHEMA_VERSION,
            "profile": self.profile,
            "max_episode_steps": self.max_episode_steps,
            "reward_spec": asdict(self.reward_spec),
            "frozen_reward_hash": self.frozen_reward_hash,
        }
        return {**resolved, "resolved_sha256": _json_hash(resolved)}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        for value in (
            self.step_count,
            self.hold_count,
            self.fall_count,
            self.danger_count,
            self.grasp_without_lift_count,
            self.grasp_attempt_started,
            self.ever_lifted,
            self.reference_contact,
            self.reference_contact_valid,
        ):
            value[indices] = 0

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
    def _finger_opposition(state: GpuTargetState, active: torch.Tensor) -> torch.Tensor:
        center = torch.where(
            state.contact.has_object_contact_center[:, None],
            state.contact.object_contact_center_w,
            state.object_pos_w,
        )
        vectors = torch.nn.functional.normalize(
            state.distal_pos_w - center[:, None], dim=-1, eps=1e-6
        )
        support = torch.nn.functional.normalize(
            vectors[:, 1:].mean(dim=1), dim=-1, eps=1e-6
        )
        opposition = ((1.0 - (vectors[:, 0] * support).sum(dim=-1)) * 0.5).clamp(
            0.0, 1.0
        )
        return torch.where(active, opposition, torch.zeros_like(opposition))

    def _joint_limit_penalty(self) -> torch.Tensor:
        position = self.sim.data.qpos[:, self.state_reader.qpos_indices]
        margin = 0.05 * self.joint_span.clamp_min(1e-5)
        distance = torch.minimum(position - self.joint_low, self.joint_high - position)
        penalty = ((margin - distance) / margin).clamp(0.0, 1.0)
        valid = self.joint_range_valid.to(penalty.dtype)
        return (penalty * valid).sum(dim=-1) / valid.sum().clamp_min(1.0)

    def compute(
        self,
        state: GpuTargetState,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        action_span: torch.Tensor,
    ) -> GpuRewardTerms:
        expected = (self.num_envs, 36)
        if action.shape != expected or previous_action.shape != expected:
            raise ValueError(f"Actions must have shape {expected}")
        if (
            str(action.device) != self.device
            or str(previous_action.device) != self.device
        ):
            raise ValueError("Reward actions must stay on the simulation device")
        self.step_count.add_(1)
        initial = self.state_reader.initial_object_pos
        lift_height = state.object_pos_w[:, 2] - initial[:, 2]
        distance = (state.hand_pos_w - state.object_pos_w).norm(dim=-1)
        surface = state.fingertip_surface_distances
        pregrasp = torch.sqrt(
            torch.exp(-10.0 * surface[:, 0])
            * torch.exp(-10.0 * torch.minimum(surface[:, 1], surface[:, 2]))
        )
        force_scores = (state.contact.group_forces / 2.0).clamp(0.0, 1.0)
        grasp_quality = torch.sqrt(
            force_scores[:, 0] * torch.maximum(force_scores[:, 1], force_scores[:, 2])
        )
        is_grasp = state.contact.is_grasp
        reach = torch.exp(-10.0 * distance)
        contact = force_scores.mean(dim=-1)
        finger = self._finger_opposition(
            state, (distance < 0.12) | state.contact.group_contacts.any(dim=-1)
        )
        contact_intent = torch.where(
            self.reference_contact_valid,
            self.reference_contact,
            is_grasp.to(torch.float32),
        )
        grail_grasp = (
            (state.contact.link_force_magnitudes > 0.1).sum(dim=-1).float() / 8.0
        ).clamp(0.0, 1.0) * contact_intent
        grail_finger_direction = (2.0 * finger - 1.0) * contact_intent
        lift = grasp_quality * (lift_height / self.reward_spec.progress_lift).clamp(
            0.0, 1.0
        )
        lift_gate = (lift_height / self.reward_spec.success_lift).clamp(0.0, 1.0)
        stability = torch.exp(
            -5.0 * state.object_lin_vel_w.square().sum(dim=-1)
            - state.object_ang_vel_w.square().sum(dim=-1)
        )
        stable = grasp_quality * lift_gate * stability
        wrist_distance = (state.wrist_pos_w - state.object_pos_w).norm(dim=-1)
        approach = state.wrist_lin_vel_w.square().sum(dim=-1) * torch.exp(
            -wrist_distance.square() / (0.25**2)
        )
        table = state.contact.hand_table_force.clamp(0.0, 1.0)
        span = torch.as_tensor(action_span, dtype=torch.float32, device=self.device)
        normalized_delta = (action - previous_action) / span.clamp_min(1e-4)
        action_rate = normalized_delta.square().mean(dim=-1).clamp(0.0, 1.0)
        joint_limit = self._joint_limit_penalty()
        target = (
            pregrasp
            + 5.0 * grail_grasp
            + 10.0 * grail_finger_direction
            + 5.0 * lift
            + 2.0 * stable
        )
        penalty = 15.0 * approach + table + 0.1 * action_rate

        success_gate = (lift_height >= self.reward_spec.success_lift) & is_grasp
        self.hold_count.copy_(
            torch.where(
                success_gate, self.hold_count + 1, (self.hold_count - 1).clamp_min(0)
            )
        )
        fallen_now = (state.pelvis_height < self.reward_spec.min_pelvis_height) | (
            state.pelvis_rot_w[:, 2, 2] < 0.5
        )
        self.fall_count.copy_(torch.where(fallen_now, self.fall_count + 1, 0))
        dangerous_now = state.contact.hand_table_force > 120.0
        self.danger_count.copy_(torch.where(dangerous_now, self.danger_count + 1, 0))
        self.ever_lifted.logical_or_(lift_height >= self.reward_spec.ever_lifted)
        self.grasp_attempt_started.logical_or_(is_grasp)
        self.grasp_without_lift_count.copy_(
            torch.where(
                lift_height >= 0.005,
                0,
                torch.where(
                    self.grasp_attempt_started,
                    self.grasp_without_lift_count + 1,
                    self.grasp_without_lift_count,
                ),
            )
        )

        success = self.hold_count >= self.reward_spec.success_hold_steps
        native_success = lift_height >= self.reward_spec.success_lift
        dropped = (
            self.ever_lifted & (lift_height < self.reward_spec.drop_lift) & ~is_grasp
        )
        workspace_exit = (state.object_pos_w[:, :2] - initial[:, :2]).norm(
            dim=-1
        ) > self.reward_spec.workspace_radius
        stalled = torch.zeros_like(success)
        if self.reward_spec.stalled_grasp_steps is not None:
            stalled = (
                self.grasp_without_lift_count >= self.reward_spec.stalled_grasp_steps
            )
        fallen = self.fall_count >= 3
        failure = fallen | (self.danger_count >= 3) | dropped | workspace_exit | stalled
        timeout = (self.step_count >= self.max_episode_steps) & ~(success | failure)
        terminal = torch.where(
            success,
            torch.full_like(target, 20.0),
            torch.where(
                failure,
                torch.full_like(target, -10.0),
                torch.where(
                    timeout, torch.full_like(target, -5.0), torch.zeros_like(target)
                ),
            ),
        )
        return GpuRewardTerms(
            target,
            penalty,
            terminal,
            success,
            native_success,
            failure,
            timeout,
            reach,
            pregrasp,
            contact,
            grasp_quality,
            finger,
            lift,
            stable,
            grail_grasp,
            grail_finger_direction,
            approach,
            table,
            action_rate,
            joint_limit,
            lift_height,
            is_grasp,
        )
