"""Object-contract reward shaping isolated from all legacy PPO tasks."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from simple.grasp_rl.mjlab_gpu.goal_reward import GpuGoalGraphReward, _json_hash
from simple.grasp_rl.mjlab_gpu.reward import GpuRewardTerms, finger_closure_score
from simple.grasp_rl.mjlab_gpu.state_v2 import GpuTaskStateV2
from simple.grasp_rl.task_spec import GoalStageSpec

OBJECT_GRASP_REWARD_SCHEMA_VERSION = 1
DEFAULT_OVERHEAD_FINAL_DESCENT_WEIGHT = 0.25


def overhead_approach_progress(
    distal_pos_w: torch.Tensor,
    grasp_center_w: torch.Tensor,
    *,
    clearance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape a collision-free approach toward a fixed overhead waypoint."""

    if distal_pos_w.ndim != 3 or distal_pos_w.shape[-2:] != (3, 3):
        raise ValueError("distal positions must have shape (N, 3, 3)")
    if grasp_center_w.shape != (distal_pos_w.shape[0], 3):
        raise ValueError("grasp centers must have shape (N, 3)")
    if not math.isfinite(clearance_m) or clearance_m <= 0.0:
        raise ValueError("overhead approach clearance must be positive and finite")

    centroid = distal_pos_w.mean(dim=1)
    horizontal_distance = (centroid[:, :2] - grasp_center_w[:, :2]).norm(dim=-1)
    waypoint_height = grasp_center_w[:, 2] + clearance_m
    waypoint_distance = torch.sqrt(
        horizontal_distance.square()
        + (centroid[:, 2] - waypoint_height).square()
    )
    progress = torch.exp(-4.0 * waypoint_distance)
    return progress, waypoint_distance, waypoint_height


def gated_overhead_approach_progress(
    overhead_progress: torch.Tensor,
    final_descent_progress: torch.Tensor,
    waypoint_reached: torch.Tensor,
    *,
    final_descent_weight: float = DEFAULT_OVERHEAD_FINAL_DESCENT_WEIGHT,
) -> torch.Tensor:
    """Allocate approach potential between the overhead waypoint and descent."""

    if not math.isfinite(final_descent_weight) or not 0.0 < final_descent_weight < 1.0:
        raise ValueError("overhead final descent weight must be finite and in (0, 1)")
    waypoint_weight = 1.0 - final_descent_weight
    return torch.where(
        waypoint_reached,
        waypoint_weight + final_descent_weight * final_descent_progress,
        waypoint_weight * overhead_progress,
    )


def object_grasp_geometry(
    distal_pos_w: torch.Tensor,
    grasp_center_w: torch.Tensor,
    *,
    grip_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return band reach, opposition and nearest-center distance.

    The three distal link origins are a stable CUDA-native proxy for the thumb,
    index and middle fingertips.  Reach targets a shell around the configured
    grasp frame rather than the object's root frame, while opposition rewards a
    thumb on the other side of the frame from the two supporting fingers.
    """

    if distal_pos_w.ndim != 3 or distal_pos_w.shape[-2:] != (3, 3):
        raise ValueError("distal positions must have shape (N, 3, 3)")
    if grasp_center_w.shape != (distal_pos_w.shape[0], 3):
        raise ValueError("grasp centers must have shape (N, 3)")
    radius = max(float(grip_width_m) * 0.5, 1e-3)
    vectors = distal_pos_w - grasp_center_w[:, None]
    center_distances = vectors.norm(dim=-1)
    shell_error = (center_distances - radius).abs()
    band_reach = torch.exp(-24.0 * shell_error.mean(dim=-1))
    directions = torch.nn.functional.normalize(vectors, dim=-1, eps=1e-6)
    support = torch.nn.functional.normalize(
        directions[:, 1:].mean(dim=1), dim=-1, eps=1e-6
    )
    opposition = (
        (1.0 - (directions[:, 0] * support).sum(dim=-1)) * 0.5
    ).clamp(0.0, 1.0)
    return band_reach, opposition, center_distances.amin(dim=-1)


def object_grasp_progress(
    band_reach: torch.Tensor,
    opposition: torch.Tensor,
    closure: torch.Tensor,
    contact_quality: torch.Tensor,
) -> torch.Tensor:
    """Dense pre-contact progress with force contact as the strongest signal."""

    reach = band_reach.clamp(0.0, 1.0)
    geometric = reach * (
        0.55
        + 0.25 * opposition.clamp(0.0, 1.0)
        + 0.10 * closure.clamp(0.0, 1.0)
    )
    return (geometric + 0.10 * contact_quality.clamp(0.0, 1.0)).clamp(0.0, 1.0)


class GpuObjectGraspReward(GpuGoalGraphReward):
    """Per-object grasp shaping selected only by an explicit object contract."""

    profile = "object_grasp_v1"

    def __init__(
        self,
        *args,
        overhead_approach_clearance_m: float = 0.0,
        overhead_final_descent_weight: float = DEFAULT_OVERHEAD_FINAL_DESCENT_WEIGHT,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.object_contract = self.state_reader.gpu.bundle.manifest.get(
            "object_contract"
        )
        if self.spec.name != "grasp_anything" or self.object_contract is None:
            raise ValueError(
                "GpuObjectGraspReward requires grasp_anything and an object contract"
            )
        self.grasp_frame_position = torch.as_tensor(
            self.object_contract["grasp_frame_position_m"],
            dtype=torch.float32,
            device=self.device,
        )
        if self.grasp_frame_position.shape != (3,):
            raise ValueError("object grasp frame position must contain three values")
        self.grip_width_m = float(self.object_contract["grip_width_m"])
        self.grasp_closure_threshold = min(
            1.0, max(0.55, 0.55 + (0.09 - self.grip_width_m) * 7.5)
        )
        self.maximum_grip_force_newtons = float(
            self.object_contract["maximum_grip_force_newtons"]
        )
        self.overhead_approach_clearance_m = float(
            overhead_approach_clearance_m
        )
        self.overhead_final_descent_weight = float(overhead_final_descent_weight)
        if not (
            math.isfinite(self.overhead_approach_clearance_m)
            and 0.0 <= self.overhead_approach_clearance_m <= 0.5
        ):
            raise ValueError(
                "overhead approach clearance must be finite and in [0, 0.5]"
            )
        if not (
            math.isfinite(self.overhead_final_descent_weight)
            and 0.0 < self.overhead_final_descent_weight < 1.0
        ):
            raise ValueError(
                "overhead final descent weight must be finite and in (0, 1)"
            )
        self.last_grasp_band_reach = torch.zeros(
            self.num_envs, device=self.device
        )
        self.last_finger_opposition = torch.zeros_like(
            self.last_grasp_band_reach
        )
        self.last_closure = torch.zeros_like(self.last_grasp_band_reach)
        self.last_contact_quality = torch.zeros_like(self.last_grasp_band_reach)
        self.last_min_fingertip_distance = torch.zeros_like(
            self.last_grasp_band_reach
        )
        self.last_squeeze_penalty = torch.zeros_like(
            self.last_grasp_band_reach
        )
        self.last_overhead_approach_progress = torch.zeros_like(
            self.last_grasp_band_reach
        )
        self.last_approach_waypoint_distance = torch.zeros_like(
            self.last_grasp_band_reach
        )
        self.overhead_waypoint_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def metadata(self) -> dict[str, object]:
        resolved = {
            "schema_version": OBJECT_GRASP_REWARD_SCHEMA_VERSION,
            "profile": self.profile,
            "base_reward": super().metadata(),
            "task": self.spec.name,
            "object_id": self.object_contract["object_id"],
            "grasp_frame_position_m": self.grasp_frame_position.tolist(),
            "grip_width_m": self.grip_width_m,
            "closure_threshold": self.grasp_closure_threshold,
            "maximum_grip_force_newtons": self.maximum_grip_force_newtons,
            "stable_lift_required": True,
        }
        if self.overhead_approach_clearance_m > 0.0:
            resolved["overhead_approach_clearance_m"] = (
                self.overhead_approach_clearance_m
            )
            resolved["overhead_approach_profile"] = "fixed_clearance_gate_v2"
            if (
                self.overhead_final_descent_weight
                != DEFAULT_OVERHEAD_FINAL_DESCENT_WEIGHT
            ):
                resolved["overhead_final_descent_weight"] = (
                    self.overhead_final_descent_weight
                )
                resolved["overhead_approach_profile"] = "fixed_clearance_gate_v3"
        return {**resolved, "resolved_sha256": _json_hash(resolved)}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        for value in (
            self.last_grasp_band_reach,
            self.last_finger_opposition,
            self.last_closure,
            self.last_contact_quality,
            self.last_min_fingertip_distance,
            self.last_squeeze_penalty,
            self.last_overhead_approach_progress,
            self.last_approach_waypoint_distance,
        ):
            value[indices] = 0.0
        self.overhead_waypoint_reached[indices] = False

    def _approach_reward_progress(
        self,
        state: GpuTaskStateV2,
        fingertip_distances: torch.Tensor,
    ) -> torch.Tensor:
        legacy = super()._approach_reward_progress(state, fingertip_distances)
        if self.overhead_approach_clearance_m <= 0.0:
            return legacy
        grasp_center = state.primary.pos_w + torch.einsum(
            "bij,j->bi", state.primary.rot_w, self.grasp_frame_position
        )
        overhead, distance, _ = overhead_approach_progress(
            state.distal_pos_w[:, 1],
            grasp_center,
            clearance_m=self.overhead_approach_clearance_m,
        )
        centroid = state.distal_pos_w[:, 1].mean(dim=1)
        horizontal_distance = (centroid[:, :2] - grasp_center[:, :2]).norm(dim=-1)
        clearance_height = grasp_center[:, 2] + self.overhead_approach_clearance_m
        height_error = (centroid[:, 2] - clearance_height).abs()
        reached = (
            horizontal_distance <= max(0.5 * self.overhead_approach_clearance_m, 0.05)
        ) & (
            height_error <= max(0.35 * self.overhead_approach_clearance_m, 0.04)
        )
        self.overhead_waypoint_reached.logical_or_(reached)
        self.last_overhead_approach_progress.copy_(overhead)
        self.last_approach_waypoint_distance.copy_(distance)
        # Before the hand has physically cleared the support, direct low-path
        # proximity receives no credit. Once the fixed waypoint is reached,
        # retain that progress floor and shape the final descent to the object.
        return gated_overhead_approach_progress(
            overhead,
            legacy,
            self.overhead_waypoint_reached,
            final_descent_weight=self.overhead_final_descent_weight,
        )

    def _hand_grasp(
        self, state: GpuTaskStateV2, hand_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitudes = state.contact_forces_pelvis[:, hand_index].norm(dim=-1)
        thumb = magnitudes[:, 1:4].amax(dim=-1)
        support = magnitudes[:, 4:8].amax(dim=-1)
        closure = finger_closure_score(
            self.sim.data.qpos[:, self.hand_qpos_indices[hand_index]],
            self.initial_hand_qpos[:, hand_index],
        )
        normalized_closure = (
            closure / self.grasp_closure_threshold
        ).clamp(0.0, 1.0)
        quality = (
            (torch.minimum(thumb, support) / 8.0).clamp(0.0, 1.0)
            * normalized_closure
        )
        contact = state.predicates[:, hand_index].bool()
        grasp = (
            contact
            & (thumb > 2.0)
            & (support > 2.0)
            & (closure >= self.grasp_closure_threshold)
        )
        return grasp, quality

    def _object_metrics(
        self, state: GpuTaskStateV2, right_quality: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grasp_center = state.primary.pos_w + torch.einsum(
            "bij,j->bi", state.primary.rot_w, self.grasp_frame_position
        )
        reach, opposition, nearest = object_grasp_geometry(
            state.distal_pos_w[:, 1],
            grasp_center,
            grip_width_m=self.grip_width_m,
        )
        closure = finger_closure_score(
            self.sim.data.qpos[:, self.hand_qpos_indices[1]],
            self.initial_hand_qpos[:, 1],
        )
        self.last_grasp_band_reach.copy_(reach)
        self.last_finger_opposition.copy_(opposition)
        self.last_closure.copy_(closure)
        self.last_contact_quality.copy_(right_quality)
        self.last_min_fingertip_distance.copy_(nearest)
        return reach, opposition, closure, right_quality, nearest

    def _stage(
        self,
        stage: GoalStageSpec,
        state: GpuTaskStateV2,
        left_grasp: torch.Tensor,
        left_quality: torch.Tensor,
        right_grasp: torch.Tensor,
        right_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stage.hand == "right" and stage.primitive in ("approach", "grasp"):
            reach, opposition, closure, contact_quality, _ = self._object_metrics(
                state, right_quality
            )
            if stage.primitive == "approach":
                surface_distance = state.fingertip_distances[:, 1].amin(dim=-1)
                physical_contact = right_grasp | (contact_quality >= 0.05)
                complete = (surface_distance <= stage.threshold) & (
                    (reach >= 0.20) | physical_contact
                )
                return torch.maximum(reach, contact_quality), complete
            return (
                object_grasp_progress(
                    reach, opposition, closure, contact_quality
                ),
                right_grasp,
            )
        progress, complete = super()._stage(
            stage,
            state,
            left_grasp,
            left_quality,
            right_grasp,
            right_quality,
        )
        if stage.primitive == "lift":
            stable = (
                state.primary.lin_vel_w.norm(dim=-1)
                < self.spec.stable_linear_speed
            ) & (
                state.primary.ang_vel_w.norm(dim=-1)
                < self.spec.stable_angular_speed
            )
            complete &= stable
        return progress, complete

    def compute(
        self,
        state: GpuTaskStateV2,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        action_span: torch.Tensor,
    ) -> GpuRewardTerms:
        terms = super().compute(state, action, previous_action, action_span)
        # Refresh diagnostics during lift, where _stage no longer calls the
        # object-specific approach/grasp branch.
        _, right_quality = self._hand_grasp(state, 1)
        self._object_metrics(state, right_quality)
        magnitudes = state.contact_forces_pelvis[:, 1].norm(dim=-1)
        total_grip_force = magnitudes.sum(dim=-1)
        self.last_squeeze_penalty.copy_(
            (
                (total_grip_force - self.maximum_grip_force_newtons)
                / self.maximum_grip_force_newtons
            ).clamp(0.0, 2.0)
        )
        return replace(
            terms,
            penalty=terms.penalty + 0.25 * self.last_squeeze_penalty,
        )
