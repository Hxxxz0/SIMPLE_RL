"""Reference-free GRAIL-style grasp reward components."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from simple.grasp_rl.state import MujocoStateExtractor, TargetState
from simple.grasp_rl.schema import MAX_EPISODE_STEPS


REWARD_VARIANTS = (
    "task_only",
    "smp_additive",
    "smp_product",
    "smp_product_strict",
)
TASK_REWARD_PROFILES = (
    "dense_v1",
    "progress_v2",
    "grail_v3",
    "grail_release_v1",
)
DEFAULT_TASK_REWARD_PROFILE = "grail_release_v1"


def compose_reward(
    target,
    penalty,
    terminal_adjustment,
    smp,
    smp_active,
    variant: str,
    task_weight: float = 0.02,
    smp_weight: float = 0.01,
):
    """Combine task and frozen-prior rewards identically in train and eval.

    SMP's released ``task_smp_product`` multiplies the task target by its
    unconditional diffusion score; ``smp_product_strict`` preserves that
    structure while leaving safety penalties outside the product.  Additive is
    a requested ablation for sparse grasp rewards, not the original SMP rule.
    """
    if variant not in REWARD_VARIANTS:
        raise ValueError(f"Unknown reward variant {variant}")
    task_component = task_weight * (target - penalty)
    if variant == "task_only":
        shaped = task_component
    elif variant == "smp_additive":
        shaped = task_component + smp_weight * smp * smp_active
    elif variant == "smp_product":
        shaped = task_component * smp
    else:
        shaped = task_weight * (target * smp - penalty)
    smp_contribution = shaped - task_component
    return shaped + terminal_adjustment, task_component, smp_contribution


@dataclass
class RewardTerms:
    reach: float
    pregrasp: float
    contact: float
    grasp_quality: float
    finger: float
    xy: float
    lift: float
    stable: float
    hold: float
    progress: float
    progress_bonus: float
    grail_grasp: float
    grail_finger_direction: float
    approach_penalty: float
    table_penalty: float
    action_rate_penalty: float
    joint_limit_penalty: float
    target_reward: float
    penalty: float
    terminal_adjustment: float
    success: bool
    failure: bool
    timeout: bool
    lift_height: float
    is_grasp: bool
    hand_table_force: float

    def to_dict(self) -> dict:
        return asdict(self)


class GraspReward:
    def __init__(
        self,
        extractor: MujocoStateExtractor,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        profile: str = "dense_v1",
    ):
        if profile not in TASK_REWARD_PROFILES:
            raise ValueError(f"Unknown task reward profile {profile}")
        self.extractor = extractor
        self.max_episode_steps = max_episode_steps
        self.profile = profile
        self.reset()

    def reset(self) -> None:
        self.step_count = 0
        self.hold_count = 0
        self.fall_count = 0
        self.danger_count = 0
        self.grasp_without_lift_count = 0
        self.grasp_attempt_started = False
        self.ever_lifted = False
        self.best_progress = 0.0
        self.previous_progress: float | None = None
        self.reference_should_contact: float | None = None

    def set_reference_contact(self, should_contact: float | None) -> None:
        """Set GRAIL's per-frame reference contact intent for the next step."""
        self.reference_should_contact = (
            None if should_contact is None else float(bool(should_contact))
        )

    @staticmethod
    def _terminal_adjustment(
        success: bool,
        fallen: bool,
        dropped: bool,
        failure: bool = False,
        timeout: bool = False,
        profile: str = "dense_v1",
    ) -> float:
        """Make completing the grasp preferable to stalling until timeout."""
        if profile in {"grail_v3", "grail_release_v1"}:
            if success:
                return 20.0
            if fallen or dropped or failure:
                return -10.0
            if timeout:
                return -5.0
            return 0.0
        if profile == "progress_v2":
            if success:
                return 10.0
            if fallen:
                return -5.0
            if dropped:
                return -3.0
            if failure or timeout:
                return -2.0
            return 0.0
        if success:
            return 5.0
        if fallen:
            return -1.0
        if dropped:
            return -0.5
        return 0.0

    @staticmethod
    def _finger_opposition(state: TargetState, active: bool) -> float:
        if not active:
            return 0.0
        # GRAIL pnp_table explicitly enables ``use_contact_center``.  MuJoCo
        # exposes the corresponding object/finger contact positions directly;
        # use their center while contact exists and fall back to object CoM for
        # the pre-contact geometry signal.
        center = (
            state.contact.object_contact_center_w
            if state.contact.has_object_contact_center
            else state.object_pos_w
        )
        vectors = state.distal_pos_w - center[None, :]
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True).clip(1e-6)
        support = vectors[1:].mean(axis=0)
        support /= max(np.linalg.norm(support), 1e-6)
        return float(np.clip((1.0 - np.dot(vectors[0], support)) * 0.5, 0.0, 1.0))

    @staticmethod
    def _progress_target(
        progress: float,
        best_progress: float,
        hold: float,
        stable: float,
    ) -> tuple[float, float, float]:
        """Return non-farmable progress shaping and the updated best value."""
        progress_bonus = 50.0 * max(progress - best_progress, 0.0)
        best_progress = max(best_progress, progress)
        target = progress_bonus + 10.0 * hold + 2.0 * stable
        return target, best_progress, progress_bonus

    @staticmethod
    def _grail_target(
        progress: float,
        previous_progress: float | None,
        lifted_grasp: bool,
        stable: float,
        gamma: float = 0.99,
    ) -> tuple[float, float, float]:
        """Potential shaping plus a short lifted-grasp hold objective.

        GRAIL emits grasp and reference-tracking rewards every frame.  That is
        appropriate while a time-indexed reference advances, but an absolute
        reach/contact reward is farmable in this reference-free task.  The
        potential difference preserves dense approach/grasp/lift feedback and
        telescopes under the PPO discount.  A stationary table-height contact
        therefore cannot accumulate positive return.
        """
        potential_delta = (
            0.0
            if previous_progress is None
            else gamma * progress - previous_progress
        )
        shaping = 50.0 * potential_delta
        target = shaping + 5.0 * float(lifted_grasp) + 2.0 * stable
        return target, progress, shaping

    @staticmethod
    def _grail_release_target(
        pregrasp: float,
        grasp: float,
        finger_direction: float,
        lift: float,
        stable: float,
    ) -> float:
        """GRAIL target terms plus reference-free approach/lift feedback."""
        return (
            pregrasp
            + 5.0 * grasp
            + 10.0 * finger_direction
            + 5.0 * lift
            + 2.0 * stable
        )

    def _joint_limit_penalty(self) -> float:
        penalties = []
        for name in self.extractor.robot.joint_names:
            joint = self.extractor.data.joint(name)
            low, high = self.extractor.model.jnt_range[joint.id]
            span = high - low
            if span <= 1e-5:
                continue
            margin = 0.05 * span
            distance = min(joint.qpos.item() - low, high - joint.qpos.item())
            penalties.append(np.clip((margin - distance) / margin, 0.0, 1.0))
        return float(np.mean(penalties)) if penalties else 0.0

    def compute(
        self,
        state: TargetState,
        action: np.ndarray,
        previous_action: np.ndarray,
        action_span: np.ndarray,
    ) -> RewardTerms:
        self.step_count += 1
        initial = self.extractor.initial_object_pos
        lift_height = float(state.object_pos_w[2] - initial[2])
        distance = float(np.linalg.norm(state.hand_pos_w - state.object_pos_w))
        fingertip_distances = np.linalg.norm(
            state.distal_pos_w - state.object_pos_w[None, :], axis=-1
        )
        # Center-distance minus an approximate chips-can radius gives a smooth
        # surface-distance curriculum.  The geometric mean requires both the
        # thumb and at least one opposing finger to approach the object.
        surface_distances = np.maximum(fingertip_distances - 0.05, 0.0)
        thumb_near = np.exp(-10.0 * surface_distances[0])
        support_near = np.exp(-10.0 * min(surface_distances[1], surface_distances[2]))
        pregrasp = float(np.sqrt(thumb_near * support_near))
        contacts = state.contact.group_contacts
        force_scores = np.clip(state.contact.group_forces / 2.0, 0.0, 1.0)
        support_force = max(float(force_scores[1]), float(force_scores[2]))
        grasp_quality = float(np.sqrt(float(force_scores[0]) * support_force))
        is_grasp = state.contact.is_grasp

        reach = float(np.exp(-10.0 * distance))
        contact = float(force_scores.mean())
        finger = self._finger_opposition(state, distance < 0.12 or bool(contacts.any()))
        # Released GRAIL pnp_table counts all eight palm/finger sensor links and
        # divides by min_contacts=8.  Use the same link-level contact fraction
        # and its signed thumb-vs-support direction term here.
        grail_contacts = state.contact.link_force_magnitudes > 0.1
        contact_intent = (
            float(is_grasp)
            if self.reference_should_contact is None
            else self.reference_should_contact
        )
        grail_grasp = (
            float(np.clip(grail_contacts.sum() / 8.0, 0.0, 1.0))
            * contact_intent
        )
        # GRAIL can gate this term with a time-indexed reference contact label.
        # SMP has no such phase label, so require the simulated bilateral grasp;
        # gating on any single contact would permit a one-finger reward exploit.
        grail_finger_direction = (2.0 * finger - 1.0) * contact_intent
        xy_error_sq = float(np.sum((state.object_pos_w[:2] - initial[:2]) ** 2))
        xy = grasp_quality * float(np.exp(-50.0 * xy_error_sq))
        lift = grasp_quality * float(np.clip(lift_height / 0.025, 0.0, 1.0))
        lift_gate = float(np.clip(lift_height / 0.02, 0.0, 1.0))
        motion_stability = float(
            np.exp(
                -5.0 * np.sum(state.object_lin_vel_w**2)
                - np.sum(state.object_ang_vel_w**2)
            )
        )
        stable = grasp_quality * lift_gate * motion_stability
        hold = grasp_quality * lift_gate
        progress = (
            0.5 * reach
            + pregrasp
            + 1.5 * grasp_quality
            + 0.5 * finger
            + 2.0 * lift
            + stable
        )
        progress_target, self.best_progress, progress_bonus = self._progress_target(
            progress, self.best_progress, hold, stable
        )
        # GRAIL gates object tracking with simulated grasp contact.  Without an
        # object reference trajectory, absolute low velocity is not equivalent
        # to velocity tracking: it would reject a correctly controlled upward
        # lift. Require a bilateral grasp while lifted instead. A thrown object
        # loses ``is_grasp`` and therefore cannot accumulate the hold counter.
        success_gate = lift_height >= 0.02 and is_grasp

        grail_target, self.previous_progress, grail_shaping = self._grail_target(
            progress,
            self.previous_progress,
            success_gate,
            stable,
        )

        if self.profile == "grail_release_v1":
            # Exact released GRAIL approach term: absolute wrist speed, Gaussian
            # wrist-to-object distance gate, and disable_after_contact=false.
            wrist_distance = float(
                np.linalg.norm(state.wrist_pos_w - state.object_pos_w)
            )
            approach = float(
                np.sum(state.wrist_lin_vel_w**2)
                * np.exp(-(wrist_distance**2) / (0.25**2))
            )
            # GRAIL clamps total table/hand contact force directly to [0, 1].
            table = float(np.clip(state.contact.hand_table_force, 0.0, 1.0))
        else:
            relative_velocity = state.hand_lin_vel_w - state.object_lin_vel_w
            approach = float(
                np.clip(
                    np.sum(relative_velocity**2)
                    * np.exp(-(distance**2) / (0.25**2))
                    * (not is_grasp),
                    0.0,
                    1.0,
                )
            )
            table = float(
                np.clip(
                    (state.contact.hand_table_force - 50.0) / 70.0,
                    0.0,
                    1.0,
                )
            )
        normalized_delta = (action - previous_action) / np.maximum(action_span, 1e-4)
        action_rate = float(np.clip(np.mean(normalized_delta**2), 0.0, 1.0))
        joint_limit = self._joint_limit_penalty()

        if self.profile == "grail_release_v1":
            # Direct adaptation of the released GRAIL pnp_table task terms:
            # grasp=5, opposing finger direction=10.  GRAIL obtains lift intent
            # and approach guidance from its advancing robot-motion reference.
            # SMP has no target direction, so enable GRAIL's implemented (but
            # release-config-disabled) hand/object distance term and add small
            # contact-gated lift/stability terms.
            target = self._grail_release_target(
                pregrasp,
                grail_grasp,
                grail_finger_direction,
                lift,
                stable,
            )
            progress_bonus = 0.0
        elif self.profile == "grail_v3":
            target = grail_target
            progress_bonus = grail_shaping
        elif self.profile == "progress_v2":
            # Reward only new task progress, plus lifted bilateral hold.  The
            # dense-v1 absolute contact terms can otherwise be farmed at table
            # height for the full horizon and occasionally outscore an early
            # successful grasp.
            target = progress_target
        else:
            target = (
                reach
                + 2.0 * pregrasp
                + contact
                + 2.0 * grasp_quality
                + 0.5 * finger
                + 0.25 * xy
                + 4.0 * lift
                + stable
                + 4.0 * hold
            )
        if self.profile == "grail_release_v1":
            penalty = 15.0 * approach + table + 0.1 * action_rate
        elif self.profile == "grail_v3":
            # Relative magnitudes follow GRAIL pnp_table: approach velocity is
            # the dominant dense safety term, followed by hand/table contact
            # and small action/joint regularizers.  ``compose_reward`` applies
            # the common task scale afterwards.
            penalty = (
                15.0 * approach
                + table
                + 0.1 * action_rate
                + 0.1 * joint_limit
            )
        else:
            penalty = 0.05 * approach + 0.05 * table + 0.005 * action_rate + 0.01 * joint_limit

        self.hold_count = (
            self.hold_count + 1 if success_gate else max(self.hold_count - 1, 0)
        )
        self.fall_count = (
            self.fall_count + 1
            if state.pelvis_height < 0.55 or state.pelvis_rot_w[2, 2] < 0.5
            else 0
        )
        self.danger_count = self.danger_count + 1 if state.contact.hand_table_force > 120.0 else 0
        self.ever_lifted |= lift_height >= 0.015
        self.grasp_attempt_started |= is_grasp
        if lift_height >= 0.005:
            self.grasp_without_lift_count = 0
        elif self.grasp_attempt_started:
            # Match GRAIL's first-contact grace-period semantics: losing contact
            # briefly must not reset the stalled-grasp timer and enable farming.
            self.grasp_without_lift_count += 1

        success = self.hold_count >= 13
        dropped = self.ever_lifted and lift_height < 0.005 and not is_grasp
        workspace_exit = np.linalg.norm(state.object_pos_w[:2] - initial[:2]) > 0.30
        stalled_grasp = (
            self.profile == "grail_release_v1"
            and self.grasp_without_lift_count >= 40
        )
        failure = bool(
            self.fall_count >= 3
            or self.danger_count >= 3
            or dropped
            or workspace_exit
            or stalled_grasp
        )
        success = bool(success)
        timeout = bool(self.step_count >= self.max_episode_steps and not (success or failure))
        terminal_adjustment = self._terminal_adjustment(
            success=success,
            fallen=self.fall_count >= 3,
            dropped=bool(dropped),
            failure=failure,
            timeout=timeout,
            profile=self.profile,
        )
        return RewardTerms(
            reach=reach,
            pregrasp=pregrasp,
            contact=contact,
            grasp_quality=grasp_quality,
            finger=finger,
            xy=xy,
            lift=lift,
            stable=stable,
            hold=hold,
            progress=progress,
            progress_bonus=progress_bonus,
            grail_grasp=grail_grasp,
            grail_finger_direction=grail_finger_direction,
            approach_penalty=approach,
            table_penalty=table,
            action_rate_penalty=action_rate,
            joint_limit_penalty=joint_limit,
            target_reward=target,
            penalty=penalty,
            terminal_adjustment=terminal_adjustment,
            success=success,
            failure=failure,
            timeout=timeout,
            lift_height=lift_height,
            is_grasp=is_grasp,
            hand_table_force=float(state.contact.hand_table_force),
        )
