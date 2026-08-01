"""Single source of truth for grasp-RL tensor layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

ACTION_DIM: Final = 36
# The original grasp-only contract remains frozen for released checkpoints.
ACTOR_OBS_DIM: Final = 192
# GRAIL conditions its actor on ten future reference frames spaced by 0.1 s.
# SIMPLE runs at 50 Hz, so the equivalent offsets are five control steps apart.
REFERENCE_FUTURE_OFFSETS: Final[tuple[int, ...]] = tuple(range(0, 50, 5))
# Per future frame: complete normalized tracker command, object-position delta,
# and the reference bilateral-contact label.  A final scalar stores phase.
REFERENCE_FRAME_DIM: Final = ACTION_DIM + 3 + 1
REFERENCE_CONTEXT_DIM: Final = (
    len(REFERENCE_FUTURE_OFFSETS) * REFERENCE_FRAME_DIM + 1
)
REFERENCE_ACTOR_OBS_DIM: Final = ACTOR_OBS_DIM + REFERENCE_CONTEXT_DIM

# Role-based task schema used by manipulation, transport, handover and
# articulated-object tasks.  It deliberately keeps a fixed size across tasks;
# absent entities/joints are zero-filled and accompanied by presence masks.
ACTOR_OBS_V2_DIM: Final = 331
REFERENCE_FRAME_V2_DIM: Final = 51
REFERENCE_CONTEXT_V2_DIM: Final = (
    len(REFERENCE_FUTURE_OFFSETS) * REFERENCE_FRAME_V2_DIM + 1
)
REFERENCE_ACTOR_OBS_V2_DIM: Final = ACTOR_OBS_V2_DIM + REFERENCE_CONTEXT_V2_DIM
MOTION_WINDOW: Final = 10
MOTION_FRAME_DIM: Final = 80
MOTION_FEATURE_DIM: Final = 82
MAX_EPISODE_STEPS: Final = 192

JOINT_NAMES: Final[tuple[str, ...]] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

MOTION_LINK_NAMES: Final[tuple[str, ...]] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_hand_thumb_2_link",
    "left_hand_index_1_link",
    "left_hand_middle_1_link",
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
)

RIGHT_CONTACT_LINK_NAMES: Final[tuple[str, ...]] = (
    "right_wrist_yaw_link",  # fixed proxy: this model has no palm body
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
)

LEFT_CONTACT_LINK_NAMES: Final[tuple[str, ...]] = tuple(
    name.replace("right_", "left_", 1) for name in RIGHT_CONTACT_LINK_NAMES
)

RIGHT_DISTAL_LINK_NAMES: Final[tuple[str, str, str]] = (
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
)
LEFT_DISTAL_LINK_NAMES: Final[tuple[str, str, str]] = tuple(
    name.replace("right_", "left_", 1) for name in RIGHT_DISTAL_LINK_NAMES
)


@dataclass(frozen=True)
class SliceSpec:
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


ACTION_SLICES: Final[dict[str, SliceSpec]] = {
    "left_hand": SliceSpec(0, 7),
    "right_hand": SliceSpec(7, 14),
    "left_arm": SliceSpec(14, 21),
    "right_arm": SliceSpec(21, 28),
    "torso_rpy": SliceSpec(28, 31),
    "base_height": SliceSpec(31, 32),
    "torso_vx": SliceSpec(32, 33),
    "torso_vy": SliceSpec(33, 34),
    "turning_flag": SliceSpec(34, 35),
    "target_yaw": SliceSpec(35, 36),
}


def schema_dict() -> dict:
    return {
        "action_dim": ACTION_DIM,
        "actor_obs_dim": ACTOR_OBS_DIM,
        "reference_actor_obs_dim": REFERENCE_ACTOR_OBS_DIM,
        "reference_future_offsets": list(REFERENCE_FUTURE_OFFSETS),
        "reference_frame_dim": REFERENCE_FRAME_DIM,
        "reference_context_dim": REFERENCE_CONTEXT_DIM,
        "actor_obs_v2_dim": ACTOR_OBS_V2_DIM,
        "reference_actor_obs_v2_dim": REFERENCE_ACTOR_OBS_V2_DIM,
        "reference_frame_v2_dim": REFERENCE_FRAME_V2_DIM,
        "reference_context_v2_dim": REFERENCE_CONTEXT_V2_DIM,
        "motion_window": MOTION_WINDOW,
        "motion_frame_dim": MOTION_FRAME_DIM,
        "motion_feature_dim": MOTION_FEATURE_DIM,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "joint_names": list(JOINT_NAMES),
        "motion_link_names": list(MOTION_LINK_NAMES),
        "right_contact_link_names": list(RIGHT_CONTACT_LINK_NAMES),
        "left_contact_link_names": list(LEFT_CONTACT_LINK_NAMES),
        "action_slices": {key: asdict(value) for key, value in ACTION_SLICES.items()},
    }


def validate_schema() -> None:
    assert len(JOINT_NAMES) == 43
    assert len(MOTION_LINK_NAMES) == 8
    assert len(RIGHT_CONTACT_LINK_NAMES) == 8
    assert len(LEFT_CONTACT_LINK_NAMES) == 8
    assert max(spec.stop for spec in ACTION_SLICES.values()) == ACTION_DIM
    assert REFERENCE_FRAME_DIM == 40
    assert REFERENCE_CONTEXT_DIM == 401
    assert REFERENCE_ACTOR_OBS_DIM == 593
    assert REFERENCE_CONTEXT_V2_DIM == 511
    assert REFERENCE_ACTOR_OBS_V2_DIM == 842
    assert 3 + 4 + 3 + 3 + len(JOINT_NAMES) + 3 * len(MOTION_LINK_NAMES) == MOTION_FRAME_DIM
    assert 3 + 6 + len(JOINT_NAMES) + 3 * len(MOTION_LINK_NAMES) + 3 + 3 == MOTION_FEATURE_DIM


validate_schema()


def base_observation_dim(observation_dim: int) -> int:
    """Return the state prefix length for a supported actor observation."""

    if observation_dim in (ACTOR_OBS_DIM, REFERENCE_ACTOR_OBS_DIM):
        return ACTOR_OBS_DIM
    if observation_dim in (ACTOR_OBS_V2_DIM, REFERENCE_ACTOR_OBS_V2_DIM):
        return ACTOR_OBS_V2_DIM
    raise ValueError(f"Unsupported actor observation dimension {observation_dim}")


def reference_observation_dim(actor_observation_dim: int) -> int:
    if actor_observation_dim == ACTOR_OBS_DIM:
        return REFERENCE_ACTOR_OBS_DIM
    if actor_observation_dim == ACTOR_OBS_V2_DIM:
        return REFERENCE_ACTOR_OBS_V2_DIM
    raise ValueError(f"Unsupported base observation dimension {actor_observation_dim}")
