"""Declarative SIMPLE task catalogue for the shared full-command RL stack.

Version one entries preserve the released grasp-only observation/checkpoints.
Version two entries use role-based entities and an ordered goal graph while
retaining the same complete 36-D tracker command.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Any, Literal

from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    ACTOR_OBS_V2_DIM,
    REFERENCE_ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_V2_DIM,
)


TASK_SCHEMA_VERSION = 2
DEFAULT_TASK = "tabletop_grasp"
TaskFamily = Literal["grasp", "place", "handover", "articulation", "push", "compound"]
ControllerBackend = Literal["amo", "sonic_wbc"]


@dataclass(frozen=True)
class GraspLiftRewardSpec:
    goal_lift: float
    success_lift: float
    success_hold_steps: int = 13
    ever_lifted: float = 0.015
    drop_lift: float = 0.005
    progress_lift: float = 0.02
    workspace_radius: float = 0.30
    stalled_grasp_steps: int | None = 40
    min_pelvis_height: float = 0.55


@dataclass(frozen=True)
class EntityRoles:
    primary: str = "target"
    destination: str | None = "table"
    auxiliary: str | None = None


@dataclass(frozen=True)
class GoalStageSpec:
    """One monotonic task stage interpreted by :mod:`goal_reward`."""

    name: str
    primitive: str
    threshold: float = 0.0
    hold_steps: int = 3
    hand: Literal["left", "right", "both", "none"] = "right"


@dataclass(frozen=True)
class ArticulationGoalSpec:
    joints: tuple[str, ...]
    targets: tuple[float, ...]
    comparison: Literal["ge", "le", "abs_ge", "any_ge"] = "ge"


@dataclass(frozen=True)
class GraspTaskSpec:
    """Frozen legacy task contract."""

    name: str
    registry_uid: str
    dataset_name: str
    label: str
    target_object: str
    max_episode_steps: int
    reward: GraspLiftRewardSpec
    native_target_x: tuple[float, float]
    native_target_y: tuple[float, float]
    native_yaw_jitter: float = 0.15
    robot_uid: str = "g1_wholebody"
    target_role: str = "target"
    support_role: str = "table"
    fast_reset: bool = True
    observation_schema: str = "grasp_privileged_v1"
    reference_schema: str = "complete_plan_10x40_plus_phase_v1"
    action_schema: str = "g1_wholebody_tracker_36_v1"
    schema_version: int = 1

    @property
    def actor_observation_dim(self) -> int:
        return ACTOR_OBS_DIM

    @property
    def reference_actor_observation_dim(self) -> int:
        return REFERENCE_ACTOR_OBS_DIM

    def dataset_path(self, root: str | Path = "data/simple") -> Path:
        return Path(root) / self.dataset_name

    def processed_path(self, root: str | Path = "data/grasp_rl") -> Path:
        return Path(root) / self.dataset_name

    def metadata(self) -> dict[str, Any]:
        return _metadata(self)


@dataclass(frozen=True)
class TaskSpecV2:
    name: str
    registry_uid: str
    source_uids: tuple[str, ...]
    dataset_name: str
    label: str
    family: TaskFamily
    controller_backend: ControllerBackend
    roles: EntityRoles
    stages: tuple[GoalStageSpec, ...]
    max_episode_steps: int = 800
    target_object: str | None = None
    articulation: ArticulationGoalSpec | None = None
    lift_height: float = 0.08
    placement_xy_tolerance: float = 0.12
    stable_linear_speed: float = 0.10
    stable_angular_speed: float = 1.0
    native_target_x: tuple[float, float] = (0.0, 0.0)
    native_target_y: tuple[float, float] = (0.0, 0.0)
    native_yaw_jitter: float = 0.15
    robot_uid: str = "g1_sonic"
    fast_reset: bool = False
    observation_schema: str = "task_privileged_v2"
    reference_schema: str = "complete_plan_10x51_plus_phase_v2"
    action_schema: str = "g1_wholebody_tracker_36_v1"
    schema_version: int = 2

    @property
    def target_role(self) -> str:
        return self.roles.primary

    @property
    def support_role(self) -> str:
        return self.roles.destination or "table"

    @property
    def actor_observation_dim(self) -> int:
        return ACTOR_OBS_V2_DIM

    @property
    def reference_actor_observation_dim(self) -> int:
        return REFERENCE_ACTOR_OBS_V2_DIM

    def dataset_path(self, root: str | Path = "data/simple") -> Path:
        return Path(root) / self.dataset_name

    def processed_path(self, root: str | Path = "data/grasp_rl") -> Path:
        return Path(root) / self.dataset_name / "v2"

    def metadata(self) -> dict[str, Any]:
        return _metadata(self)


TaskSpec = GraspTaskSpec | TaskSpecV2


def _metadata(spec: TaskSpec) -> dict[str, Any]:
    return {
        "task_schema_version": spec.schema_version,
        "task": spec.name,
        "registry_uid": spec.registry_uid,
        "robot_uid": spec.robot_uid,
        "observation_schema": spec.observation_schema,
        "reference_schema": spec.reference_schema,
        "action_schema": spec.action_schema,
        "actor_observation_dim": spec.actor_observation_dim,
        "reference_actor_observation_dim": spec.reference_actor_observation_dim,
        "action_dim": ACTION_DIM,
        "spec": asdict(spec),
    }


def _stages(*items: tuple[str, str, float, int, str]) -> tuple[GoalStageSpec, ...]:
    return tuple(GoalStageSpec(*item) for item in items)


_TASKS: dict[str, TaskSpec] = {
    "tabletop_grasp": GraspTaskSpec(
        name="tabletop_grasp", registry_uid="g1_wholebody_tabletop_grasp_mp",
        dataset_name="G1WholebodyTabletopGraspMP-v0", label="Tabletop grasp",
        target_object="graspnet1b:10", max_episode_steps=192,
        reward=GraspLiftRewardSpec(0.025, 0.020),
        native_target_x=(-0.67, -0.62), native_target_y=(-0.03, 0.03),
    ),
    "bend_pick": GraspTaskSpec(
        name="bend_pick", registry_uid="g1_wholebody_bend_pick_mp",
        dataset_name="G1WholebodyBendPickMP-v0", label="Bend pick MP",
        target_object="graspnet1b:0", max_episode_steps=300,
        reward=GraspLiftRewardSpec(0.09, 0.09, ever_lifted=0.03,
            drop_lift=0.015, progress_lift=0.09, stalled_grasp_steps=None,
            min_pelvis_height=0.50),
        native_target_x=(-0.32, -0.29), native_target_y=(-0.08, -0.04),
    ),
    "bend_pick_and_place": TaskSpecV2(
        "bend_pick_and_place", "g1_wholebody_bend_pick_and_place_teleop",
        ("g1_wholebody_bend_pick_and_place_teleop",),
        "G1WholebodyBendPickAndPlaceTeleop-v0", "Bend pick and place", "place",
        "sonic_wbc", EntityRoles("target", "container"),
        _stages(("approach", "approach", .06, 1, "right"),
                ("grasp", "grasp", 0, 3, "right"),
                ("lift", "lift", .05, 3, "right"),
                ("transport", "transport", .20, 1, "right"),
                ("place", "place", .12, 3, "right"),
                ("settle", "release_settle", 0, 13, "none")),
    ),
    "bend_pick_teleop": TaskSpecV2(
        "bend_pick_teleop", "g1_wholebody_bend_pick_teleop",
        ("g1_wholebody_bend_pick_teleop",), "G1WholebodyBendPickTeleop-v0",
        "Bend pick Teleop", "grasp", "sonic_wbc", EntityRoles("target", "table"),
        _stages(("approach", "approach", .06, 1, "right"),
                ("grasp", "grasp", 0, 3, "right"),
                ("lift", "lift", .08, 13, "right")), lift_height=.08,
    ),
    "close_door": TaskSpecV2(
        "close_door", "g1_wholebody_close_door_teleop",
        ("g1_wholebody_close_door_teleop",), "G1WholebodyCloseDoorTeleop-v0",
        "Close door", "articulation", "sonic_wbc", EntityRoles("target", None),
        _stages(("approach", "approach", .08, 1, "right"),
                ("contact", "contact", 0, 3, "right"),
                ("joint_goal", "articulation", 0, 13, "right")),
        articulation=ArticulationGoalSpec(("articulate_joint_1",), (-.16,), "le"),
    ),
    "handover_and_place": TaskSpecV2(
        "handover_and_place", "g1_wholebody_handover_teleop",
        ("g1_wholebody_handover_teleop",), "G1WholebodyHandoverTeleop-v0",
        "Handover and place", "handover", "sonic_wbc", EntityRoles("target", "container"),
        _stages(("right_grasp", "grasp", 0, 3, "right"),
                ("bimanual", "bimanual", 0, 3, "both"),
                ("left_only", "handover", 0, 3, "left"),
                ("place", "place", .12, 3, "left"),
                ("settle", "release_settle", 0, 13, "none")),
    ),
    "locomotion_pick_between_tables": TaskSpecV2(
        "locomotion_pick_between_tables",
        "g1_wholebody_locomotion_pick_between_tables_variant5_mp",
        ("g1_wholebody_locomotion_pick_between_tables_variant5",
         "g1_wholebody_locomotion_pick_between_tables_variant5_mp"),
        "G1WholebodyLocomotionPickBetweenTablesMixed-v0", "Cross-table pick and place",
        "place", "amo", EntityRoles("target", "container", "table2"),
        _stages(("approach", "approach", .06, 1, "right"),
                ("grasp", "grasp", 0, 3, "right"),
                ("lift", "lift", .05, 3, "right"),
                ("transport", "transport", .25, 1, "right"),
                ("place", "place", .12, 3, "right"),
                ("settle", "release_settle", 0, 13, "none")),
        # Repaired source plus feedback completion spans 827--1013 control
        # steps; 800 made successful expert trajectories time out before the
        # release-and-settle stage during policy evaluation.
        max_episode_steps=1100, target_object="graspnet1b:12", robot_uid="g1_wholebody",
    ),
    "open_faucet": TaskSpecV2(
        "open_faucet", "g1_wholebody_open_faucet_teleop",
        ("g1_wholebody_open_faucet_teleop",), "G1WholebodyOpenFaucetTeleop-v0",
        "Open faucet", "articulation", "sonic_wbc", EntityRoles("target", None),
        _stages(("approach", "approach", .08, 1, "right"),
                ("contact", "contact", 0, 3, "right"),
                ("joint_goal", "articulation", 0, 13, "right")),
        articulation=ArticulationGoalSpec(("articulate_joint_0",), (.7,), "abs_ge"),
    ),
    "open_oven": TaskSpecV2(
        "open_oven", "g1_wholebody_open_oven_teleop",
        ("g1_wholebody_open_oven_teleop",), "G1WholebodyOpenOvenTeleop-v0",
        "Open oven", "articulation", "sonic_wbc", EntityRoles("target", None),
        _stages(("approach", "approach", .08, 1, "right"),
                ("contact", "contact", 0, 3, "right"),
                ("joint_goal", "articulation", 0, 13, "right")),
        articulation=ArticulationGoalSpec(("articulate_joint_1", "articulate_joint_2"),
                                          (1.4, 1.4), "any_ge"),
    ),
    "open_trash_can": TaskSpecV2(
        "open_trash_can", "g1_wholebody_open_trash_can_teleop",
        ("g1_wholebody_open_trash_can_teleop",), "G1WholebodyOpenTrashCanTeleop-v0",
        "Open trash can", "articulation", "sonic_wbc", EntityRoles("target", None),
        _stages(("approach", "approach", .08, 1, "right"),
                ("contact", "contact", 0, 3, "right"),
                ("joint_goal", "articulation", 0, 13, "right")),
        articulation=ArticulationGoalSpec(("articulate_joint_0",), (.5,), "ge"),
    ),
    "pick_place_hug_container": TaskSpecV2(
        "pick_place_hug_container", "g1_wholebody_pick_and_place_and_hug_container_teleop",
        ("g1_wholebody_pick_and_place_and_hug_container_teleop",),
        "G1WholebodyPickAndPlaceAndHugContainerTeleop-v0", "Pick, hug and carry container",
        "compound", "sonic_wbc", EntityRoles("target", "table2", "container"),
        _stages(("item_grasp", "grasp", 0, 3, "right"),
                ("item_place", "place_aux", .12, 5, "right"),
                ("hug", "aux_bimanual", 0, 3, "both"),
                ("carry", "aux_transport", .25, 1, "both"),
                ("container_place", "aux_place", .15, 3, "both"),
                ("settle", "aux_release_settle", 0, 13, "none")),
    ),
    "push_office_chair": TaskSpecV2(
        "push_office_chair", "g1_wholebody_push_office_chair_teleop",
        ("g1_wholebody_push_office_chair_teleop",), "G1WholebodyPushOfficeChairTeleop-v0",
        "Push office chair", "push", "sonic_wbc", EntityRoles("target", "table"),
        _stages(("approach", "approach", .10, 1, "both"),
                ("contact", "contact", 0, 3, "both"),
                ("push", "push", .8, 13, "both")),
    ),
    "xmove_bend_pick": TaskSpecV2(
        "xmove_bend_pick", "g1_wholebody_xmove_bend_pick_teleop",
        ("g1_wholebody_xmove_bend_pick_teleop", "g1_sonic_xmove_bend_pick_teleop"), "G1WholebodyXMoveBendPickTeleop-v0",
        "Move and bend pick", "grasp", "sonic_wbc", EntityRoles("target", "table"),
        _stages(("approach", "approach", .06, 1, "right"),
                ("grasp", "grasp", 0, 3, "right"),
                ("lift", "lift", .08, 13, "right")), lift_height=.08,
    ),
    "xmove_pick": TaskSpecV2(
        "xmove_pick", "g1_wholebody_xmove_pick_teleop",
        ("g1_wholebody_xmove_pick_teleop",), "G1WholebodyXMovePickTeleop-v0",
        "Move and pick", "grasp", "sonic_wbc", EntityRoles("target", "table"),
        _stages(("approach", "approach", .06, 1, "right"),
                ("grasp", "grasp", 0, 3, "right"),
                ("lift", "lift", .09, 13, "right")), lift_height=.09,
    ),
    # GPU-only, one-object-per-policy specialization.  It deliberately reuses
    # the audited xmove task implementation and observation/action schemas;
    # object identity and geometry are frozen in a separate derived asset
    # bundle, so registering this entry cannot alter any existing task.
    "grasp_anything": TaskSpecV2(
        "grasp_anything", "g1_wholebody_xmove_pick_teleop",
        ("g1_wholebody_xmove_pick_teleop",),
        "G1WholebodyGraspAnythingPhysicalPPO-v0",
        "Grasp one frozen object", "grasp", "sonic_wbc",
        EntityRoles("target", "table"),
        _stages(("approach", "approach", .06, 1, "right"),
                ("grasp", "grasp", 0, 3, "right"),
                ("lift", "lift", .09, 13, "right")), lift_height=.09,
    ),
}


_ALIASES: dict[str, str] = {
    "tabletop": "tabletop_grasp", "bend": "bend_pick",
}
for _name, _spec in _TASKS.items():
    _ALIASES[_spec.dataset_name] = _name
    # grasp_anything deliberately reuses the xmove runtime UID, but the public
    # UID alias must continue resolving to the released xmove_pick task.  The
    # new specialization is selected only by its own task/dataset name.
    if _name == "grasp_anything":
        continue
    _ALIASES[_spec.registry_uid] = _name
    if isinstance(_spec, TaskSpecV2):
        for _uid in _spec.source_uids:
            _ALIASES[_uid] = _name


def task_names() -> tuple[str, ...]:
    return tuple(_TASKS)


def get_task_spec(task: str | TaskSpec | None = None) -> TaskSpec:
    if isinstance(task, (GraspTaskSpec, TaskSpecV2)):
        return task
    name = DEFAULT_TASK if task is None else _ALIASES.get(task, task)
    try:
        return _TASKS[name]
    except KeyError as error:
        raise ValueError(f"Unknown grasp-RL task {task!r}; choose one of {', '.join(task_names())}") from error


def task_from_manifest(processed_dir: str | Path) -> TaskSpec:
    import json
    path = Path(processed_dir) / "manifest.json"
    if not path.exists():
        return get_task_spec()
    return get_task_spec(json.loads(path.read_text()).get("task", DEFAULT_TASK))


def checkpoint_task_metadata(task: str | TaskSpec, action_transform: str | Path | None = None) -> dict[str, Any]:
    metadata = get_task_spec(task).metadata()
    if action_transform is not None:
        metadata["action_transform_sha256"] = hashlib.sha256(Path(action_transform).read_bytes()).hexdigest()
    return metadata


def validate_task_metadata(payload: dict[str, Any], task: str | TaskSpec, *,
                           checkpoint: str | Path | None = None,
                           action_transform: str | Path | None = None) -> None:
    expected = get_task_spec(task)
    metadata = payload.get("task_metadata")
    if metadata is None:
        if expected.name == DEFAULT_TASK:
            return
        location = f" {checkpoint}" if checkpoint is not None else ""
        raise ValueError(f"Checkpoint{location} has no task metadata and is only compatible with {DEFAULT_TASK}, not {expected.name}")
    mismatches = []
    for key, wanted in (
        ("task", expected.name), ("registry_uid", expected.registry_uid),
        ("observation_schema", expected.observation_schema),
        ("action_schema", expected.action_schema),
        ("actor_observation_dim", expected.actor_observation_dim),
        ("action_dim", ACTION_DIM),
    ):
        if metadata.get(key) != wanted:
            mismatches.append(f"{key}={metadata.get(key)!r} (expected {wanted!r})")
    if mismatches:
        raise ValueError(f"Checkpoint/task mismatch for {expected.name}: " + "; ".join(mismatches))
    expected_hash = metadata.get("action_transform_sha256")
    if expected_hash is not None and action_transform is not None:
        actual_hash = hashlib.sha256(Path(action_transform).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Checkpoint/action-transform mismatch: policy output scaling does not match the selected processed dataset")


class GraspTaskAdapter:
    """Construct legacy and v2 tasks while tolerating historical UID aliases."""

    def __init__(self, task: str | TaskSpec | None = None):
        self.spec = get_task_spec(task)

    def make_task(self, target_object: str | None = None,
                  sonic_config: dict[str, Any] | None = None):
        import simple.tasks  # noqa: F401
        from simple.tasks.registry import TaskRegistry

        task_cls = TaskRegistry._registry[self.spec.registry_uid]
        # Sonic's low-level controller is designed for four 5 ms physics steps
        # per 50 Hz command.  Passing the historical AMO default (2 ms) here
        # silently overwrote the task metadata and made recorded Sonic commands
        # physically non-reproducible during headless replay.
        physics_dt = (
            float(sonic_config["SIMULATE_DT"])
            if sonic_config is not None
            else .002
        )
        kwargs: dict[str, Any] = dict(
            render_hz=50,
            physics_dt=physics_dt,
            dr_level=0,
            split="train",
        )
        if sonic_config is not None:
            kwargs["sonic_config"] = sonic_config
        signature = inspect.signature(task_cls)
        selected = target_object or self.spec.target_object
        if selected is not None:
            if "target_object" in signature.parameters:
                kwargs["target_object"] = selected
            elif "target" in signature.parameters:
                kwargs["target"] = selected
        return task_cls(**kwargs)

    def native_success(self, lift_height: float) -> bool:
        if isinstance(self.spec, GraspTaskSpec):
            return bool(lift_height >= self.spec.reward.success_lift)
        return False
