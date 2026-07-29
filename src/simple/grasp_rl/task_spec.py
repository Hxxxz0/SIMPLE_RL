"""Task adapters for reusing the grasp-RL pipeline across SIMPLE tasks.

The policy/tracker contract is deliberately shared: every adapter in this
module consumes the frozen 192-D privileged observation (593-D with a plan)
and emits SIMPLE's complete 36-D whole-body tracker command.  Task-specific
scene construction, reset distribution, goal, and termination live here
instead of being scattered through training and evaluation code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

from simple.grasp_rl.schema import ACTION_DIM, ACTOR_OBS_DIM, REFERENCE_ACTOR_OBS_DIM


TASK_SCHEMA_VERSION = 1
DEFAULT_TASK = "tabletop_grasp"


@dataclass(frozen=True)
class GraspLiftRewardSpec:
    """Physical thresholds for one grasp-and-lift task."""

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
class GraspTaskSpec:
    """Declarative contract between a SIMPLE task and the shared RL stack."""

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

    def dataset_path(self, root: str | Path = "data/simple") -> Path:
        return Path(root) / self.dataset_name

    def processed_path(self, root: str | Path = "data/grasp_rl") -> Path:
        return Path(root) / self.dataset_name

    def metadata(self) -> dict[str, Any]:
        return {
            "task_schema_version": TASK_SCHEMA_VERSION,
            "task": self.name,
            "registry_uid": self.registry_uid,
            "robot_uid": self.robot_uid,
            "observation_schema": self.observation_schema,
            "reference_schema": self.reference_schema,
            "action_schema": self.action_schema,
            "actor_observation_dim": ACTOR_OBS_DIM,
            "reference_actor_observation_dim": REFERENCE_ACTOR_OBS_DIM,
            "action_dim": ACTION_DIM,
            "spec": asdict(self),
        }


_TASKS = {
    "tabletop_grasp": GraspTaskSpec(
        name="tabletop_grasp",
        registry_uid="g1_wholebody_tabletop_grasp_mp",
        dataset_name="G1WholebodyTabletopGraspMP-v0",
        label="G1 Wholebody Tabletop Grasp",
        target_object="graspnet1b:10",
        max_episode_steps=192,
        reward=GraspLiftRewardSpec(
            goal_lift=0.025,
            success_lift=0.020,
            ever_lifted=0.015,
            drop_lift=0.005,
            progress_lift=0.020,
        ),
        native_target_x=(-0.67, -0.62),
        native_target_y=(-0.03, 0.03),
    ),
    "bend_pick": GraspTaskSpec(
        name="bend_pick",
        registry_uid="g1_wholebody_bend_pick_mp",
        dataset_name="G1WholebodyBendPickMP-v0",
        label="G1 Wholebody Bend Pick",
        target_object="graspnet1b:0",
        max_episode_steps=300,
        # The native task declares LIFT_HEIGHT=0.10 and success_criteria=0.9.
        # Thus 9 cm is the native physical success boundary.
        reward=GraspLiftRewardSpec(
            goal_lift=0.090,
            success_lift=0.090,
            ever_lifted=0.030,
            drop_lift=0.015,
            progress_lift=0.090,
            # Bend trajectories can make incidental early finger contact long
            # before the torso finishes descending.  GRAIL's short-horizon
            # first-contact timeout is therefore not a valid failure signal.
            stalled_grasp_steps=None,
            # Valid demonstrations reach roughly 0.533 m ankle-relative
            # pelvis height while the torso remains upright.
            min_pelvis_height=0.50,
        ),
        native_target_x=(-0.32, -0.29),
        native_target_y=(-0.08, -0.04),
    ),
}

_ALIASES = {
    "G1WholebodyTabletopGraspMP-v0": "tabletop_grasp",
    "g1_wholebody_tabletop_grasp_mp": "tabletop_grasp",
    "tabletop": "tabletop_grasp",
    "G1WholebodyBendPickMP-v0": "bend_pick",
    "g1_wholebody_bend_pick_mp": "bend_pick",
    "bend": "bend_pick",
}


def task_names() -> tuple[str, ...]:
    return tuple(_TASKS)


def get_task_spec(task: str | GraspTaskSpec | None = None) -> GraspTaskSpec:
    if isinstance(task, GraspTaskSpec):
        return task
    name = DEFAULT_TASK if task is None else _ALIASES.get(task, task)
    try:
        return _TASKS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown grasp-RL task {task!r}; choose one of {', '.join(task_names())}"
        ) from error


def task_from_manifest(processed_dir: str | Path) -> GraspTaskSpec:
    import json

    manifest_path = Path(processed_dir) / "manifest.json"
    if not manifest_path.exists():
        return get_task_spec()
    manifest = json.loads(manifest_path.read_text())
    return get_task_spec(manifest.get("task", DEFAULT_TASK))


def checkpoint_task_metadata(
    task: str | GraspTaskSpec,
    action_transform: str | Path | None = None,
) -> dict[str, Any]:
    metadata = get_task_spec(task).metadata()
    if action_transform is not None:
        path = Path(action_transform)
        metadata["action_transform_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return metadata


def validate_task_metadata(
    payload: dict[str, Any],
    task: str | GraspTaskSpec,
    *,
    checkpoint: str | Path | None = None,
    action_transform: str | Path | None = None,
) -> None:
    """Fail early when a policy is paired with an incompatible task schema."""

    expected = get_task_spec(task)
    metadata = payload.get("task_metadata")
    if metadata is None:
        # Existing released checkpoints predate task metadata and are known to
        # use the original tabletop schema.  They must never silently run on a
        # newly added task.
        if expected.name == DEFAULT_TASK:
            return
        location = f" {checkpoint}" if checkpoint is not None else ""
        raise ValueError(
            f"Checkpoint{location} has no task metadata and is only compatible "
            f"with {DEFAULT_TASK}, not {expected.name}"
        )
    actual = get_task_spec(metadata.get("task"))
    mismatches = []
    for key, wanted in (
        ("task", expected.name),
        ("registry_uid", expected.registry_uid),
        ("observation_schema", expected.observation_schema),
        ("action_schema", expected.action_schema),
        ("actor_observation_dim", ACTOR_OBS_DIM),
        ("action_dim", ACTION_DIM),
    ):
        got = metadata.get(key)
        if got != wanted:
            mismatches.append(f"{key}={got!r} (expected {wanted!r})")
    if actual.name != expected.name or mismatches:
        raise ValueError(
            f"Checkpoint/task mismatch for {expected.name}: " + "; ".join(mismatches)
        )
    expected_transform = metadata.get("action_transform_sha256")
    if expected_transform is not None and action_transform is not None:
        actual_transform = hashlib.sha256(
            Path(action_transform).read_bytes()
        ).hexdigest()
        if actual_transform != expected_transform:
            raise ValueError(
                "Checkpoint/action-transform mismatch: policy output scaling "
                "does not match the selected processed dataset"
            )


class GraspTaskAdapter:
    """Runtime adapter that constructs and evaluates a configured SIMPLE task."""

    def __init__(self, task: str | GraspTaskSpec | None = None):
        self.spec = get_task_spec(task)

    def make_task(self, target_object: str | None = None):
        import simple.tasks  # noqa: F401 -- populate TaskRegistry
        from simple.tasks.registry import TaskRegistry

        task_cls = TaskRegistry._registry[self.spec.registry_uid]
        return task_cls(
            target_object=target_object or self.spec.target_object,
            render_hz=50,
            physics_dt=0.002,
            dr_level=0,
            split="train",
        )

    def native_success(self, lift_height: float) -> bool:
        return bool(lift_height >= self.spec.reward.success_lift)
