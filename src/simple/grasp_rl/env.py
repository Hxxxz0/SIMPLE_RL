"""Headless single-process SIMPLE environment used by RL workers and replay."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import mujoco
import numpy as np

from simple.engines.mujoco import MujocoSimulator
from simple.grasp_rl.motion import MotionFrameExtractor
from simple.grasp_rl.controller_backend import ControllerBackend, make_controller_backend
from simple.grasp_rl.goal_reward import GoalGraphReward
from simple.grasp_rl.rewards import (
    DEFAULT_TASK_REWARD_PROFILE,
    GraspReward,
    RewardTerms,
)
from simple.grasp_rl.state import MujocoStateExtractor
from simple.grasp_rl.state_v2 import MujocoTaskStateExtractor
from simple.grasp_rl.task_spec import GraspTaskAdapter, TaskSpec, TaskSpecV2
from simple.grasp_rl.tracker import ActionTransform


@dataclass
class EnvStep:
    actor_observation: np.ndarray
    motion_frame: np.ndarray
    terms: RewardTerms
    done: bool


class GraspRlEnv:
    """One task instance and one MuJoCo simulator, without camera rendering."""

    def __init__(
        self,
        action_transform: ActionTransform,
        seed: int = 0,
        target_object: str | None = None,
        warmup_steps: int = 60,
        max_episode_steps: int | None = None,
        fast_reset: bool | None = None,
        task_reward_profile: str = DEFAULT_TASK_REWARD_PROFILE,
        task: str | TaskSpec | None = None,
        enable_renderers: bool = False,
    ) -> None:
        self.task_adapter = GraspTaskAdapter(task)
        self.task_spec = self.task_adapter.spec
        self.sonic_config: dict[str, Any] | None = None
        if isinstance(self.task_spec, TaskSpecV2) and self.task_spec.controller_backend == "sonic_wbc":
            from simple.cli.eval_decoupled_wbc import _make_sonic_config
            self.sonic_config = _make_sonic_config()
        self.task = self.task_adapter.make_task(target_object, sonic_config=self.sonic_config)
        physics_dt = (
            float(self.sonic_config["SIMULATE_DT"])
            if self.sonic_config is not None
            else 0.002
        )
        self.sim = MujocoSimulator(
            self.task,
            render_hz=50,
            physics_dt=physics_dt,
            headless=True,
            enable_renderers=enable_renderers,
        )
        self.control_decimation = (
            int(round(1.0 / (50.0 * physics_dt)))
            if self.sonic_config is not None
            else 1
        )
        if self.sonic_config is not None and not np.isclose(
            self.control_decimation * physics_dt, 1.0 / 50.0
        ):
            raise ValueError(
                "Sonic physics_dt must divide the 50 Hz policy interval exactly"
            )
        self.action_transform = action_transform
        self.seed = seed
        self.warmup_steps = warmup_steps
        self.max_episode_steps = (
            self.task_spec.max_episode_steps
            if max_episode_steps is None
            else max_episode_steps
        )
        self.fast_reset = self.task_spec.fast_reset if fast_reset is None else fast_reset
        self.task_reward_profile = task_reward_profile
        self.episode_index = 0
        self.rng = np.random.default_rng(seed)
        self._data_snapshot: dict[str, np.ndarray | float] | None = None
        self._robot_snapshot: dict[str, Any] | None = None
        self._controller_snapshot: dict[str, Any] | None = None
        self._previous_action_snapshot: np.ndarray | None = None
        self._initial_object_pos_snapshot: np.ndarray | None = None
        self._goal_pos_snapshot: np.ndarray | None = None
        self._target_qpos_adr: int | None = None
        self._target_base_position: np.ndarray | None = None
        self._target_base_quat: np.ndarray | None = None
        self._fast_reset_randomize_target = True
        self._fast_reset_position_jitter_xy: tuple[float, float] | None = None
        self._fast_reset_yaw_jitter: float | None = None
        self.previous_physical_action = action_transform.center.copy()
        self.state: MujocoStateExtractor | None = None
        self.motion: MotionFrameExtractor | None = None
        self.reward: GraspReward | None = None
        self.controller: ControllerBackend | None = None
        self._training_dr_model: mujoco.MjModel | None = None
        self._training_dr_body_mass: np.ndarray | None = None
        self._training_dr_geom_friction: np.ndarray | None = None

    def _configure_target(self, target_object: str | None) -> None:
        if target_object is None:
            return
        target = self.task.dr.get_randomizer("target")
        if target is None:
            raise RuntimeError("Task has no target randomizer")
        target.res_id, target.obj_id = target_object.split(":")

    def reset(
        self,
        state_dict: dict[str, Any] | None = None,
        target_object: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if state_dict is not None:
            state_uid = state_dict.get("uid")
            valid_uids = (
                self.task_spec.source_uids
                if isinstance(self.task_spec, TaskSpecV2)
                else (self.task_spec.registry_uid,)
            )
            if state_uid not in valid_uids:
                raise ValueError(
                    f"Environment state belongs to {state_uid!r}, not "
                    f"task {self.task_spec.registry_uid!r}"
                )
            runtime_uid = self.task.uid
            if state_uid != runtime_uid:
                state_dict = copy.deepcopy(state_dict)
                state_dict["uid"] = runtime_uid
        can_fast_reset = (
            self.fast_reset
            and self._data_snapshot is not None
            and state_dict is None
            and target_object is None
        )
        if can_fast_reset:
            self._restore_snapshot_and_randomize_target()
        else:
            self._configure_target(target_object)
            reset_seed = self.seed + self.episode_index
            self.task.reset(
                seed=reset_seed,
                options={"state_dict": state_dict} if state_dict is not None else None,
            )
            if self.sonic_config is None:
                self.sim.update_layout()
            else:
                self.sim.update_layout(sonic_config=self.sonic_config)
            if self.controller is None:
                self.controller = make_controller_backend(
                    self.task_spec.controller_backend
                    if isinstance(self.task_spec, TaskSpecV2)
                    else "amo",
                    self.task.robot,
                )
            self.controller.reset()
            self.sim.step(render=False)
            warmup_count = 0
            while True:
                stabilized = (
                    bool(self.task.robot.stabilized)
                    if self.sonic_config is not None
                    else True
                )
                if warmup_count >= self.warmup_steps and stabilized:
                    break
                if warmup_count >= 300:
                    # Match the bounded evaluation loop: proceed after the
                    # stabilization budget even when the very strict floating
                    # base velocity latch has not fired.
                    break
                command = self.controller.stabilize_command()
                for _ in range(self.control_decimation):
                    self.sim.apply_action(command)
                    self.sim.step(render=False)
                warmup_count += 1
            self.previous_physical_action = self.action_transform.center.copy()
            if self.fast_reset and state_dict is None:
                self._capture_snapshot()

        if isinstance(self.task_spec, TaskSpecV2):
            self.state = MujocoTaskStateExtractor(
                self.sim, self.task_spec, self.previous_physical_action
            )
        else:
            self.state = MujocoStateExtractor(
                self.sim,
                self.previous_physical_action,
                goal_lift=self.task_spec.reward.goal_lift,
                target_role=self.task_spec.target_role,
                support_role=self.task_spec.support_role,
            )
        if (
            can_fast_reset
            and not self._fast_reset_randomize_target
            and not isinstance(self.task_spec, TaskSpecV2)
        ):
            assert self._initial_object_pos_snapshot is not None
            assert self._goal_pos_snapshot is not None
            self.state.initial_object_pos = self._initial_object_pos_snapshot.copy()
            self.state.goal_pos = self._goal_pos_snapshot.copy()
        self.motion = MotionFrameExtractor(self.sim.mjModel, self.sim.mjData, self.task.robot)
        self.reward = (
            GoalGraphReward(
                self.state,
                self.task_spec,
                self.max_episode_steps,
                profile=self.task_reward_profile,
            )
            if isinstance(self.task_spec, TaskSpecV2)
            else GraspReward(
                self.state,
                max_episode_steps=self.max_episode_steps,
                profile=self.task_reward_profile,
                reward_spec=self.task_spec.reward,
            )
        )
        observation, _ = self.state.actor_observation()
        frame = self.motion.extract()
        self.episode_index += 1
        return observation, frame

    def _capture_snapshot(self, preserve_task_origin: bool = False) -> None:
        """Cache a stabilized MuJoCo state for inexpensive training resets."""
        data = self.sim.mjData
        self._data_snapshot = {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "act": data.act.copy(),
            "ctrl": data.ctrl.copy(),
            "qacc_warmstart": data.qacc_warmstart.copy(),
            "mocap_pos": data.mocap_pos.copy(),
            "mocap_quat": data.mocap_quat.copy(),
            "time": float(data.time),
        }
        self._robot_snapshot = self.task.robot.get_runtime_state()
        assert self.controller is not None
        self._controller_snapshot = self.controller.get_runtime_state()
        self._previous_action_snapshot = self.previous_physical_action.copy()
        is_v2 = isinstance(self.task_spec, TaskSpecV2)
        if preserve_task_origin and self.state is not None and not is_v2:
            self._initial_object_pos_snapshot = self.state.initial_object_pos.copy()
            self._goal_pos_snapshot = self.state.goal_pos.copy()
        else:
            target_body_id = (
                self.state.role_body_ids["primary"]
                if is_v2 and self.state is not None
                else self.sim.mj_objects[self.task_spec.target_role].id
            )
            if target_body_id is None:
                raise RuntimeError("Task has no primary body for fast reset")
            current = data.body(target_body_id).xpos.copy()
            self._initial_object_pos_snapshot = current
            self._goal_pos_snapshot = current + np.array(
                [
                    0.0,
                    0.0,
                    self.task_spec.lift_height
                    if is_v2
                    else self.task_spec.reward.goal_lift,
                ]
            )
        target_body_id = (
            self.state.role_body_ids["primary"]
            if is_v2 and self.state is not None
            else self.sim.mj_objects[self.task_spec.target_role].id
        )
        if target_body_id is None:
            raise RuntimeError("Task has no primary body for fast reset")
        joint_id = int(self.sim.mjModel.body_jntadr[target_body_id])
        if joint_id < 0 or self.sim.mjModel.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("The grasp target must have a free joint for fast reset")
        self._target_qpos_adr = int(self.sim.mjModel.jnt_qposadr[joint_id])
        self._target_base_position = data.qpos[
            self._target_qpos_adr : self._target_qpos_adr + 3
        ].copy()
        self._target_base_quat = data.qpos[self._target_qpos_adr + 3 : self._target_qpos_adr + 7].copy()

    def capture_fast_reset_snapshot(
        self,
        randomize_target: bool = True,
        position_jitter_xy: tuple[float, float] | None = None,
        yaw_jitter: float | None = None,
    ) -> None:
        """Make the current state the fast-reset origin.

        RSI uses ``randomize_target=False`` so a demonstration prefix is replayed
        only in its exact recorded scene.  Randomized plan-conditioned training
        instead keeps the recorded robot/reference plan while moving the target
        relative to its recorded pose.  This directly trains the feedback policy
        required for producing new trajectories rather than merely replaying one.
        """
        if position_jitter_xy is not None:
            if len(position_jitter_xy) != 2 or any(
                value < 0.0 for value in position_jitter_xy
            ):
                raise ValueError("position_jitter_xy must contain two non-negative values")
            position_jitter_xy = tuple(float(value) for value in position_jitter_xy)
        if yaw_jitter is not None and yaw_jitter < 0.0:
            raise ValueError("yaw_jitter must be non-negative")
        self._capture_snapshot(preserve_task_origin=True)
        self._fast_reset_randomize_target = randomize_target
        self._fast_reset_position_jitter_xy = position_jitter_xy
        self._fast_reset_yaw_jitter = (
            None if yaw_jitter is None else float(yaw_jitter)
        )
        self.fast_reset = True

    @staticmethod
    def _yaw_quat_product(base: np.ndarray, desired_yaw: float) -> np.ndarray:
        w, x, y, z = base
        initial_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        half = 0.5 * (desired_yaw - initial_yaw)
        cw, sz = np.cos(half), np.sin(half)
        result = np.array(
            [cw * w - sz * z, cw * x - sz * y, cw * y + sz * x, cw * z + sz * w],
            dtype=np.float64,
        )
        return result / np.linalg.norm(result)

    def _restore_snapshot_and_randomize_target(self) -> None:
        """Restore robot/dynamics state without rebuilding the MuJoCo model."""
        assert self._data_snapshot is not None
        assert self._target_qpos_adr is not None
        assert self._target_base_position is not None
        assert self._target_base_quat is not None
        assert self._robot_snapshot is not None
        assert self._controller_snapshot is not None
        assert self._previous_action_snapshot is not None
        self.task.robot.set_runtime_state(self._robot_snapshot)
        assert self.controller is not None
        self.controller.set_runtime_state(self._controller_snapshot)
        self.previous_physical_action = self._previous_action_snapshot.copy()
        data = self.sim.mjData
        for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat"):
            getattr(data, name)[:] = self._data_snapshot[name]
        data.time = float(self._data_snapshot["time"])
        if self._fast_reset_randomize_target:
            address = self._target_qpos_adr
            if self._fast_reset_position_jitter_xy is None:
                # Preserve the original task distribution for legacy callers.
                data.qpos[address] = self.rng.uniform(
                    *self.task_spec.native_target_x
                )
                data.qpos[address + 1] = self.rng.uniform(
                    *self.task_spec.native_target_y
                )
            else:
                jitter_x, jitter_y = self._fast_reset_position_jitter_xy
                data.qpos[address] = self._target_base_position[0] + self.rng.uniform(
                    -jitter_x, jitter_x
                )
                data.qpos[address + 1] = self._target_base_position[1] + self.rng.uniform(
                    -jitter_y, jitter_y
                )
            yaw_jitter = (
                self.task_spec.native_yaw_jitter
                if self._fast_reset_yaw_jitter is None
                else self._fast_reset_yaw_jitter
            )
            base = self._target_base_quat
            w, x, y, z = base
            base_yaw = float(
                np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            )
            data.qpos[address + 3 : address + 7] = self._yaw_quat_product(
                base, base_yaw + self.rng.uniform(-yaw_jitter, yaw_jitter)
            )
        self.sim.render_step = 0
        mujoco.mj_forward(self.sim.mjModel, data)

    def reset_to_target_pose(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Restore the cached scene at an explicit target pose.

        This is used for fair fallback-plan attempts: every reference rank sees
        exactly the same robot, target, and tracker initial state instead of a
        newly sampled easier task.
        """
        if isinstance(self.task_spec, TaskSpecV2):
            raise RuntimeError("reset_to_target_pose is a legacy single-object operation")
        if self._data_snapshot is None or self._target_qpos_adr is None:
            raise RuntimeError("capture_fast_reset_snapshot must be called first")
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("position/quaternion must have shapes (3,) and (4,)")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-8:
            raise ValueError("target quaternion must be non-zero")
        randomize = self._fast_reset_randomize_target
        self._fast_reset_randomize_target = False
        try:
            self.reset()
        finally:
            self._fast_reset_randomize_target = randomize
        address = self._target_qpos_adr
        self.sim.mjData.qpos[address : address + 3] = position
        self.sim.mjData.qpos[address + 3 : address + 7] = quaternion / norm
        mujoco.mj_forward(self.sim.mjModel, self.sim.mjData)
        self.state = MujocoStateExtractor(
            self.sim,
            self.previous_physical_action,
            goal_lift=self.task_spec.reward.goal_lift,
            target_role=self.task_spec.target_role,
            support_role=self.task_spec.support_role,
        )
        self.motion = MotionFrameExtractor(
            self.sim.mjModel, self.sim.mjData, self.task.robot
        )
        self.reward = GraspReward(
            self.state,
            max_episode_steps=self.max_episode_steps,
            profile=self.task_reward_profile,
            reward_spec=self.task_spec.reward,
        )
        observation, _ = self.state.actor_observation()
        return observation, self.motion.extract()

    def target_freejoint_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the target free-joint position and quaternion."""
        if self._target_qpos_adr is None:
            raise RuntimeError("The target free joint has not been configured")
        address = self._target_qpos_adr
        return (
            self.sim.mjData.qpos[address : address + 3].copy(),
            self.sim.mjData.qpos[address + 3 : address + 7].copy(),
        )

    def randomize_primary_pose(
        self,
        position_jitter_xy: tuple[float, float],
        yaw_jitter: float,
        position_offset_center_xy: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Offset and jitter a v2 primary body, then rebuild task state.

        A non-zero center supports hard-example curricula: the simulator still
        samples a continuous neighbourhood, but most episodes lie on the side
        of the recorded reference where replay is known to fail.
        """

        if not isinstance(self.task_spec, TaskSpecV2) or self.state is None:
            raise RuntimeError("randomize_primary_pose requires a reset v2 task")
        joint_id, address = self._primary_freejoint()
        base_position = self.sim.mjData.qpos[address:address + 3].copy()
        base_quat = self.sim.mjData.qpos[address + 3:address + 7].copy()
        jitter_x, jitter_y = position_jitter_xy
        center_x, center_y = position_offset_center_xy
        self.sim.mjData.qpos[address] += center_x + self.rng.uniform(
            -jitter_x, jitter_x
        )
        self.sim.mjData.qpos[address + 1] += center_y + self.rng.uniform(
            -jitter_y, jitter_y
        )
        w, x, y, z = base_quat
        base_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        self.sim.mjData.qpos[address + 3:address + 7] = self._yaw_quat_product(
            base_quat, base_yaw + self.rng.uniform(-yaw_jitter, yaw_jitter)
        )
        dof_address = int(self.sim.mjModel.jnt_dofadr[joint_id])
        self.sim.mjData.qvel[dof_address:dof_address + 6] = 0.0
        mujoco.mj_forward(self.sim.mjModel, self.sim.mjData)
        self._rebuild_v2_state()
        return base_position, base_quat

    def _primary_freejoint(self) -> tuple[int, int]:
        if not isinstance(self.task_spec, TaskSpecV2) or self.state is None:
            raise RuntimeError("primary free joint requires a reset v2 task")
        primary_id = self.state.role_body_ids["primary"]
        if primary_id is None:
            raise RuntimeError("v2 task has no primary entity")
        joint_id = int(self.sim.mjModel.body_jntadr[primary_id])
        if (
            joint_id < 0
            or self.sim.mjModel.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise RuntimeError("v2 primary entity does not have a free joint")
        return joint_id, int(self.sim.mjModel.jnt_qposadr[joint_id])

    def _rebuild_v2_state(self) -> None:
        self.state = MujocoTaskStateExtractor(self.sim, self.task_spec, self.previous_physical_action)
        self.motion = MotionFrameExtractor(self.sim.mjModel, self.sim.mjData, self.task.robot)
        self.reward = GoalGraphReward(
            self.state,
            self.task_spec,
            self.max_episode_steps,
            profile=self.task_reward_profile,
        )

    def primary_freejoint_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current v2 primary-object pose."""

        _, address = self._primary_freejoint()
        return (
            self.sim.mjData.qpos[address:address + 3].copy(),
            self.sim.mjData.qpos[address + 3:address + 7].copy(),
        )

    def set_primary_pose(
        self, position: np.ndarray, quaternion: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Set a v2 free primary pose and reset task-relative reward state."""

        joint_id, address = self._primary_freejoint()
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("position/quaternion must have shapes (3,) and (4,)")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-8:
            raise ValueError("primary quaternion must be non-zero")
        self.sim.mjData.qpos[address:address + 3] = position
        self.sim.mjData.qpos[address + 3:address + 7] = quaternion / norm
        dof_address = int(self.sim.mjModel.jnt_dofadr[joint_id])
        self.sim.mjData.qvel[dof_address:dof_address + 6] = 0.0
        mujoco.mj_forward(self.sim.mjModel, self.sim.mjData)
        self._rebuild_v2_state()
        observation, _ = self.state.actor_observation()
        return observation, self.motion.extract()

    def restore_v2_state(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Restore an evaluated v2 initial state for exact closed-loop replay."""

        if not isinstance(self.task_spec, TaskSpecV2) or self.state is None:
            raise RuntimeError("restore_v2_state requires a reset v2 task")
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        if qpos.shape != self.sim.mjData.qpos.shape:
            raise ValueError(
                f"Expected saved qpos shape {self.sim.mjData.qpos.shape}, got {qpos.shape}"
            )
        if qvel.shape != self.sim.mjData.qvel.shape:
            raise ValueError(
                f"Expected saved qvel shape {self.sim.mjData.qvel.shape}, got {qvel.shape}"
            )
        self.sim.mjData.qpos[:] = qpos
        self.sim.mjData.qvel[:] = qvel
        mujoco.mj_forward(self.sim.mjModel, self.sim.mjData)
        self._rebuild_v2_state()
        observation, _ = self.state.actor_observation()
        return observation, self.motion.extract()

    def randomize_training_physics(
        self,
        target_mass_scale: float,
        friction_scale: float,
    ) -> dict[str, float]:
        """Apply reset-scoped training DR without changing replay semantics.

        The baseline arrays are restored before every sample.  Callers must opt
        in explicitly; reward audits and ``step_physical`` never invoke this.
        """

        if self.state is None:
            raise RuntimeError("Call reset before applying training DR")
        if target_mass_scale <= 0.0 or friction_scale <= 0.0:
            raise ValueError("Training DR scales must be positive")
        model = self.sim.mjModel
        if self._training_dr_model is not model:
            self._training_dr_model = model
            self._training_dr_body_mass = model.body_mass.copy()
            self._training_dr_geom_friction = model.geom_friction.copy()
        assert self._training_dr_body_mass is not None
        assert self._training_dr_geom_friction is not None
        model.body_mass[:] = self._training_dr_body_mass
        model.geom_friction[:] = self._training_dr_geom_friction

        if isinstance(self.task_spec, TaskSpecV2):
            target_bodies = self.state.role_body_sets["primary"]
            friction_geoms = set(self.state.role_geom_ids["primary"])
            friction_geoms.update(self.state.role_geom_ids["support"])
            friction_geoms.update(self.state.role_geom_ids["destination"])
        else:
            target_bodies = {
                body_id
                for body_id in range(model.nbody)
                if self.state._is_descendant(body_id, self.state.target_id)
            }
            friction_geoms = set(self.state.target_geom_ids)
            friction_geoms.update(self.state._subtree_geom_ids(self.state.table_id))

        body_indices = np.fromiter(sorted(target_bodies), dtype=np.int32)
        geom_indices = np.fromiter(sorted(friction_geoms), dtype=np.int32)
        if body_indices.size:
            model.body_mass[body_indices] *= float(target_mass_scale)
        if geom_indices.size:
            model.geom_friction[geom_indices] *= float(friction_scale)
        mujoco.mj_forward(model, self.sim.mjData)
        return {
            "target_mass_scale": float(target_mass_scale),
            "friction_scale": float(friction_scale),
        }

    def target_pose_from_qpos(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract the target pose from a saved full-model qpos vector."""
        if self._target_qpos_adr is None:
            raise RuntimeError("The target free joint has not been configured")
        qpos = np.asarray(qpos)
        if qpos.shape != self.sim.mjData.qpos.shape:
            raise ValueError(
                f"Expected saved qpos shape {self.sim.mjData.qpos.shape}, got {qpos.shape}"
            )
        address = self._target_qpos_adr
        return (
            qpos[address : address + 3].copy(),
            qpos[address + 3 : address + 7].copy(),
        )

    def step_raw(self, raw_action: np.ndarray) -> EnvStep:
        if self.state is None or self.motion is None or self.reward is None:
            raise RuntimeError("Call reset before step")
        physical = self.action_transform.decode(raw_action, self.previous_physical_action)
        return self.step_physical(physical)

    def step_physical(self, physical_action: np.ndarray) -> EnvStep:
        """Execute an already-decoded tracker action (used for exact replay)."""
        if self.state is None or self.motion is None or self.reward is None:
            raise RuntimeError("Call reset before step")
        physical = np.asarray(physical_action, dtype=np.float32)
        if physical.shape != (36,):
            raise ValueError(f"Expected physical action (36,), got {physical.shape}")
        previous = self.previous_physical_action.copy()
        if self.controller is None:
            raise RuntimeError("Controller backend is not initialized")
        command = self.controller.command(physical)
        for _ in range(self.control_decimation):
            self.sim.apply_action(command)
            self.sim.step(render=False)
        self.state.set_previous_action(physical)
        observation, target_state = self.state.actor_observation()
        terms = self.reward.compute(
            target_state,
            physical,
            previous,
            self.action_transform.high - self.action_transform.low,
        )
        if isinstance(self.task_spec, TaskSpecV2):
            # Include the stage transition caused by this state in the policy's
            # next observation instead of exposing a stale one-step context.
            observation, _ = self.state.actor_observation()
        self.previous_physical_action = physical
        done = terms.success or terms.failure or terms.timeout
        return EnvStep(
            actor_observation=observation,
            motion_frame=self.motion.extract(),
            terms=terms,
            done=done,
        )

    def close(self) -> None:
        self.sim.close()
