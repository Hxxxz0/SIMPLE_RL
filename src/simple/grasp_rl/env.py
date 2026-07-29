"""Headless single-process SIMPLE environment used by RL workers and replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

import simple.tasks  # noqa: F401 -- registers task classes
from simple.engines.mujoco import MujocoSimulator
from simple.grasp_rl.motion import MotionFrameExtractor
from simple.grasp_rl.rewards import (
    DEFAULT_TASK_REWARD_PROFILE,
    GraspReward,
    RewardTerms,
)
from simple.grasp_rl.schema import MAX_EPISODE_STEPS
from simple.grasp_rl.state import MujocoStateExtractor
from simple.grasp_rl.tracker import ActionTransform, stand_command, tracker_action_to_cmd
from simple.tasks.registry import TaskRegistry


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
        target_object: str = "graspnet1b:10",
        warmup_steps: int = 60,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        fast_reset: bool = True,
        task_reward_profile: str = DEFAULT_TASK_REWARD_PROFILE,
    ) -> None:
        task_cls = TaskRegistry._registry["g1_wholebody_tabletop_grasp_mp"]
        self.task = task_cls(
            target_object=target_object,
            render_hz=50,
            physics_dt=0.002,
            dr_level=0,
            split="train",
        )
        self.sim = MujocoSimulator(self.task, render_hz=50, physics_dt=0.002, headless=True)
        self.action_transform = action_transform
        self.seed = seed
        self.warmup_steps = warmup_steps
        self.max_episode_steps = max_episode_steps
        self.fast_reset = fast_reset
        self.task_reward_profile = task_reward_profile
        self.episode_index = 0
        self.rng = np.random.default_rng(seed)
        self._data_snapshot: dict[str, np.ndarray | float] | None = None
        self._robot_snapshot: dict[str, Any] | None = None
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
            self.sim.update_layout()
            self.sim.step(render=False)
            stand = stand_command()
            for _ in range(self.warmup_steps):
                self.sim.apply_action(stand)
                self.sim.step(render=False)
            self.previous_physical_action = self.action_transform.center.copy()
            if self.fast_reset and state_dict is None:
                self._capture_snapshot()

        self.state = MujocoStateExtractor(self.sim, self.previous_physical_action)
        if can_fast_reset and not self._fast_reset_randomize_target:
            assert self._initial_object_pos_snapshot is not None
            assert self._goal_pos_snapshot is not None
            self.state.initial_object_pos = self._initial_object_pos_snapshot.copy()
            self.state.goal_pos = self._goal_pos_snapshot.copy()
        self.motion = MotionFrameExtractor(self.sim.mjModel, self.sim.mjData, self.task.robot)
        self.reward = GraspReward(
            self.state,
            max_episode_steps=self.max_episode_steps,
            profile=self.task_reward_profile,
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
        self._previous_action_snapshot = self.previous_physical_action.copy()
        if preserve_task_origin and self.state is not None:
            self._initial_object_pos_snapshot = self.state.initial_object_pos.copy()
            self._goal_pos_snapshot = self.state.goal_pos.copy()
        else:
            current = data.body(self.sim.mj_objects["target"].id).xpos.copy()
            self._initial_object_pos_snapshot = current
            self._goal_pos_snapshot = current + np.array([0.0, 0.0, 0.025])
        target_body_id = self.sim.mj_objects["target"].id
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
        assert self._previous_action_snapshot is not None
        self.task.robot.set_runtime_state(self._robot_snapshot)
        self.previous_physical_action = self._previous_action_snapshot.copy()
        data = self.sim.mjData
        for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat"):
            getattr(data, name)[:] = self._data_snapshot[name]
        data.time = float(self._data_snapshot["time"])
        if self._fast_reset_randomize_target:
            address = self._target_qpos_adr
            if self._fast_reset_position_jitter_xy is None:
                # Preserve the original task distribution for legacy callers.
                data.qpos[address] = self.rng.uniform(-0.67, -0.62)
                data.qpos[address + 1] = self.rng.uniform(-0.03, 0.03)
            else:
                jitter_x, jitter_y = self._fast_reset_position_jitter_xy
                data.qpos[address] = self._target_base_position[0] + self.rng.uniform(
                    -jitter_x, jitter_x
                )
                data.qpos[address + 1] = self._target_base_position[1] + self.rng.uniform(
                    -jitter_y, jitter_y
                )
            yaw_jitter = (
                0.15
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
        self.state = MujocoStateExtractor(self.sim, self.previous_physical_action)
        self.motion = MotionFrameExtractor(
            self.sim.mjModel, self.sim.mjData, self.task.robot
        )
        self.reward = GraspReward(
            self.state,
            max_episode_steps=self.max_episode_steps,
            profile=self.task_reward_profile,
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
        self.sim.apply_action(tracker_action_to_cmd(physical, self.task.robot))
        self.sim.step(render=False)
        self.state.set_previous_action(physical)
        observation, target_state = self.state.actor_observation()
        terms = self.reward.compute(
            target_state,
            physical,
            previous,
            self.action_transform.high - self.action_transform.low,
        )
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
