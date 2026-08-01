"""Execution adapters for the two whole-body controller families in SIMPLE."""

from __future__ import annotations

import copy
from typing import Protocol
import time

import numpy as np

from simple.core.action import ActionCmd
from simple.grasp_rl.schema import ACTION_DIM
from simple.grasp_rl.tracker import stand_command, tracker_action_to_cmd, upper_joints_from_tracker


class ControllerBackend(Protocol):
    def reset(self) -> None: ...
    def command(self, action: np.ndarray) -> ActionCmd: ...
    def stabilize_command(self) -> ActionCmd: ...
    def get_runtime_state(self) -> dict: ...
    def set_runtime_state(self, state: dict) -> None: ...


class AmoControllerBackend:
    def __init__(self, robot):
        self.robot = robot

    def reset(self) -> None:
        return None

    def command(self, action: np.ndarray) -> ActionCmd:
        return tracker_action_to_cmd(action, self.robot)

    def stabilize_command(self) -> ActionCmd:
        return stand_command()

    def get_runtime_state(self) -> dict:
        return {}

    def set_runtime_state(self, state: dict) -> None:
        if state:
            raise ValueError("AMO controller has no runtime state")


class SonicWbcControllerBackend:
    """Convert the shared 36-D command through SIMPLE's decoupled WBC."""

    def __init__(self, robot):
        from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent

        sonic_config = getattr(robot, "sonic_config", None)
        if not sonic_config:
            raise ValueError("g1_sonic must be initialized with sonic_config")
        self.robot = robot
        self.agent = SonicDecoupledWbcAgent(robot, sonic_config=sonic_config)
        self._control_time = time.monotonic()
        indices = self.agent._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.upper_names = [name for name, index in self.agent._dwbc_robot_model.joint_to_dof_index.items()
                            if index in indices]

    def reset(self) -> None:
        self.agent.reset()
        # Official Sonic replay/eval explicitly engages the learned lower-body
        # controller before stabilization.  Without this flag the robot never
        # reaches the recorded walking state and eventually falls when 36-D
        # navigation commands begin.
        self.agent._wbc_policy.lower_body_policy.use_policy_action = True
        self._control_time = time.monotonic()

    def _next_control_time(self) -> float:
        now = self._control_time
        self._control_time += 1.0 / self.agent._control_frequency
        return now

    def stabilize_command(self) -> ActionCmd:
        return self.agent.get_stabilize_action(
            None,
            control_time=self._next_control_time(),
        )

    def command(self, action: np.ndarray) -> ActionCmd:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected ({ACTION_DIM},), got {action.shape}")
        mapped = upper_joints_from_tracker(action)
        targets = dict(zip(self.robot.joint_names[15:], mapped, strict=True))
        targets.update({
            "waist_yaw_joint": float(action[30]),
            "waist_roll_joint": float(action[28]),
            "waist_pitch_joint": float(action[29]),
        })
        target_upper = np.asarray([targets[name] for name in self.upper_names], dtype=np.float32)
        proprio = self.robot.prepare_obs()
        observation = self.agent._build_wbc_observation(proprio)
        self.agent._wbc_policy.set_observation(observation)
        now = self._next_control_time()
        frequency = self.agent._control_frequency
        self.agent._wbc_policy.set_goal({
            "target_upper_body_pose": target_upper,
            "navigate_cmd": action[32:36],
            "base_height_command": action[31:32],
            "target_time": now + 1 / frequency,
            "interpolation_garbage_collection_time": now - 2 / frequency,
            "timestamp": now,
        })
        # The upstream communication watchdog records wall time internally.
        # Headless replay deliberately advances a deterministic control clock,
        # so keep the watchdog in that same clock domain before querying WBC.
        self.agent._wbc_policy.last_goal_time = now
        wbc_action = self.agent._wbc_policy.get_action(time=now)
        model = self.agent._dwbc_robot_model
        return ActionCmd(
            "decoupled_wbc",
            target_q=model.get_body_actuated_joints(wbc_action["q"]),
            left_hand_q=model.get_hand_actuated_joints(wbc_action["q"], side="left"),
            right_hand_q=model.get_hand_actuated_joints(wbc_action["q"], side="right"),
        )

    @staticmethod
    def _copy_fields(instance, names: tuple[str, ...]) -> dict:
        return {
            name: copy.deepcopy(getattr(instance, name))
            for name in names
            if hasattr(instance, name)
        }

    def get_runtime_state(self) -> dict:
        """Capture mutable Sonic histories without copying ONNX sessions."""

        whole = self.agent._wbc_policy
        upper = whole.upper_body_policy
        lower = whole.lower_body_policy
        return {
            "control_time": self._control_time,
            "agent": self._copy_fields(
                self.agent,
                (
                    "_cached_target_q",
                    "_cached_left_hand_q",
                    "_cached_right_hand_q",
                ),
            ),
            "whole": self._copy_fields(
                whole, ("last_goal_time", "is_in_teleop_mode", "last_action")
            ),
            "upper": self._copy_fields(
                upper,
                ("interp", "last_waypoint_time", "last_action"),
            ),
            "lower": self._copy_fields(
                lower,
                (
                    "observation",
                    "obs_history",
                    "obs_buffer",
                    "obs_tensor",
                    "counter",
                    "use_policy_action",
                    "use_teleop_policy_cmd",
                    "action",
                    "target_dof_pos",
                    "cmd",
                    "height_cmd",
                    "freq_cmd",
                    "roll_cmd",
                    "pitch_cmd",
                    "yaw_cmd",
                    "target_yaw_cmd",
                    "gait_indices",
                    "foot_indices",
                    "clock_inputs",
                    "last_action",
                ),
            ),
        }

    def set_runtime_state(self, state: dict) -> None:
        """Restore the exact interpolation and lower-body policy histories."""

        self._control_time = float(state["control_time"])
        whole = self.agent._wbc_policy
        targets = (
            (self.agent, state["agent"]),
            (whole, state["whole"]),
            (whole.upper_body_policy, state["upper"]),
            (whole.lower_body_policy, state["lower"]),
        )
        for instance, values in targets:
            for name, value in values.items():
                setattr(instance, name, copy.deepcopy(value))


def make_controller_backend(name: str, robot) -> ControllerBackend:
    if name == "amo":
        return AmoControllerBackend(robot)
    if name == "sonic_wbc":
        return SonicWbcControllerBackend(robot)
    raise ValueError(f"Unknown controller backend {name!r}")
