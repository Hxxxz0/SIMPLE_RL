"""MuJoCo proprioception, object state, and contact extraction."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from simple.grasp_rl.schema import (
    ACTOR_OBS_DIM,
    JOINT_NAMES,
    RIGHT_CONTACT_LINK_NAMES,
    RIGHT_DISTAL_LINK_NAMES,
)


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)


def _body_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
    )
    return velocity[3:].copy(), velocity[:3].copy()


@dataclass
class ContactState:
    link_forces_pelvis: np.ndarray
    link_force_magnitudes: np.ndarray
    group_forces: np.ndarray
    object_contact_center_w: np.ndarray
    has_object_contact_center: bool
    hand_table_force: float

    @property
    def group_contacts(self) -> np.ndarray:
        return self.group_forces > 2.0

    @property
    def is_grasp(self) -> bool:
        thumb, index, middle = self.group_contacts
        return bool(thumb and (index or middle))


@dataclass
class TargetState:
    object_pos_w: np.ndarray
    object_rot_w: np.ndarray
    object_lin_vel_w: np.ndarray
    object_ang_vel_w: np.ndarray
    hand_pos_w: np.ndarray
    hand_rot_w: np.ndarray
    hand_lin_vel_w: np.ndarray
    wrist_pos_w: np.ndarray
    wrist_lin_vel_w: np.ndarray
    distal_pos_w: np.ndarray
    table_pos_w: np.ndarray
    table_rot_w: np.ndarray
    pelvis_pos_w: np.ndarray
    pelvis_rot_w: np.ndarray
    pelvis_height: float
    contact: ContactState


class MujocoStateExtractor:
    """Resolve model IDs once and expose the fixed 192-D actor observation."""

    def __init__(self, simulator, previous_action: np.ndarray | None = None):
        self.sim = simulator
        self.model = simulator.mjModel
        self.data = simulator.mjData
        self.robot = simulator.task.robot
        if tuple(self.robot.joint_names) != JOINT_NAMES:
            raise ValueError("Unexpected G1 joint order")

        self.pelvis_id = self.model.body("pelvis").id
        self.ankle_ids = (
            self.model.body("left_ankle_roll_link").id,
            self.model.body("right_ankle_roll_link").id,
        )
        self.table_id = self.model.body("table").id
        self.target_id = simulator.mj_objects["target"].id
        self.contact_body_ids = tuple(
            self.model.body(name).id for name in RIGHT_CONTACT_LINK_NAMES
        )
        self.contact_index_by_body = {
            body_id: i for i, body_id in enumerate(self.contact_body_ids)
        }
        self.distal_ids = tuple(
            self.model.body(name).id for name in RIGHT_DISTAL_LINK_NAMES
        )
        self.wrist_id = self.model.body("right_wrist_yaw_link").id
        self.previous_action = (
            np.zeros(36, dtype=np.float32)
            if previous_action is None
            else np.asarray(previous_action, dtype=np.float32).copy()
        )
        self.initial_object_pos = self.data.body(self.target_id).xpos.copy()
        self.goal_pos = self.initial_object_pos + np.array([0.0, 0.0, 0.025])

    def set_previous_action(self, action: np.ndarray) -> None:
        self.previous_action = np.asarray(action, dtype=np.float32).copy()

    def _contacts(self, pelvis_rot: np.ndarray) -> ContactState:
        force_world = np.zeros((8, 3), dtype=np.float64)
        object_contact_positions: list[np.ndarray] = []
        hand_table = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body_1 = int(self.model.geom_bodyid[contact.geom1])
            body_2 = int(self.model.geom_bodyid[contact.geom2])
            local_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, local_force)
            world = np.asarray(contact.frame).reshape(3, 3).T @ local_force[:3]

            if self.target_id in (body_1, body_2):
                hand_body = body_2 if body_1 == self.target_id else body_1
                link_index = self.contact_index_by_body.get(hand_body)
                if link_index is not None:
                    # Keep a consistent "force on hand" sign. Reward uses magnitude.
                    force_world[link_index] += world if hand_body == body_1 else -world
                    object_contact_positions.append(np.asarray(contact.pos).copy())

            if self.table_id in (body_1, body_2):
                hand_body = body_2 if body_1 == self.table_id else body_1
                if hand_body in self.contact_index_by_body:
                    hand_table += float(np.linalg.norm(local_force[:3]))

        force_pelvis = force_world @ pelvis_rot
        magnitude = np.linalg.norm(force_world, axis=-1)
        group_forces = np.array(
            [
                magnitude[1:4].max(initial=0.0),
                magnitude[4:6].max(initial=0.0),
                magnitude[6:8].max(initial=0.0),
            ],
            dtype=np.float32,
        )
        has_contact_center = bool(object_contact_positions)
        contact_center = (
            np.mean(object_contact_positions, axis=0)
            if has_contact_center
            else np.zeros(3, dtype=np.float64)
        )
        return ContactState(
            link_forces_pelvis=np.clip(force_pelvis, -100.0, 100.0).astype(np.float32),
            link_force_magnitudes=magnitude.astype(np.float32),
            group_forces=group_forces,
            object_contact_center_w=np.asarray(contact_center, dtype=np.float32),
            has_object_contact_center=has_contact_center,
            hand_table_force=hand_table,
        )

    def target_state(self) -> TargetState:
        pelvis = self.data.body(self.pelvis_id)
        obj = self.data.body(self.target_id)
        table = self.data.body(self.table_id)
        pelvis_rot = np.asarray(pelvis.xmat).reshape(3, 3).copy()
        object_rot = np.asarray(obj.xmat).reshape(3, 3).copy()
        table_rot = np.asarray(table.xmat).reshape(3, 3).copy()
        wrist_rot = np.asarray(self.data.body(self.wrist_id).xmat).reshape(3, 3).copy()

        obj_lin, obj_ang = _body_velocity(self.model, self.data, self.target_id)
        wrist_pos = self.data.body(self.wrist_id).xpos.copy()
        wrist_lin, _ = _body_velocity(self.model, self.data, self.wrist_id)
        distal_pos = np.stack([self.data.body(body_id).xpos.copy() for body_id in self.distal_ids])
        distal_vel = np.stack([_body_velocity(self.model, self.data, body_id)[0] for body_id in self.distal_ids])
        ankle_height = float(
            np.mean([self.data.body(body_id).xpos[2] for body_id in self.ankle_ids])
        )
        return TargetState(
            object_pos_w=obj.xpos.copy(),
            object_rot_w=object_rot,
            object_lin_vel_w=obj_lin,
            object_ang_vel_w=obj_ang,
            hand_pos_w=distal_pos.mean(axis=0),
            hand_rot_w=wrist_rot,
            hand_lin_vel_w=distal_vel.mean(axis=0),
            wrist_pos_w=wrist_pos,
            wrist_lin_vel_w=wrist_lin,
            distal_pos_w=distal_pos,
            table_pos_w=table.xpos.copy(),
            table_rot_w=table_rot,
            pelvis_pos_w=pelvis.xpos.copy(),
            pelvis_rot_w=pelvis_rot,
            pelvis_height=float(pelvis.xpos[2] - ankle_height),
            contact=self._contacts(pelvis_rot),
        )

    def actor_observation(self) -> tuple[np.ndarray, TargetState]:
        state = self.target_state()
        rotation_inv = state.pelvis_rot_w.T
        joint_pos = np.fromiter(
            (self.data.joint(name).qpos.item() for name in JOINT_NAMES),
            dtype=np.float32,
            count=43,
        )
        joint_vel = np.fromiter(
            (self.data.joint(name).qvel.item() for name in JOINT_NAMES),
            dtype=np.float32,
            count=43,
        )
        pelvis_lin, pelvis_ang = _body_velocity(self.model, self.data, self.pelvis_id)
        gravity = rotation_inv @ np.array([0.0, 0.0, -1.0])

        object_pos_b = rotation_inv @ (state.object_pos_w - state.pelvis_pos_w)
        object_rot_b = rotation_inv @ state.object_rot_w
        object_lin_b = rotation_inv @ state.object_lin_vel_w
        object_ang_b = rotation_inv @ state.object_ang_vel_w

        hand_inv = state.hand_rot_w.T
        object_pos_hand = hand_inv @ (state.object_pos_w - state.hand_pos_w)
        object_rot_hand = hand_inv @ state.object_rot_w
        table_pos_b = rotation_inv @ (state.table_pos_w - state.pelvis_pos_w)
        table_rot_b = rotation_inv @ state.table_rot_w
        goal_delta_b = rotation_inv @ (self.goal_pos - state.object_pos_w)

        observation = np.concatenate(
            [
                joint_pos,
                joint_vel,
                gravity,
                rotation_inv @ pelvis_lin,
                rotation_inv @ pelvis_ang,
                np.array([state.pelvis_height], dtype=np.float32),
                self.previous_action,
                object_pos_b,
                _rotation_6d(object_rot_b),
                object_lin_b,
                object_ang_b,
                object_pos_hand,
                _rotation_6d(object_rot_hand),
                table_pos_b,
                _rotation_6d(table_rot_b),
                state.contact.link_forces_pelvis.reshape(-1),
                goal_delta_b,
            ]
        ).astype(np.float32)
        if observation.shape != (ACTOR_OBS_DIM,):
            raise RuntimeError(f"Actor observation has shape {observation.shape}")
        return observation, state
