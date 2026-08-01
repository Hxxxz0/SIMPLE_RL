"""Role-based privileged MuJoCo state for the multi-task v2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_V2_DIM,
    JOINT_NAMES,
    LEFT_CONTACT_LINK_NAMES,
    LEFT_DISTAL_LINK_NAMES,
    RIGHT_CONTACT_LINK_NAMES,
    RIGHT_DISTAL_LINK_NAMES,
)
from simple.grasp_rl.state import _body_velocity, _rotation_6d
from simple.grasp_rl.task_spec import TaskSpecV2


V2_SLICES = {
    "joint_pos": slice(0, 43),
    "joint_vel": slice(43, 86),
    "root": slice(86, 96),
    "previous_action": slice(96, 132),
    "hands": slice(132, 162),
    "entities": slice(162, 219),
    "contact_forces": slice(219, 267),
    "fingertip_distances": slice(267, 273),
    "primary_in_hands": slice(273, 291),
    "primary_to_destination": slice(291, 300),
    "predicates": slice(300, 308),
    "articulation": slice(308, 316),
    "task_context": slice(316, 331),
}

FAMILIES = ("grasp", "place", "handover", "articulation", "push", "compound")


@dataclass(frozen=True)
class EntityState:
    present: bool
    body_id: int | None
    pos_w: np.ndarray
    rot_w: np.ndarray
    lin_vel_w: np.ndarray
    ang_vel_w: np.ndarray
    extents: np.ndarray


@dataclass(frozen=True)
class HandStateV2:
    pos_w: np.ndarray
    rot_w: np.ndarray
    lin_vel_w: np.ndarray
    ang_vel_w: np.ndarray
    primary_contact: bool
    auxiliary_contact: bool


@dataclass(frozen=True)
class TaskStateV2:
    pelvis_pos_w: np.ndarray
    pelvis_rot_w: np.ndarray
    pelvis_height: float
    hands: tuple[HandStateV2, HandStateV2]
    primary: EntityState
    destination: EntityState
    auxiliary: EntityState
    contact_forces_pelvis: np.ndarray
    distal_pos_w: np.ndarray
    primary_contact_centers_w: np.ndarray
    has_primary_contact_center: np.ndarray
    fingertip_distances: np.ndarray
    predicates: np.ndarray
    articulation: np.ndarray
    articulation_raw: tuple[float, ...]
    initial_primary_pos: np.ndarray
    initial_auxiliary_pos: np.ndarray

    @property
    def left_primary_contact(self) -> bool:
        return bool(self.predicates[0])

    @property
    def right_primary_contact(self) -> bool:
        return bool(self.predicates[1])

    @property
    def primary_destination_contact(self) -> bool:
        return bool(self.predicates[4])


class MujocoTaskStateExtractor:
    """Extract the fixed 331-D observation from task-role declarations."""

    def __init__(self, simulator, spec: TaskSpecV2, previous_action: np.ndarray | None = None):
        self.sim = simulator
        self.model: mujoco.MjModel = simulator.mjModel
        self.data: mujoco.MjData = simulator.mjData
        self.spec = spec
        self.robot = simulator.task.robot
        if tuple(self.robot.joint_names) != JOINT_NAMES:
            raise ValueError("Unexpected G1 joint order for task_privileged_v2")
        self.pelvis_id = self.model.body("pelvis").id
        self.ankle_ids = tuple(self.model.body(name).id for name in
                               ("left_ankle_roll_link", "right_ankle_roll_link"))
        self.hand_body_ids = (
            self.model.body("left_wrist_yaw_link").id,
            self.model.body("right_wrist_yaw_link").id,
        )
        self.hand_link_ids = (
            tuple(self.model.body(name).id for name in LEFT_CONTACT_LINK_NAMES),
            tuple(self.model.body(name).id for name in RIGHT_CONTACT_LINK_NAMES),
        )
        self.distal_ids = (
            tuple(self.model.body(name).id for name in LEFT_DISTAL_LINK_NAMES),
            tuple(self.model.body(name).id for name in RIGHT_DISTAL_LINK_NAMES),
        )
        self.role_body_ids = {
            "primary": self._resolve_role(spec.roles.primary),
            "destination": self._resolve_role(spec.roles.destination),
            "auxiliary": self._resolve_role(spec.roles.auxiliary),
            "support": self._resolve_role("table"),
        }
        self.role_body_sets = {
            key: self._subtree_body_ids(value) if value is not None else set()
            for key, value in self.role_body_ids.items()
        }
        self.role_geom_ids = {
            key: self._subtree_geom_ids(value) if value is not None else tuple()
            for key, value in self.role_body_ids.items()
        }
        self.hand_body_sets = tuple(set(ids) for ids in self.hand_link_ids)
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.set_previous_action(previous_action if previous_action is not None else self.previous_action)
        self.stage_index = 0
        self.stage_progress = 0.0
        primary = self._entity("primary")
        auxiliary = self._entity("auxiliary")
        self.initial_primary_pos = primary.pos_w.copy()
        self.initial_auxiliary_pos = auxiliary.pos_w.copy()

    def _resolve_role(self, role: str | None) -> int | None:
        if role is None:
            return None
        obj = getattr(self.sim, "mj_objects", {}).get(role)
        if obj is not None:
            return int(obj.id)
        try:
            return int(self.model.body(role).id)
        except KeyError:
            return None

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        current = int(body_id)
        while current > 0 and current != ancestor_id:
            current = int(self.model.body_parentid[current])
        return current == ancestor_id

    def _subtree_body_ids(self, body_id: int) -> set[int]:
        return {i for i in range(self.model.nbody) if self._is_descendant(i, body_id)}

    def _subtree_geom_ids(self, body_id: int) -> tuple[int, ...]:
        bodies = self._subtree_body_ids(body_id)
        return tuple(i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) in bodies)

    def _entity(self, role: str) -> EntityState:
        body_id = self.role_body_ids[role]
        if body_id is None:
            z3 = np.zeros(3, dtype=np.float64)
            return EntityState(False, None, z3.copy(), np.eye(3), z3.copy(), z3.copy(), z3.copy())
        body = self.data.body(body_id)
        lin, ang = _body_velocity(self.model, self.data, body_id)
        geoms = self.role_geom_ids[role]
        radius = max((float(self.model.geom_rbound[g]) for g in geoms), default=0.0)
        return EntityState(True, body_id, body.xpos.copy(), body.xmat.reshape(3, 3).copy(),
                           lin, ang, np.full(3, radius, dtype=np.float64))

    def _pair_contact(self, first: set[int], second: set[int]) -> bool:
        if not first or not second:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if (b1 in first and b2 in second) or (b2 in first and b1 in second):
                return True
        return False

    def _hand_forces(self, pelvis_rot: np.ndarray) -> np.ndarray:
        primary = self.role_body_sets["primary"]
        result = np.zeros((2, 8, 3), dtype=np.float64)
        maps = tuple({body: i for i, body in enumerate(ids)} for ids in self.hand_link_ids)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            if b1 not in primary and b2 not in primary:
                continue
            hand_body = b2 if b1 in primary else b1
            local = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, local)
            world = np.asarray(contact.frame).reshape(3, 3).T @ local[:3]
            for hand_index, mapping in enumerate(maps):
                link_index = mapping.get(hand_body)
                if link_index is not None:
                    result[hand_index, link_index] += world if hand_body == b1 else -world
        return result @ pelvis_rot

    def _hand_contact_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-hand primary contact centres from MuJoCo contact points."""

        primary = self.role_body_sets["primary"]
        points: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = int(self.model.geom_bodyid[contact.geom1])
            second = int(self.model.geom_bodyid[contact.geom2])
            for hand_index, hand_bodies in enumerate(self.hand_body_sets):
                if (
                    (first in primary and second in hand_bodies)
                    or (second in primary and first in hand_bodies)
                ):
                    points[hand_index].append(np.asarray(contact.pos).copy())
        centres = np.zeros((2, 3), dtype=np.float64)
        valid = np.zeros(2, dtype=bool)
        for hand_index, hand_points in enumerate(points):
            if hand_points:
                centres[hand_index] = np.mean(hand_points, axis=0)
                valid[hand_index] = True
        return centres, valid

    def _surface_distances(self, entity: EntityState) -> np.ndarray:
        if not entity.present:
            return np.ones(6, dtype=np.float32)
        target_geoms = self.role_geom_ids["primary"]
        values: list[float] = []
        for hand in self.distal_ids:
            for body_id in hand:
                finger_geoms = self._subtree_geom_ids(body_id)
                candidates = [float(mujoco.mj_geomDistance(self.model, self.data, fg, tg, 2.0, None))
                              for fg in finger_geoms for tg in target_geoms]
                values.append(max(min(candidates), 0.0) if candidates else 1.0)
        return np.asarray(values, dtype=np.float32)

    def _articulation(self) -> tuple[np.ndarray, tuple[float, ...]]:
        encoded = np.zeros(8, dtype=np.float32)
        raw: list[float] = []
        goal = self.spec.articulation
        if goal is None:
            return encoded, tuple(raw)
        for slot, (name, target) in enumerate(zip(goal.joints[:2], goal.targets[:2], strict=True)):
            try:
                joint = self.data.joint(name)
                joint_id = self.model.joint(name).id
            except KeyError:
                continue
            q, qd = float(joint.qpos[0]), float(joint.qvel[0])
            raw.append(q)
            limited = bool(self.model.jnt_limited[joint_id])
            low, high = self.model.jnt_range[joint_id] if limited else (-max(abs(target), 1.0), max(abs(target), 1.0))
            span = max(float(high - low), 1e-3)
            offset = 4 * slot
            encoded[offset] = 1.0
            encoded[offset + 1] = np.clip(2.0 * (q - low) / span - 1.0, -2.0, 2.0)
            encoded[offset + 2] = np.clip(qd / 5.0, -2.0, 2.0)
            encoded[offset + 3] = np.clip((target - q) / span, -2.0, 2.0)
        return encoded, tuple(raw)

    def set_previous_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected previous action ({ACTION_DIM},), got {action.shape}")
        self.previous_action = action.copy()

    def set_stage(self, index: int, progress: float) -> None:
        self.stage_index = int(np.clip(index, 0, 7))
        self.stage_progress = float(np.clip(progress, 0.0, 1.0))

    def actor_observation(self) -> tuple[np.ndarray, TaskStateV2]:
        pelvis = self.data.body(self.pelvis_id)
        pelvis_rot = pelvis.xmat.reshape(3, 3).copy()
        inv = pelvis_rot.T
        pelvis_lin, pelvis_ang = _body_velocity(self.model, self.data, self.pelvis_id)
        ankle_height = float(np.mean([self.data.body(i).xpos[2] for i in self.ankle_ids]))
        primary, destination, auxiliary = (self._entity(name) for name in ("primary", "destination", "auxiliary"))
        contact_forces = self._hand_forces(pelvis_rot)
        contact_centres, has_contact_centre = self._hand_contact_centers()
        distances = self._surface_distances(primary)
        distal_pos = np.asarray(
            [
                [self.data.body(body_id).xpos.copy() for body_id in hand]
                for hand in self.distal_ids
            ],
            dtype=np.float64,
        )

        hands: list[HandStateV2] = []
        for index, body_id in enumerate(self.hand_body_ids):
            body = self.data.body(body_id)
            lin, ang = _body_velocity(self.model, self.data, body_id)
            hands.append(HandStateV2(body.xpos.copy(), body.xmat.reshape(3, 3).copy(), lin, ang,
                                     self._pair_contact(self.hand_body_sets[index], self.role_body_sets["primary"]),
                                     self._pair_contact(self.hand_body_sets[index], self.role_body_sets["auxiliary"])))
        predicates = np.asarray([
            hands[0].primary_contact, hands[1].primary_contact,
            hands[0].auxiliary_contact, hands[1].auxiliary_contact,
            self._pair_contact(self.role_body_sets["primary"], self.role_body_sets["destination"]),
            self._pair_contact(self.role_body_sets["primary"], self.role_body_sets["auxiliary"]),
            self._pair_contact(self.role_body_sets["auxiliary"], self.role_body_sets["destination"]),
            any(self._pair_contact(hand, self.role_body_sets["support"]) for hand in self.hand_body_sets),
        ], dtype=np.float32)
        articulation, articulation_raw = self._articulation()
        state = TaskStateV2(pelvis.xpos.copy(), pelvis_rot, float(pelvis.xpos[2] - ankle_height),
                            (hands[0], hands[1]), primary, destination, auxiliary,
                            contact_forces, distal_pos, contact_centres,
                            has_contact_centre, distances, predicates,
                            articulation, articulation_raw,
                            self.initial_primary_pos.copy(), self.initial_auxiliary_pos.copy())

        joint_pos = np.fromiter((self.data.joint(n).qpos.item() for n in JOINT_NAMES), dtype=np.float32, count=43)
        joint_vel = np.fromiter((self.data.joint(n).qvel.item() for n in JOINT_NAMES), dtype=np.float32, count=43)
        root = np.concatenate((inv @ np.array([0., 0., -1.]), inv @ pelvis_lin, inv @ pelvis_ang,
                               [state.pelvis_height])).astype(np.float32)
        hand_features = []
        for hand in hands:
            hand_features.extend((inv @ (hand.pos_w - pelvis.xpos), _rotation_6d(inv @ hand.rot_w),
                                  inv @ hand.lin_vel_w, inv @ hand.ang_vel_w))
        entity_features = []
        for entity in (primary, destination, auxiliary):
            if entity.present:
                entity_features.extend(([1.0], inv @ (entity.pos_w - pelvis.xpos),
                                        _rotation_6d(inv @ entity.rot_w), inv @ entity.lin_vel_w,
                                        inv @ entity.ang_vel_w, entity.extents))
            else:
                entity_features.append(np.zeros(19, dtype=np.float32))
        primary_in_hands = []
        for hand in hands:
            if primary.present:
                hand_inv = hand.rot_w.T
                primary_in_hands.extend((hand_inv @ (primary.pos_w - hand.pos_w),
                                         _rotation_6d(hand_inv @ primary.rot_w)))
            else:
                primary_in_hands.append(np.zeros(9, dtype=np.float32))
        if primary.present and destination.present:
            primary_to_destination = np.concatenate((inv @ (destination.pos_w - primary.pos_w),
                                                      _rotation_6d(inv @ destination.rot_w)))
        else:
            primary_to_destination = np.zeros(9, dtype=np.float32)
        family = np.zeros(6, dtype=np.float32)
        family[FAMILIES.index(self.spec.family)] = 1.0
        stage = np.zeros(8, dtype=np.float32)
        stage[self.stage_index] = 1.0
        context = np.concatenate((family, stage, [self.stage_progress]))
        observation = np.concatenate((joint_pos, joint_vel, root, self.previous_action,
                                      *hand_features, *entity_features,
                                      contact_forces.reshape(-1), distances,
                                      *primary_in_hands, primary_to_destination, predicates,
                                      articulation, context)).astype(np.float32)
        if observation.shape != (ACTOR_OBS_V2_DIM,):
            raise RuntimeError(f"V2 actor observation has shape {observation.shape}, expected {(ACTOR_OBS_V2_DIM,)}")
        return observation, state
