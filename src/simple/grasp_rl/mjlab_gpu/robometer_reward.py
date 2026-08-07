"""Experimental Robometer task-reward override for bounded GPU PPO validation."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import torch

ROBOMETER_REWARD_SCHEMA_VERSION = 1
ROBOMETER_TASK = "bend_pick_and_place"
ROBOMETER_TASK_INSTRUCTIONS = {
    "tabletop_grasp": "Pick up the object from the table.",
    "bend_pick": "Bend down and pick up the object from the table.",
    "bend_pick_teleop": "Bend down and pick up the object from the table.",
    "xmove_bend_pick": (
        "Move to the object, bend down, and pick it up from the table."
    ),
    "xmove_pick": "Move to the object and pick it up from the table.",
    ROBOMETER_TASK: (
        "Bend the robot and pick up the object, then place it on the container."
    ),
    "locomotion_pick_between_tables": (
        "Pick up the object from one table, carry it to the other table, "
        "and place it in the container."
    ),
}
ROBOMETER_TASKS = frozenset(ROBOMETER_TASK_INSTRUCTIONS)
# Replacement remains bounded to the task/instruction pairs exercised by the
# multi-task shadow-reward and PPO A/B validation. Unknown tasks stay rejected.
ROBOMETER_REPLACE_TASKS = ROBOMETER_TASKS
ROBOMETER_INSTRUCTION = ROBOMETER_TASK_INSTRUCTIONS[ROBOMETER_TASK]


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RobometerTaskRewardConfig:
    """Configuration kept outside ``MjlabPpoConfig`` for legacy compatibility."""

    server_url: str
    mode: str = "replace"
    task: str = ROBOMETER_TASK
    instruction: str = ""
    inference_interval_steps: int = 25
    progress_scale: float = 1.0
    delta_clip: float = 0.25
    max_frames: int = 8
    render_width: int = 320
    render_height: int = 180
    request_timeout_s: float = 120.0
    schema_version: int = ROBOMETER_REWARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROBOMETER_REWARD_SCHEMA_VERSION:
            raise ValueError("unsupported Robometer reward schema version")
        if self.mode not in ("replace", "shadow"):
            raise ValueError("Robometer reward mode must be replace or shadow")
        if self.task not in ROBOMETER_TASKS:
            supported = ", ".join(sorted(ROBOMETER_TASKS))
            raise ValueError(f"Robometer reward task must be one of: {supported}")
        if self.instruction == "":
            object.__setattr__(
                self, "instruction", ROBOMETER_TASK_INSTRUCTIONS[self.task]
            )
        elif not self.instruction.strip():
            raise ValueError("Robometer instruction must not be empty")
        if not self.server_url.startswith(("http://", "https://")):
            raise ValueError("Robometer server URL must use HTTP or HTTPS")
        if self.inference_interval_steps < 1:
            raise ValueError("Robometer inference interval must be positive")
        if not np.isfinite(self.progress_scale) or self.progress_scale <= 0.0:
            raise ValueError("Robometer progress scale must be positive and finite")
        if not np.isfinite(self.delta_clip) or self.delta_clip <= 0.0:
            raise ValueError("Robometer delta clip must be positive and finite")
        if self.max_frames < 2:
            raise ValueError("Robometer max_frames must be at least two")
        if self.render_width < 1 or self.render_height < 1:
            raise ValueError("Robometer render dimensions must be positive")
        if self.request_timeout_s <= 0.0:
            raise ValueError("Robometer request timeout must be positive")

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        return {**payload, "resolved_sha256": _canonical_hash(payload)}


@dataclass(frozen=True)
class RobometerBatchResult:
    progress: np.ndarray
    success_probability: np.ndarray
    latency_ms: float


class RobometerClient(Protocol):
    def evaluate(
        self,
        frame_batches: Sequence[np.ndarray],
        instruction: str,
        sample_ids: Sequence[str],
    ) -> RobometerBatchResult: ...


class RobometerHttpClient:
    """Small dependency-light client for Robometer's existing NPY endpoint."""

    def __init__(self, server_url: str, *, timeout_s: float = 120.0):
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    @staticmethod
    def _npy_file(value: np.ndarray, filename: str) -> tuple[str, io.BytesIO, str]:
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        buffer.seek(0)
        return filename, buffer, "application/octet-stream"

    def evaluate(
        self,
        frame_batches: Sequence[np.ndarray],
        instruction: str,
        sample_ids: Sequence[str],
    ) -> RobometerBatchResult:
        if len(frame_batches) != len(sample_ids) or not frame_batches:
            raise ValueError(
                "Robometer request batches and IDs must be non-empty and aligned"
            )
        groups: dict[int, list[int]] = {}
        for index, frames in enumerate(frame_batches):
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError("Robometer frames must be uint8 [T,H,W,3]")
            groups.setdefault(int(frames.shape[0]), []).append(index)

        progress = np.empty(len(frame_batches), dtype=np.float32)
        success = np.empty_like(progress)
        latency_ms = 0.0
        for indices in groups.values():
            result = self._evaluate_aligned(
                [frame_batches[index] for index in indices],
                instruction,
                [sample_ids[index] for index in indices],
            )
            progress[indices] = result.progress
            success[indices] = result.success_probability
            latency_ms += result.latency_ms
        return RobometerBatchResult(progress, success, latency_ms)

    def _evaluate_aligned(
        self,
        frame_batches: Sequence[np.ndarray],
        instruction: str,
        sample_ids: Sequence[str],
    ) -> RobometerBatchResult:
        """Send one request whose trajectories all have the same frame count."""

        try:
            import requests
        except ImportError as exc:  # pragma: no cover - guarded by optional extra
            raise RuntimeError(
                "Robometer reward requires the grasp-rl optional dependencies"
            ) from exc

        files: dict[str, Any] = {}
        data: dict[str, str] = {"use_frame_steps": "false"}
        for index, (frames, sample_id) in enumerate(
            zip(frame_batches, sample_ids, strict=True)
        ):
            file_key = f"sample_{index}_trajectory_frames"
            files[file_key] = self._npy_file(frames, f"{file_key}.npy")
            sample = {
                "sample_type": "progress",
                "trajectory": {
                    "frames": {"__numpy_file__": file_key},
                    "frames_shape": list(frames.shape),
                    "task": instruction,
                    "id": str(sample_id),
                    "metadata": {"subsequence_length": int(frames.shape[0])},
                    "video_embeddings": None,
                },
            }
            data[f"sample_{index}"] = json.dumps(sample)

        started = time.perf_counter()
        response = requests.post(
            self.server_url + "/evaluate_batch_npy",
            files=files,
            data=data,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = 1000.0 * (time.perf_counter() - started)

        progress_outputs = payload.get("outputs_progress", payload)
        progress_rows = progress_outputs.get("progress_pred", [])
        success_rows = (payload.get("outputs_success", {}) or {}).get(
            "success_probs", []
        )
        expected = len(frame_batches)
        if len(progress_rows) != expected or len(success_rows) != expected:
            raise RuntimeError(
                "Robometer response sample count does not match the request"
            )
        progress = np.asarray(
            [float(row[-1]) for row in progress_rows], dtype=np.float32
        )
        success = np.asarray([float(row[-1]) for row in success_rows], dtype=np.float32)
        if not np.isfinite(progress).all() or not np.isfinite(success).all():
            raise RuntimeError("Robometer returned non-finite predictions")
        return RobometerBatchResult(progress, success, latency_ms)


class RobometerRenderer(Protocol):
    def reset(self, env_ids: Sequence[int], qpos: np.ndarray) -> list[np.ndarray]: ...

    def render(self, env_ids: Sequence[int], qpos: np.ndarray) -> list[np.ndarray]: ...

    def close(self) -> None: ...


def _load_render_model(gpu: Any) -> mujoco.MjModel:
    root = Path(gpu.bundle.root)
    manifest_path = root / "render_manifest.json"
    if not manifest_path.is_file():
        return gpu.sim.mj_model
    manifest = json.loads(manifest_path.read_text())
    unhashed = dict(manifest)
    expected_hash = unhashed.pop("manifest_hash", None)
    actual_hash = _canonical_hash(unhashed)
    if expected_hash != actual_hash:
        raise ValueError("Render sidecar manifest hash mismatch")
    if manifest.get("base_manifest_hash") != gpu.bundle.manifest["manifest_hash"]:
        raise ValueError("Render sidecar belongs to a different physics bundle")
    scene = (root / manifest["scene_file"]).resolve()
    if not scene.is_relative_to(root) or _sha256(scene) != manifest["scene_sha256"]:
        raise ValueError("Render sidecar scene hash mismatch")
    for asset in manifest["assets"]:
        asset_path = (root / asset["bundle_path"]).resolve()
        if (
            not asset_path.is_relative_to(root)
            or _sha256(asset_path) != asset["sha256"]
        ):
            raise ValueError("Render sidecar asset hash mismatch")
    return mujoco.MjModel.from_xml_path(str(scene))


class MujocoRobometerRenderer:
    """Sequential offscreen renderer for the bounded eight-environment pilot."""

    def __init__(self, gpu: Any, *, width: int, height: int, num_envs: int):
        self.model = _load_render_model(gpu)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.target_id = self.model.body(gpu.bundle.manifest["roles"]["primary"]).id
        self.pelvis_id = self.model.body("pelvis").id
        self.cameras = [mujoco.MjvCamera() for _ in range(num_envs)]
        for camera in self.cameras:
            mujoco.mjv_defaultFreeCamera(self.model, camera)

    def _set_qpos(self, value: np.ndarray) -> None:
        if value.shape != (self.model.nq,):
            raise ValueError(
                f"Render qpos has shape {value.shape}; expected {(self.model.nq,)}"
            )
        self.data.qpos[:] = value
        mujoco.mj_forward(self.model, self.data)

    def _render_one(self, env_id: int, value: np.ndarray) -> np.ndarray:
        self._set_qpos(value)
        self.renderer.update_scene(self.data, camera=self.cameras[env_id])
        return np.asarray(self.renderer.render(), dtype=np.uint8).copy()

    def reset(self, env_ids: Sequence[int], qpos: np.ndarray) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for row, env_id in enumerate(env_ids):
            self._set_qpos(qpos[row])
            camera = self.cameras[env_id]
            mujoco.mjv_defaultFreeCamera(self.model, camera)
            camera.azimuth = 135.0
            camera.elevation = -14.0
            camera.distance = 2.5
            camera.lookat[:] = (
                0.5 * self.data.xpos[self.target_id]
                + 0.5 * self.data.xpos[self.pelvis_id]
            )
            camera.lookat[2] += 0.05
            frames.append(self._render_one(env_id, qpos[row]))
        return frames

    def render(self, env_ids: Sequence[int], qpos: np.ndarray) -> list[np.ndarray]:
        return [
            self._render_one(env_id, qpos[row]) for row, env_id in enumerate(env_ids)
        ]

    def close(self) -> None:
        self.renderer.close()


class RobometerTaskReward:
    """Tracks video prefixes and returns sparse relative-progress rewards."""

    def __init__(
        self,
        gpu: Any,
        *,
        config: RobometerTaskRewardConfig,
        num_envs: int,
        device: str,
        client: RobometerClient | None = None,
        renderer: RobometerRenderer | None = None,
    ):
        self.config = config
        self.num_envs = int(num_envs)
        self.device = device
        self.client = client or RobometerHttpClient(
            config.server_url, timeout_s=config.request_timeout_s
        )
        self.renderer = renderer or MujocoRobometerRenderer(
            gpu,
            width=config.render_width,
            height=config.render_height,
            num_envs=num_envs,
        )
        self.histories: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        self.previous_progress = np.full(num_envs, np.nan, dtype=np.float32)
        self.last_progress = torch.full(
            (num_envs,), float("nan"), dtype=torch.float32, device=device
        )
        self.last_success_probability = torch.full_like(
            self.last_progress, float("nan")
        )
        self.last_dense_reward = torch.zeros(num_envs, device=device)
        self.last_progress_delta = torch.zeros(num_envs, device=device)
        self.last_inferred = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_latency_ms = 0.0

    @staticmethod
    def _cpu_qpos(qpos: torch.Tensor, env_ids: torch.Tensor) -> np.ndarray:
        return qpos[env_ids].detach().cpu().numpy()

    def reset(self, env_ids: torch.Tensor, qpos: torch.Tensor) -> None:
        ids = [int(value) for value in env_ids.detach().cpu().tolist()]
        frames = self.renderer.reset(ids, self._cpu_qpos(qpos, env_ids))
        for env_id, frame in zip(ids, frames, strict=True):
            self.histories[env_id] = [frame]
            self.previous_progress[env_id] = np.nan
            self.last_progress[env_id] = float("nan")
            self.last_success_probability[env_id] = float("nan")
            self.last_dense_reward[env_id] = 0.0
            self.last_progress_delta[env_id] = 0.0
            self.last_inferred[env_id] = False

    def _sample_history(self, env_id: int) -> np.ndarray:
        history = self.histories[env_id]
        if not history:
            raise RuntimeError("Robometer frame history is empty")
        if len(history) <= self.config.max_frames:
            selected = history
        else:
            indices = np.linspace(
                0, len(history) - 1, self.config.max_frames, dtype=np.int64
            )
            selected = [history[int(index)] for index in indices]
        return np.stack(selected, axis=0)

    def compute(
        self,
        *,
        qpos: torch.Tensor,
        terminal: torch.Tensor,
        vector_step: int,
    ) -> torch.Tensor:
        self.last_dense_reward.zero_()
        self.last_progress_delta.zero_()
        self.last_inferred.zero_()
        scheduled = (int(vector_step) + 1) % self.config.inference_interval_steps == 0
        selected = terminal.clone()
        if scheduled:
            selected.fill_(True)
        env_ids = selected.nonzero(as_tuple=False).flatten()
        if not len(env_ids):
            self.last_latency_ms = 0.0
            return self.last_dense_reward.clone()

        ids = [int(value) for value in env_ids.detach().cpu().tolist()]
        frames = self.renderer.render(ids, self._cpu_qpos(qpos, env_ids))
        for env_id, frame in zip(ids, frames, strict=True):
            self.histories[env_id].append(frame)
        batches = [self._sample_history(env_id) for env_id in ids]
        result = self.client.evaluate(
            batches,
            self.config.instruction,
            [f"env-{env_id}-step-{int(vector_step) + 1}" for env_id in ids],
        )
        if result.progress.shape != (len(ids),) or result.success_probability.shape != (
            len(ids),
        ):
            raise RuntimeError("Robometer client returned incorrectly shaped output")
        deltas = np.zeros(len(ids), dtype=np.float32)
        for row, env_id in enumerate(ids):
            current = float(result.progress[row])
            previous = float(self.previous_progress[env_id])
            if np.isfinite(previous):
                deltas[row] = np.clip(
                    current - previous,
                    -self.config.delta_clip,
                    self.config.delta_clip,
                )
            self.previous_progress[env_id] = current
        dense = self.config.progress_scale * deltas
        dense_tensor = torch.as_tensor(dense, device=self.device)
        progress_tensor = torch.as_tensor(result.progress, device=self.device)
        success_tensor = torch.as_tensor(result.success_probability, device=self.device)
        self.last_dense_reward[env_ids] = dense_tensor
        self.last_progress_delta[env_ids] = torch.as_tensor(deltas, device=self.device)
        self.last_progress[env_ids] = progress_tensor
        self.last_success_probability[env_ids] = success_tensor
        self.last_inferred[env_ids] = True
        self.last_latency_ms = float(result.latency_ms)
        return self.last_dense_reward.clone()

    def snapshot(self) -> dict[str, object]:
        return {
            "dense_reward": self.last_dense_reward.clone(),
            "progress_delta": self.last_progress_delta.clone(),
            "progress": self.last_progress.clone(),
            "success_probability": self.last_success_probability.clone(),
            "inferred": self.last_inferred.clone(),
            "latency_ms": self.last_latency_ms,
        }

    def metadata(self) -> dict[str, object]:
        return self.config.metadata()

    def close(self) -> None:
        self.renderer.close()
