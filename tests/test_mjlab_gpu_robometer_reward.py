import json
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch

import simple.grasp_rl.mjlab_gpu.cli as mjlab_gpu_cli
from simple.grasp_rl.mjlab_gpu.cli import (
    _parser,
    _robometer_config,
    _train,
    _validate_robometer_run,
)
from simple.grasp_rl.mjlab_gpu.config import MjlabPpoConfig
from simple.grasp_rl.mjlab_gpu.robometer_reward import (
    ROBOMETER_INSTRUCTION,
    ROBOMETER_REPLACE_TASKS,
    ROBOMETER_REWARD_SCHEMA_VERSION,
    ROBOMETER_TASK_INSTRUCTIONS,
    RobometerBatchResult,
    RobometerHttpClient,
    RobometerTaskReward,
    RobometerTaskRewardConfig,
)
from simple.grasp_rl.mjlab_gpu.runner import GpuPpoRunner


class _FakeRenderer:
    def __init__(self) -> None:
        self.reset_calls: list[list[int]] = []
        self.render_calls: list[list[int]] = []
        self.closed = False

    @staticmethod
    def _frames(env_ids: list[int]) -> list[np.ndarray]:
        return [np.full((2, 3, 3), env_id, dtype=np.uint8) for env_id in env_ids]

    def reset(self, env_ids: list[int], qpos: np.ndarray) -> list[np.ndarray]:
        assert len(env_ids) == len(qpos)
        self.reset_calls.append(list(env_ids))
        return self._frames(env_ids)

    def render(self, env_ids: list[int], qpos: np.ndarray) -> list[np.ndarray]:
        assert len(env_ids) == len(qpos)
        self.render_calls.append(list(env_ids))
        return self._frames(env_ids)

    def close(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, results: list[RobometerBatchResult]) -> None:
        self.results = results
        self.calls: list[tuple[list[np.ndarray], str, list[str]]] = []

    def evaluate(
        self,
        frame_batches: list[np.ndarray],
        instruction: str,
        sample_ids: list[str],
    ) -> RobometerBatchResult:
        self.calls.append((frame_batches, instruction, sample_ids))
        return self.results.pop(0)


def _batch(
    progress: list[float], success: list[float], latency_ms: float = 4.0
) -> RobometerBatchResult:
    return RobometerBatchResult(
        np.asarray(progress, dtype=np.float32),
        np.asarray(success, dtype=np.float32),
        latency_ms,
    )


def test_robometer_config_is_bounded_and_hashed() -> None:
    config = RobometerTaskRewardConfig("http://127.0.0.1:8000")
    metadata = config.metadata()
    assert metadata["schema_version"] == ROBOMETER_REWARD_SCHEMA_VERSION
    assert len(str(metadata["resolved_sha256"])) == 64
    assert metadata == config.metadata()

    invalid = (
        {"server_url": "file:///model"},
        {"server_url": "http://server", "task": "close_door"},
        {"server_url": "http://server", "mode": "invalid"},
        {"server_url": "http://server", "inference_interval_steps": 0},
        {"server_url": "http://server", "progress_scale": float("nan")},
        {"server_url": "http://server", "delta_clip": 0.0},
        {"server_url": "http://server", "max_frames": 1},
        {"server_url": "http://server", "instruction": "  "},
    )
    for values in invalid:
        with pytest.raises(ValueError):
            RobometerTaskRewardConfig(**values)


def test_relative_progress_interval_terminal_clip_and_reset() -> None:
    renderer = _FakeRenderer()
    client = _FakeClient(
        [
            _batch([0.1, 0.8], [0.2, 0.3]),
            _batch([0.7], [0.9]),
            _batch([0.0, 0.7], [0.1, 0.4]),
            _batch([0.9], [0.8]),
        ]
    )
    reward = RobometerTaskReward(
        object(),
        config=RobometerTaskRewardConfig(
            "http://server",
            inference_interval_steps=2,
            progress_scale=2.0,
            max_frames=3,
        ),
        num_envs=2,
        device="cpu",
        client=client,
        renderer=renderer,
    )
    qpos = torch.zeros(2, 4)
    ids = torch.arange(2)
    reward.reset(ids, qpos)

    torch.testing.assert_close(
        reward.compute(
            qpos=qpos, terminal=torch.zeros(2, dtype=torch.bool), vector_step=0
        ),
        torch.zeros(2),
    )
    assert not client.calls

    first = reward.compute(
        qpos=qpos, terminal=torch.zeros(2, dtype=torch.bool), vector_step=1
    )
    torch.testing.assert_close(first, torch.zeros(2))
    assert client.calls[0][0][0].shape == (2, 2, 3, 3)

    terminal = reward.compute(
        qpos=qpos, terminal=torch.tensor([True, False]), vector_step=2
    )
    torch.testing.assert_close(terminal, torch.tensor([0.5, 0.0]))
    snapshot = reward.snapshot()
    torch.testing.assert_close(snapshot["progress_delta"], torch.tensor([0.25, 0.0]))
    assert renderer.render_calls[-1] == [0]

    scheduled = reward.compute(
        qpos=qpos, terminal=torch.zeros(2, dtype=torch.bool), vector_step=3
    )
    torch.testing.assert_close(scheduled, torch.tensor([-0.5, -0.2]))
    assert client.calls[-1][0][0].shape[0] == 3

    reward.reset(torch.tensor([0]), qpos)
    after_reset = reward.compute(
        qpos=qpos, terminal=torch.tensor([True, False]), vector_step=4
    )
    torch.testing.assert_close(after_reset, torch.zeros(2))
    assert reward.previous_progress[0] == pytest.approx(0.9)
    reward.close()
    assert renderer.closed


def test_robometer_rejects_incorrect_client_shapes() -> None:
    reward = RobometerTaskReward(
        object(),
        config=RobometerTaskRewardConfig("http://server"),
        num_envs=1,
        device="cpu",
        client=_FakeClient([_batch([], [])]),
        renderer=_FakeRenderer(),
    )
    qpos = torch.zeros(1, 2)
    reward.reset(torch.tensor([0]), qpos)
    with pytest.raises(RuntimeError, match="incorrectly shaped"):
        reward.compute(qpos=qpos, terminal=torch.tensor([True]), vector_step=0)


class _Response:
    def __init__(self, payload: dict[str, object], error: Exception | None = None):
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> dict[str, object]:
        return self.payload


def test_http_client_sends_npy_and_reads_last_prediction(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def post(url: str, **kwargs: object) -> _Response:
        captured.update(url=url, **kwargs)
        return _Response(
            {
                "outputs_progress": {"progress_pred": [[0.1, 0.6]]},
                "outputs_success": {"success_probs": [[0.2, 0.75]]},
            }
        )

    monkeypatch.setattr(requests, "post", post)
    frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    result = RobometerHttpClient("http://server/").evaluate(
        [frames], "instruction", ["sample"]
    )

    assert captured["url"] == "http://server/evaluate_batch_npy"
    assert "sample_0_trajectory_frames" in captured["files"]
    sample = json.loads(captured["data"]["sample_0"])
    assert sample["trajectory"]["frames_shape"] == [2, 4, 5, 3]
    np.testing.assert_allclose(result.progress, [0.6])
    np.testing.assert_allclose(result.success_probability, [0.75])


def test_http_client_groups_heterogeneous_trajectory_lengths(monkeypatch) -> None:
    requests_seen: list[list[str]] = []

    def post(url: str, **kwargs: object) -> _Response:
        samples = [
            json.loads(value)
            for key, value in sorted(kwargs["data"].items())
            if key.startswith("sample_")
        ]
        requests_seen.append([sample["trajectory"]["id"] for sample in samples])
        outputs = [[float(sample["trajectory"]["id"])] for sample in samples]
        return _Response(
            {
                "outputs_progress": {"progress_pred": outputs},
                "outputs_success": {"success_probs": outputs},
            }
        )

    monkeypatch.setattr(requests, "post", post)
    batches = [
        np.zeros((2, 4, 5, 3), dtype=np.uint8),
        np.zeros((8, 4, 5, 3), dtype=np.uint8),
        np.zeros((2, 4, 5, 3), dtype=np.uint8),
    ]
    result = RobometerHttpClient("http://server").evaluate(
        batches, "task", ["0", "1", "2"]
    )

    assert requests_seen == [["0", "2"], ["1"]]
    np.testing.assert_allclose(result.progress, [0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    "payload, match",
    [
        (
            {
                "outputs_progress": {"progress_pred": [[0.2]]},
                "outputs_success": {"success_probs": []},
            },
            "sample count",
        ),
        (
            {
                "outputs_progress": {"progress_pred": [[float("nan")]]},
                "outputs_success": {"success_probs": [[0.2]]},
            },
            "non-finite",
        ),
    ],
)
def test_http_client_rejects_invalid_response(
    monkeypatch, payload: dict[str, object], match: str
) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _Response(payload))
    frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match=match):
        RobometerHttpClient("http://server").evaluate([frames], "task", ["0"])


def test_http_client_propagates_http_error(monkeypatch) -> None:
    error = requests.HTTPError("server failed")
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _Response({}, error),
    )
    frames = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    with pytest.raises(requests.HTTPError, match="server failed"):
        RobometerHttpClient("http://server").evaluate([frames], "task", ["0"])


def _train_args(*extra: str) -> list[str]:
    return [
        "train",
        "--task",
        "bend_pick",
        "--asset-bundle",
        "assets",
        "--reference-processed",
        "reference",
        "--output",
        "output",
        *extra,
    ]


def test_cli_defaults_to_legacy_physical_reward() -> None:
    args = _parser().parse_args(_train_args())
    assert args.task_reward_source == "physical"
    assert _robometer_config(args) is None


def test_cli_robometer_requires_url_smoke_and_env_limit_but_allows_scratch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    missing_url = _parser().parse_args(_train_args("--task-reward-source", "robometer"))
    with pytest.raises(ValueError, match="server-url"):
        _robometer_config(missing_url)

    no_smoke = _parser().parse_args(
        _train_args(
            "--task-reward-source",
            "robometer",
            "--robometer-server-url",
            "http://server",
            "--warm-start",
            "model.pt",
        )
    )
    with pytest.raises(ValueError, match="smoke"):
        _train(no_smoke)

    too_many = _parser().parse_args(
        _train_args(
            "--task-reward-source",
            "robometer",
            "--robometer-server-url",
            "http://server",
            "--smoke",
            "--num-envs",
            "9",
            "--warm-start",
            "model.pt",
        )
    )
    with pytest.raises(ValueError, match="eight"):
        _train(too_many)

    no_warm_start = _parser().parse_args(
        _train_args(
            "--task-reward-source",
            "robometer",
            "--robometer-server-url",
            "http://server",
            "--smoke",
            "--num-envs",
            "8",
        )
    )

    class ScratchEnvironmentReached(RuntimeError):
        pass

    def reach_scratch_environment(*args, **kwargs):
        raise ScratchEnvironmentReached

    monkeypatch.setattr(mjlab_gpu_cli, "GpuGraspVecEnv", reach_scratch_environment)
    with pytest.raises(ScratchEnvironmentReached):
        _train(no_warm_start)

    exact_resume = _parser().parse_args(
        _train_args(
            "--task-reward-source",
            "robometer",
            "--robometer-server-url",
            "http://server",
            "--smoke",
            "--num-envs",
            "8",
            "--resume",
            "model.pt",
        )
    )
    with pytest.raises(ValueError, match="exact resume"):
        _train(exact_resume)


def test_cli_robometer_task_allowlist_and_shadow_defaults(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    wrong_task = _parser().parse_args(
        [
            "train",
            "--task",
            "close_door",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "reference",
            "--output",
            "output",
            "--task-reward-source",
            "robometer",
            "--robometer-server-url",
            "http://server",
        ]
    )
    with pytest.raises(ValueError, match="task must be one of"):
        _robometer_config(wrong_task)

    shadow = _parser().parse_args(
        [
            "shadow-reward",
            "--task",
            "bend_pick_and_place",
            "--asset-bundle",
            "assets",
            "--reference-processed",
            "reference",
            "--proposal-only",
            "--output",
            "shadow.json",
            "--robometer-server-url",
            "http://server",
        ]
    )
    assert shadow.smoke
    assert shadow.num_envs == 8
    config = _robometer_config(shadow, mode="shadow")
    assert config.mode == "shadow"
    assert config.instruction == ROBOMETER_INSTRUCTION

    shadow.robometer_instruction = "custom task instruction"
    assert (
        _robometer_config(shadow, mode="shadow").instruction
        == "custom task instruction"
    )

    assert ROBOMETER_REPLACE_TASKS == frozenset(ROBOMETER_TASK_INSTRUCTIONS)
    for task in ROBOMETER_REPLACE_TASKS:
        _validate_robometer_run(
            MjlabPpoConfig(task, "assets", num_envs=8, smoke_mode=True),
            RobometerTaskRewardConfig("http://server", mode="replace", task=task),
        )


@pytest.mark.parametrize("task,instruction", ROBOMETER_TASK_INSTRUCTIONS.items())
def test_robometer_tasks_resolve_distinct_default_instructions(
    task: str, instruction: str
) -> None:
    config = RobometerTaskRewardConfig("http://server", task=task)
    assert config.instruction == instruction
    assert config.metadata()["instruction"] == instruction


def test_default_runner_checkpoint_metadata_has_no_reward_override(tmp_path) -> None:
    env = SimpleNamespace(
        config=MjlabPpoConfig("tabletop_grasp", str(tmp_path)),
        gpu=SimpleNamespace(
            bundle=SimpleNamespace(manifest={"manifest_hash": "asset-hash"})
        ),
        controller=SimpleNamespace(metadata=lambda: {"backend": "amo"}),
        reward=SimpleNamespace(metadata=lambda: {"reward": "physical"}),
        reference=SimpleNamespace(metadata=lambda: {"reference": "bc"}),
        robometer_reward_metadata=lambda: None,
    )
    runner = object.__new__(GpuPpoRunner)
    runner.env = env
    runner.actor_learning_rate_scale = 1.0

    metadata = runner.checkpoint_metadata()
    assert "task_reward_override" not in metadata
    assert metadata == runner.checkpoint_metadata()
