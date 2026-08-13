from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/grasp_rl/grasp_anything_objects.py"
SPEC = importlib.util.spec_from_file_location("grasp_anything_objects", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
OBJECT_RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OBJECT_RUNNER
SPEC.loader.exec_module(OBJECT_RUNNER)


def test_catalog_covers_distinct_grasp_geometries() -> None:
    assert set(OBJECT_RUNNER.OBJECTS) == {
        "Apple_1",
        "Bottle_1",
        "Bowl_1",
        "Soap_Bottle_1",
    }
    assert {spec.category for spec in OBJECT_RUNNER.OBJECTS.values()} == {
        "bottle",
        "bowl",
        "round",
    }
    for spec in OBJECT_RUNNER.OBJECTS.values():
        spec.validate()


def test_object_outputs_are_isolated_from_cup_and_each_other() -> None:
    paths = {
        OBJECT_RUNNER.stage_output(object_id, "narrow")
        for object_id in OBJECT_RUNNER.OBJECTS
    }
    assert len(paths) == len(OBJECT_RUNNER.OBJECTS)
    assert all("/Cup_6/" not in str(path) for path in paths)
    assert all(path.name == "narrow" for path in paths)


def test_workspace_starts_inside_original_edge_position() -> None:
    narrow = OBJECT_RUNNER.STAGES["narrow"]
    workspace = OBJECT_RUNNER.STAGES["workspace"]
    assert narrow.center_xy_m[0] - narrow.jitter_xy_m[0] >= 0.08
    assert workspace.center_xy_m[0] - workspace.jitter_xy_m[0] >= 0.08
    assert workspace.jitter_xy_m[1] > narrow.jitter_xy_m[1]


def test_environment_uses_single_reference_and_opt_in_pose_retargeting() -> None:
    arguments = OBJECT_RUNNER._environment_args(
        OBJECT_RUNNER.OBJECTS["Apple_1"], "narrow", 123
    )
    assert arguments[arguments.index("--strict-reference-episode") + 1] == "82"
    assert arguments[arguments.index("--dr-profile") + 1] == "pose_only"
    assert arguments[
        arguments.index("--reference-target-positive-y-arm-gains") + 1 :
        arguments.index("--reference-target-positive-y-arm-gains") + 3
    ] == ["22", "0"]


def test_existing_run_is_never_overwritten(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "model_0.pt").write_bytes(b"owned")
    warm_start = tmp_path / "warm.pt"
    warm_start.write_bytes(b"checkpoint")
    monkeypatch.setattr(OBJECT_RUNNER, "stage_output", lambda *_: output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        OBJECT_RUNNER.train(
            OBJECT_RUNNER.OBJECTS["Apple_1"],
            stage="narrow",
            gpu=0,
            seed=1,
            num_envs=1,
            iterations=1,
            warm_start=warm_start,
        )


def test_narrow_scratch_training_does_not_require_cross_object_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "scratch"
    captured = {}
    monkeypatch.setattr(OBJECT_RUNNER, "stage_output", lambda *_: output)
    monkeypatch.setattr(
        OBJECT_RUNNER,
        "_run_gpu",
        lambda arguments, **kwargs: captured.update(arguments=arguments, **kwargs),
    )

    checkpoint = OBJECT_RUNNER.train(
        OBJECT_RUNNER.OBJECTS["Apple_1"],
        stage="narrow",
        gpu=0,
        seed=1,
        num_envs=64,
        iterations=1,
        warm_start=None,
    )

    assert "--plan-conditioned-actor" in captured["arguments"]
    assert "--warm-start" not in captured["arguments"]
    assert checkpoint == output / "model_0.pt"


def test_gpu_evaluation_log_extracts_structured_result(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "evaluation.json"
    output = 'runtime banner\n{"result":{"success_rate":0.5}}\ntrailer\n'

    class Completed:
        returncode = 0
        stdout = output

    monkeypatch.setattr(OBJECT_RUNNER.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(
        OBJECT_RUNNER,
        "_gpu_cli",
        lambda gpu: (["python", "gpu_cli.py"], {}),
    )

    OBJECT_RUNNER._run_gpu(["evaluate"], gpu=0, log=log)

    assert OBJECT_RUNNER.json.loads(log.read_text()) == {
        "result": {"success_rate": 0.5}
    }
