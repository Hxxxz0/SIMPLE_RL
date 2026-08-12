#!/usr/bin/env python3
"""Opt-in adaptive X curriculum for the Cup_6 grasp-anything policy."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
CUP6_SCRIPT = REPO_ROOT / "scripts/grasp_rl/grasp_anything_cup6.sh"
RUN_ROOT = (
    REPO_ROOT
    / "outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Cup_6"
)
DEFAULT_STATE_DIR = RUN_ROOT / "adaptive_x_curriculum_v1"
DEFAULT_INITIAL_RUN = RUN_ROOT / (
    "v_object_reward3_single_ep82_interior_x008_020_y010_seed20260824_"
    "env8192_roll24_frontier_s0185x_g75_std025_lr05_focus70_20"
)
DEFAULT_INITIAL_CHECKPOINT = DEFAULT_INITIAL_RUN / "model_19.pt"
DEFAULT_INITIAL_EVALUATIONS = (
    DEFAULT_INITIAL_RUN
    / "acceptance/tournament_model19_seed20260824_512.json",
    DEFAULT_INITIAL_RUN / "acceptance/seed20260825_frontier_512.json",
    DEFAULT_INITIAL_RUN / "acceptance/seed20260826_frontier_512.json",
)
EVALUATION_CONTRACT_FILES = (
    CUP6_SCRIPT,
    REPO_ROOT
    / "outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/Cup_6_object_reward_v2/manifest.json",
    REPO_ROOT
    / "outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/Cup_6_object_reward_v2/scene.xml",
    REPO_ROOT
    / "data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2_single_ref_ep82_shared_transform/manifest.json",
    REPO_ROOT
    / "data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2_single_ref_ep82_shared_transform/action_transform.npz",
    REPO_ROOT
    / "data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2_single_ref_ep82_shared_transform/episodes/episode_000082.npz",
    *sorted((REPO_ROOT / "src/simple/grasp_rl/mjlab_gpu").glob("*.py")),
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_result(path: Path) -> dict[str, Any]:
    text = path.read_text()
    marker = '{\n  "result":'
    start = text.rfind(marker)
    if start < 0:
        raise ValueError(f"evaluation result not found in {path}")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"evaluation result is invalid in {path}")
    return result


def _gate(
    results: list[dict[str, Any]],
    *,
    minimum_seed_rate: float,
    minimum_pooled_rate: float,
) -> dict[str, Any]:
    if len(results) != 3:
        raise ValueError(f"three evaluation results are required, got {len(results)}")
    seed_rows = []
    total_successes = 0
    total_episodes = 0
    for result in results:
        successes = int(result["successes"])
        episodes = int(result["episodes"])
        rate = float(result["success_rate"])
        if episodes < 1 or successes < 0 or successes > episodes:
            raise ValueError("evaluation result contains invalid episode counts")
        if not math.isclose(rate, successes / episodes, abs_tol=1e-12):
            raise ValueError("evaluation success_rate disagrees with episode counts")
        seed_rows.append(
            {
                "seed": int(result["policy_seed"]),
                "successes": successes,
                "episodes": episodes,
                "success_rate": rate,
                "passed": rate >= minimum_seed_rate,
            }
        )
        total_successes += successes
        total_episodes += episodes
    if len({row["seed"] for row in seed_rows}) != 3:
        raise ValueError("three distinct evaluation seeds are required")
    pooled_rate = total_successes / total_episodes
    return {
        "passed": (
            all(row["passed"] for row in seed_rows)
            and pooled_rate >= minimum_pooled_rate
        ),
        "minimum_seed_rate": minimum_seed_rate,
        "minimum_pooled_rate": minimum_pooled_rate,
        "seeds": seed_rows,
        "pooled_successes": total_successes,
        "pooled_episodes": total_episodes,
        "pooled_success_rate": pooled_rate,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _append_history(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(event, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_lock(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "curriculum.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"adaptive curriculum already holds {lock_path}") from error
        yield


def _initial_gate() -> dict[str, Any]:
    return _gate(
        [_json_result(path) for path in DEFAULT_INITIAL_EVALUATIONS],
        minimum_seed_rate=0.65,
        minimum_pooled_rate=0.70,
    )


def _initial_state(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _canonical(args.initial_checkpoint).resolve()
    if checkpoint == DEFAULT_INITIAL_CHECKPOINT.resolve():
        gate = _initial_gate()
        if not gate["passed"]:
            raise RuntimeError("the default s0185x checkpoint no longer passes its gate")
        evidence = [str(path.resolve()) for path in DEFAULT_INITIAL_EVALUATIONS]
    else:
        if not args.allow_unverified_initial_checkpoint:
            raise ValueError(
                "a custom initial checkpoint requires "
                "--allow-unverified-initial-checkpoint"
            )
        gate = None
        evidence = []
    if not checkpoint.is_file():
        raise FileNotFoundError(f"initial checkpoint does not exist: {checkpoint}")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "status": "ready",
        "next_stage_index": 1,
        "consecutive_successes": 0,
        "active_attempt": None,
        "accepted": {
            "x_min_m": args.x_min,
            "x_upper_m": args.initial_x_upper,
            "checkpoint": str(checkpoint),
            "gate": gate,
            "evidence": evidence,
            "gate_profile": "legacy_initial_envelope",
        },
        "curriculum": {
            "axis": "x",
            "reach_gate_x_upper_m": args.reach_gate,
            "step_m": args.initial_step,
            "minimum_step_m": args.minimum_step,
            "maximum_step_m": args.maximum_step,
            "grow_after_successes": args.grow_after_successes,
            "growth_factor": args.growth_factor,
            "y_jitter_m": args.y_jitter,
            "yaw_jitter_rad": args.yaw_jitter,
            "focus_probability": args.focus_probability,
            "frontier_width_m": args.frontier_width,
            "iterations": args.iterations,
            "training_seed": args.training_seed,
            "selection_seed": args.selection_seed,
            "evaluation_seeds": list(args.evaluation_seeds),
            "acceptance_envs": args.acceptance_envs,
            "minimum_seed_rate": args.minimum_seed_rate,
            "minimum_pooled_rate": args.minimum_pooled_rate,
            "minimum_frontier_seed_rate": args.minimum_frontier_seed_rate,
            "minimum_frontier_pooled_rate": args.minimum_frontier_pooled_rate,
            "minimum_slice_seed_rate": args.minimum_slice_seed_rate,
            "minimum_slice_pooled_rate": args.minimum_slice_pooled_rate,
        },
    }


def _validate_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported state schema: {state.get('schema_version')}")
    curriculum = state["curriculum"]
    accepted = state["accepted"]
    if curriculum["axis"] != "x":
        raise ValueError("this driver only supports the X curriculum")
    if accepted["x_min_m"] >= accepted["x_upper_m"]:
        raise ValueError("accepted X bounds are invalid")
    if curriculum["reach_gate_x_upper_m"] > 0.06 + 1e-12:
        raise ValueError("this driver cannot advance beyond the 0.06 m reach gate")
    if not (
        0 < curriculum["minimum_step_m"]
        <= curriculum["step_m"]
        <= curriculum["maximum_step_m"]
    ):
        raise ValueError("curriculum step bounds are invalid")
    if len(curriculum["evaluation_seeds"]) != 3 or len(
        set(curriculum["evaluation_seeds"])
    ) != 3:
        raise ValueError("exactly three distinct evaluation seeds are required")
    if curriculum["selection_seed"] in curriculum["evaluation_seeds"]:
        raise ValueError("selection seed must be distinct from evaluation seeds")
    if float(curriculum["frontier_width_m"]) <= 0:
        raise ValueError("frontier width must be positive")
    for key in (
        "focus_probability",
        "minimum_seed_rate",
        "minimum_pooled_rate",
        "minimum_frontier_seed_rate",
        "minimum_frontier_pooled_rate",
        "minimum_slice_seed_rate",
        "minimum_slice_pooled_rate",
    ):
        if not 0 <= float(curriculum[key]) <= 1:
            raise ValueError(f"curriculum {key} must be in [0, 1]")


def _load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text())
    version = state.get("schema_version")
    if version not in {1, SCHEMA_VERSION}:
        raise ValueError(f"unsupported state schema: {version}")
    # Schema-1 states remain valid and are atomically persisted as schema 2 at
    # the next record. Existing checkpoints and accepted evidence are kept.
    curriculum = state.get("curriculum", {})
    curriculum.setdefault("frontier_width_m", 0.005)
    curriculum.setdefault("selection_seed", 20260823)
    curriculum.setdefault("minimum_frontier_seed_rate", 0.35)
    curriculum.setdefault("minimum_frontier_pooled_rate", 0.45)
    curriculum.setdefault("minimum_slice_seed_rate", 0.35)
    curriculum.setdefault("minimum_slice_pooled_rate", 0.45)
    state["schema_version"] = SCHEMA_VERSION
    state.get("accepted", {}).setdefault(
        "gate_profile", "legacy_trust_root"
    )
    attempt = state.get("active_attempt")
    if attempt is not None:
        attempt.setdefault("frontier_evaluations", [])
        attempt.setdefault("slice_evaluations", [])
        attempt.setdefault("selected_checkpoint", None)
        attempt.setdefault("selection_seed", None)
        attempt.setdefault("selection_profile", None)
    _validate_state(state)
    return state


def _format_mm(value: float) -> str:
    return f"{round(value * 10000):04d}"


def _new_attempt(state: dict[str, Any], state_dir: Path) -> dict[str, Any]:
    curriculum = state["curriculum"]
    accepted = state["accepted"]
    x_min = float(accepted["x_min_m"])
    old_upper = float(accepted["x_upper_m"])
    new_upper = min(
        old_upper + float(curriculum["step_m"]),
        float(curriculum["reach_gate_x_upper_m"]),
    )
    stage_index = int(state["next_stage_index"])
    label = (
        f"adaptive_x_{stage_index:04d}_lo{_format_mm(x_min)}_"
        f"hi{_format_mm(new_upper)}_step{_format_mm(new_upper - old_upper)}"
    )
    output = (state_dir / "stages" / label).resolve()
    return {
        "stage_index": stage_index,
        "label": label,
        "phase": "planned",
        "started_at": _timestamp(),
        "x_min_m": x_min,
        "previous_x_upper_m": old_upper,
        "candidate_x_upper_m": new_upper,
        "step_m": new_upper - old_upper,
        "warm_start": accepted["checkpoint"],
        "output": str(output),
        "checkpoint": str(output / f"model_{int(curriculum['iterations']) - 1}.pt"),
        "selected_checkpoint": None,
        "tournament": [],
        "evaluations": [],
        "frontier_evaluations": [],
        "slice_evaluations": [],
        "selection_seed": None,
        "selection_profile": None,
    }


def _stage_environment(state: dict[str, Any], attempt: dict[str, Any]) -> dict[str, str]:
    curriculum = state["curriculum"]
    x_min = float(attempt["x_min_m"])
    old_upper = float(attempt["previous_x_upper_m"])
    new_upper = float(attempt["candidate_x_upper_m"])
    frontier_lower = max(
        x_min, new_upper - float(curriculum["frontier_width_m"])
    )
    return {
        "SIMPLE_PPO_INTERIOR_SEED": str(curriculum["training_seed"]),
        "SIMPLE_PPO_FRONTIER_LABEL": str(attempt["label"]),
        "SIMPLE_PPO_FRONTIER_OUTPUT": str(attempt["output"]),
        "SIMPLE_PPO_FRONTIER_ITERATIONS": str(curriculum["iterations"]),
        "SIMPLE_PPO_FRONTIER_X_CENTER": f"{(x_min + new_upper) / 2:.9f}",
        "SIMPLE_PPO_FRONTIER_X_JITTER": f"{(new_upper - x_min) / 2:.9f}",
        "SIMPLE_PPO_FRONTIER_Y_JITTER": str(curriculum["y_jitter_m"]),
        "SIMPLE_PPO_FRONTIER_YAW_JITTER": str(curriculum["yaw_jitter_rad"]),
        "SIMPLE_PPO_FRONTIER_FOCUS_PROBABILITY": str(
            curriculum["focus_probability"]
        ),
        # Keep a trailing frontier band in focus, rather than only the newly
        # added slice. This repairs weak poses near the previous boundary as
        # X advances. The remaining samples still cover the full envelope.
        "SIMPLE_PPO_FRONTIER_FOCUS_X_CENTER": f"{(frontier_lower + new_upper) / 2:.9f}",
        "SIMPLE_PPO_FRONTIER_FOCUS_X_JITTER": f"{(new_upper - frontier_lower) / 2:.9f}",
        "SIMPLE_PPO_FRONTIER_FOCUS_Y_JITTER": str(curriculum["y_jitter_m"]),
        "SIMPLE_PPO_FRONTIER_EXPLORATION_STD": "0.025",
        "SIMPLE_PPO_FRONTIER_ACTOR_LR_SCALE": "0.05",
        "SIMPLE_PPO_FRONTIER_X_SHOULDER_GAIN": "-7.5",
        "SIMPLE_PPO_FRONTIER_X_ELBOW_GAIN": "2.4",
        "SIMPLE_PPO_FRONTIER_Y_SHOULDER_GAIN": "0",
        "SIMPLE_PPO_FRONTIER_Y_WRIST_GAIN": "0",
        "SIMPLE_PPO_FRONTIER_MIN_TABLE_MARGIN": "-0.03",
        "SIMPLE_PPO_CHECKPOINT_INTERIOR_FRONTIER_WARM_START": str(
            attempt["warm_start"]
        ),
        "SIMPLE_PPO_CHECKPOINT_INTERIOR_FRONTIER": str(attempt["checkpoint"]),
    }


def _command(stage: str) -> list[str]:
    return ["bash", str(CUP6_SCRIPT), stage]


def _print_plan(state: dict[str, Any], attempt: dict[str, Any]) -> None:
    environment = _stage_environment(state, attempt)
    acceptance = []
    for seed in state["curriculum"]["evaluation_seeds"]:
        envs = state["curriculum"]["acceptance_envs"]
        output = Path(attempt["output"]) / "acceptance" / f"seed{seed}_{envs}.json"
        acceptance.append(
            {
                "seed": seed,
                "output": str(output),
                "command": _command("accept_interior_frontier_s017x"),
            }
        )
    frontier_environment = _frontier_evaluation_environment(state, attempt, environment)
    print(
        json.dumps(
            {
                "dry_run": True,
                "state_would_be_created": state,
                "attempt": attempt,
                "environment": environment,
                "train_command": _command("train_interior_frontier_s017x"),
                "selection_seed": state["curriculum"]["selection_seed"],
                "tournament_checkpoints": [
                    str(path) for path in _checkpoint_candidates(state, attempt)
                ],
                "acceptance": acceptance,
                "frontier_acceptance_environment": frontier_environment,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _record(
    state: dict[str, Any], state_path: Path, history_path: Path, event: dict[str, Any]
) -> None:
    state["updated_at"] = _timestamp()
    event = {"timestamp": state["updated_at"], **event}
    _append_history(history_path, event)
    _atomic_write_json(state_path, state)


def _frontier_stall_from_event(
    state: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any] | None:
    """Describe a structural boundary stall without changing the trust root."""

    if event.get("event") != "attempt_rejected":
        return None
    gate = event.get("gate")
    if not isinstance(gate, dict) or not bool(gate.get("envelope_passed")):
        return None
    frontier = gate.get("frontier")
    new_slice = gate.get("new_slice")
    if not isinstance(frontier, dict) or not isinstance(new_slice, dict):
        return None
    if bool(frontier.get("passed")) and bool(new_slice.get("passed")):
        return None
    minimum_step = float(state["curriculum"]["minimum_step_m"])
    next_step = float(event.get("next_step_m", math.inf))
    candidate = float(event["candidate_x_upper_m"])
    accepted_upper = float(state["accepted"]["x_upper_m"])
    attempted_step = candidate - accepted_upper
    if attempted_step <= minimum_step + 1e-12:
        return None
    if attempted_step > 2 * minimum_step + 1e-12:
        return None
    if next_step > minimum_step + 1e-12:
        return None
    return {
        "boundary_x_m": accepted_upper,
        "candidate_x_upper_m": candidate,
        "failed_attempt": str(event.get("label", "unknown")),
        "reason": "boundary_gate_plateau",
        "recommended_action": "proposal_experiment",
        "gate": gate,
    }


def _reconcile_frontier_stall(
    state: dict[str, Any], state_path: Path, history_path: Path
) -> bool:
    if state["status"] == "frontier_stalled":
        return False
    if not history_path.is_file():
        raise FileNotFoundError(f"curriculum history does not exist: {history_path}")
    events = [json.loads(line) for line in history_path.read_text().splitlines() if line]
    stall = next(
        (
            candidate
            for event in reversed(events)
            if (candidate := _frontier_stall_from_event(state, event)) is not None
        ),
        None,
    )
    if stall is None:
        return False
    state["status"] = "frontier_stalled"
    state["frontier_stall"] = stall
    _record(
        state,
        state_path,
        history_path,
        {"event": "frontier_stalled", "frontier_stall": stall},
    )
    return True


def _run_command(stage: str, environment: dict[str, str]) -> None:
    subprocess.run(
        _command(stage),
        cwd=REPO_ROOT,
        env={**os.environ, **environment},
        check=True,
    )


def _checkpoint_candidates(
    state: dict[str, Any], attempt: dict[str, Any]
) -> list[Path]:
    iterations = int(state["curriculum"]["iterations"])
    indices = list(range(0, iterations, 4))
    indices.append(iterations - 1)
    return [
        Path(attempt["output"]) / f"model_{index}.pt"
        for index in sorted(set(indices))
    ]


def _x_bounds(
    state: dict[str, Any], attempt: dict[str, Any], profile: str
) -> tuple[float, float]:
    x_min = float(attempt["x_min_m"])
    previous = float(attempt["previous_x_upper_m"])
    upper = float(attempt["candidate_x_upper_m"])
    if profile == "envelope":
        return x_min, upper
    if profile == "trailing_frontier":
        return max(x_min, upper - float(state["curriculum"]["frontier_width_m"])), upper
    if profile == "new_slice":
        return previous, upper
    raise ValueError(f"unsupported evaluation profile: {profile}")


def _evaluation_fingerprint(
    state: dict[str, Any],
    attempt: dict[str, Any],
    *,
    profile: str,
    envs: int,
    checkpoint: Path,
    environment: dict[str, str],
) -> str:
    lower, upper = _x_bounds(state, attempt, profile)
    payload = {
        "profile": profile,
        "x_bounds_m": [lower, upper],
        "y_jitter_m": float(state["curriculum"]["y_jitter_m"]),
        "yaw_jitter_rad": float(state["curriculum"]["yaw_jitter_rad"]),
        "dr_profile": "pose_only",
        "dr_strength": 1.0,
        "reference_noise": False,
        "envs": int(envs),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "contract_files": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in EVALUATION_CONTRACT_FILES
        },
        "environment": {
            key: value
            for key, value in sorted(environment.items())
            if key.startswith("SIMPLE_PPO_")
            and key
            not in {
                "SIMPLE_PPO_ACCEPTANCE_OUTPUT",
                "SIMPLE_PPO_CHECKPOINT_INTERIOR_FRONTIER",
                "SIMPLE_PPO_CHECKPOINT_INTERIOR_FRONTIER_WARM_START",
                "SIMPLE_PPO_EVAL_SEED",
            }
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _validated_evidence_result(
    row: dict[str, Any],
    *,
    checkpoint: Path,
    seed: int,
    envs: int,
    fingerprint: str,
) -> dict[str, Any]:
    if int(row.get("seed", -1)) != seed:
        raise ValueError("evaluation evidence row has the wrong seed")
    if row.get("evaluation_profile") != fingerprint:
        raise ValueError("evaluation evidence row has the wrong fingerprint")
    row_checkpoint = Path(str(row.get("checkpoint", checkpoint)))
    if row_checkpoint.resolve() != checkpoint.resolve():
        raise ValueError("evaluation evidence row has the wrong checkpoint")
    output = Path(str(row["output"]))
    result = _json_result(output)
    if int(result["policy_seed"]) != seed:
        raise ValueError(f"evaluation seed mismatch in {output}")
    if int(result["episodes"]) != envs:
        raise ValueError(f"evaluation episode count mismatch in {output}")
    if Path(str(result["checkpoint"])).resolve() != checkpoint.resolve():
        raise ValueError(f"evaluation checkpoint mismatch in {output}")
    for key in ("successes", "episodes"):
        if int(row[key]) != int(result[key]):
            raise ValueError(f"evaluation {key} evidence mismatch in {output}")
    if not math.isclose(
        float(row["success_rate"]), float(result["success_rate"]), abs_tol=1e-12
    ):
        raise ValueError(f"evaluation success-rate evidence mismatch in {output}")
    return result


def _evidence_by_seed(
    rows: list[dict[str, Any]], fingerprint: str
) -> dict[int, dict[str, Any]]:
    matching = [
        row for row in rows if row.get("evaluation_profile") == fingerprint
    ]
    seeds = [int(row["seed"]) for row in matching]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate evaluation evidence seed")
    return {int(row["seed"]): row for row in matching}


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    output: Path,
    seed: int,
    envs: int,
    environment: dict[str, str],
) -> tuple[dict[str, Any], bool]:
    recovered = output.exists()
    if recovered:
        try:
            result = _json_result(output)
        except (json.JSONDecodeError, ValueError) as error:
            raise FileExistsError(
                "incomplete acceptance output requires manual inspection; "
                f"refusing to overwrite {output}"
            ) from error
    else:
        evaluation_environment = {
            **environment,
            "SIMPLE_PPO_CHECKPOINT_INTERIOR_FRONTIER": str(checkpoint),
            "SIMPLE_PPO_EVAL_SEED": str(seed),
            "SIMPLE_PPO_ACCEPTANCE_ENVS": str(envs),
            "SIMPLE_PPO_ACCEPTANCE_OUTPUT": str(output),
            "SIMPLE_PPO_MINIMUM_SUCCESS_RATE": "0",
        }
        _run_command("accept_interior_frontier_s017x", evaluation_environment)
        result = _json_result(output)
    if int(result["policy_seed"]) != seed:
        raise ValueError(f"evaluation seed mismatch in {output}")
    if int(result["episodes"]) != envs:
        raise ValueError(f"evaluation episode count mismatch in {output}")
    if Path(str(result["checkpoint"])).resolve() != checkpoint.resolve():
        raise ValueError(f"evaluation checkpoint mismatch in {output}")
    return result, recovered


def _run_tournament(
    state: dict[str, Any],
    attempt: dict[str, Any],
    state_path: Path,
    history_path: Path,
    environment: dict[str, str],
) -> None:
    seed = int(state["curriculum"]["selection_seed"])
    envs = int(state["curriculum"]["acceptance_envs"])
    profile = "new_slice"
    selection_environment = _profile_evaluation_environment(
        state, attempt, environment, profile=profile
    )
    if attempt.get("selected_checkpoint"):
        selected = Path(attempt["selected_checkpoint"])
        if not selected.is_file():
            raise FileNotFoundError(f"selected checkpoint is missing: {selected}")
        selected_fingerprint = _evaluation_fingerprint(
            state,
            attempt,
            profile=profile,
            envs=envs,
            checkpoint=selected,
            environment=selection_environment,
        )
        selection_is_current = (
            int(attempt.get("selection_seed") or -1) == seed
            and attempt.get("selection_profile") == selected_fingerprint
        )
        if selection_is_current:
            selected_rows = [
                row
                for row in attempt.get("tournament", [])
                if row.get("checkpoint") == str(selected)
                and int(row.get("seed", -1)) == seed
                and row.get("evaluation_profile") == selected_fingerprint
            ]
            if len(selected_rows) != 1:
                raise ValueError("selected tournament evidence is missing or duplicated")
            _validated_evidence_result(
                selected_rows[0],
                checkpoint=selected,
                seed=seed,
                envs=envs,
                fingerprint=selected_fingerprint,
            )
            return

    # A legacy or semantically different selection must not bypass the
    # independent selection seed. Gate evidence is tied to the selected model.
    attempt["selected_checkpoint"] = None
    attempt["selection_seed"] = None
    attempt["selection_profile"] = None
    attempt["evaluations"] = []
    attempt["frontier_evaluations"] = []
    attempt["slice_evaluations"] = []
    matching_rows = []
    for checkpoint in _checkpoint_candidates(state, attempt):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"tournament checkpoint is missing: {checkpoint}")
        key = str(checkpoint)
        fingerprint = _evaluation_fingerprint(
            state,
            attempt,
            profile=profile,
            envs=envs,
            checkpoint=checkpoint,
            environment=selection_environment,
        )
        known = [
            row
            for row in attempt.get("tournament", [])
            if row.get("checkpoint") == key
            and int(row.get("seed", -1)) == seed
            and row.get("evaluation_profile") == fingerprint
        ]
        if len(known) > 1:
            raise ValueError(f"duplicate tournament evidence for {checkpoint}")
        if known:
            _validated_evidence_result(
                known[0],
                checkpoint=checkpoint,
                seed=seed,
                envs=envs,
                fingerprint=fingerprint,
            )
            matching_rows.append(known[0])
            continue
        model_index = checkpoint.stem.removeprefix("model_")
        output = (
            Path(attempt["output"])
            / "acceptance"
            / f"tournament_slice_{fingerprint}_model{model_index}_seed{seed}.json"
        )
        result, recovered = _evaluate_checkpoint(
            checkpoint=checkpoint,
            output=output,
            seed=seed,
            envs=envs,
            environment=selection_environment,
        )
        row = {
            "checkpoint": key,
            "seed": seed,
            "evaluation_profile": fingerprint,
            "output": str(output),
            "successes": int(result["successes"]),
            "episodes": int(result["episodes"]),
            "success_rate": float(result["success_rate"]),
            "grasp_episode_rate": float(result.get("grasp_episode_rate", 0.0)),
        }
        attempt.setdefault("tournament", []).append(row)
        matching_rows.append(row)
        _record(
            state,
            state_path,
            history_path,
            {
                "event": (
                    "tournament_evaluation_recovered"
                    if recovered
                    else "tournament_checkpoint_evaluated"
                ),
                "label": attempt["label"],
                **row,
            },
        )

    if len(matching_rows) != len(_checkpoint_candidates(state, attempt)):
        raise RuntimeError("selection tournament evidence is incomplete")
    winner = max(
        matching_rows,
        key=lambda row: (
            row["success_rate"],
            row["grasp_episode_rate"],
            row["successes"],
        ),
    )
    attempt["selected_checkpoint"] = winner["checkpoint"]
    attempt["selection_seed"] = seed
    attempt["selection_profile"] = winner["evaluation_profile"]
    # Tournament evidence is deliberately not reused by the held-out gate.
    # All three acceptance seeds remain independent of checkpoint selection.
    attempt["evaluations"] = []
    _record(
        state,
        state_path,
        history_path,
        {
            "event": "tournament_winner_selected",
            "label": attempt["label"],
            "checkpoint": winner["checkpoint"],
            "seed": seed,
            "evaluation_profile": winner["evaluation_profile"],
            "success_rate": winner["success_rate"],
        },
    )


def _profile_evaluation_environment(
    state: dict[str, Any],
    attempt: dict[str, Any],
    environment: dict[str, str],
    *,
    profile: str,
) -> dict[str, str]:
    lower, upper = _x_bounds(state, attempt, profile)
    return {
        **environment,
        "SIMPLE_PPO_FRONTIER_X_CENTER": f"{(lower + upper) / 2:.9f}",
        "SIMPLE_PPO_FRONTIER_X_JITTER": f"{(upper - lower) / 2:.9f}",
        "SIMPLE_PPO_FRONTIER_FOCUS_PROBABILITY": "0",
    }


def _frontier_evaluation_environment(
    state: dict[str, Any],
    attempt: dict[str, Any],
    environment: dict[str, str],
) -> dict[str, str]:
    return _profile_evaluation_environment(
        state, attempt, environment, profile="trailing_frontier"
    )


def _run_attempt(
    state: dict[str, Any], state_path: Path, history_path: Path, state_dir: Path
) -> bool:
    attempt = state.get("active_attempt")
    if attempt is None:
        attempt = _new_attempt(state, state_dir)
        output = Path(attempt["output"])
        if output.exists():
            raise FileExistsError(f"refusing to reuse adaptive PPO output: {output}")
        state["active_attempt"] = attempt
        _record(
            state,
            state_path,
            history_path,
            {"event": "attempt_started", "attempt": attempt.copy()},
        )
    environment = _stage_environment(state, attempt)
    checkpoint = Path(attempt["checkpoint"])
    if attempt["phase"] == "planned":
        recovered = checkpoint.is_file()
        if not recovered:
            output = Path(attempt["output"])
            if output.exists():
                raise FileExistsError(
                    "incomplete adaptive PPO output requires manual inspection; "
                    f"refusing to overwrite {output}"
                )
            _run_command("train_interior_frontier_s017x", environment)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"training did not produce {checkpoint}")
        attempt["phase"] = "evaluating"
        _record(
            state,
            state_path,
            history_path,
            {
                "event": (
                    "training_recovered" if recovered else "training_completed"
                ),
                "label": attempt["label"],
            },
        )
    elif attempt["phase"] != "evaluating":
        raise ValueError(f"unsupported active attempt phase: {attempt['phase']}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"active attempt checkpoint is missing: {checkpoint}")

    _run_tournament(
        state, attempt, state_path, history_path, environment
    )
    selected_checkpoint = Path(attempt["selected_checkpoint"])

    envs = int(state["curriculum"]["acceptance_envs"])
    envelope_environment = _profile_evaluation_environment(
        state, attempt, environment, profile="envelope"
    )
    envelope_fingerprint = _evaluation_fingerprint(
        state,
        attempt,
        profile="envelope",
        envs=envs,
        checkpoint=selected_checkpoint,
        environment=envelope_environment,
    )
    known = _evidence_by_seed(attempt["evaluations"], envelope_fingerprint)
    for seed, row in known.items():
        _validated_evidence_result(
            row,
            checkpoint=selected_checkpoint,
            seed=seed,
            envs=envs,
            fingerprint=envelope_fingerprint,
        )
    for seed in state["curriculum"]["evaluation_seeds"]:
        seed = int(seed)
        if seed in known:
            continue
        output = (
            Path(attempt["output"])
            / "acceptance"
            / f"envelope_{envelope_fingerprint}_seed{seed}.json"
        )
        result, recovered = _evaluate_checkpoint(
            checkpoint=selected_checkpoint,
            output=output,
            seed=seed,
            envs=envs,
            environment=envelope_environment,
        )
        row = {
            "checkpoint": str(selected_checkpoint),
            "seed": seed,
            "evaluation_profile": envelope_fingerprint,
            "output": str(output),
            "successes": int(result["successes"]),
            "episodes": int(result["episodes"]),
            "success_rate": float(result["success_rate"]),
        }
        attempt["evaluations"].append(row)
        known[seed] = row
        _record(
            state,
            state_path,
            history_path,
            {
                "event": "seed_evaluation_recovered" if recovered else "seed_evaluated",
                "label": attempt["label"],
                **row,
            },
        )

    results = [_json_result(Path(row["output"])) for row in known.values()]
    gate = _gate(
        results,
        minimum_seed_rate=float(state["curriculum"]["minimum_seed_rate"]),
        minimum_pooled_rate=float(state["curriculum"]["minimum_pooled_rate"]),
    )
    envelope_passed = bool(gate["passed"])

    attempt.setdefault("frontier_evaluations", [])
    frontier_environment = _frontier_evaluation_environment(
        state, attempt, environment
    )
    frontier_fingerprint = _evaluation_fingerprint(
        state,
        attempt,
        profile="trailing_frontier",
        envs=envs,
        checkpoint=selected_checkpoint,
        environment=frontier_environment,
    )
    known_frontier = _evidence_by_seed(
        attempt["frontier_evaluations"], frontier_fingerprint
    )
    for seed, row in known_frontier.items():
        _validated_evidence_result(
            row,
            checkpoint=selected_checkpoint,
            seed=seed,
            envs=envs,
            fingerprint=frontier_fingerprint,
        )
    for seed in state["curriculum"]["evaluation_seeds"]:
        seed = int(seed)
        if seed in known_frontier:
            continue
        output = (
            Path(attempt["output"])
            / "acceptance"
            / f"frontier_{frontier_fingerprint}_seed{seed}.json"
        )
        result, recovered = _evaluate_checkpoint(
            checkpoint=selected_checkpoint,
            output=output,
            seed=seed,
            envs=envs,
            environment=frontier_environment,
        )
        row = {
            "checkpoint": str(selected_checkpoint),
            "seed": seed,
            "evaluation_profile": frontier_fingerprint,
            "output": str(output),
            "successes": int(result["successes"]),
            "episodes": int(result["episodes"]),
            "success_rate": float(result["success_rate"]),
        }
        attempt["frontier_evaluations"].append(row)
        known_frontier[seed] = row
        _record(
            state,
            state_path,
            history_path,
            {
                "event": (
                    "frontier_seed_evaluation_recovered"
                    if recovered
                    else "frontier_seed_evaluated"
                ),
                "label": attempt["label"],
                **row,
            },
        )

    frontier_results = [
        _json_result(Path(row["output"])) for row in known_frontier.values()
    ]
    frontier_gate = _gate(
        frontier_results,
        minimum_seed_rate=float(
            state["curriculum"]["minimum_frontier_seed_rate"]
        ),
        minimum_pooled_rate=float(
            state["curriculum"]["minimum_frontier_pooled_rate"]
        ),
    )

    attempt.setdefault("slice_evaluations", [])
    slice_environment = _profile_evaluation_environment(
        state, attempt, environment, profile="new_slice"
    )
    slice_fingerprint = _evaluation_fingerprint(
        state,
        attempt,
        profile="new_slice",
        envs=envs,
        checkpoint=selected_checkpoint,
        environment=slice_environment,
    )
    known_slice = _evidence_by_seed(
        attempt["slice_evaluations"], slice_fingerprint
    )
    for seed, row in known_slice.items():
        _validated_evidence_result(
            row,
            checkpoint=selected_checkpoint,
            seed=seed,
            envs=envs,
            fingerprint=slice_fingerprint,
        )
    for seed in state["curriculum"]["evaluation_seeds"]:
        seed = int(seed)
        if seed in known_slice:
            continue
        output = (
            Path(attempt["output"])
            / "acceptance"
            / f"slice_{slice_fingerprint}_seed{seed}.json"
        )
        result, recovered = _evaluate_checkpoint(
            checkpoint=selected_checkpoint,
            output=output,
            seed=seed,
            envs=envs,
            environment=slice_environment,
        )
        row = {
            "checkpoint": str(selected_checkpoint),
            "seed": seed,
            "evaluation_profile": slice_fingerprint,
            "output": str(output),
            "successes": int(result["successes"]),
            "episodes": int(result["episodes"]),
            "success_rate": float(result["success_rate"]),
        }
        attempt["slice_evaluations"].append(row)
        known_slice[seed] = row
        _record(
            state,
            state_path,
            history_path,
            {
                "event": (
                    "slice_seed_evaluation_recovered"
                    if recovered
                    else "slice_seed_evaluated"
                ),
                "label": attempt["label"],
                **row,
            },
        )
    slice_results = [
        _json_result(Path(row["output"])) for row in known_slice.values()
    ]
    slice_gate = _gate(
        slice_results,
        minimum_seed_rate=float(
            state["curriculum"]["minimum_slice_seed_rate"]
        ),
        minimum_pooled_rate=float(
            state["curriculum"]["minimum_slice_pooled_rate"]
        ),
    )
    gate["envelope_passed"] = envelope_passed
    gate["frontier"] = frontier_gate
    gate["new_slice"] = slice_gate
    gate["passed"] = (
        envelope_passed
        and bool(frontier_gate["passed"])
        and bool(slice_gate["passed"])
    )
    state["next_stage_index"] = int(state["next_stage_index"]) + 1
    state["active_attempt"] = None
    if gate["passed"]:
        state["accepted"] = {
            "x_min_m": attempt["x_min_m"],
            "x_upper_m": attempt["candidate_x_upper_m"],
            "checkpoint": attempt["selected_checkpoint"],
            "gate": gate,
            "evidence": [
                row["output"]
                for row in (
                    list(known.values())
                    + list(known_frontier.values())
                    + list(known_slice.values())
                )
            ],
            "gate_profile": "adaptive_x_triple_v2",
        }
        successes = int(state["consecutive_successes"]) + 1
        state["consecutive_successes"] = successes
        if successes >= int(state["curriculum"]["grow_after_successes"]):
            state["curriculum"]["step_m"] = min(
                float(state["curriculum"]["maximum_step_m"]),
                float(state["curriculum"]["step_m"])
                * float(state["curriculum"]["growth_factor"]),
            )
            state["consecutive_successes"] = 0
        if float(state["accepted"]["x_upper_m"]) >= float(
            state["curriculum"]["reach_gate_x_upper_m"]
        ) - 1e-12:
            state["status"] = "reach_gate_complete"
        event_name = "attempt_promoted"
    else:
        state["consecutive_successes"] = 0
        minimum_step = float(state["curriculum"]["minimum_step_m"])
        state["curriculum"]["step_m"] = max(
            minimum_step,
            float(attempt["step_m"]) / 2,
        )
        if float(attempt["step_m"]) <= minimum_step + 1e-12:
            state["status"] = "minimum_step_failed"
        event_name = "attempt_rejected"
    _record(
        state,
        state_path,
        history_path,
        {
            "event": event_name,
            "label": attempt["label"],
            "checkpoint": attempt["selected_checkpoint"],
            "candidate_x_upper_m": attempt["candidate_x_upper_m"],
            "next_step_m": state["curriculum"]["step_m"],
            "gate": gate,
        },
    )
    if not gate["passed"]:
        stall = _frontier_stall_from_event(
            state,
            {
                "event": event_name,
                "label": attempt["label"],
                "candidate_x_upper_m": attempt["candidate_x_upper_m"],
                "next_step_m": state["curriculum"]["step_m"],
                "gate": gate,
            },
        )
        if stall is not None:
            state["status"] = "frontier_stalled"
            state["frontier_stall"] = stall
            _record(
                state,
                state_path,
                history_path,
                {"event": "frontier_stalled", "frontier_stall": stall},
            )
    return bool(gate["passed"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("run", "status", "reconcile-stall"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-minimum-step-probe",
        action="store_true",
        help="Explicitly run one minimum-step diagnostic after a frontier stall",
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--max-stages", type=int, default=1)
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT
    )
    parser.add_argument("--allow-unverified-initial-checkpoint", action="store_true")
    parser.add_argument("--x-min", type=float, default=0.014)
    parser.add_argument("--initial-x-upper", type=float, default=0.033)
    parser.add_argument("--reach-gate", type=float, default=0.06)
    parser.add_argument("--initial-step", type=float, default=0.001)
    parser.add_argument("--minimum-step", type=float, default=0.00025)
    parser.add_argument("--maximum-step", type=float, default=0.002)
    parser.add_argument("--grow-after-successes", type=int, default=2)
    parser.add_argument("--growth-factor", type=float, default=2.0)
    parser.add_argument("--y-jitter", type=float, default=0.015)
    parser.add_argument("--yaw-jitter", type=float, default=0.0225)
    parser.add_argument("--focus-probability", type=float, default=0.70)
    parser.add_argument("--frontier-width", type=float, default=0.005)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--training-seed", type=int, default=20260824)
    parser.add_argument("--selection-seed", type=int, default=20260823)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs=3,
        default=(20260824, 20260825, 20260826),
    )
    parser.add_argument("--acceptance-envs", type=int, default=512)
    parser.add_argument("--minimum-seed-rate", type=float, default=0.65)
    parser.add_argument("--minimum-pooled-rate", type=float, default=0.70)
    parser.add_argument("--minimum-frontier-seed-rate", type=float, default=0.35)
    parser.add_argument("--minimum-frontier-pooled-rate", type=float, default=0.45)
    parser.add_argument("--minimum-slice-seed-rate", type=float, default=0.35)
    parser.add_argument("--minimum-slice-pooled-rate", type=float, default=0.45)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_stages < 0:
        raise ValueError("max-stages must be non-negative (zero means unlimited)")
    if not args.x_min < args.initial_x_upper <= args.reach_gate <= 0.06:
        raise ValueError(
            "X bounds must satisfy x-min < initial upper <= reach gate <= 0.06"
        )
    if not 0 < args.minimum_step <= args.initial_step <= args.maximum_step:
        raise ValueError("step sizes must be positive and ordered")
    if args.grow_after_successes < 1 or args.growth_factor < 1:
        raise ValueError("success growth settings are invalid")
    if len(set(args.evaluation_seeds)) != 3:
        raise ValueError("evaluation seeds must be distinct")
    if args.acceptance_envs < 1 or args.iterations < 1:
        raise ValueError("acceptance-envs and iterations must be positive")
    if args.frontier_width <= 0:
        raise ValueError("frontier-width must be positive")
    if args.selection_seed in args.evaluation_seeds:
        raise ValueError("selection seed must be distinct from evaluation seeds")
    for name in (
        "minimum_seed_rate",
        "minimum_pooled_rate",
        "minimum_frontier_seed_rate",
        "minimum_frontier_pooled_rate",
        "minimum_slice_seed_rate",
        "minimum_slice_pooled_rate",
        "focus_probability",
    ):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1]")


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    state_dir = _canonical(args.state_dir).resolve()
    state_path = state_dir / "state.json"
    history_path = state_dir / "history.jsonl"
    if state_path.exists():
        state = _load_state(state_path)
    else:
        state = _initial_state(args)

    if args.action == "status":
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    if args.action == "reconcile-stall":
        if not state_path.exists():
            raise FileNotFoundError(f"curriculum state does not exist: {state_path}")
        with _exclusive_lock(state_dir):
            state = _load_state(state_path)
            _reconcile_frontier_stall(state, state_path, history_path)
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    if state["status"] in {
        "reach_gate_complete",
        "minimum_step_failed",
        "frontier_stalled",
    } and not (
        state["status"] == "frontier_stalled" and args.force_minimum_step_probe
    ):
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    if args.dry_run:
        attempt = state.get("active_attempt") or _new_attempt(state, state_dir)
        if state.get("active_attempt") is None and Path(attempt["output"]).exists():
            raise FileExistsError(
                f"refusing to reuse adaptive PPO output: {attempt['output']}"
            )
        _print_plan(state, attempt)
        return

    with _exclusive_lock(state_dir):
        if state_path.exists():
            state = _load_state(state_path)
        else:
            _record(
                state,
                state_path,
                history_path,
                {"event": "curriculum_initialized", "accepted": state["accepted"]},
            )
        if state["status"] == "frontier_stalled":
            if not args.force_minimum_step_probe:
                print(json.dumps(state, indent=2, sort_keys=True))
                return
            state["status"] = "ready"
            _record(
                state,
                state_path,
                history_path,
                {
                    "event": "minimum_step_probe_forced",
                    "frontier_stall": state.get("frontier_stall"),
                },
            )
        completed = 0
        while state["status"] not in {
            "reach_gate_complete",
            "minimum_step_failed",
            "frontier_stalled",
        }:
            _run_attempt(state, state_path, history_path, state_dir)
            completed += 1
            if args.max_stages and completed >= args.max_stages:
                break
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"adaptive Cup_6 X curriculum: {error}", file=sys.stderr)
        raise SystemExit(2) from error
