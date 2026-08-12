import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/grasp_rl/grasp_anything_cup6_adaptive_x.py"
)
SPEC = importlib.util.spec_from_file_location("grasp_anything_cup6_adaptive_x", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADAPTIVE_X = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTIVE_X)


def _state() -> dict:
    return {
        "curriculum": {
            "iterations": 20,
            "training_seed": 20260824,
            "y_jitter_m": 0.015,
            "yaw_jitter_rad": 0.0225,
            "focus_probability": 0.7,
            "frontier_width_m": 0.005,
        }
    }


def _attempt(tmp_path: Path) -> dict:
    return {
        "label": "candidate",
        "x_min_m": 0.014,
        "previous_x_upper_m": 0.033,
        "candidate_x_upper_m": 0.034,
        "step_m": 0.001,
        "warm_start": "/tmp/warm.pt",
        "output": str(tmp_path),
        "checkpoint": str(tmp_path / "model_19.pt"),
    }


def test_stage_focuses_new_x_slab(tmp_path: Path) -> None:
    environment = ADAPTIVE_X._stage_environment(_state(), _attempt(tmp_path))

    assert environment["SIMPLE_PPO_FRONTIER_X_CENTER"] == "0.024000000"
    assert environment["SIMPLE_PPO_FRONTIER_X_JITTER"] == "0.010000000"
    assert environment["SIMPLE_PPO_FRONTIER_FOCUS_X_CENTER"] == "0.031500000"
    assert environment["SIMPLE_PPO_FRONTIER_FOCUS_X_JITTER"] == "0.002500000"
    assert environment["SIMPLE_PPO_FRONTIER_Y_SHOULDER_GAIN"] == "0"
    assert environment["SIMPLE_PPO_FRONTIER_Y_WRIST_GAIN"] == "0"


def _failed_boundary_gate() -> dict:
    return {
        "passed": False,
        "envelope_passed": True,
        "frontier": {"passed": False, "pooled_success_rate": 0.38},
        "new_slice": {"passed": False, "pooled_success_rate": 0.18},
    }


def test_half_millimetre_boundary_failure_stalls_before_minimum_probe() -> None:
    state = {
        "accepted": {"x_upper_m": 0.039},
        "curriculum": {"minimum_step_m": 0.00025},
    }
    event = {
        "event": "attempt_rejected",
        "label": "stage7",
        "candidate_x_upper_m": 0.0395,
        "next_step_m": 0.00025,
        "gate": _failed_boundary_gate(),
    }

    stall = ADAPTIVE_X._frontier_stall_from_event(state, event)

    assert stall is not None
    assert stall["boundary_x_m"] == 0.039
    assert stall["recommended_action"] == "proposal_experiment"


def test_minimum_step_failure_does_not_relabel_as_frontier_stall() -> None:
    state = {
        "accepted": {"x_upper_m": 0.039},
        "curriculum": {"minimum_step_m": 0.00025},
    }
    event = {
        "event": "attempt_rejected",
        "label": "minimum-probe",
        "candidate_x_upper_m": 0.03925,
        "next_step_m": 0.00025,
        "gate": _failed_boundary_gate(),
    }

    assert ADAPTIVE_X._frontier_stall_from_event(state, event) is None


def test_reconcile_stall_uses_existing_evidence_without_changing_accepted(
    tmp_path: Path,
) -> None:
    state = _runtime_state(tmp_path)
    state["accepted"]["x_upper_m"] = 0.039
    state["curriculum"]["step_m"] = 0.00025
    state["curriculum"]["minimum_step_m"] = 0.00025
    state["active_attempt"] = None
    accepted = dict(state["accepted"])
    state_path = tmp_path / "state.json"
    history_path = tmp_path / "history.jsonl"
    state_path.write_text(json.dumps(state))
    history_path.write_text(
        json.dumps(
            {
                "event": "attempt_rejected",
                "label": "stage7",
                "candidate_x_upper_m": 0.0395,
                "next_step_m": 0.00025,
                "gate": _failed_boundary_gate(),
            }
        )
        + "\n"
    )

    changed = ADAPTIVE_X._reconcile_frontier_stall(
        state, state_path, history_path
    )

    persisted = json.loads(state_path.read_text())
    assert changed
    assert persisted["status"] == "frontier_stalled"
    assert persisted["accepted"] == accepted
    assert persisted["frontier_stall"]["failed_attempt"] == "stage7"


def test_frontier_evaluation_covers_trailing_band(tmp_path: Path) -> None:
    state = _state()
    attempt = _attempt(tmp_path)
    environment = ADAPTIVE_X._stage_environment(state, attempt)

    frontier = ADAPTIVE_X._frontier_evaluation_environment(
        state, attempt, environment
    )

    assert frontier["SIMPLE_PPO_FRONTIER_X_CENTER"] == "0.031500000"
    assert frontier["SIMPLE_PPO_FRONTIER_X_JITTER"] == "0.002500000"
    assert frontier["SIMPLE_PPO_FRONTIER_FOCUS_PROBABILITY"] == "0"


def test_new_slice_evaluation_covers_only_candidate_increment(tmp_path: Path) -> None:
    state = _state()
    attempt = _attempt(tmp_path)
    environment = ADAPTIVE_X._stage_environment(state, attempt)

    new_slice = ADAPTIVE_X._profile_evaluation_environment(
        state, attempt, environment, profile="new_slice"
    )

    assert new_slice["SIMPLE_PPO_FRONTIER_X_CENTER"] == "0.033500000"
    assert new_slice["SIMPLE_PPO_FRONTIER_X_JITTER"] == "0.000500000"
    assert new_slice["SIMPLE_PPO_FRONTIER_FOCUS_PROBABILITY"] == "0"


def test_evaluation_fingerprints_separate_profiles(tmp_path: Path) -> None:
    state = _state()
    attempt = _attempt(tmp_path)
    checkpoint = Path(attempt["checkpoint"])
    checkpoint.write_bytes(b"checkpoint")
    environment = ADAPTIVE_X._stage_environment(state, attempt)

    fingerprints = {
        ADAPTIVE_X._evaluation_fingerprint(
            state,
            attempt,
            profile=profile,
            envs=512,
            checkpoint=checkpoint,
            environment=ADAPTIVE_X._profile_evaluation_environment(
                state, attempt, environment, profile=profile
            ),
        )
        for profile in ("envelope", "trailing_frontier", "new_slice")
    }

    assert len(fingerprints) == 3


def test_evaluation_fingerprint_changes_with_checkpoint_content(
    tmp_path: Path,
) -> None:
    state = _state()
    attempt = _attempt(tmp_path)
    checkpoint = Path(attempt["checkpoint"])
    environment = ADAPTIVE_X._profile_evaluation_environment(
        state,
        attempt,
        ADAPTIVE_X._stage_environment(state, attempt),
        profile="new_slice",
    )
    checkpoint.write_bytes(b"first")
    first = ADAPTIVE_X._evaluation_fingerprint(
        state,
        attempt,
        profile="new_slice",
        envs=512,
        checkpoint=checkpoint,
        environment=environment,
    )
    checkpoint.write_bytes(b"second")
    second = ADAPTIVE_X._evaluation_fingerprint(
        state,
        attempt,
        profile="new_slice",
        envs=512,
        checkpoint=checkpoint,
        environment=environment,
    )

    assert first != second


def test_tournament_includes_all_saved_checkpoints(tmp_path: Path) -> None:
    candidates = ADAPTIVE_X._checkpoint_candidates(_state(), _attempt(tmp_path))

    assert [path.name for path in candidates] == [
        "model_0.pt",
        "model_4.pt",
        "model_8.pt",
        "model_12.pt",
        "model_16.pt",
        "model_19.pt",
    ]


def test_gate_requires_each_seed_and_pooled_rate() -> None:
    results = [
        {"policy_seed": 1, "successes": 70, "episodes": 100, "success_rate": 0.7},
        {"policy_seed": 2, "successes": 70, "episodes": 100, "success_rate": 0.7},
        {"policy_seed": 3, "successes": 69, "episodes": 100, "success_rate": 0.69},
    ]

    assert ADAPTIVE_X._gate(
        results, minimum_seed_rate=0.65, minimum_pooled_rate=0.69
    )["passed"]
    assert not ADAPTIVE_X._gate(
        results, minimum_seed_rate=0.70, minimum_pooled_rate=0.69
    )["passed"]
    assert not ADAPTIVE_X._gate(
        results, minimum_seed_rate=0.65, minimum_pooled_rate=0.71
    )["passed"]


def test_winner_prefers_success_then_grasp_rate() -> None:
    rows = [
        {"success_rate": 0.70, "grasp_episode_rate": 0.80, "successes": 70},
        {"success_rate": 0.70, "grasp_episode_rate": 0.85, "successes": 70},
        {"success_rate": 0.69, "grasp_episode_rate": 0.99, "successes": 69},
    ]

    winner = max(
        rows,
        key=lambda row: (
            row["success_rate"], row["grasp_episode_rate"], row["successes"]
        ),
    )
    assert winner is rows[1]


def test_state_rejects_selection_seed_reused_for_acceptance() -> None:
    state = {
        "schema_version": ADAPTIVE_X.SCHEMA_VERSION,
        "accepted": {"x_min_m": 0.014, "x_upper_m": 0.034},
        "curriculum": {
            "axis": "x",
            "reach_gate_x_upper_m": 0.06,
            "minimum_step_m": 0.00025,
            "step_m": 0.001,
            "maximum_step_m": 0.002,
            "evaluation_seeds": [1, 2, 3],
            "selection_seed": 1,
            "frontier_width_m": 0.005,
            "focus_probability": 0.7,
            "minimum_seed_rate": 0.65,
            "minimum_pooled_rate": 0.70,
            "minimum_frontier_seed_rate": 0.35,
            "minimum_frontier_pooled_rate": 0.45,
            "minimum_slice_seed_rate": 0.35,
            "minimum_slice_pooled_rate": 0.45,
        },
    }

    with pytest.raises(ValueError, match="selection seed"):
        ADAPTIVE_X._validate_state(state)


def _runtime_state(tmp_path: Path) -> dict:
    state = {
        "schema_version": ADAPTIVE_X.SCHEMA_VERSION,
        "status": "ready",
        "next_stage_index": 1,
        "consecutive_successes": 0,
        "accepted": {
            "x_min_m": 0.014,
            "x_upper_m": 0.033,
            "checkpoint": str(tmp_path / "warm.pt"),
        },
        "curriculum": {
            **_state()["curriculum"],
            "axis": "x",
            "reach_gate_x_upper_m": 0.06,
            "minimum_step_m": 0.00025,
            "step_m": 0.001,
            "maximum_step_m": 0.002,
            "grow_after_successes": 2,
            "growth_factor": 2.0,
            "selection_seed": 20260823,
            "evaluation_seeds": [20260824, 20260825, 20260826],
            "acceptance_envs": 512,
            "minimum_seed_rate": 0.65,
            "minimum_pooled_rate": 0.70,
            "minimum_frontier_seed_rate": 0.35,
            "minimum_frontier_pooled_rate": 0.45,
            "minimum_slice_seed_rate": 0.35,
            "minimum_slice_pooled_rate": 0.45,
        },
    }
    attempt = {
        **_attempt(tmp_path),
        "phase": "evaluating",
        "selected_checkpoint": None,
        "selection_seed": None,
        "selection_profile": None,
        "tournament": [],
        "evaluations": [],
        "frontier_evaluations": [],
        "slice_evaluations": [],
    }
    state["active_attempt"] = attempt
    return state


def _evaluation_result(checkpoint: Path, seed: int, rate: float) -> dict:
    episodes = 100
    successes = round(rate * episodes)
    return {
        "checkpoint": str(checkpoint),
        "policy_seed": seed,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "grasp_episode_rate": successes / episodes,
    }


def test_schema_one_migration_preserves_active_training(tmp_path: Path) -> None:
    state = _runtime_state(tmp_path)
    state["schema_version"] = 1
    attempt = state["active_attempt"]
    for key in (
        "frontier_evaluations",
        "slice_evaluations",
        "selection_seed",
        "selection_profile",
    ):
        attempt.pop(key)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))

    migrated = ADAPTIVE_X._load_state(state_path)

    assert migrated["schema_version"] == ADAPTIVE_X.SCHEMA_VERSION
    assert migrated["active_attempt"]["phase"] == "evaluating"
    assert migrated["active_attempt"]["checkpoint"] == attempt["checkpoint"]
    assert migrated["active_attempt"]["slice_evaluations"] == []


def test_tournament_ignores_legacy_rows_and_stale_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _runtime_state(tmp_path)
    attempt = state["active_attempt"]
    checkpoints = ADAPTIVE_X._checkpoint_candidates(state, attempt)
    for checkpoint in checkpoints:
        checkpoint.touch()
    stale = checkpoints[0]
    attempt["selected_checkpoint"] = str(stale)
    attempt["selection_seed"] = 20260823
    attempt["selection_profile"] = "trailing_frontier"
    attempt["tournament"] = [
        {
            "checkpoint": str(stale),
            "seed": 20260823,
            "evaluation_profile": "trailing_frontier",
            "success_rate": 1.0,
            "grasp_episode_rate": 1.0,
            "successes": 100,
        }
    ]
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        checkpoint = kwargs["checkpoint"]
        index = int(checkpoint.stem.removeprefix("model_"))
        return _evaluation_result(
            checkpoint, kwargs["seed"], 0.40 + index / 100
        ), False

    monkeypatch.setattr(ADAPTIVE_X, "_evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr(ADAPTIVE_X, "_record", lambda *args, **kwargs: None)
    environment = ADAPTIVE_X._stage_environment(state, attempt)

    ADAPTIVE_X._run_tournament(
        state,
        attempt,
        tmp_path / "state.json",
        tmp_path / "history.jsonl",
        environment,
    )

    fingerprint = ADAPTIVE_X._evaluation_fingerprint(
        state,
        attempt,
        profile="new_slice",
        envs=512,
        checkpoint=checkpoints[-1],
        environment=ADAPTIVE_X._profile_evaluation_environment(
            state, attempt, environment, profile="new_slice"
        ),
    )
    assert len(calls) == len(checkpoints)
    assert all(
        call["environment"]["SIMPLE_PPO_FRONTIER_X_CENTER"] == "0.033500000"
        for call in calls
    )
    assert attempt["selected_checkpoint"] == str(checkpoints[-1])
    assert attempt["selection_profile"] == fingerprint
    assert attempt["evaluations"] == []
    assert attempt["frontier_evaluations"] == []
    assert attempt["slice_evaluations"] == []


def test_run_attempt_gates_fresh_envelope_frontier_and_slice_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _runtime_state(tmp_path)
    attempt = state["active_attempt"]
    selected = Path(attempt["checkpoint"])
    selected.touch()
    attempt["selected_checkpoint"] = str(selected)
    attempt["selection_seed"] = state["curriculum"]["selection_seed"]
    environment = ADAPTIVE_X._stage_environment(state, attempt)
    attempt["selection_profile"] = ADAPTIVE_X._evaluation_fingerprint(
        state,
        attempt,
        profile="new_slice",
        envs=512,
        checkpoint=selected,
        environment=ADAPTIVE_X._profile_evaluation_environment(
            state, attempt, environment, profile="new_slice"
        ),
    )
    tournament_output = tmp_path / "selected.json"
    tournament_result = _evaluation_result(
        selected, state["curriculum"]["selection_seed"], 0.50
    )
    tournament_result["episodes"] = 512
    tournament_result["successes"] = 256
    tournament_result["success_rate"] = 0.50
    tournament_output.write_text(json.dumps({"result": tournament_result}, indent=2))
    attempt["tournament"] = [
        {
            "checkpoint": str(selected),
            "seed": state["curriculum"]["selection_seed"],
            "evaluation_profile": attempt["selection_profile"],
            "output": str(tournament_output),
            "successes": 256,
            "episodes": 512,
            "success_rate": 0.50,
            "grasp_episode_rate": 0.50,
        }
    ]

    def fake_evaluate(**kwargs):
        environment = kwargs["environment"]
        jitter = float(environment["SIMPLE_PPO_FRONTIER_X_JITTER"])
        # Envelope passes while both difficult boundary profiles fail. This
        # proves all newly appended rows reach their respective three-seed gate.
        rate = 0.75 if jitter > 0.005 else 0.40
        result = _evaluation_result(
            kwargs["checkpoint"], kwargs["seed"], rate
        )
        output = kwargs["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"result": result}, indent=2))
        return result, False

    monkeypatch.setattr(ADAPTIVE_X, "_evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr(ADAPTIVE_X, "_record", lambda *args, **kwargs: None)

    promoted = ADAPTIVE_X._run_attempt(
        state,
        tmp_path / "state.json",
        tmp_path / "history.jsonl",
        tmp_path,
    )

    assert not promoted
    assert state["accepted"]["x_upper_m"] == 0.033
    assert state["active_attempt"] is None
    assert state["curriculum"]["step_m"] == pytest.approx(0.0005)
