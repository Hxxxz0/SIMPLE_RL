import json

import pytest

from simple.grasp_rl.curriculum import load_curriculum


def _write_curriculum(path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phases": [
                    {
                        "name": "rsi_heavy",
                        "start_update": 0,
                        "rsi_probability": 0.7,
                        "target_mix": {
                            "uniform": 0.5,
                            "hard": 0.35,
                            "native": 0.15,
                        },
                        "rsi_stage_weights": {
                            "pregrasp": 0.6,
                            "grasp_to_lift": 0.4,
                        },
                        "domain_randomization": {
                            "target_mass_scale": [0.85, 1.15],
                            "friction_scale": [0.8, 1.2],
                            "manipulation_action_noise_std": 0.01,
                            "action_delay_max_steps": 1,
                        },
                    },
                    {
                        "name": "full_start_heavy",
                        "start_update": 100,
                        "rsi_probability": 0.25,
                    },
                ],
            }
        )
    )


def test_curriculum_lookup_and_hashes_are_reproducible(tmp_path) -> None:
    path = tmp_path / "curriculum.json"
    _write_curriculum(path)
    first = load_curriculum(path)
    second = load_curriculum(path)

    assert first.phase_for_update(0).name == "rsi_heavy"
    assert first.phase_for_update(99).name == "rsi_heavy"
    assert first.phase_for_update(100).name == "full_start_heavy"
    assert first.source_sha256 == second.source_sha256
    assert first.resolved_sha256 == second.resolved_sha256
    assert first.phases[0].target_mix == {
        "uniform": 0.5,
        "hard": 0.35,
        "native": 0.15,
    }


def test_curriculum_rejects_invalid_domain_randomization(tmp_path) -> None:
    path = tmp_path / "curriculum.json"
    _write_curriculum(path)
    document = json.loads(path.read_text())
    document["phases"][0]["domain_randomization"]["action_delay_max_steps"] = 2
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="must be 0 or 1"):
        load_curriculum(path)


def test_curriculum_requires_sorted_phase_boundaries(tmp_path) -> None:
    path = tmp_path / "curriculum.json"
    _write_curriculum(path)
    document = json.loads(path.read_text())
    document["phases"][1]["start_update"] = 0
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="unique and sorted"):
        load_curriculum(path)
