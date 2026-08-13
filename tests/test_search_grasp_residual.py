from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts/grasp_rl/search_cup6_residual.py"
SPEC = importlib.util.spec_from_file_location("search_grasp_residual", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEARCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEARCH
SPEC.loader.exec_module(SEARCH)


def test_candidate_zero_retains_requested_object_specific_base() -> None:
    hand = tuple(value / 100.0 for value in range(7))
    arm = tuple(-value / 100.0 for value in range(7))
    candidates = SEARCH._candidate_corrections(
        16,
        seed=1,
        device="cpu",
        base_hand=hand,
        base_arm=arm,
    )

    assert candidates.shape == (16, 36)
    torch.testing.assert_close(
        candidates[0, list(SEARCH.HAND_INDICES)], torch.tensor(hand)
    )
    torch.testing.assert_close(
        candidates[0, list(SEARCH.ARM_INDICES)], torch.tensor(arm)
    )
    inactive = sorted(set(range(36)) - set(SEARCH.ACTIVE_INDICES))
    assert torch.count_nonzero(candidates[:, inactive]) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_hand": (0.0,) * 6}, "seven values"),
        ({"base_arm": (0.0,) * 8}, "seven values"),
        ({"hand_scale": 0.0}, "positive"),
        ({"arm_scale": -1.0}, "positive"),
    ],
)
def test_candidate_search_rejects_invalid_shape_or_scale(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SEARCH._candidate_corrections(16, seed=1, device="cpu", **kwargs)
