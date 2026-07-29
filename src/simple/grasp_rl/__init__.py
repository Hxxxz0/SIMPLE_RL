"""SMP-guided full-trajectory grasp reinforcement learning for SIMPLE.

The package deliberately keeps three contracts separate:

* an unconditional diffusion prior over executed robot motion;
* a task-conditioned PPO actor that emits a complete 36-D tracker command; and
* the existing SIMPLE AMO tracker that executes the command in MuJoCo.
"""

import os

# Training and evaluation normally run on headless GPU nodes.  This executes
# before any grasp_rl submodule can import MuJoCo, while still allowing callers
# to opt into GLFW or OSMesa explicitly.
os.environ.setdefault("MUJOCO_GL", "egl")

from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    MOTION_FEATURE_DIM,
    MOTION_FRAME_DIM,
    MOTION_WINDOW,
)

__all__ = [
    "ACTION_DIM",
    "ACTOR_OBS_DIM",
    "MOTION_FEATURE_DIM",
    "MOTION_FRAME_DIM",
    "MOTION_WINDOW",
]
