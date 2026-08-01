"""Full-command multi-task reinforcement learning for SIMPLE.

The package deliberately keeps three contracts separate:

* one task policy emits a complete 36-D tracker command;
* role-based observations and ordered goal rewards are shared across tasks;
* AMO and Sonic decoupled-WBC execute the same public command layout.
"""

import os

# Training and evaluation normally run on headless GPU nodes.  This executes
# before any grasp_rl submodule can import MuJoCo, while still allowing callers
# to opt into GLFW or OSMesa explicitly.
os.environ.setdefault("MUJOCO_GL", "egl")

from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    ACTOR_OBS_V2_DIM,
    REFERENCE_ACTOR_OBS_V2_DIM,
    MOTION_FEATURE_DIM,
    MOTION_FRAME_DIM,
    MOTION_WINDOW,
)

__all__ = [
    "ACTION_DIM",
    "ACTOR_OBS_DIM",
    "ACTOR_OBS_V2_DIM",
    "REFERENCE_ACTOR_OBS_V2_DIM",
    "MOTION_FEATURE_DIM",
    "MOTION_FRAME_DIM",
    "MOTION_WINDOW",
]
