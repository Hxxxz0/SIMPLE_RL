# Grasp RL pipeline

This package implements the design and commands documented in
[`FULL_TRAJECTORY_GRASP_RL_PLAN.md`](../../../FULL_TRAJECTORY_GRASP_RL_PLAN.md).

Core contracts are fixed in `schema.py`: a 192D task-conditioned PPO observation,
a complete 36D SIMPLE tracker command, and a 10×82D executed-motion window for a
frozen unconditional diffusion prior. The diffusion model is not the policy and
does not receive object or goal conditions.

The main experimental path is PPO warm-started from the real-trajectory BC/DAgger
actor. `grail_release_v1` is the current task reward: it maps the released GRAIL
`pnp_table` grasp, finger-direction, approach-velocity, table-contact and action-rate
terms to SIMPLE's MuJoCo signals. `dense_v1`, `progress_v2`, and `grail_v3` remain
as explicit reward ablations.

Run the CLI from the repository root with `PYTHONPATH=src`:

```bash
python -m simple.grasp_rl.cli --help
python -m simple.grasp_rl.cli prepare --help
python -m simple.grasp_rl.cli prepare-bc --help
python -m simple.grasp_rl.cli audit-reward --help
python -m simple.grasp_rl.cli build-knn --help
python -m simple.grasp_rl.cli pretrain-actor --help
python -m simple.grasp_rl.cli collect-dagger --help
python -m simple.grasp_rl.cli pretrain --help
python -m simple.grasp_rl.cli train --help
python -m simple.grasp_rl.cli evaluate --help
python -m simple.grasp_rl.cli render --help
```

The adapted GRAIL task term is evaluated at 50 Hz:

```text
target = pregrasp_distance_kernel
       + 5 * link_contact_fraction
       + 10 * bilateral_finger_direction
       + 5 * contact_gated_lift
       + 2 * contact_gated_stability
penalty = 15 * distance_gated_wrist_speed_squared
        + clamp(hand_table_force, 0, 1)
        + 0.1 * normalized_action_rate
step_reward = 0.02 * (target - penalty) + terminal
```

Finger direction uses the MuJoCo object/finger contact center, matching GRAIL's
`use_contact_center=true`, and is gated by a simulated thumb–support grasp to avoid
single-finger reward farming. Success is a 2 cm bilateral lift held for 13 frames;
contact without lift for 40 frames is a failure. Run `audit-reward` before PPO to
compare exact expert replay against stationary, truncated, open-hand, stalled-grasp,
and post-lift-release counterfactuals.

Training artifacts live under `outputs/grasp_rl`; processed schemas, splits and
normalizers live under `data/grasp_rl`. Use `--resume` for an exact continuation
and `--warm-start` when transferring PPO weights into a new curriculum or reward
experiment. Use `--actor-warm-start` for a BC actor; the loader follows chained
BC initializations to recover the nearest PPO critic. Evaluation supports
`--episode-offset` for disjoint parallel scene slices.

The current single-object experiment is a verified feasibility baseline, not a
general grasping result. Exact expert replay succeeds in 99/100 scenes. The
current learned-policy curriculum reaches 43/100 from a fixed 90% demonstration
phase, but the same checkpoint is 0/100 from a full start. Another 50 uniform
PPO iterations give 42/100, and failure-scene mining gives 39/100, so these are
scene exchanges rather than a monotonic improvement. A matched additive-SMP
ablation gives 11/20 on the first late-phase slice versus 12/20 for task-only;
the frozen diffusion prior is therefore not used by the main policy.

A replay-derived stateless nearest-neighbor state-feedback teacher reaches
90/100 from full starts; adding same-trajectory local temporal continuity reaches
91/100. This is an all-data diagnostic/DAgger teacher, not the deployable SMP MLP
and not an RL result. It proves that the 192D MuJoCo observation, 36D tracker
command, reward termination, and SIMPLE execution path admit >90% success without
an object reference trajectory. Distillation and teacher-guided PPO still reach
only 1/20 and 0/20 respectively from full starts. The strongest teacher-guided
PPO ablation lifts the object by 2 cm in 16/20 but has no bilateral grasp, which
localizes the learned-policy failure to contact preservation rather than reach or
lift motion. Do not report the 91% teacher score as PPO success.

The successful teacher video is
`outputs/grasp_rl/videos/knn_temporal_episode0_success.mp4` (H.264/yuv420p), with
machine-readable success metadata in the adjacent JSON. Reward-audit and exact
checkpoint paths are recorded in the top-level plan.

The final wrist-velocity-corrected 100-scene audit passes every acceptance
check: expert replay is 99/100, all five counterfactual success rates are 0/100,
successful-expert minimum return is 25.43, maximum counterfactual return is
1.00, and repeat outcomes match in 100/100 scenes. See
`outputs/grasp_rl/reward_audit_grail_wrist_full100/reward_audit.json`.
