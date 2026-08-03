# Grasp RL pipeline

## Reproducible mjlab GPU PPO release

The self-contained `xmove_pick` CUDA release is
[`releases/grasp_rl/mjlab_gpu/xmove_pick/v1`](../../../releases/grasp_rl/mjlab_gpu/xmove_pick/v1).
It includes locked dependencies, frozen assets/reference data, audited PPO
checkpoints, paired full-DR evaluation, successful NPZ trajectories and
full-robot/close-up videos.  Across seeds 42 and 43, selected PPO and the
noise-matched proposal-only baseline both complete 173/256 worlds; PPO raises
mean task return from 24.124 to 24.264.  This is a no-regression release, not a
claim of statistically significant success-rate improvement.

## Current GRAIL-v7 PPO release

The current selected real-PPO checkpoints are:

```text
outputs/grasp_rl/xmove_pick/ppo_reference_grail_v7_fastreset_300/model_200.pt
outputs/grasp_rl/xmove_bend_pick/ppo_reference_grail_v7_fastreset_300/model_180.pt
```

Both training runs completed through `model_299.pt`; selection is based on
paired validation rather than last-iteration convention.

They are 842D plan-conditioned actors that emit the final complete 36D Sonic
tracker command.  On 100 exact-state pairs at `0.04,0.05` metre target jitter
and `0.25` rad yaw jitter, xmove improves from reference-only 32% to 49%
(17 rescues, zero regressions, exact `p=1.53e-5`); bend improves from 43% to
54% (12 rescues, one regression, exact `p=0.00342`).  Reports are
`paired_model200_100.json` and `paired_model180_100.json` below their training
directories.

The paired baseline must use `--reference-action-override all`.  V2 action
proposals begin after the 331D base observation, not the legacy 192D base.
`compare-paired` verifies exact equality of initial qpos/qvel, target pose and
reference IDs before calculating the exact McNemar/binomial test:

```bash
PYTHONPATH=src .venv/bin/python -m simple.grasp_rl.cli compare-paired \
  --policy-evaluation path/to/eval_pair100_ppo \
  --reference-evaluation path/to/eval_pair100_reference \
  --output path/to/paired.json
```

The grasp task reward uses GRAIL's released `5 grasp + 10 finger_direction -
15 approach_velocity - hand_table - 0.1 residual_rate` coefficients.  Robot
tracking is additive, both terms are scaled by the 50 Hz `0.02` interval, and
the episode ends immediately after the final reference command.  Fast reset
restores MuJoCo, Sonic controller interpolation/history and deterministic
control time; repeated 20-step rollouts from one snapshot are bit-exact.

Filtered production at the training envelope (`0.04,0.05`, `0.25`) produced
20 finite successful trajectories in 28 targets for xmove and 24 targets for
bend.  At the standard (`0.025,0.03`, `0.15`) envelope, bend produced 20/21
targets (95.24%); xmove's bounded 22-target run produced only 18/22 and is
retained as an incomplete negative result rather than reported as 90%.

## Multi-task v2: complete-command policies

The v2 path extends the same BC -> PPO stack to the 14 task descriptions in
`data/simple/simple_merged_v30`.  Policies are still trained **per task**; the
task catalogue, role-based observation, ordered reward graph, audits and
controller interface are shared.

```text
331D task_privileged_v2 + 511D retrieved complete plan
    -> 842D PlanConditionedMLPModel (512, 256, 128)
    -> complete normalized 36D command (never a hidden external residual)
    -> per-task ActionTransform
    -> AMO tracker or Sonic decoupled-WBC
```

The 331D state is robot/history 132D, two hands 30D, three masked entity roles
57D, contact/geometry relations 89D, two articulation slots 8D and task-stage
context 15D.  `primary`, `destination` and `auxiliary` mean the currently
manipulated object/link, its receptacle/goal and an optional second object.
MuJoCo supplies their current state.  No object-reference trajectory is used
by the task reward.

For v2 plan-conditioned PPO, the internal correction is structurally limited
to right arm/hand dimensions: fingers through approach/grasp/lift and the arm
through transport.  At place and release the output is exactly the plan command.  This keeps policy
output and PPO likelihood 36D while preventing a grasp curriculum from erasing
the already-audited walking/turning/placement tail.

`GoalGraphReward` composes approach, bilateral grasp, lift, transport,
place/release/settle, handover, articulation, push and compound-container
stages.  It rewards only newly achieved potential and stage events, so an
unchanged contact cannot farm reward.  Final success must hold in the current
physical state.  The six task families are `grasp`, `place`, `handover`,
`articulation`, `push` and `compound`.

The processed parquet `states` vector has only 32 policy-state values, not a
full MuJoCo snapshot.  Therefore object/contact observations are reconstructed
by replaying the complete commands in the matching controller backend.  Source
files remain immutable and every derived manifest stores source SHA256 and
repair provenance.

### Verified Sonic movement tasks

Registration is not a trained-policy claim. As of 2026-07-31, two additional
Sonic tasks have completed replay, reward audit, training, held-out evaluation
and randomized-target evaluation:

| task | source dataset | replay-success only | test | randomized target |
| --- | --- | ---: | ---: | ---: |
| `xmove_pick` | `G1WholebodyXMovePickTeleop-v0` | 72/99 | 8/8 | 18/20 (90%) |
| `xmove_bend_pick` | `G1WholebodyXMoveBendPickTeleop-v0` | 24/100 | 3/3 | 18/19 (94.74%) |

The randomized screens use x/y/yaw jitter. `xmove_pick` uses nearest plan rank
0 at +/-2.0 cm, +/-2.5 cm and +/-0.12 rad; `xmove_bend_pick` uses the current
scene's base plan at +/-1.5 cm, +/-2.0 cm and +/-0.10 rad. These are
single-rollout rates, not unions across retries. Fixed results use the
manifest's independent `test` split (`--evaluation-split test`).

Both rewards require force-verified grasp followed by the native object-height
predicate for 13 consecutive 50 Hz commands: 9 cm for `xmove_pick` and 8 cm for
`xmove_bend_pick`. Entering the final lift stage is not success. Both audits are
expert/repeat 10/10 and 0/10 for no-motion, halfway hold, open hand, contact
hold, release-after-lift, time shuffle and throw. Artifacts are:

```text
outputs/grasp_rl/xmove_pick/reward_audit/reward_audit_v2.json
outputs/grasp_rl/xmove_bend_pick/reward_audit/reward_audit_v2.json
```

Preparation runs the official Sonic clock at 5 ms physics steps with four
substeps per 50 Hz command and reuses its controller across resets. Only replay
successes enter the splits. `successful_replay_cover_v1` covers every accepted
physical command: sequential `encode -> decode(previous=...)` has maximum error
`5.96e-8` for both datasets. This fixes the old quantile slew limit, which
could clip an intentional gripper transition by about 1.5 rad.

Reproduce preparation, audit and BC training by substituting either task:

```bash
task=xmove_pick  # or xmove_bend_pick
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m simple.grasp_rl.cli prepare \
  --task "$task" --workers 8
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m simple.grasp_rl.cli audit-reward \
  --task "$task" --episodes 10
PYTHONPATH=src .venv/bin/python -m simple.grasp_rl.cli pretrain-actor \
  --task "$task" --reference-conditioning --plan-conditioned \
  --epochs 200 --batch-size 2048 \
  --output "outputs/grasp_rl/$task/bc_plan_v2_reversible"
```

The published checkpoints are:

```text
outputs/grasp_rl/xmove_pick/bc_plan_v2_reversible/best.pt
outputs/grasp_rl/xmove_bend_pick/bc_plan_v2_reversible/best.pt
```

They are deterministic plan-conditioned BC policies emitting the final full
36D normalized tracker command. Sonic decoupled-WBC remains the low-level
tracker. PPO and diffusion/SMP are not used for these published checkpoints.

A split-safe fixed test and randomized screen are run as follows:

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m simple.grasp_rl.cli evaluate \
  --task xmove_pick \
  --checkpoint outputs/grasp_rl/xmove_pick/bc_plan_v2_reversible/best.pt \
  --output outputs/grasp_rl/xmove_pick/eval_test_reversible \
  --evaluation-split test --episodes 100

PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m simple.grasp_rl.cli evaluate \
  --task xmove_pick \
  --checkpoint outputs/grasp_rl/xmove_pick/bc_plan_v2_reversible/best.pt \
  --output outputs/grasp_rl/xmove_pick/eval_random20_nearest \
  --evaluation-split train --episodes 20 --randomize-target \
  --target-position-jitter-xy 0.02,0.025 --target-yaw-jitter 0.12 \
  --reference-rank 0 --no-reference-base-episode
```

`collect-policy` first restricts base scenes to the processed manifest's
replay-success episode IDs, then writes only verified policy successes to
`trajectories/`. Every failed target and same-state plan retry remains in
`manifest.jsonl`:

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m simple.grasp_rl.cli collect-policy \
  --task xmove_bend_pick \
  --checkpoint outputs/grasp_rl/xmove_bend_pick/bc_plan_v2_reversible/best.pt \
  --output outputs/grasp_rl/xmove_bend_pick/policy_dataset_random20_replay_gated_v1 \
  --successes 20 --max-attempts 22 --scene-hold-attempts 1 \
  --target-position-jitter-xy 0.025,0.03 --target-yaw-jitter 0.15 \
  --reference-ranks 0,1 --base-reference-fallback --base-reference-first \
  --seed 20260731
```

That replay-gated production run completed 20/21 targets (95.24%, 28 plan
rollouts). The corresponding `xmove_pick` complete run at the same larger
jitter envelope completed 20/24 (83.33%, 35 plan rollouts) under
`policy_dataset_random20_replay_gated_complete_v1`; its earlier 18/20 random
screen must not be presented as a stable 90% production rate. Both datasets
contain exactly 20 finite NPZs with 331D state, 842D policy input, 36D
normalized/physical command and 80D motion frame.

Closed-loop randomized success videos are present in the processed-data tree:

```text
data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2/videos/random_policy_episode_000002_success.mp4
data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2/videos/random_policy_episode_000004_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/random_policy_episode_000007_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/random_policy_episode_000014_success.mp4
```

All four metadata files say `closed_loop=true` and `success=true`; `ffprobe`
reports H.264, yuv420p, 640x360 at 50 fps. `render --camera auto` selects the
legacy front camera or Sonic head-left camera as appropriate. The inspected
final-frame contact sheet is `data/grasp_rl/movement_policy_video_final_frames.png`;
it shows stage 2, grasp 1 and 11.1/12.6/19.8/12.5 cm lift.

Explicit OOD evaluation is available through `evaluate --target-position-xy`
and `--robot-position-xy`. Unlike random jitter, these flags make it possible
to place an object or robot provably outside the source-pose bounding box. The
current joint OOD results include 4/4 for `xmove_pick` at robot `(-0.78, 0)` and
object `(-0.28, -0.06)`, although the dataset robot is always `(-0.8, 0)` and
object x never exceeds `-0.29045`. `xmove_bend_pick` is 4/4 at robot
`(-0.95, 0.02)` and object `(-0.275, -0.025)`, while dataset robot y is always
zero and object y never exceeds `-0.04005`. Other directions range from 0/4 to
3/4, so this demonstrates bounded, asymmetric interpolation/extrapolation, not
universal planning. See `outputs/grasp_rl/ood_initial_pose_analysis.json`.

Checkpoint inspection shows that both published actors have an exactly zero
final correction layer (maximum absolute weight and bias are both 0). Therefore
their effective action is exactly the current 36D command from the retrieved
training plan. They are not replaying an expert trajectory recorded for the
modified test scene, but they are still retrieval-and-playback systems rather
than learned feedback or RL policies. The OOD results measure local plan/tracker
robustness; they do not establish goal-conditioned trajectory generation.

### Real PPO movement-task experiment (2026-07-31)

The retrieval checkpoints above remain labelled BC/playback. Four separate
online PPO runs were completed for the two Sonic tasks: a reference-conditioned
842D actor and a state-only 331D actor for each task. Every final checkpoint
has `iter=99` (updates 0--99), actor and critic state, and a two-group Adam
optimizer state. PPO changed the zero reference-correction layer; for example,
its maximum absolute weight is `1.54e-4` in `xmove_pick/model_10.pt` and
`2.27e-4` in `xmove_bend_pick/model_20.pt`.

The deterministic results below use full starts. The random screen uses the
same first 20 replay-gated scenes for every row and independent x/y/yaw target
jitter of `0.025,0.03,0.15`:

| task / checkpoint | held-out test | random target, one rollout |
| --- | ---: | ---: |
| `xmove_pick/bc_plan_v2_reversible/best.pt` | 8/8 | 15/20 |
| `xmove_pick/ppo_reference_v1_100/model_10.pt` | 8/8 | 14/20 |
| `xmove_pick/ppo_reference_v1_100/model_99.pt` | 8/8 | 15/20 |
| `xmove_pick/ppo_state_v1_100/model_99.pt` | 0/8 | not promoted |
| `xmove_bend_pick/bc_plan_v2_reversible/best.pt` | 3/3 | 16/20 |
| `xmove_bend_pick/ppo_reference_v1_100/model_20.pt` | 3/3 | 16/20 |
| `xmove_bend_pick/ppo_reference_v1_100/model_99.pt` | 3/3 | 16/20 |
| `xmove_bend_pick/ppo_state_v1_100/model_99.pt` | 0/3 | not promoted |

The stronger causal check fixes one base state and one reference for all 100
rollouts, randomizes only target x/y by `0.04/0.05 m` and yaw by `0.25 rad`,
and uses exactly paired robot qpos/qvel and target poses:

| task | reference only | reference + PPO | PPO-only | reference-only | exact paired p |
| --- | ---: | ---: | ---: | ---: | ---: |
| `xmove_pick`, base/reference 82 | 59/100 | 59/100 | 0 | 0 | 1.0 |
| `xmove_bend_pick`, base/reference 96 | 77/100 | 79/100 | 2 | 0 | 0.5 |

All 100 xmove outcomes are identical. Bend PPO uniquely rescues repeats 18
and 67 but the two-sided exact McNemar/binomial result is not significant.
Mean absolute changes to the physical 36D command are only `8.78e-5` for
xmove and `5.15e-5` for bend (maxima `1.34e-3` and `1.87e-3`). The summaries
are stored in each checkpoint's
`eval_fixed_reference*_random100_large/summary.json` directory.

Thus the reference PPO policies genuinely succeed, but this controlled test
does not establish a repeatable improvement over the reference-only baseline;
most success is attributable to the complete plan and Sonic tracker tolerance.
The state-only MLP PPO
and an additional state-only GRU BC actor both score zero from full starts, so
they are retained as negative ablations rather than relabelled as successful
policies.

Both selected reference PPO checkpoints score 4/4 on the explicit joint
robot/object OOD poses documented above. Production with identical-state
fallback across the base plan and ranks 0 through 4 collected 20 finite closed-
loop PPO trajectories per task:

| task / PPO checkpoint | target success | plan rollouts |
| --- | ---: | ---: |
| `xmove_pick/model_10.pt` | 20/24 (83.33%) | 50 |
| `xmove_bend_pick/model_20.pt` | 20/20 (100%) | 30 |

The production roots are
`outputs/grasp_rl/xmove_pick/ppo_reference_production20_v1` and
`outputs/grasp_rl/xmove_bend_pick/ppo_reference_production20_v1`. Each NPZ has
finite 331D state, 842D policy input, 36D normalized/physical complete command,
and 80D motion frame arrays. Verified RL videos are:

```text
data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2/videos/ppo_reference_model10_random_episode000002_success.mp4
data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2/videos/ppo_reference_model10_production_episode000000_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/ppo_reference_model20_random_episode000007_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/ppo_reference_model20_production_episode000000_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/paired_fixed_reference96_repeat67_ppo_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/paired_fixed_reference96_repeat67_reference_failure.mp4
```

Their sidecar metadata records `closed_loop=true`, the actual success flag, and
the corresponding checkpoint. `ffprobe` reports H.264/yuv420p, 640x360 at 50 fps. Resume now
constructs the decoupled actor/critic optimizer groups before loading optimizer
state, so interrupted runs retain Adam moments instead of silently restarting
the optimizer.

The final two videos are the exact-state repeat-67 pair: PPO lifts 14.78 cm and
succeeds, while the zero-correction reference lifts only 0.64 cm and fails.
Rendering restores the saved full v2 `initial_qpos/qvel`, so these closed-loop
videos reproduce the evaluation metrics instead of reconstructing only the
object pose.

An additional 50-update `xmove_pick` refinement increased the task weight from
0.05 to 0.1, used a five-times larger actor learning rate, reduced the reference
weight to 0.0002, and anchored to `model_99`. `model_148` still scores 8/8 on
the held-out test, but random-target results are 15/20 for `model_110` and 14/20
for `model_140` and `model_148`; none beats the original `model_99` at 15/20.
Likewise, expanding identical-state production fallback to ranks 0--10 yields
only 18/22 targets (81.82%, 72 full-plan rollouts). These artifacts remain
negative ablations under `ppo_reference_refine_v2_50` and
`ppo_reference_model99_production20_rank10_v1` and are not promoted.

The first complex production task is cross-table pick-and-place:

```bash
PYTHONPATH=src python -m simple.grasp_rl.cli repair-data \
  --task locomotion_pick_between_tables \
  --output data/grasp_rl/G1WholebodyLocomotionPickBetweenTablesMixed-v0/repaired

PYTHONPATH=src python -m simple.grasp_rl.cli prepare \
  --task locomotion_pick_between_tables --workers 8

PYTHONPATH=src python -m simple.grasp_rl.cli audit-reward \
  --task locomotion_pick_between_tables --episodes 100

PYTHONPATH=src python -m simple.grasp_rl.cli pretrain-actor \
  --task locomotion_pick_between_tables --reference-conditioning \
  --plan-conditioned --device cuda:5
```

The accepted replay and reward gates for the current prepared set are 50/50
executable demonstrations and 10/10 repeated expert successes.  Every audited
counterfactual (`no_motion`, truncated/held motion, open hand, early release,
time shuffle and teleport/throw) is 0/10.  The deterministic plan actor succeeds
on 10/10 recorded initial scenes.  On the same 50 independently randomized
targets (x +/-2.5 cm, y +/-3 cm, yaw +/-0.15 rad), the base plan succeeds on
37/50 and rank 0 on 28/50.  Trying the base plan and ranks 0 through 9 from an
identical initial robot/object/MuJoCo state covers 46/50 targets (92%).  This is
target-level multi-plan fallback success, not a single-rollout or PPO score.

The randomized PPO curriculum uses the audited, repaired `physical_actions` for
stage initialization.  The latest conservative run samples 90% of episodes near
the replay-detected pregrasp boundary and 10% from complete initial state, then
moves the primary object before policy control.  Task potential/terminal reward
and the robot/command reference score are added; unconditional diffusion is
disabled:

```bash
PYTHONPATH=src MUJOCO_GL=egl python -m simple.grasp_rl.cli train \
  --task locomotion_pick_between_tables \
  --processed data/grasp_rl/G1WholebodyLocomotionPickBetweenTablesMixed-v0/v2 \
  --output outputs/grasp_rl/locomotion_pick_between_tables/ppo_stage_mask_v3_50 \
  --actor-warm-start outputs/grasp_rl/locomotion_pick_between_tables/bc_plan_v2_smoke/best.pt \
  --reference-processed data/grasp_rl/G1WholebodyLocomotionPickBetweenTablesMixed-v0/v2 \
  --reference-reward-weight 0.001 --task-reward-weight 0.05 \
  --reward-audit outputs/grasp_rl/locomotion_pick_between_tables/reward_audit_v2_dropfix/reward_audit_v2.json \
  --rsi-dataset data/simple/G1WholebodyLocomotionPickBetweenTablesMixed-v0 \
  --rsi-stage pregrasp --rsi-probability 0.9 --rsi-randomize-target \
  --target-position-jitter-xy 0.025,0.03 --target-yaw-jitter 0.15 \
  --variant task_only --num-envs 24 --iterations 50 \
  --action-std 0.001 --manipulation-action-std 0.015 \
  --learning-rate 0.0001 --actor-lr-scale 0.02 --learning-schedule fixed \
  --freeze-actor-normalizer \
  --teacher-anchor-checkpoint outputs/grasp_rl/locomotion_pick_between_tables/bc_plan_v2_smoke/best.pt \
  --teacher-anchor-weight 0.01
```

Do not select a checkpoint from training success logs alone.  Run deterministic
full-start evaluation on randomized objects; keep the best checkpoint only if it
also preserves at least 90% on non-jittered scenes.  `collect-policy` stores only
successful trajectories below `trajectories/` but records every failed attempt
and every reference-plan fallback in `manifest.jsonl`.

That selection gate rejects both checkpoints from the run above: model 25 and
model 49 preserve 10/10 fixed-scene success but each reaches only 5/10 on the
large-jitter random screen, versus 7/10 for the BC initialization.  They remain
documented negative ablations and are not production policies.  Production uses
the BC plan actor with same-state multi-plan fallback.  The 11-plan fixed-50
evaluation is 46/50 (92% target success); its best single plan is only 37/50, so
the two metrics must not be conflated.

For data production, try the high-coverage base plan first and then the measured
complementary neighbors.  A target is never resampled between plan attempts:

```bash
PYTHONPATH=src python -m simple.grasp_rl.cli collect-policy \
  --task locomotion_pick_between_tables \
  --checkpoint outputs/grasp_rl/locomotion_pick_between_tables/bc_plan_v2_smoke/best.pt \
  --successes 20 --max-attempts 30 \
  --output outputs/grasp_rl/locomotion_pick_between_tables/policy_dataset_random20_v1 \
  --target-position-jitter-xy 0.025,0.03 --target-yaw-jitter 0.15 \
  --reference-ranks 6,4,5,2,0,3,8,1,7,9 \
  --base-reference-fallback --base-reference-first
```

The independent seed-20260730 production run completed 20 successes in 22
random target attempts (90.91% target success) using 52 complete-plan rollouts.
The base plan solved 14 targets; ranks 6, 4 and 5 rescued 3, 2 and 1 targets.
Both final failures, including all eleven failed plan attempts for each, remain
in `manifest.jsonl`.  All 20 saved NPZ files were checked for finite values and
the exact 331D base observation, 842D policy observation, 36D raw/physical
command and 80D motion-frame schemas.

The historical cross-table exporter zeroed `turning_flag` and `target_yaw`.
`cross_table_yaw_v1` restores only dimensions 34/35.  Since the old MP command
also stops while still holding the object short of table two, preparation adds
a manifest-labelled, half-weight closed-loop completion segment: approach the
container, settle locomotion, open the right hand and verify 13 stable frames.
This derived segment never overwrites `data/simple`.

The remaining sections document the frozen v1 Tabletop/BendPick experiments.

This package implements the design and commands documented in
[`FULL_TRAJECTORY_GRASP_RL_PLAN.md`](../../../FULL_TRAJECTORY_GRASP_RL_PLAN.md).

Core contracts are fixed in `schema.py`: a 192D task-conditioned PPO observation,
a complete 36D SIMPLE tracker command, and a 10×82D executed-motion window for a
frozen unconditional diffusion prior. The diffusion model is not the policy and
does not receive object or goal conditions.

## Current production design

The released path in this branch is a **retrieval-plan-conditioned full-command
policy**:

```text
192D online MuJoCo state + 401D retrieved future plan
    -> 593D PlanConditionedMLPModel
    -> complete normalized 36D tracker command
    -> task ActionTransform
    -> SIMPLE AMO whole-body tracker
```

The 192D state layout is:

```text
q[43], qdot[43], projected_gravity[3], pelvis_velocity[6], pelvis_height[1],
previous_complete_command[36], object_in_pelvis[3+6], object_velocity[3+3],
object_in_hand[3+6], table_in_pelvis[3+6], hand_contact_forces[24], goal_delta[3]
```

`ReferenceLibrary` retrieves a replay plan by current object/hand/table geometry.
At future control offsets `0,5,...,45`, each context frame contains a complete
36D command, the replayed object-position delta relative to the current object,
and a bilateral-contact label. Ten 40D frames plus phase give 401D. The first
command is the proposal and the MLP predicts a state-feedback correction:

```text
complete_command = plan_command_t + correction_36d
```

The PPO distribution and log probability are defined over `complete_command`.
The tracker does not execute a second hidden residual. Raw source trajectories
need only contain commands and environment configs: replay through the real
tracker/MuJoCo constructs the state, object delta, and contact context.

This distinction is important: the current system is not target-conditioned
diffusion and not reference-free. It uses a processed replay library at runtime.
It also does not use an object-reference tracking reward; physical success is
computed solely from the current simulated object/contact state and task goal.
The frozen unconditional SMP diffusion remains an optional reward ablation and
is disabled in the selected BendPick run.

The output layout is fixed by `ACTION_SLICES`: left/right hands (14D), left/right
arms (14D), torso RPY (3D), base height (1D), torso vx/vy (2D), turning flag (1D),
and target yaw (1D). Each task has its own demonstrated action transform; task,
schema, and transform hash are recorded in checkpoints.

The main experimental path is PPO warm-started from the real-trajectory BC/DAgger
actor. `task_spec.py` keeps scene construction, reset bounds, success semantics,
and checkpoint compatibility task-specific while preserving the shared 192/593D
input and complete 36D output contract. `grail_release_v1` is the current task reward: it maps the released GRAIL
`pnp_table` grasp, finger-direction, approach-velocity, table-contact and action-rate
terms to SIMPLE's MuJoCo signals. `dense_v1`, `progress_v2`, and `grail_v3` remain
as explicit reward ablations.

Run the CLI from the repository root with `PYTHONPATH=src`:

```bash
python -m simple.grasp_rl.cli --help
python -m simple.grasp_rl.cli list-tasks
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
python -m simple.grasp_rl.cli collect-policy --help
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
single-finger reward farming. Tabletop success is a 2 cm bilateral lift held for
13 frames. BendPick success is its native 9 cm lift with the same hold; its valid
demonstrations make early contact while bending, so it intentionally has no
40-frame stalled-contact termination. Run `audit-reward` before PPO to
compare exact expert replay against stationary, truncated, open-hand, stalled-grasp,
and post-lift-release counterfactuals.

For the first extension task, add `--task bend_pick` to every command. Its processed
data lives in `data/grasp_rl/G1WholebodyBendPickMP-v0`; its passing audit is
`outputs/grasp_rl/bend_pick/reward_audit_v3/reward_audit.json`, and its exact-plan
complete-command BC initialization is
`outputs/grasp_rl/bend_pick/bc_plan_v1/best.pt`. The native 100-scene BC baseline is
97/100 at the strict 9 cm held-lift criterion. This model uses the same policy
class as SMP-style PPO, not a diffusion-policy architecture.

The selected BendPick PPO checkpoint is
`outputs/grasp_rl/bend_pick/ppo_native_v1_300/model_100.pt`. On 100 fixed native-
range randomized targets it achieves 90/100 with rank 0; trying rank 1 from the
same initial target/state raises the union to 91/100. A separate production seed
collected 100 successes in 106 target attempts (94.34%) at
`outputs/grasp_rl/bend_pick/policy_dataset_random_100_v1`. Its manifest includes
all six failures and all 118 plan rollouts; the trajectory directory alone must
not be used to claim a 100% policy success rate.

Training artifacts live under `outputs/grasp_rl`; processed schemas, splits and
normalizers live under `data/grasp_rl`. Use `--resume` for an exact continuation
and `--warm-start` when transferring PPO weights into a new curriculum or reward
experiment. Use `--actor-warm-start` for a BC actor; the loader follows chained
BC initializations to recover the nearest PPO critic. Evaluation supports
`--episode-offset` for disjoint parallel scene slices.

## Legacy Tabletop diagnostics

The following results explain design choices but are not the current BendPick
production score. On the earlier Tabletop task, a late-phase curriculum reached
43/100, while its full-start actor failed; uniform continuation and failure mining
did not improve it monotonically. A matched additive-SMP ablation gave 11/20
versus 12/20 for task-only. These experiments are why the selected production
path disables the unconditional diffusion reward and starts near a complete
retrieved command plan.

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
