# Apple multi-region frontier PPO v11

This document records the Apple experiment that targets spatial coverage rather
than replay-like average success. One checkpoint is evaluated independently in
five target-position regions and selected by its worst region, with core-region
regression protection. The result is a measured improvement, not a claim of
stable success over the full `+/-20 cm` square.

Success is the native physical gate used by the rest of this release: the free
Apple body must be grasped, lifted by at least 0.09 m, and held for 13 control
steps. Contact, closure, shaped reward, and reference agreement do not count as
success.

## Backward-compatible runtime changes

The opt-in repeatable argument below adds a weighted rectangular mixture:

```text
--target-position-focus-region CENTER_X CENTER_Y JITTER_X JITTER_Y PROBABILITY
```

It can be repeated. Probabilities must be positive and sum to at most one; any
remainder keeps the original target-position distribution. The old single
focus option, all old defaults, old checkpoint metadata, and all existing
commands remain valid. Single-focus and multi-region focus are mutually
exclusive. Evaluation and video output names include region and checkpoint
hashes so different regions cannot silently overwrite one another.

The evaluation report now includes an 8 x 8 target-translation success grid at
`initial_pose_diagnostics.target_translation_xy.grid`.

## Training contract

All accepted-line training uses 1024 environments, 240 steps per environment
per PPO update, temporally held exploration noise with standard deviation 0.08
and hold length 4, full DR strength from the first reset, frozen actor
normalization, warm-started actor and critic, and an actor anchor. Every update
contains 245,760 fresh on-policy transitions. No rollout is reused.

The policy still receives the reference-conditioned observation. It is not
rewarded for imitation: `reference_reward_weight=0`. The normalized residual
limit is `max_reference_action_deviation=2`, so the arm and hand are not held to
a small reference-action tube.

| Run | Region mixture | Goal scale/negative clip | Updates | Fresh transitions | Result |
| :--- | :--- | :---: | ---: | ---: | :--- |
| v8 | five regions, 20% each | 5/0.25 | 20 | 4,915,200 | model 5 improved the old worst region; later models oscillated |
| v9 | right 60%, four others 10% | 5/0.25 | 10 | 2,457,600 | model 5 improved right; model 9 regressed |
| v10 | right 45%, left 30%, back 20%, core/front 2.5% | 5/0.25 | 6 | 1,474,560 | rejected in favor of v11 |
| v11 | same weak-region mixture as v10 | 20/1 | 6 | 1,474,560 | selected model 5 |

The complete four-run experiment contains 10,321,920 fresh transitions. The
selected checkpoint lineage contains 4,423,680 new transitions after the v5
warm start: v8 model 5, v9 model 5, and v11 model 5, each after six 240-step
updates.

Selected checkpoint:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_frontier_v11_model_5.pt
```

SHA256:

```text
d2e8dcf6c5ae6250b1bda34a16ad43551ef22db6b58e44f8a23bae78360e2170
```

## Region definitions

Coordinates are target translations in metres under the episode-11 Apple
contract.

| Region | Center XY | Symmetric jitter XY | Training probability in v11 |
| :--- | :--- | :--- | ---: |
| core | `(-0.020, -0.065)` | `(0.040, 0.055)` | 2.5% |
| left | `(-0.060, -0.065)` | `(0.025, 0.055)` | 30% |
| right | `(+0.025, -0.065)` | `(0.025, 0.055)` | 45% |
| back | `(-0.020, -0.115)` | `(0.040, 0.025)` | 20% |
| front | `(-0.020, -0.025)` | `(0.040, 0.025)` | 2.5% |

These rectangles overlap by design and cover the empirically reachable
frontier, approximately X `[-8.5,+5.0] cm` and Y `[-14,0] cm`. They are not the
uniform 40 x 40 cm `target_xy_200mm` square.

## Cross-seed physical evaluation

Each row is the same v11 checkpoint evaluated with two independent 512-world
seeds. Model selection uses the aggregate worst region, not the average across
regions.

The seed identifier is not a policy input. It only changes the sampled initial
worlds and stochastic simulator/randomization stream. No checkpoint selection,
fine-tuning, or per-seed policy switch occurs between the two evaluations, so a
stable-policy claim requires comparable success across both seeds.

| Region | Seed 1 | Seed 2 | Aggregate | Aggregate rate |
| :--- | ---: | ---: | ---: | ---: |
| core | 402/512 | 398/512 | 800/1024 | 78.12% |
| left | 185/512 | 175/512 | 360/1024 | 35.16% |
| right | 170/512 | 187/512 | 357/1024 | 34.86% |
| back | 190/512 | 221/512 | 411/1024 | 40.14% |
| front | 359/512 | 341/512 | 700/1024 | 68.36% |

The old v5 checkpoint's first-seed region results were core 375, left 183,
right 116, back 246, and front 248 out of 512. Its right-region aggregate over
the two directly repeated seeds was 264/1024 (25.78%). The v11 right aggregate
is 357/1024 (34.86%), an increase of 93 successes and 9.08 percentage points.
Core also remains above the old first-seed result.

The acceptance criterion is only partially met. The result is seed-repeatable
and raises the worst region, but the three weak regions remain below 50%.
Therefore this checkpoint is released as an additional frontier-improvement
policy; it does not replace the v5 `+/-1 cm` stable-core checkpoint.

Absolute evaluation directories:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/frontier_v11_evaluations/model_5
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/frontier_v11_evaluations/model_5_replication
```

## Strict +/-20 cm result

The final v11 checkpoint scored 44/1024 (4.30%) on seed 20261531 under uniform
independent X/Y jitter of `+/-0.20 m`. This remains sparse and does not establish
robust `+/-20 cm` support.

Absolute log:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/frontier_v11_evaluations/model_5_strict_xy200mm_seed20261531_1024.log
```

## Audited position videos

Five full-robot videos use the same checkpoint. The representative points were
chosen from successful cells in the independent 8 x 8 grids so the displayed
positions do not collapse into one small cluster. The left video uses a
`+/-3 mm` representative window inside the evaluated left rectangle because
two full-left recording batches produced no recordable success despite the
independent evaluation result. This does not change the region evaluation.

Every sidecar reports native simulator success, right-finger grasp, a free and
physically unattached target, and exactly 0 m of target motion before first hand
contact.

| Region | Recorded target XY | Max lift | Absolute video |
| :--- | :--- | ---: | :--- |
| core | `(-2.29,-7.29) cm` | 0.1772 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_frontier_v11_model_5/core/gpu_ppo_success_full_dr_01.mp4` |
| left | `(-4.96,-4.77) cm` | 0.1217 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_frontier_v11_model_5/left/gpu_ppo_success_full_dr_01.mp4` |
| right | `(+2.36,-4.17) cm` | 0.1643 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_frontier_v11_model_5/right/gpu_ppo_success_full_dr_01.mp4` |
| back | `(-1.60,-9.01) cm` | 0.2144 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_frontier_v11_model_5/back/gpu_ppo_success_full_dr_01.mp4` |
| front | `(-0.22,-0.71) cm` | 0.1341 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_frontier_v11_model_5/front/gpu_ppo_success_full_dr_01.mp4` |

The convenient review directory is:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/video_review/apple_frontier_v11_regions
```

## Source run directories

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v8_balanced_regions_longrollout_warmcritic_freezenorm_anchor10_warm_v5m100_seed20261406_env1024_steps240_std080_hold4_20
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v9_right60_longrollout_warmcritic_freezenorm_anchor20_warm_v8m5_seed20261407_env1024_steps240_std080_hold4_10
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v10_weakregions_right45_left30_back20_longrollout_warmcritic_freezenorm_anchor30_warm_v9m5_seed20261408_env1024_steps240_std080_hold4_6
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v11_weakregions_goal20_right45_left30_back20_longrollout_warmcritic_freezenorm_anchor30_warm_v9m5_seed20261409_env1024_steps240_std080_hold4_6
```
