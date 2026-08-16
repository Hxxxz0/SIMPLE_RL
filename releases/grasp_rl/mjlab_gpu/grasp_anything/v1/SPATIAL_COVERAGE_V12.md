# Apple exact-cell spatial PPO v12

This document records the Apple follow-up that fixes spatial PPO accounting and
targets positions where the previous policy had few or no successes. The v12
artifacts add an opt-in checkpoint router and three exact-cell specialists. They
do not replace any existing checkpoint or establish uniform success over the
full workspace.

Success is the native physical gate used throughout this release: the free and
unattached Apple body must be grasped, lifted by at least 0.09 m, and held for
13 control steps. Contact, finger closure, shaped reward, reference agreement,
and a successful video from a neighboring position do not count.

## What was fixed

The changes are opt-in and preserve old command and checkpoint behavior.

- Stratified target-position sampling assigns one world to every grid cell,
  then assigns the remaining worlds to explicitly focused cells. A focused run
  now fails closed when `num_envs` is not greater than the grid cell count.
- Evaluation reports `stratified_target_position_grid` from the randomizer's
  exact cell IDs. It no longer reconstructs those cells from the observed
  sample minimum and maximum.
- Spatial PPO advantage normalization has a new `sample` weighting. The legacy
  `cell` default gives every cell equal aggregate weight. `sample` preserves
  the intended extra PPO weight from focused-cell oversampling.
- The bootstrap gate can use `--bootstrap-gate-spatial-scope focus`, which
  requires every requested focus cell to meet the threshold instead of using a
  global average that can hide a zero-success cell.
- Evaluation has `--minimum-focus-cell-success-rate` for exact-cell acceptance.
- Multi-checkpoint evaluation, collection, and recording can route explicitly
  listed grid cells to specialist checkpoints. The main checkpoint remains the
  default everywhere else.
- Evaluation constructs all routed actors before the final DR reset. Adding an
  expert therefore cannot change the physical worlds in a same-seed A/B test.
- Recording has `--maximum-precontact-target-motion-m`. The legacy default is
  still 0.003 m; v12 release videos use the strict 0.0001 m limit.

The old defaults remain `cell` advantage weighting, all-world bootstrap gating,
no stratified grid, no checkpoint router, and the 0.003 m video threshold. Old
weights do not need conversion.

## Spatial contract

The v12 experiment uses a 32 x 32 stratified grid over target X/Y jitter of
`+/-0.05 m` around center `(0.0, -0.09 m)`. The sampled rectangle is therefore
X `[-0.05,+0.05] m` and Y `[-0.14,-0.04] m`. A cell ID is
`x_index * 32 + y_index`.

The three newly specialized frontier cells are only 3.125 mm wide per axis:

| Cell | ID | X bounds | Y bounds |
| :--- | ---: | :--- | :--- |
| `(29,31)` | 959 | `[+0.040625,+0.043750] m` | `[-0.043125,-0.040000] m` |
| `(30,30)` | 990 | `[+0.043750,+0.046875] m` | `[-0.046250,-0.043125] m` |
| `(31,29)` | 1021 | `[+0.046875,+0.050000] m` | `[-0.049375,-0.046250] m` |

These are target translations under the episode-11 asset contract. They are
not `+/-20 cm` edge positions. The existing strict uniform `+/-20 cm` results
remain sparse, and robust `+/-20 cm` support is not claimed.

## Training contract

All v12 lines use fresh on-policy RSL-RL PPO rollouts, full DR strength from the
first reset, 120 steps per environment per update, no rollout reuse, 3 learning
epochs, 4 minibatches, and 12 optimizer steps per update. The coarse and
four-cell lines use 1024 environments, or 122,880 transitions per update. Each
exact-cell specialist uses 2048 environments, or 245,760 transitions per
update.

All five training lines completed 100 updates. The complete training budget is:

| Line | Environments | Updates | Fresh transitions |
| :--- | ---: | ---: | ---: |
| coarse | 1024 | 100 | 12,288,000 |
| shared four-cell | 1024 | 100 | 12,288,000 |
| specialist `(29,31)` | 2048 | 100 | 24,576,000 |
| specialist `(30,30)` | 2048 | 100 | 24,576,000 |
| specialist `(31,29)` | 2048 | 100 | 24,576,000 |
| total | | 500 | 98,304,000 |

Reference conditioning remains in the policy observation. It is not an
imitation objective: `reference_reward_weight=0`, the reference contact gate is
disabled, and `max_reference_action_deviation=2`. The enabled residual groups
are right hand, right arm, torso RPY, base height, base X velocity, and base Y
velocity. The policy can change arm direction instead of being restricted to a
small reference-action tube.

The released router contains these files:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_spatial_v12_router.json
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_spatial_v12_coarse_model_50.pt
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_spatial_v12_four_cell_model_99.pt
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_spatial_v12_cell29_31_model_5.pt
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_spatial_v12_cell30_30_model_5.pt
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_spatial_v12_cell31_29_model_5.pt
```

The router covers 13 explicitly measured frontier cells. Ten use the shared
four-cell-line checkpoint, and the three cells above use individual
specialists. All other cells fall back to the coarse checkpoint.

## Exact-cell cross-seed evaluation

Every exact-cell evaluation uses 2048 unique worlds: one world in each of the
1024 grid cells and the remaining 1024 worlds in the requested focus cell.
Thus the focus cell has 1025 trials. The deterministic policy is identical
between seeds; the seed only selects initial worlds and simulator streams and
is not a policy input.

| Specialist | Seed 20261634 | Seed 20261635 | Repeat gate | Measured claim |
| :--- | ---: | ---: | :--- | :--- |
| `(29,31)` model 5 | 5/1025 (0.49%) | 3/1025 (0.29%) | 3/1025 and 5/1025 at a 0.2% gate | new but very sparse reachability |
| `(30,30)` model 5 | 32/1025 (3.12%) | 36/1025 (3.51%) | 40/1025 and 41/1025; both pass 3% | repeatable low-rate reachability |
| `(31,29)` model 5 | 22/1025 (2.15%) | 22/1025 (2.15%) | 20/1025 and 23/1025; one misses 2% | new but threshold-unstable reachability |

The `(29,31)` initialized/warm policy scored 0/1025 on both seeds before the
specialist update. These results prove that corrected PPO can produce new
successful trajectories in an old zero-success cell. They do not prove stable
success at every position. In particular, `(29,31)` remains below 1% and must
not be described as supported in the ordinary robust sense.

Absolute exact-cell logs are under:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance
```

The relevant names begin with `focus29_31_model5_exact`,
`focus30_30_model5_exact`, and `focus31_29_model5_exact`.

## Routed frontier evaluation

The router is evaluated on the same 13-cell focus set with 2048 unique worlds.
This test measures aggregate frontier improvement and also reports each exact
cell. It is not accepted solely from the global average: the minimum focus-cell
rate is retained as a separate gate and conclusion.

The post-fix paired A/B table is recorded after actor construction and the
final DR reset, so coarse and routed rows with the same seed have identical
`initial_world_sha256` values.

| Seed | Coarse checkpoint | v12 router | Difference |
| :--- | ---: | ---: | ---: |
| 20261634 | 1070/2048 (52.25%) | 1256/2048 (61.33%) | +186 / +9.08 points |
| 20261635 | 1088/2048 (53.12%) | 1278/2048 (62.40%) | +190 / +9.28 points |
| aggregate | 2158/4096 (52.69%) | 2534/4096 (61.87%) | +376 / +9.18 points |

The paired world hashes are
`18e9de5db2e324e68131742ff524b3cd08a1907a751d1c054cfd587e1ff00df7`
for seed 20261634 and
`0b42cc02ab98263514082fbf1663c85284d37612a68ac70a0f711d33929062fc`
for seed 20261635. Within each seed, the coarse and routed policy-state and
proposal-context hashes also match exactly.

Absolute paired logs:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/coarse50_focus13_postloadreset_full_dr_seed20261634_2048.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/coarse50_focus13_postloadreset_full_dr_seed20261635_2048.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/routed_spatial_v12_postloadreset_focus13_full_dr_seed20261634_2048.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/routed_spatial_v12_postloadreset_focus13_full_dr_seed20261635_2048.log
```

Even when the router improves the aggregate, `(29,31)` can produce 0 successes
in an 80-world routed batch because its independently measured probability is
only about 0.3-0.5%. Therefore the routed policy is released as experimental
partial spatial coverage, not stable coverage of all 13 cells.

## Strictly audited videos

The three videos below report native simulator success, a passed right-finger
grasp audit, a free and physically unattached target, and exactly 0 m target
motion before first hand contact. Their strict allowed pre-contact displacement
is 0.0001 m.

| Cell | Recorded target XY | Max lift | Absolute video |
| :--- | :--- | ---: | :--- |
| `(29,31)` | `(+0.041382,-0.042383) m` | 0.2375 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_spatial_v12/cell_29_31/gpu_ppo_success_full_dr_01.mp4` |
| `(30,30)` | `(+0.043915,-0.046068) m` | 0.2343 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_spatial_v12/cell_30_30/gpu_ppo_success_full_dr_01.mp4` |
| `(31,29)` | `(+0.047666,-0.048883) m` | 0.2385 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_spatial_v12/cell_31_29/gpu_ppo_success_full_dr_01.mp4` |

The combined position-review directory contains the five older v11 videos and
the three v12 exact-cell videos with position-bearing names:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/video_review/apple_spatial_v12_positions
```

## Source training directories

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_targetsupport_v3_xy50mm_yaw0_focus77_sampleweighted_longrollout120_warm_focus11m300_anchor100_arm08_actorlr25_seed20261652_env1024_steps120_hold1_updates100
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_targetsupport_v3_xy50mm_yaw0_focus77_subgrid16x16_focus4_sampleweighted_longrollout120_warm_focus11m300_anchor100_arm12_actorlr25_seed20261654_env1024_steps120_hold1_updates100
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_targetsupport_v3_xy50mm_yaw0_subgrid32x32_focus29_31_sampleweighted_longrollout120_warm_four99_anchorfour99_arm12_actorlr25_seed20261661_env2048_steps120_hold1_updates100
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_targetsupport_v3_xy50mm_yaw0_subgrid32x32_focus30_30_phase3_sampleweighted_longrollout120_warm_focusm40_anchorfocusm40_arm16_actorlr25_seed20261660_env2048_steps120_hold1_updates100
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_targetsupport_v3_xy50mm_yaw0_subgrid32x32_focus31_29_sampleweighted_longrollout120_warm_four99_anchorfour99_arm12_actorlr25_seed20261662_env2048_steps120_hold1_updates100
```

Checkpoint selection is based on cross-seed exact-cell physical evaluation,
not on training reward and not on the last update. Every specialist directory
has 100 integrity records, and every final record reports on-policy collection,
no rollout reuse, 245,760 transitions, and 12 optimizer steps.

All three final `model_99` checkpoints regressed to 0/1025 in their requested
focus cell on both evaluation seeds, even though their global success counts
remained substantial:

| Final checkpoint | Seed 20261634 | Seed 20261635 | Selected release checkpoint |
| :--- | :--- | :--- | :--- |
| `(29,31)` model 99 | global 748/2048; cell 0/1025 | global 763/2048; cell 0/1025 | model 5 |
| `(30,30)` model 99 | global 721/2048; cell 0/1025 | global 734/2048; cell 0/1025 | model 5 |
| `(31,29)` model 99 | global 735/2048; cell 0/1025 | global 742/2048; cell 0/1025 | model 5 |

These are spatial policy regressions, not insufficient update counts. The
released `model_5` checkpoints are retained because selection uses exact-cell
cross-seed physical results, including independent strict repeats, instead of
the global average or final training update.

Absolute final-checkpoint logs:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/focus30_30_model99_longrun_final_exact_seed20261634_1025.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/focus30_30_model99_longrun_final_exact_seed20261635_1025.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/focus29_31_model99_longrun_final_exact_seed20261634_1025.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/focus29_31_model99_longrun_final_exact_seed20261635_1025.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/focus31_29_model99_longrun_final_exact_seed20261634_1025.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/focus31_29_model99_longrun_final_exact_seed20261635_1025.log
```
