# Tomato and Potato stable-physics workspace PPO

This document records the extension of the accepted Apple stable-physics
residual PPO method to `Tomato_1`, `Potato_1`, and `Bowl_1`. Tomato and Potato
passed physical acceptance and are released. Bowl completed the same training
budget but failed checkpoint screening, so no Bowl workspace PPO is released.

## Accepted scope

Success is a native simulator grasp that lifts the object by at least 0.09 m
and holds the lift for 13 control steps. The core profile samples target X and
Y independently within `+/-1 cm` around `(x=0, y=-0.09 m)`, a 2 x 2 cm plane.

| Object | Selected checkpoint | Core confirmation | Paired reference | `+/-2.5 cm` generalization | Paired reference |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `Tomato_1` | `model_150` | 739/1024 (72.17%) | 310/1024 (30.27%) | 199/512 (38.87%) | 45/512 (8.79%) |
| `Potato_1` | `model_50` | 1024/1024 (100.00%) | 618/1024 (60.35%) | 429/512 (83.79%) | 135/512 (26.37%) |

The `+/-1 cm` rows are the accepted trained scope. The `+/-2.5 cm` rows are
one-seed evaluations outside the training distribution and are reported only
as measured generalization. Neither result is a claim of reliable operation
over tens of centimeters.

The paired checkpoint and reference evaluations used identical seeds and
identical initial-world hashes:

| Object/profile | Seed | Initial-world SHA-256 |
| :--- | ---: | :--- |
| Tomato `+/-1 cm` | 20261131 | `44361426f89a7cba6d7d97faa273c4ea22a8449380527f25f9ba2ee929b4ad5e` |
| Potato `+/-1 cm` | 20261132 | `eec42ff27c6e0f2d8f17c7af03acb1da6df920a4eb0e97a3d01066e6809d415a` |
| Potato `+/-2.5 cm` | 20261133 | `b144a774818722bee9259d7a10b4ae8f563e805054c3ce0b3150f66d61be13dd` |
| Tomato `+/-2.5 cm` | 20261134 | `fc74765f19bafe971fab7085c796d84b981ae753146c423c4974abcabf382342` |

## Training and selection

All three objects used the same scratch-training configuration:

```text
num_envs = 1024
updates = 200
steps_per_env_per_update = 24
transitions_per_update = 24,576
total_transitions_per_completed_run = 4,915,200
exploration_std = 0.08
exploration_hold_steps = 4
dr_initial_strength = 1
reference_reward_weight = 0
max_reference_action_deviation = 2
goal_potential_scale = 20
goal_potential_negative_clip = 1
success_bonus = 80
```

Every `ppo_integrity.jsonl` contains 200 fresh on-policy records and no reused
rollout. The selected checkpoint transition counts include the zero-indexed
checkpoint update: Tomato `model_150` contains 151 rollouts and 3,710,976
transitions; Potato `model_50` contains 51 rollouts and 1,253,376 transitions.
The full 200-update runs were retained even though an earlier checkpoint was
selected.

Checkpoint selection used native physical success, not shaped return or the
last update:

| Object | `model_50` | `model_100` | `model_150` | `model_199` | Selection |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Tomato | 27/512 (5.27%) | 97/512 (18.95%) | 369/512 (72.07%) | 308/512 (60.16%) | `model_150` |
| Potato | 511/512 (99.80%) | 23/512 (4.49%) | 0/512 | 0/512 | `model_50` |
| Bowl | 0/512 | 0/512 | 0/512 | 0/512 | rejected |

The different seeds in this screen prevent acceptance-log filename collisions;
the log filenames do not encode checkpoint update. The independent 1024-world
confirmations above were then run only for the selected Tomato and Potato
checkpoints.

## Reference conditioning and action freedom

These are reference-conditioned residual policies, not reference-free task
policies. The episode-11 `xmove_bend_pick` reference still supplies the
reference action/phase context and the residual action is applied around that
baseline. Setting `reference_reward_weight=0` removes the imitation reward,
while `max_reference_action_deviation=2` spans the full possible difference
between normalized actions in `[-1, 1]`. The right-hand and right-arm residuals
remain active through placement. PPO can therefore learn a flexible arm
correction without being rewarded for copying the reference exactly.

The accepted deterministic evaluations observed maximum effective normalized
action deltas of 0.328 for Tomato and 0.169 for Potato in the core profile.
These values are learned corrections below the configured limit, not clipping
at the reference boundary.

## Stable physics and compatibility

The v5 bundles place the object at zero artificial table clearance and preserve
free-joint damping `0.1`. This prevents the pre-contact rolling seen in the
rejected Apple v4 video. The validated manifest hashes are:

| Object | Stable manifest hash | Stable asset directory |
| :--- | :--- | :--- |
| Tomato | `5f50193f98855ddbe31fba96aa1fd1f15e9b0a1edae06c41eb0dcc6d672f89f9` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/Tomato_1_object_reward_v5_stable_xmove_bend_ep11` |
| Potato | `5ef2bafb44337eb6ffef62aa71367ebeb1fa22ad16495705622b4349598d08ea` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/Potato_1_object_reward_v5_stable_xmove_bend_ep11` |
| Bowl | `ac2af94d5eb11b62904983b67b498d8185e8aa9e4ac38ab43906036a69aea9c9` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/Bowl_1_object_reward_v5_stable_xmove_bend_ep11` |

Compatibility is opt-in. Tomato, Potato, and Bowl still resolve their existing
v4 assets when old commands are used. `derive`, `verify`, `evaluate`, `train`,
and `record` select v5 only with `--stable-physics`. Apple already used v5 by
default, so the flag resolves to the same Apple bundle.

Example evaluation from the repository root:

```bash
scripts/grasp_rl/grasp_anything_bend_objects.sh evaluate Potato_1 \
  --stable-physics --profile target_xy_10mm --seed 20261132 \
  --episodes 1024 \
  --checkpoint releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/potato_1_xmove_bend_workspace_v5_model_50.pt \
  --goal-potential-scale 20 --goal-potential-negative-clip 1 \
  --success-bonus 80 --reference-reward-weight 0 \
  --max-reference-action-deviation 2
```

## Bowl rejection

The stable Bowl reference baseline was 24/1024 (2.34%) under `+/-1 cm` target
randomization. The 200-update PPO run collected 4,915,200 transitions, but all
four screened checkpoints scored 0/512. The run is retained locally for
diagnosis; no Bowl v5 checkpoint or success video is presented as accepted.
The earlier v4 Bowl fixed and `+/-2.5 mm` reference-only results remain in
`DR_BEND_FOLLOWUP.md` and are not replaced.

## Absolute checkpoint and evaluation paths

| Artifact | Absolute path |
| :--- | :--- |
| Tomato release checkpoint | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/tomato_1_xmove_bend_workspace_v5_model_150.pt` |
| Tomato source checkpoint | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/ppo_workspace_v5_stable_target10mm_seed20261110_env1024_std080_hold4_200/model_150.pt` |
| Tomato training directory | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/ppo_workspace_v5_stable_target10mm_seed20261110_env1024_std080_hold4_200` |
| Tomato core PPO confirmation | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_10mm_seed20261131_1024.log` |
| Tomato core paired reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_10mm_seed20261131_1024.log` |
| Tomato `+/-2.5 cm` PPO | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_25mm_seed20261134_512.log` |
| Tomato `+/-2.5 cm` paired reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_25mm_seed20261134_512.log` |
| Potato release checkpoint | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/potato_1_xmove_bend_workspace_v5_model_50.pt` |
| Potato source checkpoint | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/ppo_workspace_v5_stable_target10mm_seed20261111_env1024_std080_hold4_200/model_50.pt` |
| Potato training directory | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/ppo_workspace_v5_stable_target10mm_seed20261111_env1024_std080_hold4_200` |
| Potato core PPO confirmation | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_10mm_seed20261132_1024.log` |
| Potato core paired reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_10mm_seed20261132_1024.log` |
| Potato `+/-2.5 cm` PPO | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_25mm_seed20261133_512.log` |
| Potato `+/-2.5 cm` paired reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_25mm_seed20261133_512.log` |
| Bowl rejected training directory | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/xmove_bend_ep11_v1/ppo_workspace_v5_stable_target10mm_seed20261112_env1024_std080_hold4_200` |
| Bowl final screen | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_10mm_seed20261135_512.log` |

## Absolute video paths

Each accepted object has one full-robot video and one grasp close-up. Both
camera recordings independently passed simulator success, finger/contact, and
pre-contact target-motion audits.

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/tomato_1_xmove_bend_workspace_v5_model_150/full_robot/gpu_ppo_success_full_dr_01.mp4
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/tomato_1_xmove_bend_workspace_v5_model_150/grasp_closeup/gpu_ppo_success_full_dr_01.mp4
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/potato_1_xmove_bend_workspace_v5_model_50/full_robot/gpu_ppo_success_full_dr_01.mp4
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/potato_1_xmove_bend_workspace_v5_model_50/grasp_closeup/gpu_ppo_success_full_dr_01.mp4
```

Tomato recorded zero pre-contact displacement in both videos. Potato recorded
`4.47e-8 m`, numerical noise far below the `0.003 m` limit. The release JSON
sidecars preserve the exact randomized target positions, first-contact steps,
finger audits, camera definitions, checkpoint provenance, and physical target
audit.
