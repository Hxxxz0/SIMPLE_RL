# Strict +/-20 cm workspace experiment

This document records the v6 attempt to extend the accepted stable-physics
episode-11 bend policies to target X/Y jitter of `+/-0.20 m` per axis. X and Y
are sampled independently around `(x=0, y=-0.09 m)`, so the evaluation domain
is a 40 x 40 cm square and its corners are 28.3 cm from the center. Evaluation
always uses the uniform square; focused samples are used only as an opt-in
training curriculum.

Success is the existing native physical gate: lift the object by at least
0.09 m and hold the lift for 13 control steps. Contact, closure, shaped return,
and training proxy metrics are not counted as success.

## Backward-compatible runner changes

The runner adds the opt-in `target_xy_200mm` profile and opt-in training-only
focus controls:

```text
--profile target_xy_200mm
--target-focus-probability P
--target-focus-jitter-xy-m X Y
```

All old profiles, defaults, assets, commands, and checkpoints are unchanged.
The focus arguments are omitted from the underlying environment command when
their probability remains at the default zero. Stable physics remains opt-in
for Tomato, Potato, and Bowl through `--stable-physics`; Apple continues to
resolve the same v5 asset with or without that flag.

`--max-reference-initial-position-offset 0.12` is intentionally unchanged. It
checks the frozen asset against the initial reference frame at startup; it does
not clip or reject the per-world target DR. The baseline logs below show actual
samples spanning approximately `[-0.20,+0.20] m` in X and `[-0.29,+0.11] m` in
Y.

Reference conditioning remains present, but every v6 training and evaluation
uses `reference_reward_weight=0` and `max_reference_action_deviation=2`. The
policy is not rewarded for imitation and has the full normalized residual
correction range.

## Honest pre-training baseline

The accepted v5 checkpoints and paired references were evaluated on identical
1024-world seeds before new training:

| Object | Existing checkpoint | Paired reference | Interpretation |
| :--- | ---: | ---: | :--- |
| Apple | 39/1024 (3.81%) | 29/1024 (2.83%) | measured sparse generalization only |
| Potato | 40/1024 (3.91%) | 29/1024 (2.83%) | measured sparse generalization only |
| Tomato | 31/1024 (3.03%) | 13/1024 (1.27%) | measured sparse generalization only |

Absolute baseline logs:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261200_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_200mm_seed20261200_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261201_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_200mm_seed20261201_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261202_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_200mm_seed20261202_1024.log
```

## Training protocol

Every run uses 1024 environments, 24 steps per environment per update,
temporally coherent exploration with standard deviation 0.08 held for four
steps, learning rate `5e-5`, zero reference reward, full residual range, stable
physics, and fresh on-policy rollouts. A 300-update run therefore contains
7,372,800 transitions; 400 updates contain 9,830,400; 800 updates contain
19,660,800.

The direct jump to `+/-20 cm` mixed 50% uniform full-square samples and 50%
`+/-2.5 cm` focused samples. It ran 400 updates per object and failed:

| Object | model_50 | model_200 | model_399 | Result |
| :--- | ---: | ---: | ---: | :--- |
| Apple | 24/512 (4.69%) | 2/512 (0.39%) | 0/512 | rejected |
| Potato | 13/512 (2.54%) | 0/512 | 0/512 | rejected |
| Tomato | 8/512 (1.56%) | 0/512 | 0/512 | rejected |

The staged curriculum instead expands `+/-2.5 cm`, `+/-5 cm`, `+/-10 cm`,
then `+/-20 cm`, always warm-starting from a physically screened checkpoint:

| Object/stage | model_50 | model_100/150 | Selected |
| :--- | ---: | ---: | :--- |
| Apple `+/-2.5 cm` | 221/512 (43.16%) | not selected | original v5 model_100 (known 72.75%) |
| Potato `+/-2.5 cm` | 502/512 (98.05%) | not selected | model_50 |
| Tomato `+/-2.5 cm` | 281/512 (54.88%) | not selected | model_50 |
| Apple `+/-5 cm` | 240/512 (46.88%) | model_100: 182/512 (35.55%) | model_50 |
| Potato `+/-5 cm` | 293/512 (57.23%) | model_150: 0/512 | model_50 |
| Tomato `+/-5 cm` | 122/512 (23.83%) | model_100: 0/512 | model_50 |
| Apple `+/-10 cm` | 83/512 (16.21%) | later checkpoints not selected | model_50 |
| Potato `+/-10 cm` | 37/512 (7.23%) | model_100: 1/512 | model_50 |
| Tomato `+/-10 cm` | model_50: 0/512 | model_100: 0/512 | rejected; no final expansion |

Tomato stops at the failed `+/-10 cm` gate. Apple and Potato continued through
complete 800-update final `+/-20 cm` lines, using 50% full-square samples and
50% `+/-10 cm` focused samples. Each run has exactly 800 integrity records and
19,660,800 fresh transitions. Every record is on-policy and reports that its
rollout was not reused. A checkpoint is not released merely because its
training run completed.

## Final +/-20 cm screening

| Object | model_50 | model_100 | model_150 | model_200 | model_250 | model_300 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple | 4/512 (0.78%) | 2/512 (0.39%) | 1/512 (0.20%) | 0/512 | 0/512 | 0/512 |
| Potato | 4/512 (0.78%) | 0/512 | n/a | 0/512 | 0/512 | 0/512 |

| Object | model_350 | model_400 | model_500 | model_600 | model_700 | model_799 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple | 0/512 | 0/512 | 0/512 | 0/512 | 0/512 | 0/512 |
| Potato | 0/512 | 0/512 | 0/512 | 0/512 | 0/512 | 0/512 |

The final lines are rejected. They do not beat the accepted v5 checkpoints'
strict-profile baselines, so no v6 checkpoint is added to the release. Longer
training caused policy collapse rather than broader workspace support. The
release continues to select the best physically evaluated checkpoint, not the
last checkpoint.

Absolute final-screen logs:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261319_512.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261321_512.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261323_512.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261320_512.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261322_512.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_200mm_seed20261324_512.log
```

## Absolute training directories

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v6_stage200mm_focus50x100_warm100m50_seed20261241_env1024_std080_hold4_800
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/ppo_workspace_v6_stage200mm_focus50x100_warm100m50_seed20261240_env1024_std080_hold4_800
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/ppo_workspace_v6_stage100mm_focus25x50_warm50m50_seed20261232_env1024_std080_hold4_400
```

## Audited sparse-success videos

These recordings use the accepted v5 checkpoints under the strict
`target_xy_200mm` sampling profile. They prove sparse measured generalization
inside that distribution; they do not prove success at the `+/-20 cm` edges.
The six successful samples are 6.34-9.29 cm from the distribution center. All
sidecars pass native simulator success, right-hand grasp, physically unattached
target, and pre-contact target-motion audits.

| Object/view | Sampled target XY offset | Pre-contact motion | Absolute release video |
| :--- | :--- | ---: | :--- |
| Apple full robot | `(-3.06, -5.65) cm` | 0 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_target_xy_200mm_sparse_v5/full_robot/gpu_ppo_success_full_dr_01.mp4` |
| Apple close-up | `(-3.08, -8.77) cm` | 0 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_target_xy_200mm_sparse_v5/grasp_closeup/gpu_ppo_success_full_dr_01.mp4` |
| Potato full robot | `(-5.58, -3.02) cm` | `7.45e-8 m` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/potato_1_xmove_bend_target_xy_200mm_sparse_v5/full_robot/gpu_ppo_success_full_dr_01.mp4` |
| Potato close-up | `(-5.70, -5.80) cm` | `0.001940 m` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/potato_1_xmove_bend_target_xy_200mm_sparse_v5/grasp_closeup/gpu_ppo_success_full_dr_01.mp4` |
| Tomato full robot | `(-4.09, -7.94) cm` | 0 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/tomato_1_xmove_bend_target_xy_200mm_sparse_v5/full_robot/gpu_ppo_success_full_dr_01.mp4` |
| Tomato close-up | `(-1.43, -8.44) cm` | 0 m | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/tomato_1_xmove_bend_target_xy_200mm_sparse_v5/grasp_closeup/gpu_ppo_success_full_dr_01.mp4` |

The strict 1024-world baseline is therefore the final measured claim: Apple
39/1024 (3.81%), Potato 40/1024 (3.91%), and Tomato 31/1024 (3.03%). Robust
`+/-20 cm` support was attempted but not achieved.
