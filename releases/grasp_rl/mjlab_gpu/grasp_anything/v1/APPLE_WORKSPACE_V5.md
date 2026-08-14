# Apple stable-physics workspace PPO

This document records the replacement Apple episode-11 PPO experiment after
the pre-contact object-motion defect was found in the earlier v4 asset. The v4
checkpoint and videos remain in the release for provenance, but they are
rejected and must not be used as accepted evidence.

## Accepted result

The selected policy is `model_100` from the v5 stable-physics scratch run. It
uses 1024 environments, 24 steps per environment per update, 100 selected
updates, and 2,482,176 fresh on-policy transitions. PPO was allowed to correct
the complete normalized right-hand and right-arm range through placement:

```text
reference_reward_weight = 0
max_reference_action_deviation = 2
exploration_std = 0.08
exploration_hold_steps = 4
```

Holding a noise direction for four policy steps was important. On the same v5
asset, `std=0.12, hold=8` collapsed to 0/1024 at update 100, while the accepted
`std=0.08, hold=4` run reached 991/1024. Longer training was retained as an
experiment rather than treated as automatically better: update 175 had already
fallen to 748/1024 versus its paired 712/1024 reference.

The V2 residual actor owns every normalized correction dimension of the right
hand (`[7:14]`) and right arm (`[21:28]`) through the active placement stages.
The configured deviation limit of 2 spans the complete possible difference
between normalized actions in `[-1, 1]`, and the reference imitation reward is
zero. Thus the learned arm direction is not forced to remain near the
reference. The accepted deterministic policy used a maximum observed action
delta of 0.323 in the core evaluation and 0.339 in the 20 x 20 cm diagnostic;
it learned a smaller correction than the allowed bound rather than being
clipped by it.

The target center is `(x=0, y=-0.09 m)`. Profile values below are independent
uniform jitter limits on both X and Y, so `+/-10 cm` is a 20 x 20 cm sampled
plane, not a 10 cm total span.

| Target X/Y jitter | Sampled plane | PPO | Paired reference | Interpretation |
| :--- | :--- | ---: | ---: | :--- |
| `+/-1 cm` | 2 x 2 cm | 1969/2048 (96.14%) | 1383/2048 (67.53%) | accepted robust core workspace, two independent seeds |
| `+/-2.5 cm` | 5 x 5 cm | 745/1024 (72.75%) | 502/1024 (49.02%) | measured one-seed generalization |
| `+/-5 cm` | 10 x 10 cm | 485/1024 (47.36%) | 428/1024 (41.80%) | measured one-seed generalization |
| `+/-10 cm` | 20 x 20 cm | 340/2048 (16.60%) | 231/2048 (11.28%) | repeated improvement, but not robust at the outer workspace |

The accepted claim is therefore 96.14% in the `+/-1 cm` core. The policy can
produce successful grasps throughout the larger 20 x 20 cm sampled plane and
improves its paired reference there, but a 16.60% aggregate rate is not a
robust-workspace claim.

## Full exploration sweep

Four 1024-environment training lines were run through update 799 or 800. Every
update collected 24 steps per environment. Together they collected 78,667,776
fresh on-policy transitions. The two wider lines warm-started from the accepted
`model_100`, but their transition counts below include only newly collected
rollouts.

| Training line | Target X/Y jitter | Exploration | Last checkpoint | Fresh transitions | Final 512-world screen |
| :--- | :--- | :--- | :--- | ---: | ---: |
| core scratch plus exact resume | `+/-1 cm` | `std=0.08, hold=4` | `model_800` | 19,685,376 | 0/512, seed 20261083 |
| high-noise scratch control | `+/-1 cm` | `std=0.12, hold=8` | `model_799` | 19,660,800 | 0/512, seed 20261080 |
| `model_100` warm start, 5 x 5 cm plane | `+/-2.5 cm` | `std=0.08, hold=4` | `model_799` | 19,660,800 | 0/512, seed 20261081 |
| `model_100` warm start, 20 x 20 cm plane | `+/-10 cm` | `std=0.08, hold=4` | `model_799` | 19,660,800 | 0/512, seed 20261082 |

Intermediate screens also showed the same late-training decline: the core
`model_520`, high-noise `model_670`, and 5 x 5 cm `model_480` each scored
0/512; the 20 x 20 cm `model_440` scored 29/512. These completed experiments
show that more updates and larger persistent noise did not improve this PPO
objective. Checkpoint selection must use physical evaluation rather than the
last update, so `model_100` remains selected.

## Physics correction and video audit

The old Apple asset discarded the source object's free-joint damping and added
2 mm of table clearance. The nearly spherical Apple could settle and roll
before the hand arrived. The v5 asset preserves source free-joint damping
`0.1` and uses zero table clearance.

Every newly accepted recording also enforces a pre-contact motion audit. A
candidate is rejected if target XY displacement exceeds 3 mm before first hand
contact. Both released v5 videos measured exactly `0.0 m` pre-contact motion:

| Camera | Sampled target offset | First hand contact | Pre-contact motion |
| :--- | :--- | ---: | ---: |
| full robot | `(-3.81 cm, -6.79 cm)` | step 226 | `0.0 m`, passed |
| grasp close-up | `(-5.18 cm, -8.13 cm)` | step 220 | `0.0 m`, passed |

The simulator lift criterion, right-finger closure/contact audit, and
pre-contact target-motion audit all pass in both JSON sidecars.

## Absolute artifact paths

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_workspace_v5_model_100.pt
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v2_stable_stage10mm_free_seed20261040_env1024_std080_hold4_800/model_100.pt
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/Apple_1_object_reward_v5_stable_xmove_bend_ep11
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/ppo_target_xy_100mm_full_robot_seed20261060/gpu_ppo_success_full_dr_01.mp4
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/ppo_target_xy_100mm_grasp_closeup_seed20261061/gpu_ppo_success_full_dr_01.mp4
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_workspace_v5_model_100/full_robot/gpu_ppo_success_full_dr_01.mp4
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_workspace_v5_model_100/grasp_closeup/gpu_ppo_success_full_dr_01.mp4
```

Completed long-training directories:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v2_stable_stage10mm_free_seed20261040_env1024_std080_hold4_800
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v2_stable_stage10mm_free_seed20261040_env1024_std080_hold4_resume280_to800
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v2_stable_stage10mm_free_seed20261041_env1024_std120_hold8_800
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v2_stable_stage25mm_warm10m100_seed20261050_env1024_std080_hold4_800
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v2_stable_stage100mm_warm10m100_seed20261051_env1024_std080_hold4_800
```

Paired evaluation logs:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_10mm_seed20261046_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_10mm_seed20261046_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_10mm_seed20261055_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_10mm_seed20261055_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_100mm_seed20261045_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_100mm_seed20261045_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_100mm_seed20261056_1024.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_100mm_seed20261056_1024.log
```

Final long-training screening logs:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_10mm_seed20261080_512.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_25mm_seed20261081_512.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_100mm_seed20261082_512.log
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_target_xy_10mm_seed20261083_512.log
```

## Object support

The combined release has measured physical successes for seven unique objects:
`Soap_Bottle_1`, `Bottle_1`, `Apple_1`, `Bowl_1`, `Cup_6`, `Potato_1`, and
`Tomato_1`. The opt-in low-object bend runner accepts `Apple_1`, `Bowl_1`,
`Potato_1`, and `Tomato_1`. Apple, Potato, and Tomato now have accepted v5
workspace PPO results; Bowl retains its earlier measured reference-only scope
after its v5 PPO experiment was rejected. See
[`MULTI_OBJECT_WORKSPACE_V5.md`](MULTI_OBJECT_WORKSPACE_V5.md).

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/grasp_rl/grasp_anything_bend_objects.py evaluate Apple_1 \
  --profile target_xy_10mm --seed 20261055 --episodes 1024 \
  --checkpoint releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_workspace_v5_model_100.pt \
  --goal-potential-scale 20 --goal-potential-negative-clip 1 \
  --success-bonus 80 --reference-reward-weight 0 \
  --max-reference-action-deviation 2
```

The wider diagnostic uses `--profile target_xy_100mm`. The old v4
`apple_1_xmove_bend_rewardalign_model_300.pt` is retained only for provenance.
