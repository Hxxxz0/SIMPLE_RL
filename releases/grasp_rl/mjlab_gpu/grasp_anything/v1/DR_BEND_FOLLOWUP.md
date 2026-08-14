# Position DR and low-object bend follow-up

> Superseded result: the Apple v4 `model_300` result below was rejected after
> its video showed the Apple moving before hand contact. It remains documented
> only for provenance. The accepted stable-physics replacement is recorded in
> [`APPLE_WORKSPACE_V5.md`](APPLE_WORKSPACE_V5.md).

This document records the opt-in episode-11 `xmove_bend_pick` follow-up. It
does not replace the episode-82 route, its five released RL checkpoints, or any
default CLI behavior.

## Outcome

Success means a native physical grasp, at least 0.09 m of object lift, and 13
consecutive control steps holding that lift. Contact or finger closure alone is
not a success.

| Object | Policy | Fixed pose | Target XY jitter +/-2.5 mm | Position-DR status |
| :--- | :--- | ---: | ---: | :--- |
| `Apple_1` | legacy v4 reward-aligned PPO, now rejected | 512/512 reference only | PPO 185/1024 (18.07%) vs paired reference 166/1024 (16.21%) | superseded after pre-contact target motion was found |
| `Bowl_1` | staged reference only | 389/512 (75.98%) | 270/512 (52.73%) | moderate |
| `Potato_1` | staged reference only | 510/512 (99.61%) | 490/512 (95.70%) | strong within the measured +/-2.5 mm scope |
| `Tomato_1` | staged reference only | 512/512 (100.00%) | 496/512 (96.88%) | strong within the measured +/-2.5 mm scope |

The fixed Apple reference result remains batch sensitive. Position randomization exposes
that the reference-only 512/512 number was a fixed-geometry result rather than
evidence of domain-randomized robustness. The new Apple position-DR reference
improves paired-seed success from 81/512 to 95/512, but 18.55% is still low and
must not be described as robust DR. A repeated Warp run can also differ by a
few worlds because contact simulation is not perfectly run-to-run stable.

The first two Apple PPO experiments used 40 updates, 8192 environments and
7,864,320 fresh transitions each. On paired seed 20260914, the reference scored
46/512, while the final `std=0.01` and `std=0.02` checkpoints scored 44/512
and 45/512. They were rejected and are backed up only for provenance:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/experiments/rejected_checkpoints/apple_1_position_dr_2p5mm_std010_model_39.pt
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/experiments/rejected_checkpoints/apple_1_position_dr_2p5mm_std020_model_39.pt
```

Longer 1024-environment experiments then tested the user's proposed
many-update/many-exploration setting. Each update collected 24,576 fresh
transitions. Higher exploration was harmful: on the same 512-world seed, the
400-update `std=0.03/0.05/0.08` checkpoints scored 69/58/5, versus 84 for the
reference. A slower DR curriculum with `std=0.03` matched the reference at
84/512 after 400 updates, but exact resume to 800 updates (19,660,800
transitions) did not improve either independent seed: 80 vs 84 and 81 vs 82.

The diagnosis was reward alignment rather than insufficient action noise. The
legacy goal potential clipped an equal progress loss to one quarter of the
corresponding progress gain. New opt-in settings make that loss symmetric,
increase the physically grounded goal-potential scale and increase the terminal
success bonus. All defaults preserve the old behavior, and exact-resume
compatibility accepts legacy metadata that does not contain the new fields:

```text
--grasp-anything-goal-potential-scale 20
--grasp-anything-goal-potential-negative-clip 1
--grasp-anything-success-bonus 80
```

Two 400-update, 1024-environment warm-start runs tested potential scales 20 and
50 with `std=0.03`, a 4800-vector-step DR ramp and a fresh critic/Adam state.
Each completed 9,830,400 fresh transitions. Scale 50 increased shaped return
without improving lift and was rejected. Scale 20 produced the accepted
`model_300`; the later `model_399` was only 75/512 versus 73/512 and was
superseded rather than assumed to be better.

The accepted checkpoint passed two independent 512-world paired evaluations:

| Seed | PPO model 300 | Reference | Difference |
| :--- | ---: | ---: | ---: |
| 20260974 | 93/512 | 77/512 | +16 |
| 20260975 | 92/512 | 89/512 | +3 |
| Aggregate | 185/1024 (18.07%) | 166/1024 (16.21%) | +19, 11.4% relative |

The first seed improved substantially and the second only slightly. The result
therefore establishes a repeated positive direction and an aggregate
improvement, not a stable 20% gain and not robust position DR. The release now
contains six accepted RL checkpoints. Bowl, Potato and Tomato bend support
remains reference-only.

## Supported objects

The combined release has measured successful physical grasps for seven unique
objects: `Soap_Bottle_1`, `Bottle_1`, `Apple_1`, `Bowl_1`, `Cup_6`, `Potato_1`
and `Tomato_1`. The opt-in bend runner supports `Apple_1`, `Bowl_1`, `Potato_1`
and `Tomato_1`.

`Egg_1` was screened with its original scale and an approximately 64 mm scaled
variant, using both Apple-like and Bowl-like arm initializations. All searches
returned 0/4096 successful candidates because no bilateral grasp formed, so
`Egg_1` is explicitly not in the supported table. Other small objects may fit
the route, but must pass the same physical evaluation before being called
supported.

## Domain randomization

The bend runner exposes named, opt-in pose profiles:

| Profile | Target X/Y jitter | Robot base X/Y jitter | Yaw jitter |
| :--- | :--- | :--- | :--- |
| `fixed` | 0 | 0 | 0 |
| `target_xy_2p5mm` | +/-2.5 mm | 0 | 0 |
| `target_xy_5mm` | +/-5 mm | 0 | 0 |
| `target_xy_10mm` | +/-10 mm | 0 | 0 |
| `target_xy_25mm` | +/-25 mm | 0 | 0 |
| `target_xy_50mm` | +/-50 mm | 0 | 0 |
| `target_xy_100mm` | +/-100 mm | 0 | 0 |
| `target_base_xy_2p5mm` | +/-2.5 mm | +/-2.5 mm | 0 |

The staged-search CLI separately accepts target X/Y and yaw jitter plus robot
base X/Y and yaw jitter. Every new option defaults to zero, preserving old
search behavior. GPU evaluation now reports initial-pose distributions and
up to eight success-rate bins for target X, target Y, target yaw, robot-base X,
robot-base Y and robot-base yaw (constant axes collapse to one bin). This makes pose coverage visible instead of
letting a fixed reference-only score be mistaken for DR.

The `target_xy_100mm` profile independently samples both axes over `+/-10 cm`,
so its target plane is 20 x 20 cm. The accepted Apple v5 checkpoint is robust
only in the `+/-1 cm` core (96.14% over two seeds); its 16.60% result in the
20 x 20 cm plane is measured outer-workspace capability, not robust coverage.
The complete 78.7M-transition exploration sweep and final-checkpoint screens
are recorded in `APPLE_WORKSPACE_V5.md`.

Example evaluations from the repository root:

```bash
scripts/grasp_rl/grasp_anything_bend_objects.sh evaluate Potato_1 \
  --profile target_xy_2p5mm --seed 20260940 --episodes 512

scripts/grasp_rl/grasp_anything_bend_objects.sh evaluate Tomato_1 \
  --profile target_xy_2p5mm --seed 20260942 --episodes 512
```

The runner is deliberately separate from
`scripts/grasp_rl/grasp_anything_objects.sh`; selecting this route is explicit
and existing commands keep their old reference, episode and output paths.

## Absolute artifact paths

| Artifact | Absolute path |
| :--- | :--- |
| Apple position-DR reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/references/grasp_anything/Apple_1_xmove_bend_ep11_position_dr_2p5mm_staged_native_v2` |
| Rejected legacy Apple PPO release checkpoint | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_rewardalign_model_300.pt` |
| Rejected legacy Apple PPO source checkpoint | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_rewardalign_v1_scale20_clip1_bonus80_warm_model399_seed20260968_env1024_std030_400/model_300.pt` |
| Scale-20 training directory | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_rewardalign_v1_scale20_clip1_bonus80_warm_model399_seed20260968_env1024_std030_400` |
| Scale-50 rejected training directory | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_rewardalign_v1_scale50_clip1_bonus80_warm_model399_seed20260969_env1024_std030_400` |
| Apple PPO paired acceptance, seed 20260974 | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_rewardalign_v1_scale20_model300_reference-v2_target_xy_2p5mm_seed20260974_512.log` |
| Apple PPO paired acceptance, seed 20260975 | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/ppo_rewardalign_v1_scale20_model300_reference-v2_target_xy_2p5mm_seed20260975_512.log` |
| Apple paired acceptance, new reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_reference-Apple_1_xmove_bend_ep11_position_dr_2p5mm_staged_native_v2_target_xy_2p5mm_seed20260938_512.log` |
| Apple paired acceptance, old reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_2p5mm_seed20260938_512.log` |
| Bowl reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/references/grasp_anything/Bowl_1_xmove_bend_ep11_staged_native_v1` |
| Bowl fixed acceptance | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/xmove_bend_ep11_v1/acceptance/reference_fixed_seed20260924_512.log` |
| Bowl position-DR acceptance | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_2p5mm_seed20260925_512.log` |
| Potato reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/references/grasp_anything/Potato_1_xmove_bend_ep11_staged_native_v1` |
| Potato fixed acceptance | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/reference_fixed_seed20260939_512.log` |
| Potato position-DR acceptance | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_2p5mm_seed20260940_512.log` |
| Tomato reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/references/grasp_anything/Tomato_1_xmove_bend_ep11_staged_native_v1` |
| Tomato fixed acceptance | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/acceptance/reference_fixed_seed20260941_512.log` |
| Tomato position-DR acceptance | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/acceptance/reference_target_xy_2p5mm_seed20260942_512.log` |

## Absolute video paths

All four bend-supported objects have successful full-robot and grasp-closeup
videos. The new position-DR recordings are:

| Object and profile | Absolute local directory |
| :--- | :--- |
| Apple new reference, target XY +/-2.5 mm | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/reference_reference-Apple_1_xmove_bend_ep11_position_dr_2p5mm_staged_native_v2_target_xy_2p5mm_full_robot_seed20260937` |
| Apple new reference close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/reference_reference-Apple_1_xmove_bend_ep11_position_dr_2p5mm_staged_native_v2_target_xy_2p5mm_grasp_closeup_seed20260937` |
| Apple rejected legacy PPO, target XY +/-2.5 mm | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/ppo_reference-Apple_1_xmove_bend_ep11_position_dr_2p5mm_staged_native_v2_target_xy_2p5mm_full_robot_seed20260977` |
| Apple rejected legacy PPO close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/ppo_reference-Apple_1_xmove_bend_ep11_position_dr_2p5mm_staged_native_v2_target_xy_2p5mm_grasp_closeup_seed20260977` |
| Bowl target XY +/-2.5 mm | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_full_robot_seed20260933` |
| Bowl target XY +/-2.5 mm close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_grasp_closeup_seed20260933` |
| Potato target XY +/-2.5 mm | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_full_robot_seed20260943` |
| Potato target XY +/-2.5 mm close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_grasp_closeup_seed20260943` |
| Tomato target XY +/-2.5 mm | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_full_robot_seed20260944` |
| Tomato target XY +/-2.5 mm close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_grasp_closeup_seed20260944` |

Release-relative copies, including JSON audit sidecars, are under
`videos/*_xmove_bend_ep11_reference` and
`videos/apple_1_xmove_bend_rewardalign_model_300`. Integrity hashes are in
`SHA256SUMS`.
