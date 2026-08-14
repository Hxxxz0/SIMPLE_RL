# Position DR and low-object bend follow-up

This document records the opt-in episode-11 `xmove_bend_pick` follow-up. It
does not replace the episode-82 route, its five released RL checkpoints, or any
default CLI behavior.

## Outcome

Success means a native physical grasp, at least 0.09 m of object lift, and 13
consecutive control steps holding that lift. Contact or finger closure alone is
not a success.

| Object | Policy | Fixed pose | Target XY jitter +/-2.5 mm | Position-DR status |
| :--- | :--- | ---: | ---: | :--- |
| `Apple_1` | position-DR staged reference only | 512/512 | 95/512 (18.55%), seed 20260938 | low; improved over the paired old reference at 81/512, but not robust |
| `Bowl_1` | staged reference only | 389/512 (75.98%) | 270/512 (52.73%) | moderate |
| `Potato_1` | staged reference only | 510/512 (99.61%) | 490/512 (95.70%) | strong within the measured +/-2.5 mm scope |
| `Tomato_1` | staged reference only | 512/512 (100.00%) | 496/512 (96.88%) | strong within the measured +/-2.5 mm scope |

The fixed Apple result remains batch sensitive. Position randomization exposes
that the reference-only 512/512 number was a fixed-geometry result rather than
evidence of domain-randomized robustness. The new Apple position-DR reference
improves paired-seed success from 81/512 to 95/512, but 18.55% is still low and
must not be described as robust DR. A repeated Warp run can also differ by a
few worlds because contact simulation is not perfectly run-to-run stable.

Two Apple PPO experiments used 40 updates, 8192 environments and 7,864,320
fresh transitions each. On paired seed 20260914, the reference scored 46/512,
while the final `std=0.01` and `std=0.02` checkpoints scored 44/512 and 45/512.
They were rejected and are backed up only for provenance:

```text
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/experiments/rejected_checkpoints/apple_1_position_dr_2p5mm_std010_model_39.pt
/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/experiments/rejected_checkpoints/apple_1_position_dr_2p5mm_std020_model_39.pt
```

These are not accepted checkpoints. The release still contains exactly five
accepted RL checkpoints. The new Apple, Bowl, Potato and Tomato bend results
are reference-only policies, not RL weights.

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
| `target_base_xy_2p5mm` | +/-2.5 mm | +/-2.5 mm | 0 |

The staged-search CLI separately accepts target X/Y and yaw jitter plus robot
base X/Y and yaw jitter. Every new option defaults to zero, preserving old
search behavior. GPU evaluation now reports initial-pose distributions and
up to eight success-rate bins for target X, target Y, target yaw, robot-base X,
robot-base Y and robot-base yaw (constant axes collapse to one bin). This makes pose coverage visible instead of
letting a fixed reference-only score be mistaken for DR.

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
| Bowl target XY +/-2.5 mm | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_full_robot_seed20260933` |
| Bowl target XY +/-2.5 mm close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_grasp_closeup_seed20260933` |
| Potato target XY +/-2.5 mm | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_full_robot_seed20260943` |
| Potato target XY +/-2.5 mm close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_grasp_closeup_seed20260943` |
| Tomato target XY +/-2.5 mm | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_full_robot_seed20260944` |
| Tomato target XY +/-2.5 mm close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/videos/reference_target_xy_2p5mm_grasp_closeup_seed20260944` |

Release-relative copies, including JSON audit sidecars, are under
`videos/*_xmove_bend_ep11_reference`. Integrity hashes are in `SHA256SUMS`.
