# grasp_anything mjlab GPU PPO v1

This release backs up the selected object-specific RL checkpoints and the
corresponding evaluation record for the current `grasp_anything` work. It is a
checkpoint release, not a self-contained asset release: the object bundles,
processed references, acceptance logs, and videos remain under the local
`outputs/` and `data/` trees.

Git LFS is required to download the five `.pt` files:

```bash
git lfs pull
```

## Supported objects and measured scope

A success is a native physical grasp that lifts the object at least 0.09 m and
holds that lift for 13 control steps. Contact or finger closure without the
stable lift is not counted as success.

| Object | Released checkpoint | Accepted evaluation | Supported scope |
| :--- | :--- | ---: | :--- |
| `Soap_Bottle_1` | `checkpoints/soap_bottle_1_model_199.pt` | 512/512 (100.00%) | narrow pose DR; requires the runtime lift-arm-decay variant |
| `Bottle_1` | `checkpoints/bottle_1_model_199.pt` | 492/512 (96.09%) | narrow pose DR; requires the runtime lift-arm-decay variant |
| `Apple_1` | `checkpoints/apple_1_model_0.pt` | 112/512 (21.88%) | fixed-pose baseline only |
| `Bowl_1` | `checkpoints/bowl_1_model_0.pt` | 194/512 (37.89%) | fixed-pose rim-grasp baseline only |
| `Cup_6` | `checkpoints/cup_6_model_39.pt` | 1523/1536 (99.15%) over three seeds | full DR profile evaluated at strength 0.2 with the Cup X-conditioned proposal |

The two narrow bottle policies were evaluated with target translation jitter
of +/-5 mm in X/Y and target yaw jitter of +/-0.015 rad. Their accepted behavior
also depends on these runtime options:

```text
--grasp-anything-lift-arm-residual-min-scale 0.1
--grasp-anything-lift-arm-residual-decay-steps 10
--grasp-anything-lift-arm-residual-grasp-steps 3
```

The Apple result does not establish narrow DR support (its recorded narrow
result is 3/512). The Bowl result also does not establish narrow or workspace
DR support. Every checkpoint is tied to its own object contract and reference;
cross-object checkpoint compatibility is not supported.

Exact evaluation seeds, world hashes, lift measurements, checkpoint hashes,
and training transition counts are recorded in `release.json`.

## Local evaluation

From the repository root, the four catalog objects can be evaluated with the
object runner. For example:

```bash
scripts/grasp_rl/grasp_anything_objects.sh evaluate Soap_Bottle_1 \
  --stage narrow \
  --checkpoint releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/soap_bottle_1_model_199.pt \
  --lift-arm-decay --seed 20260870 --episodes 512

scripts/grasp_rl/grasp_anything_objects.sh evaluate Apple_1 \
  --stage fixed \
  --checkpoint releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_model_0.pt \
  --seed 20260884 --episodes 512
```

Use the same pattern for `Bottle_1` and `Bowl_1`, preserving the stage and
lift-arm-decay requirements in the table. The accepted Cup command is encoded
by `cup6_candidate_acceptance` in
`scripts/grasp_rl/grasp_anything_cup6.sh`; point
`SIMPLE_PPO_CHECKPOINT_CUP6_CANDIDATE` at the released Cup checkpoint before
running it.

## Video evidence on this machine

Every released object has recorded successful videos. These videos are not
duplicated in Git because their original local outputs are already retained.
The absolute directories are:

| Object | Absolute video directory | Videos |
| :--- | :--- | ---: |
| `Soap_Bottle_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Soap_Bottle_1/single_ref_ep82_per_object_v1/videos_narrow_dr1_lift_arm_decay_v1` | 3 grasp close-ups |
| `Bottle_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bottle_1/single_ref_ep82_per_object_v1/videos_narrow_dr1_lift_arm_decay_v1` | 3 grasp close-ups |
| `Apple_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/single_ref_ep82_small_v4_multireplay/videos_fixed_dr1_model_0` | 3 full-robot + 3 close-ups |
| `Bowl_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/single_ref_ep82_rim_v4_staged/videos_fixed_dr1_model_0` | 3 full-robot + 3 close-ups |
| `Cup_6` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Cup_6/release_single_ref_ep82_dr02_xonly_s12p5_e4/videos_dr02_final` | 3 full-robot + 3 close-ups |

## Integrity

Verify the downloaded LFS objects from this directory:

```bash
sha256sum -c SHA256SUMS
```
