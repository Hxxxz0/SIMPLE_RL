# grasp_anything mjlab GPU PPO v1

This release backs up the selected object-specific RL checkpoints and the
corresponding evaluation record for the current `grasp_anything` work. It also
contains the small Apple `xmove_bend_pick` staged reference and its two audited
videos. The derived object bundles and full experiment logs remain under the
local `outputs/` and `data/` trees.

Git LFS is required to download the `.pt`, `.npz`, and `.mp4` files:

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

These five checkpoints are unchanged. In particular, the Apple episode-82
checkpoint remains the 112/512 fixed-pose baseline; it was not replaced or
used to drive the new episode-11 result because the action-transform hashes do
not match.

## Apple low-object reference variant

The lower `xmove_bend_pick` trajectory is a substantially better geometric fit
for `Apple_1` than the existing `xmove_pick` episode 82. Episode 11 lowers the
pelvis to 0.5052 m, about 17.5 cm below the lowest episode-82 pose, and its
original lift is 0.1668 m versus 0.1177 m for episode 82.

The raw episode-11 reference without an Apple-specific correction scored
0/128. A staged arm/hand search with the fixed target offset `(x=0, y=-0.09 m)`
reached 3839/4096 (93.73%) in CEM round 7. The exported staged reference then
passed two independent 512-world checks:

| Variant | Policy | Accepted evaluation | Mean/max lift | Scope |
| :--- | :--- | ---: | :--- | :--- |
| `Apple_1` episode 11 | reference only, no PPO | 512/512 (100.00%) | 0.26416/0.26889 m | fixed geometry, exact 512-world Warp batch contract |

The exact same reference scored 0/32 with `num_envs=32`; the current MuJoCo-Warp
contact result is batch-size sensitive. Therefore this is not a general DR
claim and must not be quoted without the 512-world runtime contract. PPO was
not trained for this route because the reference itself already achieved
512/512, and adding PPO at that point would add regression risk without an
established gain.

Compatibility is opt-in:

- Existing commands still default to `xmove_pick` and episode 82 where they did
  before this addition.
- `xmove_bend_pick` is accepted only after checking its V2 331D/842D/36D schema.
- Legacy `bend_pick` remains rejected because its 192D schema is incompatible.
- `record --reference-only` is mutually exclusive with `--checkpoint`; the old
  PPO recording path and checkpoint provenance checks are unchanged.

The staged reference is backed up at
`references/apple_1_xmove_bend_ep11_staged_native_v1`. Its action-transform
SHA256 is
`0b0ac2e1f69c2e304f2cc720fadbdc7fab5d678f0f492ad24815d5ae4e9002be`;
the staged episode SHA256 is
`f377e28458badee178b7c15641196c3de9202911c61b6380966378470510c823`.

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

The Apple bend reference must use all of the following conditions: strict
episode 11, `num_envs=512`, target offset `0 -0.09`, zero pose jitter, unit
mass/friction/damping/actuator scales, zero action delay, zero reference noise,
zero reference retarget gains, and evaluation DR strength 1. It is evaluated or
recorded with `--reference-only`, never with the episode-82 Apple checkpoint.

## Apple experiment paths

The absolute local paths are recorded here because the asset and detailed logs
are intentionally outside Git:

| Artifact | Absolute path |
| :--- | :--- |
| Derived asset | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/Apple_1_object_reward_v4_xmove_bend_ep11` |
| Strict source reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/references/grasp_anything/Apple_1_xmove_bend_ep11_strict_v1` |
| Successful staged reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/references/grasp_anything/Apple_1_xmove_bend_ep11_staged_native_v1` |
| 4096-world search result | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/diagnostics/staged_surface_yminus09_seed20260901_4096.json` |
| Independent export replay | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/diagnostics/staged_export_seed20260902_512.json` |
| Formal acceptance | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_fixed_seed20260903_512.json` |

## Video evidence on this machine

Every released object has recorded successful videos. The two new Apple bend
videos are backed up in Git LFS; older videos remain only in their retained
local output directories. The absolute directories are:

| Object | Absolute video directory | Videos |
| :--- | :--- | ---: |
| `Soap_Bottle_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Soap_Bottle_1/single_ref_ep82_per_object_v1/videos_narrow_dr1_lift_arm_decay_v1` | 3 grasp close-ups |
| `Bottle_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bottle_1/single_ref_ep82_per_object_v1/videos_narrow_dr1_lift_arm_decay_v1` | 3 grasp close-ups |
| `Apple_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/single_ref_ep82_small_v4_multireplay/videos_fixed_dr1_model_0` | 3 full-robot + 3 close-ups |
| `Bowl_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/single_ref_ep82_rim_v4_staged/videos_fixed_dr1_model_0` | 3 full-robot + 3 close-ups |
| `Cup_6` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Cup_6/release_single_ref_ep82_dr02_xonly_s12p5_e4/videos_dr02_final` | 3 full-robot + 3 close-ups |
| `Apple_1` bend reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos_fixed_reference` | 1 audited full-robot + 1 audited close-up |

The new videos are also backed up in this release under
`videos/apple_1_xmove_bend_ep11_reference`. Both are H.264, 960x540, 50 fps,
319 frames, and 6.38 seconds. Their JSON sidecars report native simulator
success, the lift stage, and 86 consecutive valid right-hand grasp steps.

## Integrity

Verify the downloaded LFS objects from this directory:

```bash
sha256sum -c SHA256SUMS
```
