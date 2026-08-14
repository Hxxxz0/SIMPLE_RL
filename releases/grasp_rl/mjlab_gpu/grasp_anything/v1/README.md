# grasp_anything mjlab GPU PPO v1

This release backs up eight accepted object-specific RL checkpoints and the
corresponding evaluation record for the current `grasp_anything` work. It also
contains opt-in episode-11 `xmove_bend_pick` references for Apple, Bowl, Potato
and Tomato, plus audited fixed/position-DR videos. The derived object bundles
and full experiment logs remain under the local `outputs/` and `data/` trees.

The accepted stable-physics Apple workspace result, including absolute
artifact and video paths, is in
[`APPLE_WORKSPACE_V5.md`](APPLE_WORKSPACE_V5.md). The accepted Tomato/Potato
extension and rejected Bowl experiment are in
[`MULTI_OBJECT_WORKSPACE_V5.md`](MULTI_OBJECT_WORKSPACE_V5.md). The older
position-DR history is retained in
[`DR_BEND_FOLLOWUP.md`](DR_BEND_FOLLOWUP.md). The strict `+/-20 cm` profile,
full training record, rejected long-run checkpoints, and audited sparse-success
videos are in [`WORKSPACE_200MM_V6.md`](WORKSPACE_200MM_V6.md). In short, seven
unique objects now have measured physical successes: `Soap_Bottle_1`,
`Bottle_1`, `Apple_1`, `Bowl_1`, `Cup_6`, `Potato_1` and `Tomato_1`. There are
now eight accepted RL checkpoints. Apple, Potato, and Tomato have accepted
stable-physics episode-11 workspace PPOs. Bowl retains its earlier fixed-pose
checkpoint and v4 reference-only bend evidence; its stable workspace PPO was
rejected after all four screened checkpoints scored 0/512.

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
| `Apple_1` bend v5 | `checkpoints/apple_1_xmove_bend_workspace_v5_model_100.pt` | 1969/2048 (96.14%) vs paired reference 1383/2048 | stable physics; target XY +/-1 cm per axis over two seeds |
| `Potato_1` bend v5 | `checkpoints/potato_1_xmove_bend_workspace_v5_model_50.pt` | 1024/1024 (100.00%) vs paired reference 618/1024 | stable physics; trained target XY +/-1 cm; measured +/-2.5 cm generalization 429/512 |
| `Tomato_1` bend v5 | `checkpoints/tomato_1_xmove_bend_workspace_v5_model_150.pt` | 739/1024 (72.17%) vs paired reference 310/1024 | stable physics; trained target XY +/-1 cm; measured +/-2.5 cm generalization 199/512 |
| `Bowl_1` | `checkpoints/bowl_1_model_0.pt` | 194/512 (37.89%) | fixed-pose rim-grasp baseline only |
| `Cup_6` | `checkpoints/cup_6_model_39.pt` | 1523/1536 (99.15%) over three seeds | full DR profile evaluated at strength 0.2 with the Cup X-conditioned proposal |

The opt-in `target_xy_200mm` profile samples X and Y independently over
`+/-0.20 m`, a 40 x 40 cm plane. Existing v5 checkpoints have sparse measured
success there: Apple 39/1024 (3.81%), Potato 40/1024 (3.91%), and Tomato
31/1024 (3.03%). Complete 800-update, 19.66M-transition Apple/Potato expansion
runs collapsed to 0/512 at their final checkpoints, so no new weight is
released and robust `+/-20 cm` support is not claimed.

The original five checkpoints are unchanged. The new Potato and Tomato v5
checkpoints are additional opt-in bend policies and do not replace older
artifacts. The Apple episode-82 checkpoint
remains the 112/512 fixed-pose baseline; it was not replaced or used to drive
the episode-11 result because the action-transform hashes do not match. The v5
Apple bend checkpoint replaces the accepted status of the old v4 bend policy.
The old checkpoint remains for provenance, but its video showed target motion
before hand contact and is rejected.

The Apple v5 follow-up did not stop at the selected early checkpoint. Four
1024-environment lines were run through update 799/800 for 78,667,776 fresh
transitions in total, including stronger temporally persistent noise and
5 x 5 cm / 20 x 20 cm expansion from `model_100`. All four final checkpoints
scored 0/512. This is why the release selects the physically evaluated
`model_100`, not the last checkpoint. The actor has full right-hand/right-arm
normalized residual range during placement and no reference imitation reward;
the decline is not caused by a tight reference-action clamp.

The separate bend route previously had these v4 reference-only position
results. These historical rows remain valid in their measured scope but do not
describe the newer stable-physics PPOs:

| Object | Released bend reference | Target XY +/-2.5 mm | Measured scope |
| :--- | :--- | ---: | :--- |
| `Apple_1` | `references/apple_1_xmove_bend_ep11_position_dr_2p5mm_staged_native_v2` | legacy v4 PPO 185/1024 vs paired reference 166/1024 | historical result rejected after pre-contact target motion was found |
| `Bowl_1` | `references/bowl_1_xmove_bend_ep11_staged_native_v1` | 270/512 (52.73%) | moderate position tolerance |
| `Potato_1` | `references/potato_1_xmove_bend_ep11_staged_native_v1` | 490/512 (95.70%) | strong within measured +/-2.5 mm scope |
| `Tomato_1` | `references/tomato_1_xmove_bend_ep11_staged_native_v1` | 496/512 (96.88%) | strong within measured +/-2.5 mm scope |

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
claim and must not be quoted without the 512-world runtime contract. The later
position-DR audit showed that the fixed score does not generalize: the improved
Apple reference reached 95/512 under target X/Y jitter of +/-2.5 mm. Early
40-update PPO and later 800-update high-exploration/legacy-reward trials did not
beat their references. The historical reward-aligned v4 `model_300` used 1024
environments, `std=0.03`, 24 steps per environment per update, and 7,397,376
fresh transitions after actor warm start. It scored 93/512 vs 77/512 and
92/512 vs 89/512 on two independent seeds. The aggregate gain is 19 successes
or 11.4% relative, but the result is now rejected because the Apple moved
before hand contact. The accepted v5 replacement is documented in
[`APPLE_WORKSPACE_V5.md`](APPLE_WORKSPACE_V5.md).

Compatibility is opt-in:

- Existing commands still default to `xmove_pick` and episode 82 where they did
  before this addition.
- `xmove_bend_pick` is accepted only after checking its V2 331D/842D/36D schema.
- Legacy `bend_pick` remains rejected because its 192D schema is incompatible.
- `record --reference-only` is mutually exclusive with `--checkpoint`; the old
  PPO recording path and checkpoint provenance checks are unchanged.
- Tomato, Potato, and Bowl keep resolving their v4 asset unless
  `--stable-physics` is supplied to `derive`, `verify`, `evaluate`, `train`, or
  `record`. Existing commands therefore keep their prior behavior.

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

The Apple episode-82 result does not establish narrow DR support (its recorded
narrow result is 3/512). The accepted v5 Apple bend PPO establishes a 96.14%
`+/-1 cm` core target-position scope. Its measured `+/-10 cm` result is only
16.60%, so the outer 20 x 20 cm plane is not a robust-workspace claim. Potato
and Tomato establish 100.00% and 72.17% respectively in independently confirmed
`+/-1 cm` core evaluations. Their `+/-2.5 cm` results are measured
generalization, not their robust core claim. The Bowl result does not establish
narrow or workspace DR support. Every checkpoint is tied to its own object
contract and reference; cross-object checkpoint compatibility is not supported.

Exact evaluation seeds, world hashes, lift measurements, checkpoint hashes,
and training transition counts are recorded in `release.json`.

## Local evaluation

From the repository root, the original catalog objects can be evaluated with
the object runner. For example:

```bash
scripts/grasp_rl/grasp_anything_objects.sh evaluate Soap_Bottle_1 \
  --stage narrow \
  --checkpoint releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/soap_bottle_1_model_199.pt \
  --lift-arm-decay --seed 20260870 --episodes 512

scripts/grasp_rl/grasp_anything_objects.sh evaluate Apple_1 \
  --stage fixed \
  --checkpoint releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_model_0.pt \
  --seed 20260884 --episodes 512

.venv/bin/python scripts/grasp_rl/grasp_anything_bend_objects.py evaluate Apple_1 \
  --profile target_xy_10mm --seed 20261055 --episodes 1024 \
  --checkpoint releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_workspace_v5_model_100.pt \
  --goal-potential-scale 20 --goal-potential-negative-clip 1 \
  --success-bonus 80 --reference-reward-weight 0 \
  --max-reference-action-deviation 2
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
| Derived stable asset | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/Apple_1_object_reward_v5_stable_xmove_bend_ep11` |
| Strict source reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/references/grasp_anything/Apple_1_xmove_bend_ep11_strict_v1` |
| Successful staged reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/references/grasp_anything/Apple_1_xmove_bend_ep11_staged_native_v1` |
| 4096-world search result | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/diagnostics/staged_surface_yminus09_seed20260901_4096.json` |
| Independent export replay | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/diagnostics/staged_export_seed20260902_512.json` |
| Formal acceptance | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/acceptance/reference_fixed_seed20260903_512.json` |
| Accepted bend PPO checkpoint | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/checkpoints/apple_1_xmove_bend_workspace_v5_model_100.pt` |
| Accepted bend PPO source run | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/ppo_workspace_v2_stable_stage10mm_free_seed20261040_env1024_std080_hold4_800` |

## Video evidence on this machine

Every released object has recorded successful videos. The accepted Apple,
Tomato, and Potato v5 bend videos are backed up in Git LFS; older videos remain
in their retained local output directories. The absolute directories are:

| Object | Absolute video directory | Videos |
| :--- | :--- | ---: |
| `Soap_Bottle_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Soap_Bottle_1/single_ref_ep82_per_object_v1/videos_narrow_dr1_lift_arm_decay_v1` | 3 grasp close-ups |
| `Bottle_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bottle_1/single_ref_ep82_per_object_v1/videos_narrow_dr1_lift_arm_decay_v1` | 3 grasp close-ups |
| `Apple_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/single_ref_ep82_small_v4_multireplay/videos_fixed_dr1_model_0` | 3 full-robot + 3 close-ups |
| `Bowl_1` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Bowl_1/single_ref_ep82_rim_v4_staged/videos_fixed_dr1_model_0` | 3 full-robot + 3 close-ups |
| `Cup_6` | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Cup_6/release_single_ref_ep82_dr02_xonly_s12p5_e4/videos_dr02_final` | 3 full-robot + 3 close-ups |
| `Apple_1` bend v5 reference | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/reference_reference-Apple_1_xmove_bend_ep11_position_dr_2p5mm_staged_native_v2_target_xy_2p5mm_full_robot_seed20261020` | 1 full-robot success with 0 m audited pre-contact motion |
| `Apple_1` bend v5 PPO full robot | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/ppo_target_xy_100mm_full_robot_seed20261060` | 1 audited `+/-10 cm` success |
| `Apple_1` bend v5 PPO close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Apple_1/xmove_bend_ep11_v1/videos/ppo_target_xy_100mm_grasp_closeup_seed20261061` | 1 audited `+/-10 cm` success |
| `Potato_1` bend v5 PPO full robot | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/videos/ppo_target_xy_10mm_full_robot_seed20261142` | 1 audited `+/-1 cm` success |
| `Potato_1` bend v5 PPO close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Potato_1/xmove_bend_ep11_v1/videos/ppo_target_xy_10mm_grasp_closeup_seed20261143` | 1 audited `+/-1 cm` success |
| `Tomato_1` bend v5 PPO full robot | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/videos/ppo_target_xy_10mm_full_robot_seed20261140` | 1 audited `+/-1 cm` success |
| `Tomato_1` bend v5 PPO close-up | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/Tomato_1/xmove_bend_ep11_v1/videos/ppo_target_xy_10mm_grasp_closeup_seed20261141` | 1 audited `+/-1 cm` success |
| `Apple_1` strict `+/-20 cm` profile | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/apple_1_xmove_bend_target_xy_200mm_sparse_v5` | 1 full-robot + 1 close-up sparse success; not an edge claim |
| `Potato_1` strict `+/-20 cm` profile | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/potato_1_xmove_bend_target_xy_200mm_sparse_v5` | 1 full-robot + 1 close-up sparse success; not an edge claim |
| `Tomato_1` strict `+/-20 cm` profile | `/mnt/workspace/Jensen/project/g1datagen/SIMPLE/releases/grasp_rl/mjlab_gpu/grasp_anything/v1/videos/tomato_1_xmove_bend_target_xy_200mm_sparse_v5` | 1 full-robot + 1 close-up sparse success; not an edge claim |

The original v4 Apple fixed videos are retained under
`videos/apple_1_xmove_bend_ep11_reference` as legacy evidence. The follow-up additionally backs
up Apple target/base position-DR recordings, Bowl fixed and target-position
recordings, and Potato/Tomato target-position recordings. Every MP4 has a JSON
sidecar reporting native simulator success, lift and finger-contact audit.
Exact release-relative and absolute local paths are listed in
`APPLE_WORKSPACE_V5.md`, `MULTI_OBJECT_WORKSPACE_V5.md`, and
`DR_BEND_FOLLOWUP.md`, and checksummed in `SHA256SUMS`.

## Integrity

Verify the downloaded LFS objects from this directory:

```bash
sha256sum -c SHA256SUMS
```
