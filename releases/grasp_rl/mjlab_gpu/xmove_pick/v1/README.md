# xmove_pick mjlab GPU PPO v1

This is a self-contained, audited `xmove_pick` release for the CUDA
MuJoCo-Warp backend.  It contains the selected RSL-RL PPO checkpoint, its
training initialization, frozen simulator/controller assets, the processed
reference bank, paired evaluation reports, successful trajectories and two
successful full-DR videos.

## Result and scope

The selected `policy_model_299.pt` collected 58,982,400 fresh on-policy
transitions.  Its last update used 8,192 environments x 24 steps, five PPO
epochs and four minibatches.  The checkpoint records 20 CUDA optimizer steps
and non-zero actor/critic parameter deltas.

On two 128-world full-DR seeds, deterministic PPO completed 173/256 episodes
(67.58%).  The noise-matched proposal-only baseline also completed 173/256;
PPO mean task return was 24.264 versus 24.124.  Thus this release demonstrates
no success regression and a small return gain, not a statistically established
success-rate improvement.  It is below the 70% target.  `reference-only` is a
clean expert replay upper-bound diagnostic; `proposal-only` is the correct
no-PPO comparison because it receives the same noisy reference proposal as
the actor.

## Install and verify

Git LFS is required because checkpoints, meshes, videos and trajectory NPZs
are tracked as LFS objects.

```bash
git lfs pull
uv sync --project mjlab_gpu --frozen
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
PY=mjlab_gpu/.venv/bin/python
REL=releases/grasp_rl/mjlab_gpu/xmove_pick/v1

$PY -m simple.grasp_rl.mjlab_gpu.cli verify-release --release-dir "$REL"
```

The lock requires Python 3.10 and pins Torch 2.7.0/cu126, MuJoCo 3.11.0,
RSL-RL 5.2.0 and mjlab commit
`731fa27b8c023d3d5258501c694979425d2a37fd`.

## Reproduce the paired benchmark

`benchmark` constructs reference-only, proposal-only and PPO environments
from the same seed.  It rejects mismatched physical-world or proposal-context
hashes and reports the exact paired McNemar test.

```bash
$PY -m simple.grasp_rl.mjlab_gpu.cli benchmark \
  --task xmove_pick \
  --asset-bundle "$REL/assets/episode82" \
  --reference-processed "$REL/reference/v2" \
  --checkpoint "$REL/checkpoints/policy_model_299.pt" \
  --num-envs 128 --episodes 128 --smoke --device cuda:0 \
  --seed 42 --stress-domain-randomization \
  --output /tmp/xmove_pick_seed42.json
```

MuJoCo-Warp contact/sensor reductions are not guaranteed to be bitwise stable
across independent simulator construction.  Exact initial physical-world and
proposal-context hashes are therefore hard gates; policy-state bitwise equality
is retained as a diagnostic in each report.

## Collect trajectories and videos

Only simulator-success episodes are exported by `collect`.  Video selection
also requires at least five consecutive steps of real finger closure and
diverse finger contact; diagnostic fallback is disabled here.

```bash
$PY -m simple.grasp_rl.mjlab_gpu.cli collect \
  --task xmove_pick --asset-bundle "$REL/assets/episode82" \
  --reference-processed "$REL/reference/v2" \
  --checkpoint "$REL/checkpoints/policy_model_299.pt" \
  --num-envs 64 --device cuda:0 --seed 44 \
  --successes 3 --max-attempts 128 --stress-domain-randomization \
  --output-dir /tmp/xmove_pick_trajectories

$PY -m simple.grasp_rl.mjlab_gpu.cli record \
  --task xmove_pick --asset-bundle "$REL/assets/episode82" \
  --reference-processed "$REL/reference/v2" \
  --checkpoint "$REL/checkpoints/policy_model_299.pt" \
  --num-envs 32 --device cuda:0 --seed 42 \
  --videos 1 --max-attempts 128 --width 960 --height 540 --fps 50 \
  --camera-view full_robot --stress-domain-randomization \
  --output-dir /tmp/xmove_pick_video
```

## Train, warm-start, resume and fine-tune

The reviewed formal configuration uses 8,192 CUDA environments.  This command
restarts at full DR from the published initialization and restores the critic,
but creates a fresh optimizer:

```bash
$PY -m simple.grasp_rl.mjlab_gpu.cli train \
  --task xmove_pick --asset-bundle "$REL/assets/episode82" \
  --reference-processed "$REL/reference/v2" \
  --warm-start "$REL/checkpoints/initial_model_100.pt" --warm-start-critic \
  --num-envs 8192 --device cuda:0 --seed 344 --iterations 20000 \
  --initial-vector-step 240000 --dr-initial-strength 1.0 \
  --dr-ramp-steps 240000 --dr-profile full \
  --learning-rate 1e-3 --actor-learning-rate-scale 0.02 \
  --schedule fixed --exploration-std 0.05 --ppo-clip-param 0.2 \
  --ppo-learning-epochs 5 --ppo-max-grad-norm 0.1 \
  --ppo-steps-per-env 24 --freeze-actor-normalizer \
  --output outputs/grasp_rl/xmove_pick/reproduced_full_dr
```

For exact continuation, replace the two warm-start options and the initial
vector step with:

```text
--resume "$REL/checkpoints/policy_model_299.pt"
```

Keep task, seed, environment count, DR settings, optimizer split and PPO
hyperparameters identical for exact resume.  To fine-tune with a new optimizer,
use the selected policy with `--warm-start ... --warm-start-critic` instead.
Legacy CPU checkpoints remain supported only as explicit actor warm starts;
they cannot be presented as exact GPU PPO resumes.
