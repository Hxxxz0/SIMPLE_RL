#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

stage="${1:-smoke64}"
physical_gpu="${SIMPLE_PPO_GPU:-0}"
training_seed="${SIMPLE_PPO_SEED:-20260811}"
evaluation_seed="${SIMPLE_PPO_EVAL_SEED:-20260812}"

processed="data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2"
asset="outputs/grasp_rl/other/assets/mjlab_assets/xmove_pick/episode82"
run="outputs/grasp_rl/other/raw_runs/mjlab_gpu/xmove_pick/v_multi_ref_balanced_cycle_wide_seed${training_seed}_env12288_roll24_5000"
checkpoint="${SIMPLE_PPO_CHECKPOINT:-${run}/model_4999.pt}"

export PYTHONPATH="${repo_root}/src"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export MUJOCO_GL=egl

gpu_cli=(uv run --project mjlab_gpu --no-sync python -m simple.grasp_rl.mjlab_gpu.cli)

common=(
  --task xmove_pick
  --asset-bundle "${asset}"
  --reference-processed "${processed}"
  --device cuda:0
  --seed "${training_seed}"
  --reference-selection balanced
  --max-reference-initial-position-offset 0.05
  --reference-reward-weight 0.05
  --max-reference-action-deviation 0.5
  --dr-initial-strength 0.1
  --dr-warmup-steps 0
  --dr-ramp-steps 48000
  --dr-profile full
  --target-position-jitter-xy 0.05 0.05
  --target-yaw-jitter 0.30
  --destination-position-jitter-xy 0 0
  --destination-yaw-jitter 0
  --distractor-position-jitter-xy 0.05 0.05
  --distractor-yaw-jitter 0.40
  --robot-base-position-jitter-xy 0.02 0.02
  --robot-base-yaw-jitter 0.05
  --target-mass-scale 0.6 1.4
  --friction-scale 0.5 1.5
  --joint-damping-scale 0.8 1.2
  --actuator-strength-scale 0.85 1.15
  --action-delay-max-steps 1
  --reference-action-noise-std 0.004
  --reference-position-noise-std 0.005
  --reference-phase-noise-std 0.02
  --reference-future-dropout-probability 0.05
)

train=(
  --plan-conditioned-actor
  --scratch-actor-output-scale 0.001
  --learning-rate 0.001
  --actor-learning-rate-scale 0.02
  --schedule fixed
  --exploration-std 0.05
  --ppo-clip-param 0.2
  --ppo-learning-epochs 5
  --ppo-max-grad-norm 0.1
  --ppo-steps-per-env 24
)

verify_multi_reference_contract() {
  uv run --no-sync python - "${asset}" "${processed}" 0.05 <<'PY'
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

asset_root = Path(sys.argv[1])
reference_root = Path(sys.argv[2])
maximum_offset = float(sys.argv[3])
asset = json.loads((asset_root / "manifest.json").read_text())
reference = json.loads((reference_root / "manifest.json").read_text())
episodes = sorted({
    int(episode)
    for split in ("train", "val", "test")
    for episode in reference["splits"][split]
})
base_episode = asset.get("base_episode")
if base_episode not in episodes:
    raise SystemExit(
        f"asset base_episode={base_episode!r} is not in the processed references"
    )
if len(episodes) < 2:
    raise SystemExit("multi-reference training requires at least two usable episodes")
reference_action_hash = reference.get("action_transform_sha256")
if reference_action_hash is None:
    reference_action_hash = hashlib.sha256(
        (reference_root / "action_transform.npz").read_bytes()
    ).hexdigest()
if asset["action_transform_sha256"] != reference_action_hash:
    raise SystemExit("asset/reference action transforms do not match")

asset_observation = np.asarray(
    asset["reset"]["actor_observation"], dtype=np.float64
)
offsets = {}
for episode in episodes:
    path = reference_root / "bc" / f"episode_{episode:06d}.npz"
    with np.load(path, allow_pickle=False) as saved:
        observation = saved["observations"][0]
    if observation.shape == (331,):
        asset_primary = asset_observation[163:166]
        reference_primary = observation[163:166]
    elif observation.shape == (192,):
        asset_primary = asset_observation[132:135]
        reference_primary = observation[132:135]
    else:
        raise SystemExit(
            f"episode {episode} has unsupported observation shape {observation.shape}"
        )
    offsets[episode] = float(np.linalg.norm(asset_primary - reference_primary))
worst_episode = max(offsets, key=offsets.get)
worst_offset = offsets[worst_episode]
if worst_offset > maximum_offset:
    raise SystemExit(
        "asset/reference initial primary-position mismatch: "
        f"episode {worst_episode} is {worst_offset:.6f} m away, "
        f"limit is {maximum_offset:.6f} m"
    )
print(json.dumps({
    "task": reference["task"],
    "usable_reference_count": len(episodes),
    "usable_reference_episodes": episodes,
    "asset_base_episode": base_episode,
    "action_transform_sha256": asset["action_transform_sha256"],
    "initial_position_offset_mean_metres": float(np.mean(list(offsets.values()))),
    "initial_position_offset_max_metres": worst_offset,
    "initial_position_offset_max_episode": worst_episode,
    "initial_position_offset_limit_metres": maximum_offset,
}, indent=2))
PY
}

case "${stage}" in
  verify)
    verify_multi_reference_contract
    uv run --no-sync python -m simple.grasp_rl.cli validate-mjlab-assets \
      --bundle "${asset}"
    ;;
  reference_gate72)
    verify_multi_reference_contract
    "${gpu_cli[@]}" evaluate "${common[@]}" \
      --reference-only \
      --num-envs 72 \
      --episodes 72 \
      --minimum-success-rate 0.8 \
      --smoke
    ;;
  smoke64)
    verify_multi_reference_contract
    "${gpu_cli[@]}" train "${common[@]}" "${train[@]}" \
      --output "${run}_smoke64" \
      --num-envs 64 \
      --iterations 1 \
      --smoke
    ;;
  capacity12288)
    verify_multi_reference_contract
    "${gpu_cli[@]}" train "${common[@]}" "${train[@]}" \
      --output "${run}_capacity12288" \
      --num-envs 12288 \
      --iterations 2
    ;;
  train5000)
    verify_multi_reference_contract
    "${gpu_cli[@]}" train "${common[@]}" "${train[@]}" \
      --output "${run}" \
      --num-envs 12288 \
      --iterations 5000
    ;;
  benchmark256)
    verify_multi_reference_contract
    "${gpu_cli[@]}" benchmark "${common[@]}" \
      --checkpoint "${checkpoint}" \
      --output "${run}/benchmark_seed${evaluation_seed}_wide_dr_256.json" \
      --seed "${evaluation_seed}" \
      --num-envs 256 \
      --episodes 256 \
      --stress-domain-randomization \
      --smoke
    ;;
  *)
    echo "usage: $0 {verify|reference_gate72|smoke64|capacity12288|train5000|benchmark256}" >&2
    exit 2
    ;;
esac
