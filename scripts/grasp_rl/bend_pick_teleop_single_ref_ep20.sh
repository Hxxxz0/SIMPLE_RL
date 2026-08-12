#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

stage="${1:-smoke64}"
physical_gpu="${SIMPLE_PPO_GPU:-4}"
evaluation_seed="${SIMPLE_PPO_EVAL_SEED:-20260809}"
processed="data/grasp_rl/G1WholebodyBendPickTeleop-v0/v2_single_ref_ep20"
asset="outputs/grasp_rl/other/assets/mjlab_assets/bend_pick_teleop/episode20_single_ref_v1"
run="outputs/grasp_rl/other/raw_runs/mjlab_gpu/bend_pick_teleop/v_single_ref_ep20_wide_seed20260808_env12288_roll24_3000"
checkpoint="${SIMPLE_PPO_CHECKPOINT:-${run}/model_2999.pt}"
ppo_dataset="data/ppo/G1WholebodyBendPickPhysicalPPO-v0"
groot_source="data/simple-teleop/G1WholebodyBendPickTeleop-v0/level-0"
psi0_template="data/simple/G1WholebodyBendPickTeleop-v0"

export PYTHONPATH="${repo_root}/src"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export MUJOCO_GL=egl

gpu_cli=(uv run --project mjlab_gpu --no-sync python -m simple.grasp_rl.mjlab_gpu.cli)
common=(
  --task bend_pick_teleop
  --asset-bundle "${asset}"
  --reference-processed "${processed}"
  --strict-reference-episode 20
  --device cuda:0
  --seed 20260808
  --reference-reward-weight 0.05
  --max-reference-action-deviation 0.5
  --dr-initial-strength 0.1
  --dr-warmup-steps 0
  --dr-ramp-steps 24000
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
  --ppo-learning-epochs 5
  --ppo-max-grad-norm 0.1
  --ppo-steps-per-env 24
)

ensure_render_assets() {
  if [[ -f "${asset}/render_manifest.json" && -f "${asset}/render_scene.xml" ]]; then
    return
  fi
  uv run --no-sync python -m simple.grasp_rl.cli export-mjlab-render-assets \
    --bundle "${asset}"
}

collect_dataset_split() {
  local split="$1"
  local successes="$2"
  local max_attempts="$3"
  ensure_render_assets
  "${gpu_cli[@]}" collect-dataset "${common[@]}" \
    --checkpoint "${checkpoint}" \
    --output-root "${ppo_dataset}/${split}" \
    --source-dataset "${groot_source}" \
    --psi0-template "${psi0_template}" \
    --num-envs 64 \
    --successes "${successes}" \
    --max-attempts "${max_attempts}" \
    --camera head_stereo_left \
    --width 640 \
    --height 360 \
    --fps 50 \
    --stress-domain-randomization \
    --smoke
}

audit_dataset_split() {
  local split="$1"
  local successes="$2"
  shift 2
  "${gpu_cli[@]}" audit-dataset \
    --dataset-root "${ppo_dataset}/${split}" \
    --expected-successes "${successes}" \
    --expected-task bend_pick_teleop \
    --expected-dr-strength 1 \
    "$@"
}

case "${stage}" in
  prepare)
    export GRASP_RL_WORKER_DEVICES="${physical_gpu}"
    uv run --no-sync python -m simple.grasp_rl.cli prepare \
      --task bend_pick_teleop \
      --dataset data/simple/G1WholebodyBendPickTeleop-v0 \
      --output "${processed}" \
      --episode-id 20 \
      --workers 1 \
      --warmup-steps 60
    ;;
  assets)
    uv run --no-sync python -m simple.grasp_rl.cli export-mjlab-assets \
      --task bend_pick_teleop \
      --output "${asset}" \
      --base-episode 20 \
      --action-transform "${processed}/action_transform.npz" \
      --seed 42 \
      --warmup-steps 60
    ;;
  render_assets)
    ensure_render_assets
    ;;
  smoke64)
    "${gpu_cli[@]}" train "${common[@]}" "${train[@]}" \
      --output "${run}_smoke64" \
      --num-envs 64 \
      --iterations 1 \
      --smoke
    ;;
  capacity12288)
    "${gpu_cli[@]}" train "${common[@]}" "${train[@]}" \
      --output "${run}_capacity2" \
      --num-envs 12288 \
      --iterations 2
    ;;
  train3000)
    "${gpu_cli[@]}" train "${common[@]}" "${train[@]}" \
      --output "${run}" \
      --num-envs 12288 \
      --iterations 3000
    ;;
  benchmark256)
    "${gpu_cli[@]}" benchmark "${common[@]}" \
      --checkpoint "${checkpoint}" \
      --output "${run}/benchmark_seed${evaluation_seed}_full_dr_256.json" \
      --seed "${evaluation_seed}" \
      --num-envs 256 \
      --episodes 256 \
      --stress-domain-randomization \
      --smoke
    ;;
  continue10000)
    "${gpu_cli[@]}" train "${common[@]}" \
      --plan-conditioned-actor \
      --learning-rate 0.001 \
      --actor-learning-rate-scale 0.02 \
      --schedule fixed \
      --exploration-std 0.05 \
      --ppo-learning-epochs 5 \
      --ppo-max-grad-norm 0.1 \
      --ppo-steps-per-env 24 \
      --output "${run}_continue3000_to10000" \
      --num-envs 12288 \
      --iterations 7000 \
      --resume "${checkpoint}"
    ;;
  collect500)
    "${gpu_cli[@]}" collect "${common[@]}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${run}/full_dr_successful_rollouts_500" \
      --num-envs 512 \
      --successes 500 \
      --max-attempts 5000 \
      --stress-domain-randomization \
      --smoke
    ;;
  dataset_review1)
    collect_dataset_split review 1 100
    audit_dataset_split review 1
    ;;
  dataset_production500)
    if [[ ! -f "${ppo_dataset}/review/audit/randomization_coverage.json" ]]; then
      echo "run dataset_review1 and inspect its contact sheet first" >&2
      exit 1
    fi
    collect_dataset_split production 500 5000
    audit_dataset_split production 500 --require-full-dr-coverage
    ;;
  dataset_all)
    collect_dataset_split review 1 100
    audit_dataset_split review 1
    collect_dataset_split production 500 5000
    audit_dataset_split production 500 --require-full-dr-coverage
    ;;
  *)
    echo "usage: $0 {prepare|assets|render_assets|smoke64|capacity12288|train3000|benchmark256|continue10000|collect500|dataset_review1|dataset_production500|dataset_all}" >&2
    exit 2
    ;;
esac
