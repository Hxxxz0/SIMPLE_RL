#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

stage="${1:-verify}"
physical_gpu="${SIMPLE_PPO_GPU:-0}"
training_seed="${SIMPLE_PPO_SEED:-20260812}"
evaluation_seed="${SIMPLE_PPO_EVAL_SEED:-20260813}"
acceptance_envs="${SIMPLE_PPO_ACCEPTANCE_ENVS:-512}"
acceptance_minimum="${SIMPLE_PPO_ACCEPTANCE_MINIMUM:-0.98}"
object_id="Cup_6"

source_mjcf="${GRASP_ANYTHING_SOURCE_MJCF:-/mnt/workspace/Jensen/.cache/molmo-spaces-resources/objects/thor/20251117/Kitchen Objects/Cup/Prefabs/Cup_6/Cup_6.xml}"
processed_multi="data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2"
processed_single="data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2_single_ref_ep82_shared_transform"
asset="outputs/grasp_rl/other/assets/mjlab_assets/grasp_anything/${object_id}_object_reward_v2"
asset_single="${asset}"
base_asset_multi="outputs/grasp_rl/other/assets/mjlab_assets/xmove_pick/episode82"
run_root="outputs/grasp_rl/other/raw_runs/mjlab_gpu/grasp_anything/${object_id}"
run_bootstrap="${run_root}/v_object_reward3_search_best_single_ep82_seed${training_seed}_env8192_roll24_bootstrap200"
run_dr02="${run_root}/v_object_reward3_search_best_single_ep82_seed${training_seed}_env8192_roll24_dr02_lownoise_frozen_warm_model19_20"
run_dr02_correlated="${run_root}/v_object_reward3_search_best_single_ep82_seed${training_seed}_env8192_roll24_dr02_corr8_std01_warm_model10_20"
run_dr04_pose="${run_root}/v_object_reward3_search_best_single_ep82_seed${training_seed}_env8192_roll24_dr04_pose_lownoise_warm_model10_20"
run_dr04_focused="${run_root}/v_object_reward3_search_best_single_ep82_seed${training_seed}_env8192_roll24_dr04_pose_center008_lr05_warm_model10_20"
run_multi="${run_root}/v_object_reward2_multi_ref_balanced_wide_seed${training_seed}_env12288_roll24_5000"
run_single="${run_bootstrap}"
checkpoint_bootstrap="${SIMPLE_PPO_CHECKPOINT_BOOTSTRAP:-${run_bootstrap}/model_19.pt}"
checkpoint_dr02="${SIMPLE_PPO_CHECKPOINT_DR02:-${run_dr02}/model_10.pt}"
checkpoint_dr04_pose="${SIMPLE_PPO_CHECKPOINT_DR04_POSE:-${run_dr04_pose}/model_19.pt}"
checkpoint_cup6_candidate="${SIMPLE_PPO_CHECKPOINT_CUP6_CANDIDATE:-${run_root}/v_object_reward3_anchor10_mix_global80_focus20_x000_010_y002_010_seed20260818_env8192_roll24_dr02_lr01_fixed_warm_model10_40/model_39.pt}"
checkpoint_multi="${SIMPLE_PPO_CHECKPOINT_MULTI:-${run_multi}/model_4999.pt}"
checkpoint_single="${SIMPLE_PPO_CHECKPOINT_SINGLE:-${run_single}/model_4999.pt}"
ppo_dataset="data/ppo/G1WholebodyGraspAnythingPhysicalPPO-v0/${object_id}"
groot_source="data/simple-teleop/G1WholebodyXMovePickTeleop-v0/level-0"
psi0_template="data/simple/G1WholebodyXMovePickTeleop-v0"

export PYTHONPATH="${repo_root}/src"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export MUJOCO_GL=egl

gpu_cli=(uv run --project mjlab_gpu --no-sync python -m simple.grasp_rl.mjlab_gpu.cli)

wide_dr_profile=(
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

wide_dr=(
  --reference-reward-weight 0.005
  --max-reference-action-deviation 0.7
  --dr-initial-strength 0.05
  --dr-warmup-steps 0
  --dr-ramp-steps 48000
  "${wide_dr_profile[@]}"
)

common_multi=(
  --task grasp_anything
  --asset-bundle "${asset}"
  --reference-processed "${processed_multi}"
  --device cuda:0
  --seed "${training_seed}"
  --reference-selection balanced
  --max-reference-initial-position-offset 0.08
  "${wide_dr[@]}"
)

common_single=(
  --task grasp_anything
  --asset-bundle "${asset_single}"
  --reference-processed "${processed_single}"
  --strict-reference-episode 82
  --device cuda:0
  --seed "${training_seed}"
  --max-reference-initial-position-offset 0.08
  "${wide_dr[@]}"
)

train_core=(
  --learning-rate 0.0003
  --actor-learning-rate-scale 0.05
  --schedule fixed
  --exploration-std 0.02
  --ppo-clip-param 0.1
  --ppo-learning-epochs 5
  --ppo-max-grad-norm 0.5
  --ppo-steps-per-env 24
  --save-interval 10
)

train_scratch=(
  --plan-conditioned-actor
  --scratch-actor-output-scale 0.000001
  --scratch-right-hand-correction 0.00324760 -0.09997928 0.00110806 -0.02096313 -0.05184034 0.00053336 0.12066098
  --scratch-right-arm-correction -0.63216424 -0.05362973 0.14593603 0.29428789 -0.10058338 0.43392509 -0.24306557
  "${train_core[@]}"
)

train_dr_core=(
  --learning-rate 0.0003
  --actor-learning-rate-scale 0.01
  --schedule fixed
  --exploration-std 0.005
  --ppo-clip-param 0.05
  --ppo-learning-epochs 2
  --ppo-max-grad-norm 0.2
  --ppo-steps-per-env 24
  --save-interval 10
)

train_dr_correlated=(
  --learning-rate 0.0003
  --actor-learning-rate-scale 0.01
  --schedule fixed
  --exploration-std 0.01
  --exploration-hold-steps 8
  --ppo-clip-param 0.05
  --ppo-learning-epochs 2
  --ppo-max-grad-norm 0.2
  --ppo-steps-per-env 24
  --save-interval 2
)

train_dr_focused=(
  --learning-rate 0.0003
  --actor-learning-rate-scale 0.05
  --schedule fixed
  --exploration-std 0.005
  --ppo-clip-param 0.05
  --ppo-learning-epochs 2
  --ppo-max-grad-norm 0.2
  --ppo-steps-per-env 24
  --save-interval 2
)

bootstrap_common=(
  --task grasp_anything
  --asset-bundle "${asset_single}"
  --reference-processed "${processed_single}"
  --strict-reference-episode 82
  --device cuda:0
  --seed "${training_seed}"
  --max-reference-initial-position-offset 0.08
  --reference-reward-weight 0.005
  --max-reference-action-deviation 0.7
  --disable-domain-randomization
)

dr02_common=(
  --task grasp_anything
  --asset-bundle "${asset_single}"
  --reference-processed "${processed_single}"
  --strict-reference-episode 82
  --device cuda:0
  --seed "${training_seed}"
  --max-reference-initial-position-offset 0.08
  --reference-reward-weight 0.005
  --max-reference-action-deviation 0.7
  --dr-initial-strength 0.2
  --dr-warmup-steps 0
  --dr-ramp-steps 48000
  "${wide_dr_profile[@]}"
)

dr04_pose_common=(
  --task grasp_anything
  --asset-bundle "${asset_single}"
  --reference-processed "${processed_single}"
  --strict-reference-episode 82
  --device cuda:0
  --seed "${training_seed}"
  --max-reference-initial-position-offset 0.08
  --reference-reward-weight 0.005
  --max-reference-action-deviation 0.7
  --dr-initial-strength 0.4
  --dr-warmup-steps 0
  --dr-ramp-steps 48000
  "${wide_dr_profile[@]}"
  --dr-profile pose_only
)

dr04_focused_common=(
  "${dr04_pose_common[@]}"
  --target-position-jitter-xy 0.025 0.025
  --target-position-offset-center-xy 0.02 0.02
)

# Accepted Cup_6 candidate: one episode-82 reference, global DR 0.2 and an
# object-X-conditioned proposal.  These gains are opt-in so legacy tasks and
# previously trained grasp_anything checkpoints keep their original behavior.
cup6_candidate_common=(
  "${dr02_common[@]}"
  --reference-target-x-arm-gains -12.5 4.0
  --reference-target-y-arm-gains 0 0
)

cup6_candidate_release="${run_root}/release_single_ref_ep82_dr02_xonly_s12p5_e4"

verify_contract() {
  uv run --no-sync python - "${asset}" "${processed_multi}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

asset_root = Path(sys.argv[1])
reference_root = Path(sys.argv[2])
asset = json.loads((asset_root / "manifest.json").read_text())
reference = json.loads((reference_root / "manifest.json").read_text())
contract = asset["object_contract"]
if asset["task"] != "grasp_anything" or contract["reference_task"] != "xmove_pick":
    raise SystemExit("grasp-anything task/reference contract mismatch")
if asset["object_id"] != "Cup_6" or contract["object_id"] != "Cup_6":
    raise SystemExit("this policy directory is not frozen to Cup_6")
episodes = sorted({
    int(episode)
    for split in ("train", "val", "test")
    for episode in reference["splits"][split]
})
if len(episodes) < 2 or 82 not in episodes:
    raise SystemExit("multi-reference data must contain episode 82 and other episodes")
transform_hash = hashlib.sha256(
    (reference_root / "action_transform.npz").read_bytes()
).hexdigest()
if transform_hash != asset["action_transform_sha256"]:
    raise SystemExit("asset/reference action transforms do not match")

frozen = np.asarray(asset["reset"]["actor_observation"], dtype=np.float64)
offsets = {}
for episode in episodes:
    with np.load(
        reference_root / "bc" / f"episode_{episode:06d}.npz",
        allow_pickle=False,
    ) as saved:
        observation = saved["observations"][0]
    offsets[episode] = float(
        np.linalg.norm(frozen[163:166] - observation[163:166])
    )
worst_episode = max(offsets, key=offsets.get)
if offsets[worst_episode] > 0.08:
    raise SystemExit(
        f"reference {worst_episode} offset {offsets[worst_episode]:.6f} m "
        "exceeds the reviewed 0.08 m object-shape allowance"
    )

render = json.loads((asset_root / "render_manifest.json").read_text())
if render["base_manifest_hash"] != asset["manifest_hash"]:
    raise SystemExit("render sidecar is not bound to the Cup_6 physics bundle")
print(json.dumps({
    "task": asset["task"],
    "object_id": asset["object_id"],
    "reference_task": contract["reference_task"],
    "reference_count": len(episodes),
    "reference_offset_mean_m": float(np.mean(list(offsets.values()))),
    "reference_offset_max_m": offsets[worst_episode],
    "reference_offset_max_episode": worst_episode,
    "asset_manifest_hash": asset["manifest_hash"],
    "render_manifest_hash": render["manifest_hash"],
}, indent=2))
PY
}

collect_dataset_split() {
  local split="$1"
  local successes="$2"
  local max_attempts="$3"
  "${gpu_cli[@]}" collect-dataset "${common_multi[@]}" \
    --checkpoint "${checkpoint_multi}" \
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
    --expected-task grasp_anything \
    --expected-dr-strength 1 \
    "$@"
}

evaluate_three_seeds() {
  local mode="$1"
  local checkpoint="$2"
  local minimum="$3"
  local run="$4"
  shift 4
  local -a mode_common=("$@")
  mkdir -p "${run}/acceptance"
  for seed in "${evaluation_seed}" "$((evaluation_seed + 1))" "$((evaluation_seed + 2))"; do
    "${gpu_cli[@]}" evaluate "${mode_common[@]}" \
      --checkpoint "${checkpoint}" \
      --seed "${seed}" \
      --num-envs 512 \
      --episodes 512 \
      --minimum-success-rate "${minimum}" \
      --stress-domain-randomization \
      --smoke | tee "${run}/acceptance/${mode}_seed${seed}_dr1.json"
  done
}

case "${stage}" in
  derive_assets)
    if [[ ! -f "${base_asset_multi}/manifest.json" ]]; then
      echo "derive the episode-82 xmove base asset first" >&2
      exit 1
    fi
    uv run --no-sync python -m simple.grasp_rl.cli derive-grasp-anything-assets \
      --base-bundle "${base_asset_multi}" \
      --source-mjcf "${source_mjcf}" \
      --output "${asset}" \
      --object-id "${object_id}" \
      --grip-width-m 0.083 \
      --mass-kg 0.25 \
      --upright-quaternion-wxyz 0.70710678 0.70710678 0 0 \
      --grasp-frame-position-m 0 -0.027 0 \
      --maximum-grip-force-newtons 80 \
      --table-clearance-m 0.002
    ;;
  derive_assets_single)
    if [[ ! -f "${asset_single}/manifest.json" ]]; then
      "${BASH_SOURCE[0]}" derive_assets
    else
      uv run --no-sync python -m simple.grasp_rl.cli validate-mjlab-assets \
        --bundle "${asset_single}"
    fi
    ;;
  prepare_single)
    uv run --no-sync python -m simple.grasp_rl.cli derive-strict-reference \
      --processed "${processed_multi}" \
      --output "${processed_single}" \
      --episode 82
    ;;
  verify)
    uv run --no-sync python -m simple.grasp_rl.cli validate-mjlab-assets \
      --bundle "${asset}"
    verify_contract
    if [[ -f "${asset_single}/manifest.json" ]]; then
      uv run --no-sync python -m simple.grasp_rl.cli validate-mjlab-assets \
        --bundle "${asset_single}"
    fi
    ;;
  reference_smoke64)
    verify_contract
    "${gpu_cli[@]}" evaluate "${common_multi[@]}" \
      --reference-only \
      --num-envs 64 \
      --episodes 64 \
      --smoke
    ;;
  smoke64_multi)
    verify_contract
    "${gpu_cli[@]}" train "${common_multi[@]}" "${train_scratch[@]}" \
      --output "${run_multi}_smoke64" \
      --num-envs 64 \
      --iterations 1 \
      --smoke
    ;;
  smoke64_single)
    "${gpu_cli[@]}" train "${common_single[@]}" "${train_scratch[@]}" \
      --output "${run_single}_smoke64" \
      --num-envs 64 \
      --iterations 1 \
      --smoke
    ;;
  capacity12288_multi)
    verify_contract
    "${gpu_cli[@]}" train "${common_multi[@]}" "${train_scratch[@]}" \
      --output "${run_multi}_capacity12288" \
      --num-envs 12288 \
      --iterations 2
    ;;
  capacity12288_single)
    "${gpu_cli[@]}" train "${common_single[@]}" "${train_scratch[@]}" \
      --output "${run_single}_capacity12288" \
      --num-envs 12288 \
      --iterations 2
    ;;
  bootstrap_smoke64)
    "${gpu_cli[@]}" train "${bootstrap_common[@]}" "${train_scratch[@]}" \
      --output "${run_bootstrap}_smoke64" \
      --num-envs 64 \
      --iterations 1 \
      --smoke
    ;;
  bootstrap20)
    "${gpu_cli[@]}" train "${bootstrap_common[@]}" "${train_scratch[@]}" \
      --output "${run_bootstrap}" \
      --num-envs 8192 \
      --iterations 20
    ;;
  bootstrap200)
    "${gpu_cli[@]}" train "${bootstrap_common[@]}" "${train_scratch[@]}" \
      --output "${run_bootstrap}" \
      --num-envs 8192 \
      --iterations 200
    ;;
  bootstrap_acceptance)
    "${gpu_cli[@]}" evaluate "${bootstrap_common[@]}" \
      --checkpoint "${checkpoint_bootstrap}" \
      --num-envs 512 \
      --episodes 512 \
      --minimum-success-rate 0.90 \
      --smoke
    ;;
  train20_dr02_warmstart)
    "${gpu_cli[@]}" train "${dr02_common[@]}" "${train_dr_core[@]}" \
      --output "${run_dr02}" \
      --warm-start "${checkpoint_bootstrap}" \
      --warm-start-critic \
      --freeze-actor-normalizer \
      --num-envs 8192 \
      --iterations 20
    ;;
  train20_dr02_correlated_warmstart)
    "${gpu_cli[@]}" train "${dr02_common[@]}" "${train_dr_correlated[@]}" \
      --output "${run_dr02_correlated}" \
      --warm-start "${run_dr02}/model_10.pt" \
      --warm-start-critic \
      --freeze-actor-normalizer \
      --num-envs 8192 \
      --iterations 20
    ;;
  smoke64_dr02_correlated_warmstart)
    "${gpu_cli[@]}" train "${dr02_common[@]}" "${train_dr_correlated[@]}" \
      --output "${run_dr02_correlated}_smoke64" \
      --warm-start "${run_dr02}/model_10.pt" \
      --warm-start-critic \
      --freeze-actor-normalizer \
      --num-envs 64 \
      --iterations 1 \
      --smoke
    ;;
  train20_dr04_pose_warmstart)
    "${gpu_cli[@]}" train "${dr04_pose_common[@]}" "${train_dr_core[@]}" \
      --output "${run_dr04_pose}" \
      --warm-start "${run_dr02}/model_10.pt" \
      --warm-start-critic \
      --freeze-actor-normalizer \
      --save-interval 2 \
      --num-envs 8192 \
      --iterations 20
    ;;
  smoke64_dr04_focused_warmstart)
    "${gpu_cli[@]}" train "${dr04_focused_common[@]}" "${train_dr_focused[@]}" \
      --output "${run_dr04_focused}_smoke64" \
      --warm-start "${run_dr02}/model_10.pt" \
      --warm-start-critic \
      --freeze-actor-normalizer \
      --num-envs 64 \
      --iterations 1 \
      --smoke
    ;;
  train20_dr04_focused_warmstart)
    "${gpu_cli[@]}" train "${dr04_focused_common[@]}" "${train_dr_focused[@]}" \
      --output "${run_dr04_focused}" \
      --warm-start "${run_dr02}/model_10.pt" \
      --warm-start-critic \
      --freeze-actor-normalizer \
      --num-envs 8192 \
      --iterations 20
    ;;
  dr02_acceptance)
    "${gpu_cli[@]}" evaluate "${dr02_common[@]}" \
      --checkpoint "${checkpoint_dr02}" \
      --seed "${evaluation_seed}" \
      --num-envs "${acceptance_envs}" \
      --episodes "${acceptance_envs}" \
      --evaluation-dr-strength 0.2 \
      --minimum-success-rate "${acceptance_minimum}" \
      --smoke
    ;;
  dr04_pose_acceptance)
    "${gpu_cli[@]}" evaluate "${dr04_pose_common[@]}" \
      --checkpoint "${checkpoint_dr04_pose}" \
      --seed "${evaluation_seed}" \
      --num-envs "${acceptance_envs}" \
      --episodes "${acceptance_envs}" \
      --evaluation-dr-strength 0.4 \
      --minimum-success-rate "${acceptance_minimum}" \
      --smoke
    ;;
  cup6_candidate_acceptance)
    mkdir -p "${cup6_candidate_release}/acceptance"
    for seed in 20260814 20260815 20260819; do
      "${gpu_cli[@]}" evaluate "${cup6_candidate_common[@]}" \
        --checkpoint "${checkpoint_cup6_candidate}" \
        --seed "${seed}" \
        --num-envs 512 \
        --episodes 512 \
        --evaluation-dr-strength 0.2 \
        --minimum-success-rate 0.98 \
        --smoke | tee "${cup6_candidate_release}/acceptance/seed${seed}_dr02_512.json"
    done
    ;;
  train5000_multi_warmstart)
    verify_contract
    "${gpu_cli[@]}" train "${common_multi[@]}" "${train_core[@]}" \
      --output "${run_multi}" \
      --warm-start "${checkpoint_bootstrap}" \
      --num-envs 12288 \
      --iterations 5000
    ;;
  train5000_multi)
    verify_contract
    "${gpu_cli[@]}" train "${common_multi[@]}" "${train_scratch[@]}" \
      --output "${run_multi}" \
      --num-envs 12288 \
      --iterations 5000
    ;;
  train5000_single)
    "${gpu_cli[@]}" train "${common_single[@]}" "${train_scratch[@]}" \
      --output "${run_single}" \
      --num-envs 12288 \
      --iterations 5000
    ;;
  acceptance_multi)
    evaluate_three_seeds multi "${checkpoint_multi}" 0.85 "${run_multi}" \
      "${common_multi[@]}"
    ;;
  acceptance_single)
    evaluate_three_seeds single_ep82 "${checkpoint_single}" 0.60 "${run_single}" \
      "${common_single[@]}"
    ;;
  record_multi)
    "${gpu_cli[@]}" record "${common_multi[@]}" \
      --checkpoint "${checkpoint_multi}" \
      --output-dir "${run_multi}/videos_dr1" \
      --num-envs 32 \
      --videos 6 \
      --max-attempts 300 \
      --camera-view grasp_closeup \
      --stress-domain-randomization \
      --smoke
    ;;
  record_cup6_candidate)
    for camera_view in grasp_closeup full_robot; do
      "${gpu_cli[@]}" record "${cup6_candidate_common[@]}" \
        --checkpoint "${checkpoint_cup6_candidate}" \
        --output-dir "${cup6_candidate_release}/videos_dr02_final/${camera_view}" \
        --seed 20260815 \
        --num-envs 32 \
        --videos 3 \
        --max-attempts 100 \
        --camera-view "${camera_view}" \
        --evaluation-dr-strength 0.2 \
        --smoke
    done
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
    echo "usage: $0 {derive_assets|derive_assets_single|prepare_single|verify|reference_smoke64|smoke64_multi|smoke64_single|capacity12288_multi|capacity12288_single|bootstrap_smoke64|bootstrap20|bootstrap200|bootstrap_acceptance|train20_dr02_warmstart|smoke64_dr02_correlated_warmstart|train20_dr02_correlated_warmstart|train20_dr04_pose_warmstart|smoke64_dr04_focused_warmstart|train20_dr04_focused_warmstart|dr02_acceptance|dr04_pose_acceptance|cup6_candidate_acceptance|train5000_multi_warmstart|train5000_multi|train5000_single|acceptance_multi|acceptance_single|record_multi|record_cup6_candidate|dataset_review1|dataset_production500|dataset_all}" >&2
    exit 2
    ;;
esac
