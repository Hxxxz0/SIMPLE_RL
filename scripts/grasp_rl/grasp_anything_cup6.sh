#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

stage="${1:-verify}"
physical_gpu="${SIMPLE_PPO_GPU:-0}"
training_seed="${SIMPLE_PPO_SEED:-20260812}"
# Keep the new interior curriculum reproducible by default without changing
# the seed used by any of the pre-existing PPO stages.  An explicit global
# seed still overrides this default for backwards-compatible automation.
interior_training_seed="${SIMPLE_PPO_INTERIOR_SEED:-${SIMPLE_PPO_SEED:-20260824}}"
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
interior_stage_root="${run_root}/v_object_reward3_single_ep82_interior_x008_020_y010_seed${interior_training_seed}_env8192_roll24"
run_interior_s010="${interior_stage_root}_fixed_s010_20"
run_interior_s015="${interior_stage_root}_fixed_s015_20"
run_interior_s020="${interior_stage_root}_fixed_s020_20"
frontier_label="${SIMPLE_PPO_FRONTIER_LABEL:-s017x_g75_std02_lr05_focus50}"
frontier_iterations="${SIMPLE_PPO_FRONTIER_ITERATIONS:-12}"
run_interior_frontier="${SIMPLE_PPO_FRONTIER_OUTPUT:-${interior_stage_root}_frontier_${frontier_label}_${frontier_iterations}}"
run_interior_s030="${interior_stage_root}_fixed_s030_20"
run_interior_s040="${interior_stage_root}_fixed_s040_20"
run_interior_s060="${interior_stage_root}_fixed_s060_20"
run_interior_s080="${interior_stage_root}_fixed_s080_20"
run_interior="${interior_stage_root}_fixed_s100_100"
run_multi="${run_root}/v_object_reward2_multi_ref_balanced_wide_seed${training_seed}_env12288_roll24_5000"
run_single="${run_bootstrap}"
checkpoint_bootstrap="${SIMPLE_PPO_CHECKPOINT_BOOTSTRAP:-${run_bootstrap}/model_19.pt}"
checkpoint_dr02="${SIMPLE_PPO_CHECKPOINT_DR02:-${run_dr02}/model_10.pt}"
checkpoint_dr04_pose="${SIMPLE_PPO_CHECKPOINT_DR04_POSE:-${run_dr04_pose}/model_19.pt}"
checkpoint_cup6_candidate="${SIMPLE_PPO_CHECKPOINT_CUP6_CANDIDATE:-${run_root}/v_object_reward3_anchor10_mix_global80_focus20_x000_010_y002_010_seed20260818_env8192_roll24_dr02_lr01_fixed_warm_model10_40/model_39.pt}"
checkpoint_interior_s010="${SIMPLE_PPO_CHECKPOINT_INTERIOR_S010:-${run_interior_s010}/model_19.pt}"
# Fixed 512-world evaluation selected model_10 (76.76%) over model_19
# (76.37%) for curriculum promotion.  The environment seed is unchanged.
checkpoint_interior_s015="${SIMPLE_PPO_CHECKPOINT_INTERIOR_S015:-${run_interior_s015}/model_10.pt}"
checkpoint_interior_s020="${SIMPLE_PPO_CHECKPOINT_INTERIOR_S020:-${run_interior_s020}/model_19.pt}"
checkpoint_interior_frontier="${SIMPLE_PPO_CHECKPOINT_INTERIOR_FRONTIER:-${run_interior_frontier}/model_$((frontier_iterations - 1)).pt}"
checkpoint_interior_frontier_warm_start="${SIMPLE_PPO_CHECKPOINT_INTERIOR_FRONTIER_WARM_START:-${checkpoint_interior_s015}}"
checkpoint_interior_s030="${SIMPLE_PPO_CHECKPOINT_INTERIOR_S030:-${run_interior_s030}/model_19.pt}"
checkpoint_interior_s040="${SIMPLE_PPO_CHECKPOINT_INTERIOR_S040:-${run_interior_s040}/model_19.pt}"
checkpoint_interior_s060="${SIMPLE_PPO_CHECKPOINT_INTERIOR_S060:-${run_interior_s060}/model_19.pt}"
checkpoint_interior_s080="${SIMPLE_PPO_CHECKPOINT_INTERIOR_S080:-${run_interior_s080}/model_19.pt}"
checkpoint_interior="${SIMPLE_PPO_CHECKPOINT_INTERIOR:-${run_interior}/model_99.pt}"
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

# The source trajectory places the Cup_6 center at x=-0.313 m, only 12 mm
# inside the table edge.  At full curriculum strength this profile moves the
# cup into a safe, reachable interior rectangle: x offset +0.08..+0.20 m and
# y offset -0.10..+0.10 m.  Scaling both the center and jitter makes the first
# training worlds overlap the accepted policy before PPO expands the reach.
interior_center_x=0.14
interior_center_y=0.0
interior_jitter_x=0.06
interior_jitter_y=0.10
interior_common=(
  --task grasp_anything
  --asset-bundle "${asset_single}"
  --reference-processed "${processed_single}"
  --strict-reference-episode 82
  --device cuda:0
  --seed "${interior_training_seed}"
  --max-reference-initial-position-offset 0.08
  --reference-reward-weight 0.005
  --max-reference-action-deviation 0.7
  --dr-initial-strength 1
  --dr-warmup-steps 0
  --dr-ramp-steps 1
  --dr-profile pose_only
  --target-position-jitter-xy "${interior_jitter_x}" "${interior_jitter_y}"
  --target-position-offset-center-xy "${interior_center_x}" "${interior_center_y}"
  --target-yaw-jitter 0.15
  --destination-position-jitter-xy 0 0
  --destination-yaw-jitter 0
  --distractor-position-jitter-xy 0 0
  --distractor-yaw-jitter 0
  --robot-base-position-jitter-xy 0 0
  --robot-base-yaw-jitter 0
  # The accepted small-DR proposal (-12.5, 4.0) over-corrects in this
  # one-sided interior workspace.  A fixed 512-world scan selected this milder
  # proposal; PPO still owns the residual adaptation for each observed pose.
  --reference-target-x-arm-gains -7.5 2.4
  --reference-target-y-arm-gains 0 0
)

interior_s010_pose=(
  --target-position-jitter-xy 0.006 0.010
  --target-position-offset-center-xy 0.014 0
  --target-yaw-jitter 0.015
)
interior_s015_pose=(
  --target-position-jitter-xy 0.009 0.015
  --target-position-offset-center-xy 0.021 0
  --target-yaw-jitter 0.0225
)
interior_s020_pose=(
  --target-position-jitter-xy 0.012 0.020
  --target-position-offset-center-xy 0.028 0
  --target-yaw-jitter 0.030
)
# Move the difficult X frontier by only 2 mm while preserving half of each
# training batch in the accepted s015 distribution.  Evaluation remains
# uniform over the new 0.014..0.032 m X interval, so the focus mixture cannot
# inflate the acceptance result.
interior_frontier_s017x_pose=(
  --target-position-jitter-xy
    "${SIMPLE_PPO_FRONTIER_X_JITTER:-0.009}"
    "${SIMPLE_PPO_FRONTIER_Y_JITTER:-0.015}"
  --target-position-offset-center-xy
    "${SIMPLE_PPO_FRONTIER_X_CENTER:-0.023}" 0
  --target-position-focus-probability "${SIMPLE_PPO_FRONTIER_FOCUS_PROBABILITY:-0.50}"
  --target-position-focus-jitter-xy
    "${SIMPLE_PPO_FRONTIER_FOCUS_X_JITTER:-0.009}"
    "${SIMPLE_PPO_FRONTIER_FOCUS_Y_JITTER:-0.015}"
  --target-position-focus-offset-center-xy
    "${SIMPLE_PPO_FRONTIER_FOCUS_X_CENTER:-0.021}" 0
  --target-yaw-jitter "${SIMPLE_PPO_FRONTIER_YAW_JITTER:-0.0225}"
)
interior_frontier_s017x_eval_pose=(
  --target-position-jitter-xy
    "${SIMPLE_PPO_FRONTIER_X_JITTER:-0.009}"
    "${SIMPLE_PPO_FRONTIER_Y_JITTER:-0.015}"
  --target-position-offset-center-xy
    "${SIMPLE_PPO_FRONTIER_X_CENTER:-0.023}" 0
  --target-position-focus-probability 0
  --target-yaw-jitter "${SIMPLE_PPO_FRONTIER_YAW_JITTER:-0.0225}"
)
interior_s030_pose=(
  --target-position-jitter-xy 0.018 0.030
  --target-position-offset-center-xy 0.042 0
  --target-yaw-jitter 0.045
)
interior_s040_pose=(
  --target-position-jitter-xy 0.024 0.040
  --target-position-offset-center-xy 0.056 0
  --target-yaw-jitter 0.060
)
interior_s060_pose=(
  --target-position-jitter-xy 0.036 0.060
  --target-position-offset-center-xy 0.084 0
  --target-yaw-jitter 0.090
)
interior_s080_pose=(
  --target-position-jitter-xy 0.048 0.080
  --target-position-offset-center-xy 0.112 0
  --target-yaw-jitter 0.120
)
interior_s100_pose=(
  --target-position-jitter-xy 0.060 0.100
  --target-position-offset-center-xy 0.140 0
  --target-yaw-jitter 0.150
)

train_interior=(
  --learning-rate 0.0003
  --actor-learning-rate-scale 0.02
  --schedule fixed
  --exploration-std 0.01
  --exploration-hold-steps 8
  --ppo-clip-param 0.05
  --ppo-learning-epochs 2
  --ppo-max-grad-norm 0.2
  --ppo-steps-per-env 24
  --save-interval 5
)

train_interior_frontier=(
  --learning-rate 0.0003
  --actor-learning-rate-scale "${SIMPLE_PPO_FRONTIER_ACTOR_LR_SCALE:-0.05}"
  --schedule fixed
  --exploration-std "${SIMPLE_PPO_FRONTIER_EXPLORATION_STD:-0.02}"
  --exploration-hold-steps 8
  --ppo-clip-param 0.05
  --ppo-learning-epochs 2
  --ppo-max-grad-norm 0.2
  --ppo-steps-per-env 24
  --save-interval 4
)

interior_frontier_x_gains=(
  "${SIMPLE_PPO_FRONTIER_X_SHOULDER_GAIN:--7.5}"
  "${SIMPLE_PPO_FRONTIER_X_ELBOW_GAIN:-2.4}"
)
interior_frontier_y_gains=(
  "${SIMPLE_PPO_FRONTIER_Y_SHOULDER_GAIN:-0}"
  "${SIMPLE_PPO_FRONTIER_Y_WRIST_GAIN:-0}"
)
interior_frontier_positive_y_args=()
if [[ -n "${SIMPLE_PPO_FRONTIER_POSITIVE_Y_SHOULDER_GAIN:-}" ]] || \
   [[ -n "${SIMPLE_PPO_FRONTIER_POSITIVE_Y_WRIST_GAIN:-}" ]]; then
  if [[ -z "${SIMPLE_PPO_FRONTIER_POSITIVE_Y_SHOULDER_GAIN:-}" ]] || \
     [[ -z "${SIMPLE_PPO_FRONTIER_POSITIVE_Y_WRIST_GAIN:-}" ]]; then
    echo "positive-Y shoulder and wrist gains must be set together" >&2
    exit 2
  fi
  interior_frontier_positive_y_args=(
    --reference-target-positive-y-arm-gains
    "${SIMPLE_PPO_FRONTIER_POSITIVE_Y_SHOULDER_GAIN}"
    "${SIMPLE_PPO_FRONTIER_POSITIVE_Y_WRIST_GAIN}"
  )
fi

train_fixed_interior_stage() {
  local output="$1"
  local warm_start="$2"
  local iterations="$3"
  local pose_name="$4"
  local -n pose_args="${pose_name}"
  verify_interior_workspace
  require_fresh_training_output "${output}"
  "${gpu_cli[@]}" train "${interior_common[@]}" "${pose_args[@]}" \
    "${train_interior[@]}" \
    --output "${output}" \
    --warm-start "${warm_start}" \
    --warm-start-critic \
    --num-envs 8192 \
    --iterations "${iterations}"
}

require_fresh_training_output() {
  local output="$1"
  if [[ -d "${output}" ]] && [[ -z "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
    return
  fi
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite existing PPO output: ${output}" >&2
    echo "choose a new stage/label or explicitly point at the existing checkpoint" >&2
    exit 1
  fi
}

evaluate_fixed_interior_stage() {
  local checkpoint="$1"
  local output="$2"
  local pose_name="$3"
  local minimum="$4"
  local effective_output="${SIMPLE_PPO_ACCEPTANCE_OUTPUT:-${output}}"
  local effective_minimum="${SIMPLE_PPO_MINIMUM_SUCCESS_RATE:-${minimum}}"
  local -n pose_args="${pose_name}"
  mkdir -p "$(dirname "${effective_output}")"
  "${gpu_cli[@]}" evaluate "${interior_common[@]}" "${pose_args[@]}" \
    --checkpoint "${checkpoint}" \
    --seed "${evaluation_seed}" \
    --num-envs "${acceptance_envs}" \
    --episodes "${acceptance_envs}" \
    --evaluation-dr-strength 1 \
    --minimum-success-rate "${effective_minimum}" \
    --smoke | tee "${effective_output}"
}

verify_interior_workspace() {
  local center_x="${1:-${interior_center_x}}"
  local center_y="${2:-${interior_center_y}}"
  local jitter_x="${3:-${interior_jitter_x}}"
  local jitter_y="${4:-${interior_jitter_y}}"
  local minimum_margin="${5:-0.03}"
  uv run --no-sync python - \
    "${asset_single}" \
    "${center_x}" "${center_y}" \
    "${jitter_x}" "${jitter_y}" "${minimum_margin}" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

asset_root = Path(sys.argv[1])
center_x, center_y, jitter_x, jitter_y, minimum_margin = map(float, sys.argv[2:])
manifest = json.loads((asset_root / "manifest.json").read_text())
root = ET.parse(asset_root / "scene.xml").getroot()
table_body = root.find(".//body[@name='table']")
if table_body is None:
    raise SystemExit("Cup_6 scene is missing the table body")
table_geom = table_body.find("geom[@name='table_geom']")
if table_geom is None:
    raise SystemExit("Cup_6 scene is missing table_geom")

table_center = [float(value) for value in table_body.attrib["pos"].split()[:2]]
table_half = [float(value) for value in table_geom.attrib["size"].split()[:2]]
object_center = [float(value) for value in manifest["reset"]["initial_object_pos"][:2]]
object_half = [float(value) for value in manifest["object_contract"]["half_extents_m"][:2]]
offset_bounds = (
    (center_x - jitter_x, center_x + jitter_x),
    (center_y - jitter_y, center_y + jitter_y),
)
object_bounds = tuple(
    (object_center[axis] + offset_bounds[axis][0],
     object_center[axis] + offset_bounds[axis][1])
    for axis in range(2)
)
table_bounds = tuple(
    (table_center[axis] - table_half[axis],
     table_center[axis] + table_half[axis])
    for axis in range(2)
)
margins = {
    "x_min": object_bounds[0][0] - object_half[0] - table_bounds[0][0],
    "x_max": table_bounds[0][1] - object_bounds[0][1] - object_half[0],
    "y_min": object_bounds[1][0] - object_half[1] - table_bounds[1][0],
    "y_max": table_bounds[1][1] - object_bounds[1][1] - object_half[1],
}
if min(margins.values()) < minimum_margin:
    raise SystemExit(
        f"interior workspace violates {minimum_margin:.3f} m table margin: {margins}"
    )
print(json.dumps({
    "target_offset_bounds_xy_m": offset_bounds,
    "target_center_bounds_xy_m": object_bounds,
    "table_bounds_xy_m": table_bounds,
    "object_half_extents_xy_m": object_half,
    "table_edge_margins_m": margins,
    "minimum_required_margin_m": minimum_margin,
}, indent=2))
PY
}

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
  verify_interior_workspace)
    verify_interior_workspace
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
  smoke64_interior)
    verify_interior_workspace
    "${gpu_cli[@]}" train "${interior_common[@]}" "${interior_s010_pose[@]}" \
      "${train_interior[@]}" \
      --output "${run_interior_s010}_smoke64" \
      --warm-start "${checkpoint_cup6_candidate}" \
      --warm-start-critic \
      --num-envs 64 \
      --iterations 1 \
      --smoke
    ;;
  train_interior_s010)
    train_fixed_interior_stage "${run_interior_s010}" \
      "${checkpoint_cup6_candidate}" 20 interior_s010_pose
    ;;
  train_interior_s015)
    train_fixed_interior_stage "${run_interior_s015}" \
      "${checkpoint_interior_s010}" 20 interior_s015_pose
    ;;
  train_interior_s020)
    train_fixed_interior_stage "${run_interior_s020}" \
      "${checkpoint_interior_s015}" 20 interior_s020_pose
    ;;
  train_interior_frontier_s017x)
    verify_interior_workspace \
      "${SIMPLE_PPO_FRONTIER_X_CENTER:-0.023}" 0 \
      "${SIMPLE_PPO_FRONTIER_X_JITTER:-0.009}" \
      "${SIMPLE_PPO_FRONTIER_Y_JITTER:-0.015}" \
      "${SIMPLE_PPO_FRONTIER_MIN_TABLE_MARGIN:--0.03}"
    require_fresh_training_output "${run_interior_frontier}"
    "${gpu_cli[@]}" train "${interior_common[@]}" \
      --reference-target-x-arm-gains "${interior_frontier_x_gains[@]}" \
      --reference-target-y-arm-gains "${interior_frontier_y_gains[@]}" \
      "${interior_frontier_positive_y_args[@]}" \
      "${interior_frontier_s017x_pose[@]}" \
      "${train_interior_frontier[@]}" \
      --output "${run_interior_frontier}" \
      --warm-start "${checkpoint_interior_frontier_warm_start}" \
      --warm-start-critic \
      --num-envs 8192 \
      --iterations "${frontier_iterations}"
    ;;
  train_interior_s030)
    train_fixed_interior_stage "${run_interior_s030}" \
      "${checkpoint_interior_s020}" 20 interior_s030_pose
    ;;
  train_interior_s040)
    train_fixed_interior_stage "${run_interior_s040}" \
      "${checkpoint_interior_s030}" 20 interior_s040_pose
    ;;
  train_interior_s060)
    train_fixed_interior_stage "${run_interior_s060}" \
      "${checkpoint_interior_s040}" 20 interior_s060_pose
    ;;
  train_interior_s080)
    train_fixed_interior_stage "${run_interior_s080}" \
      "${checkpoint_interior_s060}" 20 interior_s080_pose
    ;;
  train_interior_s100)
    train_fixed_interior_stage "${run_interior}" \
      "${checkpoint_interior_s080}" 100 interior_s100_pose
    ;;
  accept_interior_s010)
    evaluate_fixed_interior_stage "${checkpoint_interior_s010}" \
      "${run_interior_s010}/acceptance/seed20260824_s010_512.json" \
      interior_s010_pose 0.85
    ;;
  accept_interior_s015)
    evaluate_fixed_interior_stage "${checkpoint_interior_s015}" \
      "${run_interior_s015}/acceptance/seed20260824_s015_512.json" \
      interior_s015_pose 0.75
    ;;
  accept_interior_s020)
    evaluate_fixed_interior_stage "${checkpoint_interior_s020}" \
      "${run_interior_s020}/acceptance/seed20260824_s020_512.json" \
      interior_s020_pose 0.70
    ;;
  accept_interior_frontier_s017x)
    verify_interior_workspace \
      "${SIMPLE_PPO_FRONTIER_X_CENTER:-0.023}" 0 \
      "${SIMPLE_PPO_FRONTIER_X_JITTER:-0.009}" \
      "${SIMPLE_PPO_FRONTIER_Y_JITTER:-0.015}" \
      "${SIMPLE_PPO_FRONTIER_MIN_TABLE_MARGIN:--0.03}"
    original_interior_common=("${interior_common[@]}")
    interior_common+=(
      --reference-target-x-arm-gains "${interior_frontier_x_gains[@]}"
      --reference-target-y-arm-gains "${interior_frontier_y_gains[@]}"
      "${interior_frontier_positive_y_args[@]}"
    )
    evaluate_fixed_interior_stage "${checkpoint_interior_frontier}" \
      "${run_interior_frontier}/acceptance/seed20260824_frontier_512.json" \
      interior_frontier_s017x_eval_pose 0.70
    interior_common=("${original_interior_common[@]}")
    ;;
  accept_interior_s030)
    evaluate_fixed_interior_stage "${checkpoint_interior_s030}" \
      "${run_interior_s030}/acceptance/seed20260824_s030_512.json" \
      interior_s030_pose 0.65
    ;;
  accept_interior_s040)
    evaluate_fixed_interior_stage "${checkpoint_interior_s040}" \
      "${run_interior_s040}/acceptance/seed20260824_s040_512.json" \
      interior_s040_pose 0.60
    ;;
  accept_interior_s060)
    evaluate_fixed_interior_stage "${checkpoint_interior_s060}" \
      "${run_interior_s060}/acceptance/seed20260824_s060_512.json" \
      interior_s060_pose 0.55
    ;;
  accept_interior_s080)
    evaluate_fixed_interior_stage "${checkpoint_interior_s080}" \
      "${run_interior_s080}/acceptance/seed20260824_s080_512.json" \
      interior_s080_pose 0.50
    ;;
  train_interior_curriculum)
    "${BASH_SOURCE[0]}" train_interior_s010
    "${BASH_SOURCE[0]}" accept_interior_s010
    "${BASH_SOURCE[0]}" train_interior_s015
    "${BASH_SOURCE[0]}" accept_interior_s015
    "${BASH_SOURCE[0]}" train_interior_s020
    "${BASH_SOURCE[0]}" accept_interior_s020
    "${BASH_SOURCE[0]}" train_interior_s030
    "${BASH_SOURCE[0]}" accept_interior_s030
    "${BASH_SOURCE[0]}" train_interior_s040
    "${BASH_SOURCE[0]}" accept_interior_s040
    "${BASH_SOURCE[0]}" train_interior_s060
    "${BASH_SOURCE[0]}" accept_interior_s060
    "${BASH_SOURCE[0]}" train_interior_s080
    "${BASH_SOURCE[0]}" accept_interior_s080
    "${BASH_SOURCE[0]}" train_interior_s100
    ;;
  interior_acceptance)
    verify_interior_workspace
    mkdir -p "${run_interior}/acceptance"
    for seed in 20260824 20260825 20260826; do
      "${gpu_cli[@]}" evaluate "${interior_common[@]}" "${interior_s100_pose[@]}" \
        --checkpoint "${checkpoint_interior}" \
        --seed "${seed}" \
        --num-envs 512 \
        --episodes 512 \
        --evaluation-dr-strength 1 \
        --minimum-success-rate 0.85 \
        --smoke | tee "${run_interior}/acceptance/seed${seed}_interior_dr1_512.json"
    done
    ;;
  record_interior)
    verify_interior_workspace
    "${gpu_cli[@]}" record "${interior_common[@]}" "${interior_s100_pose[@]}" \
      --checkpoint "${checkpoint_interior}" \
      --output-dir "${run_interior}/videos_interior_dr1" \
      --seed 20260825 \
      --num-envs 32 \
      --videos 6 \
      --max-attempts 300 \
      --camera-view grasp_closeup \
      --evaluation-dr-strength 1 \
      --smoke
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
    echo "usage: $0 {derive_assets|derive_assets_single|prepare_single|verify|verify_interior_workspace|reference_smoke64|smoke64_multi|smoke64_single|capacity12288_multi|capacity12288_single|bootstrap_smoke64|bootstrap20|bootstrap200|bootstrap_acceptance|train20_dr02_warmstart|smoke64_dr02_correlated_warmstart|train20_dr02_correlated_warmstart|train20_dr04_pose_warmstart|smoke64_dr04_focused_warmstart|train20_dr04_focused_warmstart|dr02_acceptance|dr04_pose_acceptance|cup6_candidate_acceptance|smoke64_interior|train_interior_s010|train_interior_s015|train_interior_s020|train_interior_frontier_s017x|train_interior_s030|train_interior_s040|train_interior_s060|train_interior_s080|train_interior_s100|accept_interior_s010|accept_interior_s015|accept_interior_s020|accept_interior_frontier_s017x|accept_interior_s030|accept_interior_s040|accept_interior_s060|accept_interior_s080|train_interior_curriculum|interior_acceptance|record_interior|train5000_multi_warmstart|train5000_multi|train5000_single|acceptance_multi|acceptance_single|record_multi|record_cup6_candidate|dataset_review1|dataset_production500|dataset_all}" >&2
    exit 2
    ;;
esac
