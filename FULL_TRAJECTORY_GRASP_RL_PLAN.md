# SIMPLE 全轨迹抓取：mjlab GPU PPO 方案与实测

更新日期：2026-08-02

## 1. 目标与验收口径

新版训练只认真实的 `mjlab + MuJoCo-Warp CUDA + RSL-RL PPO`。CPU MuJoCo
训练和旧 checkpoint 继续兼容，但其结果必须标明 backend，不能冒充新版 GPU PPO。

当前发布验收口径为：

1. 在相同 seed、相同初始 worlds、相同 full-DR 下比较 `reference-only` 和 PPO；
2. PPO 必须对完整任务成功率产生明显的大幅提升；
3. 抓住或抬起但未完成最后 goal stage 仍算失败；
4. 70% 是后续优化目标，90% 是理想目标，不再作为本轮发布的硬阻塞条件；
5. 只发布通过 CUDA/PPO 完整性审计且可复现的 checkpoint。

`68.75% DR` 表示 domain-randomization 强度为 `0.6875`，不表示成功率。

## 2. GPU 训练配置

正式训练使用 8192 个并行 environment。物理 GPU 由
`CUDA_VISIBLE_DEVICES=4|5|6` 选择，进程内部固定使用逻辑 `cuda:0`；GPU 7 不使用。
4096 或更少 environment 只用于 smoke、诊断和短评测。

每个 8192-env、240-step update 收集：

```text
8192 * 240 = 1,966,080 fresh on-policy transitions
1 PPO epoch * 4 minibatches = 4 CUDA Adam optimizer steps
```

训练、controller、MuJoCo-Warp physics、actor、critic、rollout、loss 和 Adam state
全部留在 CUDA。不存在 CPU physics fallback。

## 3. PPO 真实性门槛

每次 update 的 `ppo_integrity.jsonl` 必须同时满足：

```text
algorithm == rsl_rl.algorithms.ppo.PPO
on_policy == true
rollout_reused == false
transitions == num_envs * num_steps_per_env
optimizer_steps == num_learning_epochs * num_mini_batches
actor_parameter_delta_l2 > 0
critic_parameter_delta_l2 > 0
optimizer tensor state is CUDA
```

PPO actor 输出 36D normalized action，并参与 Gaussian sampling、log-prob、GAE 和
clipped surrogate。环境只在任务允许的右手和右臂维度执行相对 clean reference 的有界
residual；reward 和终止始终来自新的 simulation state。`reference-only` 直接执行 reference，
不会创建 PPO actor。

当前发布候选 `v192/model_0.pt` 的审计记录为：

```text
algorithm: rsl_rl.algorithms.ppo.PPO
transitions: 1,966,080
rollout_reused: false
optimizer_steps: 4
actor_parameter_delta_l2: 6.3792746e-4
critic_parameter_delta_l2: 1.3579301e-3
stochastic_action_noise_rms: 0.0199977
```

## 4. Domain randomization

full-DR 同时包含：

- target X/Y translation 和 yaw；
- target mass/inertia、friction、joint damping、actuator strength；
- 0--1 step action delay；
- actor-only reference action/position/phase noise 和 future dropout。

target pose 的场景变换会同步到 reference geometry。reference noise 只进入 actor view；
critic 使用 clean reference；reward、success、failure 和 termination 不读取 noisy reference。

为了诊断单轴瓶颈，CLI 还支持 `pose_only`、`target_x_only`、`target_y_only` 和
`target_yaw_only`。这些 profile 只能用于 curriculum，最终评测必须使用 `full`。

## 5. Reward 与完整任务成功

当前 GPU goal-graph reward schema 为 v8：

- ordered stages：`approach -> grasp -> lift`，place 类任务继续执行 transport/place/settle；
- approach 和 grasp 使用 multi-finger shaping；
- grasp stage 使用真实 contact force 验证；
- lift 的 dense potential 必须同时保有对应手的真实 force-grasp quality，避免 PPO
  通过碰撞或松手后的物体位移获取虚假抬升奖励；
- place 的 dense potential 同时要求 XY 对齐和向 destination 下放，避免只靠近容器上方后
  reward 饱和；该 shaping 不改变容器接触、连续 hold、settle 或最终 success 条件；
- stage shaping 使用 potential delta，不重复乘 discount；
- success `+40`、failure `-10`、timeout `-5`；
- reference contact 只表达 clean reference 的接触意图，不控制 success；
- 物体跌落、落到 support 以下、机器人跌倒、危险接触或数值异常按失败处理。

评测额外输出 `max_stage_counts`、grasp episode rate、最大 lift 和最大 grasp quality，
用于区分“未形成抓取”和“抓取后未完成 lift”，但不改变 success 语义。

## 6. 支持范围与兼容性

同一套 GPU vec-env、goal graph、DR、PPO runner 和评测路径支持：

- `xmove_pick`
- `xmove_bend_pick`
- `bend_pick_teleop`
- `bend_pick_and_place`
- `locomotion_pick_between_tables`

旧 CPU/BC actor 只能通过显式 warm start 加载。task、observation dimension、asset manifest
或 action-transform 不匹配时拒绝加载；已有 CPU 训练入口和旧 checkpoint 格式不改。
CLI 的 `--max-reference-action-deviation` 可按任务放宽 residual action 上限；默认仍为
`0.35`，因此旧命令和旧行为不变。改变该值后必须 fresh warm start，不能伪装成 exact resume。

## 7. 2026-08-02 实测结果

### xmove_pick 发布候选

Checkpoint：

```text
outputs/grasp_rl/mjlab_gpu/xmove_pick/
v192_reward6_full_stage06875_from_v184m0_seed44_env8192_roll240_lr5e7_std02_freezenorm/model_0.pt
```

以下是 full physics DR + full actor reference noise、deterministic actor、每个 seed 200 个
独立 worlds 的完整任务结果：

| seed | reference-only | PPO | 绝对提升 | PPO grasp episodes | PPO reach lift stage |
|---|---:|---:|---:|---:|---:|
| 42 | 0/200 (0%) | 42/200 (21%) | +21 pp | 108/200 | 100/200 |
| 43 | 0/200 (0%) | 48/200 (24%) | +24 pp | 111/200 | 109/200 |
| 合计 | 0/400 (0%) | 90/400 (22.5%) | +22.5 pp | 219/400 | 209/400 |

这满足当前“PPO 相对 reference-only 有明显完整成功提升”的验收口径。它不满足 70%
stretch target，因此不能描述为 70% 成功率。

在开发用的 `0.6875` DR 强度上，`v192` 相比上一 PPO checkpoint `v184` 的 paired
结果为 seed42 `62 -> 64/128`、seed43 `58 -> 60/128`，两个 seed 都无回退。

### full-DR reward-v7 长训复核

从 reward-v7 的三条 8192-env 分支分别 exact-resume 4 个 update；每条分支累计采集
`9,830,400` 条 fresh on-policy transition，每个 update 执行 4 次 CUDA Adam step，
actor/critic 参数变化均非零。相同 seed42、128 个 full-DR worlds 的筛选结果为：

| policy | 完整成功 | reference-only | 绝对提升 |
|---|---:|---:|---:|
| v208/model_4.pt | 37/128 (28.9%) | 0/128 (0%) | +28.9 pp |
| v209/model_4.pt | 29/128 (22.7%) | 0/128 (0%) | +22.7 pp |
| v210/model_4.pt | 36/128 (28.1%) | 0/128 (0%) | +28.1 pp |

三条长训都再次满足“明显超过 reference-only”的完成标准，但单 seed 小筛选存在
MuJoCo-Warp 接触波动，且没有形成稳定超过现有候选的证据。因此不以这组 128-world
开发筛选替换经过双 seed、400 episodes 正式评测的 `v192`。

### 其他任务

GPU 工程路径已经覆盖，但当前 checkpoint 尚未通过“完整成功大幅高于 reference-only”：

- `bend_pick_and_place`：reward-v8 在 place shaping 中加入垂直下放进度，但不放宽
  success。三条真实 8192-env、240-step PPO checkpoint 在同一未见 seed90 full-DR
  worlds 上均为 `2/128 (1.56%)`，paired reference-only 为 `0/128`。每条分支都形成
  非零 actor/critic 更新，但 `2/128` 的幅度不足以认定为明显提升，因此不发布。
- `xmove_bend_pick`：reward-v7 的真实 8192-env、240-step PPO update 后 clean 仍为
  `0/128`。静态 residual 诊断仅在上限 `0.35` 附近找到 `7/1024` 抓取，而增大 iid
  exploration std 后仍为零，说明下一步应学习持续姿态修正，而不是继续放大随机噪声。
- `bend_pick_teleop`：默认 residual 上限 `0.35` 的扫描没有形成抓取；临时诊断上限
  `0.7` 得到 `2/1024` 抓取，证明资产和接触链有效，也说明该任务需要单独 warm-start
  训练，不能直接复用旧配置 exact-resume。
- `locomotion_pick_between_tables`：reward-v8 checkpoint 在未见 seed90 full-DR 仍为
  `0/128`，其中 `119/128` 形成抓取、`93/128` 到 transport、`22/128` 到 place；
  place 可达率高于 reward-v7 诊断，但完整成功没有提升，故仍不发布。

reward-v7/v8 的上述正式 update 均收集 `1,966,080` 条 fresh transition、执行 4 次
CUDA Adam step，actor/critic 参数变化均非零；它们是有效 PPO 训练，但没有达到完整
任务验收，因此不发布这些 checkpoint。reward-v8 locomotion 分支的 actor/critic delta
分别为 `3.815e-4` 和 `7.325e-4`，且 optimizer state 位于 CUDA。

因此本轮可以发布的是 `xmove_pick`，不能宣称所有 task 的策略效果都已通过。

## 8. 视频

最终视频只从 simulator truth 判定成功的 full-DR episode 中录制。渲染使用与冻结 physics
bundle 状态布局一致的 full-visual sidecar，并强制使用外部 MuJoCo full-robot camera，
不使用机器人头部 stereo camera。

```text
outputs/grasp_rl/mjlab_gpu/xmove_pick/release_v192_videos_full_visual/
  xmove_pick_full_dr_seed42_01.mp4
  xmove_pick_full_dr_seed42_02.mp4
  xmove_pick_full_dr_seed42_03.mp4
```

每个同名 JSON 包含 checkpoint SHA256、PPO integrity、随机化参数、最大 lift、分辨率和
`render_source=full_visual_sidecar`。

## 9. 可复现实验命令

正式训练示例：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=4 mjlab_gpu/.venv/bin/python \
  -m simple.grasp_rl.mjlab_gpu.cli train \
  --task xmove_pick \
  --asset-bundle outputs/grasp_rl/mjlab_assets/xmove_pick/episode82 \
  --reference-processed data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2 \
  --reference-target-x-arm-gains -10 2 \
  --num-envs 8192 --device cuda:0 --iterations 1 \
  --warm-start <audited-checkpoint.pt> --warm-start-critic \
  --learning-rate 5e-7 --schedule fixed --exploration-std 0.02 \
  --ppo-clip-param 0.025 --ppo-learning-epochs 1 \
  --ppo-steps-per-env 240 --freeze-actor-normalizer \
  --dr-initial-strength <fixed-stage-strength> \
  --dr-ramp-steps 240000 --dr-profile full \
  --output <output-dir>
```

full-DR paired evaluation 必须分别运行 `--reference-only` 和 `--checkpoint`，并保持
`--seed`、`--num-envs`、`--episodes` 与其他参数完全一致。

## 10. 后续提升方向

若继续追求 70%--90%，优先解决进入 lift stage 后掉落或超时的问题，不再扩大无依据的
reward 改动。每次只做一个小改动，并在 seed42/43 paired worlds 上同时无回退后才保留；
训练仍使用 GPU 4--6 的 8192-env 独立分支，GPU 7 保持不使用。
