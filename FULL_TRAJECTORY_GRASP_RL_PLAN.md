# SIMPLE 全轨迹抓取：mjlab GPU PPO 方案

更新日期：2026-08-01

## 目标

新版训练只认真实 `mjlab + MuJoCo-Warp CUDA + RSL-RL PPO`。正式 run 使用
4096 env；CPU/MuJoCo 旧代码和 checkpoint 继续兼容，但旧结果不冒充新版 GPU PPO。

当前 GPU 路径覆盖：

- `xmove_pick`
- `xmove_bend_pick`
- `bend_pick_teleop`
- `bend_pick_and_place`
- `locomotion_pick_between_tables`

每个任务的发布目标为 clean 和 full-DR success 都 >= 70%，优选 >= 90%。
没有达到就明确标记未达标，不用 seed union、成功样本筛选或 reference replay
替代失败结果。

## 已实现路径

- physics、controller、actor、critic、rollout、loss、Adam state 全在 CUDA。
- Sonic WBC 与 AMO 都有 batched GPU controller。
- v2 任务使用 331D online state、511D reference context、842D actor/critic input。
- frozen bundle 固定 scene、mesh、controller、action transform、reset、physics、task spec、
  reward hash 和原始 episode id。
- reference library 加载 train/val/test，但每个 frozen scene 只绑定 manifest 中的
  `base_episode`，防止 scene 与 trajectory 错配。
- 旧 CPU/BC actor 仅可作为显式 warm start；task、observation dim 和 action-transform
  hash 不一致时拒绝加载。旧 CPU 训练入口不变。

## PPO 真实性门槛

每个 update 的 `ppo_integrity.jsonl` 必须同时满足：

```text
algorithm == rsl_rl.algorithms.ppo.PPO
on_policy == true
rollout_reused == false
transitions == num_envs * num_steps_per_env
optimizer_steps == num_learning_epochs * num_mini_batches
actor_parameter_delta_l2 > 0
critic_parameter_delta_l2 > 0
```

默认 update 是 24 steps/env、98,304 fresh transitions、5 epochs x 4 minibatches
= 20 Adam steps。完整轨迹诊断可用 `--ppo-steps-per-env 240`；当前保守设置为
1 epoch x 4 minibatches，因此每个 update 是 983,040 fresh transitions 和 4 Adam
steps，仍不复用 rollout。

runner 还会检查 actor、critic、Adam tensor state 均为逻辑 `cuda:0`，Adam
`capturable=True`。物理卡由 `CUDA_VISIBLE_DEVICES` 选择；reference-only 是独立 baseline，
不会创建 PPO actor，也不能作为 PPO 成功率。

## Policy 与 reference

PPO actor 输出完整 36D normalized action，并参与 Gaussian sampling、log-prob 和 clipped
surrogate。环境把它解释成围绕当前 reference 的有界 residual：

- correction bound 默认 `0.35`；
- v2 pick 从 approach 到 place 允许 actor 改右手 `7:14` 和右臂
  `21:28`；release/settle（包括 handover 的 stage 4）恢复 clean reference；
- 其余维度执行 clean reference，避免无控制权的探索噪声破坏行走和平衡；
- clean reference 只用于 residual anchor、critic clean context 和很小的 tracking reward。

这仍是真 PPO：实际执行的 residual 来自 actor distribution，reward 来自新 simulation
state，所有 rollout 都是 fresh on-policy trajectory。手工校准 policy 只能叫 seed；只有后续
PPO 相对该 seed 的 paired 提升才能叫 PPO 有效。

## Domain randomization

物理 DR 包括 target XY/yaw、质量/惯量、摩擦、joint damping、actuator strength 和 0–1
step action delay。target pose 的同一变换同步到 reference 几何。

reference noise 只进入 actor view：

- action std `0.002`；
- position std `0.0025` m；
- phase std `0.01`；
- future dropout probability `0.02`。

critic 使用 clean reference；reward、success、failure 和 termination 只使用 simulation
truth。stress evaluation 同时开启 full physics DR 和 full reference noise。

连续 ramp 只用于诊断，不再作为正式 checkpoint 选择方法。实测表明 fast ramp 和 400-update
slow ramp 都会在难度上升时覆盖掉较低强度的已验证能力。正式训练改为 performance-gated
固定强度阶段：

```text
30% -> 40% -> 50% -> 60% -> 70% -> 80% -> 90% -> 100% full DR
每档固定强度 30 updates 后 paired eval
有净增益才允许到 60 updates；60 不优于 30 就回选 model_30
退化立即停止并保留上一档最佳 checkpoint
若 10% 难度跳跃退化，则二分插入 5% 固定阶段，不直接跨过失败区间
```

CLI 支持 `--dr-initial-strength`、`--dr-warmup-steps`、`--dr-ramp-steps` 和
`--exploration-std`。固定阶段把 `--dr-initial-strength` 设为目标强度，同时给出足够大的
`--dr-ramp-steps`，使 30/60-update 筛选窗口内强度不发生可见漂移。默认值保持前向兼容，
所有 resolved 值写入 `config.json`。

若 full-DR 诊断表明失败集中在目标位姿两端，可以用 `--dr-profile pose_only` 做显式
分阶段预训练：保留目标 XY/yaw curriculum，暂时把质量、摩擦、阻尼、执行器强度、
action delay 和 reference noise 固定为 nominal。该 profile 只负责先学空间适配；其
checkpoint 必须再 warm-start 到默认 `full` profile，最终仍按完整 physics DR + reference
noise 验收，不能把 pose-only 数字当发布结果。

若 paired 诊断进一步确认只有目标 X 平移未通过，可以短暂使用
`--dr-profile target_x_only` 隔离该轴；它同样只是 curriculum，不是发布评估 profile。

## Reward

v2 任务在 GPU 上执行 frozen ordered goal graph：approach -> grasp -> lift，以及 place
任务后续的 transport -> place -> settle。当前 reward schema 为 v4：grasp 前使用受 support
约束的 multi-finger reach potential；stage potential 不再二次乘 discount；所有 goal-graph
任务统一 success `+20`、failure `-10`、timeout `-5`。这避免静态抓握累积绝对 reward 超过
完成 lift/终止的收益。reward 仍只由 simulation truth 计算。

约束：

- success 必须完成最后 goal stage 的 hold；
- 物体跌落、落到 support 以下、机器人跌倒或危险接触会失败；
- reference contact 只表示接触意图，不决定 success；
- 数值异常按单个 world 标记并 subset reset；
- reward 上升而真实 success 不升时，不选择该 checkpoint。

## 训练与选择

1. 导出并 hash task-specific frozen asset bundle。
2. 跑 reference-only、pre-PPO clean/full-DR paired baseline。
3. 验证旧 checkpoint 的 task/action-transform hash；不匹配则拒绝。旧 CPU/GPU actor 只能
   显式 warm start，不能伪装成新 PPO 结果。
4. 若 replay 无法物理抓取，先找可达的非 PPO calibration seed，并明确标注。
5. 4096 env 启动真实 PPO；每 5 updates 保存 checkpoint，并逐 update 审计 fresh rollout、
   CUDA optimizer 和 actor/critic parameter delta。
6. 诊断阶段可用 `target_x_only` 或 `pose_only`，但正式晋级只认含 physics DR 和 reference
   noise 的 `full` profile。
7. full profile 从 30% 开始按固定强度逐档训练。每档用相同 seed/world 比较 warm-start
   checkpoint、`model_30` 和 `model_60`；30 updates 退化就停止，60 不优于 30 就回选 30。
8. 开发筛选用 seed42 和 seed43 各 128 个独立 world；只有两个 seed 都不退化才晋级下一档。
9. 100% full-DR 达标后再做每 seed 至少 200 episode 的发布评估；只为发布 checkpoint
   生成最终视频。

任务已有 audited GPU checkpoint 后，先在固定强度用 30-update 门控验证 PPO 能否在
paired worlds 上产生增益。当前保守命令示例：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=4 mjlab_gpu/.venv/bin/python \
  -m simple.grasp_rl.mjlab_gpu.cli train \
  --task xmove_pick \
  --asset-bundle outputs/grasp_rl/mjlab_assets/xmove_pick/episode82 \
  --reference-processed data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2 \
  --num-envs 4096 --device cuda:0 --iterations 60 \
  --warm-start <audited-gpu-checkpoint.pt> --warm-start-critic \
  --learning-rate 1e-6 --schedule fixed --exploration-std 0.02 \
  --dr-initial-strength 0.5 --dr-ramp-steps 240000 \
  --dr-profile full --reference-target-x-arm-gains -10 2 \
  --output <output-dir>
```

默认 24 steps/env 用于固定阶段长跑。完整轨迹诊断时，`xmove_pick` 可用 240，497-step
的 `bend_pick_and_place` 使用 512；不要把长 rollout 盲目用于每个 update。先跑一个
update，检查显存、integrity 和 paired success，退化就保留旧 checkpoint。

## 验收报告

每个任务至少报告：

- clean PPO success / failure / timeout；
- full-DR PPO success / failure / timeout；
- grasp episode rate、mean/max lift、numerical failure；
- 相同 seed/world 的 reference-only 和 pre-PPO 对照；
- PPO integrity、CUDA optimizer、asset/reward/reference hash；
- 旧 CPU 结果仅作为注明 backend 的历史对照。

200-episode clean success 至少 140/200 才达到 raw 70%。如果声明“统计上高于 70%”，
还要求 Wilson 95% lower bound > 70%（约需 153/200）。

## 当前实测状态

以下均为 2026-08-01 的确定性 paired evaluation；训练 batch 中的随机成功率不替代
这些数字。

- `xmove_pick` 的 pre-PPO calibration seed 是 `v117/model_30`。30% pose-only 上，
  seed42 从 119/128 到 125/128（+6），seed43 从 116/128 到 123/128（+7）；
  15% pose-only 的 `v125/model_60` 从 125/128 到 128/128（+3）。这些是 actor/critic
  都发生 CUDA Adam 更新后的 paired 净增益，证明 PPO 不是 reference replay。
- 30% full-DR（含 reference noise）上，`v126/model_120` 从 115/128 到 124/128
  （+9）。但连续 ramp 到 40% 后 `model_160` 只有 102/128，低于 pre-PPO 107/128；
  因此保留 `model_120`，停止该 ramp。
- fast ramp 负对照在 100% pose/full 分别只有 50/128、35/128。连续 slow pose 在
  40% 从 111/128 退化到 102/128，pose-to-full 长训也从 121/128 退化到 102/128。
  这些结果是改用固定强度门控的直接依据。
- 固定 40% full-DR 的 `v131/model_30` 在 seed42 从 109/128 到 110/128，在 seed43
  从 104/128 到 107/128；`model_60` 回落到 109/128，所以选择 `model_30`。
  小步长 `v132/model_30` 在 seed42 为 112/128，但 seed43 为 103/128（低于 104），
  不作为晋级 checkpoint。冻结 normalizer 的 `v133/model_30` 只有 105/128。
- 50% full-DR warm-start 基线为 89/128；标准步长 `v134/model_30` 退化到 82/128，
  已停止。现插入 45% full-DR 阶段，其 seed42/43 基线分别为 99/128、91/128；
  标准步长和小步长固定阶段正在进行，50% 小步长分支保留作并行对照。
- `bend_pick_and_place` reference-only clean 为 0/128，虽有 128/128 grasp，但最大 lift
  仅 0.27 cm。一次 reward-v4、4096-env、512-step PPO update 通过 integrity，结果仍为
  0/128，grasp 113/128，因此停止该分支，不把 grasp 冒充完整成功。
- `xmove_bend_pick/model_119` 与 `bend_pick_teleop/model_50` clean 均为 0/128，且没有
  形成 grasp。`xmove_bend_pick` 的一次 reward-v4、4096-env、300-step PPO update 通过
  integrity 后仍为 0/128、grasp 0/128，因此也停止该分支。
  `locomotion_pick_between_tables/model_160` clean 为 0/128，但 grasp
  128/128、平均最大 lift 11.85 cm；其失败仍集中在 transport/place/settle。

当前没有一个新版 v2 mjlab GPU policy 已通过 full-DR 70% 发布门槛；达到前不录制
最终发布视频，也不宣称完成。
