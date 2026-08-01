# SIMPLE 全轨迹抓取：mjlab GPU PPO 方案

更新日期：2026-08-02

## 目标

新版训练只认真实 `mjlab + MuJoCo-Warp CUDA + RSL-RL PPO`。正式 run 使用
4096 env；CPU/MuJoCo 旧代码和 checkpoint 继续兼容，但旧结果不冒充新版 GPU PPO。

当前 GPU 路径覆盖：

- `xmove_pick`
- `xmove_bend_pick`
- `bend_pick_teleop`
- `bend_pick_and_place`
- `locomotion_pick_between_tables`

每个任务的发布目标为 clean success >= 70%，优选 >= 90%；full-DR robustness
单独报告。没有达到就明确标记未达标，不用 seed union、成功样本筛选或 reference replay
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
transitions == 4096 * 24 == 98304
optimizer_steps == 5 * 4 == 20
actor_parameter_delta_l2 > 0
critic_parameter_delta_l2 > 0
```

runner 还会检查 actor、critic、Adam tensor state 均为逻辑 `cuda:0`，Adam
`capturable=True`。物理卡由 `CUDA_VISIBLE_DEVICES` 选择；reference-only 是独立 baseline，
不会创建 PPO actor，也不能作为 PPO 成功率。

## Policy 与 reference

PPO actor 输出完整 36D normalized action，并参与 Gaussian sampling、log-prob 和 clipped
surrogate。环境把它解释成围绕当前 reference 的有界 residual：

- correction bound 默认 `0.35`；
- v2 pick 阶段只允许 actor 改右手 `7:14` 和右臂 `21:28`；
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

短验证使用 2400 vector-step ramp（约 100 PPO updates）。正式长跑使用可配置慢 ramp，
推荐：

```text
initial strength = 0.10
ramp steps       = 9600  # 400 updates
full-DR settle   = 100 updates
total            = 500 updates = 49,152,000 fresh transitions
```

CLI 支持 `--dr-initial-strength`、`--dr-warmup-steps`、`--dr-ramp-steps` 和
`--exploration-std`；默认值保持前向兼容，所有 resolved 值写入 `config.json`。

## Reward

v2 任务在 GPU 上执行 frozen ordered goal graph：approach -> grasp -> lift，以及 place
任务后续的 transport -> place -> settle。reward 由 stage potential delta、stage completion、
真实 finger opposition/contact、真实 lift 和 terminal bonus 组成。

约束：

- success 必须完成最后 goal stage 的 hold；
- 物体跌落、落到 support 以下、机器人跌倒或危险接触会失败；
- reference contact 只表示接触意图，不决定 success；
- 数值异常按单个 world 标记并 subset reset；
- reward 上升而真实 success 不升时，不选择该 checkpoint。

## 训练与选择

1. 导出并 hash task-specific frozen asset bundle。
2. 跑 reference-only clean/full-DR paired baseline。
3. 验证旧 BC checkpoint 的 task/action-transform hash；不匹配则拒绝。
4. 若 replay 无法物理抓取，先找可达的非 PPO calibration seed，并明确标注。
5. 4096 env 启动真实 PPO；每 5 updates 保存 checkpoint。
6. 先用短 ramp 排除坏 reward/坏 seed，再从最佳 checkpoint 做 500-update 慢 ramp。
7. 在固定 world 上比较 reference-only、pre-PPO seed/BC 和 PPO。
8. 只为真正成功的 GPU PPO checkpoint 生成视频。

正式命令示例：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=4 mjlab_gpu/.venv/bin/python \
  -m simple.grasp_rl.mjlab_gpu.cli train \
  --task xmove_pick \
  --asset-bundle outputs/grasp_rl/mjlab_assets/xmove_pick/episode82 \
  --reference-processed data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2 \
  --num-envs 4096 --device cuda:0 --iterations 500 \
  --warm-start <validated-checkpoint.pt> --learning-rate 3e-5 --schedule fixed \
  --exploration-std 0.05 --dr-ramp-steps 9600 --output <output-dir>
```

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

- `xmove_pick` reference-only clean/full-DR 都是 0%。非 PPO calibration seed clean
  256/256，full-DR 20/256（7.81%）。真实 PPO 快 ramp `model_80` full-DR
  23/256（8.98%），只提升 3 个 world，尚不能称为明显有效；已转入 500-update 慢 ramp。
- `xmove_bend_pick`、`bend_pick_teleop`、`bend_pick_and_place` 的 reference-only 完整
  success 当前均为 0%，需要各自 calibration/PPO，不可复用 `xmove_pick` 数字。
- `locomotion_pick_between_tables` reference-only 能 grasp 并抬升约 11.6 cm，但完整
  pick-and-place 为 0%，主要失败在后续 transport/place/settle。

当前没有一个新版 v2 mjlab GPU policy 已通过 70% 发布门槛；训练与 paired evaluation
继续进行，达到前不会宣称完成。
