# SIMPLE 抓取任务：真实 PPO 训练与验收计划

更新日期：2026-08-01

本文档是当前唯一执行口径，替代此前关于 diffusion/SMP、旧 GRAIL-v7、reference-test、
小样本 production 和旧发布 checkpoint 的结论。旧输出只用于诊断，不得作为新版 PPO
发布证据。

## 1. 目标

对每个 pick 相关任务分别训练真实的 on-policy PPO policy。策略必须在随机目标和 dynamics
扰动下稳定完成任务，最终在锁定的 200 个 unseen test targets 上通过统计验收。

当前纳入六个任务：

| task | 类型 | 当前正式输出 |
| --- | --- | --- |
| `tabletop_grasp` | 最简单桌面抓取 | `tabletop_grasp/ppo_curriculum_clean_v1_seed42_to3200` |
| `bend_pick_teleop` | 弯腰抓取 | `bend_pick_teleop/ppo_curriculum_v1_seed42_winner_to3200` |
| `bend_pick_and_place` | 弯腰抓放 | `bend_pick_and_place/ppo_curriculum_v1_seed42_winner_to3200` |
| `xmove_pick` | 移动抓取 | `xmove_pick/ppo_curriculum_clean_v2_seed42_to3200` |
| `xmove_bend_pick` | 移动并弯腰抓取 | `xmove_bend_pick/ppo_curriculum_clean_v1_seed42_to3200` |
| `locomotion_pick_between_tables` | 跨桌抓放 | `locomotion_pick_between_tables/ppo_curriculum_v1_seed42_winner_to3200` |

旧 `bend_pick` MP 实验不计入这六条正式结果；如再次训练，也必须完整采用本文方法。

## 2. 不可放宽的发布门槛

每个任务必须同时满足：

1. final evaluation 恰好使用 200 个预先锁定、训练阶段从未使用的 test targets；
2. PPO 至少成功 153/200；
3. Wilson 95% success-rate lower bound 严格大于 70%；
4. 相对同状态 reference-only baseline 至少提升 5 percentage points；
5. paired exact McNemar test `p < 0.05`；
6. 初始 `qpos/qvel`、目标 pose、base episode 和 reference episode 必须逐对完全相同；
7. policy checkpoint 必须通过 PPO 完整性审计。

任何单独的 70% raw success、小样本 20/20、多个 rank 的 union、只保存成功轨迹，或
reference playback 都不能替代上述门槛。若某任务的 reference baseline 高于 95%，则
“提升 5pp”在数学上不可实现；必须在 final test 前预注册更有区分度的随机目标分布，不能
看完 test 后修改协议。

## 3. 数据隔离

- 原始 `data/simple` 保持只读；派生数据写入 `data/grasp_rl`。
- BC 初始化只训练 `train`，以 `val` 做 early stopping，不读 `test`。
- PPO 的 reference library 固定为 `--reference-splits train,val`。
- hard-target mining 只允许来自 `train` 或 `val` failure manifest。
- checkpoint 选择、reward 调整和 curriculum 调整只看 development validation。
- final test manifest 在 validation 达标之前保持锁定且不可读取。
- 使用过 test reference、继承过污染 PPO history 或训练扰动范围不一致的目录必须写入
  `NOT_RELEASE_CANDIDATE.json`，不得继续 resume 为发布候选。

已知不可发布的旧分支包括旧 `xmove_pick/ppo_curriculum_v1*`、
`xmove_pick/ppo_curriculum_clean_v1*`、旧 `xmove_bend_pick/ppo_curriculum_v1*`，以及
`tabletop_grasp` 的旧 `ppo_task_random_stable_v1_300`。旧 Tabletop run 使用过
`reference_splits=train,val,test`，也没有新版逐 update 完整性日志。

## 4. Policy 与真实 PPO 定义

### 4.1 Policy 接口

- v2 任务 actor 输入为 331D 在线 MuJoCo/task state 加 511D retrieved plan context，共 842D。
- legacy `tabletop_grasp` 输入为 192D 在线状态加 401D plan context，共 593D。
- actor 直接输出最终 normalized 36D tracker command。
- PPO distribution、sample、log-prob 和 clipped surrogate 都定义在这 36D policy action 上。
- 不允许在 PPO 外部播放 reference action，再把结果称作 RL policy。
- retrieved plan 只是 policy observation/context；最终命令由 actor distribution 产生。
- v2 task 使用 task-specific `ActionTransform` 后交给 Sonic 或 AMO tracker。

v2 任务从 train-only plan-conditioned BC actor 初始化。`tabletop_grasp` 新分支从零 correction
的 plan-conditioned actor 初始化，不继承旧的 test 污染 PPO checkpoint。

### 4.2 每个 update 的完整性条件

`ppo_integrity.jsonl` 每行必须满足：

```text
algorithm == rsl_rl.algorithms.ppo.PPO
on_policy == true
rollout_reused == false
transitions == num_envs * num_steps_per_env
optimizer_steps == num_learning_epochs * num_mini_batches
actor_parameter_delta_l2 > 0
policy_version_collected == update
```

正式配置为 36 env、24 steps，因此每个 update 必须包含 864 条 fresh transitions；
5 PPO epochs × 4 mini-batches，因此每个 update 必须有 20 次 optimizer step。只有 checkpoint
文件而没有这些证据，不足以证明 PPO 有效执行。

## 5. 统一训练预算

- `num_envs = 36`
- `num_steps_per_env = 24`
- `iterations = 3200`，update 编号为 0–3199
- 总预算 `36 × 24 × 3200 = 2,764,800 transitions/task`
- `ppo_epochs = 5`
- `num_mini_batches = 4`
- 每 50 updates 保存 checkpoint
- seed 42 为主训练 seed；必要时用独立 seed 验证，不用 seed union 隐藏失败
- 同时最多运行三条 36-env 训练，避免 GPU/CPU contention 破坏吞吐和 rollout timing

任务特定的保守 actor 设置：

| task family | base LR | actor LR scale | manipulation std | task/ref reward |
| --- | ---: | ---: | ---: | ---: |
| bend grasp | 3e-4 | 0.10 | 0.03 | 默认 / 0.005 |
| xmove grasp | 3e-4 | 0.05 | 0.05 | 0.02 / 0.005 |
| tabletop | 3e-4 | 0.02 | 0.01 | 0.02 / 0.005 |
| locomotion place | 1e-4 | 0.02 | 0.015 | 0.05 / 0.001 |

BC 初始化任务冻结 actor normalizer；Tabletop 从零 actor 开始，允许 normalizer 用新 rollout
更新。locomotion 额外使用很小的 teacher anchor 防止已学会的 walking/place tail 被擦除。

## 6. Curriculum 与 domain randomization

抓取任务使用 `configs/grasp_rl/pick_grasp_curriculum_v1.json`，抓放任务使用
`configs/grasp_rl/pick_place_curriculum_v1.json`。两个 curriculum 都包含三个阶段：

| updates | RSI probability | reference rank max | reference noise | mass scale | friction scale | action delay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–1064 | 0.70 | 2 | 0.01 | 0.95–1.05 | 0.90–1.10 | 0 |
| 1065–2394 | 0.50 | 4 | 0.02 | 0.90–1.10 | 0.85–1.15 | 0–1 |
| 2395–3199 | 0.25 | 4 | 0.025 | 0.85–1.15 | 0.80–1.20 | 0–1 |

每阶段 target mix 固定为 50% uniform、35% hard、15% native。抓取 RSI 在
`pregrasp/grasp_to_lift` 间按 60/40 采样；抓放 RSI 覆盖 approach、grasp、lift、transport。
课程后期逐步减少 RSI、增加 full-start rollout，同时扩大 dynamics、action noise 和 delay。

移动任务和新版 Tabletop 的正式训练/验证 envelope 为目标 x/y jitter `±4/5 cm`、yaw
`±0.25 rad`。其他 bend 任务当前 development screen 使用 `±2.5/3 cm`、`±0.15 rad`；
训练与最终验证若改变 envelope，必须生成新版本协议并保持 train/validation 一致。

## 7. Reward 方法

- 每个任务在训练前必须通过真实 tracker/MuJoCo expert replay reward audit。
- 抓取使用 `grail_release_v1` 的 approach、bilateral grasp、finger direction、lift、stability、
  table/contact penalty 和 residual-rate penalty。
- 抓放使用 ordered goal graph：approach → grasp → lift → transport → place → release/settle。
- success 必须来自当前物理状态并连续保持，不能只因进入 stage 或短暂接触而成功。
- reference reward 只是低权重 shaping，不能压过 task reward。
- 当前正式方法不使用 diffusion/SMP reward；旧 SMP 对照不进入发布路径。
- 训练中持续记录 task component、reference contribution、各 stage、drop、table contact、
  success RSI/full-start 和 domain-randomization telemetry。

如果 return 上升但 success 不升，首先检查 reward farming、stage transition 和成功保持条件，
不能只继续增大 reward scale。

## 8. Checkpoint 选择与验证阶梯

1. 训练前生成 `model_initial` 的 paired val-40 baseline。
2. 训练中按固定 seed 对 checkpoint 做 paired val-40；早期可额外测 model50。
3. 同一 target 上分别运行 PPO 与 `--reference-action-override all`。
4. `compare-paired` 必须验证配对状态完全一致，并输出 rescue、regression、Wilson lower 和
   exact McNemar p-value。
5. 选择 validation 最佳 checkpoint，不默认选择最后一个 checkpoint。
6. 任一 checkpoint 达到至少 28/40 后，先扩大到预注册的 val-100，再到 val-200。
7. 只有扩大 validation 后仍稳定达标，才允许一次性运行锁定的 final test-200。
8. final test 不用于调参、重训、选择 seed 或选择 checkpoint。

val-40 只是快速筛选，不能据此宣称最终 70% 成功率。`28/40` 也不满足 Wilson lower >70%；
它只是启动更大 validation 的门槛。

## 9. 退化和 plateau 的处理

- 曲线先升后降：保留历史最佳，不用 last checkpoint 覆盖；从最佳点建立独立低 actor-LR
  continuation，并减小 reference/task reward imbalance。
- success 长期低于 70%：按 approach、contact、grasp、lift、place 分解失败，先修复占比最高
  的阶段；每个分支只改变一个主要因素。
- PPO actor delta 为零、rollout 被复用或 reference override 泄漏：立即判定该 run 无效。
- reference-only 很强：扩大预注册随机化难度或提升 robustness，不能把 reference playback
  计为 PPO success。
- final checkpoint 退化：发布选择仍使用 validation 最佳 checkpoint。
- test 泄漏：整个 ancestry 标记不可发布，从 clean BC 重新开始。

## 10. 资源队列

最多同时保留三条正式训练：

1. 当前运行 `bend_pick_teleop`、`bend_pick_and_place`、`xmove_pick clean v2`；
2. bend 完成并完成 final validation screen 后，在同一资源槽启动 `xmove_bend_pick clean`；
3. place 完成后启动 `locomotion_pick_between_tables`；
4. xmove_pick 完成并生成 `paired_val_model3199_40.json` 后启动 `tabletop_grasp clean`。

所有 watcher 只等待 checkpoint/paired report，不得抢先读取 test，也不得把正在等待的 watcher
误杀。每条新训练启动后，先检查首个 `ppo_integrity.jsonl` 记录，再允许继续长训。

## 11. 当前状态快照

以下仅是 2026-08-01 development validation 快照，不是 final test：

| task | update / 3199 | transitions | 当前最佳 val-40 | reference | exact p | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bend_pick_teleop` | 2347 | 2,028,672 | 21/40 (52.5%) | 8/40 | 0.000244 | 训练中 |
| `bend_pick_and_place` | 2066 | 1,785,888 | 26/40 (65.0%) | 19/40 | 0.143 | 训练中 |
| `xmove_pick` | 686 | 593,568 | 14/40 (35.0%) | 4/40 | 0.00195 | 训练中 |
| `xmove_bend_pick` | — | — | — | — | — | 已排队 |
| `locomotion_pick_between_tables` | — | — | initial 0/40 | — | — | 已排队 |
| `tabletop_grasp` | — | — | 待新版 wide-envelope baseline | — | — | 已排队 |

当前没有任何任务获准运行 final test-200。最接近 raw 70% 的是
`bend_pick_and_place/model_1700.pt`，但只有 26/40，且相对 reference 的 McNemar 结果不显著。

新版 Tabletop 已完成隔离 smoke：真实 `rsl_rl PPO`、20 optimizer steps、actor delta
`0.0229`、`on_policy=true`、`rollout_reused=false`。正式 run 使用 36 env；旧 99/100 结果只
证明任务容易，不是新版合规 PPO 结果。

## 12. 最终交付物

每个任务的发布包必须包含：

- 选定的 PPO checkpoint 与 SHA256；
- 完整训练 `config.json` 和 resolved curriculum；
- 全量 `ppo_integrity.jsonl`；
- reward audit；
- train/val/test manifest hash 和 split provenance；
- paired val-40、扩大 validation 和 locked final test-200 报告；
- reference-only paired baseline；
- success/failure stage breakdown；
- 至少若干从保存初态重新闭环运行的成功与失败视频；
- 明确记录未达标项，禁止只发布成功样本。

只有六项任务级门槛全部通过后，才能把对应 checkpoint 标记为 release candidate。
