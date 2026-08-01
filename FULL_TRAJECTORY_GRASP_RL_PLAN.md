# SIMPLE 全轨迹抓取：mjlab GPU PPO 执行方案

更新日期：2026-08-01

本文是新 GPU 路径的唯一执行口径。旧 CPU PPO、36-env 实验、SMP/diffusion
reward 和旧成功率只能作为兼容性资料，不能作为新版 PPO 的发布证据。

## 1. 目标与当前边界

- 使用真实 `mjlab + MuJoCo-Warp + RSL-RL 5.2 PPO`，simulation、AMO、actor、
  critic、rollout、loss 和 Adam state 全部留在 GPU。
- 正式训练从 2048 env 起步，显存允许时提升到 4096 env；小 env 仅用于 smoke。
- 每个任务独立训练、独立 frozen asset bundle、独立 reward audit 和 locked test。
- locked test-200 的目标是成功率 70% 以上，优选 90% 以上；未达到不得声称完成。
- 旧 CPU 环境、命令和 checkpoint loader 保持可用，但不会被包装成 GPU 结果。

当前已经接通 GPU 的是 legacy AMO + 192D state + 401D reference 的
`tabletop_grasp` 和 `bend_pick`。Sonic/v2 的 `bend_pick_teleop`、
`bend_pick_and_place`、`xmove_pick`、`xmove_bend_pick`，以及虽然使用 AMO、但仍是
v2 schema 的 `locomotion_pick_between_tables`，必须在各自 batch controller、842D
observation 和 goal-graph reward 完成 GPU parity 后再训练；当前不得宣称它们已迁移完成。

## 2. 不可伪造的 PPO 条件

每个 update 的 `ppo_integrity.jsonl` 必须证明：

```text
algorithm == rsl_rl.algorithms.ppo.PPO
on_policy == true
rollout_reused == false
transitions == num_envs * num_steps_per_env
optimizer_steps == num_learning_epochs * num_mini_batches
actor_parameter_delta_l2 > 0
critic_parameter_delta_l2 > 0
```

还必须检查 Adam `capturable=True`，optimizer tensor state 全在 `cuda:0`。AMO
TorchScript 内部固定使用逻辑 `cuda:0`，因此一张物理 GPU 启动一个进程，并通过
`CUDA_VISIBLE_DEVICES=<physical GPU>` 选择设备。

reference action 只能作为 actor observation/context 或独立 baseline。禁止在 PPO 外部
播放 reference action，再把成功称作 PPO policy success。最终 36D command 必须来自
actor distribution，并参与 PPO log-prob 和 clipped surrogate。

## 3. Frozen asset 与前向兼容

每个 GPU task bundle 固定并校验：

- portable `scene.xml` 与全部 mesh hash；
- MuJoCo timestep、solver、cone、impratio 等 physics 参数；
- reset qpos/qvel/ctrl/qacc、目标/桌子角色和 reward spec；
- task-specific `action_transform.npz`；
- AMO policy、adapter、normalization stats 及 warm controller state；
- AMO 的 history、last action、gait、initial quaternion 和 flags。

GPU 代码按 joint/body/actuator/sensor 名称解析索引，不依赖 XML 数字顺序。旧 CPU
checkpoint 可作为显式 warm start；新版 GPU checkpoint 额外保存 config、asset、reward、
reference hash、DR RNG、reference-noise RNG、adaptive LR 和 next iteration，严格 resume。
loader 根据旧 checkpoint 的 `_plan_conditioned_actor` marker 恢复原来的
`reference proposal + PPO correction` 语义；不能把 correction 当成完整 command。

## 4. Reference 与 domain randomization

reference 分成两个视图：

- actor：训练时使用 noisy reference；
- critic：始终使用 clean reference；
- reward/success/failure/termination：只使用 clean replay 标签和当前仿真真值；
- locked eval：关闭物理 DR 和 reference noise。

训练 reference noise 默认包含 normalized action、物体相对位置、phase 和 future-frame
dropout 的小扰动。action std 固定为经 200-episode ablation 验证的 `0.002`；旧候选
`0.015` 会让 plan-conditioned actor 把白噪声直接带入完整 command，已弃用。噪声使用
独立 CUDA RNG，同一 simulation step 重复读取保持一致，不能改变之后的物理 DR 序列。
目标 XY jitter 通过 object offset 同步进 reference；target yaw 作为当前物体姿态扰动由
在线 base observation 提供，不能伪造进 reward truth。

物理 DR 真实写入 MuJoCo-Warp per-world model field：目标质量/惯量、摩擦、joint damping、
actuator strength、目标 XY/yaw 和 0–1 step action delay。训练不从 full DR 硬启动：

| vector steps | DR strength | 目的 |
| ---: | ---: | --- |
| 0–1199 | 0 | 先修正 BC 闭环漂移 |
| 1200–4799 | 0 → 1 | 同步放大 physics 与 reference noise |
| 4800+ | 1 | 完整随机化训练与 robustness 验收 |

数值发散按 world 标记为 `numerical_failure` 并 subset reset，不能让单个异常 world 杀死
全部 rollout，也不能静默当作正常 transition。

## 5. Reward 原则

legacy 抓取继续使用 frozen `grail_release_v1`：pregrasp、双侧 contact、finger
opposition、lift、stability、approach/table/action-rate penalty，以及连续 hold success。
success 必须来自当前物体真实 lift + grasp；reference contact 只表达接触意图。

训练前必须先通过 clean expert replay：成功率至少 90%，contact、lift 和 terminal 状态与
CPU 基线一致。若 expert replay 失败，先修 simulation/controller/reward，禁止靠调 reward
掩盖迁移错误。

## 6. Tabletop 训练阶梯

1. **Parity gate**：CPU/GPU first-step、10-step、完整 expert trajectory；clean expert
   replay 至少 90%。
2. **Initial baseline**：BC 在同一 locked validation 上记录 success、failure、timeout、
   max lift、grasp rate；不因 BC loss 很低就假定闭环成功。
3. **Clean PPO**：2048 env，24 steps/env，5 epochs，4 mini-batches；先跑 50–200 updates，
   以真实 success 而非 return 选择是否继续。
4. **DR ramp**：继续训练到 strength=1，观察 numerical failure、contact、lift 和 success，
   不允许 reward 上升但 success 长期不升。
5. **Capacity**：记录实际 contact/constraint 峰值后降低过度保守的 `nconmax/njmax`，再从
   2048 提升到 4096；每次扩容重跑 PPO integrity 和 checkpoint resume。
6. **Validation**：固定 val-40 快筛，达到 28/40 后扩大到 val-100/200；只按 validation
   选择 checkpoint。
7. **Final**：一次性 locked test-200，不用于调参。

从已有强 policy 做 DR continuation 时，首轮 150 updates、4096 env 已包含
`4096 * 24 * 150 = 14,745,600` 条 fresh transitions，应按 25-update checkpoint 的
full-DR 曲线决定是否追加；从头训练仍建议至少 3000 updates。训练中每 25–50 updates
保存 checkpoint，避免长任务因异常丢失全部进度，也避免默认把 last 当作 best。

## 7. 70%–90% 验收

每个 task release candidate 同时满足：

1. locked clean test 恰好 200 个未参与训练/调参的 targets；
2. raw success 至少 140/200；推荐至少 180/200；
3. 若声明“统计上高于 70%”，Wilson 95% lower bound 必须大于 70%（约需 153/200）；
4. full-DR robustness validation 单独报告，不与 clean test 混算；
5. 相同初态的 pre-PPO/BC、reference-only、PPO 三方 paired 对比；
6. PPO integrity、CUDA optimizer、asset/reward/reference hash 全部通过；
7. 报告 failure、timeout、native lift、grasp episode rate、mean/max lift 和 numerical failure。

“PPO 明显有效”至少要求 PPO 相对 pre-PPO BC 的 paired success 有实际提升，并且 actor
参数变化来自 fresh on-policy optimizer steps。单独的 expert/reference replay 100%、短 smoke
actor delta 或训练 return 上升，都不能替代最终成功率。

## 8. 多任务迁移队列

多 GPU 只并行已经通过 parity gate 的任务，避免同时修改共享实现：

1. `tabletop_grasp`：4096-env full-DR PPO 已完成，`model_149` 通过 robustness gate；
2. `bend_pick`：独立 AMO bundle、expert replay、旧 policy GPU parity 和 4096-env
   full-DR PPO 已完成；当前 robustness 最好为 547/800，尚未通过 70% gate；
3. Sonic grasp tasks：先实现 frozen Sonic batch controller 与 842D GPU state parity；
4. place tasks：在 Sonic parity 之后迁移 ordered goal-graph reward；
5. 每个任务达到同一验收门槛后才标记 release candidate。

任何任务未达到 70% 时，交付真实诊断与 checkpoint，不用其它 task、seed union、成功样本
筛选或 reference override 替代该任务的失败结果。

## 9. 2026-08-01 已完成的实测 gate

- `tabletop_grasp` clean expert replay：16/16；clean locked parity：200/200。locked
  parity 的 200 个 world 可能共享等价初态，因此只用于检查回归，不能冒充 200 个独立
  robustness target。
- reference action noise 的 200-episode 消融中，`0.015` 会把 plan-conditioned 完整
  command 破坏到 0/200；正式值已锁定为 `0.002`。full-DR 评估同时开启 physics DR
  和 reference noise，critic/reward/success 仍使用 clean truth。
- `tabletop_grasp` full-DR warm-start 基线在 seed 142/143/144 为
  140/132/133，合计 405/600（67.5%）。4096-env adaptive PPO `model_149` 为
  158/157/157，合计 472/600（78.67%），提升 67 个成功样本、11.17 个百分点；
  三个单 seed 的 Wilson 95% lower bound 均高于 72%。
- 同任务 fixed `1e-5` 的 `model_100` 为 145/141/143，合计 429/600
  （71.5%），虽然有效但显著弱于 adaptive final，因此不选作 release candidate。
- `bend_pick` clean expert replay：16/16；clean locked parity：200/200。原 adaptive
  `model_100` 在 seed 144/145/146/147 为 129/142/138/135，合计 544/800
  （68%）。从该 checkpoint 以 fixed `1e-5` 在 full DR 下稳定化 25 updates 后为
  129/142/144/132，合计 547/800（68.38%），是当前最佳，但仍未达到发布门槛。
  作为对照，从 stronger-clean `model_49` 出发的 adaptive `model_100` 为
  128/141/142/133（544/800），从 clean policy 开始的 fixed `1e-5 model_75` 为
  128/144/136/132（540/800），fixed `3e-5 model_75` 为 129/141/142/131
  （543/800）；增加训练量或更换学习率均未稳定越过 70%，不得宣称 Bend 已达标。
- 正式 PPO 每 update 使用 4096×24=98,304 条 fresh transitions、5 epochs、4
  mini-batches、20 optimizer steps；`tabletop_grasp model_149` 累计 14,745,600
  条 fresh transitions。actor/critic delta 均非零，rollout 未复用，Adam state 全在 CUDA。
