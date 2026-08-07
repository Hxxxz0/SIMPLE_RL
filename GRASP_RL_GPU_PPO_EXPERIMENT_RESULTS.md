# Grasp RL GPU PPO 长训实验记录

记录日期：2026-08-04
状态：长训已停止；checkpoint、配置和完整评测报告均已保留。

## 1. 结论

本轮 GPU PPO 长训可以结束。`bend_pick_and_place` 和 `bend_pick_teleop`
在后续 checkpoint 上已进入平台期；`xmove_pick` 的 full-DR 分支只有小幅后期收益，
弱 DR 长训分支则明显退化。

最终建议使用：

| 任务 | 采用 checkpoint | full-DR PPO 成功率 | 同初态 proposal-only | PPO 提升 |
|---|---|---:|---:|---:|
| `bend_pick_and_place` | `model_500.pt` | 114/128（89.06%） | 53/128（41.41%） | +47.66 pp |
| `bend_pick_teleop` | `model_500.pt` | 103/128（80.47%） | 67/128（52.34%） | +28.13 pp |
| `xmove_pick` | full-DR `model_500.pt` | 89/128（69.53%） | 85/128（66.41%） | +3.12 pp |

其中 place 和 teleop 的 PPO 提升明确；xmove 没有下降，但 `p=0.125`，当前
128-world 测试还不能证明其小幅提升具有统计显著性。

## 2. 固定评测协议

所有表格结果均使用同一套 GPU paired benchmark：

- seed：`42`
- worlds / episodes：`128 / 128`
- 参数：`--smoke --stress-domain-randomization`
- 策略：确定性 actor mean
- 同一次 benchmark 同时运行 `reference-only`、`proposal-only` 和 PPO
- `physical_worlds_match=true`
- `noise_matched_proposals_match=true`
- PPO 的直接对照是同噪声的 `proposal-only`，不是另一组初始位置

MuJoCo-Warp 独立构造模拟器时的 contact reduction 不保证跨进程 bitwise
稳定，因此只比较同一个 paired report 内的 proposal-only 与 PPO；初始物理世界和
proposal context 的哈希是硬门槛。

## 3. 最终 paired 结果

| 任务 / checkpoint | reference-only | proposal-only | PPO | PPO - proposal | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| place `model_500` | 53 | 53 | 114 | +61 | `1.16e-16` |
| place `model_800` | 55 | 53 | 114 | +61 | `1.16e-16` |
| teleop `model_500` | 57 | 67 | 103 | +36 | `5.63e-9` |
| teleop `model_1000` | 63 | 75 | 103 | +28 | `4.26e-6` |
| xmove full-DR `model_100` | 85 | 85 | 85 | 0 | `1.0` |
| xmove full-DR `model_500` | 85 | 85 | 89 | +4 | `0.125` |
| xmove 弱 DR `model_1000` | 85 | 85 | 56 | -29 | `3.73e-9` |

完整 JSON 报告：

```text
outputs/grasp_rl/other/longrun_final_benchmarks/20260804/
```

## 4. 长训停止依据

- place：`model_500` 和 `model_800` 都是 `114/128`，继续训练无收益。
- teleop：`model_500` 和 `model_1000` 都是 `103/128`；`model_1000` 的平均
  action delta 还由 `0.03010` 增至 `0.03847`，因此保留漂移更小的
  `model_500`。
- xmove full-DR：从 `model_100` 的 `85/128` 增至 `model_500` 的
  `89/128`，只有 4 个 paired rescue、0 个 regression，采用 `model_500`，
  但不宣称显著提升。
- xmove 弱 DR：`model_1000` 降至 `56/128`，明确拒绝该 checkpoint。

停止时三个训练进程已正常退出，GPU 0--6 已释放。没有删除任何 run 或 checkpoint。

## 5. 采用的权重

```text
outputs/grasp_rl/other/raw_runs/mjlab_gpu/bend_pick_and_place/
  v400_grail20k_reward9_seed341_env8192_roll24_dr01_to_full_actor2e5_critic1e3_resume500/model_500.pt

outputs/grasp_rl/other/raw_runs/mjlab_gpu/bend_pick_teleop/
  v400_grail20k_reward9_seed342_env8192_roll24_dr01_to_full_actor2e5_critic1e3/model_500.pt

outputs/grasp_rl/other/raw_runs/mjlab_gpu/xmove_pick/
  v420_full_dr_reward9_seed233_resume99_env8192_roll24_actor1e5_critic1e3_20k/model_500.pt
```

place 的 `model_800`、teleop 的 `model_1000`/`model_1100` 及所有 xmove
历史 checkpoint 继续保留用于回归分析，但不作为默认发布权重。

## 6. PPO 训练真实性

本轮不是 reference replay 冒充 PPO。长训配置为：

- `8192` 个 GPU 环境
- 每个环境每次 update 采集 `24` steps
- 每次 update 使用 `196,608` 条 fresh transitions
- `5` learning epochs、`4` mini-batches，即 `20` 次 optimizer steps/update
- on-policy rollout，不复用旧 rollout
- actor、critic 均实际反向传播更新
- physics DR 与 reference noise 同步，最终使用 full-DR paired benchmark 验收

每个 run 目录中的 `config.json` 和训练日志是训练配置与 update 统计的原始记录。

## 7. 复现实验

以下为 xmove 最终权重的精确复测命令；place 和 teleop 只需替换 task、asset、
reference 与 checkpoint 路径。

```bash
cd SIMPLE
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  mjlab_gpu/.venv/bin/python -m simple.grasp_rl.mjlab_gpu.cli benchmark \
  --task xmove_pick \
  --asset-bundle outputs/grasp_rl/other/assets/mjlab_assets/xmove_pick/episode82 \
  --reference-processed data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2 \
  --checkpoint outputs/grasp_rl/other/raw_runs/mjlab_gpu/xmove_pick/v420_full_dr_reward9_seed233_resume99_env8192_roll24_actor1e5_critic1e3_20k/model_500.pt \
  --num-envs 128 --episodes 128 --smoke --device cuda:0 \
  --seed 42 --stress-domain-randomization \
  --output /tmp/xmove_model500_seed42_full_dr_128.json
```

验收时必须再次确认输出中的两个字段：

```text
physical_worlds_match=true
noise_matched_proposals_match=true
```

## 8. Robometer task-reward A/B 实验（2026-08-04）

### 8.1 实验问题与隔离范围

本实验验证 Robometer 是否可作为 `bend_pick` 的 PPO task reward，并且不改变前述
长训任务或已有 checkpoint。A/B 两组只改变 task reward 来源：

- Physical 组使用原始 simulator-truth `physical_task_reward`。
- Robometer 组使用 `robometer_dense + terminal_adjustment` 替换 physical task
  reward；reference regularizer、terminal 判定、成功条件、PPO、DR 和 actor 结构不变。
- Robometer 指令固定为
  `Bend down and pick up the object from the table.`，每 25 个 vector steps 推理一次。
- 两组分别从各自 30-update pilot 的 `model_29.pt` warm-start actor；由于
  Robometer 路径不支持精确恢复 PPO 状态，两组都重新初始化 critic 和 optimizer，
  避免不对称续训。

续训参数为 full DR、8 env、64 steps/env、90 updates、固定 action std `0.02`、
actor LR `3e-5`、critic LR `3e-4`，每组新增 `46,080` 条 fresh transitions。
actor normalizer 冻结。PPO 审计确认两组均为 on-policy、未复用 rollout，actor/critic
参数均发生变化，且未观察到 numerical failure。

### 8.2 固定评测与结果

评测使用 seed `20260804`、128 个 full-DR worlds、确定性 actor mean。每份报告内部
均满足 `physical_worlds_match=true` 和 `noise_matched_proposals_match=true`。由于
MuJoCo-Warp 独立进程存在少量 contact reduction 抖动，`PPO - proposal` 和对应
McNemar 检验只在同一份 paired report 内解释。

表中的 `+30/+60/+90` 是相对 30-update pilot actor 的新增 update 数：

| reward 来源 | checkpoint | PPO | proposal-only | PPO - proposal | exact McNemar p |
|---|---|---:|---:|---:|---:|
| Physical | pilot | 97 | 97 | 0 | `1.0` |
| Physical | `+30` / `model_29` | 95 | 95 | 0 | `1.0` |
| Physical | `+60` / `model_59` | 94 | 95 | -1 | `1.0` |
| Physical | `+90` / `model_89` | 95 | 92 | +3 | `0.375` |
| Robometer | pilot | 96 | 94 | +2 | `0.6875` |
| Robometer | `+30` / `model_29` | 97 | 93 | +4 | `0.125` |
| Robometer | `+60` / `model_59` | 93 | 96 | -3 | `0.375` |
| Robometer | `+90` / `model_89` | 94 | 94 | 0 | `1.0` |

Robometer 在 `+30` checkpoint 上出现过短暂的 `+4/128` paired 优势，但不显著，
随后在 `+60` 反转为 `-3/128`，在 `+90` 回到 0。最终 Robometer 与 Physical
checkpoint 的成功数为 `94/128` 对 `95/128`；相同 world ID 的直接对比中，
Robometer-only 成功 3 个、Physical-only 成功 4 个，exact McNemar `p=1.0`。

结论：当前指令、采样频率和 reward 构造下，继续 Robometer-reward PPO 没有形成
持续或统计显著的提升，不能替代 physical task reward 作为默认训练配置。短暂高分
应视为 checkpoint 波动，不能视为 reward 已验证有效。该结论不影响第 1--7 节的
已有任务和采用权重。

完整训练目录、6 份 paired benchmark 和机器可读摘要：

```text
outputs/grasp_rl/other/robometer_reward_ppo_ab/20260804_seed4242_continue90/
outputs/grasp_rl/other/robometer_reward_ppo_ab/20260804_seed4242_continue90/summary.json
```

### 8.3 其余任务的多任务 A/B（2026-08-04）

在 `bend_pick` 之外，又对 Robometer 已定义指令的其余 6 个任务执行相同的
30-update Physical/Robometer pilot。每个任务的两组从同一个任务专用 actor
warm-start，并同时使用 fresh critic/optimizer；两份规范化 `config.json` 删除
`task_reward_override` 后完全一致。每组使用 8 env、64 steps/env、full DR、
actor LR `3e-5`、critic LR `3e-4`，采集 `15,360` 条 fresh transitions。

全部 12 个 run 均完成 30/30 update，审计结果为 on-policy、无 rollout reuse，
actor/critic 参数变化非零。评测使用 seed `20260804`、128 worlds、full DR；共 18 份
initial/Physical/Robometer paired report，均满足两个 world/proposal 匹配门槛。

| 任务 | 初始 PPO | Physical PPO | Robometer PPO | Robometer - Physical | 两策略直接 paired p* |
|---|---:|---:|---:|---:|---:|
| `tabletop_grasp` | 110 | 109 | 107 | -2 | `0.6875` |
| `bend_pick_and_place` | 114 | 115 | 111 | -4 | `0.4240` |
| `bend_pick_teleop` | 93 | 98 | 99 | +1 | `1.0` |
| `xmove_bend_pick` | 127 | 127 | 127 | 0 | `1.0` |
| `xmove_pick` | 95 | 95 | 97 | +2 | `0.625` |
| `locomotion_pick_between_tables` | 0 | 0 | 0 | 0 | `1.0` |
| **合计** | **539/768** | **544/768** | **541/768** | **-3** | **`0.7283`** |

`*` 两策略由独立 MuJoCo-Warp 进程评测，因此该列是相同 task/world ID 上的描述性
McNemar 结果；正式的 PPO-vs-proposal 检验仍只解释同一 report 内的配对结果：

| 任务 | Physical：PPO - proposal（p） | Robometer：PPO - proposal（p） |
|---|---:|---:|
| `tabletop_grasp` | +20（`1.91e-6`） | +20（`1.91e-6`） |
| `bend_pick_and_place` | +59（`2.48e-15`） | +50（`1.04e-11`） |
| `bend_pick_teleop` | +35（`1.77e-7`） | +40（`1.03e-8`） |
| `xmove_bend_pick` | -1（`1.0`） | -1（`1.0`） |
| `xmove_pick` | +2（`0.5`） | +4（`0.125`） |
| `locomotion_pick_between_tables` | 0（`1.0`） | 0（`1.0`） |

Robometer 在 `bend_pick_teleop` 和 `xmove_pick` 上分别比 Physical 多 1 和 2 个成功，
但均不显著；在 tabletop 和 place 上分别少 2 和 4 个，另外两项持平。6 任务合计
Robometer 为 `541/768`，低于 Physical 的 `544/768`，`p=0.7283`。加上 8.2 的
`bend_pick` 长续训结果，目前 7 个已支持任务中没有一个证明 Robometer replacement
优于 physical task reward。因此默认配置继续使用 Physical；Robometer replacement
只保留为显式实验选项，不进入默认训练或既有发布权重。

本轮还修复了 Robometer 上游服务的 Qwen3-VL 推理 dtype 问题：Unsloth eval 路径会
使视觉塔混合 Float32/BF16，服务端现以标准 Transformers 路径加载，权重与 reward
语义不变。修复后已通过真实 512-transition smoke 和全部多任务训练请求。

完整训练、18 份报告及机器可读摘要：

```text
outputs/grasp_rl/other/robometer_reward_ppo_multitask/20260804_seed5252/
outputs/grasp_rl/other/robometer_reward_ppo_multitask/20260804_seed5252/summary.json
```

### 8.4 无 actor warm-start 的冷启动 A/B（2026-08-05）

8.2--8.3 的 A/B 都从已有任务 actor warm-start，因此只能回答“已有策略继续训练时
替换 reward 是否有益”，不能把 PPO 相对 reference-only 的优势归因于 Robometer。
本轮补做 `tabletop_grasp` 冷启动对照：Physical 和 Robometer 不加载任何 actor、
critic 或 optimizer 权重，并在环境构造后重新用 seed `6464` 初始化。配置记录的
初始 actor SHA 均为
`b0d698793b6a240dbcfcb124db715a141d5ec10187087a2a6615a03e0733892a`，
critic SHA 均为
`643c6e8a7f397f9fc045c0560543952cc0f97a4650be38e95a81b644f2bedc74`。

完全使用默认幅度随机输出时，普通 MLP 的 initial/Physical/Robometer 均为 `0/128`；
同架构的 plan-conditioned actor 为 `2/128 -> 0/128, 0/128`。这两项只说明 8 env、
30 updates（15,360 transitions）不足以从随机大动作学会任务，不能比较 reward。
正式 pilot 因此使用常见的 residual-policy 冷启动：隐藏层和输出头仍随机初始化，
但将随机 correction head 缩放到 `0.001`，使未训练 actor 从可执行 proposal 附近
开始；这不是加载或复制任何已有策略权重。

两组均使用 8 env、64 steps/env、30 updates、full DR、actor LR `3e-5`、critic LR
`3e-4`、std `0.02`。均完成 30/30 updates、15,360 fresh transitions，on-policy、
无 rollout reuse、无 numerical failure。评测使用 seed `20260805`、128 worlds；
三份报告均满足 world/proposal 匹配门槛。

| checkpoint | PPO | proposal-only | PPO - proposal | exact McNemar p |
|---|---:|---:|---:|---:|
| 随机小残差初始 actor | 82 | 74 | +8 | `0.0386` |
| Physical PPO | 83 | 83 | 0 | `1.0` |
| Robometer PPO | 74 | 78 | -4 | `0.3877` |

按相同 task/world ID 描述性直比，Robometer 比 Physical 少 9 个成功；
Robometer-only 2 个、Physical-only 11 个，McNemar `p=0.0225`。由于两 checkpoint
由独立 MuJoCo-Warp 进程评测，该 p 值仍按 8.3 的规则视为描述性结果；正式的
同-report 结论是 Physical 没有改变 proposal 成功数，而 Robometer 点估计下降 4，
且没有显著优于 proposal。相对同一随机初始 actor，Physical 为 `+1/128`，Robometer
为 `-8/128`。

结论：去掉 actor warm-start 后，当前 30-update pilot 仍未显示 Robometer reward
带来任何提升，反而出现较差点估计。因此原结论不变：默认保持 Physical，不能用
先前相对 reference-only 的高分宣称 Robometer task reward 有效。

完整 run、9 份冷启动/控制 benchmark 及机器可读摘要：

```text
outputs/grasp_rl/other/robometer_reward_ppo_scratch/20260805_seed6262/
outputs/grasp_rl/other/robometer_reward_ppo_scratch_plan/20260805_seed6363/
outputs/grasp_rl/other/robometer_reward_ppo_scratch_scaled/20260805_seed6464/
outputs/grasp_rl/other/robometer_reward_ppo_scratch_scaled/20260805_seed6464/summary.json
```
