# SIMPLE 全轨迹抓取 RL：设计、执行计划与当前结果

## 1. 目标与边界

第一阶段只使用：

```text
task/data: simple/G1WholebodyTabletopGraspMP-v0
robot:     G1 whole-body + Dex3
rate:      50 Hz
object:    当前数据中的单一 tabletop object
```

当前生产系统不是 Diffusion Policy。它从真实轨迹库检索完整计划，learned policy 根据
MuJoCo self/target state 对完整 tracker command 做闭环修正：

```text
self + object + goal observation + retrieved complete-plan context
              ↓
   plan-conditioned PPO actor
              ↓
   每步完整 36D tracker input
              ↓
        SIMPLE AMO tracker
              ↓
         MuJoCo execution
          ↙            ↘
GRAIL-style task reward  optional executed-motion window
                              ↓
                    frozen unconditional diffusion
                              ↓
                    additive SMP ablation only
```

episode 内连续的 actor 输出自然构成 `[T, 36]` 完整轨迹。网络内部以检索计划为 proposal
并预测 state-feedback correction，但对外 action、PPO distribution 和 log-prob 始终定义在
最终完整36D command上。Tracker 必须保留，因为36D是高层 whole-body command，
不是 torque 或完整关节控制量。
当前可用路径为完整计划检索 + plan-conditioned task-only PPO；推理时不加载 diffusion。

## 2. 已冻结的输入输出契约

### 2.1 Actor 输入：192D

```text
43  joint position
43  joint velocity
 3  projected gravity
 3  pelvis linear velocity
 3  pelvis angular velocity
 1  pelvis height above mean ankle height
36  previous physical tracker command
 3  object position in pelvis frame
 6  object rotation-6D in pelvis frame
 3  object linear velocity
 3  object angular velocity
 3  object position in right-hand frame
 6  object rotation-6D in right-hand frame
 3  table position in pelvis frame
 6  table rotation-6D in pelvis frame
24  eight right-hand link contact-force vectors in pelvis frame
 3  object-to-goal translation
---
192
```

这些 target 信息全部在 RL 时由 MuJoCo `MjData` 获得，不要求数据集记录物体参考轨迹。
与 GRAIL 相比，保留 current object、hand-object、table、contact force；删除 current/future
reference-object trajectory，并加入 actual object velocity 和 reset-relative lift goal。

无计划 actor 的 critic 输入为最近10帧，即`10 × 192 = 1920D`。当前生产 policy 另加入10个
0.1秒间隔的未来计划帧；每帧为36D完整命令、3D reference-object delta和1D contact intent，
再加phase，总 actor 输入为`192 + 10×40 + 1 = 593D`，critic history为5930D。

### 2.2 Actor 输出：完整 36D tracker input

```text
[ 0: 7] left hand
[ 7:14] right hand
[14:21] left arm
[21:28] right arm
[28:31] torso rpy
[31]    base height
[32]    torso vx
[33]    torso vy
[34]    turning flag
[35]    target yaw
```

Actor 输出为 `[-1,1]` 归一化量，经 train-data piecewise affine range 和每维 slew limit
解码成物理 36D command，再进入现有 `eval_move_actuators`/AMO 执行链。

### 2.3 无条件 diffusion 输入：10 × 82D executed motion

原始在线/离线 frame 为 80D：

```text
pelvis position/quaternion/linear velocity/angular velocity  13
43 joint positions                                         43
8 wrist/fingertip world positions × 3                      24
                                                             --
                                                             80
```

每个 10 帧窗口锚定到最后一帧 pelvis xy/yaw，转换为 82D SMP feature：

```text
root local position 3 + rotation-6D 6 + joints 43
+ 8 local link positions × 3 + root linear/angular velocity 6 = 82
```

Diffusion 接口严格是 `model(x_t, diffusion_timestep)`，没有 object/goal condition，
也不被 PPO 更新。

## 3. Diffusion 与 SMP reward

数据按 episode 划分为 80/10/10，窗口不跨 episode：

```text
train windows: 9,066
val windows:   1,138
test windows:  1,122
total frames:  12,226
```

模型沿用 SMP 量级：DiT hidden 128、4 heads、2 layers、50 diffusion steps，训练目标为
epsilon-prediction L1。在线评分固定使用 `K={8,15,22}`：

```text
raw_error = mean_k MSE(model(add_noise(x,k), k), noise)
r_smp     = exp(-ws * running-mean-normalized-error), ws=4
```

reward composition 同时保留 SMP 原始乘法与用户要求的加法消融；`smp_active` 只在
10 帧窗口填满后为1：

```text
task_component:     task_weight * (target - penalty)
task_only:          task_component + terminal
smp_additive:       task_component + smp_weight * r_smp * smp_active + terminal
smp_product:        task_component * r_smp + terminal
smp_product_strict: task_weight * (target * r_smp - penalty) + terminal
```

SMP 发布代码的 `task_smp_product` 是 task target 与 prior 相乘，`smp_product_strict`
对应这一规则，并将 safety penalty 留在乘法之外；`smp_additive` 是本项目按用户建议加入的
消融。本项目的正式对照也证明，在抓取尚不稳定时原始乘法会压低 task signal，而即使使用
小权重 additive SMP 也未提高成功率，
所以当前主训练配置是完全不加载 diffusion 的 `task_only`。

## 4. Reference-free GRAIL-style reward

Task reward 使用 MuJoCo 当前物体状态，不读取 object reference trajectory。旧版
`dense_v1` 为：

```text
target = reach
       + 2.0 * pregrasp
       + contact
       + 2.0 * grasp_quality
       + 0.5 * finger_opposition
       + 0.25 * gated_xy
       + 4.0 * gated_lift
       + stable
       + 4.0 * hold
```

关键定义：

- `pregrasp`：thumb 与 index/middle support 到物体表面的几何均值接近奖励。
- `grasp_quality`：thumb force score 与最佳 support force score 的几何均值。
- `lift/hold/stable`：由 grasp quality、lift fraction 和 object velocity 连续门控。
- penalty：near-object approach velocity、hand-table force、action rate、joint limit。
- success：物体相对 reset 抬升至少 2 cm，同时保持 thumb 与 index/middle 的双侧
  MuJoCo grasp contact，净持续 13 帧。
- success terminal bonus `+5`；fall `-1`；lift 后 drop `-0.5`。

真实轨迹审计发现 `dense_v1` 虽然能正确给成功轨迹高分，但绝对 contact/grasp reward 每帧
重复发放：不抬升、只在桌面高度保持接触直到 timeout 的轨迹最高可得 15.59，已超过部分
成功轨迹的 13.82。因此新增、不覆盖旧实验的 `progress_v2`：

```text
progress = 0.5*reach + pregrasp + 1.5*grasp_quality
         + 0.5*finger_opposition + 2.0*lift + stable
target_v2 = 50 * max(progress - best_progress_so_far, 0)
          + 10 * lifted_bilateral_hold + 2 * stable
terminal_v2 = success +10 / fall -5 / drop -3 / other failure or timeout -2
```

同一接触状态不再反复获得 progress reward；只有刷新进度以及已经抬起后的双侧保持有持续
回报。profile 通过 `--task-reward-profile dense_v1|progress_v2` 显式选择并写入 checkpoint
配置，旧结果仍可复现。

GRAIL 原始 `object_tracking_reward` 使用 object position/orientation/linear/angular velocity
reference error，并由接触门控；`reward_grasp` 使用接触手指数。这里没有 object reference
trajectory，不能把“绝对低速度”误当成 velocity tracking，否则正常上抬也会被拒绝。
因此 stable velocity 只作为软奖励，成功由“受控接触 + 抬升 + 持续时间”定义，抛掷会因
失去 grasp contact 无法累计成功计数。

### 4.1 GRAIL 发布配置的直接适配：`grail_release_v1`

重新逐项核对 `GRAIL/imports/SONIC/.../hoi/pnp_table.yaml` 后，发布配置的实际权重为：

- `object_tracking_reward=0`、`finger_primitive_limit=0`；
- `grasp_reward=5`，0.1 N link threshold、`min_contacts=8`；
- `grasp_finger_direction=10`，并显式使用 contact center；
- `tracking_anchor_ori=2.5`、`tracking_relative_body_ori=5`；
- `approach_velocity_penalty=-15`、`hand_table_contact_penalty=-1`；
- `meta_action_rate_l2=-0.1`、`full_latent_rate_l2=-0.01`、异常终止 `-10`。

GRAIL 的 actor 同时观测当前 proprioception/object/table、object future delta、robot
multi-future reference；因此其逐帧 robot tracking reward 不会允许静止刷分。本项目按最初的
SMP 设计不向 actor 输入逐帧 reference，robot motion quality 由无条件 diffusion guidance
负责，不能伪造一个 GRAIL reference tracking term。`grail_release_v1` 只直接迁移 GRAIL 的
target/safety 部分，并增加 reference-free 抬升判据：

```text
target = pregrasp_distance_kernel
       + 5 * link_contact_fraction
       + 10 * bilateral_finger_direction
       + 5 * contact_gated_lift
       + 2 * contact_gated_stability

penalty = 15 * ||wrist_velocity||^2 * exp(-distance^2 / 0.25^2)
        + clamp(hand_table_force, 0, 1)
        + 0.1 * normalized_action_rate

step_reward = 0.02 * (target - penalty) + terminal
terminal = success:+20 / fall,drop,other failure:-10 / timeout:-5
```

具体适配约束：

- link contact fraction 直接统计 MuJoCo 的8个 palm/finger proxy links，复现 GRAIL
  `min_contacts=8` 的比例奖励；
- GRAIL 实现中已有 `hand_fingers_object_distance`，但发布配置因有逐帧 reference tracking
  将其注释；SMP prior 不包含目标方向，因此以权重1启用 thumb–support surface pregrasp
  distance kernel，给接触前阶段提供可学习的目标信号；
- finger direction 相对 MuJoCo object–finger contact center 计算，而不是物体质心；数据回放
  已证明用质心会错误地把专家抓取判为负值；
- GRAIL 用 reference contact label 门控 finger direction；SMP 没有此标签，所以改为真实
  thumb–support 双侧抓取门控，防止单指触碰刷分；
- 成功为抬升2 cm并保持双侧抓取13帧；首次双侧抓取后40帧仍未抬升5 mm则失败，短暂松手
  不重置宽限期；
- 真实 SIMPLE 轨迹在合法抓取时会有手指轻触桌面，不能照搬 GRAIL 的1 N table-contact
  immediate termination，但保留其逐帧 `-clamp(force,1)` 惩罚。

新增 `grasp-rl audit-reward`，在同一 scene 上成对执行 expert+hold、no-motion、halfway
hold、open hand、contact hold、lift 后 release，以及末尾重复 expert。只有专家成功率、
反事实失败率、逐 episode 排序、全局 return separation 和 reset outcome repeatability 全部
过门槛，奖励才允许进入 PPO。

最终100场景审计保存于
`outputs/grasp_rl/reward_audit_grail_pregrasp_full100/reward_audit.json`：

- expert+hold成功99/100；no-motion、halfway hold、open hand、contact hold、lift 后 release
  均为0/100；
- `grail_release_v1` expert平均return 28.192；成功expert最低25.362；所有反事实最高
  0.978，全局margin 24.384；
- contact hold平均-4.203，lift 后 release平均-5.053，说明接触或瞬时抬升不能刷出成功回报；
- expert末尾重复回放仍为99/100，100% episode outcome一致，最大return差0.375；
- 全部自动acceptance checks通过，`grail_release_v1` 因而替换 `dense_v1` 成为默认 profile。

## 5. 实现布局

```text
src/simple/grasp_rl/
  schema.py       tensor/action 唯一 schema
  tracker.py      36D decode、slew limit、SIMPLE ActionCmd adapter
  motion.py       offline/online 共用 80D frame → 82D feature
  state.py        object/hand/table/contact MuJoCo extractor
  rewards.py      task reward、success、termination
  env.py          headless SIMPLE 单环境与 exact fast reset
  vec_env.py      multi-process workers、batched diffusion guidance
  prepare.py      replay audit、split、normalization/action transform
  diffusion.py    unconditional model/train/resume/score
  policy.py       BC/PPO 共用 SMP 量级 actor 构造与 checkpoint 加载
  bc.py           物理重放 BC、接触 curriculum DAgger、actor pretrain
  train.py        RSL-RL PPO、resume/warm-start/curriculum options
  evaluate.py     deterministic full-start/RSI/offset evaluation
  collect.py      fast-reset随机任务、同目标plan fallback、成功轨迹数据集生产
  cli.py          prepare/pretrain/train/evaluate/collect-policy/render
```

为保证快重置正确，`G1Wholebody` 和 `AMO_Policy` 增加 runtime-state capture/restore。
只恢复 `qpos/qvel/ctrl` 会清空 AMO history，导致 tracker 与物理状态不一致；这是本次实验中
已定位并修复的关键问题。接触后的 MuJoCo solver state 不适合直接缓存，因此 RSI 仍从
站立快照物理重放 demonstration prefix。

## 6. 可复现命令

在 SIMPLE 根目录运行：

```bash
PYTHONPATH=src .venv/bin/python -m simple.grasp_rl.cli prepare \
  --dataset data/simple/G1WholebodyTabletopGraspMP-v0 \
  --output data/grasp_rl/G1WholebodyTabletopGraspMP-v0

PYTHONPATH=src .venv/bin/python -m simple.grasp_rl.cli pretrain \
  --processed data/grasp_rl/G1WholebodyTabletopGraspMP-v0 \
  --output outputs/grasp_rl/diffusion --epochs 1000 --min-epochs 1000

PYTHONPATH=src .venv/bin/python -m simple.grasp_rl.cli prepare-bc \
  --dataset data/simple/G1WholebodyTabletopGraspMP-v0 \
  --processed data/grasp_rl/G1WholebodyTabletopGraspMP-v0 --workers 7

PYTHONPATH=src .venv/bin/python -m simple.grasp_rl.cli pretrain-actor \
  --processed data/grasp_rl/G1WholebodyTabletopGraspMP-v0 \
  --output outputs/grasp_rl/bc_actor --sources bc --epochs 500

PYTHONPATH=src .venv/bin/python -m simple.grasp_rl.cli train \
  --processed data/grasp_rl/G1WholebodyTabletopGraspMP-v0 \
  --output outputs/grasp_rl/ppo_task_progress_v2 --variant task_only \
  --task-reward-profile progress_v2 \
  --num-envs 42 --iterations 500 --worker-devices 1,2,3,4,5,6 \
  --action-std 0.005 --exploration-hold-steps 8 \
  --no-observation-noise --learning-rate 0.0001 \
  --actor-lr-scale 0.01 --learning-schedule fixed \
  --ppo-epochs 1 --num-mini-batches 4 --freeze-actor-normalizer \
  --actor-warm-start outputs/grasp_rl/bc_actor_curriculum_p80/best.pt

PYTHONPATH=src .venv/bin/python -m simple.grasp_rl.cli train \
  --processed data/grasp_rl/G1WholebodyTabletopGraspMP-v0 \
  --output outputs/grasp_rl/ppo_smp_additive \
  --diffusion outputs/grasp_rl/diffusion/best.pt \
  --variant smp_additive --num-envs 42 --iterations 500 \
  --worker-devices 1,2,3,4,5,6 \
  --task-reward-weight 0.02 --smp-reward-weight 0.01 \
  --action-std 0.005 --exploration-hold-steps 8 \
  --no-observation-noise --learning-rate 0.0001 \
  --actor-lr-scale 0.01 --learning-schedule fixed \
  --ppo-epochs 1 --num-mini-batches 4 --freeze-actor-normalizer \
  --actor-warm-start outputs/grasp_rl/bc_actor_curriculum_p80/best.pt

# 当前可用的随机目标数据生产命令
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m simple.grasp_rl.cli collect-policy \
  --checkpoint outputs/grasp_rl/ppo_task_random_stable_v1_300/model_299.pt \
  --processed data/grasp_rl/G1WholebodyTabletopGraspMP-v0 \
  --dataset data/simple/G1WholebodyTabletopGraspMP-v0 \
  --output outputs/grasp_rl/policy_dataset_task_random_100_v1 \
  --successes 100 --max-attempts 150 --scene-hold-attempts 16 \
  --target-position-jitter-xy task --target-yaw-jitter 0.15 \
  --reference-ranks 0,1 --seed 20260801 --device cuda:0
```

RSI curriculum 可用 `--rsi-dataset ... --rsi-prefix 100,100 --rsi-probability 1`；
`--warm-start` 载入 PPO actor+critic，`--actor-warm-start` 可沿多级 BC 初始化链恢复最近
PPO critic，`--resume` 用于同一实验精确续训。`evaluate --episode-offset N` 可并行评估
互不重叠的场景切片。

## 7. 已完成实验（2026-07-28）

### 7.1 数据与 diffusion

- 100 episodes 全部 finite；36D tracker replay 可运行。
- median joint replay RMSE 约 0.00813 rad；arm RMSE 约 0.000895 rad。
- diffusion 完成 1,000 epochs；best validation L1 为 0.128898。
- test real-window median MSE 0.04618；time-shuffle 0.08982，ratio 1.945。

### 7.2 轨迹长度与保持监督修复

- 原数据最长 142 帧，旧环境只允许 128 帧；示范又在刚抬起后立即结束，而成功需要持续
  hold，导致数据与成功定义结构性不相容。
- 环境、VecEnv、reward、evaluation 现统一为 192 步。
- BC 物理重放在每条记录动作后，通过真实 SIMPLE tracker + MuJoCo 重复末端完整 36D
  command 40 帧；不是复制 observation 或伪造 object state。
- 新 BC 数据为 16,226 帧，100/100 有 grasp contact，99/100 抬升至少 2 cm，99/100
  满足持续成功；平均 tracker slew error 为 `5.47e-5`。

### 7.3 BC/DAgger curriculum

- 纯 teacher-forcing BC 离线损失低但存在闭环 covariate shift；右臂/右手在第2–3步后
  逐渐偏离。
- prefix110 on-policy DAgger：5,134 帧，94/100 接触、67/100 抬升≥2 cm。
- 聚合 actor 在 prefix110 的14场景评估为 8/14 成功；向前移动到 prefix80 时曾达到
  5/14 成功。
- 最佳 full-start actor 为 `outputs/grasp_rl/bc_actor_curriculum_p80/best.pt`。100个不重叠
  数据场景确定性评估：2/100 成功、76/100 接触、28/100 抬升≥2 cm，平均最大抬升
  0.01832 m。成功 episode 为1和53。

### 7.4 奖励审计、CLI 修复与正式 PPO 对照

奖励审计使用原始100条 recorded 36D command，经真实 SIMPLE tracker/MuJoCo 执行，并在
第一次 terminal transition 截断 reward（BC 仍可继续收集末端 hold 监督）：

- 严格只执行原始12,226帧、不追加 hold：100/100有双侧接触、97/100抬升≥2 cm，但
  **0/100满足13帧持续成功**。这说明 terminal success 与原始数据长度并不对齐。
- 通过 tracker/MuJoCo 继续执行末端真实36D command 40帧后，`dense_v1` 为99/100成功，
  平均 task-only return 14.985、平均每步 raw target 3.803。
- 同样的100条延长重放在 `progress_v2` 下为99/100成功，平均 return 19.476、平均每步
  raw target 3.629。
- `dense_v1` PPO 的3条成功轨迹平均 return 19.80；timeout 平均7.41，但最高15.59，证实
  旧 absolute dense reward 存在接触刷分漏洞。

逐项专家均值还显示：`dense_v1` 每步3.803中，reach/pregrasp贡献1.467、contact/grasp/finger
贡献1.119、xy/lift/stable/hold贡献1.217；penalty 总计仅0.00128/步，约为 positive target 的
0.034%，实际几乎不起作用。`progress_v2` 的3.629由 progress bonus 2.091、lifted hold 1.343、
stable 0.195组成，排序更合理，但目前 PPO 成功率没有提高。因此 task state/reward signal 可用，
而旧 shaping 权重和13帧 terminal gate 不能称为已验证正确。

此前写入本节的“task/SMP PPO 均退化到站立”实验无效：CLI 虽解析了
`--actor-warm-start`、`--actor-lr-scale` 和 `--learning-schedule`，却没有把它们传入
`PpoTrainConfig`，实际训练从随机 actor 或错误学习率开始。现已修复并有 CLI 回归测试；
下面只报告修复后的42环境、500 iterations、504,000 transitions、100固定场景确定性结果：

| actor/checkpoint | task profile | diffusion | success | contact | lift≥2 cm | mean return | mean max lift |
|---|---|---|---:|---:|---:|---:|---:|
| BC curriculum best | dense_v1 | none | 2/100 | 76/100 | 28/100 | 4.935 | 0.01832 m |
| task-only PPO `model_300.pt` | dense_v1 | none | **3/100** | 78/100 | 20/100 | 6.387 | 0.01520 m |
| additive-SMP PPO `model_300.pt` | dense_v1 | frozen unconditional | 1/100 | 72/100 | 18/100 | 5.751 | 0.01580 m |
| task-only PPO `model_100.pt` | progress_v2 | none | 2/100 | **84/100** | **34/100** | 1.491* | **0.01873 m** |

`*` 不同 task profile 的 return 数值不可直接横向比较。全部 PPO 使用 action std 0.005、
噪声保持8个控制步、actor LR `1e-6`/critic LR `1e-4`、fixed schedule、单 PPO epoch。

结论：去掉 SMP 比 additive SMP 更好；`progress_v2` 修复了奖励排序并提高接触/2 cm抬升率，
但没有把 full-start 成功率提高到 BC 以上。当前主要失败已收窄为“能接触、部分能短暂抬起，
但不能稳定保持13帧”，因此 reward 有问题但不是唯一原因。2–3%仍不是可用通用抓取策略；
单物体100条数据也不能支持 unseen-object 泛化结论。

视频核验又发现一个独立问题：100场景 sweep 可重复得到 episode 26/80/99 成功，但把这些
episode 单独启动会失败；先按原顺序运行前置 episode 后再录制，则 episode 26 再次成功。
这说明 state-dict reset 仍未完全隔离 AMO/tracker 隐藏状态。因此表中的3/100是当前 evaluator
顺序下的可重复结果，不能视为3个相互独立的成功样本。渲染工具支持 `--context-start 0`，
并为每个视频写出含 `success` 的同名 JSON，防止把不成功的 command replay 当成成功视频。

### 7.5 GRAIL reward、归一化 RSI 与 state-feedback teacher（2026-07-29）

重新对照 GRAIL 发布实现后，`grail_release_v1` 已成为默认 task profile。修正 wrist
velocity 为 MuJoCo 真实 wrist link 速度后的20场景审计为：expert 20/20，五类反事实均
0/20，成功 expert 最低 return 26.057，失败最高 -0.431，margin 26.488；全部 acceptance
checks 通过。完整100场景报告写入
`outputs/grasp_rl/reward_audit_grail_wrist_full100/reward_audit.json`。

最终100场景 wrist-corrected audit 同样全部通过：expert+hold 99/100，no-motion、
halfway-hold、open-hand、contact-hold、lift后release 均0/100；成功expert最低return
25.432，全部反事实最高1.001，全局margin 24.431。expert repeat仍为99/100，outcome
一致率100%，最大return差0.379。contact-hold与lift后release分别100/100触发失败，
证明桌面接触或瞬时抬升不能刷成成功。

绝对 prefix 会因示范长度108–142帧而混合不同任务阶段，现已实现：

- `--rsi-phase low,high`：按每条轨迹归一化 phase；
- `--rsi-stage pregrasp|grasp_to_lift|lift`：由真实 bilateral contact/lift 状态确定边界；
- `--rsi-episodes ...`：仅重采样独立评估中的失败 scene；
- exact scene snapshot hold、RSI/full-start 分离在线指标。

固定 phase=0.90 的100场景结果：低噪声 task-only PPO model50 为43/100；继续50轮为
42/100；57个失败场景 mining + 旧式独立 Adam BC anchor 为39/100。它们分别新增场景、
同时丢失原成功场景，不能视为净进步。matched additive-SMP model50 在前20场景为11/20，
task-only baseline 为12/20；joint BC-gradient anchor 为10/20。因此当前仍关闭 diffusion，
且 task/SMP 相加没有优于纯 task reward。

为检验 reward/tracker 与监督瓶颈，新增 `build-knn` 和 replay-derived state-feedback
teacher：

- stateless 192D state→36D command kNN：full-start 90/100；
- reset 后轨迹起点检索 + same-trajectory ±4帧状态反馈：full-start **91/100**；
- 不读取 object reference trajectory，输出仍由 SIMPLE tracker 执行；
- 这是 all-data 非参数诊断/DAgger teacher，不是 PPO、也不是最终 SMP MLP。

结果证明 MuJoCo target observation、GRAIL task/success reward 和 tracker path 能支持
>90% 的闭环执行；learned policy 的主要问题是轨迹偏离后的动作映射。实现并验证了
teacher block mixing DAgger（每8帧选择完整 teacher/student command，禁止36D逐维线性
平均）、初始化 checkpoint normalizer 保留冻结、BC/PPO joint-gradient anchor、以及
on-policy teacher-gradient PPO。当前结果：

- β=0.9 block teacher rollout：74% contact、39% lift≥2cm、26% success；
- 线性0.9 action blend 只有24% contact、6% lift/success，已淘汰；
- 蒸馏后的普通512–256–128 MLP 最好为 full-start 1/20；
- teacher-guided PPO model99 为0/20 success、0/20 bilateral contact，但16/20短暂
  lift≥2cm，说明它学到了完整抬升运动却平均掉了抓取闭合；
- teacher 本身91/100，不能作为“RL成功率”报告。

可播放的闭环 teacher 成功视频：
`outputs/grasp_rl/videos/knn_temporal_episode0_success.mp4`，H.264/yuv420p/640×360/50fps；
相邻 JSON 记录 `success=true`、最大抬升3.26cm。

### 7.6 PPO 失败根因与 plan-conditioned 上界（2026-07-29）

进一步逐项对照 SMP/GRAIL 后，task reward 不是当前 0–10% PPO 的主故障。证据链如下：

- `grail_release_v1` 的100场景 replay audit 仍是 expert 99/100、五类反事实0/100，成功与
  失败 return 全局 margin 24.431；reward/success 判据可以区分真实完成与刷分。
- 普通 reference-conditioned MLP 经归一化修复后只有2/20；500轮 PPO 的最佳 checkpoint
  在扩展20场景仅1/20。减小右臂/手探索、actor LR降到`2e-6`、单PPO epoch、严格
  GRAIL robot-reference reward和BC joint-gradient anchor后，六个 checkpoint 在相同前10
  场景仍只有0–1/10，未超过初始化。
- 动作干预精确定位到右侧操作14维：仅替换右手为 reference action 是2/10；同时替换右臂
  和右手是10/10；替换全部36维同样10/10。故障不在腿/躯干、tracker、MuJoCo或成功判据，
  而在普通 MLP 对右臂/右手完整命令的微小回归误差。
- GRAIL `pnp_table` 使用4096环境×24步×20,000轮，约19.66亿 transitions；当前正式 run
  为56×24×500=67.2万，相差约2926倍。GRAIL还从预训练 tokenizer、0.1 latent residual和
  离散手部 primitive附近搜索；当前普通 MLP 是直接回归36个连续命令，起点只有约10%成功，
  PPO采集的大部分自然是失败轨迹。

为验证“必须先进入成功轨迹邻域”而新增 `PlanConditionedMLPModel`。它的对外 action 仍是
完整36D tracker command，PPO概率分布/log-prob也定义在该完整命令上；零初始化时采用计划
当前命令，网络可用自身/目标状态修正全部36维。当前仅用 exact replay plan 做诊断上界：

| checkpoint | 场景 | success | contact | lift≥2 cm | mean return |
|---|---:|---:|---:|---:|---:|
| zero-update plan-conditioned | 100 | 99/100 | 100/100 | 99/100 | 53.10 |
| PPO `model_100.pt` | 100 | **98/100** | 100/100 | 98/100 | 52.97 |
| PPO `model_499.pt` | 100 | **98/100** | 100/100 | 98/100 | 53.25 |

这证明 PPO 在高成功率动作邻域可以保持>90%，也证明此前失败不是简单的 reward 权重问题。
但这三个数字使用了数据集 exact action plan，**不能报告为 reference-free 或通用抓取成功率**；
它们是 pipeline/reward/PPO 的上界实验。当前 SMP diffusion 是无条件 motion-score prior，并
不会根据桌上目标生成对应的36D抓取计划，所以不能直接替代这个 plan slot。最终要么增加
target-guided action-plan sampling/trajectory generator，要么继续把普通无reference actor训练到
高成功率；在完成前，98%不能勾选通用策略验收项。

闭环 PPO 视频为
`outputs/grasp_rl/videos/ppo_plan_conditioned_model499_episode0_success.mp4`，相邻 JSON 明确记录
checkpoint、`closed_loop=true`、`success=true`、最大抬升3.21 cm。视频编码为
H.264/yuv420p/640×360/50fps。

### 7.7 随机目标完整 policy 与批量数据生产（2026-07-29）

为区分 exact replay 与真实随机目标，训练 reset 现在固定记录中的机器人/桌面场景，同时独立
重采样目标位置和yaw。原数据目标范围约为`x[-0.670,-0.624]`、`y[-0.030,0.027]`；生产评测
使用 SIMPLE 任务绝对范围`x[-0.670,-0.620]`、`y[-0.030,0.030]`。ReferenceLibrary 根据当前
object/hand/table geometry检索完整轨迹，policy仍输出最终36D tracker command。

两次正式随机化 PPO：

- broad relative-jitter run运行到577轮后发生遗忘，固定30任务的最佳已测checkpoint不超过
  80%，因此早停并保留所有checkpoint；
- task-distribution stable run为56环境×24步×300轮=`403,200` transitions，actor LR为
  `5e-6`、右臂/手std为0.002、reference reward weight为0.0002；`model_299.pt`在固定30任务
  为25/30=83.3%，高于初始化22/30=73.3%。

固定30任务上，rank-0与rank-1各为25/30，但失败集合部分互补；同目标依次尝试rank-0、rank-1
为27/30=90%。rank-2/3没有继续增加覆盖。单独写`torso_vx`的0.15/0.30/0.50三档消融完全
不改变成功集合，故不进入生产默认路径。

最终用未见seed 20260801运行106个独立随机目标：

| 指标 | 结果 |
|---|---:|
| rank-0单次成功 | 97/106 = **91.51%** |
| rank-1同目标救回 | 3 |
| rank-0→1系统级成功 | 100/106 = **94.34%** |
| 最终失败 | 6/106 |
| 成功轨迹吞吐 | 0.555条/秒 |
| 保存成功轨迹 | 100条，36 MB |

失败任务主要是相对记录场景x偏移3.6–4.7cm的边界组合。生产器不会重采样一个更容易的目标
冒充fallback：它恢复完全相同的robot runtime state、qpos/qvel和target pose后才切换
reference rank。manifest记录全部106个目标和115次plan rollout；只有100条成功轨迹写入
`trajectories/`，summary仍报告真实94.34%目标级成功率。

输出目录为`outputs/grasp_rl/policy_dataset_task_random_100_v1`。每条NPZ包含192D base
observation、593D policy observation、raw/physical 36D action、motion frames、task reward/
penalty、初始qpos/qvel、target pose、base/reference episode及reference rank。

对rank-0失败、rank-1救回的episode47做独立物理command replay，仍为success，最大抬升
2.98cm。视频为
`outputs/grasp_rl/videos/policy_dataset_random_rank1_rescue_episode47.mp4`，H.264/yuv420p、
640×360、132帧；同名JSON记录`success=true`。另外从相同随机初态重新运行checkpoint而非
回放保存动作，闭环policy同样成功、最大抬升2.97cm：
`outputs/grasp_rl/videos/policy_closed_loop_random_rank1_episode47.mp4`（132帧，
H.264/yuv420p）；同名JSON记录`closed_loop=true`和`success=true`。

## 8. 下一阶段执行顺序

不要继续加入 diffusion 或只改标量权重。下一阶段优先级为：

1. 当前单物体任务直接用`collect-policy`扩大成功数据，并把6个边界失败加入failure mining；
   rank-0与rank-1必须分别保留指标，不能只报告筛选后的100%数据集。
2. 训练显式base/reach动作前必须扩大包含机器人-目标极端相对位姿的数据；现有专家的
   `torso_vx/vy`恒为0，不能指望小std PPO凭空学会走近。
3. 若要求无轨迹库部署，再把检索计划蒸馏为target-conditioned完整36D generator；当前
   unconditional SMP diffusion仍只适合作motion prior，不能声称会生成目标抓取计划。
4. target-specific generator或普通actor达到稳定成功后才重做 task-only/SMP additive/product
   消融；当前 unconditional additive低于task-only，默认关闭。
5. 最后加入多物体数据和 shape/BPS/point-cloud input、质量/摩擦/尺寸随机化，单独报告
   held-out object success rate。

## 9. 验收状态

- [x] 完整 36D policy output，经 SIMPLE tracker 执行。
- [x] 192D self/target/goal observation，target 在线来自 MuJoCo。
- [x] 10×82D unconditional diffusion，无 target condition。
- [x] SMP fixed-timestep guidance 与 task-only/additive/product ablation。
- [x] 数据 prepare、diffusion train/resume、PPO train/resume/warm-start、evaluate CLI。
- [x] 多进程训练、headless EGL、tracker-aware fast reset、RSI、checkpoint 与测试。
- [x] 修正 warm-start 后的 task-only、additive SMP、dense_v1/progress_v2 正式实验和100场景评估。
- [x] full-start learned MLP 非零抓取成功；teacher 上限91/100。
- [x] exact replay-plan-conditioned PPO >90%（98/100 pipeline上界，不能替代下一项）。
- [x] 单物体任务随机目标production policy：rank-0 91.5%，rank-0→1同目标fallback 94.3%。
- [x] 批量成功轨迹生产CLI与100条真实输出，失败/重试未从指标中隐藏。
- [ ] generated-plan或无reference full-start learned MLP/PPO >90%；retrieval分数不得替代。
- [ ] observation/dynamics noise鲁棒性与极端reach组合>90%。
- [ ] multi-object/unseen-object generalization。
