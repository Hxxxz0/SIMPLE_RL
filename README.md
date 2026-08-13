<h1 align="center">SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation
</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2606.08278-df2a2a.svg)](https://arxiv.org/abs/2606.08278)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://psi-lab.ai/SIMPLE)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/USC-PSI-Lab/psi-model)
[![Data](https://img.shields.io/badge/Hugging%20Face-Data-pink)](https://huggingface.co/datasets/USC-PSI-Lab/psi-data)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

</div>


<p align="center">
  <img src="assets/teaser.webp" alt="SIMPLE teaser image" />
</p>


Contributors: [Songlin Wei](https://songlin.github.io/)\*, [Zhenhao Ni](https://nizhenhao-3.github.io/)\*, [Jie Liu](https://jie0530.github.io/)\*, [Zhenyu Zhao](https://zhenyuzhao.com/)\*, [Junjie Ye](https://junjieye.com/), [Hongyi Jing](https://hongyijing.me/), Junkai Xia, [Xiawei Liu](https://www.xiaweiliu.com/), [Michael Leong](https://leongmichael.github.io/), [Liang Heng](https://liangheng121.github.io/), Di Huang, [Yue Wang](https://yuewang.xyz/)†

> 


## 📢 News & Updates
+ [2026-07-14] We released support for World Action Models: [Cosmos3](https://github.com/songlin/cosmos-framework/blob/main/docs/action_policy_simple_posttrain.md) and [DreamZero](https://github.com/physical-superintelligence-lab/Psi0/blob/main/baselines/dreamzero/README.md). 
+ [ ] Integrate SONIC whole-body controller.

## grasp_anything 物体抓取发布（2026-08-14）

`grasp_anything` 当前已有 5 种物体的独立 RL checkpoint：`Soap_Bottle_1`、
`Bottle_1`、`Apple_1`、`Bowl_1` 和 `Cup_6`。选定权重、SHA256、验收数据、
适用边界和本机视频绝对路径已归档到
[`releases/grasp_rl/mjlab_gpu/grasp_anything/v1`](releases/grasp_rl/mjlab_gpu/grasp_anything/v1)，
大文件通过 Git LFS 发布。

验收成功要求物体在原生物理中被抓住、抬高至少 9 cm 并保持 13 个控制步，
不把仅接触或闭合手指计为成功。`Soap_Bottle_1` 和 `Bottle_1` 支持 narrow
pose DR，但需要 lift-arm-decay 运行时变体；`Apple_1` 和 `Bowl_1` 目前只是
fixed-pose baseline，不宣称 narrow/workspace DR 支持。五种物体都已有成功视频。

## mjlab CUDA PPO 可复现发布（2026-08-03）

`xmove_pick` 的首个自包含 GPU release 位于
[`releases/grasp_rl/mjlab_gpu/xmove_pick/v1`](releases/grasp_rl/mjlab_gpu/xmove_pick/v1)。
它包含选中的 RSL-RL PPO 权重、训练初始权重、冻结的 mjlab/控制器资产、processed
reference、双 seed 同世界 benchmark、3 条成功 NPZ 轨迹，以及全身和抓取近景成功视频。
依赖锁定在 `mjlab_gpu/uv.lock`，并通过 Git LFS 发布大文件。

选中的 `model_299` 累计使用 58,982,400 条 fresh on-policy transition。两个 128-world
full-DR seed 上，PPO 和 noise-matched proposal-only 都是 173/256（67.58%）；PPO mean task
return 为 24.264，proposal 为 24.124。因此当前结论是成功率不下降且 return 小幅提升，
不是“PPO 成功率显著提升”，也没有把 67.58% 写成达到 70%。完整安装、verify、benchmark、
collect、record、warm-start、exact resume 和 fine-tune 命令见 release README。

## SIMPLE_RL：完整轨迹抓取新方案

本分支在 SIMPLE 上实现了一个面向 G1 的完整轨迹抓取系统。当前主方案不是 diffusion
policy，也不是 GRAIL 式外部小残差控制，而是：

```text
MuJoCo 在线 self / object / goal 状态（192D）
                    +
检索到的未来完整 tracker plan（401D）
                    │
                    ▼
       plan-conditioned PPO policy（593D）
                    │
                    ▼
        最终完整 36D tracker command
                    │
                    ▼
       SIMPLE AMO whole-body tracker → MuJoCo
```

### 方案边界

- policy 每步输出最终的完整 36D 运控命令，SIMPLE tracker 仍负责将其变成底层关节控制；
- 当前 plan 来自 processed BC replay 库的几何近邻检索，不是 target-conditioned diffusion
  生成。因此当前结果属于“检索计划 + 状态反馈 PPO”，不是 reference-free policy；
- plan 中包含 replay 得到的未来 object position delta 和 contact label。原始轨迹不需要记录
  object state，因为这些量是在真实 SIMPLE tracker + MuJoCo replay 时生成的；
- task reward 不跟踪 object reference trajectory，只依据当前 MuJoCo 物体状态、接触和任务
  goal 判断完成情况；
- 当前 `bend_pick` 结果验证的是同一物体在原生位置/yaw范围内的随机目标抓取，不代表
  unseen-object 通用抓取。

### Policy 完整输入：593D

基础状态是以下 192D privileged MuJoCo observation：

| 索引 | 信号 | 维度 |
| :--- | :--- | ---: |
| `0:43` | G1 43个关节位置 | 43 |
| `43:86` | G1 43个关节速度 | 43 |
| `86:89` | pelvis坐标系重力方向 | 3 |
| `89:92` | pelvis线速度 | 3 |
| `92:95` | pelvis角速度 | 3 |
| `95` | pelvis相对脚踝高度 | 1 |
| `96:132` | 上一步实际执行的完整command | 36 |
| `132:135` | object相对pelvis的位置 | 3 |
| `135:141` | object相对pelvis的6D旋转 | 6 |
| `141:144` | object在pelvis坐标系中的线速度 | 3 |
| `144:147` | object在pelvis坐标系中的角速度 | 3 |
| `147:150` | object相对右手的位置 | 3 |
| `150:156` | object相对右手的6D旋转 | 6 |
| `156:159` | table相对pelvis的位置 | 3 |
| `159:165` | table相对pelvis的6D旋转 | 6 |
| `165:189` | 8个右手link的三维接触力 | 24 |
| `189:192` | 当前object到抬升goal的位置差 | 3 |

未来计划使用控制步偏移 `0,5,...,45`，即50 Hz下每0.1秒一个点，共10点。每点40D：

```text
未来完整 normalized tracker command  36D
reference object position - 当前 object position  3D
reference bilateral grasp label  1D
```

再加一个归一化 plan phase，得到 `10 × 40 + 1 = 401D`。最终输入为
`192 + 401 = 593D`。网络为 `593 → 512 → 256 → 128 → 36`，执行命令为：

```text
complete_command = current_plan_command + MLP(current_state, future_plan)
```

PPO 的 distribution、采样动作和 log-prob 都定义在这个最终完整 command 上；不存在另一个
绕过 PPO 的外部 residual executor。

### Policy 完整输出：36D

| 索引 | 内容 | 维度 |
| :--- | :--- | ---: |
| `0:7`, `7:14` | 左手、右手 | 7 + 7 |
| `14:21`, `21:28` | 左臂、右臂 | 7 + 7 |
| `28:31` | torso roll/pitch/yaw command | 3 |
| `31` | base height | 1 |
| `32:34` | torso `vx`, `vy` | 2 |
| `34` | turning flag | 1 |
| `35` | target yaw | 1 |

输出先通过该任务自己的 `action_transform.npz` 做非对称反归一化和 slew limit，再转成
SIMPLE `ActionCmd` 交给 AMO whole-body tracker。checkpoint 会保存 task/schema 和
action-transform SHA256，避免跨任务误用动作缩放。

### 通用复杂任务 RL v2

当前新增的 v2 框架覆盖合并数据集中的14类任务，但每个任务单独训练 policy。统一输入为
331D MuJoCo privileged state；使用完整计划时再拼接511D future-plan，最终输入842D。
Policy 每步直接输出完整36D tracker command，不是对专家轨迹的 residual。

统一 `GoalGraphReward` 支持抓取、放置、handover、可动关节、推物体和复合容器任务。
MP任务通过 AMO tracker 执行，Teleop任务通过 Sonic decoupled-WBC 执行。任务奖励只读取
当前物体、目标、接触和关节状态，不依赖物体参考轨迹。

首个重点复杂任务是 `G1WholebodyLocomotionPickBetweenTablesMixed-v0`。旧数据中的
`turning_flag/target_yaw` 丢失会在派生数据中修复，并通过闭环走近、稳定、开手补齐当前
tracker可执行的放置尾段；原始 `data/simple` 永不修改。详细接口与命令见
[`src/simple/grasp_rl/README.md`](src/simple/grasp_rl/README.md)。

该任务现已完成真实 tracker/MuJoCo 闭环验收：派生数据 50/50 可执行，10 条 expert 重放
与 repeat 全部成功，七组 no-motion/truncate/open-hand/early-release/time-shuffle/throw
反事实均为 0/10。发布 checkpoint 是
`outputs/grasp_rl/locomotion_pick_between_tables/bc_plan_v2_smoke/best.pt`。在同一批 50 个
独立随机目标（x ±2.5 cm、y ±3 cm、yaw ±0.15 rad）上，单一 base plan 为 37/50，
rank-0 为 28/50；base 与 rank-0...9 从完全相同初态依次尝试时覆盖 46/50，即严格的
target-level multi-plan success 为 **92%**。这不是单次 rollout 成功率，也不是 PPO 成功率。

最新 stage-masked PPO 保留固定场景 10/10，但 `model_25.pt` 和 `model_49.pt` 在大扰动
随机 10 场景都只有 5/10，低于初始化的 7/10，因此只保留为失败消融，生产路径不选 PPO。
当前生产器先试 base complete plan，再按互补 rank 回退；每次回退恢复完全相同的机器人、
物体 pose 和 MuJoCo 初态，只把最终成功轨迹写入 `trajectories/`，同时在 manifest 中保留
所有失败目标和 plan rollout。独立 seed 20260730 的实际生产结果是 20/22 个随机目标成功
（**90.91% target success**），共运行 52 条完整计划；base 成功 14 个，rank 6/4/5 又救回
3/2/1 个，两个最终失败目标各自的 11 次失败计划均保留。20 条 NPZ 已验证为完整
331D/842D observation、36D raw/physical command 和初始 MuJoCo/目标状态。当前正式路径
没有使用 SMP diffusion。

### 已训练的 Sonic 移动抓取任务（2026-07-31）

现在另有两个真正完成数据重放、reward audit、训练和随机目标验收的移动操作任务；其余
v2 task 仍只是共享代码/奖励接口支持，不能称为已有 policy：

| task | 可执行 source | 独立 test | 随机目标单 rollout |
| :--- | ---: | ---: | ---: |
| `xmove_pick` | 72/99 | 8/8 | 18/20（90%） |
| `xmove_bend_pick` | 24/100 | 3/3 | 18/19（94.74%） |

两者都只用 controller 重放成功的 episode 训练，source 失败不会进入 train/val/test；动作
变换 `successful_replay_cover_v1` 在带 slew limit 的逐帧 `encode -> decode` 路径上最大误差
为 `5.96e-8`。奖励要求真实力接触后的连续抬升：`xmove_pick` 至少 9 cm，
`xmove_bend_pick` 至少 8 cm，保持 13 个 50 Hz control steps。两份 reward audit 都是
expert/repeat 10/10，七类反事实 0/10。

发布 checkpoint 分别为
`outputs/grasp_rl/xmove_pick/bc_plan_v2_reversible/best.pt` 和
`outputs/grasp_rl/xmove_bend_pick/bc_plan_v2_reversible/best.pt`。它们是
plan-conditioned 36D complete-command BC policy，输出仍由 Sonic tracker 执行；不是 PPO，
也没有使用 diffusion/SMP。生产器现强制只从 processed manifest 的 replay-success scenes
采样，避免把已知 controller 重放失败场景误算成 policy 失败。独立 seed 20260731 已各自
产出 20 个成功、schema/有限值检查通过的 NPZ；在更大的 ±2.5/3 cm、±0.15 rad envelope
上，真实 target success 分别为 20/24（83.33%）和 20/21（95.24%）。所有失败目标保留在
manifest，未混进成功轨迹目录。四个随机闭环成功视频已放到对应
`data/grasp_rl/.../v2/videos/`，均为 H.264/yuv420p。

显式未见初态实验进一步区分了“回放”和泛化。`xmove_pick`数据的机器人XY恒为
`(-0.8,0)`，物体XY包围盒为`[-0.3197,-0.2905] × [-0.0797,-0.0401]`；将机器人设为
`(-0.78,0)`、物体设为`(-0.28,-0.06)`后仍为4/4成功。`xmove_bend_pick`数据的机器人
y恒为0、物体y不超过`-0.0400`；机器人`(-0.95,0.02)`、物体
`(-0.275,-0.025)`同样4/4成功。因此机器人和物体初始位置均可不在原数据中，但四方向
边界测试显示成功率明显非对称，不能称为无限范围泛化。完整结果见
`outputs/grasp_rl/ood_initial_pose_analysis.json`，两段联合OOD视频也已放入对应
`data/grasp_rl/.../v2/videos/`。

需要严格说明：上面两个 `bc_plan_v2_reversible/best.pt` 的 correction MLP 最后一层
weight/bias 均严格为0，所以这两个 **BC checkpoint** 的输出等于检索计划的36D command。
它们仍是 retrieval/playback + Sonic tracker 基线，不能标成RL。下面的 PPO checkpoint 是
在此之后单独完成的在线训练结果。

### 当前 GRAIL-v7 真实 PPO 结果（2026-07-31）

当前发布候选是修复奖励和 reset 后重新训练的 on-policy PPO，不是轨迹回放重命名：

- `xmove_pick`: `ppo_reference_grail_v7_fastreset_300/model_200.pt`；
- `xmove_bend_pick`: `ppo_reference_grail_v7_fastreset_300/model_180.pt`。

Actor 输入是 331D MuJoCo task state 与 511D complete-plan context，共 842D；输出始终是
最终完整 36D tracker command，再由 Sonic decoupled-WBC 执行。网络内部只允许右手和
右臂在抓取相关 stage 修正计划，但 PPO distribution、采样 action、log-prob 和保存到数据集
的 action 都定义在最终 36D command 上。

训练使用发布 GRAIL `pnp_table` 的 dense grasp 部分：`5×grasp_contact_fraction +
10×finger_direction - 15×approach_velocity - hand_table - 0.1×policy_residual_rate`；
另以加法加入 robot/reference tracking。两项均按 50 Hz 的 `0.02` 缩放。finger direction
由 reference contact intent 门控并使用 reference contact center；缺失 center 的旧数据才
按 GRAIL fallback 使用物体中心。reference 最后一条 command 执行后立即 motion timeout，
不会继续优化到无关的 800 步。

reward audit 先在真实 tracker 回放上验收：

| task | expert | no-motion/open-hand/contact-hold | expert mean return |
| :--- | ---: | ---: | ---: |
| `xmove_pick` | 3/3 | 全部 0/3 | 15.28 |
| `xmove_bend_pick` | 3/3 | 全部 0/3 | 14.20 |

严格配对评估固定 base/reference episode（xmove 82、bend 96），只随机物体 x/y
`±4/5 cm` 和 yaw `±0.25 rad`。每对的完整 `initial_qpos/qvel`、目标位置/四元数、base ID
和 reference ID 都逐元素相同；reference-only 直接执行同一 plan 的完整 36D command。

| task / selected checkpoint | reference only | PPO | PPO-only rescue | reference-only regression | exact McNemar |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `xmove_pick/model_200` | 32/100 | **49/100** | 17 | 0 | **`p=1.53e-5`** |
| `xmove_bend_pick/model_180` | 43/100 | **54/100** | 12 | 1 | **`p=0.00342`** |

这两个结果都证明 PPO 相对同一 reference 有统计显著的闭环增益。xmove 的策略/reference
物理 action 平均绝对差为 `7.86e-3`，bend 为 `4.13e-3`；差异仅出现在右手 `7:14`
和右臂 `21:28`，其它 22 维仍由完整计划提供。配对报告分别位于：

```text
outputs/grasp_rl/xmove_pick/ppo_reference_grail_v7_fastreset_300/paired_model200_100.json
outputs/grasp_rl/xmove_bend_pick/ppo_reference_grail_v7_fastreset_300/paired_model180_100.json
```

`compare-paired` 会验证所有配对初态字段并计算双侧 exact McNemar/binomial 检验，避免只比较
两个未配对成功率。旧 evaluator 曾错误地从 V1 的 192D offset 读取 V2 reference action；
现在按活动 schema 从 331D base observation 后读取，并有 V1/V2 回归测试。因此下方 v1
配对表仅保留为历史实验记录，不再用于当前结论。

真实 PPO rescue 与同初态 reference failure 视频为 H.264/yuv420p、640×360、50 fps：

```text
data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2/videos/grail_v7_model200_fixed82_repeat15_ppo_success.mp4
data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2/videos/grail_v7_fixed82_repeat15_reference_failure.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/grail_v7_model180_fixed96_repeat1_ppo_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/grail_v7_fixed96_repeat1_reference_failure.mp4
```

训练 worker 使用完整 MuJoCo、Sonic interpolation/lower-body history 和 deterministic control
clock 快照做 fast reset。同一快照重复执行相同 20 步 action 后 observation、motion frame、
qpos 和 qvel 均逐元素完全相同。训练仍会保存每 10 轮 checkpoint；发布模型按严格配对
筛选，而不是默认选择最后一轮。两条续训都完整结束并保存 `model_299.pt`；xmove
`model_280` 已退回 8/20（reference 8/20），bend `model_299` 为 10/20，与 model 180
的小样本相同但没有新的 100 对证据，因此不替换已验证的发布 checkpoint。

成功轨迹生产器只写入通过 MuJoCo task success 的 rollout，所有失败目标和每次 plan 尝试
仍保留在 `manifest.jsonl`。新 PPO 的实际生产结果为：

| task / envelope | 成功轨迹 | target success | plan rollouts |
| :--- | ---: | ---: | ---: |
| xmove `±4/5 cm, ±0.25 rad` | 20 | 20/28 (71.43%) | 70 |
| bend `±4/5 cm, ±0.25 rad` | 20 | 20/24 (83.33%) | 56 |
| bend `±2.5/3 cm, ±0.15 rad` | 20 | **20/21 (95.24%)** | 34 |

xmove 的常规 envelope 严格 22-target run 只有 18/22 (81.82%)，未达到 90%，所以不把它
列成完整 20 条生产结果。较大 envelope 的两份完整数据都已检查：每份恰好 20 个有限 NPZ，
包含 331D state、842D policy input、36D normalized/physical command 和 80D motion frame。
路径为 `ppo_grail_v7_model200_production20_hard` 与
`ppo_grail_v7_model180_production20_hard`；bend 95.24% 路径为
`ppo_grail_v7_model180_production20_standard`。

### 历史 Sonic 移动抓取 PPO 结果（已由 GRAIL-v7 替代）

`xmove_pick` 和 `xmove_bend_pick` 均训练了100个在线PPO update（checkpoint的 `iter=99`
表示第0--99轮）。reference policy 使用842D输入并输出最终完整36D command；state-only
policy只使用331D在线MuJoCo状态。两条线都保存 actor、critic、两组Adam optimizer state，
不是BC重命名。reference actor最后一层也已从严格0更新为非零：选用的
`xmove_pick/model_10.pt` 最大绝对weight为 `1.54e-4`，
`xmove_bend_pick/model_20.pt` 为 `2.27e-4`。

统一评测使用确定性完整起点，以及同一组前20个可执行场景上的 x/y
`±2.5/3 cm`、yaw `±0.15 rad` 随机目标。结果必须和BC基线一起看：

| task / policy | 固定独立test | 随机目标单rollout |
| :--- | ---: | ---: |
| `xmove_pick` BC playback基线 | 8/8 | 15/20 |
| `xmove_pick` reference PPO `model_10` | 8/8 | 14/20 |
| `xmove_pick` reference PPO `model_99` | 8/8 | 15/20 |
| `xmove_pick` state-only PPO `model_99` | 0/8 | 未进入随机生产 |
| `xmove_bend_pick` BC playback基线 | 3/3 | 16/20 |
| `xmove_bend_pick` reference PPO `model_20` | 3/3 | 16/20 |
| `xmove_bend_pick` reference PPO `model_99` | 3/3 | 16/20 |
| `xmove_bend_pick` state-only PPO `model_99` | 0/3 | 未进入随机生产 |

为了直接判断成功究竟来自固定 reference 还是 PPO，另做了严格的100次配对实验：
`xmove_pick`固定 base/reference episode 82，`xmove_bend_pick`固定 episode 96；每对
rollout 的机器人 `qpos/qvel`、目标位置/四元数和 reference ID 都逐元素相同，只随机目标
x/y `±4/5 cm`、yaw `±0.25 rad`。BC checkpoint 的 correction 层严格为零，因此就是
“同一 reference only”基线。

| task | reference only | reference + PPO | PPO-only | reference-only | 双侧精确配对检验 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `xmove_pick` | 59/100 | 59/100 | 0 | 0 | `p=1.0` |
| `xmove_bend_pick` | 77/100 | 79/100 | 2 | 0 | `p=0.5` |

`xmove_pick`的100个成败逐个完全一致；`xmove_bend_pick`的PPO只额外救回repeat 18和67，
没有丢失reference成功。物理36D command相对reference的平均绝对变化分别只有
`8.78e-5`和`5.15e-5`，最大变化为`1.34e-3`和`1.87e-3`。所以当前证据只能说明bend
存在微弱的`+2%`信号，统计上不能确认PPO有稳定增益；xmove则没有任何可测增益。完整
summary位于各checkpoint下的`eval_fixed_reference*_random100_large/summary.json`。

因此这里可以确认“PPO policy会成功”，但不能声称PPO已经超过计划基线；大部分成功应
归因于reference plan与Sonic tracker的容差。无reference的
MLP PPO以及额外GRU-BC都在完整起点为0，说明仅靠当前331D逐帧状态监督还不能稳定恢复
长时序；这条负消融被保留，没有冒充成功policy。

两个 reference PPO 在显式联合OOD初态上均为4/4：`xmove_pick`使用机器人
`(-0.78,0)`、物体`(-0.28,-0.06)`，`xmove_bend_pick`使用机器人
`(-0.95,0.02)`、物体`(-0.275,-0.025)`。用于实际数据生产时，每个随机目标从完全相同
MuJoCo初态依次尝试base计划与rank 0--4，所有rollout仍由PPO checkpoint闭环输出：

| task | 成功轨迹 | 随机目标成功率 | 完整plan rollout |
| :--- | ---: | ---: | ---: |
| `xmove_pick` reference PPO `model_10` | 20 | 20/24（83.33%） | 50 |
| `xmove_bend_pick` reference PPO `model_20` | 20 | 20/20（100%） | 30 |

成功数据分别位于 `outputs/grasp_rl/xmove_pick/ppo_reference_production20_v1` 和
`outputs/grasp_rl/xmove_bend_pick/ppo_reference_production20_v1`。每条NPZ均已检查为有限的
331D state、842D policy input、36D normalized/physical command和80D motion frame。
PPO闭环视频位于：

```text
data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2/videos/ppo_reference_model10_random_episode000002_success.mp4
data/grasp_rl/G1WholebodyXMovePickTeleop-v0/v2/videos/ppo_reference_model10_production_episode000000_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/ppo_reference_model20_random_episode000007_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/ppo_reference_model20_production_episode000000_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/paired_fixed_reference96_repeat67_ppo_success.mp4
data/grasp_rl/G1WholebodyXMoveBendPickTeleop-v0/v2/videos/paired_fixed_reference96_repeat67_reference_failure.mp4
```

视频均为H.264/yuv420p、640x360、50 fps，metadata记录 `closed_loop=true`、实际
`success` 和对应checkpoint路径。这里的成功仍应解释为“检索完整计划 + PPO状态
反馈 + Sonic tracker”的有界泛化，不是无条件SMP，也不是任意目标的通用运动规划。
最后两段是repeat 67的严格同初态配对：PPO抬升14.78 cm并成功，零correction的reference
只抬升0.64 cm且失败。渲染器会恢复轨迹保存的完整v2 `initial_qpos/qvel`，而非只恢复物体
pose，因此该视频与评估summary逐值复现。

`xmove_pick` 还完成了一轮更激进的50轮PPO续训：task权重由0.05增至0.1、actor
学习率放大5倍、reference权重降至0.0002，并以 `model_99` 做teacher anchor。最终
`model_148`仍保持固定test 8/8，但统一随机20场中 `model_110`为15/20，
`model_140/148`均为14/20，没有超过原 `model_99` 的15/20，因此不替换。另一个不缩小
扰动、同seed的rank 0--10生产消融只有18/22（81.82%，72次完整plan rollout），也未达到
20条目标，所以保留在 `ppo_reference_model99_production20_rank10_v1` 作为失败证据。

### GRAIL 风格 task reward（旧版 grasp v1）

当前默认 `grail_release_v1` 在50 Hz下计算：

```text
target = pregrasp_distance_kernel
       + 5  × grasp_contact_fraction
       + 10 × bilateral_finger_direction
       + 5  × contact_gated_lift
       + 2  × contact_gated_stability

penalty = 15 × distance_gated_wrist_speed²
        + clamp(hand_table_force, 0, 1)
        + 0.1 × normalized_action_rate

step_reward = 0.02 × (target - penalty)
            + 0.0002 × robot_reference_reward
            + terminal_adjustment
```

`bend_pick` 的成功条件是物体相对初始高度提升至少9 cm，同时保持有效thumb-support双侧
抓持连续13个control steps。目标pose、速度、指尖表面距离、接触力和抬升高度全部来自
MuJoCo。100场景reward audit中expert为95/100，no-motion、time-shuffle和throw均为0/100。

当前正式实验使用 `task_only`，没有启用 SMP diffusion reward。仓库仍保留 frozen
unconditional SMP prior的additive/product消融接口，但它不会根据目标生成抓取计划，已有
消融也没有优于task-only。

### 训练、评测与结果

第一个扩展任务为 `G1WholebodyBendPickMP-v0`：

- 完整轨迹BC初始化：native 100场景为97/100；
- PPO：14环境 × 96步 × 300轮，共403,200 transitions；
- 固定100随机目标：rank-0为90/100，rank-1单独为87/100；同目标依次尝试rank-0和rank-1
  为91/100；
- 独立生产seed：106个目标成功100个，真实成功率94.34%；共保存100条成功完整轨迹，
  manifest同时保留6个失败和全部118次plan rollout；
- 发布checkpoint为 `outputs/grasp_rl/bend_pick/ppo_native_v1_300/model_100.pt`。训练至
  model250仍为90/100且成功集合完全相同，因此没有因iteration更大而替换发布模型。

主要产物：

```text
data/grasp_rl/G1WholebodyBendPickMP-v0/
outputs/grasp_rl/bend_pick/reward_audit_v3/reward_audit.json
outputs/grasp_rl/bend_pick/bc_plan_v1/best.pt
outputs/grasp_rl/bend_pick/ppo_native_v1_300/model_100.pt
outputs/grasp_rl/bend_pick/policy_dataset_random_100_v1/
outputs/grasp_rl/bend_pick/videos/
```

完整设计、实验消融和验收记录见
[`FULL_TRAJECTORY_GRASP_RL_PLAN.md`](FULL_TRAJECTORY_GRASP_RL_PLAN.md)，代码接口说明见
[`src/simple/grasp_rl/README.md`](src/simple/grasp_rl/README.md)。

## Table of Contents
- [SIMPLE_RL：完整轨迹抓取新方案](#simple_rl完整轨迹抓取新方案)
- [What is SIMPLE?](#what～is～SIMPLE)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [[Option 1] UV setup (Quickest)](#option-1-uv-setup-quickest)
  - [[Option 2] Nix setup](#option-2-nix-setup)
  - [[Option 3] Docker setup](#option-3-docker-setup)
- [Data Generation & Pipeline](#-data-generation--pipeline)
  - [1. Data Collection ](#1-data-collection-methods)
  - [2. Data Post-processing](#2-post-processing)
  - [3. Fine-Tuning](#3-fine-tuning)
- [Evaluation in SIMPLE](#-evaluation-in-simple)
- [📊 Simulation Benchmarking Results](#-simulation-benchmarking-results)
- [Citation](#citation)
- [License](#license)

## What is SIMPLE?

SIMPLE stands for SIMulation-based Policy Learning and Evaluation.

It is a `simple` simulation environment supports:
  + multiple agents: (franka arm/aloha bimanual arms/dexmate wheeled robot and unitree g1 humanoid!)
  + 1000+ Objaverse assets
  + 50+ Habitat HSSD scenes
  + 50+ humanoid wholebody loco-manipulation tasks

## System Requirements

SIMPLE is built on top of `IsaacSim 4.5` and `MuJoCo 3.3`.

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 |
| **CPU** | Intel Core i7 / AMD Ryzen 7 | Intel Core i9 / AMD Ryzen 9 |
| **RAM** | 32 GB | 64 GB |
| **GPU** | NVIDIA RTX 2070 (8 GB VRAM) | NVIDIA RTX 3080 Ti / 4090 (16+ GB VRAM) |
| **NVIDIA Driver** | 535.x | Latest |
| **CUDA** | 12.x | 12.x |
| **Python** | 3.10 | 3.10 |
| **Storage** | 50 GB SSD | 100+ GB NVMe SSD |

> An RTX-class NVIDIA GPU is required. GTX and older architectures are not supported.


## Installation

Clone the project:


```

git clone git@github.com:physical-superintelligence-lab/SIMPLE.git

```

Change directory to the project root:


```

cd SIMPLE

```

Pull all submodules

```

git submodule update --init --recursive

```

We offer three options for setting up SIMPLE:

## [Option 1] UV setup (Quickest)

Install `uv` if not already done
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

Install all dependencies at once

```
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match

```

Install CuRobo

```
bash scripts/install_curobo.sh

```

Activate the environment:

```
source .venv/bin/activate

```

Verify the installation by printing the version number

```
python -c "import simple; print(simple.__version__)"

```

[Optional] Build the docs.

```
make live

```

Open http://127.0.0.1:8005 in a browser to view the documentation.

> The document are working in progress. Feel free to raise questions using github issue, we will try to complete the document construction as soon as possible.

## [Option 2] Nix setup

We recommend using [nix](https://nixos.org/) on fresh new linux host, otherwise, if you alread have install NVIDIA driver and CUDA, it will be faster to setup SIMPLE through `uv`.

> [Nix](https://nixos.org/) is a modern package manager and build system that focuses on reproducibility, isolation, and declarative system configuration.

> Instead of installing software directly into your system (like apt or pip), Nix builds everything in isolated environments and stores them in the /nix/store, where each package version is uniquely identified by a hash.

1. Install Nix first, for all interactive questions, enter `y`:

```bash
sh <(curl --proto '=https' --tlsv1.2 -L [https://nixos.org/nix/install](https://nixos.org/nix/install)) --daemon

```

2. After Nix installation, open up a new shell to proceed.

If you encounter issues with `nix` command not found, try

```bash
export PATH=/nix/var/nix/profiles/default/bin:$PATH


```

3. Pull git modules recursively

```bash
git submodule update --init --recursive

```

Run the prerequisite check once on a new host:

```bash
./scripts/nix/prereq-check.sh

```

`nix develop` auto-booststraps dependencies on first entry (or when `uv.lock` / `pyproject.toml` changes).

Start the dev shell:

```bash
nix --extra-experimental-features "nix-command flakes" develop

```

Or run a single command inside the dev shell:

```bash
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop -c <command>

```

Do not activate the virtual environment directly with `source .venv/bin/activate` or `source .venv-nix/bin/activate`.
This repo expects the Nix shell and the Python environment to be used together. The virtual environment alone is not a supported runtime.
If your IDE terminal auto-sources `.venv-nix/bin/activate`, disable that behavior for this workspace or `deactivate` before entering through `nix develop`.

Check if install successfully.

```
python -c "import simple; print(simple.__version__)"

```

You should see version number printed.

* If encouter installtion or running issues, please checkout `Troubleshootings` in the Docs


### Nix Notes

The Nix runtime is documented in detail in [`docs/source/nix-runtime.md`](https://www.google.com/search?q=./docs/source/nix-runtime.md).

Short version:

* Mutually exclusive with Docker.
* Intended host baseline: Linux with NVIDIA drivers already installed, especially Ubuntu hosts.
* Run `./scripts/nix/prereq-check.sh` first on a new host.
* Nix owns userspace; the host only owns the NVIDIA driver boundary.
* The shell fails early on runtime pollution from `LD_LIBRARY_PATH`, `PYTHONPATH`, `PYTHONHOME`, or `LD_PRELOAD`.
* The default Python environment is `.venv-nix`.
* Bootstrap entry points are `./scripts/nix/bootstrap-python.sh`, `./scripts/nix/bootstrap-gpu.sh`, and `./scripts/nix/bootstrap.sh`.
* Prefer importing `simple` as a library from inside the dev shell; treat the CLI as a thin convenience layer.

Operational notes:

* Remove a root-owned `.venv` left by older Docker runs with `sudo rm -rf .venv`.
* Use `SIMPLE_AUTO_BOOTSTRAP=0` to skip auto-setup, or `SIMPLE_FORCE_BOOTSTRAP=1` to force re-bootstrap.
* If you need to run `nix` from inside the dev shell, prefer `env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" ...`.

### [Option 3] Docker setup

We also support building and running SIMPLE in docker. Please refer to the documents for [docker setup](https://www.google.com/search?q=docs/source/tutorials/docker.md).

---

## ⚙️ Data Generation & Pipeline

SIMPLE provides a scalable pipeline to generate, process, and train policies using synthesized simulation data.

### 1. Data Collection 

We support two primary interfaces for gathering  data: **Teleoperation (human-in-the-loop)** and **Automated Motion Planning**. 

Before running, adjust your environment variables to match your system topology.
```bash
# Example configurations (Adjust CUDA_VISIBLE_DEVICES and DISPLAY based on your host)
export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES="0" 
export DISPLAY=":1"
```




##### Stage 1: Teleoperation in MuJoCo

We perform the initial human-in-the-loop teleoperation inside the lightweight MuJoCo engine. This ensures minimal control loop latency and high-frequency physical interactions during the demonstration tracking.

**Example Usage:**

```bash
export TASK_NAME=G1WholebodyOpenTrashCanTeleop-v0

python -m simple.cli.teleop_decoupled_wbc \
  simple/$TASK_NAME \
  --target=graspnet1b:0 \
  --sim-mode=mujoco \
  --record \
  --no-headless \
  --success-criteria=2


```
> 🥽 **Hardware Setup:** We utilize **Pico VR headsets** for immersive human-in-the-loop teleoperation. For specific hardware configuration, controller mapping, and connection details, please refer to the [Teleoperation Setup Guide](docs/source/tutorials/teleop.md).


> 💡 *To explore additional customizable options for teleoperation, run:*
> `python -m simple.cli.teleop_decoupled_wbc --help`

Supported Wholebody Teleop Tasks Include:

* `simple/G1WholebodyOpenTrashCanTeleop-v0`
* `simple/G1WholebodyBendPickTeleop-v0`
* `simple/G1WholebodyBendPickAndPlaceTeleop-v0`
* `simple/G1WholebodyBendHandoverTeleop-v0`
* `simple/G1WholebodyPushOfficeChairTeleop-v0`
* `simple/G1WholebodyOpenFaucetTeleop-v0`
* `simple/G1WholebodyOpenOvenTeleop-v0`
* `simple/G1WholebodyCloseDoorTeleop-v0`
* `simple/G1WholebodyXMovePickTeleop-v0`
* `simple/G1WholebodyXMoveBendPickTeleop-v0`
* `simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`
* `simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0`
* `simple/G1WholebodyHandoverTeleop-v0`


##### Stage 2: Photorealistic Replay & Isaac Sim Rendering

Once raw trajectories are successfully captured, pass them into the `replay_decoupled_wbc` suite. By specifying `--sim-mode=mujoco_isaac`, this stage replays the actions in MuJoCo while driving **Isaac Sim** simultaneously as a synchronized rendering engine. This step processes the raw stream into standard dataset structures (LeRobot format).

**Example Usage:**

```bash
# Ensure $TASK_NAME matches the task used in Stage 1
python -m simple.cli.replay_decoupled_wbc \
  simple/$TASK_NAME \
  --data-dir=data/teleop_decoupled_wbc/simple/$TASK_NAME/level-0/ \
  --sim-mode=mujoco_isaac \
  --no-headless \
  --render-hz=50 \
  --save-dir=data/replay_decoupled_wbc_output \
  --record \
  --resume \
  --success-criteria=0.2

```

> 💡 **Tip:** If the replay success rate is low, try lowering the `--success-criteria` first.



#### B. Automated Motion Planning 

To bypass manual human interaction and scale up synthetic data generation, the `simple.cli.datagen` pipeline directly integrates **CuRobo for automated motion planning**. This allows us to procedurally batch-produce optimal demonstration trajectories without human teleop.

Unlike the two-stage teleoperation process, **Motion Planning can be executed in a single step**. By setting `--sim-mode=mujoco_isaac`, the pipeline resolves the fast contact physics and motion planning within MuJoCo, while simultaneously driving Isaac Sim for photorealistic rendering. This directly outputs the final dataset in the standard LeRobot format.

**Example Usage:**

```bash
export TASK_NAME=G1WholebodyTabletopHandoverMP-v0

python -m simple.cli.datagen \
  simple/$TASK_NAME \
  --sim-mode=mujoco_isaac \
  --render-hz=50 \
  --no-headless \
  --num-episodes=10

```





### 2. Post-processing

To prepare the generated datasets for policy learning, we need to post-process the raw output data to be strictly compatible with the training pipeline of our foundation model, [Psi-0](https://github.com/physical-superintelligence-lab/Psi0).

We provide two distinct post-processing scripts depending on how the data was collected:

#### A. Post-processing Motion Planning Data
For data generated via the automated motion planning pipeline (`datagen.py`), use `postprocess_psi0.py`. **This script supports wildcard matching (`*`)** to seamlessly merge data from multiple parallel generation batches into a single unified dataset.

**Example Usage:**
```bash
python scripts/postprocess_psi0.py \
  --sim-root="data/datagen*/simple/G1WholebodyXMoveBendPickMP-v0/level-0/" \
  --out-dir=data/processed_psi0/G1WholebodyXMoveBendPickMP-v0 \
  --skip=60

```

#### B. Post-processing Teleoperation Data

For data captured through human teleoperation and rendered via Isaac Sim , use `postprocess_psi0_sonic.py`. Similarly, this script utilizes wildcard matching (`*`) to merge data from multiple teleop replay sessions.

**Example Usage:**

```bash
python scripts/postprocess_psi0_sonic.py \
  --sim-root="data/replay_decoupled_wbc_output*/simple/G1WholebodyPushOfficeChairTeleop-v0/level-0/" \
  --out-dir=data/processed_psi0/G1WholebodyPushOfficeChairTeleop-v0 \
  --skip=0 \
  --total_episodes=100

```

**Key Arguments:**

* `--sim-root`: The input directory containing the generated dataset. Note that quotes `""` are highly recommended when using wildcards (`*`) to prevent premature shell expansion.
* `--out-dir`: The output directory where the Psi-0 compatible dataset will be saved.
* `--skip`: Number of initial frames to skip (useful for bypassing static setup or initialization frames).
* `--total_episodes`: Limits the total number of valid episodes to process and merge.



### 3. Fine-Tuning

To train or fine-tune foundation models directly using the structured datasets generated from the pipeline, we provide seamless integration with the **Psi-0** training stack.

> 👉 **Quick Start:** You can skip fine-tuning entirely and evaluate right away by downloading our pre-trained [checkpoints for SIMPLE](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/simple-checkpoints).

**Data Preparation:**
If you wish to train from scratch or fine-tune, download the required [SIMPLE task data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple) and extract it to your local workspace:

```bash
export TASK_NAME=G1WholebodyXMovePickTeleop-v0

hf download USC-PSI-Lab/psi-data \
  simple/$TASK_NAME.zip \
  --local-dir=data \
  --repo-type=dataset

unzip data/simple/$TASK_NAME.zip -d data/simple

```

**Training Integration:**

> 💡 **For full training instructions, please refer to the [Psi-0 Project README](https://github.com/physical-superintelligence-lab/Psi0).** >
> The Psi-0 repository contains comprehensive, up-to-date documentation on setting up training environment variables, visualizing episodes, and launching the training scripts (e.g., `bash scripts/train/psi0/finetune-simple-psi0.sh`).



## 🎯 Evaluation in SIMPLE

To rigorously evaluate the robustness and generalization of learned policies, we benchmark our foundation model [Psi-0](https://github.com/physical-superintelligence-lab/Psi0) using a decoupled **Client-Server architecture**. The server hosts the model inference, while the SIMPLE client runs the simulation environment.

---

### 🖥️ Server Side: Model Inference (Executed in the Psi-0 Repository)

#### Step 1: Environment & Checkpoint Setup
Configure the evaluation environment variables and paths within your **Psi-0** project workspace.

1. **Configure Environment Variables:** Inside the **Psi-0** project root, create and source your `.env` file based on the sample:
```bash
  cp .env.sample .env
  # Edit .env to include your HF_TOKEN, WANDB variables, and PSI_HOME path
  source .env
  echo $PSI_HOME # Verify the path is correctly set
```

2. **Download Pre-trained Weights:** Pull the Psi-0 checkpoints for the SIMPLE benchmark from our Hugging Face repository. Psi0's pre-trained weights for the SIMPLE benchmark are hosted on the Hugging Face Model Hub at [USC-PSI-Lab/psi-model](https://huggingface.co/USC-PSI-Lab/psi-model).

```bash
hf download USC-PSI-Lab/psi-model \
  --include="psi0/simple-checkpoints/*" \
  --local-dir=$PSI_HOME/.runs \
  --repo-type=model

```

### Step 2: Start the Psi-0 Inference Server

Before launching the simulation, initialize the model inference server.

```bash
# Set your target run directory and checkpoint step
export RUN_DIR=xxxx
export CKPT_STEP=40000

# Start the server (Listens on port 22085 by default)
bash scripts/deploy/serve_psi0_simple.sh $RUN_DIR $CKPT_STEP

```

> ⚠️ **Important:** Keep this terminal window open. The server must remain active for the duration of the evaluation.

### Step 3: Run the SIMPLE Simulation Client

Open a **new terminal window** to launch the environment. The execution parameters differ slightly based on the data source of the task:

* **For Teleop Tasks (suffix `*Teleop-v0`):** Use decoupled Whole-Body Control.
* `export entry=eval_decoupled_wbc`
* `export agent=psi0_decoupled_wbc`


* **For Motion Planning Tasks (suffix `*MP-v0`):** Use standard evaluation.
* `export entry=eval`
* `export agent=psi0`



**Execution Example (Teleop Task):**


#### Option A: UV Environment

```bash
export task=G1WholebodyXMovePickTeleop-v0
export agent=psi0_decoupled_wbc
export dr=level-0

TASK_NAME=$task uv run eval-decoupled-wbc \
    simple/$task \
    $agent \
    train \
    --data-format lerobot \
    --data-dir data/evals/simple-eval/$task/$dr \
    --host 127.0.0.1 \
    --port 21000 \
    --headless
```

#### Option B: Nix Environment

```bash
export task=G1WholebodyXMovePickTeleop-v0
export entry=eval_decoupled_wbc
export agent=psi0_decoupled_wbc
export dr=level-0

env -u LD_LIBRARY_PATH nix --extra-experimental-features 'nix-command flakes' develop -c \
  python -m simple.cli.$entry \
  simple/$task \
  $agent \
  train \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/$task/$dr \
  --host 127.0.0.1 \
  --port 21000 \
  --headless
```

### Step 4: View Evaluation Results & Videos

**Task Success Rate Statistics:**
Upon completion, the terminal will display a summary of the results. A detailed log is also preserved automatically:

```bash
cat data/evals_decoupled_wbc/eval_stats.txt

```

**Execution Videos:**
Visual records of each episode are automatically rendered and saved. The files are named using the pattern `episode_id/cam_name_{success_flag}.mp4` (e.g., `success` or `failed`).

```bash
# Example: Play a successful teleop evaluation video
mpv data/evals_decoupled_wbc/psi0_decoupled_wbc/G1WholebodyXMovePickTeleop-v0/level-0/episode_0/head_stereo_left_success.mp4

# Example: Play a successful motion planning evaluation video
mpv data/evals/psi0/G1WholebodyBendPickMP-v0/level-0/episode_0/front_stereo_left_success.mp4

```




## 📊 Simulation Benchmarking Results

> This is a preliminary benchmark with 6 tasks accompanying the [Psi-0](https://github.com/physical-superintelligence-lab/Psi0) project. Please also checkout Psi-0 for more details of intergrating Psi-0 with SIMPLE.

To rigorously evaluate the robustness and generalization of the learned policies, we design three evaluation levels with progressive out-of-distribution variations applied to the training environment:

> The evaluation environments are provided in the huggingface repository [USC-PSI-Lab/psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval).

* **Level 0 (Visual & Distractors):** Randomizes table materials and the types/initial positions of distractor objects.
* **Level 1 (Lighting):** Includes Level 0 variations + extreme changes in lighting conditions.
* **Level 2 (Spatial pose):** Includes Level 1 variations + perturbations to the initial positions of the target objects.

_Success rates are reported out of 10 evaluation trials per level (**Level 0 | Level 1 | Level 2**)._
| Baseline / Task | G1Wholebody<br>XMove<br>PickTeleop-v0 | G1Wholebody<br>BendPickMP-v0 | G1Wholebody<br>Handover<br>Teleop-v0 | G1Wholebody<br>Locomotion<br>PickBetweenTables<br>Teleop-v0 | G1Wholebody<br>Tabletop<br>GraspMP-v0 | G1Wholebody<br>XMove<br>BendPick<br>Teleop-v0 |
| :--------------- | :-----------------------------------: | :--------------------------: | :----------------------------------: | :---------------------------------------------------------: | :-----------------------------------: | :-------------------------------------------: |
| **Psi0** | 10 &#124; 10 &#124; 6 | 10 &#124; 10 &#124; 10 | 7 &#124; 7 &#124; 10 | 7 &#124; 5 &#124; 6 | 10 &#124; 10 &#124; 8 | 10 &#124; 9 &#124; 9 |
| **GR00T N1.6** | 10 &#124; 10 &#124; 7 | 7 &#124; 7 &#124; 6 | 1 &#124; 3 &#124; 3 | 0 &#124; 0 &#124; 0 | 9 &#124; 9 &#124; 7 | 4 &#124; 4 &#124; 1 |
| **OpenPi π0.5** | 7 &#124; 5 &#124; 1 | 10 &#124; 10 &#124; 8 | 5 &#124; 4 &#124; 5 | 3 &#124; 3 &#124; 3 | 10 &#124; 10 &#124; 8 | 0 &#124; 0 &#124; 0 |
| **InternVLA-M1** | 0 &#124; 0 &#124; 0 | 5 &#124; 5 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 3 &#124; 5 &#124; 7 |
| **H-RDT** | 0 &#124; 0 &#124; 2 | 0 &#124; 0 &#124; 1 | 0 &#124; 1 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 | 0 &#124; 0 &#124; 0 |
| **DreamZero** | 10 &#124; 10 &#124; 10 | 9 &#124; 9 &#124; 8 | 7 &#124; 8 &#124; 9 | 5 &#124; 3 &#124; 3 | 9 &#124; 10 &#124; 7 | 0 &#124; 0 &#124; 1 |
| **EgoVLA** | 0 &#124; 1 &#124; 2 | 7 &#124; 5 &#124; 8 | 0 &#124; 4 &#124; 3 | 0 &#124; 0 &#124; 0 | 10 &#124; 10 &#124; 7 | 3 &#124; 5 &#124; 4 |
| **Diff. Policy** | 3 &#124; 3 &#124; 2 | 10 &#124; 8 &#124; 6 | 3 &#124; 2 &#124; 4 | 4 &#124; 0 &#124; 0 | 8 &#124; 9 &#124; 8 | 0 &#124; 0 &#124; 0 |
| **ACT** | 10 &#124; 9 &#124; 6 | 10 &#124; 9 &#124; 9 | 4 &#124; 4 &#124; 6 | 6 &#124; 5 &#124; 7 | 10 &#124; 10 &#124; 8 | 6 &#124; 8 &#124; 8 |

_More interesting tasks, including articulated objects._

| Baseline / Task | G1Wholebody<br>CloseDoor<br>Teleop-v0 | G1Wholebody<br>OpenOven<br>Teleop-v0 | G1Wholebody<br>OpenFaucet<br>Teleop-v0 | G1Wholebody<br>PickAndPlace<br>AndHugContainer<br>Teleop-v0 | 
| :--------------- | :-----------------------------------: | :--------------------------: | :----------------------------------: | :---------------------------------------------------------: | 
| **Psi0** | 10 &#124; 10 &#124; 10 | 7 &#124; 5 &#124; 4 | 3 &#124; 3 &#124; 4 | 7 &#124; 6 &#124; 3 | 

## Citation

> Please also consider citing `Psi-0` if you use its training code.

```
@article{wei2026simple,
  title={SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation},
  author={Wei, Songlin and Ni, Zhenhao and Liu, Jie and Zhao, Zhenyu and Ye, Junjie and Jing, Hongyi and Xia, Junkai and Liu, Xiawei and Leong, Michael and Heng, Liang and Huang, Di and Wang, Yue},
  journal={arXiv preprint arXiv:2606.08278},
  year={2026}
}
```

```
@article{wei2026psi0,
  title={{$\Psi_0$}: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation},
  author={Wei, Songlin and Jing, Hongyi and Li, Boqian and Zhao, Zhenyu and Mao, Jiageng and Ni, Zhenhao and He, Sicheng and Liu, Jie and Liu, Xiawei and Kang, Kaidi and others},
  journal={arXiv preprint arXiv:2603.12263},
  year={2026}
}
```

## License

This project is licensed under the MIT.

See the [LICENSE](https://www.google.com/search?q=license.md) file for details.

# SIMPLE_RL
