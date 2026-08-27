# RankWAM v2: Action-Conditioned Future Supervision for WAM-RL

## 0. 研究主张

RankWAM 不再把一个独立 action ranker 当作最终方法。最终要验证的是：

> 对 WAM 同时施加动作监督、动作条件未来监督和 continuation-aware RL 信号，
> 是否能让模型生成更可能成功的动作，而不是只提高未来视频的重建质量。

方法由三个可分离模块组成：

```text
pi_theta(a | s, instruction)                 action policy
F_psi(z_future | s, instruction, a)          action-conditioned future model
V_omega(s, a, F_psi(s,a))                    continuation critic
```

`s` 包含双相机当前观测和 proprio，`a` 是 32 步候选动作块。`V` 预测的不是
“这个画面看起来像不像成功”，而是从候选执行后的状态继续运行同一个冻结策略
固定预算后得到的结果分布。

最终联合目标：

```text
L = lambda_action * L_action
  + lambda_future * L_future
  + lambda_value * L_value
  + lambda_rank * L_group_rank
  + lambda_rl * L_policy
  + lambda_prior * L_prior
```

- `L_action`: FastWAM 原有 action flow-matching loss；
- `L_future`: 动作条件未来 latent flow-matching / reconstruction loss；
- `L_value`: success BCE + progress/AUC regression + grasp-stability auxiliary loss；
- `L_group_rank`: 同一状态内候选的 listwise 或 Bradley-Terry loss；
- `L_policy`: 用组内相对 advantage 更新动作生成分布；
- `L_prior`: 对冻结初始策略的 KL/flow matching anchor，防止小数据破坏先验。

这里的核心不是“把仿真视频逐帧判断一遍”。仿真 continuation 只在研究阶段生成
训练标签。部署时仿真器不可用；策略可以直接生成动作，critic 只在需要 best-of-K
推理时启用。

## 1. 梯度和信用分配

对同一状态采样 `K` 个动作块，并用冻结 continuation policy `pi_0` 运行到统一的
episode policy budget `T`：

```text
G_i = success_i + beta_p * progress_auc_i + beta_g * grasp_stability_i
A_i = (G_i - mean_group(G)) / (std_group(G) + eps)
```

稀疏成功是主目标。progress 和 grasp 只作为 task-local 辅助目标，不得改变
success 的优先级；所有 `beta` 只能在验证集上固定。

第一版不反传穿过 MuJoCo，也不声称端到端 differentiable planning。策略更新采用
FastWAM 动作扩散/flow 的可计算 surrogate：对高 advantage 样本降低 action
flow-matching loss，对低 advantage 样本降低权重或使用 preference/DPO-style loss。
若后续能够稳定得到 trajectory log-prob，才替换为 PPO/GRPO。

关键梯度消融：

| 版本 | future loss | value/rank loss | policy update | 用途 |
|---|---:|---:|---:|---|
| A | 否 | 否 | 否 | 原始 FastWAM |
| B | 是 | 否 | 否 | 只做未来监督 |
| C | 否 | 是 | 是 | 无 future 的 RL/critic |
| D | 是 | 是，但 stop-grad 到 F | 是 | future 仅作表征 |
| E | 是 | 是，rank 梯度进入 F | 是 | 完整 RankWAM |

只有 `E > C` 且 `E > D`，才能主张 action-conditioned future supervision 对 RL
有额外贡献。只证明一个 action MLP 能排序候选，不构成论文主贡献。

## 2. Continuation value 的定义

标签单位是 `(episode, state, candidate)`，不是视频帧，也不是候选 pair。

```text
candidate: 从分叉点执行固定 32 个控制步
continuation: 从分叉 step t 开始，用冻结 FastWAM pi_0 闭环运行 T - t - 32 步
primary label: 统一 episode budget T=200 内 success
auxiliary labels: max/final/AUC progress, grasp fraction/loss/recovery
```

固定 60 步 continuation 只保留为 short-horizon 消融，不进入主训练集。它会系统性
截断早期状态：例如 trial 5 从 step 40 分叉时，`40 + 32 + 60 = 132`，而原策略在
step 133 才成功。主标签使用固定绝对终点而不是固定剩余长度；critic 必须输入归一化
时间 `t/T` 或 remaining budget，AUC 也按实际剩余长度归一化。

固定 continuation policy 很重要。否则标签同时混合候选质量和不同后续策略质量，
`V(s,a)` 的语义会漂移。每轮 policy 更新后不能直接混用旧标签；需要记录
`behavior_policy_hash`，并采用以下之一：

1. 每轮只用当前/最近 behavior policy 的标签；
2. 对旧数据做 importance correction；
3. 把 behavior policy id 输入 critic。

pilot 使用方案 1。

## 3. 状态采样，不固定绝对 step

绝对 step 只用于重放索引，不代表语义阶段。每条 baseline trajectory 每 10 步保存
checkpoint，然后在看候选结果之前用如下确定规则选最多三个窗口：

1. `pre_grasp`: 首次 `grasped=True` 前 10 步；
2. `first_grasp`: 首次抓取所在 checkpoint；
3. `pre_completion`: 首次 success 前 10--20 步；若 baseline 未成功，则选 grasped
   状态中 progress 最大点之前 10 步。

相邻选择若落在同一 checkpoint，只保留一次。每个原始 episode 最多三个 group，
避免一个 episode 在统计上占主导。

当前已知例子解释了为什么要这样做：task 3 trial 1 的 step 90 为 3/8 success，
step 100 为 8/8 success；信息边界出现在抓取/运输阶段，而不是某个跨 episode 固定步数。

## 4. 数据收集和缓存预算

每个状态 `K=8`：5 个 FastWAM diffusion seeds + 3 个 normalized action perturbations。
每个 group 保存：

```text
branch RGB (2 cameras), proprio, state id
candidate action [8, 32, 7]
candidate terminal RGB/proprio
continuation checkpoint RGB/proprio every 10 steps
success and vector-valued continuation targets
checkpoint/model/stats hashes and all seeds
```

不保存 MP4，不保存逐控制步 RGB，不保存重复的模型权重。RGB 用 JPEG 或压缩 NPZ；
每条 baseline trajectory 上限 50 MB，每个 candidate group 上限 50 MB。收集器必须原子
写入并支持 resume。到达预算就失败停止，不能静默扩张缓存。

## 5. 分阶段可证伪实验

### E0: 基线与重放

- 正确 LIBERO 环境的 FastWAM baseline 至少有一个成功 episode；
- prefix replay 的 MuJoCo state/proprio 最大误差 `<=1e-7`；
- 相同 candidate + continuation 重复三次，success 和物理 state 一致。

E0 已通过。现有 task 3 trial 1 baseline 在 policy step 183 成功。

### E1: 候选可控性和标签密度

先收集 20 个 informative groups，要求每组同时包含成功和失败候选。报告单位必须是
state group：

```text
informative-group rate >= 20%
oracle uplift over random >= 10 percentage points
```

若 30 个预先选择的 boundary groups 中不足 6 个 informative，停止训练 rank/value
head，优先修改候选生成或 horizon。不得用查看 outcome 后挑状态来提高比例。

### E2: future information test

这是完整模型前必须通过的关键实验。使用真实候选短期未来作为信息上限，比较：

```text
V_state_action: current RGB + proprio + action
V_future_oracle: current inputs + real candidate-terminal/future RGB
V_state_only: current RGB + proprio
V_action_only: action
```

按 episode 分组做 leave-one-episode-out 或 train/val/test，禁止同一状态候选泄漏。
在 informative groups 上评估 pairwise accuracy、success@top1、top1 regret。

通过条件：`V_future_oracle` 的跨 episode pairwise accuracy 至少 75%，且比
`V_state_action` 高至少 5 个百分点。否则未来观测对这个任务/horizon 没有足够增量，
没有理由投入 action-conditioned video model。

### E3: learned future model

实现轻量 `F_psi`，先不改完整 5B trunk：

```text
current FastWAM/VAE latent + action tokens
  -> temporal adapter (4--8 transformer blocks)
  -> future latent tokens at 10/20/32 control steps
```

冻结 VAE、文本编码器和原 FastWAM trunk，只训练 action embedding、temporal adapter
和 value heads。用真实 rollout latent 监督：

```text
L_future = latent FM/MSE + multiscale feature loss
L_value = BCE(success) + Huber(AUC/final progress) + BCE(grasp events)
```

比较 `V_future_oracle`、`V_pred_future`、`V_state_action`。若 predicted-future critic
不能达到 oracle-future 增益的一半，说明 bottleneck 在未来模型，不进入大模型联合训练。

### E4: joint WAM-RL

在 E3 通过后，解冻 action expert 的 LoRA/adapters，使用 group advantage 做加权
action supervision：

```text
w_i = clip(exp(A_i / tau), w_min, w_max)
L_policy = mean_i w_i * L_action(s_i, a_i)
```

失败候选不是直接“反向模仿”；它们只通过组内归一化降低权重并训练 critic。保留
原始离线 demonstration batch 和 `L_prior`，RL batch 与 offline batch 比例从 1:3 开始。

离线 gate：held-out group 的 top1 success/regret。在线 gate：预先固定的 unseen initial
states，每个版本至少 50 trials，报告 Wilson 95% CI、GPU-hours、reset 数和 wall time。

### E5: 完整联合训练

最后才允许 rank/value 梯度进入 future adapter，并比较第 1 节 A--E 五个版本。
论文主结论必须同时满足：

1. 完整版真实 success 显著高于原始 FastWAM；
2. 完整版显著高于无 future 的 RL 版本 C；
3. future metric 与 policy gain 的关系不是仅由更多参数/算力解释；
4. 在至少两个 task suite 或两类操作任务上复现。

## 6. Critic 结构

第一版 critic 控制在 20M 参数以内：

```text
image encoder: frozen DINO/FastWAM VAE feature -> 256-D
proprio encoder: MLP(9 -> 64)
action encoder: temporal Conv1D/Transformer([32,7]) -> 256-D
future encoder: shared image/latent encoder + temporal pooling -> 256-D
fusion: 2-layer gated MLP -> 256-D
heads: success logit, AUC/final progress, grasp-loss/recovery logits
```

success head 不输出跨任务“绝对成功率”；它估计：

```text
P(success by T | task, t/T, s, candidate a, fixed continuation policy)
```

训练采用 group-balanced BCE + listwise ranking。每个 state group 在一个 batch 内完整出现，
避免 8 个几乎相同候选被当成 8 个独立状态。校准误差只作辅助指标，排序和真实 top1
outcome 才是主指标。

## 7. 当前证据和下一步

截至当前 pilot：

- task 3 trial 0 step 70: `0/8` success，非 informative；
- task 3 trial 1 step 100: `8/8` success，非 informative；
- task 3 trial 1 step 90: `3/8` success，oracle uplift `62.5pp`，informative；
- discovery 阶段 `B=60` 共 9 个完整 group / 72 个 candidate，覆盖 5 个独立 trial；
- discovery 数据有 3 个 mixed-success group，但只有 1 个 strong-contrast group；
- 新主协议 `T=200` 已完成 trial 5 step 30/40 两组，分别为 `2/8` 和 `6/8`；
- 两个 `T=200` group 都是 strong contrast，但旧协议数据不得混入主训练集；
- 仍不能训练或报告 learned critic gain。

接下来严格按顺序执行：

1. 自动采集新的 baseline trajectory，并在 outcome 前确定 boundary steps；
2. 单次加载 FastWAM，批量生成/执行 8 个候选，并 continuation 到 `T=200`；
3. 达到至少 20 informative groups 后运行 E2；
4. E2 通过后才实现 action-conditioned future adapter；
5. E3 通过后才做 policy update。

任何阶段失败都要保留负结果并停止扩大后续模型。这样可以明确区分候选数据问题、
critic 可学习性问题、future prediction 问题和 RL policy update 问题。
