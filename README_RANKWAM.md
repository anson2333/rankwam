# RankWAM

RankWAM 是基于 FastWAM 的研究分支，目标是验证：利用同一状态下多个动作候选的
仿真 continuation 信号，能否学习出更可靠的动作排序，而不是只提高未来视频重建质量。

本仓库只同步源码、配置、测试和实验协议。模型权重、数据集、虚拟环境、缓存、日志、
视频和评测输出均不进入 Git，应通过平台挂载或单独的数据存储提供。

## 当前研究状态

### 已完成

- 固定 FastWAM 上游基线：commit `45d8e1458921d83f8ad6cf9ce993d371208dabd0`。
- 固定 LIBERO-plus 代码版本：commit `4976dc30028e805ff8094b55501d532c48fec182`。
- 4051 环境已完成 bootstrap：Python 3.10.12、PyTorch 2.7.1+cu128、MuJoCo 3.3.2、robosuite 1.4.0。
- G0 端到端评测链路已打通：模型加载、LIBERO 环境、动作推理、结果 JSON 和 MP4 均可生成。
- G1 隔离重放已验证：MuJoCo state、终态和 proprio 可重复；跨 EGL context 的 RGB 差异被单独审计，
  不用于标签判断。
- 4051 GPU0 + OSMesa 的正式参数 pilot 已完成 5 trials，结果为 `0/5`。这只是功能和环境基线，
  不能当作成功率结论；4051 EGL 多 trial 曾触发 NVIDIA Xid，默认采用逐 trial 故障隔离。
- 候选收集和 continuation 标签协议已实现，支持断点续跑、原子写入、候选组记录和汇总。
- 当前 discovery 数据共 9 个完整 group / 72 个 candidate，覆盖 5 个独立 trial；已有 mixed-success
  group，但样本量仍不足以训练或报告 learned critic 增益。

### 当前判断

目前还不能声称 RankWAM 提升了 LIBERO 或 RoboTwin 成功率。首要瓶颈是候选组信息量和可重复的
仿真标签，而不是先扩大模型或直接做在线 RL。只有通过候选 headroom、未来信息增益和 held-out
排序指标后，才进入大模型联合训练。

## 实验门槛

1. **G0：基线**
   - 正确的 LIBERO 环境和 FastWAM checkpoint 能稳定运行。
   - 先做 pilot，再扩展到 50 trials；每次运行保存 resolved config、环境信息、版本和哈希。

2. **G1：确定性隔离重放**
   - 新建环境、重置到记录的初始状态、完整重放 action prefix，再执行 candidate suffix。
   - physics terminal state、success 和 proprio 必须一致；RGB 只做独立渲染审计。

3. **G2：候选 headroom**
   - 每组使用 `K=8` 个候选，要求 informative-group rate 至少 20%。
   - simulator-oracle 相对随机选择的 uplift 至少 10 个百分点。
   - 未达到门槛时先改候选生成和状态采样，不训练 ranker。

4. **G3：离线 ranker**
   - held-out pairwise accuracy 超过 70%，top-1 regret 显著低于随机选择，且三个训练 seed 方向一致。

详细协议见 [`docs/experiment_protocol.md`](docs/experiment_protocol.md) 和
[`docs/rankwam_validation_plan.md`](docs/rankwam_validation_plan.md)。

## 目录说明

```text
src/fastwam/                 FastWAM/Wan22 模型、数据和训练核心
experiments/libero/           LIBERO 基线评测、候选收集、重放和 continuation 工具
configs/ranking/              RankWAM 实验配置
scripts/                     训练、环境报告、G0 分片运行和汇总入口
docs/                        环境 bootstrap、实验协议和验证计划
third_party/                 仅保留源码或小型补丁；仿真资产通过外部挂载提供
```

## 4051 环境与 G0 示例

```bash
cd /mnt/workspace/mishuo/rankwam
source .env.local

# 单个 smoke trial
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  model.redirect_common_files=false \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_object \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.output_dir=./outputs/rankwam/g0_smoke \
  gpu_id=0

# 逐 trial 隔离运行，支持失败重试和断点续跑
GPU_ID=0 START_TRIAL=0 TOTAL_TRIALS=5 MAX_ATTEMPTS=3 \
  OUTPUT_ROOT=./outputs/rankwam/g0_sharded_5trials \
  scripts/run_g0_sharded.sh
```

4051 的 EGL 图形上下文曾出现 Xid 31/109。需要隔离渲染时可设置：

```bash
export RANKWAM_MUJOCO_GL=osmesa
```

这会让 MuJoCo offscreen rendering 使用 CPU，FastWAM 推理仍使用 CUDA；它是稳定性隔离方案，
不是 GPU 驱动问题的完整修复。

## 数据和权重边界

以下路径必须由外部挂载或单独同步，不上传到 Git：

```text
checkpoints/       Wan/FastWAM 权重和 dataset stats
data/              LIBERO、RoboTwin 和 LeRobot 数据
outputs/           评测 JSON、MP4、缓存和 continuation 标签
runs/              训练 checkpoint、日志和 wandb 文件
.venv/             Python 虚拟环境
.env.local         本机路径和凭据
third_party/LIBERO-plus  仿真代码/资产软链接
```

`.env.example` 只包含路径模板，不包含凭据。迁移到新实例时先阅读
[`docs/bootstrap_4051.md`](docs/bootstrap_4051.md)，再按实际挂载点修改 `.env.local`。

## 后续计划

1. 完成至少 20 个 informative candidate groups，并报告候选多样性、成功率和 oracle uplift。
2. 用 episode/state clustered bootstrap 建立 offline 排序基线，避免把同一 episode 的候选当成独立样本。
3. 完成 `V_state_action` 与 `V_future_oracle` 的跨 episode 对比；只有未来信息有明确增益才实现 learned future model。
4. 通过 E3 后再做 LoRA/adapter 级联合训练，并与无 future 的 RL、只做 future supervision 的版本进行消融。
5. 最终在固定的 unseen initial states 上做至少 50 trials，报告 Wilson 95% CI、GPU-hours、reset 数和 wall time。

## 说明

本项目继承 FastWAM 的模型和评测基础设施；原始 FastWAM 文档和上游链接仍保留在
[`README.md`](README.md) 与 [`README_zh.md`](README_zh.md) 中。

