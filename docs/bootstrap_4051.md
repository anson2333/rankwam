# RankWAM 4051 Bootstrap

项目根目录：`/mnt/workspace/mishuo/rankwam`

## 固定版本

- FastWAM: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- LIBERO-plus: `4976dc30028e805ff8094b55501d532c48fec182`
- Python: `3.10.12`
- PyTorch: `2.7.1+cu128`
- MuJoCo: `3.3.2`
- robosuite: `1.4.0`

## 环境重建

4051 的 PyPI 默认链路较慢。先从阿里云镜像安装 PyTorch 的小依赖，
再从 PyTorch index 安装已固定的 CUDA wheel：

```bash
cd /mnt/workspace/mishuo/rankwam
uv venv --python 3.10

UV_LINK_MODE=copy uv pip install \
  numpy==1.26.4 sympy==1.14.0 pillow==12.0.0 \
  --index-url https://mirrors.aliyun.com/pypi/simple/

UV_LINK_MODE=copy uv pip install \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

UV_LINK_MODE=copy uv pip install -e ".[libero]" \
  --index-url https://mirrors.aliyun.com/pypi/simple/
UV_LINK_MODE=copy uv pip install -e third_party/LIBERO-plus --no-deps
uv pip check --python .venv/bin/python
```

本机复用现有 LIBERO 和 Wan2.2 文件：

```bash
ln -sfn /mnt/workspace/mishuo/ImageWAM/third_party/LIBERO-plus \
  third_party/LIBERO-plus
ln -sfn /mnt/workspace/mishuo/ImageWAM/data/libero_mujoco3.3.2 \
  data/libero_mujoco3.3.2
mkdir -p checkpoints/Wan-AI
ln -sfn /mnt/workspace/mishuo/Wan2.2/Wan2.2-TI2V-5B \
  checkpoints/Wan-AI/Wan2.2-TI2V-5B
```

`DIFFSYNTH_DOWNLOAD_SOURCE=modelscope` 已写入 `.env.example`。现有 Wan
目录不包含 UMT5 tokenizer 时，首次 G0 会从 ModelScope 下载约 20 MB。

LIBERO 路径由项目内 `.libero/config.yaml` 固定，不读取各宿主机可能过期的
`~/.libero/config.yaml`。4052 缺少系统 MagickWand 时，启动器会自动使用
`.local/imagemagick-6/lib` 中的项目级运行库，并在 `environment.json` 记录
`MAGICK_HOME`、`LD_LIBRARY_PATH` 和 `LIBERO_CONFIG_PATH`。

## Artifact 校验

```bash
sha256sum checkpoints/fastwam_release/libero_uncond_2cam224.pt
sha256sum checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

期望值：

```text
1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579  libero_uncond_2cam224.pt
30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638  libero_uncond_2cam224_dataset_stats.json
```

## G1 隔离前缀重放

```bash
source .env.local
.venv/bin/python experiments/libero/test_state_restore.py \
  --suite libero_object \
  --task-id 0 \
  --trial-id 0 \
  --repeats 3 \
  --output outputs/rankwam/state_restore_rankwam_env.json
```

当前冒烟结果：分叉 state、终态和 proprio 误差均为 `0.0`，
`physics_passed=true`。跨 EGL context 的 RGB 不逐像素相等，单独审计且不用于标签；
切换到 OSMesa 后，独立环境的两路 RGB 也逐值完全一致。

## 4051 渲染后端

4051 的共享 GPU 作业会触发 NVIDIA `Xid 31/109`（MMU fault /
context-switch timeout），随后 step latency 上升或进程被宿主直接终止。GPU0 和
GPU5 均已复现。4052 的空闲 GPU1 也在 EGL 多 episode 的第 2 条轨迹触发
`Xid 31`，内核记录的 PCI `0000:07:00` 与该卡完全对应。正式评测仍默认保持
上游的 EGL 路径；需要隔离 renderer 时显式使用：

```bash
export RANKWAM_MUJOCO_GL=osmesa
```

这只把 MuJoCo offscreen rendering 放到 CPU；FastWAM 推理仍由
`CUDA_VISIBLE_DEVICES` 指定的 GPU 执行。选择 OSMesa 是因为隔离重放的 physics、
proprio 和两路 RGB 均可重复，且能避免再增加 graphics context 争用；它不是
GPU Xid 的完整修复。EGL 与 OSMesa 的同状态 MP4 首帧审计结果为
`mean_abs=0.986/255`、`max_abs=72`、变化通道比例 `0.350`，其中变化比例包含
独立 MP4 编码误差。

EGL 可选用单独的物理 GPU 渲染，模型仍使用 `GPU_ID` 指定的第一张卡：

```bash
GPU_ID=2 EGL_GPU_ID=3 RANKWAM_MUJOCO_GL=egl scripts/run_g0_pilot.sh
```

启动器会设置 `CUDA_VISIBLE_DEVICES=2,3` 和 `MUJOCO_EGL_DEVICE_ID=3`，并拒绝
两个 id 相同。该路径必须先通过至少 5 trials，才能称为上游可比的 G0。

## G0 单 Trial

```bash
source .env.local
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  model.redirect_common_files=false \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_object \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.num_steps_wait=5 \
  EVALUATION.output_dir=./outputs/rankwam/g0_smoke \
  gpu_id=0
```

该命令已完整执行 405 步并生成结果 JSON 和 MP4。单次结果 `0/1` 只证明
端到端连通，不用于估计成功率。

当前结果：

- 4051 GPU0 + OSMesa 正式参数完整跑完 5 trials，`0/5`，575.20 秒，输出为
  `outputs/rankwam/g0_5trials_4051_gpu0_osmesa_20260819`；
- 4052 GPU1 + EGL 的 trial 0 完整结束，`0/1`；同状态两种 renderer 的首帧
  PSNR 为 40.63 dB，但闭环轨迹随后分叉；
- 4052 EGL 多 trial 两次在 episode 2 被 Xid 31 终止，不能形成正式 5-trial
  上游基线。

因此 OSMesa `0/5` 是功能基线，不是上游成功率估计。EGL 稳定性修复前不扩展到
50 trials，也不据此训练 ranker。

正式 G0 使用逐 trial 故障隔离。每个 trial 都会启动一个新模型进程，并保存
`command.txt`、`resolved_config.yaml`、`environment.json`、结果 JSON 和 MP4；
失败 shard 最多重试 3 次，已完成 shard 可断点续跑：

```bash
GPU_ID=0 START_TRIAL=0 TOTAL_TRIALS=5 MAX_ATTEMPTS=3 \
OUTPUT_ROOT=./outputs/rankwam/g0_sharded_5trials \
scripts/run_g0_sharded.sh
```

全部完成后，`summary.json` 汇总成功率、Wilson 95% 区间和各 shard 路径。
扩展到 50 trials 时只改 `TOTAL_TRIALS` 和 `OUTPUT_ROOT`。单进程多 trial 在 4051
上已连续两次于第 2 个 episode 被 Xid 终止，不再作为默认运行方式。
