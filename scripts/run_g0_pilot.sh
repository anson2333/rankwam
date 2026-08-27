#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .env.local ]]; then
  # shellcheck disable=SC1091
  source .env.local
fi

LOCAL_MAGICK_HOME="$PROJECT_ROOT/.local/imagemagick-6"
if [[ -d "$LOCAL_MAGICK_HOME/lib" ]]; then
  export MAGICK_HOME="$LOCAL_MAGICK_HOME"
  export LD_LIBRARY_PATH="$LOCAL_MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export LIBERO_CONFIG_PATH="${RANKWAM_LIBERO_CONFIG_PATH:-$PROJECT_ROOT/.libero}"

GPU_ID="${GPU_ID:-0}"
EGL_GPU_ID="${EGL_GPU_ID:-}"
NUM_TRIALS="${NUM_TRIALS:-5}"
TASK_ID="${TASK_ID:-0}"
TRIAL_START="${TRIAL_START:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/rankwam/g0_${NUM_TRIALS}trials_${RUN_TAG}}"
CHECKPOINT="${CHECKPOINT:-./checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
LIBERO_ROOT="${LIBERO_ROOT:-./third_party/LIBERO-plus}"

# Keep the upstream EGL path by default. OSMesa remains available for renderer
# isolation tests on congested hosts, without being inherited accidentally.
export MUJOCO_GL="${RANKWAM_MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == "egl" && -n "$EGL_GPU_ID" ]]; then
  if [[ "$EGL_GPU_ID" == "$GPU_ID" ]]; then
    echo "EGL_GPU_ID must differ from GPU_ID when renderer isolation is enabled" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$GPU_ID,$EGL_GPU_ID"
  export MUJOCO_EGL_DEVICE_ID="$EGL_GPU_ID"
else
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  unset MUJOCO_EGL_DEVICE_ID
fi

mkdir -p "$OUTPUT_DIR"

ARGS=(
  task=libero_uncond_2cam224_1e-4
  "ckpt=$CHECKPOINT"
  model.redirect_common_files=false
  "EVALUATION.dataset_stats_path=$DATASET_STATS"
  EVALUATION.task_suite_name=libero_object
  "EVALUATION.task_id=$TASK_ID"
  "+EVALUATION.trial_start=$TRIAL_START"
  "EVALUATION.num_trials=$NUM_TRIALS"
  "EVALUATION.output_dir=$OUTPUT_DIR"
  "gpu_id=$GPU_ID"
)

printf '%q ' .venv/bin/python experiments/libero/eval_libero_single.py "${ARGS[@]}" \
  > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"

.venv/bin/python experiments/libero/eval_libero_single.py "${ARGS[@]}" --cfg job \
  > "$OUTPUT_DIR/resolved_config.yaml"

.venv/bin/python scripts/report_environment.py \
  --output "$OUTPUT_DIR/environment.json" \
  --libero-root "$LIBERO_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --dataset-stats "$DATASET_STATS"

exec .venv/bin/python experiments/libero/eval_libero_single.py "${ARGS[@]}"
