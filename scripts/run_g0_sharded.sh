#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPU_ID="${GPU_ID:-0}"
START_TRIAL="${START_TRIAL:-0}"
TOTAL_TRIALS="${TOTAL_TRIALS:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/rankwam/g0_sharded_${TOTAL_TRIALS}trials}"

if (( START_TRIAL < 0 || TOTAL_TRIALS <= 0 || MAX_ATTEMPTS <= 0 )); then
  echo "START_TRIAL must be >= 0; TOTAL_TRIALS and MAX_ATTEMPTS must be > 0" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

for ((offset = 0; offset < TOTAL_TRIALS; offset++)); do
  trial=$((START_TRIAL + offset))
  shard_dir="$OUTPUT_ROOT/trial_$(printf '%03d' "$trial")"
  result_file="$shard_dir/libero_object/gpu${GPU_ID}_task0_results.json"
  mkdir -p "$shard_dir"

  if [[ -s "$result_file" ]]; then
    echo "trial $trial: existing result, skip"
    continue
  fi

  completed=0
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    echo "trial $trial: attempt $attempt/$MAX_ATTEMPTS"
    set +e
    GPU_ID="$GPU_ID" \
    NUM_TRIALS=1 \
    TRIAL_START="$trial" \
    OUTPUT_DIR="$shard_dir" \
      scripts/run_g0_pilot.sh > "$shard_dir/attempt_${attempt}.log" 2>&1
    rc=$?
    set -e
    if (( rc == 0 )) && [[ -s "$result_file" ]]; then
      completed=1
      break
    fi
    echo "trial $trial: attempt $attempt failed with exit code $rc" >&2
  done

  if (( completed == 0 )); then
    echo "trial $trial failed after $MAX_ATTEMPTS attempts" >&2
    exit 1
  fi
done

.venv/bin/python scripts/summarize_g0_shards.py \
  --root "$OUTPUT_ROOT" \
  --start-trial "$START_TRIAL" \
  --expected-trials "$TOTAL_TRIALS" \
  --output "$OUTPUT_ROOT/summary.json"
