#!/bin/bash
# Usage: run_training.sh <CONFIG_FILE> <DATASET> <OUTPUT_DIR> [NGPU=2]
set -euo pipefail

CONFIG_FILE="${1:?CONFIG_FILE required}"
DATASET="${2:?DATASET required}"
OUTPUT_DIR="${3:?OUTPUT_DIR required}"
NGPU="${4:-2}"
TRAIN_SIZE="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env for direct (non-PBS) runs
if [ -z "${HF_HOME:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

[ -f "$CONFIG_FILE" ] || { echo >&2 "ERROR: $CONFIG_FILE not found"; exit 1; }
[ -f "$DATASET" ]     || { echo >&2 "ERROR: $DATASET not found"; exit 1; }

# Detect adapter type
ADAPTER_TYPE="ft"
grep -q '^adapter:' "$CONFIG_FILE" && ADAPTER_TYPE="lora"

# Create temp patched config
JOB_ID="${PBS_JOBID:-$$}"
CONFIG_BASENAME="$(basename "$CONFIG_FILE" .yaml)"
TEMP_CONFIG="/tmp/axolotl_${JOB_ID}_${CONFIG_BASENAME}.yaml"
trap 'rm -f "$TEMP_CONFIG"' EXIT

ABS_OUTPUT="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$OUTPUT_DIR")"
mkdir -p "$ABS_OUTPUT"

PATCH_ARGS=(--dataset "$DATASET" --output-dir "$ABS_OUTPUT")
[ -n "$TRAIN_SIZE" ] && PATCH_ARGS+=(--train-size "$TRAIN_SIZE")

# Auto-detect sibling test.jsonl for static eval set
EVAL_DATASET="$(dirname "$DATASET")/test.jsonl"
if [ -f "$EVAL_DATASET" ]; then
    PATCH_ARGS+=(--eval-dataset "$EVAL_DATASET")
fi

python3 "$SCRIPT_DIR/patch_config.py" \
    "$CONFIG_FILE" "$TEMP_CONFIG" \
    "${PATCH_ARGS[@]}"

echo "=== Config patched: dataset=$DATASET output_dir=$ABS_OUTPUT${TRAIN_SIZE:+ train_size=$TRAIN_SIZE}${EVAL_DATASET:+ eval_dataset=$EVAL_DATASET} ==="

# Start GPU monitor (background)
MONITOR_PID=""
if python3 -c "import pynvml" 2>/dev/null; then
    python3 "$SCRIPT_DIR/gpu_monitor.py" \
        --output "$ABS_OUTPUT/gpu_metrics.json" \
        --interval 10 \
        --job-id "${PBS_JOBID:-local_$$}" \
        --config "$CONFIG_BASENAME" \
        --dataset "$DATASET" &
    MONITOR_PID=$!
    echo "=== GPU monitor started (PID $MONITOR_PID) ==="
fi

trap '
    [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null && {
        kill -TERM "$MONITOR_PID"; wait "$MONITOR_PID" 2>/dev/null || true; }
    rm -f "$TEMP_CONFIG"
' EXIT

# Run training
echo "=== Training: NGPU=$NGPU adapter=$ADAPTER_TYPE ==="
accelerate launch --num_processes "${NGPU}" -m axolotl.cli.train "$TEMP_CONFIG"

# Stop monitor
if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill -TERM "$MONITOR_PID"; wait "$MONITOR_PID" 2>/dev/null || true; MONITOR_PID=""
fi

# Post-process
bash "$SCRIPT_DIR/post_process.sh" "$TEMP_CONFIG" "$ABS_OUTPUT" "$ADAPTER_TYPE"
echo "=== Done. Outputs: $ABS_OUTPUT ==="
