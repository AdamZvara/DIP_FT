#!/bin/bash
# Usage: post_process.sh <CONFIG_FILE> <OUTPUT_DIR> <ADAPTER_TYPE>
#
# Post process tasks (LoRA weight merging)
#
# Author: Adam Zvara (xzvara01)
# Date: 03/2026

set -euo pipefail

CONFIG_FILE="${1:?required}"; OUTPUT_DIR="${2:?required}"; ADAPTER_TYPE="${3:?required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read dataset path back from (patched) config
DATASET_PATH="$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f: cfg = yaml.safe_load(f)
print(cfg['datasets'][0]['path'])
" "$CONFIG_FILE")"

# Copy artifacts
[ -f "$DATASET_PATH" ] && cp "$DATASET_PATH" "$OUTPUT_DIR/training_data.jsonl" \
    && echo "Copied dataset -> training_data.jsonl"
cp "$CONFIG_FILE" "$OUTPUT_DIR/training_config.yaml"
echo "Copied config  -> training_config.yaml"

if [ "$ADAPTER_TYPE" = "lora" ]; then
    echo "=== Merging LoRA weights ==="
    bash "$SCRIPT_DIR/merge.sh" "$CONFIG_FILE" "$OUTPUT_DIR"

elif [ "$ADAPTER_TYPE" = "ft" ]; then
    echo "=== Making latest checkpoint self-contained ==="
    LATEST_CKPT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" \
        | sort -t- -k2 -n | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "WARNING: No checkpoint-* dirs found"
    else
        echo "Latest checkpoint: $LATEST_CKPT"
        find "$OUTPUT_DIR" -maxdepth 1 -type f | while read -r fpath; do
            fname="$(basename "$fpath")"
            [ ! -f "$LATEST_CKPT/$fname" ] && cp "$fpath" "$LATEST_CKPT/$fname" \
                && echo "  Copied $fname into checkpoint"
        done
    fi
fi
