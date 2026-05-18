#!/bin/bash
# Usage: merge.sh <CONFIG_FILE> <LORA_MODEL_DIR>
# Author: Adam Zvara (xzvara01)
# Date: 03/2026
set -euo pipefail
CONFIG_FILE="${1:?CONFIG_FILE required}"
LORA_MODEL_DIR="${2:?LORA_MODEL_DIR required}"

echo "=== Merging LoRA: $LORA_MODEL_DIR ==="
CUDA_VISIBLE_DEVICES="" axolotl merge-lora "$CONFIG_FILE" --lora-model-dir="$LORA_MODEL_DIR"
echo "=== Merge complete ==="
