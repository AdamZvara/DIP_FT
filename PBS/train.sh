#!/bin/bash
# Run training with given configuration 
# Author: Adam Zvara (xzvara01)
# Date: 03/2026
#PBS -N axolotl_train
#PBS -l select=1:ncpus=2:mem=64gb:scratch_local=20gb:ngpus=2:gpu_mem=60gb
#PBS -l walltime=3:00:00

set -euo pipefail
cd "${PBS_O_WORKDIR}" || exit 2

source PBS/base.sh

# Required PBS vars passed via qsub -v
: "${CONFIG:?CONFIG PBS variable required}"
: "${DATASET:?DATASET PBS variable required}"
: "${OUTPUT_DIR:?OUTPUT_DIR PBS variable required}"
NGPU="${NGPU:-2}"
TRAIN_SIZE="${TRAIN_SIZE:-}"

export WANDB_DISABLED=true COMET_MODE=disabled HF_MLFLOW_LOG_ARTIFACTS=false
export AXOLOTL_DO_NOT_TRACK=1 AXOLOTL_TELEMETRY_DISABLED=1

bash scripts/run_training.sh \
    "${PROJECT_ROOT}/configs/qwen_${CONFIG}.yaml" \
    "${PROJECT_ROOT}/${DATASET}" \
    "${OUTPUT_DIR}" \
    "${NGPU}" \
    "${TRAIN_SIZE}"

clean_scratch
