#!/bin/bash
#PBS -N FT
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=20gb:ngpus=2:gpu_mem=48gb
#PBS -l walltime=1:00:00
#PBS -o FT.out
#PBS -e FT.err

source /storage/brno2/home/xzvara01/DIP/ft/PBS/base_eval.sh

export AXOLOTL_DO_NOT_TRACK=1
export AXOLOTL_TELEMETRY_DISABLED=1
export AXOLOTL_TELEMETRY_DISABLED=1
export AXOLOTL_DISABLE_TELEMETRY=1
export AXOLOTL_NO_TELEMETRY=1

cd $DATADIR/DIP/ft
accelerate launch \
  --num_processes=2 \
  -m axolotl.cli.train \
  $DATADIR/DIP/ft/qwen_ft.yaml

clean_scratch
