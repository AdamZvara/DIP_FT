#!/bin/bash
#PBS -N FT
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=20gb:ngpus=2:gpu_mem=48gb
#PBS -l walltime=1:00:00

source /storage/brno2/home/xzvara01/ft/PBS/base_eval.sh

export AXOLOTL_DO_NOT_TRACK=0

cd $DATADIR/ft
accelerate launch -m axolotl.cli.train $BRNO/ft/qwen_ft.yaml

clean_scratch
