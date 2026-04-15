sinclude .env            # loads PBS_OUT_DIR, HF_HOME, etc. from .env
PBS_OUT_DIR ?= .

CONFIG     ?= ft
NGPU       ?= 2
WALLTIME   ?= 1:00:00
RUNS_DIR   ?= outputs
OUTPUT_DIR ?= $(RUNS_DIR)/qwen_$(CONFIG)_$(shell date +%Y%m%d_%H%M%S)
TRAIN_SIZE ?=
GPU_MEM   ?= 60gb
# DATASET has no default - users must provide it

CONFIG_FILE := configs/qwen_$(CONFIG).yaml
_TS         := $(shell date +%Y%m%d_%H%M%S)

.PHONY: help train train-pbs merge

help:
	@echo "Targets:"
	@echo "  train      - Run training directly (no PBS)"
	@echo "  train-pbs  - Submit PBS job"
	@echo "  merge      - Merge LoRA weights (CONFIG=lora required)"
	@echo ""
	@echo "Required:"
	@echo "  DATASET=<path>   Path to .jsonl training file"
	@echo ""
	@echo "Optional:"
	@echo "  CONFIG=ft|lora     (default: ft)"
	@echo "  OUTPUT_DIR=...     (default: outputs/qwen_<CONFIG>_<timestamp>)"
	@echo "  NGPU=2             (default: 2)"
	@echo "  WALLTIME=1:00:00   (default: 1:00:00, PBS only)"
	@echo "  TRAIN_SIZE=N       Use first N samples for train, rest for eval (default: all, no eval)"

train:
	@test -f "$(DATASET)" || { echo "ERROR: DATASET not found: $(DATASET)"; exit 1; }
	@test -f "$(CONFIG_FILE)" || { echo "ERROR: $(CONFIG_FILE) not found"; exit 1; }
	bash scripts/run_training.sh "$(CONFIG_FILE)" "$(DATASET)" "$(OUTPUT_DIR)" "$(NGPU)" "$(TRAIN_SIZE)"

train-pbs:
	@test -f "$(DATASET)" || { echo "ERROR: DATASET not found: $(DATASET)"; exit 1; }
	@test -f "$(CONFIG_FILE)" || { echo "ERROR: $(CONFIG_FILE) not found"; exit 1; }
	@test -d "$(PBS_OUT_DIR)" || { echo "ERROR: PBS_OUT_DIR does not exist: $(PBS_OUT_DIR)"; exit 1; }
	@sleep 2
	qsub \
		-l select=1:ncpus=$(NGPU):mem=64gb:scratch_local=128gb:ngpus=$(NGPU):gpu_mem=$(GPU_MEM) \
		-l walltime=$(WALLTIME) \
		-o $(PBS_OUT_DIR)/$(CONFIG)_$(_TS).out \
		-e $(PBS_OUT_DIR)/$(CONFIG)_$(_TS).err \
		-v CONFIG=$(CONFIG),DATASET=$(DATASET),OUTPUT_DIR=$(OUTPUT_DIR),NGPU=$(NGPU),TRAIN_SIZE=$(TRAIN_SIZE) \
		PBS/train.sh

merge:
	@test "$(CONFIG)" = "lora" || { echo "ERROR: merge only applies to CONFIG=lora"; exit 1; }
	@test -n "$(OUTPUT_DIR)" || { echo "ERROR: OUTPUT_DIR is required"; exit 1; }
	bash scripts/post_process.sh "$(CONFIG_FILE)" "$(OUTPUT_DIR)" lora

auth-lora-sizes:
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_30.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_60.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_100.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_250.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_500.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_750.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_1000.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_1250.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=lora DATASET=data/auth/train_1500.jsonl NGPU=2 WALLTIME=3:00:00

auth-ft-sizes:
	$(MAKE) train-pbs CONFIG=ft DATASET=data/auth/train_60.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=ft DATASET=data/auth/train_100.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=ft DATASET=data/auth/train_250.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=ft DATASET=data/auth/train_500.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=ft DATASET=data/auth/train_750.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=ft DATASET=data/auth/train_1000.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=ft DATASET=data/auth/train_1250.jsonl NGPU=2 WALLTIME=3:00:00
	$(MAKE) train-pbs CONFIG=ft DATASET=data/auth/train_1500.jsonl NGPU=2 WALLTIME=3:00:00