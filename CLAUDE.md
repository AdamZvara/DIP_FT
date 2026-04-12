# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fine-tuning pipeline for **Qwen2.5-7B** using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), supporting full fine-tuning (FSDP) and LoRA. Designed for both local multi-GPU runs and PBS HPC cluster submission (MetaCentrum).

## Common Commands

All operations go through `make`. The `DATASET` variable must point to a `.jsonl` file.

```bash
# Full fine-tuning (FSDP, 2 GPUs by default)
make train CONFIG=ft DATASET=data/auth_clean.jsonl

# LoRA fine-tuning
make train CONFIG=lora DATASET=data/auth_refactored.jsonl

# Train with 80% of data, rest as eval set
make train CONFIG=ft DATASET=data/auth_clean.jsonl TRAIN_SIZE=80%

# Submit as PBS job to HPC cluster
make train-pbs CONFIG=ft DATASET=data/auth_clean.jsonl WALLTIME=4:00:00

# Merge LoRA adapter weights with base model (usually called automatically)
make merge OUTPUT_DIR=outputs/qwen_lora_<timestamp>
```

Key Makefile variables:
- `CONFIG`: `ft` (full fine-tuning) or `lora`
- `DATASET`: Path to `.jsonl` training file (required)
- `NGPU`: Number of GPUs (default: 2)
- `GPU_MEM`: GPU VRAM limit for PBS jobs (default: `60gb`)
- `TRAIN_SIZE`: Training split size — integer count or percentage (e.g., `80%`)
- `WALLTIME`: PBS wall time (default: `8:00:00`)

## Architecture

```
make train
  └── scripts/run_training.sh
        ├── scripts/patch_config.py    # Inject dataset path, output dir, train/eval split
        ├── scripts/gpu_monitor.py     # Background GPU VRAM + power metrics collector
        ├── accelerate + axolotl       # Actual training
        └── scripts/post_process.sh   # Copy artifacts; run merge.sh for LoRA
```

For PBS submission: `make train-pbs` → `qsub PBS/train.sh` → `PBS/base.sh` (env setup) → `scripts/run_training.sh`

### Key Scripts

| Script | Role |
|---|---|
| `scripts/patch_config.py` | Reads base YAML config, injects `datasets`, `output_dir`, and optionally splits data into train/eval |
| `scripts/gpu_monitor.py` | Polls `pynvml` every second; writes `gpu_metrics.json` with peak VRAM, avg power, total energy (kWh) |
| `scripts/run_training.sh` | Orchestrates: patch → monitor → train → metrics → post-process |
| `scripts/post_process.sh` | Copies dataset + config to output dir; triggers LoRA merge |
| `scripts/merge.sh` | Runs `axolotl merge-lora` on CPU (`CUDA_VISIBLE_DEVICES=""`) |
| `PBS/base.sh` | Cluster env setup: symlinks `~/.conda → $DATADIR/.conda`, activates conda env |

### Axolotl Configs

- `configs/qwen_ft.yaml` — Full fine-tuning with FSDP, bfloat16, gradient checkpointing; `lr=2e-5`, 3 epochs
- `configs/qwen_lora.yaml` — LoRA (`r=8, alpha=16`) on `q_proj`/`v_proj`; `lr=2e-4`, eval every 10 steps

`patch_config.py` overwrites `datasets` and `output_dir` fields at runtime — do not rely on hardcoded values in the YAML files.

## Data Format

Training data uses Alpaca-style JSONL (one JSON object per line):

```json
{"instruction": "Create a login system in Python", "output": "def check_login(...):\n    ..."}
```

Datasets in `data/`:
- `auth_clean.jsonl` — 211 authentication code samples
- `auth_refactored.jsonl` — 562 samples (same code, 2 refactored variants each)

## Output Structure

Each run produces a timestamped directory:

```
outputs/qwen_<CONFIG>_<TIMESTAMP>/
├── training_config.yaml     # Patched config used for this run
├── training_data.jsonl      # Copy of the dataset
├── gpu_metrics.json         # Peak VRAM, avg power, total energy
└── checkpoint-N/
    ├── model.safetensors
    ├── tokenizer.json
    └── config.json          # (LoRA: also contains merged model files)
```

## Cluster Setup

Requires a `.env` file at repo root (not committed) with:

```bash
PROJECT_ROOT=/storage/brno2/home/<username>/DIP/ft
HF_HOME=...
DATADIR=...          # Persistent storage (used for .conda symlink)
RUNS_DIR=...         # Where outputs/ lives on cluster
PBS_OUT_DIR=...      # PBS stdout/stderr logs
CONDA_ENV=thestack
```

The `PBS/base.sh` script creates `~/.conda → $DATADIR/.conda` symlink to avoid home directory quota issues on the cluster.
