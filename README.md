# Axolotl Fine-Tuning

Fine-tuning Qwen2.5-7B via [Axolotl](https://github.com/axolotl-ai-cloud/axolotl). Supports full fine-tuning (FSDP) and LoRA, with parameterized runs via `make`.

## Datasets

- `auth_clean` — ~200 samples besides codeparrot and thestack
- `auth_refactored` — ~600 samples of the same data as `auth_clean` but each row refactored 2 times (instructions duplicated)

## Setup

### 1. Configure `.env`

Copy and edit the environment file:

```bash
cp .env.example .env   # or create .env manually
```

Required variables:

| Variable | Description |
|----------|-------------|
| `HF_HOME` | HuggingFace cache directory |
| `DATADIR` | Persistent data/home directory on the cluster |
| `CONDA_ENV` | Conda environment name with Axolotl installed |
| `PBS_OUT_DIR` | Directory for PBS stdout/stderr logs |

Optional:

| Variable | Description |
|----------|-------------|
| `PROJECT_ROOT` | Absolute path to this repo on the cluster |
| `RESULTS_DIR` | Directory for experiment results |

### 2. Conda environment

The scripts expect a conda environment with Axolotl and its dependencies installed. Activate it manually for direct runs, or set `CONDA_ENV` in `.env` for PBS runs (activated automatically via `PBS/base.sh`).

### 3. Cluster: symlink `.conda`

On the cluster, `PBS/base.sh` symlinks `~/.conda` to `$DATADIR/.conda` if not already present (avoids quota issues on home directories).

## Usage

All training is driven through `make`. The only required argument is `DATASET`.

### Direct run (no PBS)

```bash
# Full fine-tuning
make train CONFIG=ft DATASET=data/auth_refactored.jsonl

# LoRA
make train CONFIG=lora DATASET=data/auth_clean.jsonl

# Custom output directory
make train CONFIG=ft DATASET=data/auth_refactored.jsonl OUTPUT_DIR=outputs/my_experiment
```

### PBS job submission

```bash
make train-pbs CONFIG=ft   DATASET=/storage/.../data.jsonl NGPU=2 WALLTIME=4:00:00
make train-pbs CONFIG=lora DATASET=/storage/.../data.jsonl NGPU=2 WALLTIME=2:00:00
```

PBS stdout/stderr go to `$PBS_OUT_DIR/<CONFIG>_<timestamp>.{out,err}`.

### Merge LoRA weights (manual)

Normally called automatically after training. To re-run manually:

```bash
make merge CONFIG=lora OUTPUT_DIR=outputs/qwen_lora_20260309_120000
```

### All options

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `ft` | Config name: `ft` or `lora` |
| `DATASET` | *(required)* | Path to `.jsonl` training file |
| `OUTPUT_DIR` | `outputs/qwen_<CONFIG>_<timestamp>` | Where to write outputs |
| `NGPU` | `2` | Number of GPUs |
| `WALLTIME` | `1:00:00` | PBS walltime (PBS only) |

## Output Structure

Each run produces a self-contained output directory:

```
outputs/qwen_ft_20260309_120000/
├── training_config.yaml     # patched config used for the run
├── training_data.jsonl      # copy of the dataset
├── gpu_metrics.json         # peak VRAM, avg power, total energy (kWh) per GPU
└── checkpoint-N/            # latest checkpoint, with tokenizer files copied in
    ├── model.safetensors
    ├── tokenizer.json
    ├── config.json
    └── ...
```

For LoRA runs, the merged weights are written alongside the checkpoint.

## Repository Structure

```
.
├── Makefile                  # entry point
├── configs/
│   ├── qwen_ft.yaml          # full fine-tuning config template
│   └── qwen_lora.yaml        # LoRA config template
├── data/                     # training datasets
├── PBS/
│   ├── base.sh               # cluster setup (scratch, conda)
│   ├── env.sh                # .env loader
│   └── train.sh              # PBS job script
└── scripts/
    ├── run_training.sh       # core training logic
    ├── post_process.sh       # artifact copy, LoRA merge, checkpoint packaging
    ├── patch_config.py       # patches dataset path + output_dir into config
    ├── gpu_monitor.py        # background GPU metrics collector
    └── merge.sh              # LoRA merge wrapper
```
