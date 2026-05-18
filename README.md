# Fine-Tuning Pipeline for Qwen2.5-7B

A fine-tuning pipeline for [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) built on top of [Axolotl](https://github.com/axolotl-ai-cloud/axolotl). Axolotl is a framework that wraps HuggingFace Transformers and PEFT to provide a unified YAML-driven interface for training large language models handling dataset loading, tokenization, distributed training setup, and checkpointing. This pipeline adds a Makefile entry point, runtime config patching (dataset path injection, train/eval splitting), background GPU metrics collection, and PBS HPC job submission on top of Axolotl.

Two training modes are supported:

- **Full fine-tuning (FSDP)** - updates all model weights across 2+ GPUs using PyTorch Fully Sharded Data Parallel
- **LoRA** - trains lightweight adapter matrices (`r=8`, targeting `q_proj`/`v_proj`) while keeping base weights frozen

## Installation

### Requirements

- conda (miniconda or anaconda)
- CUDA-capable GPU(s)
- Python 3.10+

### 1. Create conda environment and install Axolotl

```bash
conda create -n fine-tuning python=3.11
conda activate fine-tuning
pip install axolotl[flash-attn,fsdp]
pip install pynvml pyyaml          # GPU monitor + config patching
```

Refer to the [Axolotl installation docs](https://github.com/axolotl-ai-cloud/axolotl#installation) for GPU-specific instructions (Flash Attention, bitsandbytes, etc.).

### 2. Configure `.env`

Copy the example and fill in the paths:

```bash
cp .env-example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_HOME` | yes | HuggingFace model/dataset cache directory |
| `CONDA_ENV` | yes (PBS) | Conda environment name, activated in PBS jobs |
| `DATADIR` | yes (PBS) | Persistent storage root (used for `.conda` symlink on cluster) |
| `PBS_OUT_DIR` | yes (PBS) | Directory for PBS stdout/stderr log files |
| `PROJECT_ROOT` | yes (PBS) | Absolute path to this repo on the cluster |
| `RUNS_DIR` | no | Output root directory (default: `outputs/`) |

For local runs only `HF_HOME` is strictly required; the rest are used by PBS scripts.

## Running Fine-Tuning

All operations go through `make`. The `DATASET` variable (path to a `.jsonl` file) is always required.

### Local run

```bash
# Full fine-tuning (FSDP, 2 GPUs)
make train CONFIG=ft DATASET=data/auth/train.jsonl

# LoRA
make train CONFIG=lora DATASET=data/auth/train.jsonl

# Use only first 500 samples for training, rest as eval
make train CONFIG=lora DATASET=data/auth/train.jsonl TRAIN_SIZE=500

# Custom output directory
make train CONFIG=ft DATASET=data/auth/train.jsonl OUTPUT_DIR=outputs/my_run
```

If a `test.jsonl` file exists in the same directory as `DATASET`, it is automatically used as a static evaluation set.

### PBS cluster submission (MetaCentrum)

```bash
make train-pbs CONFIG=ft   DATASET=data/auth/train.jsonl WALLTIME=4:00:00
make train-pbs CONFIG=lora DATASET=data/hashing/train_500.jsonl NGPU=2 WALLTIME=2:00:00
```

PBS stdout/stderr are written to `$PBS_OUT_DIR/<CONFIG>_<timestamp>.{out,err}`.

### Merge LoRA adapter weights

LoRA merge runs automatically after training. To re-run manually:

```bash
make merge CONFIG=lora OUTPUT_DIR=outputs/qwen_lora_20260309_120000
```

### All Makefile options

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `ft` | Training mode: `ft` (full) or `lora` |
| `DATASET` | *(required)* | Path to `.jsonl` training file |
| `OUTPUT_DIR` | `outputs/qwen_<CONFIG>_<timestamp>` | Output directory |
| `NGPU` | `2` | Number of GPUs |
| `TRAIN_SIZE` | *(all)* | Integer count or percentage (e.g. `500` or `80%`) of train samples |
| `WALLTIME` | `1:00:00` | PBS wall time (PBS only) |
| `GPU_MEM` | `60gb` | Per-GPU VRAM requirement for PBS scheduler |

## Data Format

Training data uses Alpaca-style JSONL — one JSON object per line:

```json
{"instruction": "Create a login system in Python", "output": "def check_login(...):\n    ..."}
```

Datasets live under `data/`, organized by task. Each task directory typically contains a full `train.jsonl`/`test.jsonl` split plus pre-sampled subsets (`train_30.jsonl`, `train_100.jsonl`, …) for data-size ablations.

```
data/
├── auth/          # authentication code generation (~1500 samples)
├── hashing/       # hashing code generation (~2500 samples)
└── supply_chain/  # supply chain code (~1750 samples)
```

To create a random subset from an existing dataset:

```bash
python data/create_subset.py data/auth/train.jsonl 200 --seed 42
# → data/auth/train_200.jsonl
```

## Output Structure

Each run produces a timestamped directory:

```
outputs/qwen_ft_20260309_120000/
├── training_config.yaml     # patched config used for this run
├── training_data.jsonl      # copy of the dataset
├── gpu_metrics.json         # peak VRAM, avg power, total energy (kWh) per GPU
└── checkpoint-N/
    ├── model.safetensors
    ├── tokenizer.json
    └── config.json
```

For LoRA runs, the merged full weights are written to `checkpoint-N/merged/`.

## Repository Structure

```
.
├── Makefile                  # entry point for all operations
├── configs/
│   ├── qwen_ft.yaml          # full fine-tuning config template (FSDP)
│   └── qwen_lora.yaml        # LoRA config template
├── data/
│   ├── auth/                 # authentication task datasets
│   ├── hashing/              # hashing task datasets
│   ├── supply_chain/         # supply chain task datasets
│   ├── create_subset.py      # random subsample utility
│   └── split_dataset.py      # train/test split utility
├── mlflow_export/
│   ├── import_ft.py          # bulk-import run outputs into MLflow
│   ├── training_data_html.py # render training data as HTML artifact
│   └── dataset_html.py       # HTML builder helper
├── PBS/
│   ├── base.sh               # cluster env setup (scratch, conda activation)
│   ├── env.sh                # .env loader
│   └── train.sh              # PBS job script
└── scripts/
    ├── run_training.sh       # orchestrates patch → monitor → train → post-process
    ├── patch_config.py       # injects dataset path, output dir, train/eval split into YAML
    ├── gpu_monitor.py        # background GPU VRAM + power metrics collector
    ├── post_process.sh       # copies artifacts, triggers LoRA merge
    └── merge.sh              # runs axolotl merge-lora on CPU
```

## Adding a New Configuration

1. Copy an existing config as a starting point:

   ```bash
   cp configs/qwen_lora.yaml configs/qwen_lora_large.yaml
   ```

2. Edit the new file. The `datasets` and `output_dir` fields are overwritten at runtime by `patch_config.py` — set them to `INJECTED` as a placeholder:

   ```yaml
   datasets:
     - path: INJECTED
       type: alpaca
   output_dir: INJECTED
   ```

3. Use it via `make` with the config name (filename without the `qwen_` prefix and `.yaml` extension):

   ```bash
   make train CONFIG=lora_large DATASET=data/auth/train.jsonl
   ```

   The Makefile resolves `CONFIG` to `configs/qwen_<CONFIG>.yaml`.

## Exporting Runs to MLflow

The `mlflow_export/import_ft.py` script bulk-imports completed run directories into an MLflow experiment. It was tested against a local MLflow deployment (`mlflow server --host 127.0.0.1 --port 5000`).

For each run directory it logs:

- **Params** - training config (learning rate, epochs, LoRA rank, dataset path, …)
- **Metrics** - per-step loss, eval loss, perplexity, learning rate, grad norm
- **GPU summary** - peak VRAM per GPU, average power, total energy (kWh), run duration
- **Artifacts** - `training_config.yaml`, `gpu_metrics.json`, and an HTML viewer for the training data

### Quick start

```bash
pip install mlflow pyyaml

# Start MLflow tracking server (separate terminal)
mlflow server --host 127.0.0.1 --port 5000

# Edit the constants at the top of the script to point at your outputs directory
# OUTPUTS_DIR = Path("outputs/supply_chain")
# MLFLOW_URI  = "http://127.0.0.1:5000"
# EXPERIMENT_NAME = "fine_tuning_supply_chain"

python -m mlflow_export/import_ft

# Re-import from scratch (deletes existing runs for the experiment first)
python -m mlflow_export/import_ft --force
```

Open `http://127.0.0.1:5000` to browse runs, compare metrics, and view artifacts.
