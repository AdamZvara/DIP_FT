"""
Import fine-tuning run outputs into MLflow Model Registry.

Scans outputs/ directory for fine-tuning/LoRA runs and for each:
  - Logs training config as params
  - Logs train loss, lr, grad_norm, ppl as time-series metrics (per step)
  - Logs GPU usage (VRAM, power, energy, duration) as summary metrics
  - Logs dataset info (sample count, path, split)
  - Registers the run as a model version in the MLflow Model Registry

Directory structure expected:
  outputs/{run_name}/
    training_config.yaml      (optional)
    gpu_metrics.json          (optional)
    adapter_config.json       (optional, LoRA runs)
    training_data.jsonl       (optional)
    checkpoint-{N}/
      trainer_state.json      (optional)

Author: Adam Zvara (xzvara01)
Date: 04/2026
"""

import json
import re
from pathlib import Path

import mlflow
import yaml

from mlflow_export.training_data_html import log_training_data_html

OUTPUTS_DIR = Path("outputs/supply_chain")
MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "fine_tuning_supply_chain"


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the checkpoint dir with the highest step number."""
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
    )
    return checkpoints[-1] if checkpoints else None


def load_training_config(run_dir: Path) -> dict:
    config_path = run_dir / "training_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_gpu_metrics(run_dir: Path) -> dict:
    path = run_dir / "gpu_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_trainer_state(run_dir: Path) -> dict:
    ckpt = find_latest_checkpoint(run_dir)
    if ckpt is None:
        return {}
    state_path = ckpt / "trainer_state.json"
    if not state_path.exists():
        return {}
    with open(state_path) as f:
        return json.load(f)


def load_adapter_config(run_dir: Path) -> dict:
    path = run_dir / "adapter_config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def count_training_samples(run_dir: Path) -> int | None:
    path = run_dir / "training_data.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        return sum(1 for _ in f)


def detect_run_type(run_dir: Path, config: dict) -> str:
    """Return 'lora' or 'ft'."""
    if (run_dir / "adapter_config.json").exists():
        return "lora"
    adapter = config.get("adapter", "")
    if adapter and adapter != "none":
        return "lora"
    # Fallback: directory name contains 'lora'
    if "lora" in run_dir.name.lower():
        return "lora"
    return "ft"


def infer_base_model_from_dir(run_dir: Path) -> str | None:
    """Try to infer base model name from directory name."""
    name = run_dir.name.lower()
    if "qwen2.5-7b" in name or "qwen_" in name:
        return "Qwen/Qwen2.5-7B"
    return None


def build_model_name(base_model: str, run_type: str) -> str:
    """Derive a clean registered model name."""
    short = base_model.split("/")[-1] if "/" in base_model else base_model
    suffix = "LoRA" if run_type == "lora" else "FT"
    return f"{short}-{suffix}"


def params_from_config(config: dict) -> dict:
    """Extract flat params from training_config.yaml."""
    params = {}
    scalar_keys = [
        "base_model", "model_type", "adapter", "num_epochs", "learning_rate",
        "micro_batch_size", "gradient_accumulation_steps", "sequence_len",
        "optimizer", "bf16", "fp16", "gradient_checkpointing",
        "lora_r", "lora_alpha", "lora_dropout", "val_set_size",
        "logging_steps", "save_steps",
    ]
    for k in scalar_keys:
        if k in config:
            params[k] = str(config[k])

    if "lora_target_modules" in config:
        params["lora_target_modules"] = ",".join(config["lora_target_modules"])

    # Dataset
    datasets = config.get("datasets", [])
    if datasets:
        params["dataset_path"] = datasets[0].get("path", "")
        params["dataset_split"] = datasets[0].get("split", "train")
        params["dataset_type"] = datasets[0].get("type", "")

    test_datasets = config.get("test_datasets", [])
    if test_datasets:
        params["eval_dataset_split"] = test_datasets[0].get("split", "")

    return params


def params_from_adapter_config(adapter_cfg: dict) -> dict:
    """Extract LoRA params from adapter_config.json (fallback if no training_config)."""
    keys = ["r", "lora_alpha", "lora_dropout", "peft_type", "peft_version",
            "base_model_name_or_path", "bias", "use_dora", "use_rslora"]
    params = {}
    for k in keys:
        if k in adapter_cfg:
            params[f"adapter_{k}"] = str(adapter_cfg[k])
    if "target_modules" in adapter_cfg:
        params["adapter_target_modules"] = ",".join(adapter_cfg.get("target_modules") or [])
    return params


def log_time_series_metrics(log_history: list):
    """Log per-step metrics as MLflow time-series metrics."""
    metric_keys = ["loss", "eval_loss", "eval_ppl", "learning_rate", "grad_norm", "ppl",
                   "memory/max_allocated (GiB)", "memory/device_reserved (GiB)",
                   "tokens/train_per_sec_per_gpu"]

    for entry in log_history:
        step = entry.get("step", 0)
        for key in metric_keys:
            if key in entry:
                safe_key = re.sub(r"[^a-zA-Z0-9_\-. /]", "_", key).replace(" ", "_").replace("/", "_")
                mlflow.log_metric(safe_key, entry[key], step=step)


def log_gpu_summary(gpu_data: dict):
    """Log GPU usage as summary metrics."""
    if not gpu_data:
        return

    mlflow.log_metric("duration_s", gpu_data.get("duration_s", 0))
    mlflow.log_metric("total_energy_kwh", gpu_data.get("total_energy_kwh", 0))

    gpus = gpu_data.get("gpus", [])
    mlflow.log_metric("num_gpus", len(gpus))
    for g in gpus:
        idx = g.get("gpu_index", 0)
        if "peak_vram_gb" in g:
            mlflow.log_metric(f"gpu{idx}_peak_vram_gb", g["peak_vram_gb"])
        if "avg_power_w" in g:
            mlflow.log_metric(f"gpu{idx}_avg_power_w", g["avg_power_w"])

    # Aggregate across GPUs
    if gpus:
        mlflow.log_metric("peak_vram_gb_total", sum(g.get("peak_vram_gb", 0) for g in gpus))
        mlflow.log_metric("avg_power_w_total", sum(g.get("avg_power_w", 0) for g in gpus))

    for k in ("job_id", "config"):
        if k in gpu_data:
            mlflow.set_tag(f"gpu_{k}", str(gpu_data[k]))


def log_trainer_summary(state: dict):
    """Log final training summary from trainer_state."""
    for k in ("global_step", "num_input_tokens_seen", "total_flos", "num_train_epochs"):
        if k in state:
            mlflow.log_metric(k, state[k])

    log_history = state.get("log_history", [])
    if log_history:
        # Final train loss
        train_steps = [e for e in log_history if "loss" in e]
        if train_steps:
            mlflow.log_metric("final_train_loss", train_steps[-1]["loss"])
        # Final eval loss and perplexity
        eval_steps = [e for e in log_history if "eval_loss" in e]
        if eval_steps:
            last_eval = eval_steps[-1]
            mlflow.log_metric("eval_loss", last_eval["eval_loss"])
            if "eval_ppl" in last_eval:
                mlflow.log_metric("eval_ppl", last_eval["eval_ppl"])


def extract_sample_count_from_path(path: str) -> int | None:
    """Extract sample count from dataset path like .../train_10.jsonl → 10."""
    m = re.search(r"train_(\d+)\.jsonl$", path)
    return int(m.group(1)) if m else None


def import_run(run_dir: Path, experiment_id: str, client: mlflow.tracking.MlflowClient,
               existing_run_names: set[str]):
    config = load_training_config(run_dir)
    run_type = detect_run_type(run_dir, config)

    run_name = run_dir.name
    if run_type == "lora":
        datasets = config.get("datasets", [])
        dataset_path = datasets[0].get("path", "") if datasets else ""
        n = extract_sample_count_from_path(dataset_path)
        if n is not None:
            run_name = f"{run_name}_{n}"

    if run_name in existing_run_names:
        print(f"  Skipping (already exists): {run_name}")
        return

    gpu_data = load_gpu_metrics(run_dir)
    state = load_trainer_state(run_dir)
    adapter_cfg = load_adapter_config(run_dir)
    n_samples = count_training_samples(run_dir)
    base_model = (
        config.get("base_model")
        or adapter_cfg.get("base_model_name_or_path")
        or gpu_data.get("config", "").replace("qwen_ft", "Qwen/Qwen2.5-7B").replace("qwen_lora", "Qwen/Qwen2.5-7B")
        or infer_base_model_from_dir(run_dir)
        or "unknown"
    )
    model_name = build_model_name(base_model, run_type)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # --- Params ---
        params = params_from_config(config)
        if not params and adapter_cfg:
            params = params_from_adapter_config(adapter_cfg)
        params["run_type"] = run_type
        params["base_model"] = base_model
        if n_samples is not None:
            params["training_samples"] = str(n_samples)
        if gpu_data.get("dataset"):
            params.setdefault("dataset_path", gpu_data["dataset"])
        mlflow.log_params(params)

        # --- Tags ---
        mlflow.set_tags({
            "run_name": run_name,
            "run_type": run_type,
            "base_model": base_model,
            "model_registry_name": model_name,
        })

        # --- Time-series metrics ---
        log_history = state.get("log_history", [])
        if log_history:
            log_time_series_metrics(log_history)

        # --- Summary metrics ---
        log_gpu_summary(gpu_data)
        log_trainer_summary(state)

        # --- Artifacts ---
        for artifact in ["training_config.yaml", "gpu_metrics.json", "adapter_config.json"]:
            p = run_dir / artifact
            if p.exists():
                mlflow.log_artifact(str(p))

        log_training_data_html(run_dir)

        run_id = run.info.run_id

    # # --- Register model version ---
    # try:
    #     client.create_registered_model(model_name)
    # except mlflow.exceptions.MlflowException:
    #     pass  # already exists

    # mv = client.create_model_version(
    #     name=model_name,
    #     source=f"runs:/{run_id}/",
    #     run_id=run_id,
    #     description=f"Fine-tuning run: {run_name}",
    # )
    # client.set_model_version_tag(model_name, mv.version, "run_name", run_name)
    # client.set_model_version_tag(model_name, mv.version, "run_type", run_type)
    # if gpu_data:
    #     client.set_model_version_tag(model_name, mv.version, "duration_s", str(gpu_data.get("duration_s", "")))
    #     client.set_model_version_tag(model_name, mv.version, "total_energy_kwh", str(gpu_data.get("total_energy_kwh", "")))
    # if log_history:
    #     train_steps = [e for e in log_history if "loss" in e]
    #     if train_steps:
    #         client.set_model_version_tag(model_name, mv.version, "final_train_loss", f"{train_steps[-1]['loss']:.4f}")

    # print(f"  {run_name} → {model_name} v{mv.version} (run_id={run_id[:8]})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import fine-tuning runs into MLflow.")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Delete all existing runs before importing.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        existing_run_names = set()
    else:
        experiment_id = experiment.experiment_id
        if args.force:
            runs = client.search_runs(experiment_id)
            for run in runs:
                client.delete_run(run.info.run_id)
            print(f"Deleted {len(runs)} existing run(s).\n")
            existing_run_names = set()
        else:
            existing_run_names = {r.info.run_name for r in client.search_runs(experiment_id)}

    run_dirs = sorted(d for d in OUTPUTS_DIR.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No run directories found in {OUTPUTS_DIR}")
        return

    print(f"Importing {len(run_dirs)} run(s) into experiment '{EXPERIMENT_NAME}'...\n")
    imported = 0
    for run_dir in run_dirs:
        import_run(run_dir, experiment_id, client, existing_run_names)
        imported += 1

    print(f"\nDone. Imported {imported} runs into MLflow Model Registry.")


if __name__ == "__main__":
    main()
