#!/usr/bin/env python3
"""Background GPU monitor. Writes JSON metrics on SIGTERM/SIGINT."""
import argparse
import json
import signal
import sys
import time
from typing import Any, Dict, List

try:
    import pynvml
except ImportError:
    sys.exit("ERROR: pynvml not installed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--interval", type=int, default=10, help="Sample interval in seconds")
    p.add_argument("--job-id", default="unknown")
    p.add_argument("--config", default="unknown")
    p.add_argument("--dataset", default="unknown")
    return p.parse_args()


def collect_sample(handles: List[Any]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for handle in handles:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem.used / (1024 ** 3)
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except pynvml.NVMLError:
            power_w = None
        samples.append({"vram_gb": vram_gb, "power_w": power_w})
    return samples


def write_output(
    path: str,
    job_id: str,
    config: str,
    dataset: str,
    start_time: float,
    end_time: float,
    per_gpu_samples: List[List[Dict[str, Any]]],
) -> None:
    duration_s = end_time - start_time
    gpu_summaries: List[Dict[str, Any]] = []

    for gpu_idx, samples in enumerate(per_gpu_samples):
        if not samples:
            continue
        peak_vram = max(s["vram_gb"] for s in samples)
        power_readings = [s["power_w"] for s in samples if s["power_w"] is not None]
        avg_power = sum(power_readings) / len(power_readings) if power_readings else None
        gpu_summaries.append(
            {
                "gpu_index": gpu_idx,
                "peak_vram_gb": round(peak_vram, 3),
                "avg_power_w": round(avg_power, 2) if avg_power is not None else None,
            }
        )

    total_avg_power = sum(
        g["avg_power_w"] for g in gpu_summaries if g["avg_power_w"] is not None
    )
    total_energy_kwh = (total_avg_power * duration_s) / 3_600_000

    result: Dict[str, Any] = {
        "job_id": job_id,
        "config": config,
        "dataset": dataset,
        "start_time": start_time,
        "end_time": end_time,
        "duration_s": round(duration_s, 2),
        "gpus": gpu_summaries,
        "total_energy_kwh": round(total_energy_kwh, 6),
    }

    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[gpu_monitor] Metrics written to {path}", flush=True)


def main() -> None:
    args = parse_args()
    pynvml.nvmlInit()

    num_gpus = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]
    # per_gpu_samples[i] is a list of sample dicts for GPU i
    per_gpu_samples: List[List[Dict[str, Any]]] = [[] for _ in range(num_gpus)]

    start_time = time.time()
    running = True

    def handle_signal(signum: int, frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(f"[gpu_monitor] Monitoring {num_gpus} GPU(s) every {args.interval}s", flush=True)

    while running:
        samples = collect_sample(handles)
        for i, s in enumerate(samples):
            per_gpu_samples[i].append(s)
        time.sleep(args.interval)

    end_time = time.time()
    pynvml.nvmlShutdown()

    write_output(
        path=args.output,
        job_id=args.job_id,
        config=args.config,
        dataset=args.dataset,
        start_time=start_time,
        end_time=end_time,
        per_gpu_samples=per_gpu_samples,
    )


if __name__ == "__main__":
    main()
