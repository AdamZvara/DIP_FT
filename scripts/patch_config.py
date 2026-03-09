#!/usr/bin/env python3
"""Patch axolotl YAML config fields. Usage: patch_config.py <src> <dst> --dataset PATH --output-dir DIR"""
import argparse, shutil, sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: PyYAML not installed")

    with open(args.src) as f:
        cfg = yaml.safe_load(f)

    cfg["datasets"][0]["path"] = args.dataset
    cfg["output_dir"] = args.output_dir
    if "dataset_prepared_path" in cfg:
        cfg["dataset_prepared_path"] = args.output_dir + "/prepared"

    shutil.copy2(args.src, args.dst)  # preserve permissions
    with open(args.dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    main()
