#!/usr/bin/env python3
"""
Patch axolotl YAML config fields. 
Usage: patch_config.py <src> <dst> --dataset PATH --output-dir DIR [--eval-dataset PATH] [--train-size N]

Author: Adam Zvara (xzvara01)
Date: 03/2026
"""
import argparse, shutil, sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-dataset", default=None,
                        help="Static eval dataset path. When set, used as the fixed test set instead of splitting from train.")
    parser.add_argument("--train-size", default=None,
                        help="If set, use first N samples or N%% of samples for train (eval unaffected when --eval-dataset is given)")
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

    if args.train_size is not None:
        cfg["datasets"][0]["split"] = f"train[:{args.train_size}]"

    if args.eval_dataset is not None:
        # Static eval set: point test_datasets at the pre-split file, disable val_set_size splitting
        cfg["test_datasets"] = [{
            "path": args.eval_dataset,
            "type": cfg["datasets"][0].get("type", "alpaca"),
            "split": "train",
        }]
        cfg["val_set_size"] = 0
    elif args.train_size is not None:
        # Legacy behaviour: slice eval from the same file
        cfg["test_datasets"] = [{
            "path": args.dataset,
            "type": cfg["datasets"][0].get("type", "alpaca"),
            "split": f"train[{args.train_size}:]",
        }]
        cfg["val_set_size"] = 0

    shutil.copy2(args.src, args.dst)  # preserve permissions
    with open(args.dst, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    main()
