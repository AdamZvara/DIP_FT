#!/usr/bin/env python3
"""
Split a JSONL dataset into stratified train/test sets.

Stratification is based on provenance.source_dataset (or "synthetic" when null).
Test samples are guaranteed not to appear in the train set.

Usage:
    python data/split_dataset.py --test-size 200
    python data/split_dataset.py --input data/auth_full.jsonl --test-size 200 --output-dir data/ --seed 42

Author: Adam Zvara (xzvara01)
Date: 04/2026
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict


def source_key(record):
    prov = record.get("provenance", {})
    src = prov.get("source_dataset")
    return src if src else "synthetic"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno}: {e}")
    return records


def write_jsonl(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def allocate_stratified(groups, test_size):
    """
    Given groups (dict source -> list of records) and desired test_size,
    return per-source test counts that sum to exactly test_size,
    preserving proportions as closely as possible.
    """
    total = sum(len(v) for v in groups.values())
    if test_size > total:
        raise ValueError(f"test_size ({test_size}) exceeds total records ({total})")

    # Floor allocation
    alloc = {}
    for src, recs in groups.items():
        proportion = len(recs) / total
        alloc[src] = math.floor(proportion * test_size)

    # Validate each source has enough records
    for src, count in alloc.items():
        if count > len(groups[src]):
            raise ValueError(
                f"Source '{src}' has only {len(groups[src])} records but needs {count} for test set"
            )

    # Distribute remainder to largest groups first
    remainder = test_size - sum(alloc.values())
    sorted_sources = sorted(groups.keys(), key=lambda s: len(groups[s]), reverse=True)
    for src in sorted_sources:
        if remainder == 0:
            break
        if alloc[src] < len(groups[src]):
            alloc[src] += 1
            remainder -= 1

    return alloc


def print_summary(groups, test_sets, train_sets):
    total = sum(len(v) for v in groups.values())
    total_test = sum(len(v) for v in test_sets.values())
    total_train = sum(len(v) for v in train_sets.values())

    col_w = max(len(s) for s in groups) + 2
    print(f"\n{'Source':<{col_w}} {'Full':>8} {'Full%':>7}  {'Test':>6} {'Test%':>7}  {'Train':>7} {'Train%':>7}")
    print("-" * (col_w + 50))
    for src in sorted(groups.keys()):
        n_full = len(groups[src])
        n_test = len(test_sets.get(src, []))
        n_train = len(train_sets.get(src, []))
        print(
            f"{src:<{col_w}} {n_full:>8} {n_full/total*100:>6.1f}%"
            f"  {n_test:>6} {n_test/total_test*100 if total_test else 0:>6.1f}%"
            f"  {n_train:>7} {n_train/total_train*100 if total_train else 0:>6.1f}%"
        )
    print("-" * (col_w + 50))
    print(
        f"{'TOTAL':<{col_w}} {total:>8} {'100.0%':>7}  {total_test:>6} {'100.0%':>7}  {total_train:>7} {'100.0%':>7}"
    )
    print()


def main():
    parser = argparse.ArgumentParser(description="Stratified train/test split for JSONL datasets")
    parser.add_argument("--input", default="data/auth_full.jsonl", help="Source JSONL file")
    parser.add_argument("--test-size", type=int, required=True, help="Number of samples in the test set")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    print(f"Loading {args.input}...")
    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records.")

    # Group by source
    groups = defaultdict(list)
    for rec in records:
        groups[source_key(rec)].append(rec)

    # Allocate test counts per source
    alloc = allocate_stratified(groups, args.test_size)

    # Sample test set per source; remainder goes to train
    test_sets = {}
    train_sets = {}
    for src, recs in groups.items():
        shuffled = recs[:]
        random.shuffle(shuffled)
        n_test = alloc[src]
        test_sets[src] = shuffled[:n_test]
        train_sets[src] = shuffled[n_test:]

    # Flatten and shuffle final sets
    test_records = [r for recs in test_sets.values() for r in recs]
    train_records = [r for recs in train_sets.values() for r in recs]
    random.shuffle(test_records)
    random.shuffle(train_records)

    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)

    print_summary(groups, test_sets, train_sets)
    print(f"Train: {len(train_records)} records -> {train_path}")
    print(f"Test:  {len(test_records)} records -> {test_path}")


if __name__ == "__main__":
    main()
