#!/usr/bin/env python3
"""
Randomly sample N items from a .jsonl dataset and write to NAME_N.jsonl.

Author: Adam Zvara (xzvara01)
Date: 04/2026
"""

import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample N items from a .jsonl file."
    )
    parser.add_argument("dataset", help="Path to input .jsonl file")
    parser.add_argument("n", type=int, help="Number of items to sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.dataset)
    if not input_path.exists():
        parser.error(f"File not found: {input_path}")

    with open(input_path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.n > len(items):
        parser.error(f"Requested {args.n} items but dataset only has {len(items)}")

    rng = random.Random(args.seed)
    sampled = rng.sample(items, args.n)

    output_path = input_path.parent / f"{input_path.stem}_{args.n}.jsonl"
    with open(output_path, "w") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Sampled {args.n}/{len(items)} items → {output_path}")


if __name__ == "__main__":
    main()