"""Shared utilities for evaluation scripts."""

import numpy as np


def print_stats(label, values):
    if not values:
        print(f"  {label:20s}: no data")
        return
    arr = np.array(values)
    print(f"  {label:20s}: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  std={arr.std():.4f}")


def print_buckets(label, values, thresholds=(0.05, 0.1, 0.2, 0.5, 1.0)):
    if not values:
        return
    n = len(values)
    for t in thresholds:
        count = sum(1 for v in values if v < t)
        print(f"    {label} < {t}m: {count}/{n} ({100*count/n:.1f}%)")
