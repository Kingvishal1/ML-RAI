#!/usr/bin/env python3
"""Merge multiple .npz datasets into one.

Usage:
  python merge_datasets.py -o merged_arm.npz \
    data/dataset_arm.npz dagger_p1_arm.npz dagger_p2_arm.npz dagger_p3_arm.npz dagger_p4_arm.npz
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge .npz datasets")
    parser.add_argument("files", nargs="+", help="Input .npz files to merge")
    parser.add_argument("-o", "--out", required=True, help="Output .npz file")
    args = parser.parse_args()

    all_inputs = []
    all_labels = []
    all_metadata = []
    feature_names = None

    for path in args.files:
        d = np.load(path, allow_pickle=True)
        inputs = d["inputs"]
        labels = d["labels"]
        print(f"  {path}: {inputs.shape[0]} samples, {inputs.shape[1]} features, {labels.shape[1]} labels")
        all_inputs.append(inputs)
        all_labels.append(labels)

        if "metadata" in d:
            all_metadata.append(d["metadata"])
        else:
            # Fill with zeros if missing (target_id=0, step=0..n)
            all_metadata.append(np.zeros((inputs.shape[0], 2), dtype=np.float64))

        if "feature_names" in d and feature_names is None:
            feature_names = d["feature_names"]

    merged_inputs = np.concatenate(all_inputs, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_metadata = np.concatenate(all_metadata, axis=0)

    save_dict = {
        "inputs": merged_inputs,
        "labels": merged_labels,
        "metadata": merged_metadata,
    }
    if feature_names is not None:
        save_dict["feature_names"] = feature_names

    np.savez(args.out, **save_dict)
    print(f"\nMerged: {merged_inputs.shape[0]} samples -> {args.out}")


if __name__ == "__main__":
    main()
