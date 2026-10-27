#!/usr/bin/env python3
"""Load a trained BC policy checkpoint and run inference on input observations."""

import argparse
import numpy as np
import torch
import torch.nn as nn


def build_mlp(input_dim, output_dim, hidden_dim, num_layers):
    layers = []
    prev = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU())
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def load_policy(checkpoint_path, device="cpu"):
    """Load model + normalization stats from a checkpoint.

    Returns:
        model, mean, std, chunk_size, label_mean, label_std
        chunk_size defaults to 1 for older checkpoints without it.
        label_mean/label_std default to 0/1 (no-op) for older checkpoints.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_mlp(ckpt["input_dim"], ckpt["output_dim"],
                      ckpt["hidden_dim"], ckpt["num_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    mean = ckpt["input_mean"]
    std = ckpt["input_std"]
    chunk_size = ckpt.get("chunk_size", 1)
    # Backward compat: older checkpoints without label normalization
    label_mean = ckpt.get("label_mean", np.zeros(ckpt["output_dim"]))
    label_std = ckpt.get("label_std", np.ones(ckpt["output_dim"]))
    return model, mean, std, chunk_size, label_mean, label_std


def predict(model, mean, std, obs, device="cpu", chunk_size=1,
            label_mean=None, label_std=None):
    """Run inference on a single observation or batch.

    Args:
        obs: numpy array of shape (D,) or (N, D)
        chunk_size: if > 1, reshape output to (K, action_dim) per sample
        label_mean, label_std: if provided, un-normalize the output
    Returns:
        chunk_size == 1: numpy array of shape (action_dim,) or (N, action_dim)
        chunk_size  > 1 (single obs): numpy array of shape (K, action_dim)
        chunk_size  > 1 (batch):      numpy array of shape (N, K, action_dim)
    """
    single = obs.ndim == 1
    if single:
        obs = obs[np.newaxis, :]

    obs_norm = (obs - mean) / std
    x = torch.tensor(obs_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        action = model(x).cpu().numpy()

    # Un-normalize labels back to original scale
    if label_mean is not None and label_std is not None:
        # action shape is (N, K*action_dim) — tile stats for chunked outputs
        action_dim = len(label_mean)
        K = action.shape[1] // action_dim
        lm = np.tile(label_mean, K)
        ls = np.tile(label_std, K)
        action = action * ls + lm

    if chunk_size > 1:
        action = action.reshape(action.shape[0], chunk_size, -1)

    return action[0] if single else action


def main():
    parser = argparse.ArgumentParser(description="Run trained BC policy")
    parser.add_argument("--checkpoint", type=str, default="bc_policy_100.pt", help="Path to .pt checkpoint")
    parser.add_argument("--data", type=str, help="Optional: .npz dataset to run policy on (uses val set)")
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal (MPS) GPU")
    args = parser.parse_args()

    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, mean, std, chunk_size, label_mean, label_std = load_policy(args.checkpoint, device=str(device))
    print(f"Loaded policy from {args.checkpoint}  device={device}  chunk_size={chunk_size}")

    if args.data:
        # Run on dataset and print sample predictions vs ground truth
        data = np.load(args.data)
        inputs, labels = data["inputs"], data["labels"]

        # Pick 5 random samples
        rng = np.random.RandomState(0)
        idxs = rng.choice(len(inputs), size=min(100, len(inputs)), replace=False)

        preds = predict(model, mean, std, inputs[idxs], device=str(device),
                        label_mean=label_mean, label_std=label_std)
        abs_err = np.abs(preds - labels[idxs])  # (N, 11)

        for i, idx in enumerate(idxs):
            print(f"\nSample {idx}:")
            print(f"  pred:  wheel={preds[i, :4].round(4)}  torque={preds[i, 4:].round(4)}")
            print(f"  true:  wheel={labels[idx, :4].round(4)}  torque={labels[idx, 4:].round(4)}")
            print(f"  error: wheel={abs_err[i, :4].round(4)}  "
                  f"torque={abs_err[i, 4:].round(4)}")

        # --- Summary ---
        component_names = ([f"wheel_{i}" for i in range(1, 5)]
                           + [f"torque_arm_{i}" for i in range(1, 8)])
        mean_err = abs_err.mean(axis=0)
        max_err = abs_err.max(axis=0)
        worst_idx = abs_err.argmax(axis=0)  # which sample had the worst error

        print("\n" + "=" * 70)
        print("SUMMARY  (over {} samples)".format(len(idxs)))
        print("=" * 70)
        print(f"{'component':<16} {'mean_abs_err':>12} {'max_abs_err':>12} {'worst_sample':>14}")
        print("-" * 56)
        for j, name in enumerate(component_names):
            print(f"{name:<16} {mean_err[j]:>12.4f} {max_err[j]:>12.4f} {idxs[worst_idx[j]]:>14}")

        print("-" * 56)
        print(f"{'wheel (avg)':<16} {mean_err[:4].mean():>12.4f} {max_err[:4].max():>12.4f}")
        print(f"{'torque (avg)':<16} {mean_err[4:].mean():>12.4f} {max_err[4:].max():>12.4f}")
        print(f"{'overall':<16} {mean_err.mean():>12.4f} {max_err.max():>12.4f}")
    else:
        # Demo with a zero observation
        obs = np.zeros(37)
        action = predict(model, mean, std, obs, device=str(device),
                         label_mean=label_mean, label_std=label_std)
        print(f"Demo (zero input):")
        print(f"  wheel_cmds: {action[:4].round(4)}")
        print(f"  torque_arm: {action[4:].round(4)}")
        print("\nUsage in code:")
        print("  from run_policy import load_policy, predict")
        print("  model, mean, std = load_policy('bc_policy_100.pt')")
        print("  action = predict(model, mean, std, obs)  # obs shape (37,)")


if __name__ == "__main__":
    main()
