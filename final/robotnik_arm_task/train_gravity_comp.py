#!/usr/bin/env python3
"""Train a small MLP to predict gravity compensation torques from joint angles.

Gravity torques are a deterministic function of q_arm (7 joint angles),
so a small network learns this mapping near-perfectly.

Usage:
    python train_gravity_comp.py --data data/dataset_20261027_143306.npz --mps
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from activations import ACTIVATIONS, DEFAULT_ACTIVATION, get_activation


def main():
    parser = argparse.ArgumentParser(description="Train gravity compensation network")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to full .npz dataset (44-dim inputs)")
    parser.add_argument("--out", type=str, default="gravity_comp.pt",
                        help="Output checkpoint path")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, choices=list(ACTIVATIONS),
                        help=f"Hidden-layer activation function (default: {DEFAULT_ACTIVATION})")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--max_samples", type=int, default=500000,
                        help="Subsample to this many rows (gravity is deterministic, don't need millions)")
    args = parser.parse_args()

    # Load data
    data = np.load(args.data)
    inputs = data["inputs"]
    print(f"Loaded {inputs.shape[0]} samples  inputs={inputs.shape}")

    # Extract q_arm (cols 17-23) and tau_gravity (cols 37-43)
    q_arm = inputs[:, 17:24]    # 7 joint angles
    tau_grav = inputs[:, 37:44]  # 7 gravity torques
    print(f"q_arm shape: {q_arm.shape}  tau_gravity shape: {tau_grav.shape}")

    # Subsample — gravity is a deterministic function of q_arm,
    # so duplicate configurations don't help
    if len(q_arm) > args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(q_arm), args.max_samples, replace=False)
        q_arm = q_arm[idx]
        tau_grav = tau_grav[idx]
        print(f"Subsampled to {len(q_arm)} rows")

    # Train/val split (80/20)
    n = len(q_arm)
    perm = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = q_arm[train_idx], tau_grav[train_idx]
    X_val, y_val = q_arm[val_idx], tau_grav[val_idx]

    # Normalize inputs
    q_mean = X_train.mean(axis=0)
    q_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - q_mean) / q_std
    X_val = (X_val - q_mean) / q_std

    # Normalize outputs
    tau_mean = y_train.mean(axis=0)
    tau_std = y_train.std(axis=0) + 1e-8
    y_train = (y_train - tau_mean) / tau_std
    y_val = (y_val - tau_mean) / tau_std

    print(f"Train: {len(X_train)}  Val: {len(X_val)}")
    print(f"tau_gravity mean: {np.round(tau_mean, 2)}")
    print(f"tau_gravity std:  {np.round(tau_std, 2)}")

    # Device
    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Build model
    layers = []
    prev = 7
    for _ in range(args.num_layers):
        layers.append(nn.Linear(prev, args.hidden_dim))
        layers.append(get_activation(args.activation))
        prev = args.hidden_dim
    layers.append(nn.Linear(prev, 7))
    model = nn.Sequential(*layers).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params  device={device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=(device.type != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            pin_memory=(device.type != "cpu"))

    best_val = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " *"

        if epoch % 5 == 0 or epoch == 1 or tag:
            print(f"  epoch {epoch}/{args.epochs}  val_loss={val_loss:.6f}  best={best_val:.6f}{tag}")

    elapsed = time.time() - t0

    # Compute final metrics in original scale
    model.load_state_dict(best_state)
    model.eval()
    model.to(device)
    all_pred = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            all_pred.append(model(xb).cpu().numpy())
    all_pred = np.concatenate(all_pred)
    # Un-normalize
    pred_orig = all_pred * tau_std + tau_mean
    true_orig = y_val * tau_std + tau_mean
    abs_err = np.abs(pred_orig - true_orig)
    print(f"\nFinal metrics (original Nm scale):")
    print(f"  MAE per joint: {np.round(abs_err.mean(axis=0), 4)}")
    print(f"  Max per joint: {np.round(abs_err.max(axis=0), 4)}")
    print(f"  Overall MAE: {abs_err.mean():.4f} Nm")
    print(f"  Overall max: {abs_err.max():.4f} Nm")

    # Save
    torch.save({
        "model_state_dict": best_state,
        "q_mean": q_mean,
        "q_std": q_std,
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "activation": args.activation,
        "best_val_loss": best_val,
        "elapsed_sec": elapsed,
    }, args.out)
    print(f"\nSaved to {args.out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
