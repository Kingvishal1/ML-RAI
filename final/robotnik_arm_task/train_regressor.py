#!/usr/bin/env python3
"""Train a behaviour-cloning MLP regressor on expert trajectory data."""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def build_mlp(input_dim, output_dim, hidden_dim, num_layers):
    layers = []
    prev = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU())
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def split_by_trajectory(inputs, labels, metadata, val_frac=0.2, seed=42):
    """Split into train/val by trajectory ID to avoid leakage."""
    traj_ids = np.unique(metadata[:, 0])
    rng = np.random.RandomState(seed)
    rng.shuffle(traj_ids)
    split = int(len(traj_ids) * (1 - val_frac))
    train_ids = set(traj_ids[:split].tolist())

    train_mask = np.array([m in train_ids for m in metadata[:, 0]])
    val_mask = ~train_mask

    return (inputs[train_mask], labels[train_mask],
            inputs[val_mask], labels[val_mask])


def main():
    parser = argparse.ArgumentParser(description="Train BC regressor")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--out", type=str, default="bc_policy.pt", help="Checkpoint save path")
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal (MPS) GPU if available")
    args = parser.parse_args()

    # --- Load data ---
    data = np.load(args.data)
    inputs, labels, metadata = data["inputs"], data["labels"], data["metadata"]
    print(f"Loaded {inputs.shape[0]} samples  inputs={inputs.shape}  labels={labels.shape}")

    # --- Split by trajectory ---
    X_train, y_train, X_val, y_val = split_by_trajectory(inputs, labels, metadata)
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}")

    # --- Normalisation (input only) ---
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # --- Torch datasets ---
    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # --- Model ---
    model = build_mlp(inputs.shape[1], labels.shape[1],
                      args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    print(f"Model: {sum(p.numel() for p in model.parameters())} params  device={device}")

    # --- Training ---
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # Val
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
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_mean": mean,
                "input_std": std,
                "input_dim": inputs.shape[1],
                "output_dim": labels.shape[1],
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
            }, args.out)
            tag = "  *saved*"

        print(f"Epoch {epoch:4d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{tag}")

    print(f"\nBest val loss: {best_val:.6f}  Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
