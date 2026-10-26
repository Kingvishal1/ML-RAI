#!/usr/bin/env python3
"""Train a behaviour-cloning MLP regressor on expert trajectory data."""

import argparse
import time as _time
from datetime import datetime

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
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--out", type=str, default=None,
                        help="Checkpoint save path (default: auto-generated from params)")
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal (MPS) GPU if available")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes for parallel data loading (default: 0)")
    parser.add_argument("--chunk_size", type=int, default=1,
                        help="Number of future actions to predict per observation (default: 1)")
    args = parser.parse_args()

    # --- Load data ---
    data = np.load(args.data)
    inputs, labels, metadata = data["inputs"], data["labels"], data["metadata"]
    print(f"Loaded {inputs.shape[0]} samples  inputs={inputs.shape}  labels={labels.shape}")

    # --- Build action chunks ---
    K = args.chunk_size
    ACTION_DIM = labels.shape[1]  # 11
    if K > 1:
        traj_ids = np.unique(metadata[:, 0])
        chunked_inputs, chunked_labels, chunked_meta = [], [], []
        for tid in traj_ids:
            mask = metadata[:, 0] == tid
            traj_inputs = inputs[mask]
            traj_labels = labels[mask]
            traj_meta = metadata[mask]
            # Sort by timestep
            order = np.argsort(traj_meta[:, 1])
            traj_inputs = traj_inputs[order]
            traj_labels = traj_labels[order]
            traj_meta = traj_meta[order]
            T = len(traj_inputs)
            for t in range(T - K + 1):
                chunked_inputs.append(traj_inputs[t])
                chunked_labels.append(traj_labels[t:t + K].flatten())
                chunked_meta.append(traj_meta[t])
        inputs = np.array(chunked_inputs)
        labels = np.array(chunked_labels)
        metadata = np.array(chunked_meta)
        print(f"Chunked (K={K}): {inputs.shape[0]} samples  labels={labels.shape}")

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
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type != "cpu"), persistent_workers=(args.num_workers > 0))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)

    # --- Model ---
    model = build_mlp(inputs.shape[1], labels.shape[1],
                      args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    print(f"Model: {sum(p.numel() for p in model.parameters())} params  device={device}")

    # --- Label slices: wheel_cmd_1..4 (0:4), torque_arm_1..7 (4:11) per action step ---
    # With chunking, labels are [action_0, action_1, ..., action_{K-1}] each of dim 11
    wheel_cols = np.concatenate([np.arange(k * ACTION_DIM, k * ACTION_DIM + 4) for k in range(K)])
    torque_cols = [np.arange(k * ACTION_DIM + 4, k * ACTION_DIM + 11) for k in range(K)]
    arm_names = [f"arm_{i}" for i in range(1, 8)]

    # --- Checkpoint path (fixed once at start) ---
    if args.out:
        save_path = args.out
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = (f"bc_e{args.epochs}_h{args.hidden_dim}_l{args.num_layers}"
                     f"_w{args.num_workers}_c{K}_{timestamp}.pt")
    print(f"Checkpoint: {save_path}")

    # --- Training ---
    best_val = float("inf")
    t_start = _time.time()
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

        # Val — collect per-component errors
        model.eval()
        val_loss = 0.0
        val_se_wheel = 0.0       # sum of squared errors for wheel cmds
        val_se_torque = np.zeros(7)  # per-arm torque
        n_val = len(val_ds)
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                bs = xb.size(0)
                val_loss += criterion(pred, yb).item() * bs
                err = (pred - yb).cpu().numpy()
                val_se_wheel += (err[:, wheel_cols] ** 2).sum()
                for j in range(7):
                    cols_j = np.array([k * ACTION_DIM + 4 + j for k in range(K)])
                    val_se_torque[j] += (err[:, cols_j] ** 2).sum()

        val_loss /= n_val
        val_mse_wheel = val_se_wheel / (n_val * K)
        val_mse_torque = val_se_torque / (n_val * K)
        val_mse_torque_avg = val_mse_torque.mean()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            elapsed = _time.time() - t_start
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_mean": mean,
                "input_std": std,
                "input_dim": inputs.shape[1],
                "output_dim": labels.shape[1],
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "epochs": args.epochs,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_workers": args.num_workers,
                "device": str(device),
                "elapsed_sec": elapsed,
                "chunk_size": K,
                "action_dim": ACTION_DIM,
            }, save_path)
            tag = f"  *saved* → {save_path}"

        # Summary line
        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"wheel={val_mse_wheel:.6f}  torque_avg={val_mse_torque_avg:.6f}{tag}")
        # Per-arm breakdown
        arm_str = "  ".join(f"{arm_names[j]}={val_mse_torque[j]:.6f}" for j in range(7))
        print(f"         torque: {arm_str}")

    total_time = _time.time() - t_start
    total_min = int(total_time // 60)
    final_path = save_path.replace(".pt", f"_{total_min}min.pt")
    import os
    if os.path.exists(save_path):
        os.rename(save_path, final_path)
    print(f"\nBest val loss: {best_val:.6f} at epoch {best_epoch}")
    print(f"Total time: {total_time/60:.1f}min  Checkpoint: {final_path}")


if __name__ == "__main__":
    main()
