#!/usr/bin/env python3
"""Train a behaviour-cloning MLP regressor on expert trajectory data."""

import argparse
import glob
import os
import re
import time as _time

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
    parser.add_argument("--mode", type=str, default="full", choices=["full", "base", "arm"],
                        help="Metric display mode: full (4+7 breakdown), base (wheel only), arm (torque only)")
    parser.add_argument("--early_stop", type=float, default=0.001,
                        help="Stop if val loss improvement is below this for --patience epochs (default: 0.005)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs without improvement before early stopping (default: 5)")
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

    # --- Normalisation (input + output) ---
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Compute per-action label stats (ACTION_DIM), then tile for chunked labels (K*ACTION_DIM)
    label_mean_single = y_train[:, :ACTION_DIM].mean(axis=0)
    label_std_single = y_train[:, :ACTION_DIM].std(axis=0) + 1e-8
    label_mean = np.tile(label_mean_single, K)
    label_std = np.tile(label_std_single, K)
    y_train = (y_train - label_mean) / label_std
    y_val = (y_val - label_mean) / label_std
    print(f"Label stats: mean={np.round(label_mean_single, 4)}  std={np.round(label_std_single, 4)}")

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

    # --- Label slices depend on mode ---
    mode = args.mode
    if mode == "full":
        # 4 wheel + 7 arm torque per action step
        wheel_cols = np.concatenate([np.arange(k * ACTION_DIM, k * ACTION_DIM + 4) for k in range(K)])
        n_torque = 7
        torque_offset = 4
    elif mode == "base":
        # All outputs are wheel commands (ACTION_DIM == 4)
        wheel_cols = np.concatenate([np.arange(k * ACTION_DIM, k * ACTION_DIM + ACTION_DIM) for k in range(K)])
        n_torque = 0
        torque_offset = ACTION_DIM
    elif mode == "arm":
        # All outputs are arm torques (ACTION_DIM == 7)
        wheel_cols = np.array([], dtype=int)
        n_torque = ACTION_DIM
        torque_offset = 0
    arm_names = [f"arm_{i}" for i in range(1, n_torque + 1)]

    # --- Checkpoint path (fixed once at start) ---
    if args.out:
        save_path = args.out
    else:
        # Auto-increment counter from existing .pt files in cwd
        existing = glob.glob("*.pt")
        counters = [int(m.group(1)) for f in existing
                    if (m := re.match(r"^(\d+)_", f))]
        next_id = max(counters, default=0) + 1
        save_path = (f"{next_id:03d}_bc_{mode}_e{args.epochs}_h{args.hidden_dim}"
                     f"_l{args.num_layers}_c{K}.pt")
    print(f"Checkpoint: {save_path}")

    # --- Training ---
    best_val = float("inf")
    epochs_without_improvement = 0
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

        # Val — collect all predictions for detailed metrics (in original scale)
        model.eval()
        val_loss = 0.0
        all_pred_orig = []
        all_label_orig = []
        n_val = len(val_ds)
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
                # Un-scale to original units
                all_pred_orig.append(pred.cpu().numpy() * label_std + label_mean)
                all_label_orig.append(yb.cpu().numpy() * label_std + label_mean)

        val_loss /= n_val
        all_pred_orig = np.concatenate(all_pred_orig)
        all_label_orig = np.concatenate(all_label_orig)
        err_orig = all_pred_orig - all_label_orig  # signed error in original scale

        # --- Per-component MSE (existing) ---
        val_se_wheel = 0.0
        val_se_torque = np.zeros(max(n_torque, 1))
        if len(wheel_cols) > 0:
            val_se_wheel = (err_orig[:, wheel_cols] ** 2).sum()
        for j in range(n_torque):
            cols_j = np.array([k * ACTION_DIM + torque_offset + j for k in range(K)])
            val_se_torque[j] = (err_orig[:, cols_j] ** 2).sum()
        val_mse_wheel = val_se_wheel / (n_val * K) if len(wheel_cols) > 0 else 0.0
        val_mse_torque = val_se_torque[:n_torque] / (n_val * K) if n_torque > 0 else np.array([])
        val_mse_torque_avg = val_mse_torque.mean() if n_torque > 0 else 0.0

        # --- Additional metrics (first action step only for readability) ---
        pred_a0 = all_pred_orig[:, :ACTION_DIM]
        label_a0 = all_label_orig[:, :ACTION_DIM]
        err_a0 = pred_a0 - label_a0
        abs_err_a0 = np.abs(err_a0)

        mae_per_dim = abs_err_a0.mean(axis=0)
        max_err_per_dim = abs_err_a0.max(axis=0)
        p90_per_dim = np.percentile(abs_err_a0, 90, axis=0)
        p99_per_dim = np.percentile(abs_err_a0, 99, axis=0)

        # Sign accuracy: % of samples where pred and label have the same sign
        # (only meaningful where label != 0)
        nonzero_mask = np.abs(label_a0) > 1e-6
        sign_match = np.sign(pred_a0) == np.sign(label_a0)
        sign_acc_per_dim = np.array([
            sign_match[nonzero_mask[:, d], d].mean() * 100 if nonzero_mask[:, d].any() else 100.0
            for d in range(ACTION_DIM)
        ])

        # Near-zero accuracy: when expert outputs ~0, does model also output ~0?
        near_zero_thresh = 0.02
        expert_near_zero = np.abs(label_a0) < near_zero_thresh
        pred_near_zero = np.abs(pred_a0) < near_zero_thresh
        zero_acc_per_dim = np.array([
            pred_near_zero[expert_near_zero[:, d], d].mean() * 100 if expert_near_zero[:, d].any() else 100.0
            for d in range(ACTION_DIM)
        ])

        tag = ""
        if val_loss < best_val - args.early_stop:
            epochs_without_improvement = 0
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            elapsed = _time.time() - t_start
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_mean": mean,
                "input_std": std,
                "label_mean": label_mean_single,
                "label_std": label_std_single,
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
        if mode == "full":
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"wheel={val_mse_wheel:.6f}  torque_avg={val_mse_torque_avg:.6f}{tag}")
            arm_str = "  ".join(f"{arm_names[j]}={val_mse_torque[j]:.6f}" for j in range(n_torque))
            print(f"         torque: {arm_str}")
        elif mode == "base":
            dim_names = ["v_cmd", "w_cmd"]
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"wheel={val_mse_wheel:.6f}{tag}")
        elif mode == "arm":
            dim_names = [f"arm_{i}" for i in range(1, ACTION_DIM + 1)]
            arm_str = "  ".join(f"{arm_names[j]}={val_mse_torque[j]:.6f}" for j in range(n_torque))
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"torque_avg={val_mse_torque_avg:.6f}{tag}")
            print(f"         torque: {arm_str}")

        # Detailed metrics (every 5 epochs or on save)
        if epoch % 5 == 0 or tag:
            if mode in ("base", "arm"):
                names = dim_names
            else:
                names = [f"w{i}" for i in range(1, 5)] + [f"a{i}" for i in range(1, 8)]
            mae_s = "  ".join(f"{names[d]}={mae_per_dim[d]:.4f}" for d in range(ACTION_DIM))
            max_s = "  ".join(f"{names[d]}={max_err_per_dim[d]:.4f}" for d in range(ACTION_DIM))
            p90_s = "  ".join(f"{names[d]}={p90_per_dim[d]:.4f}" for d in range(ACTION_DIM))
            sign_s = "  ".join(f"{names[d]}={sign_acc_per_dim[d]:.1f}%" for d in range(ACTION_DIM))
            zero_s = "  ".join(f"{names[d]}={zero_acc_per_dim[d]:.1f}%" for d in range(ACTION_DIM))
            print(f"         MAE:      {mae_s}")
            print(f"         max_err:  {max_s}")
            print(f"         p90:      {p90_s}")
            print(f"         sign_acc: {sign_s}")
            print(f"         zero_acc: {zero_s}")

        # Early stopping
        if val_loss >= best_val - args.early_stop:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement > {args.early_stop} for {args.patience} epochs")
            break

    total_time = _time.time() - t_start
    total_min = int(total_time // 60)
    final_path = save_path.replace(".pt", f"_{total_min}min.pt")
    if os.path.exists(save_path):
        os.rename(save_path, final_path)
    print(f"\nBest val loss: {best_val:.6f} at epoch {best_epoch}")
    print(f"Total time: {total_time/60:.1f}min  Checkpoint: {final_path}")


if __name__ == "__main__":
    main()
