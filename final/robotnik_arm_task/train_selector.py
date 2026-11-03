#!/usr/bin/env python3
"""Train a target-selector MLP scorer on expert nearest-neighbor data.

The model scores individual candidate targets (3-dim input: target position in
the robot's base frame) and outputs a single scalar score.  At each decision
point the expert-chosen candidate should score highest among the remaining set.

Training uses per-decision-point cross-entropy: for each group of candidates
belonging to one decision, softmax over their scores and penalise if the
expert's choice doesn't get the highest probability.

The model is fully general — it works with any number of targets.

Usage:
    python train_selector.py --data data/dataset_selector_*.npz --epochs 100
"""
import argparse
import glob
import os
import re
import time as _time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from activations import ACTIVATIONS, DEFAULT_ACTIVATION
from regressors import build_mlp


# ---------------------------------------------------------------------------
# Dataset that yields (candidate_obs, chosen_flag, decision_id)
# ---------------------------------------------------------------------------
class SelectorDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, chosen, decision_ids):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.chosen = torch.tensor(chosen, dtype=torch.long)
        self.decision_ids = torch.tensor(decision_ids, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.chosen[idx], self.decision_ids[idx]


# ---------------------------------------------------------------------------
# Grouped cross-entropy: softmax over candidates within each decision group
# ---------------------------------------------------------------------------
def grouped_cross_entropy(scores, chosen, decision_ids):
    """Cross-entropy computed per decision group.

    Args:
        scores: (N,) scalar score per candidate
        chosen: (N,) 1 for the expert-chosen candidate, 0 for others
        decision_ids: (N,) integer grouping candidates into decision points
    Returns:
        scalar loss (mean over decisions)
    """
    unique_ids = decision_ids.unique()
    loss = torch.tensor(0.0, device=scores.device)
    correct = 0
    total = 0
    for did in unique_ids:
        mask = decision_ids == did
        group_scores = scores[mask]          # (K,)
        group_chosen = chosen[mask]          # (K,) with exactly one 1
        label = group_chosen.argmax()        # index of the chosen one within group

        # Cross-entropy: log_softmax over the group, pick the label
        log_probs = torch.nn.functional.log_softmax(group_scores, dim=0)
        loss = loss - log_probs[label]

        # Accuracy
        if group_scores.argmax() == label:
            correct += 1
        total += 1

    return loss / max(total, 1), correct, total


# ---------------------------------------------------------------------------
# Batched version: process decisions that fit in a batch
# ---------------------------------------------------------------------------
def batched_grouped_cross_entropy(model, inputs, chosen, decision_ids, device):
    """Run model and compute grouped CE for a batch of candidates."""
    inputs = inputs.to(device)
    chosen = chosen.to(device)
    decision_ids = decision_ids.to(device)
    scores = model(inputs).squeeze(-1)  # (N,)
    return grouped_cross_entropy(scores, chosen, decision_ids)


def split_by_episode(metadata, val_frac=0.2, seed=42):
    """Return train/val boolean masks splitting by episode ID."""
    episode_ids = np.unique(metadata[:, 0])
    rng = np.random.RandomState(seed)
    rng.shuffle(episode_ids)
    split = int(len(episode_ids) * (1 - val_frac))
    train_eps = set(episode_ids[:split].tolist())
    train_mask = np.array([m in train_eps for m in metadata[:, 0]])
    return train_mask, ~train_mask


def main():
    parser = argparse.ArgumentParser(description="Train target-selector scorer")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset_selector_*.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, choices=list(ACTIVATIONS))
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--out", type=str, default=None, help="Checkpoint save path")
    parser.add_argument("--mps", action="store_true")
    args = parser.parse_args()

    # Load data
    data = np.load(args.data)
    inputs = data["inputs"]       # (N, 3)
    chosen = data["chosen"]       # (N,) int  1=expert picked, 0=not
    metadata = data["metadata"]   # (N, 2)  episode_id, decision_id

    decision_ids = metadata[:, 1].astype(np.int64)
    n_decisions = len(np.unique(decision_ids))

    print(f"Loaded {inputs.shape[0]} candidate rows, {n_decisions} decisions, "
          f"input_dim={inputs.shape[1]}, chosen_ratio={chosen.mean():.3f}")

    # Split by episode
    train_mask, val_mask = split_by_episode(metadata)
    X_train, ch_train, did_train = inputs[train_mask], chosen[train_mask], decision_ids[train_mask]
    X_val, ch_val, did_val = inputs[val_mask], chosen[val_mask], decision_ids[val_mask]

    # Re-index val decision_ids to be contiguous (for efficient grouping)
    _, did_train = np.unique(did_train, return_inverse=True)
    _, did_val = np.unique(did_val, return_inverse=True)

    print(f"Train: {X_train.shape[0]} rows, {len(np.unique(did_train))} decisions")
    print(f"Val:   {X_val.shape[0]} rows, {len(np.unique(did_val))} decisions")

    # Normalize inputs
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    # Device
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Build model: 3 inputs -> 1 score
    input_dim = X_train.shape[1]
    model = build_mlp(input_dim, 1, args.hidden_dim, args.num_layers, args.activation)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Tensors (small enough to fit in memory — process all at once per epoch)
    t_X_train = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    t_ch_train = torch.tensor(ch_train, dtype=torch.long).to(device)
    t_did_train = torch.tensor(did_train, dtype=torch.long).to(device)

    t_X_val = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
    t_ch_val = torch.tensor(ch_val, dtype=torch.long).to(device)
    t_did_val = torch.tensor(did_val, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0
    t_start = _time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            # --- Train ---
            model.train()
            optimizer.zero_grad()
            scores = model(t_X_train).squeeze(-1)
            train_loss, train_correct, train_total = grouped_cross_entropy(
                scores, t_ch_train, t_did_train)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_acc = 100.0 * train_correct / max(train_total, 1)

            # --- Validate ---
            model.eval()
            with torch.no_grad():
                val_scores = model(t_X_val).squeeze(-1)
                val_loss, val_correct, val_total = grouped_cross_entropy(
                    val_scores, t_ch_val, t_did_val)
            val_loss_f = val_loss.item()
            val_acc = 100.0 * val_correct / max(val_total, 1)

            improved = val_loss_f < best_val_loss
            if improved:
                best_val_loss = val_loss_f
                best_epoch = epoch
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            tag = " *best*" if improved else ""
            if epoch % max(1, args.epochs // 10) == 0 or epoch == 1 or improved:
                print(f"Epoch {epoch:4d}/{args.epochs}  "
                      f"train_loss={train_loss.item():.4f}  train_acc={train_acc:.1f}%  "
                      f"val_loss={val_loss_f:.4f}  val_acc={val_acc:.1f}%{tag}")

            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs")
                break
    except KeyboardInterrupt:
        print(f"\nInterrupted at epoch {epoch}. Saving best checkpoint (epoch {best_epoch})...")

    elapsed = _time.time() - t_start

    if best_state:
        model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}  val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.1f}%")

    # Save checkpoint
    if args.out:
        save_path = args.out
    else:
        existing = glob.glob("*.pt")
        counters = [int(m.group(1)) for f in existing if (m := re.match(r"^(\d+)_", f))]
        next_id = max(counters, default=0) + 1
        save_path = f"{next_id:03d}_selector_{args.activation}_e{args.epochs}_h{args.hidden_dim}_l{args.num_layers}.pt"

    torch.save({
        "model_type": "selector",
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "output_dim": 1,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "activation": args.activation,
        "input_mean": mean,
        "input_std": std,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
    }, save_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved {save_path}  ({n_params} params)")


if __name__ == "__main__":
    main()
