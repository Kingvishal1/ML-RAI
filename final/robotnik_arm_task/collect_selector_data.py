#!/usr/bin/env python3
"""Collect training data for the target-selector policy.

No simulation needed — we just generate random target sets, run the
nearest-neighbor expert decision sequence, and log the observations.
At each decision point the robot is assumed to be at the last visited
target (or origin for the first decision), facing a random bearing.

Usage:
    python collect_selector_data.py --num_episodes 5000 --max_targets 5 --seed 42
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np

from robotnik_arm_task import (
    MAX_TARGETS, SELECTOR_OBS_DIM,
    target_to_base_frame, nearest_neighbor_expert,
)


def generate_episode_targets(rng, n_targets):
    """Generate n_targets random positions in the workspace."""
    xs = rng.uniform(-5, 5, n_targets)
    ys = rng.uniform(-5, 5, n_targets)
    zs = rng.uniform(0.2, 1.0, n_targets)
    return np.column_stack([xs, ys, zs])


def run_expert_decisions(targets, rng):
    """Simulate the expert decision sequence without any physics.

    The robot starts at the origin (0,0,0) with bearing 0, then after each
    decision is assumed to be at the chosen target with a random bearing
    (mimicking arrival from an arbitrary direction).

    Args:
        targets: (N, 3) array of target positions
        rng: numpy random generator

    Returns:
        obs_list: list of (3,) candidate observations
        chosen_list: list of int (1 if expert picked, 0 if not)
        decision_ids: list of int (decision index)
    """
    n = len(targets)
    remaining = set(range(n))

    # Robot starts at origin
    robot_pos = np.array([0.0, 0.0, 0.0])
    robot_bearing = 0.0

    obs_list = []
    chosen_list = []
    decision_ids = []
    decision_id = 0

    while len(remaining) > 1:
        # Expert picks nearest
        chosen_idx = nearest_neighbor_expert(robot_pos, targets, remaining)

        # Log one row per remaining candidate
        for idx in remaining:
            obs = target_to_base_frame(robot_pos, robot_bearing, targets[idx])
            obs_list.append(obs)
            chosen_list.append(1 if idx == chosen_idx else 0)
            decision_ids.append(decision_id)

        remaining.remove(chosen_idx)
        decision_id += 1

        # Teleport robot to chosen target with random bearing
        robot_pos = targets[chosen_idx].copy()
        robot_bearing = rng.uniform(-np.pi, np.pi)

    return obs_list, chosen_list, decision_ids


def main():
    parser = argparse.ArgumentParser(description="Collect selector training data (no sim needed)")
    parser.add_argument("--num_episodes", type=int, default=5000,
                        help="Number of episodes (default: 5000)")
    parser.add_argument("--max_targets", type=int, default=5,
                        help="Max targets per episode (sampled from min..max, default: 5)")
    parser.add_argument("--min_targets", type=int, default=2,
                        help="Min targets per episode (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flush_every", type=int, default=1000,
                        help="Flush to disk every N episodes (default: 1000)")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = parser.parse_args()

    assert args.min_targets >= 2
    rng = np.random.default_rng(args.seed)

    all_obs = []
    all_chosen = []
    all_did = []      # global decision id
    all_eid = []      # episode id
    global_did = 0
    chunk_paths = []
    chunk_idx = 0
    episodes_since_flush = 0

    for ep in range(args.num_episodes):
        n_targets = rng.integers(args.min_targets, args.max_targets + 1)
        targets = generate_episode_targets(rng, n_targets)

        obs_list, chosen_list, local_dids = run_expert_decisions(targets, rng)

        for obs, ch, ldid in zip(obs_list, chosen_list, local_dids):
            all_obs.append(obs)
            all_chosen.append(ch)
            all_did.append(global_did + ldid)
            all_eid.append(ep)

        global_did += max(local_dids) + 1 if local_dids else 0
        episodes_since_flush += 1

        if ep % 500 == 0 or ep == args.num_episodes - 1:
            print(f"Episode {ep+1}/{args.num_episodes}  "
                  f"rows so far: {len(all_obs)}")

        if episodes_since_flush >= args.flush_every:
            path = _flush(all_obs, all_chosen, all_eid, all_did, args.out_dir, chunk_idx)
            if path:
                chunk_paths.append(path)
                chunk_idx += 1
            all_obs, all_chosen, all_did, all_eid = [], [], [], []
            episodes_since_flush = 0

    # Flush remaining
    if all_obs:
        path = _flush(all_obs, all_chosen, all_eid, all_did, args.out_dir, chunk_idx)
        if path:
            chunk_paths.append(path)

    if not chunk_paths:
        print("No data collected.")
        sys.exit(1)

    # Merge chunks
    all_inputs, all_ch, all_meta = [], [], []
    for p in chunk_paths:
        with np.load(p) as d:
            all_inputs.append(d["inputs"])
            all_ch.append(d["chosen"])
            all_meta.append(d["metadata"])
    inputs = np.concatenate(all_inputs)
    chosen = np.concatenate(all_ch)
    metadata = np.concatenate(all_meta)

    # Re-number decision_ids contiguously
    old_dids = metadata[:, 1]
    mapping = {}
    new_id = 0
    new_dids = np.empty(len(old_dids), dtype=np.float64)
    for i, oid in enumerate(old_dids):
        if oid not in mapping:
            mapping[oid] = new_id
            new_id += 1
        new_dids[i] = mapping[oid]
    metadata[:, 1] = new_dids

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"dataset_selector_{timestamp}.npz")
    np.savez(out_path, inputs=inputs, chosen=chosen, metadata=metadata)

    n_decisions = int(new_dids.max()) + 1
    print(f"\nSaved {out_path}")
    print(f"  candidate rows: {inputs.shape[0]}  input_dim: {inputs.shape[1]}")
    print(f"  decisions: {n_decisions}  chosen ratio: {chosen.mean():.3f}")

    # Clean up chunks
    for p in chunk_paths:
        os.remove(p)
    print("Done.")


def _flush(obs, chosen, eid, did, out_dir, chunk_idx):
    if not obs:
        return None
    inputs = np.array(obs, dtype=np.float64)
    ch = np.array(chosen, dtype=np.int64)
    meta = np.column_stack([
        np.array(eid, dtype=np.float64),
        np.array(did, dtype=np.float64),
    ])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"_selector_chunk_{chunk_idx:04d}.npz")
    np.savez(path, inputs=inputs, chosen=ch, metadata=meta)
    print(f"  Flushed chunk {chunk_idx} -> {path}  ({inputs.shape[0]} rows)")
    return path


if __name__ == "__main__":
    main()
