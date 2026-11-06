#!/usr/bin/env python3
"""Collect training data for the target-selector policy.

No simulation needed — we just generate random target sets, run the
shortest-path expert decision sequence, and log the observations.
At each decision point the robot is assumed to be at the drop zone of
the last delivered ball (or origin for the first decision), facing a
random bearing.

Each row is one decision: a fixed-size scene vector (all remaining targets
packed into MAX_TARGETS slots with a validity mask) plus the label (which
slot in the packed order the expert chose).

Usage:
    python collect_selector_data.py --num_episodes 5000 --max_targets 5 --seed 42
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from robotnik_arm_task import (
    MAX_TARGETS,
    shortest_path_expert,
    build_selector_scene,
    generate_random_drop_zones,
)


def generate_episode_targets(rng, n_targets):
    """Generate n_targets random positions in the workspace."""
    xs = rng.uniform(-5, 5, n_targets)
    ys = rng.uniform(-5, 5, n_targets)
    zs = rng.uniform(0.2, 1.0, n_targets)
    return np.column_stack([xs, ys, zs])


def run_expert_decisions(targets, rng, drop_zones):
    """Multi-drop-zone expert decisions: each ball assigned to a random drop zone.

    At each decision point the robot is at the drop zone of the last delivered ball
    (or origin for the first). The expert solves the multi-drop TSP at each step.

    Returns:
        obs_list: list of (SELECTOR_INPUT_DIM,) scene observations
        label_list: list of int — index within the packed slot_order
        decision_ids: list of int
    """
    n = len(targets)
    # Randomly assign each ball to a drop zone
    ball_zone_assignments = [int(rng.integers(0, len(drop_zones))) for _ in range(n)]

    remaining = set(range(n))
    robot_pos = np.array([0.0, 0.0, 0.0])
    robot_bearing = 0.0

    obs_list = []
    label_list = []
    decision_ids = []
    decision_id = 0

    while len(remaining) > 1:
        scene_obs, slot_order = build_selector_scene(
            robot_pos, robot_bearing, targets, remaining,
            drop_zones, ball_zone_assignments)

        chosen_idx, _ = shortest_path_expert(
            robot_pos, targets, remaining, drop_zones, ball_zone_assignments)
        label = slot_order.index(chosen_idx)

        obs_list.append(scene_obs)
        label_list.append(label)
        decision_ids.append(decision_id)

        remaining.remove(chosen_idx)
        decision_id += 1

        # After picking + delivering, robot is at the assigned drop zone
        robot_pos = drop_zones[ball_zone_assignments[chosen_idx]].copy()
        robot_bearing = rng.uniform(-np.pi, np.pi)

    return obs_list, label_list, decision_ids


def main():
    parser = argparse.ArgumentParser(description="Collect selector training data (no sim needed)")
    parser.add_argument("--num_episodes", type=int, default=5000,
                        help="Number of episodes (default: 5000)")
    parser.add_argument("--max_targets", type=int, default=5,
                        help="Max targets per episode (sampled from min..max, default: 5)")
    parser.add_argument("--min_targets", type=int, default=2,
                        help="Min targets per episode (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    args = parser.parse_args()

    assert args.min_targets >= 2
    rng = np.random.default_rng(args.seed)

    print("Mode: multi-drop-zone (randomized zones per episode)")

    all_obs = []
    all_labels = []
    all_eid = []
    all_did = []
    global_did = 0

    for ep in range(args.num_episodes):
        n_targets = rng.integers(args.min_targets, args.max_targets + 1)
        targets = generate_episode_targets(rng, n_targets)

        # Generate random drop zones for this episode
        drop_zones = generate_random_drop_zones(rng)

        obs_list, label_list, local_dids = run_expert_decisions(targets, rng, drop_zones)

        for obs, lbl, ldid in zip(obs_list, label_list, local_dids):
            all_obs.append(obs)
            all_labels.append(lbl)
            all_did.append(global_did + ldid)
            all_eid.append(ep)

        global_did += max(local_dids) + 1 if local_dids else 0

        if ep % 500 == 0 or ep == args.num_episodes - 1:
            print(f"Episode {ep+1}/{args.num_episodes}  "
                  f"decisions so far: {len(all_obs)}")

    inputs = np.array(all_obs, dtype=np.float64)
    labels = np.array(all_labels, dtype=np.int64)
    metadata = np.column_stack([
        np.array(all_eid, dtype=np.float64),
        np.array(all_did, dtype=np.float64),
    ])

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"dataset_selector_{timestamp}.npz")
    np.savez(out_path, inputs=inputs, labels=labels, metadata=metadata)

    n_decisions = len(np.unique(all_did))
    print(f"\nSaved {out_path}")
    print(f"  decision rows: {inputs.shape[0]}  input_dim: {inputs.shape[1]}")
    print(f"  decisions: {n_decisions}")
    print("Done.")


if __name__ == "__main__":
    main()
