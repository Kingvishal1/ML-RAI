"""Data collection script for behaviour cloning.

Runs the expert controller across many targets, logs input/output pairs
per timestep, and saves them to disk in .npz and .csv formats.
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np

# Force headless mode for batch data collection
os.environ["ROBOENV_HEADLESS"] = "1"

from robotnik_arm_task import (
    FIXED_FINGER_ANGLES,
    setup_simulation,
    setup_kinematics,
    run_simulation_loop,
)
from robotnik_arm_control import configure_gripper_fingers


# ==============================================================================
# Target generation
# ==============================================================================

def generate_grid_targets(n=100):
    """Produce targets on a fixed grid over the workspace.

    Workspace bounds: X: [-0.8, 0.8], Y: [0.5, 1.5], Z: [0.2, 1.0].
    Grid dimensions scale automatically so that nx * ny * nz >= n.
    """
    points_per_axis = int(np.ceil(n ** (1.0 / 3.0)))
    xs = np.linspace(-0.8, 0.8, points_per_axis)
    ys = np.linspace(0.5, 1.5, points_per_axis)
    zs = np.linspace(0.2, 1.0, points_per_axis)
    targets = []
    for x in xs:
        for y in ys:
            for z in zs:
                targets.append(np.array([x, y, z]))
    return targets[:n]


def generate_random_targets(n=100, seed=42):
    """Uniform random sampling from the workspace volume."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-0.8, 0.8, n)
    ys = rng.uniform(0.5, 1.5, n)
    zs = rng.uniform(0.2, 1.0, n)
    return [np.array([xs[i], ys[i], zs[i]]) for i in range(n)]


# ==============================================================================
# Simulation runner
# ==============================================================================

def run_single_trajectory(sim, client, rid, dyn_model, ee_frame_id,
                          shoulder_frame_id, init_joint_angles,
                          lower_limits, upper_limits,
                          target_ee_pos, vary_init=False, rng=None):
    """Reset the sim and run one trajectory for a given target.

    Returns the data dict from run_simulation_loop or None on failure.
    """
    # Reset to default pose
    sim.ResetPose()

    # Optionally perturb initial base position/orientation
    if vary_init and rng is not None:
        base_pos = list(sim.GetBasePosition())
        base_ori = list(sim.GetBaseOrientation())
        # Small perturbations: ±0.3m in x/y, ±0.1 rad in yaw
        base_pos[0] += rng.uniform(-0.3, 0.3)
        base_pos[1] += rng.uniform(-0.3, 0.3)
        import pybullet as p
        yaw_perturb = rng.uniform(-0.1, 0.1)
        new_ori = p.getQuaternionFromEuler([0, 0, yaw_perturb])
        client.resetBasePositionAndOrientation(
            sim.bot[0].bot_pybullet, base_pos, new_ori,
        )

    # Re-configure gripper fingers
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Re-read joint angles after reset
    init_joint_angles_now = np.array(sim.GetMotorAngles(0))

    try:
        data = run_simulation_loop(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles_now, target_finger_angles,
            lower_limits, upper_limits,
            target_ee_pos=target_ee_pos,
        )
        return data
    except Exception as e:
        print(f"  [WARN] Trajectory failed for target {target_ee_pos}: {e}")
        return None


# ==============================================================================
# Data assembly
# ==============================================================================

def assemble_dataset(all_trajectory_data):
    """Assemble per-timestep rows from all trajectories.

    Returns (inputs, labels, metadata) as numpy arrays.
      inputs:   (N, 44) — base_pos_x, base_pos_y, base_bearing,
                           base_lin_vel_x/y/z, base_ang_vel_x/y/z,
                           q_base_1..4, qd_base_1..4,
                           q_arm_1..7, qd_arm_1..7,
                           ee_pos_x/y/z, target_x, target_y, target_z,
                           tau_gravity_1..7
      labels:   (N, 11) — wheel_cmd_1..4, torque_arm_1..7
      metadata: (N, 2)  — trajectory_id, time
    """
    input_rows = []
    label_rows = []
    meta_rows = []

    for traj_id, data in enumerate(all_trajectory_data):
        n_steps = len(data["time"])
        has_gravity = "tau_gravity" in data and len(data["tau_gravity"]) == n_steps
        for i in range(n_steps):
            base_pos_xy = data["base_pos"][i, :2]            # 2
            base_bearing = data["base_bearing"][i]            # 1
            base_lin_vel = data["base_lin_vel"][i]            # 3
            base_ang_vel = data["base_ang_vel"][i]            # 3
            q_mes = np.array(data["q_mes"][i])
            qd_mes = np.array(data["qd_mes"][i])
            q_base = q_mes[0:4]                               # 4
            qd_base = qd_mes[0:4]                             # 4
            q_arm = q_mes[4:11]                               # 7
            qd_arm = qd_mes[4:11]                             # 7
            ee_pos = data["ee_pos"][i]                         # 3
            target = data["target_ee_pos"][i]                  # 3
            tau_grav = data["tau_gravity"][i] if has_gravity else np.zeros(7)  # 7

            inp = np.concatenate([
                base_pos_xy,
                [base_bearing],
                base_lin_vel,
                base_ang_vel,
                q_base,
                qd_base,
                q_arm,
                qd_arm,
                ee_pos,
                target,
                tau_grav,
            ])  # 44 features

            wheel_cmds = data["wheel_cmds"][i]                 # 4
            torque_arm = data["torque_arm"][i]                  # 7
            lbl = np.concatenate([wheel_cmds, torque_arm])     # 11 outputs

            meta = np.array([traj_id, data["time"][i]])

            input_rows.append(inp)
            label_rows.append(lbl)
            meta_rows.append(meta)

    inputs = np.array(input_rows, dtype=np.float64)
    labels = np.array(label_rows, dtype=np.float64)
    metadata = np.array(meta_rows, dtype=np.float64)
    return inputs, labels, metadata


# ==============================================================================
# Saving
# ==============================================================================

INPUT_COLS = (
    ["base_pos_x", "base_pos_y", "base_bearing"]
    + ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]
    + ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
    + [f"q_base_{i}" for i in range(1, 5)]
    + [f"qd_base_{i}" for i in range(1, 5)]
    + [f"q_arm_{i}" for i in range(1, 8)]
    + [f"qd_arm_{i}" for i in range(1, 8)]
    + ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
    + ["target_x", "target_y", "target_z"]
    + [f"tau_gravity_{i}" for i in range(1, 8)]
)

LABEL_COLS = (
    [f"wheel_cmd_{i}" for i in range(1, 5)]
    + [f"torque_arm_{i}" for i in range(1, 8)]
)

META_COLS = ["trajectory_id", "time"]


def flush_chunk(all_trajectory_data, out_dir, chunk_idx):
    """Assemble a batch of trajectories and write to a temporary .npz chunk.

    Returns the path to the chunk file, or None if no data.
    """
    if not all_trajectory_data:
        return None
    inputs, labels, metadata = assemble_dataset(all_trajectory_data)
    os.makedirs(out_dir, exist_ok=True)
    chunk_path = os.path.join(out_dir, f"_chunk_{chunk_idx:04d}.npz")
    np.savez(chunk_path, inputs=inputs, labels=labels, metadata=metadata)
    print(f"  Flushed chunk {chunk_idx} → {chunk_path}  "
          f"({len(all_trajectory_data)} trajs, {inputs.shape[0]} rows)")
    return chunk_path


def merge_chunks(chunk_paths, out_dir):
    """Load all chunk files, concatenate, save final dataset, and clean up."""
    all_inputs, all_labels, all_meta = [], [], []
    for p in chunk_paths:
        d = np.load(p)
        all_inputs.append(d["inputs"])
        all_labels.append(d["labels"])
        all_meta.append(d["metadata"])
    inputs = np.concatenate(all_inputs)
    labels = np.concatenate(all_labels)
    metadata = np.concatenate(all_meta)

    # Re-number trajectory IDs so they are globally contiguous
    _, inverse = np.unique(metadata[:, 0], return_inverse=True)
    # Assign new sequential IDs preserving original order of first appearance
    first_occurrence = {}
    new_id = 0
    new_ids = np.empty(len(inverse), dtype=np.float64)
    for i, old_id in enumerate(metadata[:, 0]):
        if old_id not in first_occurrence:
            first_occurrence[old_id] = new_id
            new_id += 1
        new_ids[i] = first_occurrence[old_id]
    metadata[:, 0] = new_ids

    save_dataset(inputs, labels, metadata, out_dir)

    # Remove temporary chunk files
    for p in chunk_paths:
        os.remove(p)


def save_dataset(inputs, labels, metadata, out_dir):
    """Save dataset as .npz and .csv with timestamped filenames."""
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # .npz
    npz_path = os.path.join(out_dir, f"dataset_{timestamp}.npz")
    np.savez(npz_path, inputs=inputs, labels=labels, metadata=metadata)
    print(f"Saved {npz_path}  inputs={inputs.shape}  labels={labels.shape}  meta={metadata.shape}")

    # .csv
    csv_path = os.path.join(out_dir, f"dataset_{timestamp}.csv")
    all_cols = META_COLS + INPUT_COLS + LABEL_COLS
    all_data = np.hstack([metadata, inputs, labels])
    header = ",".join(all_cols)
    np.savetxt(csv_path, all_data, delimiter=",", header=header, comments="")
    print(f"Saved {csv_path}  rows={all_data.shape[0]}  cols={all_data.shape[1]}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect expert trajectories for behaviour cloning.")
    parser.add_argument("--mode", choices=["grid", "random"], default="random",
                        help="Target generation mode (default: grid)")
    parser.add_argument("--num_targets", type=int, default=100,
                        help="Number of targets to collect (default: 100)")
    parser.add_argument("--vary_init", action="store_true",
                        help="Also randomise initial base pose per trajectory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for random targets and init perturbation")
    parser.add_argument("--flush_every", type=int, default=100,
                        help="Flush trajectories to disk every N runs (default: 100)")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                        help="Output directory (default: final/robotnik_arm_task/data/)")
    args = parser.parse_args()

    # Generate targets
    if args.mode == "grid":
        targets = generate_grid_targets(n=args.num_targets)
    else:
        targets = generate_random_targets(n=args.num_targets, seed=args.seed)

    print(f"Collecting {len(targets)} trajectories (mode={args.mode}, vary_init={args.vary_init})")

    # Setup simulation once
    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, _, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    rng = np.random.default_rng(args.seed) if args.vary_init else None

    # Run trajectories, flushing to disk periodically
    all_data = []
    chunk_paths = []
    chunk_idx = 0
    for idx, target in enumerate(targets):
        print(f"\n[{idx+1}/{len(targets)}] Target: {np.round(target, 3)}")
        data = run_single_trajectory(
            sim, client, rid, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles, lower_limits, upper_limits,
            target_ee_pos=target, vary_init=args.vary_init, rng=rng,
        )
        if data is not None and len(data["time"]) > 0:
            all_data.append(data)
            print(f"  Steps: {len(data['time'])}  Final EE error: "
                  f"{np.linalg.norm(target - data['final_ee_pos']):.4f}m")
        else:
            print(f"  [SKIP] No data collected for target {target}")

        # Flush to disk every flush_every successful trajectories
        if len(all_data) >= args.flush_every:
            path = flush_chunk(all_data, args.out_dir, chunk_idx)
            if path:
                chunk_paths.append(path)
                chunk_idx += 1
            all_data = []  # free memory

    # Flush any remaining trajectories
    if all_data:
        path = flush_chunk(all_data, args.out_dir, chunk_idx)
        if path:
            chunk_paths.append(path)
        all_data = []

    if not chunk_paths:
        print("No trajectories collected. Exiting.")
        sys.exit(1)

    # Merge all chunks into final dataset
    print(f"\nMerging {len(chunk_paths)} chunks into final dataset...")
    merge_chunks(chunk_paths, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
