#!/usr/bin/env python3
"""Run the trained BC policy in the PyBullet simulation to reach a target endpoint."""

import argparse
import os
import time

import numpy as np
import pinocchio as pin
from simulation_and_control import pb, MotorCommands
from robotnik_arm_task import (
    setup_simulation, setup_kinematics, add_target_visual,
    FIXED_FINGER_ANGLES, REACH_TOLERANCE,
)
from robotnik_arm_control import configure_gripper_fingers, quaternion2bearing
from run_policy import load_policy, predict


def build_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                      q_mes, qd_mes, ee_pos, target_ee_pos):
    """Assemble the 37-dim observation vector matching the training data format."""
    return np.concatenate([
        base_pos[:2],               # 2: base_pos_x, base_pos_y
        [base_bearing],             # 1: base_bearing
        base_lin_vel,               # 3: base_lin_vel_x/y/z
        base_ang_vel,               # 3: base_ang_vel_x/y/z
        q_mes[0:4],                 # 4: q_base (wheel angles)
        qd_mes[0:4],               # 4: qd_base (wheel velocities)
        q_mes[4:11],               # 7: q_arm
        qd_mes[4:11],              # 7: qd_arm
        ee_pos,                     # 3: ee_pos_x/y/z
        target_ee_pos,              # 3: target_x/y/z
    ])  # total: 37


def main():
    parser = argparse.ArgumentParser(description="Run BC policy in simulation")
    parser.add_argument("--checkpoint", type=str, default="bc_policy_100.pt",
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 1.1, 0.5],
                        metavar=("X", "Y", "Z"),
                        help="Target end-effector position (default: 0.0 1.1 0.5)")
    parser.add_argument("--max_steps", type=int, default=40000,
                        help="Maximum simulation steps")
    parser.add_argument("--execute_steps", type=int, default=None,
                        help="Execute this many steps from each chunk before re-predicting "
                             "(MPC-style). Defaults to chunk_size (use full chunk).")
    parser.add_argument("--headless", action="store_true",
                        help="Skip visualizer updates and sleep for fast evaluation")
    parser.add_argument("--mps", action="store_true",
                        help="Use Apple Metal (MPS) GPU for inference")
    args = parser.parse_args()

    target_ee_pos = np.array(args.target)

    # --- Load policy ---
    import torch
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model, mean, std, chunk_size = load_policy(args.checkpoint, device=device)
    execute_steps = args.execute_steps if args.execute_steps is not None else chunk_size
    if execute_steps > chunk_size:
        print(f"WARNING: execute_steps ({execute_steps}) > chunk_size ({chunk_size}), clamping.")
        execute_steps = chunk_size
    print(f"Loaded policy from {args.checkpoint}  device={device}  "
          f"chunk_size={chunk_size}  execute_steps={execute_steps}")

    # --- Setup simulation ---
    if args.headless:
        os.environ["ROBOENV_HEADLESS"] = "1"
    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    add_target_visual(client, target_ee_pos)

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    time_step = sim.GetTimeStep()
    current_time = 0.0
    cmd = MotorCommands()

    print("=" * 60)
    print(f"Target EE Position : {target_ee_pos}")
    print(f"Initial EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Error (m)  : {np.linalg.norm(target_ee_pos - init_ee_pos):.4f}")
    print("=" * 60)

    best_ee_error = np.inf
    step = 0
    action_chunk = None  # cached chunk of K actions, shape (K, 11) or None
    chunk_idx = 0        # index into current chunk

    while step < args.max_steps:
        # ---- Sense (always needed for logging / exit checks) ----
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))
        base_lin_vel = np.array(sim.GetBaseLinVelocity())
        base_ang_vel = np.array(sim.GetBaseAngVelocity())

        # Forward kinematics for EE position
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)

        if ee_dist_error < best_ee_error:
            best_ee_error = ee_dist_error

        # ---- Predict new chunk if needed (MPC: re-predict every execute_steps) ----
        if action_chunk is None or chunk_idx >= execute_steps:
            obs = build_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                                    q_mes, qd_mes, ee_pos, target_ee_pos)
            raw = predict(model, mean, std, obs, device=device, chunk_size=chunk_size)
            if chunk_size > 1:
                action_chunk = raw          # shape (K, 11)
            else:
                action_chunk = raw[np.newaxis, :]  # shape (1, 11)
            chunk_idx = 0

        # ---- Pick action from chunk ----
        action = action_chunk[chunk_idx]
        chunk_idx += 1

        wheel_cmds = action[:4]
        torque_arm = action[4:]

        # ---- Send commands ----
        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        if not args.headless:
            # ---- Optional visualisation ----
            if dyn_model.visualizer:
                for index in range(len(sim.bot)):
                    q = sim.GetMotorAngles(index)
                    dyn_model.DisplayModel(q)

            # ---- Exit on 'q' key ----
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("User quit.")
                break

        # ---- Check if target reached ----
        if ee_dist_error < REACH_TOLERANCE and current_time > 8.0:
            print(f"Target reached within {REACH_TOLERANCE*1000:.0f}mm at t={current_time:.2f}s!")
            break

        # ---- Logging ----
        if step % 500 == 0:
            print(f"t={current_time:.2f}s | Base: {np.round(base_pos[:2], 3)} | "
                  f"EE: {np.round(ee_pos, 3)} | EE err: {ee_dist_error:.4f}m | "
                  f"best: {best_ee_error:.4f}m")

        if not args.headless:
            time.sleep(0.001)
        current_time += time_step
        step += 1

    # ---- Final report ----
    print("=" * 60)
    print(f"Final EE Position  : {np.round(ee_pos, 4)}")
    print(f"Target EE Position : {target_ee_pos}")
    print(f"Final Error (m)    : {ee_dist_error:.4f}")
    print(f"Best Error (m)     : {best_ee_error:.4f}")
    print(f"Steps: {step}  Time: {current_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
