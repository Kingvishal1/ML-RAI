#!/usr/bin/env python3
"""Run a trained BC policy in PyBullet simulation to reach a target endpoint.

Supports two modes:
  1. Single-model mode (--checkpoint): one policy predicts all 11 outputs.
  2. Split-model mode (--base_checkpoint + --arm_checkpoint): base policy navigates,
     then arm policy reaches, with expert PD arm-hold during base navigation.
"""

import argparse
import os
import time

import numpy as np
import pinocchio as pin
from simulation_and_control import pb, MotorCommands
from robotnik_arm_task import (
    setup_simulation, setup_kinematics, add_target_visual,
    FIXED_FINGER_ANGLES, REACH_TOLERANCE, ARM_REACH_LIMIT,
)
from robotnik_arm_control import (
    ArmControllerParams, arm_controller, configure_gripper_fingers, quaternion2bearing,
)
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


def build_base_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel, target_ee_pos):
    """Assemble the 13-dim base policy observation."""
    delta = target_ee_pos[:2] - base_pos[:2]
    dist_to_target = np.linalg.norm(delta)
    heading_to_target = np.arctan2(delta[1], delta[0])
    heading_error = np.arctan2(np.sin(heading_to_target - base_bearing),
                               np.cos(heading_to_target - base_bearing))
    return np.concatenate([
        base_pos[:2],               # 2: base_pos_x, base_pos_y
        [base_bearing],             # 1: base_bearing
        base_lin_vel,               # 3: base_lin_vel_x/y/z
        base_ang_vel,               # 3: base_ang_vel_x/y/z
        target_ee_pos[:2],          # 2: target_x, target_y
        [dist_to_target],           # 1: distance to target XY
        [heading_error],            # 1: wrapped heading error
    ])  # total: 13


def build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos):
    """Assemble the 20-dim arm policy observation."""
    return np.concatenate([
        q_mes[4:11],               # 7: q_arm
        qd_mes[4:11],              # 7: qd_arm
        ee_pos,                     # 3: ee_pos_x/y/z
        target_ee_pos,              # 3: target_x/y/z
    ])  # total: 20


def run_sim(args, target_ee_pos, device, split_mode, single_mode,
            # single-mode policy
            model=None, mean=None, std=None, chunk_size=None, execute_steps=None,
            # split-mode policies
            base_model=None, base_mean=None, base_std=None, base_chunk_size=None, base_execute=None,
            arm_model=None, arm_mean=None, arm_std=None, arm_chunk_size=None, arm_execute=None):
    """Run one simulation attempt. Returns True if finished normally, False if crashed."""

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    add_target_visual(client, target_ee_pos)

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    time_step = sim.GetTimeStep()
    current_time = 0.0
    cmd = MotorCommands()

    # Arm hold controller state (for split mode phase 1)
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])

    print("=" * 60)
    print(f"Target EE Position : {target_ee_pos}")
    print(f"Initial EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Error (m)  : {np.linalg.norm(target_ee_pos - init_ee_pos):.4f}")
    if split_mode:
        print(f"Mode               : SPLIT (base -> arm at {args.base_threshold}m)")
    else:
        print(f"Mode               : SINGLE")
    print("=" * 60)

    best_ee_error = np.inf
    step = 0
    action_chunk = None
    chunk_idx = 0
    phase = "base" if split_mode else "single"
    phase_switched = False

    # Track last known good state for the final report on crash
    ee_pos = init_ee_pos.copy()
    ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)
    base_pos = np.array(sim.GetBasePosition())

    while step < args.max_steps:
        # ---- Sense ----
        try:
            base_pos = np.array(sim.GetBasePosition())
            base_ori = np.array(sim.GetBaseOrientation())
            base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
            q_mes = np.array(sim.GetMotorAngles(0))
            qd_mes = np.array(sim.GetMotorVelocities(0))
            base_lin_vel = np.array(sim.GetBaseLinVelocity())
            base_ang_vel = np.array(sim.GetBaseAngVelocity())
        except Exception as e:
            print(f"\nSim crashed reading sensors at t={current_time:.2f}s  step={step}: {e}")
            return False

        # Forward kinematics for EE position
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)

        if ee_dist_error < best_ee_error:
            best_ee_error = ee_dist_error

        # ---- Phase transition check (split mode only) ----
        if phase == "base":
            base_xy_dist = np.linalg.norm(base_pos[:2] - target_ee_pos[:2])
            if base_xy_dist < args.base_threshold:
                phase = "arm"
                phase_switched = True
                action_chunk = None
                chunk_idx = 0
                print(f">>> Phase switch: base -> arm at t={current_time:.2f}s  "
                      f"base_xy_dist={base_xy_dist:.4f}m")

        # ---- Compute action based on mode/phase ----
        if phase == "single":
            if action_chunk is None or chunk_idx >= execute_steps:
                obs = build_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                                        q_mes, qd_mes, ee_pos, target_ee_pos)
                raw = predict(model, mean, std, obs, device=device, chunk_size=chunk_size)
                action_chunk = raw if chunk_size > 1 else raw[np.newaxis, :]
                chunk_idx = 0
            action = action_chunk[chunk_idx]
            chunk_idx += 1
            wheel_cmds = action[:4]
            torque_arm = action[4:]

        elif phase == "base":
            if action_chunk is None or chunk_idx >= base_execute:
                obs_base = build_base_observation(base_pos, base_bearing, base_lin_vel,
                                                  base_ang_vel, target_ee_pos)
                raw = predict(base_model, base_mean, base_std, obs_base,
                              device=device, chunk_size=base_chunk_size)
                action_chunk = raw if base_chunk_size > 1 else raw[np.newaxis, :]
                chunk_idx = 0
            wheel_cmds = np.clip(action_chunk[chunk_idx], -3.0, 3.0)
            chunk_idx += 1

            torque_arm, q_des_arm, qd_des_arm = arm_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11],
                q_des_arm, q_init_arm, target_ee_pos, ee_pos,
                ee_frame_id, shoulder_frame_id, arm_lower, arm_upper,
                time_step, arm_params, reach_active=False,
            )

        elif phase == "arm":
            if action_chunk is None or chunk_idx >= arm_execute:
                obs_arm = build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos)
                raw = predict(arm_model, arm_mean, arm_std, obs_arm,
                              device=device, chunk_size=arm_chunk_size)
                action_chunk = raw if arm_chunk_size > 1 else raw[np.newaxis, :]
                chunk_idx = 0
            torque_arm = np.clip(action_chunk[chunk_idx], -30.0, 30.0)
            chunk_idx += 1
            wheel_cmds = np.zeros(4)

        # ---- Send commands ----
        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        try:
            sim.Step(cmd, "torque")
        except Exception as e:
            print(f"\nSim crashed at t={current_time:.2f}s  step={step}  phase={phase}")
            print(f"  Last base_pos: {np.round(base_pos, 3)}  wheel_cmds: {np.round(wheel_cmds, 3)}")
            print(f"  Error: {e}")
            return False

        if not args.headless:
            if dyn_model.visualizer:
                for index in range(len(sim.bot)):
                    q = sim.GetMotorAngles(index)
                    dyn_model.DisplayModel(q)

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
            phase_str = f" [{phase}]" if split_mode else ""
            print(f"t={current_time:.2f}s{phase_str} | Base: {np.round(base_pos[:2], 3)} | "
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
    if split_mode:
        print(f"Phase switched     : {'Yes' if phase_switched else 'No (stayed in base phase)'}")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run BC policy in simulation")
    # Single-model mode
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained .pt checkpoint (single-model mode)")
    # Split-model mode
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Path to base policy .pt checkpoint (split mode)")
    parser.add_argument("--arm_checkpoint", type=str, default=None,
                        help="Path to arm policy .pt checkpoint (split mode)")
    parser.add_argument("--base_threshold", type=float, default=0.5,
                        help="XY distance to target below which to switch from base to arm (default: 0.15m)")
    # Common args
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
    parser.add_argument("--retries", type=int, default=0,
                        help="Number of times to restart simulation on crash (default: 0)")
    args = parser.parse_args()

    # Determine mode
    split_mode = args.base_checkpoint is not None and args.arm_checkpoint is not None
    single_mode = args.checkpoint is not None
    if not split_mode and not single_mode:
        args.checkpoint = "bc_policy_100.pt"
        single_mode = True
    if split_mode and single_mode:
        parser.error("Specify either --checkpoint OR (--base_checkpoint + --arm_checkpoint), not both.")

    target_ee_pos = np.array(args.target)

    # --- Load policy/policies (once, reused across retries) ---
    import torch
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    policy_kwargs = dict(device=device)

    if single_mode:
        model, mean, std, chunk_size = load_policy(args.checkpoint, device=device)
        execute_steps = args.execute_steps if args.execute_steps is not None else chunk_size
        if execute_steps > chunk_size:
            print(f"WARNING: execute_steps ({execute_steps}) > chunk_size ({chunk_size}), clamping.")
            execute_steps = chunk_size
        print(f"Loaded policy from {args.checkpoint}  device={device}  "
              f"chunk_size={chunk_size}  execute_steps={execute_steps}")
        policy_kwargs.update(model=model, mean=mean, std=std,
                             chunk_size=chunk_size, execute_steps=execute_steps)
    else:
        base_model, base_mean, base_std, base_chunk_size = load_policy(args.base_checkpoint, device=device)
        arm_model, arm_mean, arm_std, arm_chunk_size = load_policy(args.arm_checkpoint, device=device)
        base_execute = args.execute_steps if args.execute_steps is not None else base_chunk_size
        arm_execute = args.execute_steps if args.execute_steps is not None else arm_chunk_size
        if base_execute > base_chunk_size:
            base_execute = base_chunk_size
        if arm_execute > arm_chunk_size:
            arm_execute = arm_chunk_size
        print(f"Split mode: base={args.base_checkpoint} (chunk={base_chunk_size}, exec={base_execute})  "
              f"arm={args.arm_checkpoint} (chunk={arm_chunk_size}, exec={arm_execute})  "
              f"device={device}  base_threshold={args.base_threshold}m")
        policy_kwargs.update(base_model=base_model, base_mean=base_mean, base_std=base_std,
                             base_chunk_size=base_chunk_size, base_execute=base_execute,
                             arm_model=arm_model, arm_mean=arm_mean, arm_std=arm_std,
                             arm_chunk_size=arm_chunk_size, arm_execute=arm_execute)

    if args.headless:
        os.environ["ROBOENV_HEADLESS"] = "1"

    # --- Run with retries ---
    max_attempts = 1 + args.retries
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"\n>>> Restarting simulation (attempt {attempt}/{max_attempts}) <<<\n")
        ok = run_sim(args, target_ee_pos, split_mode=split_mode, single_mode=single_mode,
                     **policy_kwargs)
        if ok:
            break
    else:
        print(f"\nAll {max_attempts} attempts crashed.")


if __name__ == "__main__":
    main()
