#!/usr/bin/env python3
"""Run trained BC policies in PyBullet simulation.

Two modes:
  1. Single-target: base navigates to a target, arm reaches it.
  2. Multi-target pick-and-place: selector chooses ball order,
     base+arm pick each ball and deliver to assigned drop zones.
"""

import argparse
import json
import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

if "--headless" in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import numpy as np
import torch
import pinocchio as pin
from simulation_and_control import pb, MotorCommands
from robotnik_arm_task import (
    setup_simulation, setup_kinematics, add_target_visual,
    FIXED_FINGER_ANGLES, REACH_TOLERANCE, ARM_REACH_LIMIT,
    GRIP_TOLERANCE, GripState,
    DROP_ZONES_DEFAULT, DROP_ZONE_COLORS,
    spawn_ball, find_ee_link_id,
    auto_grip, auto_release,
    shortest_path_expert,
    build_selector_scene,
    generate_random_drop_zones,
)
from robotnik_arm_control import configure_gripper_fingers, quaternion2bearing
from run_policy import load_policy, predict
from regressors import build_mlp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.11
WHEEL_BASE_WIDTH = 0.66
SHOULDER_HEIGHT = 0.75

# Stall detection
STALL_WINDOW = 10.0    # seconds
STALL_EPSILON = 3e-3   # metres
BASE_STALL_WINDOW = 5.0
BASE_STALL_EPSILON = 0.05

# Phase switch heading tolerance
HEADING_TOL_SWITCH = 0.15  # rad (~8.6°)

# Wheel lock PD gains (velocity-mode braking during arm phase)
WHEEL_LOCK_KP = 200.0
WHEEL_LOCK_LIMIT = 20.0

# Hold time for target-reached check
HOLD_TIME = 1.0


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------
def build_base_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                           target_ee_pos, feature_names=None):
    """Assemble base policy observation vector.

    Linear velocities are rotated into the base frame so 'forward' and
    'lateral' are relative to the robot's heading.
    """
    delta = target_ee_pos[:2] - base_pos[:2]
    dist_to_target = np.linalg.norm(delta)
    heading_to_target = np.arctan2(delta[1], delta[0])
    heading_error = np.arctan2(np.sin(heading_to_target - base_bearing),
                               np.cos(heading_to_target - base_bearing))

    c, s = np.cos(base_bearing), np.sin(base_bearing)
    fwd_vel = c * base_lin_vel[0] + s * base_lin_vel[1]
    lat_vel = -s * base_lin_vel[0] + c * base_lin_vel[1]

    dx = target_ee_pos[0] - base_pos[0]
    dy = target_ee_pos[1] - base_pos[1]
    target_in_base_x = c * dx + s * dy
    target_in_base_y = -s * dx + c * dy

    ALL_BASE = {
        "base_pos_x": base_pos[0], "base_pos_y": base_pos[1],
        "base_bearing": base_bearing,
        "base_lin_vel_x": base_lin_vel[0], "base_lin_vel_y": base_lin_vel[1],
        "base_lin_vel_z": base_lin_vel[2],
        "base_ang_vel_x": base_ang_vel[0], "base_ang_vel_y": base_ang_vel[1],
        "base_ang_vel_z": base_ang_vel[2],
        "target_x": target_ee_pos[0], "target_y": target_ee_pos[1],
        "target_z": target_ee_pos[2],
        "dist_to_target": dist_to_target, "heading_error": heading_error,
        "fwd_vel": fwd_vel, "lat_vel": lat_vel,
        "target_in_base_x": target_in_base_x, "target_in_base_y": target_in_base_y,
    }

    if feature_names is None:
        feature_names = ["fwd_vel", "lat_vel", "base_ang_vel_z",
                         "target_in_base_x", "target_in_base_y"]

    return np.array([ALL_BASE[name] for name in feature_names])


def build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos,
                          base_pos=None, base_bearing=None,
                          feature_names=None):
    """Assemble arm policy observation vector.

    When base_pos/base_bearing are provided, ee and target positions are
    expressed in the base frame (yaw-rotated), making the observation
    independent of the robot's world-frame pose.
    """
    if base_pos is not None and base_bearing is not None:
        c, s = np.cos(base_bearing), np.sin(base_bearing)
        base_xy = base_pos[:2]
        ee_dx, ee_dy = ee_pos[0] - base_xy[0], ee_pos[1] - base_xy[1]
        tgt_dx, tgt_dy = target_ee_pos[0] - base_xy[0], target_ee_pos[1] - base_xy[1]
        ee_rel = np.array([c * ee_dx + s * ee_dy, -s * ee_dx + c * ee_dy, ee_pos[2]])
        tgt_rel = np.array([c * tgt_dx + s * tgt_dy, -s * tgt_dx + c * tgt_dy, target_ee_pos[2]])
    else:
        ee_rel, tgt_rel = ee_pos, target_ee_pos

    ALL_ARM = {}
    for i in range(7):
        ALL_ARM[f"q_arm_{i+1}"] = q_mes[4 + i]
        ALL_ARM[f"qd_arm_{i+1}"] = qd_mes[4 + i]
    ALL_ARM["ee_pos_x"] = ee_rel[0]
    ALL_ARM["ee_pos_y"] = ee_rel[1]
    ALL_ARM["ee_pos_z"] = ee_rel[2]
    ALL_ARM["target_x"] = tgt_rel[0]
    ALL_ARM["target_y"] = tgt_rel[1]
    ALL_ARM["target_z"] = tgt_rel[2]

    if feature_names is None:
        feature_names = (
            [f"q_arm_{i}" for i in range(1, 8)]
            + [f"qd_arm_{i}" for i in range(1, 8)]
            + ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
            + ["target_x", "target_y", "target_z"]
        )
    return np.array([ALL_ARM[name] for name in feature_names])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_base_threshold(target_ee_pos):
    """Base XY distance threshold for phase switching given a target height."""
    dz = abs(target_ee_pos[2] - SHOULDER_HEIGHT)
    if dz < ARM_REACH_LIMIT:
        return 0.85 * np.sqrt(ARM_REACH_LIMIT**2 - dz**2)
    return 0.3


def read_sensors(sim):
    """Read all sensor data from the simulation. Returns a dict."""
    base_pos = np.array(sim.GetBasePosition())
    base_ori = np.array(sim.GetBaseOrientation())
    base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
    q_mes = np.array(sim.GetMotorAngles(0))
    qd_mes = np.array(sim.GetMotorVelocities(0))
    base_lin_vel = np.array(sim.GetBaseLinVelocity())
    base_ang_vel = np.array(sim.GetBaseAngVelocity())
    return dict(base_pos=base_pos, base_ori=base_ori, base_bearing=base_bearing,
                q_mes=q_mes, qd_mes=qd_mes,
                base_lin_vel=base_lin_vel, base_ang_vel=base_ang_vel)


def compute_ee_pos(dyn_model, ee_frame_id, sensors):
    """Run forward kinematics and return the EE world position."""
    q_all = np.concatenate((sensors["base_pos"], sensors["base_ori"], sensors["q_mes"]))
    q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    return dyn_model.pin_data.oMf[ee_frame_id].translation.copy(), q_pin


def diff_drive(v_cmd, w_cmd):
    """Convert (v, w) to [rw, lw, lw, rw] wheel velocity commands."""
    lw = (v_cmd - w_cmd * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
    rw = (v_cmd + w_cmd * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
    return np.array([rw, lw, lw, rw])


def heading_error_to(base_pos, base_bearing, target_pos):
    """Signed heading error from base bearing to target XY position."""
    delta = target_pos[:2] - base_pos[:2]
    heading_to_target = np.arctan2(delta[1], delta[0])
    return np.arctan2(np.sin(heading_to_target - base_bearing),
                      np.cos(heading_to_target - base_bearing))


def load_selector(checkpoint_path, device="cpu"):
    """Load a trained target-selector scorer from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_mlp(ckpt["input_dim"], ckpt["output_dim"],
                      ckpt["hidden_dim"], ckpt["num_layers"],
                      ckpt.get("activation", "gelu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["input_mean"], ckpt["input_std"]


def predict_selector(sel_model, sel_mean, sel_std, scene_obs, slot_order, device="cpu"):
    """Pick the best target using the scene-level selector."""
    from robotnik_arm_task import MAX_TARGETS
    obs_part = scene_obs[:-MAX_TARGETS]
    mask = scene_obs[-MAX_TARGETS:]
    obs_norm = (obs_part - sel_mean) / sel_std
    x = np.concatenate([obs_norm, mask])
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sel_model(x_t).squeeze(0).cpu().numpy()
    logits[mask == 0] = -np.inf
    return slot_order[int(np.argmax(logits))]


# ---------------------------------------------------------------------------
# Core simulation loop — used by both single-target and multi-target modes
# ---------------------------------------------------------------------------
def run_single_target(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                      init_joint_angles, lower_limits, upper_limits,
                      target_ee_pos, base_threshold, max_steps,
                      device, policy_kwargs, args,
                      goal_tolerance=GRIP_TOLERANCE,
                      enable_stall_detection=True):
    """Run the robot towards one target position (no sim reset).

    Returns:
        dict with ok, final_err, best_err, reached, steps, time, phase_switched,
        base_best_ee, base_final_ee, arm_best_ee, arm_final_ee,
        base_xy_at_best, base_bearing_at_best, switch_time.
    """
    time_step = sim.GetTimeStep()
    current_time = 0.0
    cmd = MotorCommands()

    best_ee_error = np.inf
    step = 0
    phase_switched = False
    wheel_lock_pos = None

    in_goal_since = None
    held = False
    arm_phase_start_time = None

    # Per-phase tracking
    base_best_ee = np.inf
    base_final_ee = np.inf
    arm_best_ee = np.inf
    arm_final_ee = np.inf
    base_xy_at_best = None
    base_bearing_at_best = None
    switch_time = None

    # Stall tracking
    stall_snapshot_time = None
    stall_snapshot_error = None
    base_stall_snapshot_time = None
    base_stall_snapshot_error = None

    ee_pos = np.zeros(3)
    ee_dist_error = np.inf

    heading_threshold = getattr(args, "heading_threshold", 0.15)
    headless = getattr(args, "headless", True)

    # --- Policy setup ---
    base_model = policy_kwargs.get("base_model")
    base_mean = policy_kwargs.get("base_mean")
    base_std = policy_kwargs.get("base_std")
    base_label_mean = policy_kwargs.get("base_label_mean")
    base_label_std = policy_kwargs.get("base_label_std")
    base_feature_names = policy_kwargs.get("base_feature_names")
    has_base_phases = policy_kwargs.get("has_base_phases", True)

    arm_model = policy_kwargs.get("arm_model")
    arm_mean = policy_kwargs.get("arm_mean")
    arm_std = policy_kwargs.get("arm_std")
    arm_label_mean = policy_kwargs.get("arm_label_mean")
    arm_label_std = policy_kwargs.get("arm_label_std")
    arm_feature_names = policy_kwargs.get("arm_feature_names")

    phase = "base"
    base_sub_phase = "rotate"

    # --- Main simulation loop ---
    while step < max_steps:
        # Sense
        try:
            S = read_sensors(sim)
        except Exception as e:
            print(f"  Sim crashed at t={current_time:.2f}s: {e}")
            return {"ok": False, "final_err": float("inf"), "best_err": best_ee_error,
                    "reached": False, "steps": step, "time": current_time,
                    "phase_switched": phase_switched,
                    "base_best_ee": base_best_ee, "base_final_ee": base_final_ee,
                    "arm_best_ee": arm_best_ee, "arm_final_ee": arm_final_ee,
                    "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
                    "base_bearing_at_best": base_bearing_at_best, "switch_time": switch_time}

        ee_pos, q_pin = compute_ee_pos(dyn_model, ee_frame_id, S)
        ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)
        if ee_dist_error < best_ee_error:
            best_ee_error = ee_dist_error

        base_pos = S["base_pos"]
        base_bearing = S["base_bearing"]
        q_mes = S["q_mes"]
        qd_mes = S["qd_mes"]

        # Per-phase tracking
        if not phase_switched:
            base_final_ee = ee_dist_error
            if ee_dist_error < base_best_ee:
                base_best_ee = ee_dist_error
                base_xy_at_best = base_pos[:2].copy()
                base_bearing_at_best = base_bearing
        else:
            arm_final_ee = ee_dist_error
            if ee_dist_error < arm_best_ee:
                arm_best_ee = ee_dist_error

        # ------ Base phase ------
        if phase == "base":
            # Check phase switch
            base_xy_dist = np.linalg.norm(base_pos[:2] - target_ee_pos[:2])
            h_err = abs(heading_error_to(base_pos, base_bearing, target_ee_pos))
            heading_ok = h_err < HEADING_TOL_SWITCH or base_xy_dist < 0.1
            if base_xy_dist < base_threshold and heading_ok:
                phase_switched = True
                switch_time = current_time
                wheel_lock_pos = q_mes[0:4].copy()
                print(f"    Phase switch: base -> arm at t={current_time:.2f}s  "
                      f"xy_dist={base_xy_dist:.4f}  heading_err={np.degrees(h_err):.1f}\u00b0")
                if arm_model is None:
                    break
                phase = "arm"
                arm_phase_start_time = current_time
                stall_snapshot_time = current_time
                stall_snapshot_error = ee_dist_error

            if phase == "base":
                # Base observation
                obs_names = ([n for n in base_feature_names if n != "phase"]
                             if has_base_phases else base_feature_names)
                obs_base = build_base_observation(
                    base_pos, base_bearing, S["base_lin_vel"], S["base_ang_vel"],
                    target_ee_pos, feature_names=obs_names)

                # Rotate/drive sub-phase
                if has_base_phases:
                    h_err_signed = heading_error_to(base_pos, base_bearing, target_ee_pos)
                    abs_herr = abs(h_err_signed)
                    if base_sub_phase == "rotate" and abs_herr < heading_threshold:
                        base_sub_phase = "drive"
                    elif base_sub_phase == "drive" and abs_herr >= heading_threshold * 2:
                        base_sub_phase = "rotate"
                    obs_base = np.append(obs_base, 0.0 if base_sub_phase == "rotate" else 1.0)

                vw = predict(base_model, base_mean, base_std, obs_base,
                             device=device, label_mean=base_label_mean, label_std=base_label_std)

                # Output masking
                if has_base_phases:
                    if base_sub_phase == "rotate":
                        v_cmd, w_cmd = 0.0, vw[1]
                    else:
                        v_cmd = max(vw[0], 0.0)
                        w_cmd = np.clip(1.5 * h_err_signed, -0.3, 0.3)
                else:
                    v_cmd = vw[0] if abs(vw[0]) > 0.02 else 0.0
                    w_cmd = vw[1] if abs(vw[1]) > 0.05 else 0.0

                v_cmd = np.clip(v_cmd, -0.25, 0.25)
                w_cmd = np.clip(w_cmd, -1.0, 1.0)
                wheel_cmds = diff_drive(v_cmd, w_cmd)

                # Arm hold: target = current EE (gravity comp only)
                if arm_model is not None:
                    obs_hold = build_arm_observation(q_mes, qd_mes, ee_pos, ee_pos,
                                                     base_pos=base_pos, base_bearing=base_bearing,
                                                     feature_names=arm_feature_names)
                    torque_arm = predict(arm_model, arm_mean, arm_std, obs_hold,
                                         device=device, label_mean=arm_label_mean,
                                         label_std=arm_label_std)
                    torque_arm = np.clip(torque_arm, -20.0, 20.0)
                else:
                    torque_arm = np.zeros(7)

        # ------ Arm phase ------
        if phase == "arm":
            obs_arm = build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos,
                                            base_pos=base_pos, base_bearing=base_bearing,
                                            feature_names=arm_feature_names)
            torque_arm = predict(arm_model, arm_mean, arm_std, obs_arm,
                                 device=device, label_mean=arm_label_mean,
                                 label_std=arm_label_std)
            torque_arm = np.clip(torque_arm, -20.0, 20.0)
            wheel_cmds = np.clip(-WHEEL_LOCK_KP * (q_mes[0:4] - wheel_lock_pos),
                                 -WHEEL_LOCK_LIMIT, WHEEL_LOCK_LIMIT)

        # ------ Send commands ------
        cmd_all = np.concatenate((wheel_cmds, torque_arm, np.zeros(6)))
        cmd.SetControlCmd(cmd_all, ["velocity"] * 4 + ["torque"] * 13)
        try:
            sim.Step(cmd, "torque")
        except Exception as e:
            print(f"  Sim crashed at t={current_time:.2f}s: {e}")
            return {"ok": False, "final_err": float("inf"), "best_err": best_ee_error,
                    "reached": False, "steps": step, "time": current_time,
                    "phase_switched": phase_switched,
                    "base_best_ee": base_best_ee, "base_final_ee": base_final_ee,
                    "arm_best_ee": arm_best_ee, "arm_final_ee": arm_final_ee,
                    "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
                    "base_bearing_at_best": base_bearing_at_best, "switch_time": switch_time}

        # ------ Check target reached ------
        arm_min_time = (arm_phase_start_time or 0.0) + 4.0
        in_goal = ee_dist_error < goal_tolerance and current_time > arm_min_time
        if in_goal and in_goal_since is None:
            in_goal_since = current_time
        elif not in_goal and in_goal_since is not None:
            in_goal_since = None
        if in_goal_since is not None and current_time - in_goal_since >= HOLD_TIME:
            held = True
            print(f"    Target held at t={current_time:.2f}s")
            break

        # ------ Stall detection ------
        if enable_stall_detection:
            # Arm stall
            if stall_snapshot_time is not None:
                if current_time - stall_snapshot_time >= STALL_WINDOW:
                    if abs(ee_dist_error - stall_snapshot_error) < STALL_EPSILON:
                        print(f"    Arm stalled at t={current_time:.2f}s "
                              f"(best={best_ee_error:.4f}m, "
                              f"moved <{STALL_EPSILON*1000:.1f}mm in {STALL_WINDOW:.0f}s)")
                        break
                    stall_snapshot_time = current_time
                    stall_snapshot_error = ee_dist_error

            # Base stall (drive sub-phase only)
            if phase == "base" and base_sub_phase == "drive":
                bxy = np.linalg.norm(base_pos[:2] - target_ee_pos[:2])
                if base_stall_snapshot_time is None:
                    base_stall_snapshot_time = current_time
                    base_stall_snapshot_error = bxy
                elif current_time - base_stall_snapshot_time >= BASE_STALL_WINDOW:
                    if abs(bxy - base_stall_snapshot_error) < BASE_STALL_EPSILON:
                        print(f"    Base stalled at t={current_time:.2f}s "
                              f"(xy_dist={bxy:.4f}m, "
                              f"moved <{BASE_STALL_EPSILON*100:.0f}cm in {BASE_STALL_WINDOW:.0f}s)")
                        break
                    base_stall_snapshot_time = current_time
                    base_stall_snapshot_error = bxy
            elif phase == "base" and base_sub_phase == "rotate":
                base_stall_snapshot_time = None

        # ------ Logging ------
        if step % 1000 == 0:
            print(f"    t={current_time:.2f}s [{phase}] "
                  f"EE err: {ee_dist_error:.4f}m  best: {best_ee_error:.4f}m")

        if not headless:
            time.sleep(0.001)
        current_time += time_step
        step += 1

    return {
        "ok": True,
        "final_err": ee_dist_error,
        "best_err": best_ee_error,
        "reached": held,
        "steps": step,
        "time": current_time,
        "phase_switched": phase_switched,
        "base_best_ee": base_best_ee,
        "base_final_ee": base_final_ee,
        "arm_best_ee": arm_best_ee,
        "arm_final_ee": arm_final_ee,
        "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
        "base_bearing_at_best": base_bearing_at_best,
        "switch_time": switch_time,
    }


# ---------------------------------------------------------------------------
# Single-target entry point (creates fresh sim)
# ---------------------------------------------------------------------------
def run_sim(args, target_ee_pos, device, **policy_kwargs):
    """Run one single-target simulation (fresh sim). Returns result dict."""
    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    add_target_visual(client, target_ee_pos)

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    base_threshold = compute_base_threshold(target_ee_pos)

    print("=" * 60)
    print(f"Target EE Position : {target_ee_pos}")
    print(f"Initial EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Error (m)  : {np.linalg.norm(target_ee_pos - init_ee_pos):.4f}")
    print(f"Mode               : SPLIT (base -> arm at XY dist < {base_threshold:.2f}m)")
    print("=" * 60)

    result = run_single_target(
        sim, dyn_model, ee_frame_id, shoulder_frame_id,
        init_joint_angles, lower_limits, upper_limits,
        target_ee_pos, base_threshold, args.max_steps,
        device, policy_kwargs, args,
        goal_tolerance=REACH_TOLERANCE,
        enable_stall_detection=True,
    )

    print("=" * 60)
    print(f"Final Error (m) : {result['final_err']:.4f}")
    print(f"Best Error (m)  : {result['best_err']:.4f}")
    print(f"Reached         : {result['reached']}")
    print(f"Steps: {result['steps']}  Time: {result['time']:.2f}s")
    print("=" * 60)
    return result


# ---------------------------------------------------------------------------
# Pick-and-place: single ball cycle
# ---------------------------------------------------------------------------
def _pick_and_place_one_ball(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                             init_joint_angles, lower_limits, upper_limits,
                             ball_pos, drop_zone, grip_state,
                             device, policy_kwargs, args):
    """Pick one ball and deliver it to a drop zone. Returns per-ball result dict."""
    client = sim.GetPyBulletClient()
    rid = sim.bot[0].bot_pybullet
    ee_link_id = find_ee_link_id(client, rid)
    total_steps = 0
    total_time = 0.0

    # Phase 1: Navigate to ball and reach it
    result_ball = run_single_target(
        sim, dyn_model, ee_frame_id, shoulder_frame_id,
        init_joint_angles, lower_limits, upper_limits,
        ball_pos, compute_base_threshold(ball_pos),
        getattr(args, "per_target_max_steps", args.max_steps),
        device, policy_kwargs, args,
    )
    total_steps += result_ball.get("steps", 0)
    total_time += result_ball.get("time", 0.0)

    if not result_ball.get("ok", True):
        return {"ok": False, "gripped": False, "delivered": False,
                "steps": total_steps, "time": total_time}

    # Auto-grip check
    S = read_sensors(sim)
    ee_pos, _ = compute_ee_pos(dyn_model, ee_frame_id, S)
    ball_world_pos, _ = client.getBasePositionAndOrientation(grip_state.ball_id)
    grip_dist = np.linalg.norm(ee_pos - np.array(ball_world_pos))
    gripped = grip_dist < GRIP_TOLERANCE

    if gripped:
        auto_grip(client, rid, ee_link_id, grip_state)
        print(f"    Auto-gripped (dist={grip_dist:.4f}m)")
    else:
        print(f"    Grip failed (dist={grip_dist:.4f}m)")
        return {"ok": True, "gripped": False, "delivered": False,
                "grip_dist": grip_dist, "steps": total_steps, "time": total_time}

    # Phase 2: Navigate to drop zone and release
    result_drop = run_single_target(
        sim, dyn_model, ee_frame_id, shoulder_frame_id,
        init_joint_angles, lower_limits, upper_limits,
        drop_zone, compute_base_threshold(drop_zone),
        getattr(args, "per_target_max_steps", args.max_steps),
        device, policy_kwargs, args,
    )
    total_steps += result_drop.get("steps", 0)
    total_time += result_drop.get("time", 0.0)

    # Auto-release check
    S = read_sensors(sim)
    ee_pos_drop, _ = compute_ee_pos(dyn_model, ee_frame_id, S)
    drop_dist = np.linalg.norm(ee_pos_drop - drop_zone)
    auto_release(client, grip_state)
    delivered = drop_dist < GRIP_TOLERANCE
    print(f"    Released at drop zone (dist={drop_dist:.4f}m, delivered={delivered})")

    return {
        "ok": result_drop.get("ok", True),
        "gripped": True, "delivered": delivered,
        "grip_dist": grip_dist, "drop_dist": drop_dist,
        "pick_best_err": result_ball.get("best_err"),
        "pick_final_err": result_ball.get("final_err"),
        "drop_best_err": result_drop.get("best_err"),
        "drop_final_err": result_drop.get("final_err"),
        "steps": total_steps, "time": total_time,
    }


# ---------------------------------------------------------------------------
# Multi-target pick-and-place episode
# ---------------------------------------------------------------------------
def run_pick_and_place_sim(args, target_ee_positions, device,
                           drop_zones=None,
                           ball_zone_assignments=None,
                           sel_model=None, sel_mean=None, sel_std=None,
                           use_expert_selector=False, **policy_kwargs):
    """Run a full multi-ball pick-and-place episode."""
    import pybullet as p

    if drop_zones is None:
        drop_zones = [np.array(dz) for dz in DROP_ZONES_DEFAULT]
    else:
        drop_zones = [np.array(dz) for dz in drop_zones]

    n_balls = len(target_ee_positions)
    all_targets = np.array(target_ee_positions)

    if ball_zone_assignments is None:
        rng = np.random.default_rng(args.multi_target_seed)
        ball_zone_assignments = [int(rng.integers(0, len(drop_zones))) for _ in range(n_balls)]

    # Create simulation
    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    configure_gripper_fingers(sim, client, rid, np.array([0.1] * 6))
    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    # Spawn balls and drop zone visuals
    grip_states = []
    for i, t in enumerate(target_ee_positions):
        color = DROP_ZONE_COLORS[ball_zone_assignments[i]]
        ball_id = spawn_ball(client, t, color=color)
        grip_states.append(GripState(ball_id=ball_id, drop_zone_idx=ball_zone_assignments[i]))
        add_target_visual(client, t)

    for zi, dz in enumerate(drop_zones):
        color = DROP_ZONE_COLORS[zi]
        vis = client.createVisualShape(p.GEOM_SPHERE, radius=0.08,
                                       rgbaColor=[color[0], color[1], color[2], 0.5])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=dz.tolist())

    # Episode header
    print(f"\n{'='*60}")
    print(f"Multi-drop-zone PnP: {n_balls} balls, {len(drop_zones)} zones")
    for i, t in enumerate(target_ee_positions):
        print(f"  Ball {i}: {np.round(t, 3)} -> zone {ball_zone_assignments[i]}")
    for zi, dz in enumerate(drop_zones):
        print(f"  Zone {zi}: {np.round(dz, 3)}")
    print(f"{'='*60}")

    remaining = set(range(n_balls))
    per_ball_results = []
    visit_order = []
    total_steps = 0
    total_time = 0.0
    robot_start_pos = read_sensors(sim)["base_pos"]

    while remaining:
        # Select next ball
        S = read_sensors(sim)
        if use_expert_selector:
            chosen_idx, _ = shortest_path_expert(
                S["base_pos"], all_targets, remaining, drop_zones, ball_zone_assignments)
        else:
            scene_obs, slot_order = build_selector_scene(
                S["base_pos"], S["base_bearing"], all_targets, remaining,
                drop_zones, ball_zone_assignments)
            chosen_idx = predict_selector(sel_model, sel_mean, sel_std,
                                           scene_obs, slot_order, device)

        ball_pos = all_targets[chosen_idx]
        assigned_drop = drop_zones[ball_zone_assignments[chosen_idx]]
        remaining.remove(chosen_idx)
        visit_order.append(chosen_idx)

        print(f"\n>>> Selector chose ball {chosen_idx} ({np.round(ball_pos, 3)}) "
              f"-> zone {ball_zone_assignments[chosen_idx]} | {len(remaining)} remaining")

        # Pick and place this ball
        result_pick = _pick_and_place_one_ball(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles, lower_limits, upper_limits,
            ball_pos, assigned_drop, grip_states[chosen_idx],
            device, policy_kwargs, args,
        )
        total_steps += result_pick.get("steps", 0)
        total_time += result_pick.get("time", 0.0)

        # Arm reset: move arm back to home EE position between targets
        if remaining and result_pick.get("ok", True):
            _reset_arm_to_home(
                sim, dyn_model, ee_frame_id, shoulder_frame_id,
                init_joint_angles, lower_limits, upper_limits,
                init_ee_pos, device, policy_kwargs, args)

        per_ball_results.append({
            "slot": chosen_idx,
            "ball_pos": ball_pos.tolist(),
            "drop_zone_idx": ball_zone_assignments[chosen_idx],
            "drop_zone_pos": assigned_drop.tolist(),
            **result_pick,
        })
        if not result_pick.get("ok", True):
            break

    # Summary
    n_gripped = sum(1 for r in per_ball_results if r.get("gripped", False))
    n_delivered = sum(1 for r in per_ball_results if r.get("delivered", False))

    # Selector travel distance: sum of XY hops for actual visit order
    def _travel_distance(order, start_pos):
        dist = 0.0
        pos = np.array(start_pos[:2])
        for idx in order:
            dist += np.linalg.norm(pos - all_targets[idx, :2])
            dz = drop_zones[ball_zone_assignments[idx]]
            dist += np.linalg.norm(all_targets[idx, :2] - dz[:2])
            pos = dz[:2].copy()
        return dist

    selector_dist = _travel_distance(visit_order, robot_start_pos)
    _, expert_order = shortest_path_expert(
        robot_start_pos, all_targets, set(range(n_balls)),
        drop_zones, ball_zone_assignments)
    expert_dist = _travel_distance(expert_order, robot_start_pos)
    dist_overhead = 100.0 * (selector_dist - expert_dist) / max(expert_dist, 1e-6)

    print(f"\n{'='*60}")
    print(f"PnP complete: {n_gripped}/{n_balls} gripped, {n_delivered}/{n_balls} delivered")
    print(f"Visit order: {visit_order}  (expert: {expert_order})")
    print(f"Selector travel: {selector_dist:.2f}m  Expert: {expert_dist:.2f}m  "
          f"Overhead: {dist_overhead:+.1f}%")
    zone_counts = {}
    for r in per_ball_results:
        zi = r.get("drop_zone_idx", 0)
        zone_counts.setdefault(zi, {"total": 0, "delivered": 0})
        zone_counts[zi]["total"] += 1
        if r.get("delivered", False):
            zone_counts[zi]["delivered"] += 1
    for zi in sorted(zone_counts):
        zc = zone_counts[zi]
        print(f"  Zone {zi}: {zc['delivered']}/{zc['total']} delivered")
    print(f"Total steps: {total_steps}  Total time: {total_time:.2f}s")
    print(f"{'='*60}")

    return {
        "ok": all(r.get("ok", True) for r in per_ball_results),
        "balls_gripped": n_gripped,
        "balls_delivered": n_delivered,
        "n_balls": n_balls,
        "visit_order": visit_order,
        "expert_order": expert_order,
        "ball_zone_assignments": ball_zone_assignments,
        "selector_dist": selector_dist,
        "expert_dist": expert_dist,
        "dist_overhead_pct": dist_overhead,
        "total_steps": total_steps,
        "total_time": total_time,
        "per_ball": per_ball_results,
    }


def _reset_arm_to_home(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                        init_joint_angles, lower_limits, upper_limits,
                        init_ee_pos, device, policy_kwargs, args):
    """Move the arm back to its init EE position using the arm policy."""
    S = read_sensors(sim)
    bearing = S["base_bearing"]
    base_pos = S["base_pos"]

    # Compute home EE in world frame at the robot's current location
    c, s = np.cos(bearing), np.sin(bearing)
    home_ee = base_pos.copy()
    home_ee[0] += c * init_ee_pos[0] - s * init_ee_pos[1]
    home_ee[1] += s * init_ee_pos[0] + c * init_ee_pos[1]
    home_ee[2] = init_ee_pos[2]
    print(f"    Resetting arm to home EE: {np.round(home_ee, 3)}")

    result = run_single_target(
        sim, dyn_model, ee_frame_id, shoulder_frame_id,
        init_joint_angles, lower_limits, upper_limits,
        home_ee, 999.0,  # large threshold to skip base nav
        5000,             # limited step budget
        device, policy_kwargs, args,
        enable_stall_detection=True,
    )
    print(f"    Arm reset done (err={result.get('best_err', 0):.4f}m)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run BC policy in simulation")
    parser.add_argument("--base_checkpoint", type=str,
                        default="060_bc_base_gelu_e70_h512_l3_16min_working.pt")
    parser.add_argument("--arm_checkpoint", type=str,
                        default="051_bc_arm_gelu_e70_h512_l3_3min_working.pt")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 1.1, 0.5],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--max_steps", type=int, default=400000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--heading_threshold", type=float, default=0.15)
    # Pick-and-place
    parser.add_argument("--pick_and_place", action="store_true",
                        help="Multi-target pick-and-place mode")
    parser.add_argument("--num_targets_per_episode", type=int, default=3)
    parser.add_argument("--selector_checkpoint", type=str,
                        default="084_selector_gelu_e300_h128_l3.pt")
    parser.add_argument("--multi_target_seed", type=int, default=None)
    parser.add_argument("--per_target_max_steps", type=int, default=100000)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load policies
    base_model, base_mean, base_std, base_lm, base_ls, base_fnames = \
        load_policy(args.base_checkpoint, device=device)
    arm_model, arm_mean, arm_std, arm_lm, arm_ls, arm_fnames = \
        load_policy(args.arm_checkpoint, device=device)
    has_base_phases = base_fnames is not None and "phase" in list(base_fnames)

    print(f"Base: {args.base_checkpoint}  Arm: {args.arm_checkpoint}  device={device}  "
          f"phases={'yes' if has_base_phases else 'no'}")

    policy_kwargs = dict(
        base_model=base_model, base_mean=base_mean, base_std=base_std,
        base_label_mean=base_lm, base_label_std=base_ls,
        base_feature_names=base_fnames, has_base_phases=has_base_phases,
        arm_model=arm_model, arm_mean=arm_mean, arm_std=arm_std,
        arm_label_mean=arm_lm, arm_label_std=arm_ls, arm_feature_names=arm_fnames,
    )

    # --- Multi-target pick-and-place ---
    if args.pick_and_place:
        sel_model = sel_mean = sel_std = None
        use_expert_selector = not args.selector_checkpoint
        if args.selector_checkpoint:
            sel_model, sel_mean, sel_std = load_selector(args.selector_checkpoint, device=device)
            print(f"Selector: {args.selector_checkpoint}")
        else:
            print("Selector: expert (shortest-path)")

        seed = (args.multi_target_seed if args.multi_target_seed is not None
                else int(np.random.default_rng().integers(0, 2**31)))
        print(f"Multi-target seed: {seed}")

        # Build config: from file or generate randomly
        if args.config:
            with open(args.config) as f:
                cfg = json.load(f)
        else:
            rng = np.random.default_rng(seed)
            n = args.num_targets_per_episode
            MIN_TARGET_DIST = 1.5
            balls = []
            for _ in range(n):
                while True:
                    t = [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.2, 1.0)]
                    if np.linalg.norm(t[:2]) >= MIN_TARGET_DIST:
                        break
                balls.append(t)

            zones = [dz.tolist() for dz in
                     generate_random_drop_zones(np.random.default_rng(seed + 1200))]

            rng_assign = np.random.default_rng(seed + 20030)
            cfg = {
                "balls": balls,
                "zones": zones,
                "assignments": [int(rng_assign.integers(0, len(zones))) for _ in range(n)],
            }

        targets = [np.array(p) for p in cfg["balls"]]
        mdz_zones = [np.array(dz) for dz in cfg["zones"]]
        ball_zone_assignments = cfg.get("assignments")
        if ball_zone_assignments is None:
            rng_a = np.random.default_rng(seed + 3000)
            ball_zone_assignments = [int(rng_a.integers(0, len(mdz_zones)))
                                     for _ in range(len(targets))]

        run_pick_and_place_sim(
            args, targets, device=device,
            drop_zones=mdz_zones, ball_zone_assignments=ball_zone_assignments,
            sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
            use_expert_selector=use_expert_selector, **policy_kwargs)
        return

    # --- Single-target mode ---
    target_ee_pos = np.array(args.target)

    run_sim(args, target_ee_pos, device=device, **policy_kwargs)


if __name__ == "__main__":
    main()
