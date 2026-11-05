#!/usr/bin/env python3
"""Physical-contact DIAGNOSTIC with a trained sensor-only close/reverse subpolicy.
Approach switching and arm holding remain hybrid. This is NOT an end-to-end
learned grasp controller. No virtual object attachment or base constraint.
Original source/checkpoints are not modified. Simulator integration unverified.
"""

import argparse
import os
import sys
import time

# Parse --headless early so the env var is set before module-level reads in robotnik_arm_task
if "--headless" in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import numpy as np
import torch
import pinocchio as pin
from simulation_and_control import pb, MotorCommands
from robotnik_arm_task import (
    setup_simulation, setup_kinematics, add_target_visual,
    FIXED_FINGER_ANGLES, REACH_TOLERANCE, ARM_REACH_LIMIT,
)
from robotnik_arm_control import (
    configure_gripper_fingers, quaternion2bearing,
)
from run_policy import load_policy, predict
from regressors import build_mlp
from activations import LEGACY_ACTIVATION


def load_gravity_comp(path, device="cpu"):
    """Load gravity compensation network from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hidden_dim = ckpt["hidden_dim"]
    num_layers = ckpt["num_layers"]
    activation = ckpt.get("activation", LEGACY_ACTIVATION)
    model = build_mlp(7, 7, hidden_dim, num_layers, activation)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["q_mean"], ckpt["q_std"], ckpt["tau_mean"], ckpt["tau_std"]


def predict_gravity(grav_model, q_mean, q_std, tau_mean, tau_std, q_arm, device="cpu"):
    """Predict gravity compensation torques from joint angles."""
    q_norm = (q_arm - q_mean) / q_std
    x = torch.tensor(q_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = grav_model(x).cpu().numpy()[0]
    return pred * tau_std + tau_mean

# Wheel kinematic constants (from BaseControllerParams)
WHEEL_RADIUS = 0.11
WHEEL_BASE_WIDTH = 0.66
# Approximate height of arm shoulder joint above ground (from robot URDF)
SHOULDER_HEIGHT = 0.75


def build_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                      q_mes, qd_mes, ee_pos, target_ee_pos, tau_gravity):
    """Assemble the 44-dim observation vector matching the training data format."""
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
        tau_gravity,                # 7: gravity compensation torques
    ])  # total: 44


def build_base_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                           target_ee_pos, feature_names=None):
    """Assemble base policy observation, optionally filtered by feature_names.

    Linear velocities are rotated into the base frame so that 'forward' and
    'lateral' are always relative to the robot's heading.
    """
    delta = target_ee_pos[:2] - base_pos[:2]
    dist_to_target = np.linalg.norm(delta)
    heading_to_target = np.arctan2(delta[1], delta[0])
    heading_error = np.arctan2(np.sin(heading_to_target - base_bearing),
                               np.cos(heading_to_target - base_bearing))

    # Rotate world-frame linear velocity into base frame
    c, s = np.cos(base_bearing), np.sin(base_bearing)
    fwd_vel = c * base_lin_vel[0] + s * base_lin_vel[1]   # forward
    lat_vel = -s * base_lin_vel[0] + c * base_lin_vel[1]   # lateral (left)

    # Target in base frame (Cartesian, position-agnostic)
    dx = target_ee_pos[0] - base_pos[0]
    dy = target_ee_pos[1] - base_pos[1]
    target_in_base_x = c * dx + s * dy    # forward distance
    target_in_base_y = -s * dx + c * dy   # lateral distance

    # All possible base features
    ALL_BASE = {
        "base_pos_x": base_pos[0],
        "base_pos_y": base_pos[1],
        "base_bearing": base_bearing,
        "base_lin_vel_x": base_lin_vel[0],
        "base_lin_vel_y": base_lin_vel[1],
        "base_lin_vel_z": base_lin_vel[2],
        "base_ang_vel_x": base_ang_vel[0],
        "base_ang_vel_y": base_ang_vel[1],
        "base_ang_vel_z": base_ang_vel[2],
        "target_x": target_ee_pos[0],
        "target_y": target_ee_pos[1],
        "target_z": target_ee_pos[2],
        "dist_to_target": dist_to_target,
        "heading_error": heading_error,
        "fwd_vel": fwd_vel,
        "lat_vel": lat_vel,
        "target_in_base_x": target_in_base_x,
        "target_in_base_y": target_in_base_y,
    }

    if feature_names is None:
        # Default: base-frame velocities + Cartesian target in base frame (5-dim, fully generalizable)
        feature_names = [
            "fwd_vel", "lat_vel",
            "base_ang_vel_z",
            "target_in_base_x", "target_in_base_y",
        ]

    return np.array([ALL_BASE[name] for name in feature_names])


def build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos,
                          base_pos=None, base_bearing=None,
                          feature_names=None):
    """Assemble arm policy observation, optionally filtered by feature_names.

    If base_pos and base_bearing are provided, ee_pos and target_ee_pos are
    expressed in the base frame (translation + yaw rotation), making the
    observation independent of the base's world-frame pose.
    """
    if base_pos is not None and base_bearing is not None:
        c, s = np.cos(base_bearing), np.sin(base_bearing)
        # Rotate xy into base frame, keep z as world (base z not in training data)
        base_xy = base_pos[:2]
        ee_dx, ee_dy = ee_pos[0] - base_xy[0], ee_pos[1] - base_xy[1]
        tgt_dx, tgt_dy = target_ee_pos[0] - base_xy[0], target_ee_pos[1] - base_xy[1]
        ee_rel = np.array([c * ee_dx + s * ee_dy, -s * ee_dx + c * ee_dy, ee_pos[2]])
        tgt_rel = np.array([c * tgt_dx + s * tgt_dy, -s * tgt_dx + c * tgt_dy, target_ee_pos[2]])
    else:
        ee_rel = ee_pos
        tgt_rel = target_ee_pos

    ALL_ARM = {}
    for i in range(7):
        ALL_ARM[f"q_arm_{i+1}"] = q_mes[4 + i]
    for i in range(7):
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


def run_sim(args, target_ee_pos, device, split_mode, single_mode,
            # single-mode policy
            model=None, mean=None, std=None,
            label_mean=None, label_std=None,
            # split-mode policies
            base_model=None, base_mean=None, base_std=None,
            base_label_mean=None, base_label_std=None, base_feature_names=None,
            has_base_phases=False,
            arm_model=None, arm_mean=None, arm_std=None,
            arm_label_mean=None, arm_label_std=None, arm_feature_names=None,
            # gravity compensation
            grav_model=None, grav_q_mean=None, grav_q_std=None,
            grav_tau_mean=None, grav_tau_std=None):
    """Run one simulation attempt. Returns True if finished normally, False if crashed."""

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, np.array(FIXED_FINGER_ANGLES, dtype=float, copy=True))
    _before_marker = {
        client.getBodyUniqueId(i)
        for i in range(client.getNumBodies())
    }

    add_target_visual(client, target_ee_pos)

    target_marker_bodies = {
        client.getBodyUniqueId(i)
        for i in range(client.getNumBodies())
    } - _before_marker

        # ============================================
    # PHYSICAL BALL AT THE TARGET POSITION
    # ============================================

    p = sim.GetPyBulletClient()

    ball_radius = 0.03       # 3 cm radius
    ball_mass = 0.05         # 50 grams
    ball_position = target_ee_pos.tolist()

    # Grasp-only air test: no board/support and no virtual attachment.
    ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius,
                                      rgbaColor=[1, 0, 0, 1])
    # Keep it kinematic while the robot approaches/closes, so incidental palm
    # contact cannot knock it away. It becomes dynamic only after finger contact.
    ball_dynamic = False
    ball_id = p.createMultiBody(baseMass=0,
        baseCollisionShapeIndex=ball_collision, baseVisualShapeIndex=ball_visual,
        basePosition=ball_position)
    p.changeDynamics(ball_id, -1, lateralFriction=1.2, spinningFriction=0.01,
                     rollingFriction=0.001, restitution=0)
    print('GRASP-ONLY AIR TEST: ball suspended until finger contact; no board or attachment')
    print(f"Red ball created at {ball_position} - no rod")

    # GRASP-AND-REVERSE DEMO STATE
    # Physical finger contact is evaluated; successful grasp is not assumed.
    from robotnik_arm_control import FINGER_JOINT_NAMES
    finger_joint_ids = [
        jid for jid in range(client.getNumJoints(rid))
        if client.getJointInfo(rid, jid)[1].decode() in FINGER_JOINT_NAMES
    ]
    if len(finger_joint_ids) != 6:
        raise RuntimeError(f"Expected 6 finger joints, found {len(finger_joint_ids)}: {finger_joint_ids}")
    finger_open = target_finger_angles.copy()
    finger_closed = finger_open.copy()
    for idx, jid in enumerate(finger_joint_ids):
        info = client.getJointInfo(rid, jid)
        low, high = float(info[8]), float(info[9])
        # Most Kinova finger joints close by increasing their angle. Verify visually.
        finger_closed[idx] = (low + 0.78 * (high - low)) if high > low else (finger_open[idx] + 0.55)
    grasp_state = "approach"
    # Load learned action policy ONCE, without retraining base/arm.
    from sensor_bc import sensor_features, SensorPolicy
    grasp_bc_policy = SensorPolicy(args.grasp_checkpoint)
    print(f"LEARNED GRIP+REVERSE: {args.grasp_checkpoint}")

    v10_ball_clearance_enabled = False
    grasp_start = None
    grasp_joint_hold = None
    carry_start_xy = None
    ball_grasp_constraint = None
    gripper_link = next((
        jid for jid in range(client.getNumJoints(rid))
        if 'end_effector' in client.getJointInfo(rid, jid)[12].decode().lower()
    ), None)
    if gripper_link is None:
        raise RuntimeError("No end-effector link found; cannot safely attach ball to a guessed link.")
    print("GRASP-ONLY: learned approach + sensor-BC finger closure; reverse disabled")


    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    time_step = sim.GetTimeStep()
    current_time = 0.0
    cmd = MotorCommands()
    import atexit
    from pathlib import Path
    full_obs, full_actions, full_times = [], [], []
    if args.record_full_npz and Path(args.record_full_npz).exists():
        raise FileExistsError(args.record_full_npz)
    def save_full_trace():
        if not args.record_full_npz or not full_obs:
            return
        with Path(args.record_full_npz).open('xb') as output:
            np.savez_compressed(output,
                format=np.asarray('robotnik_physical_trace_v1'),
                obs=np.asarray(full_obs,dtype=np.float32),
                action=np.asarray(full_actions,dtype=np.float32),
                time=np.asarray(full_times), virtual_attachment=np.asarray(False),
                success_verified=np.asarray(False),
                policy_scope=np.asarray('hybrid_approach_sensor_bc_close_reverse'))
        print(f'Saved physical trace: {args.record_full_npz}; success not verified')
    atexit.register(save_full_trace)

    print("=" * 60)
    print(f"Target EE Position : {target_ee_pos}")
    print(f"Initial EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Error (m)  : {np.linalg.norm(target_ee_pos - init_ee_pos):.4f}")
    if split_mode:
        if args.base_threshold is not None:
            print(f"Mode               : SPLIT (base -> arm at XY dist < {args.base_threshold}m)")
        else:
            print(f"Mode               : SPLIT (base -> arm on settle)")
    else:
        print(f"Mode               : SINGLE")
    print("=" * 60)

    best_ee_error = np.inf
    step = 0
    phase = "base" if split_mode else "single"
    phase_switched = False

    # Only stop once the EE has stayed inside the goal for HOLD_TIME seconds in a row;
    # drifting out resets the timer and the simulation keeps running
    HOLD_TIME = 1.0
    first_reach_time = None
    in_goal_since = None
    times_drifted_out = 0
    held = False
    grasp_contact_since = None
    previous_ball_relative = None
    grasp_success = False

    # Phase switch tracking: settle detection

    # Per-phase tracking
    base_best_ee = np.inf
    base_final_ee = np.inf
    base_xy_at_switch = None
    base_xy_at_best = None
    base_bearing_at_best = None
    arm_best_ee = np.inf
    arm_final_ee = np.inf
    switch_time = None

    # Rotate/drive sub-phase state (persisted across steps for hysteresis)
    base_sub_phase = "rotate"  # start in rotate

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
            return {"ok": False, "final_err": float("inf"), "best_err": best_ee_error,
                    "reached": False, "steps": step, "time": current_time,
                    "target": target_ee_pos.tolist(), "phase_switched": phase_switched,
                    "switch_time": switch_time, "base_best_ee": base_best_ee,
                    "base_final_ee": base_final_ee,
                    "base_xy_at_switch": base_xy_at_switch.tolist() if base_xy_at_switch is not None else None,
                    "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
                    "base_bearing_at_best": base_bearing_at_best,
                    "arm_best_ee": arm_best_ee, "arm_final_ee": arm_final_ee}

        # Forward kinematics for EE position
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)

        tau_gravity_arm = np.zeros(7)  # not available at inference

        if ee_dist_error < best_ee_error:
            best_ee_error = ee_dist_error


        # V8: Move the ARM toward the ball, not the ball into the arm.
        approach_ee_target = target_ee_pos
        pocket_gap = float("inf")

        if phase == "arm" and grasp_state == "approach":
            roots, tips = [], []

            for jid in finger_joint_ids:
                name = client.getJointInfo(rid, jid)[1].decode().lower()

                xyz = np.asarray(
                    client.getLinkState(
                        rid, jid, computeForwardKinematics=True
                    )[4],
                    dtype=float
                )

                if "finger_tip_" in name:
                    tips.append(xyz)
                elif "finger_" in name:
                    roots.append(xyz)

            if len(roots) != 3 or len(tips) != 3:
                raise RuntimeError("Finger geometry unavailable")

            pocket = (
                0.4 * np.mean(roots, axis=0)
                + 0.6 * np.mean(tips, axis=0)
            )

            ball_now = np.asarray(
                p.getBasePositionAndOrientation(ball_id)[0]
            )

            pocket_gap = float(np.linalg.norm(pocket - ball_now))

            # Adjust only the final arm approach, not base navigation.
            if ee_dist_error < 0.15:
                candidate = ball_now - (pocket - ee_pos)

                if np.linalg.norm(candidate - target_ee_pos) < 0.10:
                    approach_ee_target = candidate

            if step % 500 == 0:
                print(
                    f"ALIGN: pocket-to-ball={pocket_gap:.3f}m"
                )


        # Scripted handover from approach to sensor-only close/reverse policy.
        # Start closing only near the best observed pocket alignment; an early
        # close made the arm drift away before the finger pads reached the ball.
        if phase == "arm" and grasp_state == "approach" and pocket_gap < 0.050:
            print(
                f">>> V10 GRASP: pocket={np.round(pocket, 3)}, "
                f"ball={np.round(ball_now, 3)}, "
                f"gap={pocket_gap:.3f}m"
            )
            ball_pos, ball_ori = p.getBasePositionAndOrientation(ball_id)
            ball_ee_distance = float(np.linalg.norm(np.asarray(ball_pos) - ee_pos))
            if ball_ee_distance > 0.085:
                print(f"Ball is {ball_ee_distance:.3f} m from EE; "
                      "cannot secure reliably. Stopping.")
                break

            grasp_state = "closing"
            grasp_start = current_time
            grasp_joint_hold = q_mes[4:11].copy()
            print(f">>> CLOSE FINGERS at t={current_time:.2f}s, "
                  f"ball-to-EE={ball_ee_distance:.3f} m")


            # Hand over once to the sensor-only learned close/reverse policy.
            # This trigger is still scripted; the subsequent wheel/finger mapping is learned.
            carry_start_xy = base_pos[:2].copy()
            carry_heading = np.array([np.cos(base_bearing), np.sin(base_bearing)])
            print('SENSOR POLICY ACTIVE: no close timer, no reverse-phase input')

        # ---- Phase transition check (split mode only) ----
        if phase == "base":
            base_speed = np.linalg.norm(base_lin_vel[:2])
            switch = False
            switch_reason = ""

            base_xy_dist = np.linalg.norm(base_pos[:2] - target_ee_pos[:2])
            if base_xy_dist < args.base_threshold:
                switch = True
                switch_reason = f"xy_dist={base_xy_dist:.4f}m"

            # Track base phase metrics
            if ee_dist_error < base_best_ee:
                base_best_ee = ee_dist_error
                base_xy_at_best = base_pos[:2].copy()
                base_bearing_at_best = base_bearing
            base_final_ee = ee_dist_error

            if switch:
                phase_switched = True
                switch_time = current_time
                base_xy_at_switch = base_pos[:2].copy()
                wheel_lock_pos = q_mes[0:4].copy()
                # Base is not artificially fixed to the world.
                _p = sim.GetPyBulletClient()
                _base_pos_ori = _p.getBasePositionAndOrientation(sim.bot[0].bot_pybullet)
                base_lock_cid = None  # no artificial base lock
                print(f">>> Phase switch: base -> arm at t={current_time:.2f}s  {switch_reason}")
                if arm_model is None:
                    # No arm policy (base_only mode) — stop here
                    break
                phase = "arm"

        # ---- Compute action based on mode/phase ----
        if phase == "single":
            obs = build_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                                    q_mes, qd_mes, ee_pos, target_ee_pos, tau_gravity_arm)
            action = predict(model, mean, std, obs, device=device,
                             label_mean=label_mean, label_std=label_std)
            wheel_cmds = np.clip(action[:4], -0.25 / WHEEL_RADIUS, 0.25 / WHEEL_RADIUS)
            torque_arm = np.clip(action[4:], -30.0, 30.0)

        elif phase == "base":
            # Build base observation (without phase input)
            obs_feature_names = [n for n in base_feature_names if n != "phase"] \
                if has_base_phases else base_feature_names
            obs_base = build_base_observation(base_pos, base_bearing, base_lin_vel,
                                              base_ang_vel, target_ee_pos,
                                              feature_names=obs_feature_names)

            # Heading-based phase switching
            if has_base_phases:
                delta_base = target_ee_pos[:2] - base_pos[:2]
                heading_to_target = np.arctan2(delta_base[1], delta_base[0])
                heading_err = np.arctan2(np.sin(heading_to_target - base_bearing),
                                         np.cos(heading_to_target - base_bearing))
                abs_herr = abs(heading_err)
                if base_sub_phase == "rotate" and abs_herr < args.heading_threshold:
                    base_sub_phase = "drive"
                elif base_sub_phase == "drive" and abs_herr >= args.heading_threshold * 2:
                    base_sub_phase = "rotate"
                phase_val = 0.0 if base_sub_phase == "rotate" else 1.0
                obs_base = np.append(obs_base, phase_val)

            vw = predict(base_model, base_mean, base_std, obs_base,
                         device=device,
                         label_mean=base_label_mean, label_std=base_label_std)

            # Hard output masking at inference
            if has_base_phases:
                if base_sub_phase == "rotate":
                    v_cmd = 0.0
                    w_cmd = vw[1]
                else:  # drive
                    v_cmd = max(vw[0], 0.0)  # forward only
                    w_cmd = 0.0
            else:
                v_cmd = vw[0] if abs(vw[0]) > 0.02 else 0.0
                w_cmd = vw[1] if abs(vw[1]) > 0.05 else 0.0

            v_cmd = np.clip(v_cmd, -0.25, 0.25)
            w_cmd = np.clip(w_cmd, -1.0, 1.0)
            lw = (v_cmd - w_cmd * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
            rw = (v_cmd + w_cmd * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
            wheel_cmds = np.array([rw, lw, lw, rw])

            # Arm hold: run arm policy with target=current EE so it only outputs gravity torques
            if arm_model is not None:
                obs_arm_hold = build_arm_observation(q_mes, qd_mes, ee_pos, ee_pos,
                                                     base_pos=base_pos, base_bearing=base_bearing,
                                                     feature_names=arm_feature_names)
                torque_arm = predict(arm_model, arm_mean, arm_std, obs_arm_hold,
                                     device=device,
                                     label_mean=arm_label_mean, label_std=arm_label_std)
                torque_arm = np.clip(torque_arm, -20.0, 20.0)
            else:
                torque_arm = np.zeros(7)

        elif phase == "arm":
            # Arm policy
            hold_ee_target = ee_pos if grasp_state in ("closing", "reverse") else approach_ee_target
            obs_arm = build_arm_observation(q_mes, qd_mes, ee_pos, hold_ee_target,
                                            base_pos=base_pos, base_bearing=base_bearing,
                                            feature_names=arm_feature_names)
            raw_torque = predict(arm_model, arm_mean, arm_std, obs_arm,
                                 device=device,
                                 label_mean=arm_label_mean, label_std=arm_label_std)
            # Add gravity compensation if using learned gravity network
            if grav_model is not None:
                tau_grav = predict_gravity(grav_model, grav_q_mean, grav_q_std,
                                           grav_tau_mean, grav_tau_std,
                                           q_mes[4:11], device=device)
                raw_torque = raw_torque + tau_grav
            if args.arm_damping > 0:
                raw_torque = raw_torque - args.arm_damping * qd_mes[4:11]
            torque_arm = np.clip(raw_torque, -20.0, 20.0)

            wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - wheel_lock_pos), -5.0, 5.0)
            # Track arm phase metrics
            if ee_dist_error < arm_best_ee:
                arm_best_ee = ee_dist_error
            arm_final_ee = ee_dist_error

        # ---- GRASP / CARRY OVERRIDES ----
        if grasp_state in ("closing", "reverse"):
            # Keep the arm at its grasp pose while the learned finger targets close.
            torque_arm = np.clip(
                torque_arm + 18.0 * (grasp_joint_hold - q_mes[4:11]) - 2.5 * qd_mes[4:11],
                -20.0, 20.0)
        # The network gets physical sensors only. It supplies both wheel and
        # finger commands immediately after handover, without a scripted close delay.
        if grasp_state == "closing":
            ball_xyz = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
            obs_ml = sensor_features(ee_pos, ball_xyz, q_mes[11:17], qd_mes[11:17],
                                     base_lin_vel[:2], base_bearing)
            action_ml = grasp_bc_policy.predict(obs_ml)
            target_finger_angles[:] = action_ml[:6]
            # This task ends at grasp. Ignore the network's reverse-wheel outputs.
            # Hold the mobile base at its handover pose while fingers close.
            wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - wheel_lock_pos), -5.0, 5.0)

        # Log observable physics, never claim success from robot motion alone.
        ball_xyz = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
        robot_contacts = p.getContactPoints(bodyA=rid, bodyB=ball_id)
        finger_contacts = [cp for cp in robot_contacts if cp[3] in finger_joint_ids]
        contact_groups = sum(any(cp[3] in finger_joint_ids[i:i+2]
                             for cp in finger_contacts) for i in (0, 2, 4))
        # Release into normal Bullet dynamics only after distinct finger groups
        # touch it. This prevents an approach/palm collision from launching it.
        if not ball_dynamic and grasp_state == "closing" and contact_groups >= 2:
            p.changeDynamics(ball_id, -1, mass=ball_mass)
            ball_dynamic = True
            p.resetBaseVelocity(ball_id, [0, 0, 0], [0, 0, 0])
            previous_ball_relative = ball_xyz - ee_pos
            print('BALL RELEASED: >=2 finger groups in contact; no fixed constraint')
        relative = ball_xyz - ee_pos
        rel_speed = (np.inf if previous_ball_relative is None else
                     float(np.linalg.norm(relative - previous_ball_relative) / time_step))
        previous_ball_relative = relative.copy()
        contact_ok = (grasp_state == "closing" and contact_groups >= 2 and
                      np.linalg.norm(relative) < 0.09 and rel_speed < 0.08)
        if contact_ok:
            if grasp_contact_since is None:
                grasp_contact_since = current_time
            elif current_time - grasp_contact_since >= 0.75:
                grasp_success = True
                print(f"GRASP CANDIDATE: >=2 finger groups, ball-to-EE={np.linalg.norm(relative):.3f}m, "
                      f"stable for {current_time-grasp_contact_since:.2f}s")
                break
        else:
            grasp_contact_since = None
        if step % 250 == 0:
            print(f'CONTACT ball={np.round(ball_xyz,3)} '
                  f'height_delta={ball_xyz[2]-ball_position[2]:+.4f}m '
                  f'finger_contacts={len(finger_contacts)} groups={contact_groups} '
                  f'ball_to_EE={np.linalg.norm(relative):.3f}m '
                  f'closing={grasp_state == "closing"} '
                  f'finger_targets={np.round(target_finger_angles,3).tolist()} '
                  f'finger_q={np.round(q_mes[11:17],3).tolist()} '
                  f'ball_dynamic={ball_dynamic}')
        # Full sensor/action traces for the next training iteration. Not certified demos.
        if args.record_full_npz and step % 5 == 0:
            ball_vel = np.asarray(p.getBaseVelocity(ball_id)[0])
            contact_bits = [float(any(cp[3]==j for cp in finger_contacts)) for j in finger_joint_ids]
            full_obs.append(np.concatenate([base_pos,
                [np.cos(base_bearing), np.sin(base_bearing)],base_lin_vel,base_ang_vel,
                q_mes[:17],qd_mes[:17],ee_pos,ball_xyz,ball_vel,target_ee_pos,contact_bits]))
            full_actions.append(np.concatenate([wheel_cmds,torque_arm,target_finger_angles]))
            full_times.append(current_time)

        # ---- Send commands ----
        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        try:
            # Keep the free ball suspended; gravity remains active and contact is real.
            ball_xyz_pre = p.getBasePositionAndOrientation(ball_id)[0]
            if ball_dynamic:
                p.applyExternalForce(ball_id, -1, [0, 0, ball_mass * 9.81],
                                     ball_xyz_pre, p.WORLD_FRAME)
            sim.Step(cmd, "torque")
        except Exception as e:
            print(f"\nSim crashed at t={current_time:.2f}s  step={step}  phase={phase}")
            print(f"  Last base_pos: {np.round(base_pos, 3)}  wheel_cmds: {np.round(wheel_cmds, 3)}")
            print(f"  Error: {e}")
            return {"ok": False, "final_err": float("inf"), "best_err": best_ee_error,
                    "reached": False, "steps": step, "time": current_time,
                    "target": target_ee_pos.tolist(), "phase_switched": phase_switched,
                    "switch_time": switch_time, "base_best_ee": base_best_ee,
                    "base_final_ee": base_final_ee,
                    "base_xy_at_switch": base_xy_at_switch.tolist() if base_xy_at_switch is not None else None,
                    "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
                    "base_bearing_at_best": base_bearing_at_best,
                    "arm_best_ee": arm_best_ee, "arm_final_ee": arm_final_ee}

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
        in_goal = ee_dist_error < REACH_TOLERANCE and current_time > 8.0
        if in_goal and in_goal_since is None:
            in_goal_since = current_time
            if first_reach_time is None:
                first_reach_time = current_time
            print(f"Entered goal ({REACH_TOLERANCE*1000:.0f}mm) at t={current_time:.2f}s, "
                  f"needs to stay {HOLD_TIME:.1f}s...")
        elif not in_goal and in_goal_since is not None:
            times_drifted_out += 1
            print(f"Drifted out of goal at t={current_time:.2f}s after "
                  f"{current_time - in_goal_since:.2f}s  EE err: {ee_dist_error:.4f}m, continuing")
            in_goal_since = None
        # ---- Logging ----
        if step % 500 == 0:
            phase_str = f" [{phase}]" if split_mode else ""
            if phase == "base" and has_base_phases:
                phase_str = f" [base/{base_sub_phase}]"
            print(f"t={current_time:.2f}s{phase_str} | Base: {np.round(base_pos[:2], 3)} | "
                  f"EE: {np.round(ee_pos, 3)} | EE err: {ee_dist_error:.4f}m | "
                  f"best: {best_ee_error:.4f}m")

        if not args.headless:
            time.sleep(0.001)
        current_time += time_step
        step += 1

    # ---- Final report ----
    print('GRASP-ONLY RESULT: ' + ('candidate contact hold observed' if grasp_success else
                                   'no sustained multi-finger grasp observed'))
    print('Candidate contact is simulator evidence only; inspect the video/trace for a real hold.')
    reached = False
    print("=" * 60)
    print(f"Final EE Position  : {np.round(ee_pos, 4)}")
    print(f"Target EE Position : {target_ee_pos}")
    print(f"Final Error (m)    : {ee_dist_error:.4f}")
    print(f"Best Error (m)     : {best_ee_error:.4f}")
    if first_reach_time is not None:
        print(f"First reached at   : t={first_reach_time:.2f}s  "
              f"(drifted out {times_drifted_out}x, held {HOLD_TIME:.1f}s: {'Yes' if held else 'No'})")
    print(f"Steps: {step}  Time: {current_time:.2f}s")
    if split_mode:
        print(f"Phase switched     : {'Yes' if phase_switched else 'No (stayed in base phase)'}")
    print("=" * 60)
    return {
        "ok": True,
        "grasp_candidate": grasp_success,
        "final_err": ee_dist_error,
        "best_err": best_ee_error,
        "reached": reached,
        "first_reach_time": first_reach_time,
        "times_drifted_out": times_drifted_out,
        "steps": step,
        "time": current_time,
        "target": target_ee_pos.tolist(),
        "phase_switched": phase_switched,
        "switch_time": switch_time,
        "base_best_ee": base_best_ee,
        "base_final_ee": base_final_ee,
        "base_xy_at_switch": base_xy_at_switch.tolist() if base_xy_at_switch is not None else None,
        "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
        "base_bearing_at_best": base_bearing_at_best,
        "arm_best_ee": arm_best_ee,
        "arm_final_ee": arm_final_ee,
    }


def main():
    parser = argparse.ArgumentParser(description="Attempt a grasp-only air-ball test with learned policies")
    parser.add_argument("--grasp_checkpoint", type=str, default="grasp_reverse_sensor_bc_v2.npz", help="Learned finger-closure checkpoint (reverse output ignored)")
    # Single-model mode
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained .pt checkpoint (single-model mode)")
    # Split-model mode
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Path to base policy .pt checkpoint (split mode)")
    parser.add_argument("--arm_checkpoint", type=str, default=None,
                        help="Path to arm policy .pt checkpoint (split mode)")
    # Common args
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 1.1, 0.5],
                        metavar=("X", "Y", "Z"),
                        help="Target end-effector position (default: 0.0 1.1 0.5)")
    parser.add_argument("--max_steps", type=int, default=400000,
                        help="Maximum simulation steps")
    parser.add_argument("--gravity_comp", type=str, default=None,
                        help="Path to gravity compensation network checkpoint (adds predicted gravity torques to arm output)")
    parser.add_argument("--arm_damping", type=float, default=0.0,
                        help="Velocity damping coefficient for arm joints (default: 0, e.g. 5.0)")
    parser.add_argument("--headless", action="store_true",
                        help="Skip visualizer updates and sleep for fast evaluation")
    parser.add_argument("--mps", action="store_true",
                        help="Use Apple Metal (MPS) GPU for inference")
    parser.add_argument("--retries", type=int, default=0,
                        help="Number of times to restart simulation on crash (default: 0)")
    parser.add_argument("--heading_threshold", type=float, default=0.15,
                        help="Heading error (rad) for rotate/drive sub-phase transition "
                             "(only used when base checkpoint has 'phase' feature, default: 0.15)")
    parser.add_argument('--record_full_npz', default=None,
                        help='Optional new 63-sensor/17-action physical trace, NOT certified success')
    args = parser.parse_args()
    if args.retries:
        parser.error('Retries disabled for this diagnostic; use a distinct trace file per run')
    if args.checkpoint or not args.base_checkpoint or not args.arm_checkpoint:
        parser.error('This diagnostic requires both existing base and arm checkpoints')

    # Determine mode
    split_mode = args.base_checkpoint is not None and args.arm_checkpoint is not None
    single_mode = args.checkpoint is not None
    if not split_mode and not single_mode:
        args.checkpoint = "bc_policy_100.pt"
        single_mode = True
    if split_mode and single_mode:
        parser.error("Specify either --checkpoint OR (--base_checkpoint + --arm_checkpoint), not both.")

    # --- Load policy/policies (once, reused across retries) ---
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    policy_kwargs = dict(device=device)

    if single_mode:
        model, mean, std, label_mean, label_std, _ = load_policy(args.checkpoint, device=device)
        print(f"Loaded policy from {args.checkpoint}  device={device}")
        policy_kwargs.update(model=model, mean=mean, std=std,
                             label_mean=label_mean, label_std=label_std)
    else:
        base_model, base_mean, base_std, base_lm, base_ls, base_fnames = \
            load_policy(args.base_checkpoint, device=device)
        arm_model, arm_mean, arm_std, arm_lm, arm_ls, arm_fnames = \
            load_policy(args.arm_checkpoint, device=device)
        # Detect phase-as-input model (has 'phase' in feature names)
        has_base_phases = base_fnames is not None and "phase" in list(base_fnames)
        base_desc = args.base_checkpoint
        if has_base_phases:
            phase_desc = f"  phases=input (thr={args.heading_threshold:.3f})"
        else:
            phase_desc = "  phases=no"
        print(f"Split mode: base={base_desc}  arm={args.arm_checkpoint}  "
              f"device={device}"
              f"{phase_desc}")
        policy_kwargs.update(base_model=base_model, base_mean=base_mean, base_std=base_std,
                             base_label_mean=base_lm, base_label_std=base_ls,
                             base_feature_names=base_fnames,
                             has_base_phases=has_base_phases,
                             arm_model=arm_model, arm_mean=arm_mean, arm_std=arm_std,
                             arm_label_mean=arm_lm, arm_label_std=arm_ls,
                             arm_feature_names=arm_fnames)

    # Load gravity compensation network if provided
    if args.gravity_comp:
        grav_model, grav_q_mean, grav_q_std, grav_tau_mean, grav_tau_std = \
            load_gravity_comp(args.gravity_comp, device=device)
        print(f"Gravity comp: {args.gravity_comp}")
        policy_kwargs.update(grav_model=grav_model, grav_q_mean=grav_q_mean,
                             grav_q_std=grav_q_std, grav_tau_mean=grav_tau_mean,
                             grav_tau_std=grav_tau_std)

    target_ee_pos = np.array(args.target)

    # Compute base_threshold from arm geometry (Pythagorean: horizontal reach given vertical offset)
    if split_mode:
        dz = abs(target_ee_pos[2] - SHOULDER_HEIGHT)
        if dz < ARM_REACH_LIMIT:
            base_threshold = 0.75 * np.sqrt(ARM_REACH_LIMIT**2 - dz**2)
        else:
            base_threshold = 0.3  # target z beyond arm reach, get as close as possible
        args.base_threshold = base_threshold
        print(f"Base threshold: {base_threshold:.2f}m "
              f"(arm={ARM_REACH_LIMIT}m, dz={dz:.2f}m)")

    # --- Run with retries ---
    max_attempts = 1 + args.retries
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"\n>>> Restarting simulation (attempt {attempt}/{max_attempts}) <<<\n")
        result = run_sim(args, target_ee_pos, split_mode=split_mode, single_mode=single_mode,
                         **policy_kwargs)
        if result["ok"]:
            break
    else:
        print(f"\nAll {max_attempts} attempts crashed.")


if __name__ == "__main__":
    main()
