#!/usr/bin/env python3
"""V5 EXPERT 10cm TRANSFER built on the working V4.1 grasp simulation.

DEFAULT: --carry_controller expert. The expert directly commands BASE WHEELS
for the final 10cm transfer. NO NEW MACHINE-LEARNING TRAINING IS PERFORMED.
The approach/arm/gripper remain your original pretrained models, so this is
NOT an end-to-end all-expert demonstration. The ball remains dynamic and is
never welded, constrained, teleported, or supported by anti-gravity forces.

Original version: V5 short-carry test built on the verified V4.1 ML-grasp experiment.

Base & arm: original trained checkpoints; handover and arm holding: hybrid.
Finger targets: SensorPolicy output computed from six learned neural-network
increments plus measured joint positions, with no scripted sweep,
clamping to a favourite grasp pose, post-contact squeeze, or teacher override.

For experimental alignment the target ball is static until >=2 distinct finger
groups make persistent contact. Only then is the ball released as a 50 g
freely dynamic body under normal simulated gravity. No antigravity force,
attachment or constraints. A separate freefall probe verifies gravity.

This is a trained ML gripper *evaluation*, NOT an end-to-end learned controller
and NOT a promise the current checkpoint will transport a ball reliably.
After a 2-second hold, a separate carry controller moves the robot 10 cm
forward in its initial heading. --carry_controller existing_base_ml reuses the
trained base policy; --carry_controller expert uses a scripted transfer
teacher ONLY. In both cases finger targets remain learned. No lift or release.
Checkpoint weights are never modified.
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

# Parse --headless early so the env var is set before module-level reads in robotnik_arm_task
if "--headless" in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import numpy as np
import torch
import pinocchio as pin
from simulation_and_control import pb, MotorCommands
from robotnik_arm_task import (
    setup_simulation, setup_kinematics,
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
    print("EXPERT_10CM_TRANSFER: programmed expert controls ONLY the 10cm base motion.")
    print("APPROACH/GRASP: original pretrained base, arm, and V4.1 finger networks; no retraining.")
    print("EXPERT_TRANSFER_STARTED only after a verified 2-second physical gravity hold.")

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, np.array(FIXED_FINGER_ANGLES, dtype=float, copy=True))
    # V4.1: DO NOT add_target_visual here. V4 created a second target-marker
    # sphere at the ball's initial position. After the actual ball moved,
    # that marker remained in the air and looked like a duplicate red ball.
    # We render only the single physical ball below.
    print("V4.1 VISUAL: one physical ball; duplicate target marker disabled")

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
    # Physical friction is a simulator material property, NOT a finger command.
    # Use modest rubber-like friction on BOTH sides of the contact pair.
    p.changeDynamics(ball_id, -1, lateralFriction=args.ball_friction,
                     spinningFriction=0.01, rollingFriction=0.001,
                     restitution=0.0)
    # Better contact solver convergence, particularly for small fingertip meshes.
    if args.solver_iterations:
        p.setPhysicsEngineParameter(numSolverIterations=args.solver_iterations)
    print(f"V4.1 PHYSICS: ball_mu={args.ball_friction:.2f}, "
          f"finger_mu={args.finger_friction:.2f}, "
          f"solver_iterations={args.solver_iterations}")
    print('V4 GRAVITY-HOLD: static ball during approach/closure; dynamic 50g ball after confirmed contacts')
    print(f"Red ball created at {ball_position} - no rod, no attachment")
    # Invisible free-fall gravity probe (baseVisualShapeIndex=-1). It is NOT
    # an extra visible ball. It is removed after the gravity check.
    # Free-fall gravity probe is deliberately far from the robot and target.
    # If gravity is disabled, a stationary ball could falsely appear held.
    # Start high enough to avoid early ground contact even with low target z.
    gravity_probe_initial_z = max(float(target_ee_pos[2]) + 1.0, 2.0)
    gravity_probe_id = p.createMultiBody(
        baseMass=ball_mass, baseCollisionShapeIndex=ball_collision,
        baseVisualShapeIndex=-1,
        basePosition=[float(target_ee_pos[0] + 5.0),
                      float(target_ee_pos[1] + 5.0), gravity_probe_initial_z])
    gravity_probe_passed = None
    gravity_probe_done = False

    # GRASP-AND-REVERSE DEMO STATE
    # Physical finger contact is evaluated; successful grasp is not assumed.
    from robotnik_arm_control import FINGER_JOINT_NAMES
    finger_joint_ids = [
        jid for jid in range(client.getNumJoints(rid))
        if client.getJointInfo(rid, jid)[1].decode() in FINGER_JOINT_NAMES
    ]
    if len(finger_joint_ids) != 6:
        raise RuntimeError(f"Expected 6 finger joints, found {len(finger_joint_ids)}: {finger_joint_ids}")
    hand_link = next(j for j in range(p.getNumJoints(rid))
                     if p.getJointInfo(rid,j)[12].decode() == 'robot_j2s7s300_link_7')
    # Override only the finger servo settings after the existing callback runs.
    previous_motor_callback = sim._SetMotorTorqueByIds
    def bounded_finger_callback(motor_ids, commands, index=0):
        previous_motor_callback(motor_ids, commands, index=index)
        for i,jid in enumerate(finger_joint_ids):
            lim = p.getJointInfo(rid,jid)
            p.setJointMotorControl2(rid,jid,p.POSITION_CONTROL,
                targetPosition=float(np.clip(target_finger_angles[i],lim[8],lim[9])),
                force=min(2.0,float(lim[10])), maxVelocity=0.5)
    sim._SetMotorTorqueByIds = bounded_finger_callback
    grasp_state = "approach"  # approach -> closing -> gravity_hold -> carry -> carry_settle
    print("\n=== V4 FINGER PHYSICS / JOINT MAPPING ===")
    for i, jid in enumerate(finger_joint_ids):
        ji = p.getJointInfo(rid, jid)
        collision_shapes = p.getCollisionShapeData(rid, jid)
        # Only alter friction for existing physical fingertip/finger surfaces;
        # motor forces, targets, ML outputs, and link collision meshes unchanged.
        p.changeDynamics(rid, jid, lateralFriction=args.finger_friction,
                         spinningFriction=0.01, rollingFriction=0.001)
        print(f"Finger {i + 1}: joint_id={jid}, joint_name={ji[1].decode()}, "
              f"link={ji[12].decode()}, limits=({ji[8]:.3f}, {ji[9]:.3f}), "
              f"initial_target={target_finger_angles[i]:.3f}, "
              f"max_force={ji[10]:.3f}, collision_shapes={len(collision_shapes)}")
    print("V4.1 GRASP MODE: TRAINED RESIDUAL-BC FINGERS (reverse disabled)")
    print("V4.1: trained network controls all six fingers; no scripted squeeze, weld, or anti-gravity.")

    def pocket_world():
        state = p.getLinkState(rid, hand_link, computeForwardKinematics=True)
        rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        # Original calibrated pocket with optional local displacement for later tuning.
        offset = np.array([0.0, 0.015, -0.190]) + np.asarray(args.pocket_offset_local)
        return np.asarray(state[4]) + rotation @ offset

    # Load learned action policy ONCE, without retraining base/arm.
    from sensor_bc_v4 import sensor_features, SensorPolicy
    grasp_bc_policy = SensorPolicy(args.grasp_checkpoint)
    checkpoint_hasher = hashlib.sha256()
    with open(args.grasp_checkpoint, "rb") as checkpoint_file:
        for checkpoint_chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            checkpoint_hasher.update(checkpoint_chunk)
    checkpoint_sha256 = checkpoint_hasher.hexdigest()
    print(f"SENSOR_ML_CHECKPOINT: {args.grasp_checkpoint}")
    print(f"SENSOR_ML_CHECKPOINT_SHA256: {checkpoint_sha256}")
    print("SENSOR_ML_INFERENCE: V4 network predicts six finger increments; sensed_q + learned_delta forms target; no override.")

    v10_ball_clearance_enabled = False
    grasp_start = None
    grasp_joint_hold = None
    carry_start_xy = None
    print("V5 CARRY: a 10 cm forward transfer after verified V4.1 dynamic grasp")
    print(f"V5 CARRY CONTROLLER: {args.carry_controller} (ML option uses original trained base checkpoint)")
    print("No virtual object attachment, no scripted finger sweep, ball remains dynamic under gravity")


    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    time_step = sim.GetTimeStep()
    current_time = 0.0
    cmd = MotorCommands()
    import atexit
    from pathlib import Path
    full_obs, full_actions, full_times = [], [], []
    ml_obs_records, ml_action_records, ml_time_records = [], [], []
    if args.record_full_npz and Path(args.record_full_npz).exists():
        raise FileExistsError(args.record_full_npz)
    if args.record_ml_npz and Path(args.record_ml_npz).exists():
        raise FileExistsError(args.record_ml_npz)
    if args.record_expert_npz and Path(args.record_expert_npz).exists():
        raise FileExistsError(args.record_expert_npz)
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
                policy_scope=np.asarray('hybrid_approach_ml_grasp_and_carry_v5_' + args.carry_controller))
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
    min_finger_gaps = {jid: float("inf") for jid in finger_joint_ids}
    nearest_finger_vectors = {}       # cp[5] finger world point, cp[6] ball world point
    contacted_finger_links = set()
    max_simultaneous_groups = 0
    contact_confirm_since = None
    released_at = None
    ball_release_xyz = None
    ball_pocket_offset_release = None
    max_relative_drift = 0.0
    max_ball_drop = 0.0
    # Diagnostic-only state. Never fed back as a scripted finger action.
    first_lost_contact_at = None
    first_ball_drift_at = None
    last_hold_groups = []
    peak_contact_normal_force = 0.0
    peak_ball_speed_after_release = 0.0
    hold_group_good_samples = 0
    hold_total_samples = 0
    hold_failure_reason = None
    # V5: active short transfer, separate from physical grasp certification.
    carry_started_at = None
    carry_goal_xy = None
    carry_heading = None
    carry_origin_xy = None
    carry_ball_origin = None
    carry_ball_to_pocket_start = None
    carry_last_groups = []
    carry_good_samples = 0
    carry_total_samples = 0
    carry_contact_loss_since = None
    carry_max_relative_drift = 0.0
    carry_max_ball_drop = 0.0
    carry_success = False
    carry_fail_reason = None
    carry_end_time = None
    carry_stop_wheel_q = None
    carry_obs = []  # for EXPERT only: not mislabeled as an all-expert demonstration
    carry_actions = []
    carry_times = []

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
            pocket = pocket_world()

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


        # Scripted handover from arm approach into the selected finger test mode.
        # Start closing only near the best observed pocket alignment; an early
        # close made the arm drift away before the finger pads reached the ball.
        if phase == "arm" and grasp_state == "approach" and pocket_gap < 0.010:
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

            p.performCollisionDetection()
            if any(cp[8] < -0.001 for cp in p.getContactPoints(rid, ball_id)):
                print('ALIGNMENT FAILED: ball penetrates the hand; closure not started')
                break
            # Keep the ball STATIC only until fingers actually make contact.
            # Do not apply upward forces; after release it must support its own weight.
            print('BALL STATIC UNTIL CONTACT: waiting for >=2 finger groups (no attachment)')
            grasp_state = "closing"
            grasp_start = current_time
            grasp_joint_hold = q_mes[4:11].copy()
            print(f">>> CLOSE FINGERS at t={current_time:.2f}s, "
                  f"ball-to-EE={ball_ee_distance:.3f} m")


            # This proximity trigger remains scripted in both V4 test modes.
            carry_start_xy = base_pos[:2].copy()
            carry_heading = np.array([np.cos(base_bearing), np.sin(base_bearing)])
            print('SENSOR POLICY ACTIVE: model predictions ONLY, no scripted assist')

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
            hold_ee_target = ee_pos if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "reverse") else approach_ee_target
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
        if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "reverse"):
            # Keep the arm at the grasp pose, do not add a lift command in V4.
            torque_arm = np.clip(
                torque_arm + 18.0 * (grasp_joint_hold - q_mes[4:11]) - 2.5 * qd_mes[4:11],
                -20.0, 20.0)
        # Finger targets come ONLY from the trained sensor policy. No learned
        # wheel-reverse is tested here: the mobile base is held stationary.
        if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle"):
            ball_xyz = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
            obs_ml = np.asarray(sensor_features(
                ee_pos, ball_xyz, q_mes[11:17], qd_mes[11:17],
                base_lin_vel[:2], base_bearing), dtype=np.float64)
            action_ml = np.asarray(grasp_bc_policy.predict(obs_ml), dtype=np.float64).reshape(-1)
            if action_ml.size < 6 or not np.all(np.isfinite(action_ml[:6])):
                raise RuntimeError(f"Invalid learned finger policy outputs: {action_ml}")
            target_finger_angles[:] = action_ml[:6]  # V4 SensorPolicy neural-network action, bounded to demonstrated support
            # Record ML predictions for audit; do NOT mistake these for expert demos.
            ml_obs_records.append(obs_ml.copy())
            ml_action_records.append(action_ml.copy())
            ml_time_records.append(current_time)
            if step % 250 == 0:
                print(f"ML_INFERENCE t={current_time:.2f}s "
                      f"finger_target_rad={np.round(action_ml[:6], 3).tolist()} "
                      f"finger_measured_rad={np.round(q_mes[11:17], 3).tolist()}")
            # V4.1 stationary-base hold is retained until transport is authorised.
            if grasp_state in ("closing", "gravity_hold"):
                wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - wheel_lock_pos), -5.0, 5.0)

        # -------- V5 transfer controller: explicitly labelled ML or EXPERT --------
        # During carry we KEEP the learned V4 finger predictions active above.
        if grasp_state == "carry":
            displacement = base_pos[:2] - carry_origin_xy
            progress = float(np.dot(displacement, carry_heading))
            remaining = float(args.carry_distance - progress)
            lateral = float(carry_heading[0]*displacement[1] - carry_heading[1]*displacement[0])
            bearing_error = float(np.arctan2(
                np.sin(np.arctan2(carry_heading[1], carry_heading[0]) - base_bearing),
                np.cos(np.arctan2(carry_heading[1], carry_heading[0]) - base_bearing)))
            if args.carry_controller == "existing_base_ml":
                # Reuse the ALREADY TRAINED base policy, not a new expert policy.
                # Its goal is 10 cm forward of the base pose at carry start.
                filtered = ([n for n in base_feature_names if n != "phase"]
                            if has_base_phases else base_feature_names)
                carry_xyz_goal = np.array([carry_goal_xy[0], carry_goal_xy[1], target_ee_pos[2]])
                base_obs = build_base_observation(
                    base_pos, base_bearing, base_lin_vel, base_ang_vel,
                    carry_xyz_goal, feature_names=filtered)
                if has_base_phases:
                    # Same original route-phase convention: drive (not rotate).
                    base_obs = np.append(base_obs, 1.0)
                raw_vw = np.asarray(predict(
                    base_model, base_mean, base_std, base_obs,
                    device=device, label_mean=base_label_mean,
                    label_std=base_label_std), dtype=float).reshape(-1)
                if len(raw_vw) < 2 or not np.isfinite(raw_vw[:2]).all():
                    carry_fail_reason = f"Invalid trained base policy output: {raw_vw}"
                    print(f"V5_CARRY_FAIL {carry_fail_reason}")
                    break
                # Safety speed caps ONLY, not expert velocity substitution.
                v_cmd = float(np.clip(max(raw_vw[0], 0.0), 0.0, args.carry_speed_max))
                w_cmd = float(np.clip(raw_vw[1] if not has_base_phases else 0.0,
                                      -args.carry_yaw_max, args.carry_yaw_max))
            elif args.carry_controller == "expert":
                # Distinct scripted teacher for DATA COLLECTION, never called in ML mode.
                # Proportional speed and mild lateral/heading correction.
                v_cmd = float(np.clip(args.expert_kp * max(remaining, 0.0),
                                      0.0, args.carry_speed_max))
                w_cmd = float(np.clip(1.5 * bearing_error - 1.2 * lateral,
                                      -args.carry_yaw_max, args.carry_yaw_max))
                # Observations/actions are valid teacher labels for the base transfer
                # stage ONLY. The grasp was performed by V4.1 ML.
                if step % 5 == 0:
                    fwd_vel = float(np.dot(base_lin_vel[:2], carry_heading))
                    carry_obs.append([remaining, lateral, bearing_error,
                                      fwd_vel, float(base_ang_vel[2])])
                    carry_actions.append([v_cmd, w_cmd])
                    carry_times.append(current_time - carry_started_at)
            else:
                raise ValueError(f"Invalid carry mode {args.carry_controller}")
            rw = (v_cmd + w_cmd * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
            lw = (v_cmd - w_cmd * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
            wheel_cmds = np.array([rw, lw, lw, rw], dtype=float)
            if step % 250 == 0:
                print(f"V5_CARRY t={current_time:.2f}s mode={args.carry_controller} "
                      f"progress_m={progress:.4f} remaining_m={remaining:.4f} "
                      f"lateral_m={lateral:.4f} v_cmd={v_cmd:.4f} w_cmd={w_cmd:.4f}")

        if grasp_state == "carry_settle":
            wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - carry_stop_wheel_q), -5.0, 5.0)

        # Log observable physics, never claim success from robot motion alone.
        ball_xyz = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
        robot_contacts = p.getContactPoints(bodyA=rid, bodyB=ball_id)
        finger_contacts = [cp for cp in robot_contacts if cp[3] in finger_joint_ids]
        contact_groups = sum(any(cp[3] in finger_joint_ids[i:i+2]
                             for cp in finger_contacts) for i in (0, 2, 4))
        finger_normal_force = sum(max(0.0, float(cp[9])) for cp in finger_contacts)
        peak_contact_normal_force = max(peak_contact_normal_force, finger_normal_force)
        if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle"):
            current_groups = [
                group for group in range(3)
                if any(cp[3] in finger_joint_ids[2*group:2*group+2]
                       for cp in finger_contacts)
            ]
            max_simultaneous_groups = max(max_simultaneous_groups, len(current_groups))
            contacted_finger_links.update(int(cp[3]) for cp in finger_contacts)

            # Signed collision distances: + separation, 0 touching, - penetration.
            if step % 20 == 0:
                for jid in finger_joint_ids:
                    pts = p.getClosestPoints(bodyA=rid, bodyB=ball_id,
                        distance=args.diagnostic_search_radius,
                        linkIndexA=jid, linkIndexB=-1)
                    if pts:
                        nearest = min(pts, key=lambda cp: cp[8])
                        gap = float(nearest[8])
                        if gap < min_finger_gaps[jid]:
                            min_finger_gaps[jid] = gap
                            nearest_finger_vectors[jid] = (
                                np.asarray(nearest[5]).copy(),
                                np.asarray(nearest[6]).copy())

            if step % 250 == 0:
                gaps_mm = [
                    f"{min_finger_gaps[j]*1000:+.1f}" if np.isfinite(min_finger_gaps[j])
                    else "n/a" for j in finger_joint_ids
                ]
                print(f"V4_1_CONTACT t={current_time:.2f}s state={grasp_state} "
                      f"groups={current_groups} links={sorted(set(int(cp[3]) for cp in finger_contacts))} "
                      f"normal_force_N={finger_normal_force:.3f} "
                      f"min_gaps_mm={gaps_mm} ball={np.round(ball_xyz, 3).tolist()}")

            if grasp_state == "closing":
                # Static-ball contact qualifies only when TWO DISTINCT fingers
                # (groups, not merely two points on one fingertip) touch stably.
                if len(current_groups) >= 2 and np.linalg.norm(ball_xyz-ee_pos) < 0.09:
                    if contact_confirm_since is None:
                        contact_confirm_since = current_time
                    elif current_time-contact_confirm_since >= args.contact_confirm_seconds:
                        if not gravity_probe_passed:
                            hold_failure_reason = "Gravity probe failed/not completed; cannot validate hold."
                            print(f"V4_ABORT: {hold_failure_reason}")
                            break
                        p.changeDynamics(ball_id, -1, mass=ball_mass)
                        p.resetBaseVelocity(ball_id, linearVelocity=[0,0,0],
                                            angularVelocity=[0,0,0])
                        ball_dynamic = True
                        grasp_state = "gravity_hold"
                        released_at = current_time
                        ball_release_xyz = ball_xyz.copy()
                        ball_pocket_offset_release = ball_xyz - pocket_world()
                        print(f"V4_1_RELEASE_TO_GRAVITY t={current_time:.2f}s "
                              f"contact_groups={current_groups} "
                              f"normal_force_N={finger_normal_force:.3f} "
                              f"mass={p.getDynamicsInfo(ball_id,-1)[0]:.3f}kg "
                              f"no_attachment=True no_antigravity_force=True")
                else:
                    contact_confirm_since = None

                if current_time-grasp_start >= args.max_close_seconds and released_at is None:
                    hold_failure_reason = "No stable two-group finger contact before close timeout"
                    print(f"V4_NO_RELEASE: {hold_failure_reason}")
                    break

            if grasp_state == "gravity_hold" and released_at is not None:
                hold_total_samples += 1
                if len(current_groups) >= 2:
                    hold_group_good_samples += 1
                elif first_lost_contact_at is None and current_time > released_at + time_step:
                    first_lost_contact_at = current_time
                    print(f"V4_1_FIRST_CONTACT_LOSS t={current_time:.3f}s "
                          f"groups={current_groups} normal_force_N={finger_normal_force:.3f}")
                last_hold_groups = list(current_groups)
                pocket_relative_drift = float(np.linalg.norm(
                    (ball_xyz-pocket_world()) - ball_pocket_offset_release))
                max_relative_drift = max(max_relative_drift, pocket_relative_drift)
                vertical_drop = float(ball_release_xyz[2]-ball_xyz[2])
                max_ball_drop = max(max_ball_drop, vertical_drop)
                ball_speed = float(np.linalg.norm(p.getBaseVelocity(ball_id)[0]))
                peak_ball_speed_after_release = max(peak_ball_speed_after_release, ball_speed)
                if first_ball_drift_at is None and pocket_relative_drift > 0.005:
                    first_ball_drift_at = current_time
                    print(f"V4_1_FIRST_5MM_SLIP t={current_time:.3f}s "
                          f"drift_mm={pocket_relative_drift*1000:.1f} "
                          f"ball_speed_mps={ball_speed:.3f}")
                # High-resolution, short post-release sequence shows exactly
                # when/why the ball slipped. This is passive logging only.
                if (step % 50 == 0 or
                        (current_time - released_at < 0.5 and step % 10 == 0)):
                    print(f"V4_1_GRAVITY_HOLD elapsed={current_time-released_at:.3f}s "
                          f"groups={current_groups} normal_force_N={finger_normal_force:.3f} "
                          f"ball_speed_mps={ball_speed:.3f} "
                          f"relative_drift_mm={pocket_relative_drift*1000:.1f} "
                          f"drop_mm={vertical_drop*1000:.1f} "
                          f"ball_xyz={np.round(ball_xyz, 4).tolist()}")
                if (max_relative_drift > args.max_relative_drift or
                        max_ball_drop > args.max_vertical_drop):
                    hold_failure_reason = (
                        f"Ball moved out of hold: drift={max_relative_drift*1000:.1f}mm, "
                        f"drop={max_ball_drop*1000:.1f}mm")
                    print(f"V4_HOLD_FAIL: {hold_failure_reason}")
                    break
                if current_time-released_at >= args.gravity_hold_seconds:
                    contact_fraction = hold_group_good_samples / max(1, hold_total_samples)
                    grasp_success = (
                        contact_fraction >= args.min_contact_fraction
                        and len(current_groups) >= 2
                        and max_relative_drift <= args.max_relative_drift
                        and max_ball_drop <= args.max_vertical_drop
                    )
                    if grasp_success:
                        print(f"V4_GRAVITY_HOLD_PASS mode=learned "
                              f"duration={current_time-released_at:.2f}s "
                              f"contact_fraction={contact_fraction:.3f} "
                              f"max_drift_mm={max_relative_drift*1000:.1f} "
                              f"max_drop_mm={max_ball_drop*1000:.1f}")
                        # At this moment the ball is DYNAMIC under gravity. Begin
                        # moving the base forward exactly the requested distance.
                        carry_started_at = current_time
                        carry_origin_xy = base_pos[:2].copy()
                        carry_heading = np.array([np.cos(base_bearing), np.sin(base_bearing)])
                        carry_goal_xy = carry_origin_xy + args.carry_distance * carry_heading
                        carry_ball_origin = ball_xyz.copy()
                        carry_ball_to_pocket_start = ball_xyz - pocket_world()
                        grasp_state = "carry"
                        print(f"V5_START_CARRY t={current_time:.2f}s "
                              f"mode={args.carry_controller} distance={args.carry_distance:.3f}m "
                              f"from_xy={np.round(carry_origin_xy,4).tolist()} "
                              f"to_xy={np.round(carry_goal_xy,4).tolist()} "
                              f"ball_xyz={np.round(carry_ball_origin,4).tolist()} "
                              f"no_attachment=True ball_dynamic={ball_dynamic}")
                    else:
                        hold_failure_reason = (
                            f"Insufficient sustained contacts: "
                            f"two-group fraction={contact_fraction:.3f}, "
                            f"final_groups={current_groups}")
                        print(f"V4_HOLD_FAIL: {hold_failure_reason}")
                        break
        if grasp_state in ("carry", "carry_settle"):
            # Physical validation uses ball motion relative to the gripper and
            # two independent finger groups. No constraints, forces or object reset.
            carry_total_samples += 1
            carry_last_groups = list(current_groups)
            if len(current_groups) >= 2:
                carry_good_samples += 1
                carry_contact_loss_since = None
            elif carry_contact_loss_since is None:
                carry_contact_loss_since = current_time
            carry_drift = float(np.linalg.norm(
                (ball_xyz - pocket_world()) - carry_ball_to_pocket_start))
            carry_drop = float(carry_ball_origin[2] - ball_xyz[2])
            carry_max_relative_drift = max(carry_max_relative_drift, carry_drift)
            carry_max_ball_drop = max(carry_max_ball_drop, carry_drop)
            base_displacement = base_pos[:2] - carry_origin_xy
            carry_progress = float(np.dot(base_displacement, carry_heading))
            target_remaining = args.carry_distance - carry_progress
            lat_error = float(carry_heading[0]*base_displacement[1] - carry_heading[1]*base_displacement[0])
            ball_displacement = ball_xyz[:2] - carry_ball_origin[:2]
            ball_progress = float(np.dot(ball_displacement, carry_heading))
            ball_lateral = float(carry_heading[0]*ball_displacement[1] - carry_heading[1]*ball_displacement[0])
            if step % 250 == 0:
                print(f"V5_PHYSICAL_CARRY t={current_time:.2f}s state={grasp_state} "
                      f"base_forward_m={carry_progress:.4f} ball_forward_m={ball_progress:.4f} "
                      f"base_lateral_m={lat_error:.4f} ball_lateral_m={ball_lateral:.4f} "
                      f"groups={current_groups} finger_force_N={finger_normal_force:.2f} "
                      f"ball_pocket_drift_mm={carry_drift*1000:.1f} drop_mm={carry_drop*1000:.1f}")
            if carry_max_relative_drift > args.carry_max_slip or carry_max_ball_drop > args.carry_max_drop:
                carry_fail_reason = (f"Ball slipped during transfer: pocket_drift={carry_max_relative_drift*1000:.1f}mm "
                                     f"drop={carry_max_ball_drop*1000:.1f}mm")
            elif carry_contact_loss_since is not None and current_time - carry_contact_loss_since > 0.15:
                carry_fail_reason = "Physical contact dropped below two finger groups for >0.15 s"
            elif current_time - carry_started_at > args.carry_timeout + args.carry_settle_seconds:
                carry_fail_reason = f"Carry timeout; progress {carry_progress:.3f}/{args.carry_distance:.3f}m"
            elif (grasp_state == "carry" and target_remaining <= args.carry_goal_tolerance
                  and abs(lat_error) < args.carry_lateral_tolerance):
                grasp_state = "carry_settle"
                carry_end_time = current_time
                carry_stop_wheel_q = q_mes[0:4].copy()
                print(f"V5_CARRY_REACHED at={current_time:.2f}s; checking stationary hold "
                      f"for {args.carry_settle_seconds:.1f}s")
            elif grasp_state == "carry_settle" and current_time - carry_end_time >= args.carry_settle_seconds:
                contact_fraction = carry_good_samples / max(1, carry_total_samples)
                carry_success = bool(
                    abs(args.carry_distance-carry_progress) < args.carry_goal_tolerance
                    and abs(lat_error) < args.carry_lateral_tolerance
                    and abs(ball_progress-carry_progress) < args.carry_max_slip
                    and abs(ball_lateral-lat_error) < args.carry_max_slip
                    and contact_fraction >= args.carry_min_contact_fraction
                    and len(current_groups) >= 2
                    and carry_max_relative_drift <= args.carry_max_slip
                    and carry_max_ball_drop <= args.carry_max_drop)
                if not carry_success:
                    carry_fail_reason = (f"Final physical validation failed: base={carry_progress:.3f}m "
                                         f"ball={ball_progress:.3f}m contact_fraction={contact_fraction:.3f}")
            if carry_fail_reason or carry_success:
                print(f"V5_CARRY_RESULT {'PASS' if carry_success else 'FAIL'} "
                      f"controller={args.carry_controller} "
                      f"base_m={carry_progress:.4f} ball_m={ball_progress:.4f} "
                      f"contacts={current_groups} "
                      f"slip_mm={carry_max_relative_drift*1000:.1f} "
                      f"failure={carry_fail_reason}")
                break

        if step % 250 == 0:
            print(f'CONTACT ball={np.round(ball_xyz,3)} '
                  f'height_delta={ball_xyz[2]-ball_position[2]:+.4f}m '
                  f'finger_contacts={len(finger_contacts)} groups={contact_groups} '
                  f'ball_to_EE={np.linalg.norm(ball_xyz-ee_pos):.3f}m '
                  f'state={grasp_state} '
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
            # CRITICAL: NO upward antigravity force after release.
            # No setGravity override and no robot/ball body constraint.
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

        # Verify simulator gravity with a separate free-fall test sphere.
        if not gravity_probe_done and current_time >= 0.35:
            probe_z = p.getBasePositionAndOrientation(gravity_probe_id)[0][2]
            probe_drop = gravity_probe_initial_z - float(probe_z)
            gravity_probe_passed = probe_drop >= 0.15
            print(f"V4_GRAVITY_PROBE drop_m={probe_drop:.3f} "
                  f"gravity_active={'YES' if gravity_probe_passed else 'NO'}")
            p.removeBody(gravity_probe_id)
            gravity_probe_done = True

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
    print("\n========== V4.1 DYNAMIC GRASP VALIDATION ==========")
    print("Mode: V4 LEARNED RESIDUAL-BC FINGER ACTIONS (no scripted sweep)")
    print(f"Free-fall gravity probe: {'PASS' if gravity_probe_passed else 'FAIL/NOT RUN'}")
    print(f"Two-group finger contact observed: {'YES' if max_simultaneous_groups >= 2 else 'NO'}")
    print(f"Ball released to dynamic gravity: {'YES' if released_at is not None else 'NO'}")
    for i,jid in enumerate(finger_joint_ids):
        gap = min_finger_gaps[jid]
        gap_text = f"{gap * 1000:+.2f} mm" if np.isfinite(gap) else "not measured"
        line = f"  finger[{i+1}] link={jid}: min_gap={gap_text}; " \
               f"ever_contact={'YES' if jid in contacted_finger_links else 'NO'}"
        if jid in nearest_finger_vectors:
            pos_finger, pos_ball = nearest_finger_vectors[jid]
            line += f"; ball_minus_finger_xyz={np.round(pos_ball-pos_finger,4).tolist()}m"
        print(line)
    print(f"Max simultaneous finger groups: {max_simultaneous_groups}/3")
    print(f"Distinct finger links ever contacting: {sorted(contacted_finger_links)}")
    if released_at is not None:
        final_ball_xyz = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
        print(f"Last ball xyz (m): {np.round(final_ball_xyz, 4).tolist()}")
        print(f"First loss of >=2 finger groups (s): {first_lost_contact_at}")
        print(f"First 5mm ball/pocket drift (s): {first_ball_drift_at}")
        print(f"Last contact groups: {last_hold_groups}")
        print(f"Peak finger normal contact force (N): {peak_contact_normal_force:.3f}")
        print(f"Peak ball speed after release (m/s): {peak_ball_speed_after_release:.3f}")
        print(f"Two-group contact fraction during gravity hold: "
              f"{hold_group_good_samples/max(1,hold_total_samples):.3f}")
        print(f"Maximum ball/pocket relative drift: {max_relative_drift * 1000:.1f} mm")
        print(f"Maximum downward ball drop: {max_ball_drop * 1000:.1f} mm")
    if hold_failure_reason:
        print(f"Failure reason: {hold_failure_reason}")
    print(f"V4_1_RESULT: {'GRAVITY HOLD PASS' if grasp_success else 'NOT VERIFIED / FAIL'}")
    print("========== V5 SHORT CARRY RESULT ==========")
    print(f"Controller: {args.carry_controller} "
          f"({'scripted TEACHER' if args.carry_controller == 'expert' else 'EXISTING TRAINED BASE ML'})")
    print(f"Target transfer distance: {args.carry_distance:.3f} m FORWARD in initial base heading")
    print(f"V5_CARRY_RESULT: {'PASS' if carry_success else 'FAIL / NOT VERIFIED'}")
    print(f"Carry began (s): {carry_started_at}")
    print(f"Max ball/pocket relative movement during carry: {carry_max_relative_drift*1000:.1f} mm")
    print(f"Max ball drop during carry: {carry_max_ball_drop*1000:.1f} mm")
    print(f"Carry sample fraction with >=2 contact groups: {carry_good_samples/max(1,carry_total_samples):.3f}")
    print(f"Carry failure reason: {carry_fail_reason}")
    print("This is a HYBRID experiment, not an end-to-end learned pick-and-place policy.")
    if args.record_expert_npz and args.carry_controller == 'expert':
        with Path(args.record_expert_npz).open('xb') as output:
            np.savez_compressed(output,
                feature_names=np.asarray(['remaining_m', 'lateral_m', 'heading_err_rad',
                                         'forward_speed_mps', 'yaw_rate_radps']),
                obs=np.asarray(carry_obs, dtype=np.float32).reshape(-1, 5),
                actions_vw=np.asarray(carry_actions, dtype=np.float32).reshape(-1, 2),
                elapsed_s=np.asarray(carry_times, dtype=np.float32),
                source=np.asarray('SCRIPTED_EXPERT_TRANSFER_ONLY; V4_ML_GRASP'),
                physical_success_verified=np.asarray(bool(carry_success)),
                carries_ball_with_friction=np.asarray(True),
                no_virtual_attachment=np.asarray(True),
                target_distance_m=np.asarray(args.carry_distance))
        print(f"V5_EXPERT_TRACE_SAVED {args.record_expert_npz} frames={len(carry_obs)} "
              f"verified_success={carry_success}")
    print("V4.1 uses the unchanged V4 ML model trained on UNVERIFIED virtual-attachment demonstrations.")
    print("This is an ML gripper evaluation with hybrid handover, NOT end-to-end control.")
    print("No object attachment, no scripted finger sweep and no unverified transport success claimed.")
    print("Inspect visuals before treating any candidate gravity hold as a grasp.")
    print("=================================================\n")
    if args.record_ml_npz:
        with Path(args.record_ml_npz).open('xb') as output:
            np.savez_compressed(
                output,
                observations=np.asarray(ml_obs_records, dtype=np.float32),
                policy_outputs=np.asarray(ml_action_records, dtype=np.float32),
                sim_times=np.asarray(ml_time_records, dtype=np.float32),
                checkpoint_sha256=np.asarray(checkpoint_sha256),
                label_source=np.asarray('ML_policy_predictions_not_expert_labels'),
                physical_gravity_hold_pass=np.asarray(bool(grasp_success)),
                ball_released=np.asarray(released_at is not None),
                no_virtual_attachment=np.asarray(True),
            )
        print(f"ML_PREDICTION_TRACE: saved {len(ml_obs_records)} steps to {args.record_ml_npz}")
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
        "carry_success": carry_success,
        "carry_mode": args.carry_controller,
        "grasp_mode": "learned",
        "gravity_probe_passed": gravity_probe_passed,
        "released_to_gravity": released_at is not None,
        "max_relative_drift": max_relative_drift,
        "max_ball_drop": max_ball_drop,
        "max_simultaneous_finger_groups": max_simultaneous_groups,
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
    parser = argparse.ArgumentParser(description="EXPERT 10cm TRANSFER after existing V4.1 learned grasp; NO new model training")
    parser.add_argument("--carry_controller", choices=["existing_base_ml", "expert"],
                        default="expert", help="Run the scripted expert transfer (no carry-policy ML training)")
    parser.add_argument("--carry_distance", type=float, default=0.10,
                        help="Forward travel distance from base pose at start of carry (metres)")
    parser.add_argument("--carry_speed_max", type=float, default=0.035,
                        help="Maximum speed during carry, a safety cap not an ML command")
    parser.add_argument("--carry_yaw_max", type=float, default=0.15,
                        help="Safety cap on base angular velocity")
    parser.add_argument("--expert_kp", type=float, default=0.6,
                        help="Proportional velocity gain for expert demonstration ONLY")
    parser.add_argument("--carry_timeout", type=float, default=12.0,
                        help="Maximum active carry time before reporting failure")
    parser.add_argument("--carry_settle_seconds", type=float, default=1.0,
                        help="Duration of stationary holding verification at goal")
    parser.add_argument("--carry_goal_tolerance", type=float, default=0.012,
                        help="Allowed forward displacement error in metres")
    parser.add_argument("--carry_lateral_tolerance", type=float, default=0.020,
                        help="Allowed lateral deviation in metres")
    parser.add_argument("--carry_max_slip", type=float, default=0.025,
                        help="Maximum ball displacement relative to gripper during carry in metres")
    parser.add_argument("--carry_max_drop", type=float, default=0.025,
                        help="Maximum downward ball displacement relative to carry start in metres")
    parser.add_argument("--carry_min_contact_fraction", type=float, default=0.8)
    parser.add_argument("--record_expert_npz", type=str, default=None,
                        help="Save labelled expert transfer data (allowed in --carry_controller expert only)")
    parser.add_argument("--grasp_checkpoint", type=str, default="grasp_sensor_residual_bc_v4.npz", help="V4 residual BC finger policy checkpoint (6 learned targets)")
    parser.add_argument("--ball_friction", type=float, default=1.5,
                        help="Physical ball lateral friction coefficient (default 1.5)")
    parser.add_argument("--finger_friction", type=float, default=1.5,
                        help="Physical finger lateral friction coefficient (default 1.5)")
    parser.add_argument("--solver_iterations", type=int, default=80,
                        help="PyBullet contact solver iterations, default 80 (0 keeps current engine setting)")
    parser.add_argument("--diagnostic_search_radius", type=float, default=0.15,
                        help="Closest collision distance search radius in metres")
    parser.add_argument("--contact_confirm_seconds", type=float, default=0.25,
                        help="Minimum continuous two-group contact before dynamic release")
    parser.add_argument("--max_close_seconds", type=float, default=8.0,
                        help="End test if two finger groups have not touched within this time")
    parser.add_argument("--gravity_hold_seconds", type=float, default=2.0,
                        help="Monitor dynamic object under normal gravity for this long")
    parser.add_argument("--max_relative_drift", type=float, default=0.025,
                        help="Maximum ball displacement relative to gripper pocket (m)")
    parser.add_argument("--max_vertical_drop", type=float, default=0.025,
                        help="Maximum downward displacement of ball during hold (m)")
    parser.add_argument("--min_contact_fraction", type=float, default=0.80,
                        help="Fraction of gravity hold with >=2 distinct finger groups touching")
    parser.add_argument("--pocket_offset_local", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"),
                        help="Fine tune original hand-local pocket offset (m), default zeros")
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
                        help='Optional new physical trace, NOT certified success')
    parser.add_argument('--record_ml_npz', default=None,
                        help='Save real SensorPolicy observations/predictions for audit, NOT expert labels')
    args = parser.parse_args()
    if args.record_expert_npz and args.carry_controller != "expert":
        parser.error("--record_expert_npz is only allowed with --carry_controller expert")
    if not 0 < args.carry_distance <= 0.5:
        parser.error("--carry_distance must be >0 and <= 0.5m")
    if not 0 < args.carry_speed_max <= 0.10:
        parser.error("--carry_speed_max must be >0 and <=0.10m/s")
    if not 0 < args.carry_yaw_max <= 1:
        parser.error("--carry_yaw_max must be >0 and <= 1rad/s")
    if args.carry_timeout <= 0 or args.carry_settle_seconds <= 0:
        parser.error("Carry timeout and settle duration must be positive")
    if not 0 <= args.carry_min_contact_fraction <= 1:
        parser.error("Carry contact fraction must be [0,1]")
    if any(v <= 0 for v in (args.carry_goal_tolerance, args.carry_lateral_tolerance,
                             args.carry_max_slip, args.carry_max_drop)):
        parser.error("Carry tolerances must be positive")
    if any(v <= 0 for v in (args.diagnostic_search_radius,
                             args.max_close_seconds, args.gravity_hold_seconds)):
        parser.error("Search radius, close timeout and hold duration must be positive")
    if not 0 <= args.min_contact_fraction <= 1:
        parser.error("--min_contact_fraction must be 0..1")
    if args.contact_confirm_seconds < 0:
        parser.error("Contact confirmation must be nonnegative")
    if args.max_relative_drift <= 0 or args.max_vertical_drop <= 0:
        parser.error("Drift/drop thresholds must be positive")
    if not (0 < args.ball_friction <= 3 and 0 < args.finger_friction <= 3):
        parser.error('Physical friction must be >0 and <=3')
    if args.solver_iterations < 0 or args.solver_iterations > 500:
        parser.error('--solver_iterations must be 0..500')
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
