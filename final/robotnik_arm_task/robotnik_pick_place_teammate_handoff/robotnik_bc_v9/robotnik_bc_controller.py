#!/usr/bin/env python3
"""Robotnik V9: trained BC student carry + trained BC student release ACTIONS.

Approach, arm and grasp: original pretrained ML models, unchanged.
Carry: optional 5-input / 2-output BC student (v, omega), no expert steering.
Release: optional 17-input / 6-output BC student finger targets, no expert
opening commands. A measured destination threshold determines WHEN to switch
skills (a deterministic state-machine guard, NOT a learned release decision).
Physical ball contact, gravity and success checks are preserved from V8.2.
The standalone V8.2 expert runner remains in robotnik_expert_controller.py.
The BC student has NOT been closed-loop verified on Ubuntu by the author.
"""

import argparse
import hashlib
import json
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
from candidate_bc_policy import CandidatePolicy


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
    print("V9_BC_STUDENT: pretrained ML approach/arm/gripper; BC carry and BC release actions.")
    print("V9_CONTROL carry={} release={} (state transition uses ball XY threshold)".format(
        args.carry_controller,args.release_controller))
    print("V9_MODEL_STATUS: CANDIDATE; closed-loop performance must be verified on Robotnik simulator.")
    carry_student = (CandidatePolicy(args.carry_bc_checkpoint, expected_skill='carry')
                     if args.carry_controller == 'bc' else None)
    release_student = (CandidatePolicy(args.release_bc_checkpoint, expected_skill='release')
                       if args.release_controller == 'bc' else None)
    if carry_student is not None:
        print(f"V9_CARRY_BC_MODEL {args.carry_bc_checkpoint} sha256={carry_student.sha256}")
    if release_student is not None:
        print(f"V9_RELEASE_BC_MODEL {args.release_bc_checkpoint} sha256={release_student.sha256}")
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
    # The destination is drawn as debug lines, not as a second physical ball.
    try:
        gx, gy = (float(v) for v in args.dropoff_xy)
        for a, b in (([gx-.07,gy,0.045],[gx+.07,gy,0.045]),
                     ([gx,gy-.07,0.045],[gx,gy+.07,0.045])):
            p.addUserDebugLine(a, b, [0.0,0.7,0.0], lineWidth=3, lifeTime=0)
    except Exception as exc:
        print(f"V8_DEBUG_MARKER_UNAVAILABLE: {exc}")
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
                force=min(2.0,float(lim[10])),
                maxVelocity=1.0 if grasp_state == "release_open" else 0.5)
    sim._SetMotorTorqueByIds = bounded_finger_callback
    grasp_state = "approach"  # approach -> closing -> gravity_hold -> carry -> lowering -> release_open
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
    print("V4.1 GRASP MODE: TRAINED RESIDUAL-BC FINGERS (reverse disabled); V8.2 immediate expert open-at-goal")
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
    print(f"V8 TARGET: ball world XY={np.round(args.dropoff_xy,3).tolist()} (not the base centre)")
    print(f"V9 CARRY CONTROLLER: {args.carry_controller.upper()}  RELEASE: {args.release_controller.upper()} at ball XY goal")
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
                format=np.asarray('robotnik_physical_trace_v8'),
                obs=np.asarray(full_obs,dtype=np.float32),
                action=np.asarray(full_actions,dtype=np.float32),
                time=np.asarray(full_times), virtual_attachment=np.asarray(False),
                success_verified=np.asarray(False),
                policy_scope=np.asarray('hybrid_approach_ml_grasp_expert_waypoint_expert_release_v8'))
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
    # Data to train a release policy *later*. These are expert labels, not
    # predictions by the V4 grasp network. Do not treat them as successful
    # release demonstrations unless the physical release check also passes.
    release_sensor_obs = []
    release_target_actions = []
    release_times = []
    release_contacts_seen = []
    release_ball_xy_error_m = []
    dropoff_xy = np.asarray(args.dropoff_xy, dtype=float).reshape(2)
    dropoff_initial_range = None
    dropoff_target_reached = False
    release_started_at = None
    release_ball_start = None
    release_no_contact_since = None
    release_success = False
    release_target_success = False
    release_failure_reason = None
    release_contact_count = 0
    release_max_drop = 0.0
    release_final_distance = float('inf')
    # Expert opens to physical lower limits (not 0.1-rad reset angles).
    # Build at runtime from the six physical joint limits; never touch ML grasp targets.
    expert_open_fingers = np.asarray(
        [max(0.0, float(p.getJointInfo(rid, jid)[8])) for jid in finger_joint_ids],
        dtype=float)
    slip_warning_logged_at = None


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
            hold_ee_target = ee_pos if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "release_open", "reverse") else approach_ee_target
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
        if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "release_open", "reverse"):
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

        # -------- V9: BC policy steers CARRIED BALL or V8.2 expert as requested --------
        # No expert fallback occurs in 'bc' mode. The original V4 neural network
        # keeps the gripper closed throughout this stage.
        if grasp_state == "carry":
            ball_xy_for_control = np.asarray(p.getBasePositionAndOrientation(ball_id)[0][:2], dtype=float)
            to_goal = dropoff_xy - ball_xy_for_control
            remaining = float(np.linalg.norm(to_goal))
            desired_heading = float(np.arctan2(to_goal[1], to_goal[0])) if remaining > 1e-6 else float(base_bearing)
            bearing_error = float(np.arctan2(np.sin(desired_heading - base_bearing),
                                             np.cos(desired_heading - base_bearing)))
            origin_to_goal = dropoff_xy - carry_ball_origin[:2]
            path_unit = origin_to_goal / max(np.linalg.norm(origin_to_goal), 1e-8)
            pos_from_origin = ball_xy_for_control - carry_ball_origin[:2]
            lateral = float(path_unit[0]*pos_from_origin[1] - path_unit[1]*pos_from_origin[0])
            # This observation is IDENTICAL to the stored 5-element expert dataset.
            fwd_vel = float(np.dot(base_lin_vel[:2],
                                   [np.cos(base_bearing), np.sin(base_bearing)]))
            bc_carry_obs = np.asarray([remaining, lateral, bearing_error,
                                       fwd_vel, float(base_ang_vel[2])], dtype=np.float64)
            if remaining <= args.carry_goal_tolerance:
                # Deterministic skill-transition safety guard, not a steering expert.
                v_cmd, w_cmd = 0.0, 0.0
            elif carry_student is not None:
                predicted = np.asarray(carry_student.predict(bc_carry_obs)).reshape(-1)
                v_cmd = float(np.clip(predicted[0], 0.0, args.carry_speed_max))
                w_cmd = float(np.clip(predicted[1], -args.carry_yaw_max, args.carry_yaw_max))
            else:
                # EXPERT BASELINE ONLY: used to generate teacher demonstrations.
                yaw_scale = max(0.0, float(np.cos(bearing_error)))
                if abs(bearing_error) > 0.55:
                    yaw_scale *= 0.20
                v_cmd = float(np.clip(args.expert_kp * remaining * yaw_scale,
                                      0.0, args.carry_speed_max))
                w_cmd = float(np.clip(args.expert_heading_kp * bearing_error,
                                      -args.carry_yaw_max, args.carry_yaw_max))
            if step % 5 == 0 and args.carry_controller == 'expert':
                carry_obs.append(bc_carry_obs.tolist())
                carry_actions.append([v_cmd, w_cmd])
                carry_times.append(current_time-carry_started_at)
            rw = (v_cmd + w_cmd * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
            lw = (v_cmd - w_cmd * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
            wheel_cmds = np.array([rw, lw, lw, rw], dtype=float)
            if step % 250 == 0:
                print(f"V9_{args.carry_controller.upper()}_NAV t={current_time:.2f}s "
                      f"ball_xy={np.round(ball_xy_for_control,3).tolist()} "
                      f"target_xy={np.round(dropoff_xy,3).tolist()} "
                      f"remaining_m={remaining:.3f} heading_error_rad={bearing_error:.3f} "
                      f"v={v_cmd:.3f} w={w_cmd:.3f}")

        if grasp_state in ("carry_settle", "release_open"):
            # The base must be stationary while checking the hold and releasing.
            wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - carry_stop_wheel_q), -5.0, 5.0)

        if grasp_state == "release_open":
            if release_student is not None:
                # Sensor feature order matches release_obs_17 in demonstrations.
                release_ball_xyz = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
                obs_release = np.asarray(sensor_features(
                    ee_pos, release_ball_xyz, q_mes[11:17], qd_mes[11:17],
                    base_lin_vel[:2], base_bearing), dtype=np.float64)
                target_finger_angles[:] = release_student.predict(obs_release).reshape(6)
            else:
                # Expert demonstration mode only; not executed in student mode.
                target_finger_angles[:] = expert_open_fingers
            # The trained V4 grasp policy is inactive during release_open.
            # Only chosen release controller may set motor targets this step.
            real_open = bool(np.all(np.asarray(q_mes[11:17]) <= expert_open_fingers + 0.10))
            if step % 100 == 0:
                print(f"V8_2_FINGERS_OPENING t={current_time:.2f}s "
                      f"targets={np.round(target_finger_angles,3).tolist()} "
                      f"measured={np.round(q_mes[11:17],3).tolist()} "
                      f"all_joint_open={real_open}")

        # Log observable physics, never claim success from robot motion alone.
        ball_xyz = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
        robot_contacts = p.getContactPoints(bodyA=rid, bodyB=ball_id)
        finger_contacts = [cp for cp in robot_contacts if cp[3] in finger_joint_ids]
        contact_groups = sum(any(cp[3] in finger_joint_ids[i:i+2]
                             for cp in finger_contacts) for i in (0, 2, 4))
        finger_normal_force = sum(max(0.0, float(cp[9])) for cp in finger_contacts)
        peak_contact_normal_force = max(peak_contact_normal_force, finger_normal_force)
        if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "release_open"):
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
                        # Ball is now dynamic and grasp stability was checked.
                        carry_started_at = current_time
                        carry_origin_xy = base_pos[:2].copy()
                        carry_heading = np.array([np.cos(base_bearing), np.sin(base_bearing)])
                        carry_goal_xy = dropoff_xy.copy()  # target is for BALL, not robot base
                        carry_ball_origin = ball_xyz.copy()
                        carry_ball_to_pocket_start = ball_xyz - pocket_world()
                        dropoff_initial_range = float(np.linalg.norm(dropoff_xy - ball_xyz[:2]))
                        grasp_state = "carry"
                        print(f"V8_START_EXPERT_NAV t={current_time:.2f}s "
                              f"ball_from_xy={np.round(carry_ball_origin[:2],3).tolist()} "
                              f"dropoff_ball_xy={np.round(dropoff_xy,3).tolist()} "
                              f"range_m={dropoff_initial_range:.3f} "
                              f"no_attachment=True ball_dynamic={ball_dynamic}")
                    else:
                        hold_failure_reason = (
                            f"Insufficient sustained contacts: "
                            f"two-group fraction={contact_fraction:.3f}, "
                            f"final_groups={current_groups}")
                        print(f"V4_HOLD_FAIL: {hold_failure_reason}")
                        break
        if grasp_state in ("carry", "carry_settle"):
            # Independent physical-grasp check while approaching the drop-off.
            carry_total_samples += 1
            carry_last_groups = list(current_groups)
            if len(current_groups) >= 2:
                carry_good_samples += 1
                carry_contact_loss_since = None
            elif carry_contact_loss_since is None:
                carry_contact_loss_since = current_time
            carry_drift = float(np.linalg.norm((ball_xyz-pocket_world())-carry_ball_to_pocket_start))
            carry_drop = float(carry_ball_origin[2]-ball_xyz[2])
            carry_max_relative_drift = max(carry_max_relative_drift,carry_drift)
            carry_max_ball_drop = max(carry_max_ball_drop,carry_drop)
            dropoff_error = float(np.linalg.norm(dropoff_xy-ball_xyz[:2]))
            ball_speed_xy = float(np.linalg.norm(p.getBaseVelocity(ball_id)[0][:2]))
            if step % 250 == 0:
                print(f"V8_PHYSICAL_CARRY t={current_time:.2f}s state={grasp_state} "
                      f"ball_xy={np.round(ball_xyz[:2],3).tolist()} "
                      f"dropoff_error_m={dropoff_error:.3f} "
                      f"groups={current_groups} force_N={finger_normal_force:.2f} "
                      f"pocket_slip_mm={carry_drift*1000:.1f} drop_mm={carry_drop*1000:.1f}")
            # V8.2: the requested action is simply OPEN AT THE TARGET.
            # Arrival uses the *physical ball's* measured world XY, not base XY.
            # Open immediately on reaching tolerance.  No waiting for a stationary
            # ball, repeated grip checks, or a timed opening sweep.
            if grasp_state == "carry" and dropoff_error <= args.carry_goal_tolerance:
                dropoff_target_reached = True
                carry_end_time = current_time
                carry_stop_wheel_q = q_mes[0:4].copy()
                carry_success = bool(len(current_groups) >= 2
                                     and carry_drift <= args.release_max_grip_offset
                                     and carry_drop <= args.release_max_grip_drop)
                grasp_state = "release_open"
                release_started_at = current_time
                release_ball_start = ball_xyz.copy()
                if release_student is not None:
                    obs_release = np.asarray(sensor_features(
                        ee_pos, ball_xyz, q_mes[11:17], qd_mes[11:17],
                        base_lin_vel[:2], base_bearing), dtype=np.float64)
                    target_finger_angles[:] = release_student.predict(obs_release).reshape(6)
                else:
                    target_finger_angles[:] = expert_open_fingers
                wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - carry_stop_wheel_q), -5.0, 5.0)
                print(f"V8_2_TARGET_REACHED_OPEN_NOW t={current_time:.2f}s "
                      f"ball_xy={np.round(ball_xyz[:2],3).tolist()} "
                      f"goal_xy={np.round(dropoff_xy,3).tolist()} "
                      f"xy_error_m={dropoff_error:.3f} "
                      f"finger_open_targets={np.round(expert_open_fingers,3).tolist()}")
                print(f"V9_{args.release_controller.upper()}_RELEASE_START t={current_time:.2f}s "
                      f"immediate=True ball_dynamic={ball_dynamic} "
                      f"groups={current_groups} "
                      f"carry_physically_verified={carry_success}")
                if not carry_success:
                    print("V8_2_WARNING: Ball reached goal but grip was weak. "
                          "Opening as requested; do not count as a verified carry.")
            else:
                # If contact fails *before* reaching the target we cannot honestly
                # claim an at-target ball release. Do not teleport the ball.
                if carry_max_relative_drift > args.carry_max_slip or carry_max_ball_drop > args.carry_max_drop:
                    if slip_warning_logged_at is None or current_time-slip_warning_logged_at > 3.0:
                        print(f"V8_2_CARRY_SLIP_WARNING max_slip_mm={carry_max_relative_drift*1000:.1f} "
                              f"max_drop_mm={carry_max_ball_drop*1000:.1f}")
                        slip_warning_logged_at = current_time
                if carry_drift > args.release_max_grip_offset or carry_drop > args.release_max_grip_drop:
                    carry_fail_reason=(f"GRIP BROKEN BEFORE TARGET: current_offset={carry_drift*1000:.1f}mm "
                                       f"current_drop={carry_drop*1000:.1f}mm; "
                                       f"remaining_m={dropoff_error:.3f}")
                elif carry_contact_loss_since is not None and current_time-carry_contact_loss_since > 0.15:
                    carry_fail_reason=(f"BALL LOST BEFORE TARGET: two-finger contact absent >0.15s; "
                                       f"remaining_m={dropoff_error:.3f}")
                elif current_time-carry_started_at > args.carry_timeout:
                    carry_fail_reason=(f"Carry timeout BEFORE TARGET: remaining_m={dropoff_error:.3f}")
            if carry_fail_reason:
                print(f"V8_CARRY_FAIL {carry_fail_reason}")
                break

        if grasp_state == "release_open":
            if step % 5 == 0:
                observed_ball = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
                if args.release_controller == 'expert':
                    release_sensor_obs.append(np.asarray(sensor_features(
                        ee_pos, observed_ball, q_mes[11:17], qd_mes[11:17],
                        base_lin_vel[:2], base_bearing), dtype=np.float32))
                    release_target_actions.append(np.asarray(expert_open_fingers, dtype=np.float32).copy())
                    release_times.append(float(current_time - release_started_at))
                    release_contacts_seen.append(int(len(finger_contacts)))
                    release_ball_xy_error_m.append(float(np.linalg.norm(
                        dropoff_xy - observed_ball[:2])))
            # The ball MUST detach and fall naturally; no teleport, resetBase,
            # virtual constraint, or anti-gravity force is applied.
            all_release_contacts = p.getContactPoints(bodyA=rid, bodyB=ball_id)
            release_contacts = [cp for cp in all_release_contacts
                                if cp[3] in finger_joint_ids]
            release_contact_count = len(release_contacts)
            # Detachment means the ball is clear of the ENTIRE robot, not
            # merely no longer touching a fingertip while resting on the palm.
            robot_release_contact_count = len(all_release_contacts)
            if robot_release_contact_count == 0:
                if release_no_contact_since is None:
                    release_no_contact_since = current_time
            else:
                release_no_contact_since = None
            release_drop=max(0.0, float(release_ball_start[2]-ball_xyz[2]))
            release_max_drop=max(release_max_drop,release_drop)
            release_final_distance=float(np.linalg.norm(dropoff_xy-ball_xyz[:2]))
            if step%250==0:
                print(f"V8_RELEASE_MONITOR t={current_time:.2f}s "
                      f"open_elapsed={current_time-release_started_at:.2f}s "
                      f"finger_contacts={release_contact_count} "
                      f"drop_m={release_drop:.3f} ball_xy_error_m={release_final_distance:.3f} "
                      f"ball_xyz={np.round(ball_xyz,3).tolist()}")
            if (release_no_contact_since is not None
                and current_time-release_no_contact_since >= args.release_no_contact_seconds
                and release_drop >= args.release_min_drop
                and current_time-release_started_at >= args.release_min_seconds):
                release_success=True
                release_target_success=bool(release_final_distance<=args.release_final_tolerance)
                print(f"V8_RELEASE_RESULT DETACHED_AND_FELL "
                      f"ball_xy_error_m={release_final_distance:.3f} "
                      f"placement_within_tolerance={release_target_success} "
                      f"drop_m={release_drop:.3f} no_attachment=True")
                break
            if current_time-release_started_at > args.release_timeout:
                release_failure_reason=(f"Expert opening timed out: remaining_robot_contacts={robot_release_contact_count} "
                                        f"max_drop_m={release_max_drop:.3f} ball_xy_error_m={release_final_distance:.3f}")
                print(f"V8_RELEASE_FAIL {release_failure_reason}")
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
    print("========== V9 BC STUDENT NAVIGATE + RELEASE RESULT ==========")
    print("NOTE: RELEASE_TO_GRAVITY is NOT finger-opening; real release starts at V9_*_RELEASE_START.")
    print(f"Requested BALL drop-off (world XY): {np.round(dropoff_xy,3).tolist()}")
    print(f"Transfer: {args.carry_controller.upper()} base; grasp: V4 ML; "
          f"release: {args.release_controller.upper()} finger actions")
    print(f"Carry success before opening: {carry_success}")
    print(f"V8_2_ARRIVAL_REACHED: {dropoff_target_reached}")
    print(f"V8_2_OPEN_COMMAND_SENT: {release_started_at is not None}")
    print(f"Ball detached and fell under gravity: {release_success}")
    print(f"Ball XY within {args.release_final_tolerance:.2f}m of destination after release: {release_target_success}")
    # TASK DEFINITION: physical pick, carry and RELEASE AT GOAL is success.
    # Landing accuracy remains an independent, informational metric.
    release_at_goal_success = bool(carry_success and dropoff_target_reached
                                   and release_started_at is not None and release_success)
    accurate_placement_success = bool(release_at_goal_success and release_target_success)
    print(f"V8_RELEASE_AT_TARGET_RESULT: {'PASS' if release_at_goal_success else 'FAIL/NOT_VERIFIED'}")
    print(f"V8_FINAL_PLACEMENT_DIAGNOSTIC: within_tolerance={accurate_placement_success} "
          f"final_xy_error_m={release_final_distance:.3f} "
          f"tolerance_m={args.release_final_tolerance:.3f} (not required for task success)")
    print(f"V8_PICK_CARRY_DROP_RESULT: {'PASS' if release_at_goal_success else 'FAIL/NOT_VERIFIED'}")
    print(f"V9_STUDENT_PICK_CARRY_DROP_RESULT: "
          f"{'PASS' if release_at_goal_success else 'FAIL/NOT_VERIFIED'} "
          f"carry={args.carry_controller} release={args.release_controller}")
    print(f"Maximum relative slip while carrying: {carry_max_relative_drift*1000:.1f} mm")
    print(f"Maximum downward movement while carrying: {carry_max_ball_drop*1000:.1f} mm")
    print(f"Contact samples with >=2 finger groups during carry: {carry_good_samples/max(1,carry_total_samples):.3f}")
    print(f"Release mode={args.release_controller}: nominal open limits={np.round(expert_open_fingers,3).tolist()} "
          f"max_ball_drop_m={release_max_drop:.3f} final_xy_error_m={release_final_distance:.3f}")
    print(f"Carry failure reason: {carry_fail_reason}")
    print(f"Release failure reason: {release_failure_reason}")
    print("Approach, arm and grasp remain trained ML; carry/release action source listed above.")
    if args.record_expert_npz and args.carry_controller == 'expert' and args.release_controller == 'expert':
        with Path(args.record_expert_npz).open('xb') as output:
            np.savez_compressed(output,
                feature_names=np.asarray(['remaining_m', 'lateral_m', 'heading_err_rad',
                                         'forward_speed_mps', 'yaw_rate_radps']),
                obs=np.asarray(carry_obs, dtype=np.float32).reshape(-1, 5),
                actions_vw=np.asarray(carry_actions, dtype=np.float32).reshape(-1, 2),
                elapsed_s=np.asarray(carry_times, dtype=np.float32),
                source=np.asarray('SCRIPTED_EXPERT_TRANSFER_ONLY; V4_ML_GRASP'),
                physical_success_verified=np.asarray(bool(release_at_goal_success)),
                success_definition=np.asarray('CARRIED_BALL_RELEASED_AT_TARGET_AND_PHYSICALLY_FELL; final landing accuracy not required'),
                release_at_goal_verified=np.asarray(bool(release_at_goal_success)),
                final_placement_verified=np.asarray(bool(accurate_placement_success)),
                final_ball_xy_error_m=np.asarray(float(release_final_distance)),
                final_ball_xy_tolerance_m=np.asarray(float(args.release_final_tolerance)),
                carries_ball_with_friction=np.asarray(True),
                no_virtual_attachment=np.asarray(True),
                target_distance_m=np.asarray(dropoff_initial_range if dropoff_initial_range is not None else np.nan),
                dropoff_xy=np.asarray(dropoff_xy),
                ball_released_verified=np.asarray(bool(release_success)),
                release_obs_17=np.asarray(release_sensor_obs, dtype=np.float32).reshape(-1, 17),
                release_expert_finger_targets_rad=np.asarray(release_target_actions, dtype=np.float32).reshape(-1, 6),
                release_elapsed_s=np.asarray(release_times, dtype=np.float32),
                release_finger_contact_counts=np.asarray(release_contacts_seen, dtype=np.int32),
                release_ball_goal_error_m=np.asarray(release_ball_xy_error_m, dtype=np.float32),
                release_label_source=np.asarray('EXPLICIT_PROGRAMMED_EXPERT_OPEN_NOT_LEARNED'),
                caution=np.asarray('One expert trace; only train on physically verified successful releases, with multiple varied rollouts'))
        print(f"V8_EXPERT_TRACE_SAVED {args.record_expert_npz} frames={len(carry_obs)} "
              f"verified_pick_carry_drop_success={release_at_goal_success} "
              f"release_at_goal_success={release_at_goal_success}")
    print("V4.1 uses the unchanged V4 ML model trained on unverified virtual-attachment demonstrations.")
    print("V9 BC mode is learned low-level carry and release actions; arrival state switch remains rule-based.")
    print("No object attachment, no scripted grasp sweep, and no unverified transport success claimed.")
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
        "task_success": bool(release_at_goal_success),
        "grasp_candidate": grasp_success,
        "carry_success": carry_success,
        "release_success": release_success,
        "release_target_success": release_target_success,
        "release_at_goal_success": release_at_goal_success,
        "accurate_placement_success": accurate_placement_success,
        "carry_mode": args.carry_controller,
        "release_mode": args.release_controller,
        "carry_bc_sha256": carry_student.sha256 if carry_student else None,
        "release_bc_sha256": release_student.sha256 if release_student else None,
        "dropoff_xy": dropoff_xy.tolist(),
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
    parser = argparse.ArgumentParser(description="Robotnik V9: learned pickup/grasp plus separately trained BC carry and release action policies")
    parser.add_argument("--carry_controller", choices=["expert", "bc"], default="bc",
                        help="BC for inference, expert only for collecting demonstrations")
    parser.add_argument("--release_controller", choices=["expert", "bc"], default="bc",
                        help="BC learned finger targets or the V8.2 expert baseline")
    parser.add_argument("--carry_bc_checkpoint", default=None,
                        help="Path to trained carry_candidate_bc.npz (5 observation inputs, 2 actions)")
    parser.add_argument("--release_bc_checkpoint", default=None,
                        help="Path to trained release_candidate_bc.npz (17 observation inputs, 6 actions)")
    parser.add_argument("--result_json", default=None,
                        help="Write physics/result report with actual candidate checkpoint SHA256s")
    parser.add_argument("--dropoff_xy", type=float, nargs=2, default=[3.0, 3.5],
                        metavar=("X", "Y"),
                        help="World XY position where the BALL should be released (default 3.0 3.5)")
    parser.add_argument("--carry_speed_max", type=float, default=0.025,
                        help="Maximum speed during carry, a safety cap not an ML command")
    parser.add_argument("--carry_yaw_max", type=float, default=0.12,
                        help="Safety cap on base angular velocity")
    parser.add_argument("--expert_kp", type=float, default=0.6,
                        help="Proportional velocity gain for expert demonstration ONLY")
    parser.add_argument("--carry_timeout", type=float, default=220.0,
                        help="Maximum active carry time before reporting failure")
    parser.add_argument("--carry_settle_seconds", type=float, default=1.0,
                        help="Unused legacy compatibility option; V8.2 opens immediately")
    parser.add_argument("--carry_goal_tolerance", type=float, default=0.045,
                        help="Allowed forward displacement error in metres")
    parser.add_argument("--carry_settle_allowance", type=float, default=0.025,
                        help="Additional XY tolerance after the base comes to rest")
    parser.add_argument("--carry_arrival_speed_max", type=float, default=0.03,
                        help="Maximum ball XY speed after settling at destination (m/s)")
    parser.add_argument("--expert_heading_kp", type=float, default=1.0,
                        help="Expert steering gain (rad/s per heading-error radian)")
    parser.add_argument("--release_min_seconds", type=float, default=0.5,
                        help="Minimum time to command expert finger opening")
    parser.add_argument("--release_no_contact_seconds", type=float, default=0.4,
                        help="Time with no finger contact needed to verify release")
    parser.add_argument("--release_min_drop", type=float, default=0.12,
                        help="Required actual downward ball fall to certify detached (metres)")
    parser.add_argument("--release_timeout", type=float, default=17.0,
                        help="Maximum expert finger opening observation time")
    parser.add_argument("--release_backoff_delay", type=float, default=1.8,
                        help="Unused in V8.2; backoff intentionally disabled")
    parser.add_argument("--release_backoff_speed", type=float, default=0.018,
                        help="Unused in V8.2; backoff intentionally disabled")
    parser.add_argument("--release_backoff_distance", type=float, default=0.10,
                        help="Unused in V8.2; backoff intentionally disabled")
    parser.add_argument("--release_max_grip_offset", type=float, default=0.085,
                        help="Physical ball-pocket displacement threshold to abort carrying")
    parser.add_argument("--release_max_grip_drop", type=float, default=0.07,
                        help="Physical ball downward displacement threshold to abort carrying")
    parser.add_argument("--carry_settle_max_seconds", type=float, default=6.0,
                        help="Maximum wait for stationary held ball before abort")
    parser.add_argument("--release_final_tolerance", type=float, default=0.18,
                        help="Maximum ball XY error after it falls, for placement PASS")
    parser.add_argument("--carry_max_slip", type=float, default=0.030,
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
    parser.add_argument("--max_steps", type=int, default=350000,
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
    if args.record_expert_npz and (args.carry_controller != "expert" or args.release_controller != "expert"):
        parser.error("--record_expert_npz requires both carry and release controllers to be expert")
    if args.result_json and Path(args.result_json).exists():
        parser.error("--result_json exists; refusing to overwrite")
    if args.carry_controller == "bc" and not args.carry_bc_checkpoint:
        parser.error("BC carry requires --carry_bc_checkpoint (train it first)")
    if args.release_controller == "bc" and not args.release_bc_checkpoint:
        parser.error("BC release requires --release_bc_checkpoint (train it first)")
    if len(args.dropoff_xy)!=2 or not np.all(np.isfinite(args.dropoff_xy)):
        parser.error("--dropoff_xy requires two finite world coordinates")
    if np.linalg.norm(np.asarray(args.dropoff_xy)-np.asarray(args.target[:2])) < 0.15:
        parser.error("Drop-off point must be at least 0.15m from the ball pickup point")
    if not 0 < args.carry_speed_max <= 0.10:
        parser.error("--carry_speed_max must be >0 and <=0.10m/s")
    if not 0 < args.carry_yaw_max <= 1:
        parser.error("--carry_yaw_max must be >0 and <= 1rad/s")
    if args.carry_timeout <= 0 or args.carry_settle_seconds <= 0 or args.release_timeout <= 0:
        parser.error("Carry, settling, and release times must be positive")
    if any(v<=0 for v in (args.expert_heading_kp,args.release_min_seconds,
                         args.release_no_contact_seconds,args.release_min_drop,
                         args.release_final_tolerance,args.carry_arrival_speed_max)):
        parser.error("Expert steering and release parameters must be positive")
    if not 0 <= args.carry_min_contact_fraction <= 1:
        parser.error("Carry contact fraction must be [0,1]")
    if any(v <= 0 for v in (args.carry_goal_tolerance, args.carry_settle_allowance,
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
    if not np.all(np.isfinite(args.target)):
        parser.error("Pickup --target must contain finite coordinates")
    if (args.release_backoff_delay<=0 or not 0<args.release_backoff_speed<=0.05
        or not 0<args.release_backoff_distance<=0.3 or
        args.release_max_grip_offset<=args.carry_max_slip or
        args.release_max_grip_drop<=args.carry_max_drop or
        args.carry_settle_max_seconds<args.carry_settle_seconds):
        parser.error("Invalid expert clearance or settle/grip limits")
    if args.carry_goal_tolerance >= args.release_final_tolerance:
        parser.error("--release_final_tolerance must exceed --carry_goal_tolerance")
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
            if args.result_json:
                summary = {
                    "system": "V9 candidate BC carry and release action evaluation",
                    "task_success": bool(result["task_success"]),
                    "gravity_probe_passed": bool(result["gravity_probe_passed"]),
                    "grasp_held_under_gravity": bool(result["grasp_candidate"]),
                    "carry_success": bool(result["carry_success"]),
                    "release_at_goal_success": bool(result["release_at_goal_success"]),
                    "release_success": bool(result["release_success"]),
                    "carry_mode": result["carry_mode"],
                    "release_mode": result["release_mode"],
                    "carry_bc_sha256": result["carry_bc_sha256"],
                    "release_bc_sha256": result["release_bc_sha256"],
                    "goal_xy": [float(v) for v in args.dropoff_xy],
                    "pickup_xyz": [float(v) for v in args.target],
                    "landing_within_old_tolerance": bool(result["accurate_placement_success"]),
                    "full_closed_loop_student_test": bool(result["carry_mode"] == "bc" and result["release_mode"] == "bc"),
                    "note": "Task success is physical release at goal; floor landing XY is not graded",
                }
                with Path(args.result_json).open('x') as out:
                    json.dump(summary, out, indent=2)
                    out.write('\n')
                print(f"V9_STUDENT_RESULT_JSON: {args.result_json}")
            break
    else:
        print(f"\nAll {max_attempts} attempts crashed.")


if __name__ == "__main__":
    main()
