#!/usr/bin/env python3
"""Hybrid Robotnik pick-and-place expert with guarded ground placement.

Pretrained ML base/arm/gripper policies approach and grasp a dynamic ball.
Scripted expert navigation carries it toward a user-selected world XY landing
point; arm-only IK then lowers the held ball to a stable release pose. The
gripper opens after the placement-pose checks pass, and PyBullet verifies the
gravity fall, ground contact, and final XY error. This is not end-to-end ML.
"""

import argparse
import hashlib
import os
import sys
import time
import atexit
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
from learned_release_student import LearnedReleaseStudent
from placement_policy import placement_features
from learned_return_home_policy import LearnedReturnHomePolicy
from ground_placement_control import BallCartesianLoweringPlanner
from reachability_feedback import select_base_adjustment, wheel_command_for_displacement
from drop_zone_control import CircularDropZone, draw_drop_zone


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


def arm_chain_joint_ids(pybullet, robot_id, hand_link, expected=7):
    """Return the revolute joints on the end-effector ancestry, proximal first."""
    chain = []
    link = int(hand_link)
    while link >= 0:
        info = pybullet.getJointInfo(robot_id, link)
        if info[2] != pybullet.JOINT_FIXED:
            chain.append(link)
        link = int(info[16])
    chain.reverse()
    if len(chain) != expected:
        names = [pybullet.getJointInfo(robot_id, jid)[1].decode("utf-8") for jid in chain]
        raise RuntimeError(
            f"Expected {expected} actuated joints in the arm chain to link {hand_link}; "
            f"found {len(chain)}: {names}. Refusing ground placement with an uncertain joint map.")
    return chain


def solve_arm_ik(pybullet, robot_id, hand_link, arm_joint_ids,
                 target_position, target_orientation):
    """Solve arm IK while locking wheels/fingers at their measured positions."""
    movable = [j for j in range(pybullet.getNumJoints(robot_id))
               if pybullet.getJointInfo(robot_id, j)[2] != pybullet.JOINT_FIXED]
    infos = [pybullet.getJointInfo(robot_id, j) for j in movable]
    lower, upper, ranges, rest = [], [], [], []
    arm_joint_set = set(int(j) for j in arm_joint_ids)
    for jid, info in zip(movable, infos):
        current = float(pybullet.getJointState(robot_id, jid)[0])
        lo, hi = float(info[8]), float(info[9])
        if jid in arm_joint_set:
            # Arm joints may move within their physical URDF bounds.
            if not np.isfinite(lo + hi) or hi <= lo:
                lo, hi = current - 2.0 * np.pi, current + 2.0 * np.pi
        else:
            # PyBullet IK sees wheels and fingers as movable DOFs too. If left
            # free, it can use them to satisfy the hand target, then we discard
            # those returned wheel/finger angles. Lock every non-arm DOF at its
            # measured value so the extracted arm solution is self-consistent.
            eps = 1e-6
            lo, hi = current - eps, current + eps
        lower.append(lo)
        upper.append(hi)
        ranges.append(hi - lo)
        rest.append(current)
    solution = pybullet.calculateInverseKinematics(
        robot_id, hand_link,
        targetPosition=np.asarray(target_position, dtype=float).tolist(),
        targetOrientation=np.asarray(target_orientation, dtype=float).tolist(),
        lowerLimits=lower, upperLimits=upper, jointRanges=ranges,
        restPoses=rest, maxNumIterations=200, residualThreshold=1e-4)
    if len(solution) != len(movable):
        raise RuntimeError(
            f"PyBullet IK returned {len(solution)} values for {len(movable)} movable joints")
    by_id = dict(zip(movable, solution))
    targets = np.asarray([by_id[j] for j in arm_joint_ids], dtype=float)
    if targets.shape != (7,) or not np.isfinite(targets).all():
        raise RuntimeError(f"Invalid 7-joint ground-placement IK solution: {targets}")
    arm_bounds = [pybullet.getJointInfo(robot_id, j)[8:10] for j in arm_joint_ids]
    for i, (lo, hi) in enumerate(arm_bounds):
        lo, hi = float(lo), float(hi)
        if np.isfinite(lo + hi) and hi > lo:
            targets[i] = np.clip(targets[i], lo, hi)
    return targets




def probe_ground_reachability(pybullet, robot_id, hand_link, arm_joint_ids,
                              ball_xyz, ball_minus_hand, goal_ball_xyz,
                              hand_orientation, heading, offsets=(-0.075, 0.0, 0.075)):
    """Dry-run IK/FK at real and nearby base poses; always restore dynamics.

    No simulation steps are made in the probe, and neither robot nor ball is
    moved persistently. The only physical displacement comes later from wheel
    velocity commands. Score uses FK positional residual over several heights.
    """
    saved_state = pybullet.saveState()
    samples = {}
    try:
        start_pos, start_ori = pybullet.getBasePositionAndOrientation(robot_id)
        start_pos = np.asarray(start_pos, dtype=float)
        heading_direction = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        z_start = float(ball_xyz[2])
        z_goal = float(goal_ball_xyz[2])
        heights = np.linspace(z_start, z_goal, 9)[1:]
        for offset in offsets:
            pybullet.restoreState(saved_state)
            candidate_base = start_pos.copy()
            candidate_base[:2] += float(offset) * heading_direction
            pybullet.resetBasePositionAndOrientation(robot_id, candidate_base.tolist(), start_ori)
            residuals = []
            for height in heights:
                target_ball = np.array([goal_ball_xyz[0], goal_ball_xyz[1], height])
                desired_hand = target_ball - ball_minus_hand
                try:
                    proposed_q = solve_arm_ik(pybullet, robot_id, hand_link, arm_joint_ids,
                                              desired_hand, hand_orientation)
                    for j, q in zip(arm_joint_ids, proposed_q):
                        pybullet.resetJointState(robot_id, j, targetValue=float(q))
                    achieved_hand = np.asarray(pybullet.getLinkState(
                        robot_id, hand_link, computeForwardKinematics=True)[4], dtype=float)
                    residuals.append(float(np.linalg.norm(achieved_hand - desired_hand)))
                except Exception:
                    residuals.append(1.0)
            samples[float(offset)] = float(max(residuals))
    finally:
        pybullet.restoreState(saved_state)
        pybullet.removeState(saved_state)
    return samples


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
    print(f"CONTROL SOURCES: carry={args.carry_controller}; lowering/release={args.placement_controller}")
    print("APPROACH/GRASP: existing trained base, arm and V4 finger models.")
    print("Task guards, joint tracking and physical success checks remain programmed.")
    print("New candidate policies require successful closed-loop evaluation.")

    return_home_policy = (LearnedReturnHomePolicy.load(args.return_home_checkpoint)
                          if args.return_home_controller == "bc" else None)
    print(f"RETURN_HOME_CONTROLLER={args.return_home_controller} "
          f"checkpoint={args.return_home_checkpoint if return_home_policy else 'none'}")
    return_home_demo_obs = []
    return_home_demo_actions = []
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
    if not args.headless:
        try:
            # Keep PyBullet's real physics: disable only expensive visual extras.
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            print("V20_RENDER: GUI controls/shadows disabled; physics unchanged")
        except Exception as exc:
            print(f"V20_RENDER: visual optimization unavailable ({exc})")

    ball_radius = 0.03       # 3 cm radius
    ball_mass = 0.010        # 10 grams
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
    print('V4 GRAVITY-HOLD: static ball during approach/closure; dynamic 10g ball after confirmed contacts')
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
    arm_joint_ids = arm_chain_joint_ids(p, rid, hand_link)
    print("GROUND PLACE ARM CHAIN:", [p.getJointInfo(rid, j)[1].decode() for j in arm_joint_ids])
    # Placement references are joint positions, tracked by a shared bounded
    # actuator servo in BOTH teacher and student modes.
    placement_servo_target = None
    previous_motor_callback = sim._SetMotorTorqueByIds
    def bounded_finger_callback(motor_ids, commands, index=0):
        previous_motor_callback(motor_ids, commands, index=index)
        if (index == 0 and args.place_on_ground and
                grasp_state in ("lowering", "release_open", "return_home") and
                placement_servo_target is not None):
            for i, jid in enumerate(arm_joint_ids):
                info = p.getJointInfo(rid, jid)
                target = float(placement_servo_target[i])
                if info[8] < info[9]:
                    target = float(np.clip(target, info[8], info[9]))
                effort = float(info[10])
                p.setJointMotorControl2(rid, jid, p.POSITION_CONTROL,
                    targetPosition=target,
                    force=min(args.placement_servo_force, effort) if effort > 0 else args.placement_servo_force,
                    maxVelocity=min(args.placement_servo_speed, float(info[11])) if info[11] > 0 else args.placement_servo_speed,
                    positionGain=args.placement_servo_position_gain, velocityGain=1.0)
        for i,jid in enumerate(finger_joint_ids):
            lim = p.getJointInfo(rid,jid)
            release_force = (args.ground_release_finger_force
                             if grasp_state == "release_open" and args.place_on_ground
                             else 2.0)
            p.setJointMotorControl2(rid,jid,p.POSITION_CONTROL,
                targetPosition=float(np.clip(target_finger_angles[i],lim[8],lim[9])),
                force=min(release_force,float(lim[10])),
                maxVelocity=(args.ground_release_finger_speed
                             if grasp_state == "release_open" and args.place_on_ground
                             else (1.0 if grasp_state == "release_open" else 0.5)))
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
    print(f"CARRY={args.carry_controller}; PLACEMENT={args.placement_controller}; checkpoints required in BC mode")
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
    placement_policy = (LearnedReleaseStudent(args.placement_checkpoint)
                        if args.placement_controller == 'bc' else None)
    carry_policy = (CandidatePolicy(args.carry_checkpoint, 'carry')
                    if args.carry_controller == 'bc' else None)
    placement_obs, placement_actions, placement_times = [], [], []
    # Dataset aggregation: label the student's VISITED states but NEVER use
    # counterfactual labels as motor commands during the rollout.
    correction_obs, correction_actions = [], []
    if getattr(args, "record_corrections_npz", None):
        def save_corrections():
            if not correction_obs:
                print("V20_CORRECTIONS_EMPTY: student never reached lowering", flush=True)
                return
            out = Path(args.record_corrections_npz)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, correction_obs_37=np.asarray(correction_obs, np.float32),
                correction_actions_14=np.asarray(correction_actions,np.float32),
                label_source=np.asarray("COUNTERFACTUAL_IK_TRAINING_ONLY"),
                applied_teacher_actions=np.asarray(False))
            print(f"V20_CORRECTIONS_SAVED {out} frames={len(correction_obs)} "
                  "applied_teacher_actions=False", flush=True)
        atexit.register(save_corrections)
    placement_tick = -1
    placement_record_tick = -1
    placement_arm_target = None
    placement_prediction = None
    learned_release_finger_targets = None
    print(f"ACTION_SOURCES carry={args.carry_controller} placement={args.placement_controller}")
    if placement_policy is not None:
        print("V20_RELEASE_STUDENT_INFERENCE: trained neural lowering and finger opening; "
              "scripted expert release action DISABLED; physical safety gates remain")
    ground_lower_started_at = None
    ground_lower_start_q = None
    ground_lower_target_q = None
    ground_lower_target_ball = None
    ground_lower_reached_at = None
    # V18 expert: small, feedback-controlled Cartesian steps instead of a
    # joint-space interpolation toward one distant IK solution.
    ground_planner = None
    ground_last_servo_update = None
    ground_hand_orientation = None
    ground_last_ik_update = None
    ground_ik_corrections = 0
    ground_contact_verified = False
    release_ground_since = None
    # V19: adaptive base placement under arm reachability feedback. Wheels are
    # driven by measured position error, not teleported or moved a fixed distance.
    reach_adjust_goal_xy = None
    reach_adjust_started_at = None
    reach_adjust_last_probe = None
    reach_adjust_cumulative_m = 0.0
    reach_adjust_steps = 0
    reach_adjust_score = None
    reach_adjust_from_xy = None
    reach_drift_started_at = None
    return_home_started_at = None
    return_home_in_tolerance_since = None
    return_home_success = False
    original_arm_home_q = None
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
    drop_zone = CircularDropZone(dropoff_xy, args.drop_zone_radius, ball_radius=ball_radius,
                                  release_margin=args.drop_zone_release_margin) if args.place_on_ground else None
    if drop_zone is not None:
        if not args.headless and not args.hide_drop_zone_visual:
            try:
                draw_drop_zone(p, dropoff_xy, drop_zone.radius)
            except Exception as exc:
                print(f"V20_DROP_ZONE_VISUAL_WARNING {exc}")
        print(f"V20_DROP_ZONE centre_xy={dropoff_xy.tolist()} "
              f"radius_m={drop_zone.radius:.3f} diameter_m={2*drop_zone.radius:.3f} "
              f"safe_release_radius_m={drop_zone.release_radius:.3f} "
              f"ball_footprint_landing_radius_m={drop_zone.landing_radius:.3f} "
              "physical_landing_required=True")
    # Ground release is empirically biased by the finger-opening kick. Carry to
    # the precompensated point first, so lowering does not require a large
    # horizontal IK correction while the object is held.
    ground_target_xy = dropoff_xy + np.array(
        [args.ground_release_target_offset_x,
         args.ground_release_target_offset_y], dtype=float)
    navigation_target_xy = ground_target_xy if args.place_on_ground else dropoff_xy
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
    carry_turn_direction = 0.0  # latch direction at the +/-pi branch cut

    # Track last known good state for the final report on crash
    ee_pos = init_ee_pos.copy()
    ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)
    base_pos = np.array(sim.GetBasePosition())
    arm_joint_order_verified = False

    while step < args.max_steps:
        # ---- Sense ----
        try:
            base_pos = np.array(sim.GetBasePosition())
            base_ori = np.array(sim.GetBaseOrientation())
            base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
            q_mes = np.array(sim.GetMotorAngles(0))
            qd_mes = np.array(sim.GetMotorVelocities(0))
            if original_arm_home_q is None:
                original_arm_home_q = q_mes[4:11].copy()
                print(f"V19_HOME_RECORDED arm_q={np.round(original_arm_home_q,3).tolist()}")
            base_lin_vel = np.array(sim.GetBaseLinVelocity())
            base_ang_vel = np.array(sim.GetBaseAngVelocity())
            if not arm_joint_order_verified:
                pybullet_arm_q = np.asarray([p.getJointState(rid, j)[0]
                                             for j in arm_joint_ids], dtype=float)
                controller_arm_q = q_mes[4:11]
                if np.allclose(pybullet_arm_q, controller_arm_q, atol=2e-3, rtol=0.0):
                    arm_joint_order_verified = True
                elif np.allclose(pybullet_arm_q[::-1], controller_arm_q,
                                 atol=2e-3, rtol=0.0):
                    arm_joint_ids.reverse()
                    arm_joint_order_verified = True
                    print("GROUND PLACE: reversed arm-chain order to match MotorCommands")
                else:
                    raise RuntimeError(
                        "Ground placement refused: PyBullet arm-joint order does not match "
                        f"MotorCommands q_mes[4:11]. PyBullet={np.round(pybullet_arm_q,4)}, "
                        f"MotorCommands={np.round(controller_arm_q,4)}")
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
            hold_ee_target = ee_pos if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "reach_adjust", "reach_review", "lowering", "release_open", "return_home", "reverse") else approach_ee_target
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
        if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "reach_adjust", "reach_review", "lowering", "release_open", "return_home", "reverse"):
            # Hold the grasp posture during transport. Expert ground placement
            # follows small Cartesian corrections from actual ball feedback;
            # the V4 learned finger policy still controls all six fingers.
            arm_hold_target = (q_mes[4:11].copy() if grasp_state in
                               ("reach_adjust", "reach_review") else grasp_joint_hold)
            if placement_policy is not None and grasp_state in ("lowering", "release_open"):
                # Policy sampled at 50 Hz. Its joint targets are held between samples.
                tick = int(round(current_time / 0.02))
                if tick != placement_tick:
                    ball_now = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
                    hand_now = np.asarray(p.getLinkState(rid, hand_link, computeForwardKinematics=True)[4])
                    features = placement_features(q_mes, qd_mes, ball_now,
                        p.getBaseVelocity(ball_id)[0], hand_now, dropoff_xy,
                        args.ground_ball_center_z, base_bearing, current_time-ground_lower_started_at)
                    placement_prediction = placement_policy.predict(features)
                    # Counterfactual supervision ONLY when explicitly recording.
                    # The IK-labelled action is NOT fed to motor control: the
                    # following placement_arm_target is always network output.
                    if getattr(args, "record_corrections_npz", None) and grasp_state == "lowering":
                        target_ball = np.asarray(ground_lower_target_ball, dtype=float)
                        correction = np.clip(target_ball - ball_now, -0.012, 0.012)
                        # Keep descending gradually; never ask the teacher to
                        # pull the ball sideways in one abrupt reference step.
                        if ball_now[2] > args.ground_ball_center_z + 0.012:
                            correction[2] = min(correction[2], -0.006)
                        desired_hand = hand_now + correction
                        hand_quat = p.getLinkState(rid,hand_link,computeForwardKinematics=True)[5]
                        try:
                            ref_q = solve_arm_ik(p,rid,hand_link,arm_joint_ids,desired_hand,hand_quat)
                            ref_delta = np.clip(ref_q-q_mes[4:11], -0.12, 0.12)
                            # The training oracle supplies one-step feedback
                            # labels. It cannot drive the arm in this rollout.
                            # TRAINING ORACLE only: broaden release examples to observed
                            # near-centre, moderate-height states. At runtime the
                            # network alone decides release; this oracle NEVER acts.
                            close_to_target = (np.linalg.norm(np.asarray(dropoff_xy)-ball_now[:2]) < 0.10
                                  and 0.32 <= ball_now[2] <= 0.60
                                  and np.linalg.norm(p.getBaseVelocity(ball_id)[0]) < 0.18)
                            finger_label = (np.zeros(6) if close_to_target
                                else np.asarray(target_finger_angles).copy())
                            correction_obs.append(features.copy())
                            correction_actions.append(np.r_[ref_delta, finger_label,
                                float(close_to_target)])
                        except Exception as correction_exc:
                            if len(correction_obs) == 0:
                                print("V20_CORRECTION_LABEL_FAILED", correction_exc, flush=True)
                    # Network predicts the direction and amount; this only caps
                    # joint setpoint movement per 20-ms policy update.
                    placement_arm_target = q_mes[4:11] + np.clip(
                        placement_prediction[:7], -args.placement_joint_step, args.placement_joint_step)
                    limits = np.asarray([p.getJointInfo(rid, j)[8:10] for j in arm_joint_ids])
                    bounded = limits[:, 0] < limits[:, 1]
                    placement_arm_target[bounded] = np.clip(placement_arm_target[bounded],
                        limits[bounded, 0], limits[bounded, 1])
                    placement_tick = tick
                arm_hold_target = placement_arm_target
            elif (grasp_state == "lowering" and placement_policy is None
                  and ground_planner is not None):
                # Replan at 20 Hz from PHYSICAL ball feedback, not an open-loop
                # joint-space start/end interpolation.  Descend only while the
                # ball is horizontally aligned; cap every Cartesian IK step.
                if (ground_last_servo_update is None or
                    current_time-ground_last_servo_update >= args.ground_ik_interval):
                    ball_now = np.asarray(p.getBasePositionAndOrientation(ball_id)[0], dtype=float)
                    ball_velocity = np.asarray(p.getBaseVelocity(ball_id)[0], dtype=float)
                    hand_state = p.getLinkState(rid, hand_link, computeForwardKinematics=True)
                    hand_now = np.asarray(hand_state[4], dtype=float)
                    if drop_zone is not None:
                        # Follow the occupied reachable part of the circular
                        # region. Do not fight lateral drift while inside it;
                        # only correct horizontally near/outside the boundary.
                        xy_target = drop_zone.target_for_measured_ball(ball_now[:2])
                        ground_lower_target_ball[:2] = xy_target
                        ground_planner.target[:2] = xy_target
                    desired_ball, ball_correction, descending = ground_planner.update(
                        current_time, ball_now, ball_velocity)
                    desired_hand = hand_now + ball_correction
                    try:
                        candidate_q = solve_arm_ik(
                            p, rid, hand_link, arm_joint_ids,
                            desired_hand, ground_hand_orientation)
                        # The physical joint servo has a 0.2 rad/s cap. Restrict
                        # changes in its commanded setpoint as well, avoiding a
                        # large wrist sweep that would tip the learned grasp.
                        max_joint_step = args.ground_joint_target_step
                        ground_lower_target_q = q_mes[4:11] + np.clip(
                            candidate_q-q_mes[4:11], -max_joint_step, max_joint_step)
                        ground_last_servo_update = current_time
                        ground_ik_corrections += 1
                        if ground_ik_corrections % 20 == 1:
                            print(f"V18_CARTESIAN_LOWER t={current_time:.2f}s "
                                  f"ball={np.round(ball_now,3).tolist()} "
                                  f"desired={np.round(desired_ball,3).tolist()} "
                                  f"correction={np.round(ball_correction,3).tolist()} "
                                  f"descending={descending} "
                                  f"xy_error_m={np.linalg.norm(ground_target_xy-ball_now[:2]):.3f}")
                    except Exception as exc:
                        carry_fail_reason = f"Cartesian lowering IK failed; ball kept gripped: {exc}"
                        print(f"V18_CARTESIAN_IK_FAIL {carry_fail_reason}")
                        ground_lower_target_q = q_mes[4:11].copy()
                        ground_last_servo_update = current_time
                arm_hold_target = ground_lower_target_q
            elif grasp_state == "return_home":
                # Only after ball detachment and verified stable ground support.
                # In BC mode, the neural policy supplies every homeward setpoint.
                # Home configuration is a GOAL observation, not a scripted action.
                if return_home_policy is not None:
                    return_home_delta = return_home_policy.predict_delta(
                        q_mes[4:11], qd_mes[4:11], original_arm_home_q)
                    arm_hold_target = q_mes[4:11] + return_home_delta
                    if step % 500 == 0:
                        print(f"V21_LEARNED_RETURN_ACTION t={current_time:.2f}s "
                              f"network_delta={np.round(return_home_delta,3).tolist()} "
                              "teacher_override=False", flush=True)
                else:
                    arm_hold_target = original_arm_home_q
                if args.return_home_record_npz:
                    # Explicit expert-only collection; never mix student actions
                    # into an expert-labelled dataset.
                    if return_home_policy is None and step % 5 == 0:
                        return_home_demo_obs.append(np.concatenate([
                            original_arm_home_q - q_mes[4:11], qd_mes[4:11]]))
                        return_home_demo_actions.append(np.clip(
                            original_arm_home_q - q_mes[4:11], -0.15, 0.15))
            elif (grasp_state == "release_open" and args.place_on_ground
                  and ground_lower_target_q is not None):
                # Keep the wrist fixed at the verified placement pose until
                # the fingers have cleared the ball. Returning to the original
                # grasp joints here lifts the arm while contacts still exist,
                # kicking the released ball sideways.
                arm_hold_target = ground_lower_target_q
            if grasp_state in ("lowering", "release_open", "return_home"):
                # Identical action bounds during demonstration and inference.
                arm_hold_target = q_mes[4:11] + np.clip(arm_hold_target-q_mes[4:11], -0.15, 0.15)
                if args.place_on_ground:
                    placement_servo_target = arm_hold_target.copy()
            if (args.place_on_ground and grasp_state == "return_home") or (
                    placement_policy is not None and grasp_state in ("lowering", "release_open")):
                # The simulator's arm motors use POSITION_CONTROL via the callback.
                # Do NOT mix an unrelated learned-reaching torque and a second PD
                # torque with that actuator. The policy still supplies q targets.
                torque_arm = np.zeros(7, dtype=float)
                if step % 1000 == 0:
                    print(f"V20_LEARNED_ACTUATOR t={current_time:.2f}s "
                          f"max_joint_error_rad={np.max(np.abs(arm_hold_target-q_mes[4:11])):.4f} "
                          f"mean_joint_speed={np.mean(np.abs(qd_mes[4:11])):.4f} "
                          f"force_limit_Nm={args.placement_servo_force:.1f} "
                          f"position_gain={args.placement_servo_position_gain:.2f} "
                          f"arm_setpoints={'LEARNED' if grasp_state != 'return_home' or return_home_policy is not None else 'EXPERT'} "
                          f"network_source={grasp_state != 'return_home' or return_home_policy is not None}", flush=True)
            else:
                torque_arm = np.clip(
                    torque_arm + 18.0 * (arm_hold_target - q_mes[4:11]) - 2.5 * qd_mes[4:11],
                    -20.0, 20.0)
        # Finger targets come ONLY from the trained sensor policy. No learned
        # wheel-reverse is tested here: the mobile base is held stationary.
        if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "reach_adjust", "reach_review", "lowering"):
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
            if step % 1000 == 0:
                print(f"ML_INFERENCE t={current_time:.2f}s "
                      f"finger_target_rad={np.round(action_ml[:6], 3).tolist()} "
                      f"finger_measured_rad={np.round(q_mes[11:17], 3).tolist()}")
            # V4.1 stationary-base hold is retained until transport is authorised.
            if grasp_state in ("closing", "gravity_hold"):
                wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - wheel_lock_pos), -5.0, 5.0)

        # -------- V8 expert: steer the CARRIED BALL to its world-XY destination --------
        # Grasp remains under the trained V4 neural model until release.  No
        # scripted finger squeeze or virtual ball attachment is used.
        if grasp_state == "carry":
            ball_xy_for_control = np.asarray(p.getBasePositionAndOrientation(ball_id)[0][:2], dtype=float)
            to_goal = navigation_target_xy - ball_xy_for_control
            remaining = float(np.linalg.norm(to_goal))
            desired_heading = float(np.arctan2(to_goal[1], to_goal[0])) if remaining > 1e-6 else float(base_bearing)
            bearing_error = float(np.arctan2(np.sin(desired_heading - base_bearing),
                                             np.cos(desired_heading - base_bearing)))
            # At exactly 180 degrees, tiny simulation noise makes wrapped
            # atan2 alternate between +pi and -pi. Latch one turn direction
            # until the robot has clearly left that branch cut.
            if abs(bearing_error) > 2.75 and carry_turn_direction == 0.0:
                carry_turn_direction = float(np.copysign(1.0, bearing_error))
            elif abs(bearing_error) < 2.35:
                carry_turn_direction = 0.0
            if carry_turn_direction != 0.0 and abs(bearing_error) > 2.35:
                bearing_error = carry_turn_direction * abs(bearing_error)
            origin_to_goal = navigation_target_xy - carry_ball_origin[:2]
            path_unit = origin_to_goal / max(np.linalg.norm(origin_to_goal), 1e-8)
            pos_from_origin = ball_xy_for_control - carry_ball_origin[:2]
            lateral = float(path_unit[0]*pos_from_origin[1] - path_unit[1]*pos_from_origin[0])
            # Slow down when turning to protect the physical two-/three-finger grip.
            # When the ball is near the goal, wheel control transitions to a stop.
            if remaining <= args.carry_goal_tolerance:
                v_cmd = 0.0
                w_cmd = 0.0
            else:
                yaw_scale = max(0.0, float(np.cos(bearing_error)))
                if abs(bearing_error) > 0.55:
                    yaw_scale *= 0.20
                v_cmd = float(np.clip(args.expert_kp * remaining * yaw_scale,
                                      0.0, args.carry_speed_max))
                w_cmd = float(np.clip(args.expert_heading_kp * bearing_error,
                                      -args.carry_yaw_max, args.carry_yaw_max))
            if carry_policy is not None:
                fwd_vel = float(np.dot(base_lin_vel[:2], [np.cos(base_bearing), np.sin(base_bearing)]))
                proposal = carry_policy.predict(np.array([remaining, lateral, bearing_error,
                                                          fwd_vel, float(base_ang_vel[2])]))
                v_cmd = float(np.clip(proposal[0], 0.0, args.carry_speed_max))
                w_cmd = float(np.clip(proposal[1], -args.carry_yaw_max, args.carry_yaw_max))
                if remaining <= args.carry_goal_tolerance:
                    v_cmd = w_cmd = 0.0  # arrival stop guard
            if getattr(args, 'local_drop', False):
                # No target transfer: keep wheels immobilized. This is NOT an
                # expert motion, merely disabling the carry skill entirely.
                v_cmd = 0.0
                w_cmd = 0.0
            if step % 5 == 0:
                fwd_vel = float(np.dot(base_lin_vel[:2],
                                       [np.cos(base_bearing),np.sin(base_bearing)]))
                carry_obs.append([remaining, lateral, bearing_error,
                                  fwd_vel, float(base_ang_vel[2])])
                carry_actions.append([v_cmd, w_cmd])
                carry_times.append(current_time-carry_started_at)
            rw = (v_cmd + w_cmd * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
            lw = (v_cmd - w_cmd * WHEEL_BASE_WIDTH / 2.0) / WHEEL_RADIUS
            wheel_cmds = np.array([rw, lw, lw, rw], dtype=float)
            if step % 1000 == 0:
                print(f"V8_EXPERT_NAV t={current_time:.2f}s "
                      f"ball_xy={np.round(ball_xy_for_control,3).tolist()} "
                      f"target_xy={np.round(navigation_target_xy,3).tolist()} "
                      f"remaining_m={remaining:.3f} heading_error_rad={bearing_error:.3f} "
                      f"v={v_cmd:.3f} w={w_cmd:.3f}")

        if grasp_state in ("carry_settle", "reach_adjust", "reach_review", "lowering", "release_open", "return_home"):
            # The base must be stationary while checking the hold and releasing.
            wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - carry_stop_wheel_q), -5.0, 5.0)

        if grasp_state == "reach_adjust":
            # Physical wheel motion with position feedback; no preprogrammed
            # reverse manoeuvre or forced base pose. Keep the arm at its held
            # posture while reaching a base stance that scored better in IK.
            heading_vec = np.array([np.cos(base_bearing), np.sin(base_bearing)])
            along_error = float(np.dot(reach_adjust_goal_xy-base_pos[:2], heading_vec))
            wheel_rate = wheel_command_for_displacement(
                along_error, speed_cap=args.reach_base_speed_max)
            wheel_cmds = np.full(4, wheel_rate, dtype=float)
            if step % 1000 == 0:
                print(f"V19_BASE_ADJUST t={current_time:.2f}s "
                      f"base_xy={np.round(base_pos[:2],3).tolist()} "
                      f"remaining_along_m={along_error:.3f} wheels_rad_s={wheel_rate:.3f} "
                      f"ball={np.round(p.getBasePositionAndOrientation(ball_id)[0],3).tolist()}")
            if (abs(along_error) <= 0.008 and
                    np.linalg.norm(base_lin_vel[:2]) <= 0.015 and
                    current_time - reach_adjust_started_at >= 0.3):
                reach_adjust_cumulative_m += float(np.linalg.norm(base_pos[:2]-reach_adjust_from_xy))
                carry_stop_wheel_q = q_mes[:4].copy()
                wheel_cmds = np.zeros(4)
                grasp_state = "reach_review"
                reach_adjust_last_probe = None
                print(f"V19_BASE_CORRECTION_DONE base={np.round(base_pos[:2],3).tolist()} "
                      f"total_move_m={reach_adjust_cumulative_m:.3f}")
            elif current_time - reach_adjust_started_at >= args.reach_base_adjust_timeout:
                carry_fail_reason = "Feedback base re-position timeout; retained ML grasp"
                print(f"V19_BASE_ADJUST_ABORT {carry_fail_reason}")
                break

        if grasp_state in ("reach_review", "return_home"):
            wheel_cmds = np.clip(-50.0*(q_mes[0:4]-carry_stop_wheel_q), -5.0, 5.0)

        if grasp_state in ("release_open", "return_home"):
            # Student-generated opening action is latched at the learned decision.
            # No scripted zero/open command in student mode.
            if placement_policy is not None and grasp_state == "release_open":
                if learned_release_finger_targets is None:
                    raise RuntimeError("Learned release targets missing; refusing expert fallback")
                target_finger_angles[:] = learned_release_finger_targets
            elif placement_policy is None:
                target_finger_angles[:] = expert_open_fingers
            # V8.2: ONLY open the fingers; no expert backoff/extra motion.
            # The 6 joint targets are applied through bounded_finger_callback
            # on every simulation step and cannot be overwritten by V4 ML.
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
        if grasp_state in ("closing", "gravity_hold", "carry", "carry_settle", "reach_adjust", "reach_review", "lowering", "release_open"):
            current_groups = [
                group for group in range(3)
                if any(cp[3] in finger_joint_ids[2*group:2*group+2]
                       for cp in finger_contacts)
            ]
            max_simultaneous_groups = max(max_simultaneous_groups, len(current_groups))
            contacted_finger_links.update(int(cp[3]) for cp in finger_contacts)

            # Signed collision distances: + separation, 0 touching, - penetration.
            if step % 100 == 0:
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

            if step % 1000 == 0:
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
                        carry_goal_xy = navigation_target_xy.copy()  # target is for BALL, not robot base
                        carry_ball_origin = ball_xyz.copy()
                        carry_ball_to_pocket_start = ball_xyz - pocket_world()
                        dropoff_initial_range = float(np.linalg.norm(navigation_target_xy - ball_xyz[:2]))
                        grasp_state = "carry"
                        print(f"V8_START_EXPERT_NAV t={current_time:.2f}s "
                              f"ball_from_xy={np.round(carry_ball_origin[:2],3).tolist()} "
                              f"dropoff_ball_xy={np.round(navigation_target_xy,3).tolist()} "
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
            dropoff_error = float(np.linalg.norm(navigation_target_xy-ball_xyz[:2]))
            ball_speed_xy = float(np.linalg.norm(p.getBaseVelocity(ball_id)[0][:2]))
            if step % 1000 == 0:
                print(f"V8_PHYSICAL_CARRY t={current_time:.2f}s state={grasp_state} "
                      f"ball_xy={np.round(ball_xyz[:2],3).tolist()} "
                      f"dropoff_error_m={dropoff_error:.3f} "
                      f"groups={current_groups} force_N={finger_normal_force:.2f} "
                      f"pocket_slip_mm={carry_drift*1000:.1f} drop_mm={carry_drop*1000:.1f}")
            # Arrival uses the physical ball's measured world XY, not base XY.
            # Ground mode first lowers the still-grasped ball. The legacy mode
            # keeps the original immediate-open behavior.
            if grasp_state == "carry" and (
                    (drop_zone is not None and drop_zone.distance(ball_xyz[:2]) <=
                     max(0.02, drop_zone.release_radius - args.drop_zone_approach_margin))
                    or (drop_zone is None and dropoff_error <= args.carry_goal_tolerance)):
                # Stop the BASE early upon entering a wide landing region,
                # not at the centre. The arm chooses its own measured XY.
                if drop_zone is not None:
                    print(f"V20_ZONE_ARRIVED t={current_time:.2f}s "
                          f"ball_xy={np.round(ball_xyz[:2],3).tolist()} "
                          f"centre_distance_m={dropoff_error:.3f} "
                          f"zone_radius_m={drop_zone.radius:.3f}")
                dropoff_target_reached = True
                carry_end_time = current_time
                carry_stop_wheel_q = q_mes[0:4].copy()
                carry_success = bool(len(current_groups) >= 2
                                     and carry_drift <= args.release_max_grip_offset
                                     and carry_drop <= args.release_max_grip_drop)
                wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - carry_stop_wheel_q), -5.0, 5.0)
                if args.place_on_ground:
                    link_state = p.getLinkState(rid, hand_link, computeForwardKinematics=True)
                    hand_position = np.asarray(link_state[4], dtype=float)
                    hand_orientation = np.asarray(link_state[5], dtype=float)
                    # Preserve the measured world-space hand-to-ball offset. This
                    # avoids a bad IK target from frame-convention ambiguity and
                    # keeps the ball centred over the requested drop-off XY.
                    ball_offset_world = ball_xyz - hand_position
                    # V20: Do not require the ball at the exact circle centre.
                    # Prefer its ACTUAL measured XY to avoid lateral IK strain.
                    selected_xy = (drop_zone.target_for_measured_ball(ball_xyz[:2])
                                   if drop_zone is not None else ground_target_xy)
                    ground_lower_target_ball = np.array(
                        [selected_xy[0], selected_xy[1],
                         args.ground_ball_center_z], dtype=float)
                    desired_hand_position = ground_lower_target_ball - ball_offset_world
                    # Start the expert Cartesian servo at the ACTUAL held pose.
                    # In v17, interpolating seven joint angles to a distant
                    # low-Z IK goal created ~19 cm of ball XY error and broke
                    # contact even while the ball was near the requested height.
                    ground_lower_start_q = q_mes[4:11].copy()
                    ground_lower_target_q = q_mes[4:11].copy()
                    ground_lower_started_at = current_time
                    ground_last_servo_update = None
                    ground_last_ik_update = current_time
                    ground_ik_corrections = 0
                    ground_hand_orientation = hand_orientation.copy()
                    if placement_policy is None:
                        ground_planner = BallCartesianLoweringPlanner(
                            start_ball_xyz=ball_xyz,
                            target_ball_xyz=ground_lower_target_ball,
                            vertical_speed=args.ground_cartesian_speed,
                            xy_pause_error=args.ground_xy_pause_error,
                            max_xy_step=args.ground_cartesian_xy_step,
                            max_z_step=args.ground_cartesian_z_step)
                    # V20 ground-area mode: first try a physically measured,
                    # already-reachable XY rather than repeatedly probing IK at
                    # a single unreachable centre point. Wheels stay locked.
                    grasp_state = "lowering" if drop_zone is not None else (
                        "reach_review" if placement_policy is None else "lowering")
                    reach_adjust_from_xy = base_pos[:2].copy()
                    if drop_zone is not None:
                        print(f"V20_ZONE_LOWERING_START t={current_time:.2f}s "
                              f"selected_xy={np.round(selected_xy,3).tolist()} "
                              f"ball_z={ball_xyz[2]:.3f} "
                              f"release_height_m={args.ground_ball_center_z:.3f}")
                    print(f"V9_GROUND_PLACE_START t={current_time:.2f}s "
                          f"ball_xy={np.round(ball_xyz[:2],3).tolist()} "
                          f"goal_xy={np.round(dropoff_xy,3).tolist()} "
                          f"release_xy_target={np.round(ground_lower_target_ball[:2],3).tolist()} "
                          f"ik_hand_target={np.round(desired_hand_position,3).tolist()} "
                          f"release_center_z={args.ground_ball_center_z:.3f}m "
                          f"gate_xy_m={args.ground_pre_release_xy_tolerance:.3f} "
                          f"gate_height_m={args.ground_place_height_tolerance:.3f} "
                          f"gate_speed_mps={args.ground_place_speed_max:.3f} "
                          f"settle_s={args.ground_place_settle_seconds:.2f} "
                          f"release_finger_force_N={args.ground_release_finger_force:.2f} "
                          f"carry_physically_verified={carry_success} "
                          f"controller={'V18 Cartesian feedback' if ground_planner is not None else 'BC'}")
                else:
                    grasp_state = "release_open"
                    release_started_at = current_time
                    release_ball_start = ball_xyz.copy()
                    target_finger_angles[:] = expert_open_fingers
                    print(f"V8_2_TARGET_REACHED_OPEN_NOW t={current_time:.2f}s "
                          f"ball_xy={np.round(ball_xyz[:2],3).tolist()} "
                          f"goal_xy={np.round(dropoff_xy,3).tolist()} "
                          f"xy_error_m={dropoff_error:.3f} "
                          f"finger_open_targets={np.round(expert_open_fingers,3).tolist()}")
                    print(f"V8_EXPERT_RELEASE_START t={current_time:.2f}s "
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

        if grasp_state == "reach_adjust" and (len(current_groups) < 2 or
                        np.linalg.norm((ball_xyz-pocket_world())-
                                       carry_ball_to_pocket_start) > args.release_max_grip_offset):
            carry_fail_reason = "Physical grasp degraded during base reach correction"
            print(f"V19_BASE_ADJUST_ABORT {carry_fail_reason}")
            break

        if grasp_state == "reach_review":
            if len(current_groups) < 2:
                carry_fail_reason = "Grasp lost before base reachability review"
                print(f"V19_REACH_CHECK_ABORT {carry_fail_reason}")
                break
            # Probe only once per settle; candidate poses are restored before
            # continuing the real dynamics. Prefer measured reachability to
            # a fixed 'back up 5cm' motion.
            if reach_adjust_last_probe is None:
                try:
                    hand_now = np.asarray(p.getLinkState(
                        rid, hand_link, computeForwardKinematics=True)[4], dtype=float)
                    ball_minus_hand = ball_xyz - hand_now
                    reach_samples = probe_ground_reachability(
                        p, rid, hand_link, arm_joint_ids, ball_xyz, ball_minus_hand,
                        ground_lower_target_ball, ground_hand_orientation,
                        base_bearing, offsets=(-args.reach_probe_step_m, 0.0,
                                               args.reach_probe_step_m))
                    correction, baseline_score, best_score = select_base_adjustment(
                        reach_samples, feasible_residual=args.reach_residual_max_m)
                    reach_adjust_score = baseline_score
                    reach_adjust_last_probe = current_time
                    print(f"V19_REACH_CHECK t={current_time:.2f}s "
                          f"offset_errors={reach_samples} "
                          f"baseline_m={baseline_score:.3f} best_m={best_score:.3f} "
                          f"selected_base_shift_m={correction:+.3f}")
                except Exception as exc:
                    carry_fail_reason = f"Arm workspace probe failed safely: {exc}"
                    print(f"V19_REACH_CHECK_ABORT {carry_fail_reason}")
                    break
                if (correction != 0.0 and
                        reach_adjust_cumulative_m+abs(correction) <= args.reach_max_base_shift_m and
                        reach_adjust_steps < args.reach_max_adjustments):
                    heading_vec = np.array([np.cos(base_bearing), np.sin(base_bearing)])
                    reach_adjust_goal_xy = base_pos[:2]+correction*heading_vec
                    reach_adjust_from_xy = base_pos[:2].copy()
                    reach_adjust_started_at = current_time
                    reach_adjust_steps += 1
                    grasp_state = "reach_adjust"
                    print(f"V19_BASE_CORRECTION_START t={current_time:.2f}s "
                          f"direction={'back' if correction < 0 else 'forward'} "
                          f"planned_shift_m={correction:+.3f} "
                          f"score_before_m={baseline_score:.3f} after_m={best_score:.3f}")
                else:
                    # If no remaining candidate improves the IK residual,
                    # use the current base and guard every subsequent servo
                    # command. Do not initiate an endless fixed-wheel search.
                    if baseline_score > args.reach_max_unreachable_residual_m:
                        carry_fail_reason = (
                            f"Near-ground arm workspace unreachable after feedback base review: "
                            f"predicted_FK_error={baseline_score:.3f}m")
                        print(f"V19_REACH_CHECK_ABORT {carry_fail_reason}")
                        break
                    grasp_state = "lowering"
                    ground_lower_started_at = current_time
                    ground_last_servo_update = None
                    ground_lower_reached_at = None
                    ground_planner = BallCartesianLoweringPlanner(
                        start_ball_xyz=ball_xyz,
                        target_ball_xyz=ground_lower_target_ball,
                        vertical_speed=args.ground_cartesian_speed,
                        xy_pause_error=args.ground_xy_pause_error,
                        max_xy_step=args.ground_cartesian_xy_step,
                        max_z_step=args.ground_cartesian_z_step)
                    print(f"V19_LOWERING_START t={current_time:.2f}s "
                          f"base_xy={np.round(base_pos[:2],3).tolist()} "
                          f"predicted_ik_residual_m={baseline_score:.3f}")

        if grasp_state == "lowering":
            if carry_fail_reason:
                print(f"V9_GROUND_PLACE_ABORT {carry_fail_reason}; fingers remain under learned control")
                break
            target_xy_error = float(np.linalg.norm(ground_lower_target_ball[:2]-ball_xyz[:2]))
            goal_xy_error = float(np.linalg.norm(dropoff_xy-ball_xyz[:2]))
            height_error = abs(float(ball_xyz[2])-args.ground_ball_center_z)
            ball_speed = float(np.linalg.norm(p.getBaseVelocity(ball_id)[0]))
            lower_hold_drift = float(np.linalg.norm(
                (ball_xyz-pocket_world())-carry_ball_to_pocket_start))
            # World-Z movement is intentional while lowering. Verify retention
            # from finger contacts and ball-to-pocket drift, not carry-phase drop.
            if (len(current_groups) < 2 or
                    lower_hold_drift > args.release_max_grip_offset):
                carry_fail_reason = (
                    f"Grip lost during ground lowering; groups={current_groups}, "
                    f"pocket_slip={lower_hold_drift:.3f}m")
                print(f"V9_GROUND_PLACE_ABORT {carry_fail_reason}; fingers remain under learned control")
                break
            if (drop_zone is None and placement_policy is None and
                    target_xy_error > args.reach_replan_xy_error_m and
                    current_time-ground_lower_started_at > 0.5):
                if ball_xyz[2] > args.reach_reposition_min_ball_z and \
                   reach_adjust_steps < args.reach_max_adjustments:
                    if reach_drift_started_at is None:
                        reach_drift_started_at = current_time
                    elif current_time-reach_drift_started_at >= 0.30:
                        grasp_state = "reach_review"
                        ground_planner = None
                        ground_lower_target_q = q_mes[4:11].copy()
                        reach_adjust_last_probe = None
                        print(f"V19_LOWERING_REACH_REVIEW t={current_time:.2f}s "
                              f"ball_xyz={np.round(ball_xyz,3).tolist()} "
                              f"xy_error_m={target_xy_error:.3f}; "
                              "holding ball and checking base reachability")
                else:
                    carry_fail_reason = (
                        f"Placement XY drift beyond control envelope at low ball height: "
                        f"xy_error={target_xy_error:.3f}m height={ball_xyz[2]:.3f}m")
                    print(f"V19_LOWERING_ABORT {carry_fail_reason}")
                    break
            else:
                reach_drift_started_at = None
            if grasp_state != "lowering":
                # Reposition at a safe height, never attempt low-Z wheel sweeps.
                continue_lowering_checks = False
            else:
                continue_lowering_checks = True
            placement_xy_ok = (drop_zone.can_release(ball_xyz[:2])
                               if drop_zone is not None else
                               (target_xy_error <= args.ground_pre_release_xy_tolerance
                                and goal_xy_error <= args.ground_pre_release_xy_tolerance))
            on_placement_pose = (placement_xy_ok
                                 and height_error <= args.ground_place_height_tolerance
                                 and ball_speed <= args.ground_place_speed_max) if continue_lowering_checks else False
            # Correct residual IK / joint-tracking error using the measured ball
            # position. Keep the deadband below the pre-release XY tolerance.
            position_near = (placement_xy_ok
                             and height_error <= args.ground_place_height_tolerance)
            if (drop_zone is None and placement_policy is None and ground_planner is None
                    and not position_near and ball_speed < 0.12
                    and current_time-ground_lower_started_at >= args.ground_lower_seconds
                    and ground_ik_corrections < 64
                    and ground_last_ik_update is not None
                    and current_time-ground_last_ik_update >= 0.5):
                # Correct measured XYZ pose, not just XY. The prior feedback
                # stopped when XY passed even though release height was still off.
                position_correction = ground_lower_target_ball - ball_xyz
                correction_norm = float(np.linalg.norm(position_correction))
                # Cap large errors; never amplify small errors to an 8 cm jump.
                if correction_norm > 0.025:
                    position_correction *= 0.025 / correction_norm
                link_state = p.getLinkState(rid, hand_link, computeForwardKinematics=True)
                current_hand_position = np.asarray(link_state[4], dtype=float)
                current_hand_orientation = np.asarray(link_state[5], dtype=float)
                corrected_hand_position = current_hand_position + position_correction
                try:
                    corrected_q = solve_arm_ik(
                        p, rid, hand_link, arm_joint_ids,
                        corrected_hand_position, current_hand_orientation)
                    # Bound each replan so correction targets change smoothly.
                    ground_lower_target_q = ground_lower_target_q + np.clip(
                        corrected_q-ground_lower_target_q, -0.15, 0.15)
                    ground_ik_corrections += 1
                    ground_last_ik_update = current_time
                    print(f"V9_GROUND_IK_CORRECTION n={ground_ik_corrections} "
                          f"ball_xyz_correction={np.round(position_correction,3).tolist()} "
                          f"new_hand_target={np.round(corrected_hand_position,3).tolist()}")
                except Exception as exc:
                    carry_fail_reason = f"Ground-place XYZ correction IK failed; kept gripper closed: {exc}"
                    print(f"V9_GROUND_PLACE_IK_FAIL {carry_fail_reason}")
                    break
            # In student mode a neural score triggers release, and the six
            # commanded finger angles come from the same network.  Physical
            # position/speed checks are SAFETY LIMITS, not scripted release
            # actions. Teacher mode keeps V20's original settling behaviour.
            release_decision = (placement_policy is None or
                                (placement_prediction is not None and placement_prediction[13] >= 0.5))
            # Teacher uses its V20 height/settle gates. Student is NOT
            # triggered by the expert's hard-coded release height: the neural
            # score chooses WHEN; only in-circle contact/speed guards remain.
            student_safe_release = (placement_xy_ok and len(current_groups) >= 2
                                    and ball_speed <= args.ground_place_speed_max)
            # Diagnostics only: observe the student decision and the physical
            # safety constraints. This does NOT command any arm or finger action.
            if placement_policy is not None and step % 500 == 0:
                print(f"V20_RELEASE_GATE t={current_time:.2f}s "
                      f"learned_probability={float(placement_prediction[13]) if placement_prediction is not None else float('nan'):.3f} "
                      f"xy_inside={bool(placement_xy_ok)} "
                      f"centre_error_m={goal_xy_error:.3f} "
                      f"ball_z_m={float(ball_xyz[2]):.3f} "
                      f"ball_speed_mps={ball_speed:.3f} "
                      f"speed_allowed_mps={args.ground_place_speed_max:.3f} "
                      f"contact_groups={len(current_groups)} "
                      f"release_allowed={bool(student_safe_release and release_decision)} "
                      "motor_actions=UNCHANGED", flush=True)
            if (on_placement_pose if placement_policy is None else student_safe_release) and release_decision:
                if ground_lower_reached_at is None:
                    ground_lower_reached_at = current_time
                    if placement_policy is None:
                        ground_lower_target_q = q_mes[4:11].copy()
                if (placement_policy is not None or
                    current_time-ground_lower_reached_at >= args.ground_place_settle_seconds):
                    grasp_state = "release_open"
                    release_started_at = current_time
                    release_ball_start = ball_xyz.copy()
                    if placement_policy is not None:
                        learned_release_finger_targets = np.clip(placement_prediction[7:13], 0.0, 2.0).copy()
                        target_finger_angles[:] = learned_release_finger_targets
                        print(f"V20_RELEASE_STUDENT_ACTION t={current_time:.2f}s "
                              f"network_release_score={placement_prediction[13]:.3f} "
                              f"network_finger_targets={np.round(learned_release_finger_targets,3).tolist()} "
                              f"teacher_override=False")
                    else:
                        target_finger_angles[:] = expert_open_fingers
                    print(f"V20_ZONE_RELEASE_READY centre_error_m={goal_xy_error:.3f} "
                          f"zone_release_radius_m={drop_zone.release_radius:.3f} "
                          f"valid={drop_zone.can_release(ball_xyz[:2])}"
                          if drop_zone is not None else "V20_POINT_RELEASE_READY")
                    print(f"{'V20_STUDENT_RELEASE_SAFETY_VERIFIED' if placement_policy is not None else 'V9_GROUND_PLACE_POSE_VERIFIED'} t={current_time:.2f}s "
                          f"ball_xyz={np.round(ball_xyz,3).tolist()} "
                          f"xy_error_m={target_xy_error:.3f} "
                          f"goal_xy_error_m={goal_xy_error:.3f} "
                          f"height_error_m={height_error:.3f} "
                          f"speed_mps={ball_speed:.3f}")
                    print(f"V9_GROUND_RELEASE_START t={current_time:.2f}s "
                          f"open_targets={np.round(target_finger_angles,3).tolist()} source={args.placement_controller}")
            else:
                ground_lower_reached_at = None
            if step % 1000 == 0:
                print(f"V9_GROUND_LOWERING t={current_time:.2f}s "
                      f"ball_xyz={np.round(ball_xyz,3).tolist()} "
                      f"release_target_xy_error_m={target_xy_error:.3f} "
                      f"goal_xy_error_m={goal_xy_error:.3f} height_error_m={height_error:.3f} "
                      f"speed_mps={ball_speed:.3f}")
            if (grasp_state == "lowering" and
                    current_time-ground_lower_started_at > args.ground_lower_timeout):
                carry_fail_reason = (
                    f"Ground placement timed out; gripper kept closed; "
                    f"xy_error={target_xy_error:.3f}m height_error={height_error:.3f}m")
                print(f"V9_GROUND_PLACE_FAIL {carry_fail_reason}")
                break

        if grasp_state == "release_open":
            if step % 5 == 0:
                observed_ball = np.asarray(p.getBasePositionAndOrientation(ball_id)[0])
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
            non_robot_support_contacts = [cp for cp in p.getContactPoints(bodyA=ball_id)
                                          if cp[2] != rid]
            if non_robot_support_contacts:
                if release_ground_since is None:
                    release_ground_since = current_time
            else:
                release_ground_since = None
            if robot_release_contact_count == 0:
                if release_no_contact_since is None:
                    release_no_contact_since = current_time
            else:
                release_no_contact_since = None
            release_drop=max(0.0, float(release_ball_start[2]-ball_xyz[2]))
            release_max_drop=max(release_max_drop,release_drop)
            release_final_distance=float(np.linalg.norm(dropoff_xy-ball_xyz[:2]))
            if step%1000==0:
                release_ball_velocity = np.asarray(p.getBaseVelocity(ball_id)[0], dtype=float)
                print(f"V8_RELEASE_MONITOR t={current_time:.2f}s "
                      f"open_elapsed={current_time-release_started_at:.2f}s "
                      f"finger_contacts={release_contact_count} "
                      f"drop_m={release_drop:.3f} ball_xy_error_m={release_final_distance:.3f} "
                      f"ball_speed_mps={np.linalg.norm(release_ball_velocity):.3f} "
                      f"ball_velocity={np.round(release_ball_velocity,3).tolist()} "
                      f"ball_xyz={np.round(ball_xyz,3).tolist()}")
            ground_contact_stable = bool(
                release_ground_since is not None and
                current_time-release_ground_since >= args.ground_contact_seconds)
            required_drop = args.ground_min_release_drop if args.place_on_ground else args.release_min_drop
            release_physics_verified = (ground_contact_stable if args.place_on_ground
                                        else release_drop >= required_drop)
            if (release_no_contact_since is not None
                and current_time-release_no_contact_since >= args.release_no_contact_seconds
                and release_physics_verified
                and current_time-release_started_at >= args.release_min_seconds):
                release_success=True
                ground_contact_verified = bool(ground_contact_stable)
                placement_tolerance = (args.ground_place_xy_tolerance if args.place_on_ground
                                       else args.release_final_tolerance)
                release_target_success=bool(
                    drop_zone.landed_inside(ball_xyz[:2]) if drop_zone is not None
                    else release_final_distance <= placement_tolerance)
                if drop_zone is not None:
                    print(f"V20_DROP_ZONE_LANDING t={current_time:.2f}s "
                          f"ball_xy={np.round(ball_xyz[:2],3).tolist()} "
                          f"centre_error_m={release_final_distance:.3f} "
                          f"inside_circle={release_target_success} "
                          f"stable_ground_contact={ground_contact_verified}")
                print(f"V8_RELEASE_RESULT DETACHED_AND_FELL "
                      f"ball_xy_error_m={release_final_distance:.3f} "
                      f"placement_within_tolerance={release_target_success} "
                      f"drop_m={release_drop:.3f} "
                      f"ground_contact={ground_contact_verified} no_attachment=True")
                grasp_state = "return_home"
                return_home_started_at = current_time
                return_home_in_tolerance_since = None
                carry_stop_wheel_q = q_mes[0:4].copy()
                print(f"V19_ARM_RETURN_START t={current_time:.2f}s "
                      f"home_q={np.round(original_arm_home_q,3).tolist()}")
            if current_time-release_started_at > args.release_timeout:
                release_failure_reason=(f"Expert opening timed out: remaining_robot_contacts={robot_release_contact_count} "
                                        f"max_drop_m={release_max_drop:.3f} ball_xy_error_m={release_final_distance:.3f}")
                print(f"V8_RELEASE_FAIL {release_failure_reason}")
                break

        if grasp_state == "return_home":
            joint_error = float(np.max(np.abs(q_mes[4:11]-original_arm_home_q)))
            joint_speed = float(np.max(np.abs(qd_mes[4:11])))
            if (joint_error <= args.return_home_joint_tolerance and
                    joint_speed <= args.return_home_joint_speed_max):
                if return_home_in_tolerance_since is None:
                    return_home_in_tolerance_since = current_time
                elif current_time-return_home_in_tolerance_since >= 0.40:
                    return_home_success = True
                    print(f"V19_ARM_RETURN_PASS t={current_time:.2f}s "
                          f"max_joint_error_rad={joint_error:.3f} "
                          f"max_joint_speed_rad_s={joint_speed:.3f}")
                    break
            else:
                return_home_in_tolerance_since = None
            if step % 500 == 0:
                print(f"V19_ARM_RETURN t={current_time:.2f}s "
                      f"max_joint_error_rad={joint_error:.3f} "
                      f"max_joint_speed_rad_s={joint_speed:.3f} "
                      f"joint_error_vector_rad={np.round(q_mes[4:11]-original_arm_home_q,3).tolist()} "
                      f"motor_mode=POSITION_SERVO source={args.return_home_controller}")
            if current_time-return_home_started_at >= args.return_home_timeout:
                print(f"V19_ARM_RETURN_TIMEOUT max_joint_error_rad={joint_error:.3f}; "
                      "ball release result remains independently verified")
                break

        if step % 1000 == 0:
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

        # Record the actual reference action sent to the joint controller, not IK endpoints.
        if (args.record_expert_npz and grasp_state in ("lowering", "release_open")
                and int(round(current_time/0.02)) != placement_record_tick):
            placement_record_tick = int(round(current_time/0.02))
            hand_now = np.asarray(p.getLinkState(rid, hand_link, computeForwardKinematics=True)[4])
            placement_obs.append(placement_features(q_mes, qd_mes, ball_xyz,
                p.getBaseVelocity(ball_id)[0], hand_now, dropoff_xy,
                args.ground_ball_center_z, base_bearing, current_time-ground_lower_started_at))
            placement_actions.append(np.r_[arm_hold_target-q_mes[4:11],
                target_finger_angles.copy(), float(grasp_state == "release_open")])
            placement_times.append(current_time)

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

        if not args.headless and args.gui_sleep > 0:
            time.sleep(args.gui_sleep)
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
    print("========== V8 EXPERT NAVIGATE + RELEASE RESULT ==========")
    print("NOTE: RELEASE_TO_GRAVITY is NOT finger-opening. The real drop is V8_EXPERT_RELEASE_START.")
    print(f"Requested BALL drop-off (world XY): {np.round(dropoff_xy,3).tolist()}")
    if drop_zone is not None:
        print(f"V20_DROP_ZONE radius_m={drop_zone.radius:.3f} "
              f"landing_center_radius_m={drop_zone.landing_radius:.3f}")
    print(f"Action sources: carry={args.carry_controller}; grasp=V4_ML; lowering/release={args.placement_controller}")
    print(f"Carry success before opening: {carry_success}")
    print(f"V8_2_ARRIVAL_REACHED: {dropoff_target_reached}")
    print(f"V8_2_OPEN_COMMAND_SENT: {release_started_at is not None}")
    print(f"V19_ARM_RETURN_HOME: {'PASS' if return_home_success else 'NOT_VERIFIED'}")
    print(f"Ball detached and fell under gravity: {release_success}")
    placement_tolerance = (drop_zone.landing_radius if drop_zone is not None
                           else (args.ground_place_xy_tolerance if args.place_on_ground
                                 else args.release_final_tolerance))
    if drop_zone is not None:
        print(f"Ball landed fully INSIDE circle (radius {drop_zone.radius:.2f}m): {release_target_success}")
    else:
        print(f"Ball XY within {placement_tolerance:.2f}m of destination after release: {release_target_success}")
    print(f"Ball made stable non-robot ground contact: {ground_contact_verified}")
    release_at_goal_success = bool(carry_success and dropoff_target_reached
                                   and release_started_at is not None and release_success
                                   and (not args.place_on_ground or
                                        (ground_contact_verified and release_target_success)))
    accurate_placement_success = bool(release_at_goal_success and release_target_success
                                      and (not args.place_on_ground or ground_contact_verified))
    print(f"V8_RELEASE_AT_TARGET_RESULT: {'PASS' if release_at_goal_success else 'FAIL/NOT_VERIFIED'}")
    print(f"V20_CIRCULAR_DROP_RESULT: {'PASS' if accurate_placement_success else 'FAIL/NOT_VERIFIED'} "
          f"landed_inside_circle={release_target_success} "
          f"ground_contact={ground_contact_verified} "
          f"arm_return_home={return_home_success}")
    print(f"V9_GROUND_PLACEMENT_RESULT: {'PASS' if accurate_placement_success else 'FAIL/NOT_VERIFIED'} "
          f"final_xy_error_m={release_final_distance:.3f} "
          f"tolerance_m={placement_tolerance:.3f} "
          f"ground_contact={ground_contact_verified}")
    print(f"V8_PICK_CARRY_DROP_RESULT: {'PASS' if release_at_goal_success else 'FAIL/NOT_VERIFIED'}")
    print(f"Maximum relative slip while carrying: {carry_max_relative_drift*1000:.1f} mm")
    print(f"Maximum downward movement while carrying: {carry_max_ball_drop*1000:.1f} mm")
    print(f"Contact samples with >=2 finger groups during carry: {carry_good_samples/max(1,carry_total_samples):.3f}")
    print(f"Expert release: open targets={np.round(expert_open_fingers,3).tolist()} "
          f"max_ball_drop_m={release_max_drop:.3f} final_xy_error_m={release_final_distance:.3f}")
    print(f"Carry failure reason: {carry_fail_reason}")
    print(f"Release failure reason: {release_failure_reason}")
    print("Policy outputs are tracked by programmed joint control and guarded by physics checks.")
    if args.return_home_record_npz:
        if return_home_policy is not None:
            raise RuntimeError("Cannot record expert return-home labels from a learned rollout")
        record_path = Path(args.return_home_record_npz)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open('xb') as output:
            np.savez_compressed(output,
                obs=np.asarray(return_home_demo_obs, dtype=np.float32).reshape(-1, 14),
                action_delta=np.asarray(return_home_demo_actions, dtype=np.float32).reshape(-1, 7),
                source=np.asarray('EXPERT_RETURN_HOME_SERVO'),
                successful=np.asarray(return_home_success))
        print(f"V21_RETURN_HOME_EXPERT_RECORD path={record_path} "
              f"samples={len(return_home_demo_obs)} "
              f"verified_success={return_home_success}")
    if args.record_expert_npz and args.carry_controller == 'expert':
        with Path(args.record_expert_npz).open('xb') as output:
            np.savez_compressed(output,
                placement_format=np.asarray('robotnik_placement_v1'),
                placement_label_source=np.asarray('EXPERT_JOINT_REFERENCE_AND_RELEASE_DECISION'),
                placement_obs_37=np.asarray(placement_obs, dtype=np.float32).reshape(-1, 37),
                placement_actions_14=np.asarray(placement_actions, dtype=np.float32).reshape(-1, 14),
                placement_times=np.asarray(placement_times, dtype=np.float32),
                feature_names=np.asarray(['remaining_m', 'lateral_m', 'heading_err_rad',
                                         'forward_speed_mps', 'yaw_rate_radps']),
                obs=np.asarray(carry_obs, dtype=np.float32).reshape(-1, 5),
                actions_vw=np.asarray(carry_actions, dtype=np.float32).reshape(-1, 2),
                elapsed_s=np.asarray(carry_times, dtype=np.float32),
                source=np.asarray('SCRIPTED_EXPERT_TRANSFER_ONLY; V4_ML_GRASP'),
                physical_success_verified=np.asarray(bool(release_at_goal_success)),
                success_definition=np.asarray('BALL_LOWERED_TO_GROUND_AT_DROPOFF; DETACHED; STABLE_GROUND_CONTACT; XY_WITHIN_TOLERANCE'),
                release_at_goal_verified=np.asarray(bool(release_at_goal_success)),
                final_placement_verified=np.asarray(bool(accurate_placement_success)),
                final_ball_xy_error_m=np.asarray(float(release_final_distance)),
                final_ball_xy_tolerance_m=np.asarray(float(placement_tolerance)),
                ground_contact_verified=np.asarray(bool(ground_contact_verified)),
                ground_release_center_z_m=np.asarray(float(args.ground_ball_center_z)),
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
    print("This is an ML gripper evaluation with hybrid handover, NOT end-to-end control.")
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
        "carry_mode": 'expert',
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
    parser = argparse.ArgumentParser(description="Robotnik hybrid expert: learned pickup/grasp; expert carry, lower-to-ground, open and verify.")
    parser.add_argument("--carry_controller", choices=["expert", "bc"], default="expert",
                        help="Expert controls navigation/release; existing networks control approach and grasp")
    parser.add_argument('--carry_checkpoint', default=None)
    parser.add_argument('--placement_controller', choices=['expert', 'bc'], default='expert')
    parser.add_argument('--placement_checkpoint', default=None)
    parser.add_argument('--placement_servo_force',type=float,default=35.0,
                        help='Maximum learned placement joint servo force (Nm); always limited by URDF joint effort')
    parser.add_argument('--placement_servo_speed',type=float,default=0.20,
                        help='Maximum placement actuator joint tracking speed (rad/s)')
    parser.add_argument('--placement_servo_position_gain',type=float,default=0.32,
                        help='Position tracking gain for learned joint references, not a trajectory')
    parser.add_argument('--placement_joint_step',type=float,default=0.035,
                        help='Max neural reference delta per 20ms, radians; bounded actuator setpoint')
    parser.add_argument('--local_drop',action='store_true',
                        help='Grasp, then learned arm bend/drop in same XY zone; wheel carry disabled')
    parser.add_argument("--dropoff_xy", type=float, nargs=2, default=[3.0, 3.5],
                        metavar=("X", "Y"),
                        help="World XY position where the BALL should be placed (default 3.0 3.5)")
    parser.add_argument("--place_on_ground", action="store_true",
                        help="Lower held ball to a guarded pose, then drop into the circular region")
    parser.add_argument("--drop_zone_radius", type=float, default=0.50,
                        help="Radius (metres) of circular ground landing region, default 0.50m")
    parser.add_argument("--drop_zone_release_margin", type=float, default=0.10,
                        help="Extra XY margin inside circle for natural ball drift during fall (m)")
    parser.add_argument("--drop_zone_approach_margin", type=float, default=0.07,
                        help="Stop carrying this far inside safe release radius; base stops early (m)")
    parser.add_argument("--hide_drop_zone_visual", action="store_true",
                        help="Do not render translucent green landing circle")
    parser.add_argument("--gui_sleep", type=float, default=0.0,
                        help="Optional artificial delay after each physics step in GUI (default 0)")
    parser.add_argument("--ground_ball_center_z", type=float, default=0.25,
                        help="Ball-centre release height; the ball then falls naturally to the z=0 floor")
    parser.add_argument("--ground_release_target_offset_x", type=float, default=0.0,
                        help="Ground-mode pre-release X compensation (m), applied during carry and arm lowering")
    parser.add_argument("--ground_release_target_offset_y", type=float, default=0.0,
                        help="Ground-mode pre-release Y compensation (m), applied during carry and arm lowering")
    parser.add_argument("--ground_place_xy_tolerance", type=float, default=0.05,
                        help="Maximum final ball XY placement error for ground-place PASS")
    parser.add_argument("--ground_pre_release_xy_tolerance", type=float, default=0.030,
                        help="Required ball XY accuracy before opening; leaves margin for the drop")
    parser.add_argument("--ground_release_finger_speed", type=float, default=0.25,
                        help="Finger opening speed in rad/s during ground release (max 0.25 to shorten contact while opening)")
    parser.add_argument("--ground_release_finger_force", type=float, default=1.0,
                        help="Reduced finger motor force in N during ground release")
    parser.add_argument("--ground_place_height_tolerance", type=float, default=0.025,
                        help="Allowed ball-centre height error after arm lowering")
    parser.add_argument("--ground_place_speed_max", type=float, default=0.04,
                        help="Maximum ball speed before opening at the ground placement pose")
    parser.add_argument("--ground_lower_seconds", type=float, default=3.0,
                        help="Smooth arm lowering duration in seconds")
    parser.add_argument("--ground_cartesian_speed", type=float, default=0.030,
                        help="Expert: commanded ball lowering speed (m/s); slows/pauses when XY drifts")
    parser.add_argument("--ground_ik_interval", type=float, default=0.050,
                        help="Expert: seconds between measured-ball Cartesian IK updates")
    parser.add_argument("--ground_xy_pause_error", type=float, default=0.055,
                        help="Expert: pause descent when ball XY error exceeds this many metres")
    parser.add_argument("--ground_cartesian_xy_step", type=float, default=0.012,
                        help="Expert: maximum XY hand translation per IK update (m)")
    parser.add_argument("--ground_cartesian_z_step", type=float, default=0.008,
                        help="Expert: maximum Z hand translation per IK update (m)")
    parser.add_argument("--ground_joint_target_step", type=float, default=0.035,
                        help="Expert: maximum change of each commanded arm joint target per IK update (rad)")
    parser.add_argument("--ground_lower_timeout", type=float, default=65.0,
                        help="Abort without opening if the ball cannot reach the ground pose")
    parser.add_argument("--reach_probe_step_m", type=float, default=0.065,
                        help="Adaptive IK probe half-span along measured base heading (m)")
    parser.add_argument("--reach_residual_max_m", type=float, default=0.018,
                        help="FK position error below which base is already well placed")
    parser.add_argument("--reach_max_unreachable_residual_m", type=float, default=0.085,
                        help="Abort unrecoverable ground reach if kinematic FK error stays above this")
    parser.add_argument("--reach_max_base_shift_m", type=float, default=0.26,
                        help="Maximum aggregate physical base reposition distance (m)")
    parser.add_argument("--reach_max_adjustments", type=int, default=3,
                        help="Maximum adaptive reachability-driven base reposition steps")
    parser.add_argument("--reach_base_speed_max", type=float, default=0.016,
                        help="Maximum measured-feedback correction drive speed (m/s)")
    parser.add_argument("--reach_replan_xy_error_m", type=float, default=0.055,
                        help="Interrupt lowering and recheck arm workspace if measured ball XY drifts (m)")
    parser.add_argument("--reach_reposition_min_ball_z", type=float, default=0.45,
                        help="Minimum ball height allowing a physical base correction during placement")
    parser.add_argument("--reach_base_adjust_timeout", type=float, default=18.0,
                        help="Timeout for each short base correction (s)")
    parser.add_argument("--return_home_controller", choices=["expert", "bc"], default="expert",
                        help="Expert only for collecting demonstration; bc runs trained neural policy")
    parser.add_argument("--return_home_checkpoint", default=None,
                        help="Trained neural return-home checkpoint; required for bc")
    parser.add_argument("--return_home_record_npz", default=None,
                        help="Record expert return-home states/actions for subsequent training; never BC")
    parser.add_argument("--return_home_timeout", type=float, default=24.0,
                        help="Maximum time to return arm to its recorded initial joint posture (s)")
    parser.add_argument("--return_home_joint_tolerance", type=float, default=0.075,
                        help="Joint-error threshold to verify arm returned home (rad)")
    parser.add_argument("--return_home_joint_speed_max", type=float, default=0.07,
                        help="Speed threshold to verify home-settling (rad/s)")
    parser.add_argument("--ground_place_settle_seconds", type=float, default=0.4,
                        help="Required stable near-floor pose time before opening")
    parser.add_argument("--ground_contact_seconds", type=float, default=0.2,
                        help="Required stable ball-to-floor contact after release")
    parser.add_argument("--ground_min_release_drop", type=float, default=0.005,
                        help="Minimum observed fall after opening in ground-place mode")
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
    parser.add_argument("--carry_goal_tolerance", type=float, default=0.075,
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
    parser.add_argument('--record_corrections_npz', default=None,
                        help='Save counterfactual TRAINING labels from visited student states; no teacher actions are executed')
    parser.add_argument('--record_full_npz', default=None,
                        help='Optional new physical trace, NOT certified success')
    parser.add_argument('--record_ml_npz', default=None,
                        help='Save real SensorPolicy observations/predictions for audit, NOT expert labels')
    args = parser.parse_args()
    if args.carry_controller == 'bc' and not args.carry_checkpoint:
        parser.error('--carry_controller bc requires --carry_checkpoint')
    if args.placement_controller == 'bc' and (not args.placement_checkpoint or not args.place_on_ground):
        parser.error('--placement_controller bc requires --placement_checkpoint and --place_on_ground')
    if args.return_home_controller == 'bc' and not args.return_home_checkpoint:
        parser.error('--return_home_controller bc requires --return_home_checkpoint')
    if args.return_home_record_npz and args.return_home_controller != 'expert':
        parser.error('Expert demonstrations cannot be recorded from a learned return-home policy')
    if args.record_expert_npz and args.placement_controller != 'expert':
        parser.error('Student predictions cannot be recorded as expert labels')
    if args.record_expert_npz and args.carry_controller != "expert":
        parser.error("--record_expert_npz is only allowed with --carry_controller expert")
    if len(args.dropoff_xy)!=2 or not np.all(np.isfinite(args.dropoff_xy)):
        parser.error("--dropoff_xy requires two finite world coordinates")
    if not args.local_drop and np.linalg.norm(np.asarray(args.dropoff_xy)-np.asarray(args.target[:2])) < 0.15:
        parser.error("Drop-off point must be at least 0.15m from the ball pickup point")
    if not (0 <= args.gui_sleep <= 0.01):
        parser.error("--gui_sleep must be between 0 and 0.01 seconds")
    if not (0.15 <= args.drop_zone_radius <= 1.50 and
            0 <= args.drop_zone_release_margin < args.drop_zone_radius-0.07 and
            0 <= args.drop_zone_approach_margin < args.drop_zone_radius -
            args.drop_zone_release_margin - 0.04):
        parser.error("Invalid drop zone radius/margin: need room for complete ball footprint")
    if not 0 < args.carry_speed_max <= 0.10:
        parser.error("--carry_speed_max must be >0 and <=0.10m/s")
    if not 0 < args.carry_yaw_max <= 1:
        parser.error("--carry_yaw_max must be >0 and <= 1rad/s")
    if args.carry_timeout <= 0 or args.carry_settle_seconds <= 0 or args.release_timeout <= 0:
        parser.error("Carry, settling, and release times must be positive")
    if any(v <= 0 for v in (args.ground_place_xy_tolerance,
                            args.ground_pre_release_xy_tolerance,
                            args.ground_release_finger_speed,
                            args.ground_place_height_tolerance,
                            args.ground_place_speed_max,
                            args.ground_lower_seconds,
                            args.ground_lower_timeout,
                            args.ground_place_settle_seconds,
                            args.ground_contact_seconds,
                            args.ground_min_release_drop)):
        parser.error("Ground-place tolerances, speeds, and times must be positive")
    if not (0 < args.ground_cartesian_speed <= 0.10 and
            0.02 <= args.ground_ik_interval <= 0.2 and
            0.01 < args.ground_xy_pause_error <= 0.2 and
            0.001 < args.ground_cartesian_xy_step <= 0.03 and
            0.001 < args.ground_cartesian_z_step <= 0.03 and
            0.005 <= args.ground_joint_target_step <= 0.10):
        parser.error("Unsafe ground Cartesian servo settings")
    if args.ground_release_finger_speed > 0.25:
        parser.error("--ground_release_finger_speed must be <=0.25 rad/s")
    if not 0.0 < args.ground_release_finger_force <= 2.0:
        parser.error("--ground_release_finger_force must be >0 and <=2.0 N")
    if not (5.0 <= args.placement_servo_force <= 40.0 and
            0.05 <= args.placement_servo_speed <= 0.35 and
            0.05 <= args.placement_servo_position_gain <= 0.6 and
            0.005 <= args.placement_joint_step <= 0.05):
        parser.error('Unsafe learned placement actuator parameters')
    if not 0.12 <= args.ground_ball_center_z <= 0.60:
        parser.error("--ground_ball_center_z release height must be between 0.12m and 0.60m")
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
            break
    else:
        print(f"\nAll {max_attempts} attempts crashed.")


if __name__ == "__main__":
    main()
