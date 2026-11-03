"""Controllers, stop criteria and setup helpers for the Robotnik mobile base + Kinova arm reaching task."""
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
from simulation_and_control import wrap_angle


def quaternion2bearing(q_w, q_x, q_y, q_z):
    """Convert quaternion orientation to planar bearing (yaw angle)."""
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()
    base_euler = pin.rpy.matrixToRpy(quat.toRotationMatrix())
    return base_euler[2]


@dataclass
class ArmControllerParams:
    """Tuning parameters of the arm reaching controller."""
    reach_limit: float                   # m, max EE distance from the shoulder the arm is asked to reach
    kp: float = 100.0                    # joint PD position gain
    kd: float = 30.0                     # joint PD velocity gain
    max_joint_vel: float = 1.0           # rad/s
    max_q_lead: float = 0.15             # rad, max distance of the joint reference from the measured angle
    max_torque: float = 30.0             # N*m
    cartesian_kp: float = 1.5
    max_cartesian_speed: float = 0.25    # m/s
    min_ik_damping: float = 0.01
    max_ik_damping: float = 0.10
    singularity_threshold: float = 0.15
    limit_margin: float = 0.05           # rad, joints this close to a limit are treated as saturated


def arm_controller(dyn_model, q_pin, q_arm, qd_arm, q_des_arm, q_hold_arm, target_pos, ee_pos,
                   ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, dt, params, reach_active):
    """Cartesian reaching controller for the 7-joint arm.

    Differential IK (singularity-damped, joint-limit aware) generates the joint reference, which a
    gravity-compensated joint PD law tracks. Forward kinematics must already be up to date in
    dyn_model.pin_data for q_pin. When reach_active is False the arm holds q_hold_arm.

    Returns (torque_arm, q_des_arm, qd_des_arm).
    """
    if not reach_active:
        q_des_arm = q_hold_arm.copy()
        qd_des_arm = np.zeros(7)
    else:
        # End-effector Jacobian in the world-aligned frame; columns 6..12 are the arm joints
        pin.computeJointJacobians(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        J_lwa = pin.computeFrameJacobian(
            dyn_model.pin_model, dyn_model.pin_data, q_pin, ee_frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        J_arm = J_lwa[:3, 6:13]

        # Increase damping near singular poses to keep joint velocities bounded.
        sigma_min = np.linalg.svd(J_arm, compute_uv=False)[-1]
        damping = np.interp(
            sigma_min,
            [0.0, params.singularity_threshold],
            [params.max_ik_damping, params.min_ik_damping],
        )

        # Proportional Cartesian velocity command towards the (reach-limited) target
        shoulder_pos = dyn_model.pin_data.oMf[shoulder_frame_id].translation
        offset = target_pos - shoulder_pos
        if np.linalg.norm(offset) > params.reach_limit:
            offset *= params.reach_limit / np.linalg.norm(offset)
        ik_error = shoulder_pos + offset - ee_pos
        v_cartesian = params.cartesian_kp * ik_error
        cartesian_speed = np.linalg.norm(v_cartesian)
        if cartesian_speed > params.max_cartesian_speed:
            v_cartesian *= params.max_cartesian_speed / cartesian_speed

        # Saturation-aware solve: joints at a limit that the motion would push further into are
        # frozen, and their share of the motion is redistributed to the remaining joints.
        has_limits = arm_lower < arm_upper
        free = np.ones(7, dtype=bool)
        qd_des_arm = np.zeros(7)
        for _ in range(7):
            J_free = J_arm[:, free]
            regularized_jacobian = J_free @ J_free.T + damping**2 * np.eye(3)
            qd_des_arm = np.zeros(7)
            qd_des_arm[free] = J_free.T @ np.linalg.solve(regularized_jacobian, v_cartesian)
            pushing_lower = has_limits & (q_arm <= arm_lower + params.limit_margin) & (qd_des_arm < 0)
            pushing_upper = has_limits & (q_arm >= arm_upper - params.limit_margin) & (qd_des_arm > 0)
            newly_blocked = (pushing_lower | pushing_upper) & free
            if not newly_blocked.any():
                break
            free &= ~newly_blocked
            qd_des_arm[:] = 0.0
        max_joint_speed = np.max(np.abs(qd_des_arm))
        if max_joint_speed > params.max_joint_vel:
            qd_des_arm *= params.max_joint_vel / max_joint_speed
        q_des_arm = q_des_arm + qd_des_arm * dt
        # Anti-windup: for unreachable targets the reference must not run away from the measured state
        q_des_arm = np.clip(q_des_arm, q_arm - params.max_q_lead, q_arm + params.max_q_lead)
        # Keep the reference inside the limits so PD overshoot does not hit the hard stop
        q_des_arm = np.where(
            has_limits,
            np.clip(q_des_arm, arm_lower + params.limit_margin, arm_upper - params.limit_margin),
            q_des_arm,
        )

    # Joint-space PD law with gravity compensation
    torque_arm = params.kp * (q_des_arm - q_arm) + params.kd * (qd_des_arm - qd_arm)
    tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    torque_arm = np.clip(torque_arm + tau_g[6:13], -params.max_torque, params.max_torque)
    return torque_arm, q_des_arm, qd_des_arm


@dataclass
class StopCriteria:
    """Thresholds and running state for the simulation stop conditions."""
    reach_tolerance: float                     # m
    max_steps: int                             # 0 disables the step cap
    min_time: float = 8.0                      # s before reached / stalled can trigger
    stall_window: float = 3.0                  # s without meaningful improvement of the EE error
    stall_improvement: float = 0.002           # m
    best_ee_error: float = np.inf
    best_ee_error_time: float = 0.0


def check_stop(criteria, current_time, time_step, ee_dist_error, base_navigation_done):
    """Evaluate the stop conditions for the current step.

    Returns a message describing why the simulation should stop, or None to continue.
    Stall tracking only runs once the base navigation is done (the arm is reaching).
    """
    if criteria.max_steps and current_time / time_step >= criteria.max_steps:
        return f"Reached step cap {criteria.max_steps} at t={current_time:.2f}s"

    if base_navigation_done:
        if ee_dist_error < criteria.best_ee_error - criteria.stall_improvement:
            criteria.best_ee_error = ee_dist_error
            criteria.best_ee_error_time = current_time
        elif (current_time - criteria.best_ee_error_time > criteria.stall_window
              and current_time > criteria.min_time):
            return (f"Target not reachable: closest pose reached at t={current_time:.2f}s "
                    f"(best EE error {criteria.best_ee_error:.4f} m, current {ee_dist_error:.4f} m)")

    if base_navigation_done and ee_dist_error < criteria.reach_tolerance and current_time > criteria.min_time:
        return f"Target reached within {criteria.reach_tolerance*1000:.0f}mm tolerance at t={current_time:.2f}s!"
    return None


@dataclass
class BaseControllerParams:
    """Tuning parameters of the mobile-base navigation controller."""
    nav_timeout: float             # s, max time allowed for the base to reach its goal pose
    kp_pos: float = 1.0
    kp_ori: float = 3.0
    max_lin_vel: float = 0.25      # m/s
    min_lin_vel: float = 0.05      # m/s, avoids stalling against wheel friction near the goal
    max_ang_vel: float = 1.0       # rad/s
    pos_tol: float = 0.08          # m
    ang_tol: float = 0.05          # rad
    wheel_radius: float = 0.11
    # Effective track width for skid-steer (calibrated in simulation: yaw rate ~0.34 rad/s per rad/s of wheel speed)
    wheel_base_width: float = 0.66


@dataclass
class BaseNavState:
    """Navigation phases: align (turn in place toward goal) -> drive -> orient (face target) -> done.

    With lidar, a leg can go to a detour waypoint first: align -> drive to the waypoint, then
    align -> drive to the goal (replanning at the start of every leg).
    """
    phase: str = "align"
    done: bool = False
    waypoint: np.ndarray = None    # current detour waypoint (world xy), None = heading for the goal
    leg_planned: bool = False      # the current leg has been checked against the lidar
    n_plans: int = 0


@dataclass
class LidarParams:
    """Planar 2D lidar on the base: horizontal rays all around at a fixed height."""
    n_rays: int = 72
    max_range: float = 3.0         # m, measured from the base centre
    min_range: float = 0.5         # m, rays start outside the chassis so they never hit the robot itself
    height: float = 0.2            # m, scan plane above the floor


def lidar_scan(client, base_pos, base_bearing, params):
    """Cast the lidar rays. Returns (ranges, angles): ranges in m from the base centre,
    angles in the robot frame (0 = straight ahead). A miss reads max_range."""
    angles = np.linspace(-np.pi, np.pi, params.n_rays, endpoint=False)
    world = angles + base_bearing
    dirs = np.stack([np.cos(world), np.sin(world), np.zeros_like(world)], axis=1)
    origin = np.array([base_pos[0], base_pos[1], params.height])
    hits = client.rayTestBatch((origin + params.min_range * dirs).tolist(),
                               (origin + params.max_range * dirs).tolist())
    fractions = np.array([h[2] for h in hits])
    return params.min_range + fractions * (params.max_range - params.min_range), angles


@dataclass
class AvoidParams:
    """Lidar path planning: obstacles are passed via waypoints around them."""
    robot_radius: float = 0.47     # m, chassis half-diagonal
    margin: float = 0.15           # m, extra clearance kept between chassis and obstacle
    max_plans: int = 100
    cluster_gap: float = 0.3       # m, lidar hits closer than this belong to one obstacle
    nodes_per_obstacle: int = 8
    min_obstacle_radius: float = 0.25  # m, lidar sees only the near side, so never assume smaller
    replan_slack: float = 0.06     # m, clearance given up before a driving leg counts as blocked
    waypoint_tol: float = 0.15     # m, a waypoint counts as reached within this distance
    realign_err: float = 0.35      # rad, heading error while driving that triggers a turn in place
    # m, goal tolerance with lidar. Turning in place also shifts the base centre, so chasing the
    # plain 8 cm tolerance after a detour can orbit the goal; the arm covers the extra offset.
    goal_tol: float = 0.2
    max_range: float = 3.0         # m, must match LidarParams.max_range (misses are ignored)


def lidar_obstacles(base_xy, base_bearing, ranges, angles, params):
    """Group the lidar hits into obstacles, each modelled as a circle. Returns [(centre, radius)].

    Adjacent rays whose hits are close belong to one obstacle. Only the near side is seen, so the
    centre is pushed back from the visible arc and the radius is padded by a ray spacing.
    """
    hit = ranges < params.max_range - 1e-3
    world = angles + base_bearing
    pts = base_xy + ranges[:, None] * np.stack([np.cos(world), np.sin(world)], axis=1)
    clusters, cur = [], []
    for i in range(len(ranges)):
        if hit[i] and cur and np.linalg.norm(pts[i] - pts[cur[-1]]) < params.cluster_gap:
            cur.append(i)
        else:
            if cur:
                clusters.append(cur)
            cur = [i] if hit[i] else []
    if cur:
        clusters.append(cur)
    # The scan wraps around: join the last run onto the first if they touch
    if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == len(ranges) - 1 \
            and np.linalg.norm(pts[0] - pts[-1]) < params.cluster_gap:
        clusters[0] = clusters.pop() + clusters[0]

    ray_spacing = 2 * np.pi / len(ranges)
    circles = []
    for c in clusters:
        p = pts[c]
        extent = np.linalg.norm(p[:, None] - p[None], axis=2).max()
        radius = max(params.min_obstacle_radius, extent / 2 + ranges[c].min() * ray_spacing)
        away = p.mean(axis=0) - base_xy
        away /= np.linalg.norm(away)
        circles.append((p.mean(axis=0) + 0.7 * radius * away, radius))
    return circles


def _segment_hits(zones, a, b, slack=0.0):
    """True if the segment a->b enters any keep-out zone (centre, radius) shrunk by `slack`."""
    d = b - a
    length2 = d @ d
    for centre, radius in zones:
        s = np.clip((centre - a) @ d / length2, 0.0, 1.0) if length2 > 1e-12 else 0.0
        if np.linalg.norm(a + s * d - centre) < radius - slack:
            return True
    return False


def _keep_out(circles, start, goal, clearance):
    """Inflate obstacles by `clearance` into keep-out zones for the robot centre.

    If the robot (after slipping) or the goal is already inside a full zone, that zone is shrunk to
    just exclude it. Dropping the obstacle instead would let the planner drive straight through it.
    """
    return [(c, max(0.05, min(r + clearance, np.linalg.norm(start - c) - 0.05, np.linalg.norm(goal - c) - 0.05)))
            for c, r in circles]


def plan_path(circles, start, goal, params):
    """Shortest collision-free path start -> goal as a list of points (goal last).

    Visibility graph: nodes on a polygon around each inflated obstacle, edges wherever the straight
    segment is clear, then Dijkstra. Falls back to [goal] if no path exists.
    """
    zones = _keep_out(circles, start, goal, params.robot_radius + params.margin)
    nodes = [start, goal]
    k = params.nodes_per_obstacle
    ring = np.stack([np.cos(np.linspace(0, 2 * np.pi, k, endpoint=False)),
                     np.sin(np.linspace(0, 2 * np.pi, k, endpoint=False))], axis=1)
    for centre, radius in zones:
        # Polygon just outside the zone, so its edges stay clear of it
        for n in centre + (radius / np.cos(np.pi / k) + 0.02) * ring:
            if all(np.linalg.norm(n - c) > r for c, r in zones):
                nodes.append(n)

    n_nodes = len(nodes)
    dist = np.full(n_nodes, np.inf)
    prev = np.full(n_nodes, -1)
    done = np.zeros(n_nodes, dtype=bool)
    dist[0] = 0.0
    while True:
        i = int(np.argmin(np.where(done, np.inf, dist)))
        if done[i] or not np.isfinite(dist[i]):
            return [goal]
        if i == 1:
            break
        done[i] = True
        for j in range(n_nodes):
            if done[j]:
                continue
            step = np.linalg.norm(nodes[j] - nodes[i])
            if dist[i] + step < dist[j] and not _segment_hits(zones, nodes[i], nodes[j]):
                dist[j] = dist[i] + step
                prev[j] = i
    path, i = [], 1
    while i != 0:
        path.append(nodes[i])
        i = prev[i]
    return path[::-1]


def base_controller(state, base_pos, base_bearing, goal_pos, goal_bearing, current_time, params,
                    lidar=None, avoid_params=None):
    """Skid-steer base navigation to a goal pose; updates state in place.

    lidar: optional (ranges, angles) from lidar_scan. When given, the path is checked against the scan
        and detour waypoints are planned around the obstacles on it. Every leg is still driven with the
        plain turn-in-place + straight drive: the wheels have almost no lateral grip, so turning while
        moving makes the base slide.

    Returns the wheel velocity commands [front_right, front_left, back_left, back_right] (rad/s).
    Once navigation is done the base holds position (zero wheel velocities).
    """
    if state.done:
        return np.zeros(4)

    ap = avoid_params or AvoidParams()
    base_xy = base_pos[:2]
    if lidar is not None and state.phase in ("align", "drive"):
        leg_target = state.waypoint if state.waypoint is not None else goal_pos
        circles = lidar_obstacles(base_xy, base_bearing, lidar[0], lidar[1], ap)
        # Plan at the start of each leg toward the goal, and replan whenever the current leg is blocked.
        # The blocked check uses a slightly smaller clearance than planning, so a leg that just grazes
        # the margin does not trigger a replan on every step as the obstacle estimates jitter.
        new_leg = state.waypoint is None and not state.leg_planned
        zones = _keep_out(circles, base_xy, leg_target, ap.robot_radius + ap.margin)
        blocked = _segment_hits(zones, base_xy, leg_target, ap.replan_slack)
        if (new_leg or blocked) and state.n_plans < ap.max_plans:
            state.leg_planned = True
            state.n_plans += 1
            # Skip path points that already count as reached, or the next step would replan again
            path = [p for p in plan_path(circles, base_xy, goal_pos, ap)[:-1]
                    if np.linalg.norm(p - base_xy) > 2 * ap.waypoint_tol]
            state.waypoint = path[0] if path else None
            state.phase = "align"

    target = state.waypoint if state.waypoint is not None else goal_pos
    if state.waypoint is not None:
        tol = ap.waypoint_tol
    else:
        tol = ap.goal_tol if lidar is not None else params.pos_tol
    dist_to_goal = np.linalg.norm(base_xy - target)
    to_goal = target - base_xy
    heading_to_goal = np.arctan2(to_goal[1], to_goal[0])
    v_cmd, w_cmd = 0.0, 0.0
    if state.phase == "align":
        err = wrap_angle(heading_to_goal - base_bearing)
        if dist_to_goal < tol or abs(err) < params.ang_tol:
            state.phase = "drive" if dist_to_goal >= tol else "orient"
        else:
            w_cmd = np.clip(params.kp_ori * err, -params.max_ang_vel, params.max_ang_vel)
    if state.phase == "drive":
        err = wrap_angle(heading_to_goal - base_bearing)
        passed_goal = dist_to_goal < 0.15 and np.cos(err) < 0.0
        if dist_to_goal < tol or passed_goal:
            state.phase = "orient"
        elif lidar is not None and abs(err) > ap.realign_err:
            # Slipped off the line: stop and turn in place rather than curving (which slides more)
            state.phase = "align"
        else:
            # Gentle steering while driving: skid-steer slips a lot when turning hard, so keep it mostly straight
            v_cmd = np.clip(params.kp_pos * dist_to_goal, params.min_lin_vel, params.max_lin_vel) * max(0.0, np.cos(err))
            w_cmd = np.clip(1.5 * err, -0.3, 0.3)
    if state.phase == "orient" and state.waypoint is not None:
        # Waypoint reached: start the next leg toward the goal
        state.waypoint = None
        state.leg_planned = False
        state.phase = "align"
    if state.phase == "orient":
        err = wrap_angle(goal_bearing - base_bearing)
        if abs(err) < params.ang_tol:
            state.done = True
        else:
            w_cmd = np.clip(params.kp_ori * err, -params.max_ang_vel, params.max_ang_vel)
    lw = (v_cmd - w_cmd * params.wheel_base_width / 2) / params.wheel_radius
    rw = (v_cmd + w_cmd * params.wheel_base_width / 2) / params.wheel_radius
    if current_time > params.nav_timeout:
        state.done = True
    return np.array([rw, lw, lw, rw])


FINGER_JOINT_NAMES = [
    "robot_j2s7s300_joint_finger_1",
    "robot_j2s7s300_joint_finger_tip_1",
    "robot_j2s7s300_joint_finger_2",
    "robot_j2s7s300_joint_finger_tip_2",
    "robot_j2s7s300_joint_finger_3",
    "robot_j2s7s300_joint_finger_tip_3",
]


def configure_gripper_fingers(sim, client, rid, fixed_angles=None):
    """Reset the gripper fingers to fixed angles and hold them there.

    Intercepts the motor torque commands so wheels (0..3) and arm (4..10) keep their
    normal torque control while fingers (11..16) are position-held.
    Returns the finger target angles (initial config angles if `fixed_angles` is None).
    """
    finger_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if client.getJointInfo(rid, j)[1].decode() in FINGER_JOINT_NAMES
    ]
    raw_init_angles = np.array(sim.GetInitMotorAngles())
    target_finger_angles = fixed_angles if fixed_angles is not None else raw_init_angles[11:17]
    for i, jid in enumerate(finger_joint_ids):
        client.resetJointState(rid, jid, targetValue=float(target_finger_angles[i]), targetVelocity=0.0)
    sim.ReceiveObservation()

    orig_set_torque = sim._SetMotorTorqueByIds

    def custom_set_torque(motor_ids, commands, index=0):
        orig_set_torque(motor_ids[:11], commands[:11], index=index)
        sim.pybullet_client.setJointMotorControlArray(
            bodyIndex=sim.bot[index].bot_pybullet,
            jointIndices=motor_ids[11:17],
            controlMode=sim.pybullet_client.POSITION_CONTROL,
            targetPositions=np.asarray(target_finger_angles).tolist(),
            forces=[5.0] * 6
        )

    sim._SetMotorTorqueByIds = custom_set_torque
    return target_finger_angles
