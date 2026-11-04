import numpy as np
import time
import os
from dataclasses import dataclass
from itertools import permutations

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)


# ==============================================================================
# TARGET SPECIFICATION
# ==============================================================================
TARGET_EE_POS = np.array([0.0, 1.1, 0.5])
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
BASE_NAV_TIMEOUT = 25.0
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
REACH_TOLERANCE = 0.05
MAX_TARGETS = 5


# ==============================================================================
# TORQUE NOISE (DART-style data collection)
# ==============================================================================
@dataclass
class TorqueNoiseParams:
    """Random perturbations added to the arm torque command during data collection.

    The sim executes the perturbed torque but the clean expert torque is logged, so a policy
    trained on the data sees the expert recovering from pushes (braking near the target)
    without learning the noise itself.
    """
    std: float = 2.0               # N*m, per-joint std of an active noise segment
    active_prob: float = 0.5       # chance a segment is noisy; quiet segments let the arm settle
    kick_prob: float = 0.1         # chance an active segment is a larger kick
    kick_scale: float = 3.0        # kick std = std * kick_scale
    min_duration: float = 0.05     # s, the noise value is held constant for one segment
    max_duration: float = 0.3      # s


class TorqueNoise:
    """Piecewise-constant random torque: a new value is drawn at the start of each segment."""

    def __init__(self, params, rng):
        self.params = params
        self.rng = rng
        self.value = np.zeros(7)
        self.segment_end = 0.0

    def sample(self, current_time):
        if current_time >= self.segment_end:
            p_ = self.params
            self.segment_end = current_time + self.rng.uniform(p_.min_duration, p_.max_duration)
            if self.rng.random() < p_.active_prob:
                scale = p_.std * (p_.kick_scale if self.rng.random() < p_.kick_prob else 1.0)
                self.value = self.rng.normal(0.0, scale, 7)
            else:
                self.value = np.zeros(7)
        return self.value


def setup_simulation():
    """Initialise the PyBullet simulation, configure friction and motor gains.

    Returns (sim, client, rid).
    """
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet

    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0],
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    return sim, client, rid, conf_file_name, cur_dir


def setup_kinematics(sim, conf_file_name, cur_dir):
    """Build the Pinocchio kinematic/dynamic model and compute initial EE pose.

    Returns (dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles,
             init_ee_pos, lower_limits, upper_limits).
    """
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())

    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    lower_limits, upper_limits = sim.GetBotJointsLimit()

    return dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, lower_limits, upper_limits


def add_target_visual(client, target_pos):
    """Draw a red sphere at the target position in the PyBullet viewer."""
    pos_list = target_pos.tolist()
    visual_shape_id = client.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85],
    )
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=pos_list)
    client.addUserDebugPoints(pointPositions=[pos_list], pointColorsRGB=[[1.0, 0.0, 0.0]], pointSize=10)


def compute_base_goal(target_ee_pos):
    """Compute the base standoff pose (position + bearing) for a given target.

    Returns (base_goal_pos, base_goal_bearing).
    """
    standoff = float(np.interp(target_ee_pos[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = target_ee_pos[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    return base_goal_pos, target_bearing


from dataclasses import dataclass, field


@dataclass
class TorqueNoiseParams:
    """Configuration for DART-style torque perturbation during data collection."""
    std: float = 1.0              # base noise std (N·m per joint)
    active_prob: float = 0.5      # probability each segment is noisy vs quiet
    kick_prob: float = 0.1        # probability an active segment is a larger kick
    kick_scale: float = 3.0       # kick std = kick_scale * std
    segment_steps: int = 50       # steps per noise segment


class TorqueNoise:
    """Generates per-step torque perturbation vectors for 7 arm joints."""

    def __init__(self, params: TorqueNoiseParams, rng=None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self._steps_in_segment = 0
        self._current_noise = np.zeros(7)
        self._active = False
        self._new_segment()

    def _new_segment(self):
        self._steps_in_segment = 0
        self._active = self.rng.random() < self.params.active_prob
        if self._active:
            scale = self.params.std
            if self.rng.random() < self.params.kick_prob:
                scale *= self.params.kick_scale
            self._current_noise = self.rng.normal(0, scale, size=7)
        else:
            self._current_noise = np.zeros(7)

    def sample(self):
        """Return the noise vector for this step, advancing the segment clock."""
        self._steps_in_segment += 1
        if self._steps_in_segment >= self.params.segment_steps:
            self._new_segment()
        return self._current_noise


def run_simulation_loop(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                        init_joint_angles, target_finger_angles,
                        lower_limits, upper_limits, target_ee_pos=None,
                        torque_noise=None, hold_time=0.0):
    """Run the main control loop: base navigation then arm reaching.

    torque_noise: optional TorqueNoise. While the arm is reaching, its value is added to the
        torque sent to the sim; the logged torque_arm stays the clean expert command.
    hold_time: s to keep running after the target is first reached (0 = stop on reach).

    Returns a dict of logged data arrays.
    """
    if target_ee_pos is None:
        target_ee_pos = TARGET_EE_POS

    time_step = sim.GetTimeStep()
    current_time = 0.0

    base_goal_pos, base_goal_bearing = compute_base_goal(target_ee_pos)
    base_state = BaseNavState()
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])

    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
    cmd = MotorCommands()
    reach_time = None

    # Data logs
    time_all, ee_pos_all, ee_error_all = [], [], []
    base_pos_all, q_mes_all, qd_mes_all = [], [], []
    q_d_all, qd_d_all = [], []
    torque_arm_all, wheel_cmds_all, base_bearing_all, target_ee_pos_all = [], [], [], []
    base_lin_vel_all, base_ang_vel_all = [], []
    tau_gravity_all = []
    torque_applied_all = []

    while True:
        # ---- Sense ----
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        base_lin_vel = np.array(sim.GetBaseLinVelocity())
        base_ang_vel = np.array(sim.GetBaseAngVelocity())

        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = target_ee_pos - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        # ---- Gravity compensation (for data logging) ----
        tau_g_full = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        tau_gravity_arm = tau_g_full[6:13].copy()  # 7 arm joints

        # ---- Base control ----
        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params,
        )
        base_navigation_done = base_state.done

        # ---- Arm control ----
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target_ee_pos, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=base_navigation_done,
        )

        # ---- Torque noise (executed only, never logged as the label) ----
        torque_applied = torque_arm
        if torque_noise is not None and base_navigation_done:
            torque_applied = np.clip(torque_arm + torque_noise.sample(current_time),
                                     -arm_params.max_torque, arm_params.max_torque)

        # ---- Send commands ----
        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_applied, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        # ---- Optional visualisation ----
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # ---- Exit on 'q' key ----
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # ---- Stop criteria ----
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, base_navigation_done)
        if hold_time > 0:
            if reach_time is None and stop_message and stop_message.startswith("Target reached"):
                reach_time = current_time
                print(f"{stop_message} Holding for {hold_time:.1f}s more...")
            if reach_time is not None:
                # After the first reach only the hold window and the step cap end the run
                if current_time - reach_time >= hold_time:
                    stop_message = (f"Hold window over at t={current_time:.2f}s "
                                    f"(EE error {ee_dist_error:.4f} m)")
                elif not (stop_message or "").startswith("Reached step cap"):
                    stop_message = None
        if stop_message:
            print(stop_message)
            break

        # ---- Log data ----
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(np.concatenate((q_mes[:4], q_des_arm, target_finger_angles)))
        qd_d_all.append(np.concatenate((wheel_cmds, qd_des_arm, np.zeros(6))))
        torque_arm_all.append(torque_arm)
        torque_applied_all.append(torque_applied)
        wheel_cmds_all.append(wheel_cmds)
        base_bearing_all.append(base_bearing)
        target_ee_pos_all.append(target_ee_pos.copy())
        base_lin_vel_all.append(base_lin_vel)
        base_ang_vel_all.append(base_ang_vel)
        tau_gravity_all.append(tau_gravity_arm)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | "
                  f"EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | "
                  f"yaw {base_bearing:.2f} phase {base_state.phase} nav_done {base_navigation_done}")

    return {
        "time": np.array(time_all),
        "ee_pos": np.array(ee_pos_all),
        "ee_error": np.array(ee_error_all),
        "base_pos": np.array(base_pos_all),
        "q_mes": q_mes_all,
        "qd_mes": qd_mes_all,
        "q_d": q_d_all,
        "qd_d": qd_d_all,
        "torque_arm": np.array(torque_arm_all),
        "torque_applied": np.array(torque_applied_all),
        "reach_time": reach_time,
        "wheel_cmds": np.array(wheel_cmds_all),
        "base_bearing": np.array(base_bearing_all),
        "target_ee_pos": np.array(target_ee_pos_all),
        "base_lin_vel": np.array(base_lin_vel_all),
        "base_ang_vel": np.array(base_ang_vel_all),
        "tau_gravity": np.array(tau_gravity_all),
        "final_base_pos": base_pos,
        "final_ee_pos": ee_pos,
        "final_q_mes": q_mes,
        "final_time": current_time,
        "target_finger_angles": target_finger_angles,
    }


SELECTOR_OBS_DIM = 3  # per-candidate: (in_base_x, in_base_y, z)
SELECTOR_INPUT_DIM = MAX_TARGETS * SELECTOR_OBS_DIM + MAX_TARGETS  # 5*3 + 5 = 20


def target_to_base_frame(base_pos, base_bearing, target_pos):
    """Express a target's position relative to the robot's base frame.

    Returns (3,): (forward, lateral, z).  XY are rotated into the robot's
    heading frame; Z is kept absolute (robot doesn't move vertically).
    """
    c, s = np.cos(base_bearing), np.sin(base_bearing)
    dx = target_pos[0] - base_pos[0]
    dy = target_pos[1] - base_pos[1]
    return np.array([c * dx + s * dy,      # forward
                     -s * dx + c * dy,     # lateral
                     target_pos[2]])       # z


def build_selector_candidates(base_pos, base_bearing, target_positions, remaining_indices):
    """Build per-candidate observations for the target selector.

    Args:
        base_pos: (3,) robot base position
        base_bearing: scalar, robot heading (yaw)
        target_positions: (N, 3) all target positions (world frame)
        remaining_indices: iterable of valid target slot indices
    Returns:
        candidates: dict mapping slot_index -> (3,) observation in base frame
    """
    candidates = {}
    for idx in remaining_indices:
        candidates[idx] = target_to_base_frame(base_pos, base_bearing, target_positions[idx])
    return candidates


def build_selector_scene(base_pos, base_bearing, target_positions, remaining_indices):
    """Build a fixed-size scene observation for the target selector.

    Packs all remaining candidates into MAX_TARGETS slots (sorted by slot index)
    with a validity mask.  Empty slots are zero-padded.

    Args:
        base_pos: (3,) robot base position
        base_bearing: scalar, robot heading (yaw)
        target_positions: (N, 3) all target positions (world frame)
        remaining_indices: iterable of valid target slot indices
    Returns:
        obs: (SELECTOR_INPUT_DIM,) = (MAX_TARGETS*3 + MAX_TARGETS,) flattened
        slot_order: list of slot indices in the order they appear in obs
    """
    remaining = sorted(remaining_indices)
    obs_block = np.zeros(MAX_TARGETS * SELECTOR_OBS_DIM)
    mask = np.zeros(MAX_TARGETS)
    slot_order = []

    for i, idx in enumerate(remaining):
        bf = target_to_base_frame(base_pos, base_bearing, target_positions[idx])
        obs_block[i * SELECTOR_OBS_DIM:(i + 1) * SELECTOR_OBS_DIM] = bf
        mask[i] = 1.0
        slot_order.append(idx)

    return np.concatenate([obs_block, mask]), slot_order


def nearest_neighbor_expert(base_pos, target_positions, remaining_indices):
    """Pick the nearest remaining target by XY distance to the robot.

    Args:
        base_pos: (3,) robot base position
        target_positions: (N, 3) all target positions
        remaining_indices: set of valid target slot indices
    Returns:
        int, index of the chosen target slot
    """
    best_idx = None
    best_dist = np.inf
    for idx in remaining_indices:
        dist = np.linalg.norm(base_pos[:2] - target_positions[idx, :2])
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def shortest_path_expert(base_pos, target_positions, remaining_indices):
    """Solve TSP over remaining targets by brute-force to minimise total XY distance.

    Starting from base_pos, evaluate every permutation of remaining targets and
    return the full visit order (list) plus the first target to visit.

    Args:
        base_pos: (3,) robot base position
        target_positions: (N, 3) all target positions
        remaining_indices: set of valid target slot indices
    Returns:
        chosen_idx: int, the first target to visit in the optimal order
        full_order: list[int], complete optimal visit sequence
    """
    remaining = list(remaining_indices)
    if len(remaining) == 1:
        return remaining[0], remaining

    best_dist = np.inf
    best_order = None
    for perm in permutations(remaining):
        d = np.linalg.norm(base_pos[:2] - target_positions[perm[0], :2])
        for i in range(len(perm) - 1):
            d += np.linalg.norm(target_positions[perm[i], :2] - target_positions[perm[i + 1], :2])
        if d < best_dist:
            best_dist = d
            best_order = list(perm)

    return best_order[0], best_order


def run_multi_target_loop(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                          init_joint_angles, target_finger_angles,
                          lower_limits, upper_limits,
                          target_ee_positions,
                          torque_noise_factory=None,
                          per_target_timeout=None):
    """Run an expert multi-target episode: visit each target in nearest-neighbor order.

    Args:
        sim, dyn_model, ...: same as run_simulation_loop
        target_ee_positions: list of target (x,y,z) arrays (any length)
        torque_noise_factory: callable() -> TorqueNoise or None, called per target
        per_target_timeout: max sim-time per target (seconds). None = use default max_steps.

    Returns:
        combined_data: dict with concatenated trajectory data from all targets
        selector_samples: list of dicts, one per decision point.  Each dict has:
            "candidates": dict mapping slot_idx -> (3,) base-frame observation
            "chosen_idx": int, the expert's pick (key into candidates)
            "episode_time": float
        per_target_results: list of dicts with per-target outcomes
    """
    n_targets = len(target_ee_positions)
    assert n_targets >= 1

    all_targets = np.array(target_ee_positions)  # (N, 3)

    remaining = set(range(n_targets))
    selector_samples = []
    per_target_results = []
    all_target_data = []
    episode_time = 0.0

    while remaining:
        # --- Decision point: expert picks nearest target ---
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        candidates = build_selector_candidates(base_pos, base_bearing, all_targets, remaining)
        chosen_idx, _ = shortest_path_expert(base_pos, all_targets, remaining)

        selector_samples.append({
            "candidates": candidates,
            "chosen_idx": chosen_idx,
            "episode_time": episode_time,
        })

        current_target = all_targets[chosen_idx]
        remaining.remove(chosen_idx)
        print(f"\n>>> Multi-target: visiting slot {chosen_idx} "
              f"({current_target}) | {len(remaining)} remaining")

        # --- Run single-target loop for this target ---
        noise = torque_noise_factory() if torque_noise_factory else None

        # Override max_steps for per-target timeout if given
        old_max_steps = max_steps
        if per_target_timeout is not None:
            time_step = sim.GetTimeStep()
            os.environ["ROBOENV_MAX_STEPS"] = str(int(per_target_timeout / time_step))

        data = run_simulation_loop(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            np.array(sim.GetMotorAngles(0)), target_finger_angles,
            lower_limits, upper_limits,
            target_ee_pos=current_target,
            torque_noise=noise,
        )

        if per_target_timeout is not None:
            os.environ["ROBOENV_MAX_STEPS"] = str(old_max_steps)

        target_time = data["final_time"]
        episode_time += target_time
        final_err = np.linalg.norm(current_target - data["final_ee_pos"])
        reached = final_err < REACH_TOLERANCE

        per_target_results.append({
            "slot": chosen_idx,
            "target": current_target.tolist(),
            "reached": reached,
            "final_err": final_err,
            "time": target_time,
        })
        all_target_data.append(data)
        print(f"    Slot {chosen_idx}: err={final_err:.4f}m  reached={reached}  t={target_time:.2f}s")

    # --- Combine trajectory data across targets ---
    combined_data = _combine_trajectory_data(all_target_data)
    combined_data["episode_time"] = episode_time
    combined_data["per_target_results"] = per_target_results

    return combined_data, selector_samples, per_target_results


def _combine_trajectory_data(data_list):
    """Concatenate trajectory data dicts from multiple single-target runs."""
    if not data_list:
        return {}
    combined = {}
    # Arrays that can be concatenated along axis 0
    array_keys = [
        "time", "ee_pos", "ee_error", "base_pos", "torque_arm", "torque_applied",
        "wheel_cmds", "base_bearing", "target_ee_pos", "base_lin_vel",
        "base_ang_vel", "tau_gravity",
    ]
    # List-of-array keys (q_mes, qd_mes, q_d, qd_d)
    list_keys = ["q_mes", "qd_mes", "q_d", "qd_d"]

    for key in array_keys:
        arrays = [d[key] for d in data_list if key in d and len(d[key]) > 0]
        if arrays:
            # Offset time arrays so they are monotonically increasing
            if key == "time":
                offset = 0.0
                adjusted = []
                for arr in arrays:
                    adjusted.append(arr + offset)
                    offset += arr[-1] + (arr[1] - arr[0] if len(arr) > 1 else 0.001)
                combined[key] = np.concatenate(adjusted)
            else:
                combined[key] = np.concatenate(arrays)

    for key in list_keys:
        lists = [d[key] for d in data_list if key in d and len(d[key]) > 0]
        if lists:
            combined[key] = sum(lists, [])

    # Use the last run's final values
    last = data_list[-1]
    combined["final_base_pos"] = last["final_base_pos"]
    combined["final_ee_pos"] = last["final_ee_pos"]
    combined["final_q_mes"] = last["final_q_mes"]
    combined["final_time"] = sum(d["final_time"] for d in data_list)
    combined["target_finger_angles"] = last["target_finger_angles"]
    combined["reach_time"] = last.get("reach_time")

    return combined


def print_summary(data):
    """Print final simulation results."""
    target_finger_angles = data["target_finger_angles"]
    q_mes = data["final_q_mes"]
    print("==================================================================")
    print(f"Final Base Position          : {np.round(data['final_base_pos'][:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(data['final_ee_pos'], 4)}")
    print(f"Target End-Effector Position : {TARGET_EE_POS}")
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(TARGET_EE_POS - data['final_ee_pos']):.4f}")
    print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")


def plot_results(data):
    """Show EE tracking, error, and joint profile plots."""
    time_all = data["time"]
    ee_pos_all = data["ee_pos"]
    ee_error_all = data["ee_error"]

    # 1. Cartesian EE position tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=TARGET_EE_POS[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in data["q_mes"]], label=f'Measured J{i+1}')
        plt.plot(time_all, [q[4 + i] for q in data["q_d"]], label=f'Desired J{i+1}', linestyle='--')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()


def main():
    # Setup
    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    add_target_visual(client, TARGET_EE_POS)

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    print("==================================================================")
    print(f"Target Cartesian EE Position : {TARGET_EE_POS}")
    print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Cartesian Error (m)  : {np.linalg.norm(TARGET_EE_POS - init_ee_pos):.4f}")
    print("==================================================================")

    # Run
    data = run_simulation_loop(
        sim, dyn_model, ee_frame_id, shoulder_frame_id,
        init_joint_angles, target_finger_angles,
        lower_limits, upper_limits,
    )


    # Report
    print_summary(data)

    if headless:
        print(f"Headless run finished at t={data['final_time']:.2f}s")
        return

    plot_results(data)


if __name__ == '__main__':
    main()
