"""Pure-Python command bridge for merging BC carrying with a teammate's obstacle avoidance.

No simulator stepping, no motor writes, no file-global state. A single orchestrator
calls this once per physics/control step and applies the returned command.
"""
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


@dataclass(frozen=True)
class NavigationCommand:
    linear_mps: float
    angular_radps: float
    source: str
    reason: str = ""


@dataclass(frozen=True)
class CollisionDecision:
    """Teammate's avoidance output. A veto has priority over carrying.

    An avoidance command is specified as a safe (v,w) for this step; `stop=True`
    means an emergency stop and overrides even a proposed avoidance command.
    `active=False` allows the BC carry command to pass through.
    """
    active: bool = False
    stop: bool = False
    linear_mps: float = 0.0
    angular_radps: float = 0.0
    reason: str = ""


def arbitrate_carry(carry_command: NavigationCommand,
                    collision: CollisionDecision,
                    max_linear_mps: float = 0.025,
                    max_angular_radps: float = 0.4) -> NavigationCommand:
    """Safety priority: emergency stop > collision avoidance > BC carry.

    Must be called BEFORE any wheel conversion or motor command, not afterwards.
    """
    if collision.stop:
        return NavigationCommand(0.0, 0.0, "safety_stop", collision.reason)
    if collision.active:
        candidate = NavigationCommand(collision.linear_mps,
                                      collision.angular_radps,
                                      "collision_avoidance", collision.reason)
    else:
        candidate = carry_command
    if not (math.isfinite(candidate.linear_mps) and
            math.isfinite(candidate.angular_radps)):
        return NavigationCommand(0.0, 0.0, "safety_stop", "nonfinite navigation command")
    return NavigationCommand(float(np.clip(candidate.linear_mps, -max_linear_mps, max_linear_mps)),
                             float(np.clip(candidate.angular_radps, -max_angular_radps, max_angular_radps)),
                             candidate.source, candidate.reason)


def bc_carry_observation(ball_xy, goal_xy, carry_origin_xy,
                         base_yaw, base_velocity_xy, base_yaw_rate) -> np.ndarray:
    """Exactly the five expert-training features, in the same order.

    [remaining distance, lateral path error, heading error,
     forward speed, yaw rate]. All distances in metres, angles in radians.
    """
    ball = np.asarray(ball_xy, dtype=float).reshape(2)
    goal = np.asarray(goal_xy, dtype=float).reshape(2)
    origin = np.asarray(carry_origin_xy, dtype=float).reshape(2)
    velocity = np.asarray(base_velocity_xy, dtype=float).reshape(2)
    offset = goal - ball
    remaining = float(np.linalg.norm(offset))
    desired_heading = float(math.atan2(offset[1], offset[0])) if remaining > 1e-6 else base_yaw
    heading_error = math.atan2(math.sin(desired_heading - base_yaw),
                               math.cos(desired_heading - base_yaw))
    route = goal - origin
    unit = route / max(float(np.linalg.norm(route)), 1e-8)
    progress = ball - origin
    lateral = float(unit[0] * progress[1] - unit[1] * progress[0])
    forward = float(np.dot(velocity, [math.cos(base_yaw), math.sin(base_yaw)]))
    observation = np.array([remaining, lateral, heading_error, forward, base_yaw_rate], dtype=float)
    if not np.isfinite(observation).all():
        raise ValueError("Nonfinite carry observation")
    return observation


def carry_action(policy, observation, max_linear_mps=0.025,
                 max_angular_radps=0.4) -> NavigationCommand:
    """BC-only carry; policy.predict() is the sole source of the proposed action."""
    action = np.asarray(policy.predict(observation), dtype=float).reshape(2)
    if not np.isfinite(action).all():
        raise ValueError("Nonfinite BC action")
    return NavigationCommand(float(np.clip(action[0], 0, max_linear_mps)),
                             float(np.clip(action[1], -max_angular_radps, max_angular_radps)),
                             "bc_carry")


def wheel_speeds(command: NavigationCommand, wheel_radius_m, track_width_m):
    """Return 4 wheel angular speeds in existing Robotnik order [R,L,L,R]."""
    if wheel_radius_m <= 0 or track_width_m <= 0:
        raise ValueError("Wheel geometry must be positive")
    right = (command.linear_mps + command.angular_radps * track_width_m / 2) / wheel_radius_m
    left = (command.linear_mps - command.angular_radps * track_width_m / 2) / wheel_radius_m
    return np.array([right, left, left, right], dtype=float)


def arrived_at_goal(ball_xy, goal_xy, tolerance_m=0.045) -> bool:
    """Arrival guard should only be evaluated in the carrying state."""
    return float(np.linalg.norm(np.asarray(ball_xy)-np.asarray(goal_xy))) <= tolerance_m
