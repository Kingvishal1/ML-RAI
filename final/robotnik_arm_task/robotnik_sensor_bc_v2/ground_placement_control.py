"""Conservative feedback planner for expert (not ML) ground lowering.

Commands desired *ball* motion in world XYZ. The caller converts the small
position correction to a PyBullet arm IK command using measured hand position.
Never edits finger actions, simulates object attachment, or bypasses release
verification. No PyBullet dependency so the motion limits can be unit-tested.
"""

import numpy as np


class BallCartesianLoweringPlanner:
    def __init__(self, start_ball_xyz, target_ball_xyz, vertical_speed=0.05,
                 xy_pause_error=0.055, max_xy_step=0.012, max_z_step=0.008):
        start = np.asarray(start_ball_xyz, dtype=float).reshape(3)
        target = np.asarray(target_ball_xyz, dtype=float).reshape(3)
        if not (np.isfinite(start).all() and np.isfinite(target).all()
                and start[2] >= target[2] > 0):
            raise ValueError("Invalid finite start/ground-release ball positions")
        if not all(np.isfinite(x) and x > 0 for x in (
                vertical_speed, xy_pause_error, max_xy_step, max_z_step)):
            raise ValueError("All planner speed and step limits must be positive")
        self.target = target.copy()
        self.command_z = float(start[2])
        self.vertical_speed = float(vertical_speed)
        self.xy_pause_error = float(xy_pause_error)
        self.max_xy_step = float(max_xy_step)
        self.max_z_step = float(max_z_step)
        self.last_time = None

    def update(self, sim_time, measured_ball_xyz, measured_ball_velocity):
        """Return (desired world XYZ, safe small XYZ correction, is_descending).

        The height profile is rate limited. XY drift larger than the pause
        tolerance freezes *further descent* and prioritizes XY correction.
        The planner never asks the gripper to open; safety decisions remain
        entirely with the caller's existing physical contact/release guards.
        """
        ball = np.asarray(measured_ball_xyz, dtype=float).reshape(3)
        velocity = np.asarray(measured_ball_velocity, dtype=float).reshape(3)
        t = float(sim_time)
        if not (np.isfinite(ball).all() and np.isfinite(velocity).all()
                and np.isfinite(t)):
            raise ValueError("Non-finite ball state/time in ground placement")
        if self.last_time is None:
            dt = 0.0
        else:
            dt = float(np.clip(t-self.last_time, 0.0, 0.20))
        self.last_time = t
        xy_error = float(np.linalg.norm(self.target[:2]-ball[:2]))
        # Do not drive Z down faster than the actual held ball can follow.
        self.command_z = max(self.target[2], min(self.command_z, ball[2]))
        descending = xy_error <= self.xy_pause_error and np.linalg.norm(velocity[:2]) <= 0.12
        if descending:
            self.command_z = max(self.target[2], self.command_z-self.vertical_speed*dt)
        desired_ball = np.array([self.target[0], self.target[1], self.command_z])
        error = desired_ball-ball
        planar = error[:2].copy()
        planar_norm = float(np.linalg.norm(planar))
        if planar_norm > self.max_xy_step:
            planar *= self.max_xy_step / planar_norm
        correction = np.array([
            planar[0], planar[1],
            float(np.clip(error[2], -self.max_z_step, self.max_z_step)),
        ])
        return desired_ball, correction, bool(descending)
