"""Bounded, feedback-driven base correction selection (not a prerecorded motion).

The scorer consumes candidate IK/FK residuals measured at the current base
pose and small candidate offsets along robot forward. It selects a small
forward/back wheel movement only when the predicted ground-placement arm
trajectory would otherwise be infeasible or significantly improved.
"""
import numpy as np


def select_base_adjustment(samples, feasible_residual=0.018, min_improvement=0.004):
    """Return (metres in heading direction, baseline, best score).

    samples maps signed offsets (m) -> finite errors (m). Zero MUST be probed.
    A correction is permitted only if there is a measurable reduction in IK
    error. At an already feasible pose, leave the base where it is.
    """
    if 0.0 not in samples:
        raise ValueError("Reachability scan needs a zero-offset baseline")
    if not samples or any(not np.isfinite(float(v)) or not np.isfinite(float(k))
                          for k,v in samples.items()):
        raise ValueError("Nonfinite reachability values")
    baseline = float(samples[0.0])
    if baseline <= feasible_residual:
        return 0.0, baseline, baseline
    shift, score = min(samples.items(), key=lambda kv: (float(kv[1]), abs(float(kv[0]))))
    if float(shift) == 0 or baseline - float(score) < min_improvement:
        return 0.0, baseline, float(score)
    return float(shift), baseline, float(score)


def wheel_command_for_displacement(distance_along_heading, speed_cap=0.016,
                                   kp=0.5, deadband=0.006, wheel_radius=0.11):
    """Feedback wheel speed (rad/s), zero when measured offset settled."""
    if not np.isfinite(distance_along_heading):
        raise ValueError("Nonfinite base error")
    if abs(distance_along_heading) <= deadband:
        return 0.0
    return float(np.clip(kp*distance_along_heading, -speed_cap, speed_cap)/wheel_radius)
