"""Shared 37-feature contract for teacher recording and learned placement inference.

Outputs: seven joint target offsets, six finger targets, learned release score.
The goal is expressed in the base yaw frame so world XY can be changed.
"""
import numpy as np


def placement_features(q, qd, ball, velocity, hand, goal_xy, release_z, yaw, elapsed):
    q, qd = np.asarray(q), np.asarray(qd)
    ball, hand = np.asarray(ball), np.asarray(hand)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    goal = np.r_[goal_xy, release_z]
    x = np.r_[q[4:11], qd[4:11], q[11:17], qd[11:17],
              rotation @ (goal-ball), rotation @ np.asarray(velocity),
              rotation @ (ball-hand), ball[2], min(float(elapsed), 30.0)]
    if x.shape != (37,) or not np.isfinite(x).all():
        raise ValueError('Placement observations must contain 37 finite features')
    return x.astype(np.float32)
