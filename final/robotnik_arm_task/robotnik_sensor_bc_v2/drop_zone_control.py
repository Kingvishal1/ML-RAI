"""Physical circular landing-zone geometry for Robotnik expert controller.

All logic is based on measured ball world XY; no object manipulation,
virtual attachment or predetermined trajectory is permitted.
"""
import numpy as np


class CircularDropZone:
    def __init__(self, centre_xy, radius=0.50, ball_radius=0.03,
                 release_margin=0.10):
        self.centre = np.asarray(centre_xy, dtype=float).reshape(2)
        self.radius = float(radius)
        self.ball_radius = float(ball_radius)
        self.release_margin = float(release_margin)
        if (not np.isfinite(self.centre).all() or
            not np.isfinite([self.radius, self.ball_radius, self.release_margin]).all() or
            self.radius <= self.ball_radius + self.release_margin + 0.01 or
            self.ball_radius <= 0 or self.release_margin < 0):
            raise ValueError('Invalid drop zone bounds and release margin')

    @property
    def release_radius(self):
        # Leave some room for the ball to drift naturally while falling.
        return self.radius - self.ball_radius - self.release_margin

    @property
    def landing_radius(self):
        # The entire 3D ball footprint must finish inside the visible circle.
        return self.radius - self.ball_radius

    def distance(self, ball_xy):
        pos = np.asarray(ball_xy, dtype=float).reshape(2)
        if not np.isfinite(pos).all():
            raise ValueError('Nonfinite measured ball XY')
        return float(np.linalg.norm(pos-self.centre))

    def can_release(self, ball_xy):
        return self.distance(ball_xy) <= self.release_radius

    def landed_inside(self, ball_xy):
        return self.distance(ball_xy) <= self.landing_radius

    def target_for_measured_ball(self, ball_xy):
        """Nearest admissible XY to actual held ball, not the fixed centre.

        If ball is already in the safe release zone, do not move it sideways.
        Otherwise request the nearest point slightly inside the safe zone.
        A high-level controller must still verify low motion speed/contact.
        """
        xy = np.asarray(ball_xy, dtype=float).reshape(2)
        distance = self.distance(xy)
        if distance <= self.release_radius:
            return xy.copy()
        return self.centre + (xy-self.centre) * (max(0.001,self.release_radius-0.03)/distance)


def draw_drop_zone(pybullet, centre, radius, z=0.004):
    """Optional single transparent disk, visual only; no collision shape."""
    visual = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_CYLINDER,
        radius=float(radius), length=0.006,
        rgbaColor=[0.10, 0.85, 0.22, 0.27])
    return pybullet.createMultiBody(
        baseMass=0.0, baseVisualShapeIndex=visual,
        baseCollisionShapeIndex=-1,
        basePosition=[float(centre[0]), float(centre[1]), float(z)])
