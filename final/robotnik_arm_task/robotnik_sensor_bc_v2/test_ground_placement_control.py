"""Deterministic motion-envelope tests; not a substitute for RoboEnv rollout."""
import unittest
import numpy as np
from ground_placement_control import BallCartesianLoweringPlanner


class GroundLoweringTests(unittest.TestCase):
    def make(self):
        return BallCartesianLoweringPlanner([2.972,3.465,.992],[3.,3.5,.25])

    def test_incremental_descent_not_one_shot_joint_ik(self):
        planner = self.make()
        target, step, ok = planner.update(46.29, [2.972,3.465,.992], [0,0,0])
        self.assertTrue(ok)
        self.assertAlmostEqual(target[2], .992)
        self.assertLessEqual(np.linalg.norm(step[:2]), .0120001)
        target, step, ok = planner.update(46.34,[2.974,3.467,.991],[0,0,0])
        self.assertTrue(ok)
        self.assertLess(target[2],.992)
        self.assertGreater(target[2],.25)
        self.assertLessEqual(abs(step[2]), .0080001)

    def test_pauses_descent_on_xy_drift_and_recovers(self):
        planner = self.make()
        planner.update(0, [2.972,3.465,.992], [0,0,0])
        first, _, _ = planner.update(.1,[2.977,3.475,.990],[0,0,0])
        desired, delta, descend = planner.update(.2,[3.05,3.69,.98],[0,0,0])
        self.assertFalse(descend)
        self.assertLessEqual(desired[2],first[2])
        self.assertAlmostEqual(desired[2],.98)
        self.assertLessEqual(np.linalg.norm(delta[:2]), .0120001)
        self.assertLessEqual(abs(delta[2]),.0080001)
        desired2,_,descend2=planner.update(.3,[3.01,3.51,.979],[0,0,0])
        self.assertTrue(descend2)
        self.assertLess(desired2[2], desired[2])

    def test_high_horizontal_speed_freezes_z(self):
        planner = self.make()
        planner.update(0,[2.972,3.465,.992],[0,0,0])
        desired, _, descend = planner.update(.1,[3.,3.5,.992],[0.3,0.0,0])
        self.assertFalse(descend)
        self.assertAlmostEqual(desired[2], .992)

    def test_no_command_below_release_height(self):
        planner = self.make()
        planner.update(0, [3.,3.5,.251],[0,0,0])
        goal,delta,_=planner.update(12,[3.,3.5,.25],[0,0,0])
        self.assertAlmostEqual(goal[2],.25)
        self.assertAlmostEqual(delta[2],0.0)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            BallCartesianLoweringPlanner([0,0,float('nan')],[0,0,.25])
        with self.assertRaises(ValueError):
            self.make().update(0,[1,2,3],[float('nan'),0,0])


if __name__ == '__main__':
    unittest.main()
