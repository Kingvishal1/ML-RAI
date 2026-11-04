"""Pure reach decision + mock IK scan tests. Does not claim physics success."""
import ast
from pathlib import Path
import unittest
import numpy as np
from reachability_feedback import select_base_adjustment, wheel_command_for_displacement

class ReachDecisionTests(unittest.TestCase):
    def test_adaptive_backwards_when_better(self):
        shift, baseline, best = select_base_adjustment({-0.065:.012, 0.0:.044, +0.065:.051})
        self.assertAlmostEqual(shift, -0.065)
        self.assertAlmostEqual(baseline, .044)
        self.assertAlmostEqual(best, .012)

    def test_adaptive_forward_when_better(self):
        shift, _, _ = select_base_adjustment({-0.065:.080, 0.0:.048, +0.065:.013})
        self.assertGreater(shift, 0)

    def test_no_motion_when_reachable(self):
        self.assertEqual(select_base_adjustment({-0.065:.0, 0.0:.01, .065:.0})[0], 0)

    def test_no_motion_without_real_improvement(self):
        self.assertEqual(select_base_adjustment({-.065:.035, 0.0:.038, .065:.0345})[0], 0)

    def test_closed_loop_wheel_speed(self):
        self.assertLess(wheel_command_for_displacement(-.05), 0)
        self.assertGreater(wheel_command_for_displacement(+.05), 0)
        self.assertEqual(wheel_command_for_displacement(+.002), 0)
        self.assertLessEqual(abs(wheel_command_for_displacement(10)), .016/.11 + 1e-9)
        with self.assertRaises(ValueError): wheel_command_for_displacement(float('nan'))

class FakeBulletKinematics:
    def __init__(self):
        self.base = np.zeros(3)
        self.ori = (0.,0.,0.,1.)
        self.q = np.zeros(7)
        self.saved = None
    def saveState(self):
        self.saved=(self.base.copy(),self.q.copy())
        return 1
    def restoreState(self, idx):
        assert idx==1
        self.base,self.q=[a.copy() for a in self.saved]
    def removeState(self, idx): pass
    def getBasePositionAndOrientation(self, body):
        return self.base.tolist(),self.ori
    def resetBasePositionAndOrientation(self,body,pos,ori):
        self.base=np.array(pos,dtype=float)
    def resetJointState(self,body,j,targetValue):
        self.q[j]=targetValue
    def getLinkState(self,body,link,computeForwardKinematics=True):
        pos=self.base+self.q[:3]
        return [None,None,None,None,pos.tolist(),self.ori]


def toy_ik(pb,robot_id,hand_link,arm_ids,position,orientation):
    # A toy workspace with an underreach/overreach gap in forward direction.
    relative=np.array(position)-pb.base
    relative[0]=np.clip(relative[0], .20, .60)
    return np.r_[relative,np.zeros(4)]

class ProbeTests(unittest.TestCase):
    def test_probe_restores_robot_and_finds_better_base(self):
        root=Path(__file__).parent/'robotnik_expert_controller.py'
        tree=ast.parse(root.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                and n.name=='probe_ground_reachability')
        probe_ns={'np':np,'solve_arm_ik':toy_ik}
        exec(compile(ast.Module(body=[fn], type_ignores=[]),'<probe>', 'exec'),probe_ns)
        probe=probe_ns['probe_ground_reachability']
        pb=FakeBulletKinematics()
        samples=probe(pb,1,6,list(range(7)),[.60,0,.9],[0,0,0],
                      [.65,0,.25],[0,0,0,1],0.,offsets=(-.065,0.,.065))
        # Target +.65 is unreachable from base 0 but possible after advancing .065.
        self.assertLess(samples[.065],samples[0.])
        self.assertLess(samples[.065],samples[-.065])
        self.assertTrue(np.allclose(pb.base,[0,0,0]))
        self.assertTrue(np.allclose(pb.q,[0]*7))

if __name__=='__main__':unittest.main()
