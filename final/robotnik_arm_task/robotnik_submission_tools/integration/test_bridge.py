import unittest
import numpy as np
from robotnik_navigation_bridge import (
    NavigationCommand, CollisionDecision, arbitrate_carry,
    bc_carry_observation, carry_action, wheel_speeds, arrived_at_goal)

class FakePolicy:
    def predict(self, obs):
        assert len(obs)==5
        return [0.02, 0.1]

class TestBridge(unittest.TestCase):
    def test_feature_contract(self):
        obs=bc_carry_observation([1,1],[2,1],[1,1],0,[0,0],0)
        np.testing.assert_allclose(obs,[1,0,0,0,0])
    def test_bc_and_collision_priority(self):
        a=carry_action(FakePolicy(),[1,0,0,0,0])
        self.assertEqual(arbitrate_carry(a,CollisionDecision()).source,'bc_carry')
        b=arbitrate_carry(a,CollisionDecision(active=True,linear_mps=0,angular_radps=.2))
        self.assertEqual((b.source,b.linear_mps),('collision_avoidance',0))
        stop=arbitrate_carry(a,CollisionDecision(active=True,stop=True,linear_mps=.02))
        self.assertEqual((stop.source,stop.linear_mps,stop.angular_radps),('safety_stop',0,0))
    def test_wheel_and_goal(self):
        w=wheel_speeds(NavigationCommand(.02,0,'bc'),.1,.5)
        np.testing.assert_allclose(w,[.2,.2,.2,.2])
        self.assertTrue(arrived_at_goal([3,3.5],[3,3.5]))
        self.assertFalse(arrived_at_goal([2.9,3.5],[3,3.5]))

if __name__ == '__main__': unittest.main()
