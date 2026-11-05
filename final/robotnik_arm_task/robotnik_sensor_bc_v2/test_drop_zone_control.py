import unittest
import numpy as np
from drop_zone_control import CircularDropZone


class TestCircularDropZone(unittest.TestCase):
    def setUp(self):
        self.z = CircularDropZone([3.0,3.5],radius=0.50,ball_radius=0.03,release_margin=0.10)

    def test_usable_radii_and_true_edge(self):
        self.assertAlmostEqual(self.z.release_radius,.37)
        self.assertAlmostEqual(self.z.landing_radius,.47)
        self.assertTrue(self.z.can_release([3.36,3.5]))
        self.assertFalse(self.z.can_release([3.375,3.5]))
        self.assertTrue(self.z.landed_inside([3.469,3.5]))
        self.assertFalse(self.z.landed_inside([3.471,3.5]))

    def test_no_point_chase_inside_region(self):
        x=np.array([3.14,3.20])
        np.testing.assert_array_equal(self.z.target_for_measured_ball(x),x)
        self.assertFalse(np.allclose(self.z.target_for_measured_ball(x),self.z.centre))

    def test_projects_back_inside_without_snapping_to_centre(self):
        x=np.array([3.9,3.5])
        target=self.z.target_for_measured_ball(x)
        self.assertGreater(target[0],3.2)
        self.assertLess(self.z.distance(target), self.z.release_radius)

    def test_preserves_ball_diameter_at_edge(self):
        self.assertTrue(self.z.landed_inside([3.0,3.9]))
        self.assertFalse(self.z.landed_inside([3.0,4.0]))

    def test_rejects_invalid_bounds(self):
        for radius,margin in [(0.02,0.0),(0.10,0.09),(-1,0.0)]:
            with self.assertRaises(ValueError):
                CircularDropZone([3.,3.],radius=radius,release_margin=margin)
        with self.assertRaises(ValueError):
            self.z.distance([float('nan'),3.])

if __name__=='__main__': unittest.main()
