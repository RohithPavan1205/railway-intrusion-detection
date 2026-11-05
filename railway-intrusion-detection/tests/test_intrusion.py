import unittest
from intrusion.track_zone import TrackZone

class TestIntrusion(unittest.TestCase):
    def setUp(self):
        # Create a simple rectangle zone: [10, 10] to [50, 50]
        self.zone = TrackZone("TestZone", [[10, 10], [50, 10], [50, 50], [10, 50]])

    def test_inside(self):
        # Point clearly inside
        self.assertTrue(self.zone.is_inside([30, 30]))
        # Point clearly outside
        self.assertFalse(self.zone.is_inside([0, 0]))
        # Point on boundary
        self.assertTrue(self.zone.is_inside([10, 10]))

    def test_foot_point(self):
        # BBox: [x1, y1, x2, y2]
        bbox = [10, 10, 30, 40]
        foot = self.zone.get_foot_point(bbox)
        self.assertEqual(foot, [20, 40])

if __name__ == "__main__":
    unittest.main()
