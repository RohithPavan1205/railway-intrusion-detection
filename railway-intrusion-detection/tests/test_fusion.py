import unittest
from fusion.late_fusion import LateFusion, calculate_iou

class TestFusion(unittest.TestCase):
    def setUp(self):
        config = {
            'fusion': {
                'iou_threshold': 0.3,
                'calibration_enabled': False,
                'homography_matrix': None
            }
        }
        self.fusion = LateFusion(config)

    def test_iou(self):
        boxA = [0, 0, 10, 10]
        boxB = [5, 5, 15, 15]
        iou = calculate_iou(boxA, boxB)
        self.assertAlmostEqual(iou, 25/175, places=4)

    def test_fuse_matching(self):
        opt_dets = [{'bbox': [0, 0, 10, 10], 'conf': 0.9}]
        thm_dets = [{'bbox': [1, 1, 11, 11], 'conf': 0.8}]
        
        fused = self.fusion.fuse(opt_dets, thm_dets)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]['sources'], ['optical', 'thermal'])
        self.assertAlmostEqual(fused[0]['conf'], 0.85)

    def test_fuse_no_match(self):
        opt_dets = [{'bbox': [0, 0, 10, 10], 'conf': 0.9}]
        thm_dets = [{'bbox': [100, 100, 110, 110], 'conf': 0.8}]
        
        fused = self.fusion.fuse(opt_dets, thm_dets)
        self.assertEqual(len(fused), 2)

if __name__ == "__main__":
    unittest.main()
