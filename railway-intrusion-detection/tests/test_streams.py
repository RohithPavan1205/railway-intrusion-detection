import unittest
from unittest.mock import MagicMock, patch
import os
import numpy as np
from streams.video_capture import VideoCapture
from streams.stream_manager import StreamManager

class TestStreams(unittest.TestCase):
    def test_missing_video_file(self):
        # Path that definitely doesn't exist
        cap = VideoCapture("non_existent_video.mp4")
        self.assertTrue(cap.is_missing)
        self.assertFalse(cap.open())
        self.assertFalse(cap.read()[0])

    def test_stream_manager_missing_file(self):
        config = {
            'streams': {
                'optical': {'source': 'missing.mp4', 'width': 640, 'height': 480},
                'thermal': {'source': 'valid.mp4', 'width': 640, 'height': 480}
            }
        }
        # Mock VideoCapture to simulate one missing and one valid
        with patch('streams.stream_manager.VideoCapture') as MockCap:
            # Setup mocks
            mock_opt = MagicMock()
            mock_opt.open.return_value = False
            mock_opt.is_missing = True
            
            mock_thm = MagicMock()
            mock_thm.open.return_value = True
            mock_thm.is_missing = False
            mock_thm.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            
            MockCap.side_effect = [mock_opt, mock_thm]
            
            sm = StreamManager(config)
            sm.start()
            
            # Give it a moment to initialize
            import time
            time.sleep(0.2)
            
            status = sm.get_status()
            self.assertEqual(status['optical'], "Missing")
            self.assertEqual(status['thermal'], "Connected")
            
            sm.stop()

if __name__ == "__main__":
    unittest.main()
