import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dashboard.uploads import inspect_video, pairing_warnings


class TestUploads(unittest.TestCase):
    def test_rejects_unsupported_extension_without_opening_video(self):
        result = inspect_video("bad-input.txt")
        self.assertFalse(result["valid"])
        self.assertIn("Supported", result["error"])

    @patch("dashboard.uploads.cv2.VideoCapture")
    def test_validates_metadata_from_readable_video(self, capture_class):
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.side_effect = [24.0, 240, 1280, 720]
        capture_class.return_value = capture
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            result = inspect_video(video.name)
        self.assertTrue(result["valid"])
        self.assertEqual(result["duration_seconds"], 10.0)
        capture.release.assert_called_once()

    def test_pairing_reports_single_stream_and_mismatch(self):
        warnings, paired = pairing_warnings({"fps": 30, "duration_seconds": 10, "frame_count": 300}, None)
        self.assertFalse(paired)
        self.assertIn("Single-stream", warnings[0])
        warnings, paired = pairing_warnings(
            {"fps": 30, "duration_seconds": 10, "frame_count": 300},
            {"fps": 24, "duration_seconds": 15, "frame_count": 360},
        )
        self.assertFalse(paired)
        self.assertEqual(len(warnings), 3)
