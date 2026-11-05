import unittest
from unittest.mock import MagicMock, patch

from config_runtime import DEFAULT_CONFIG, deep_merge, normalize_config


class TestConfigRuntime(unittest.TestCase):
    def test_partial_model_update_preserves_iou_and_weights(self):
        config = normalize_config(deep_merge(DEFAULT_CONFIG, {
            "models": {"optical": {"conf_threshold": 0.65}}
        }))
        self.assertEqual(config["models"]["optical"]["conf_threshold"], 0.65)
        self.assertEqual(config["models"]["optical"]["iou_threshold"], 0.45)
        self.assertEqual(config["models"]["optical"]["weights"], "models/optical/yolov8n.pt")
        self.assertEqual(config["models"]["thermal"]["iou_threshold"], 0.45)

    def test_missing_optional_keys_receive_defaults(self):
        config = normalize_config({"models": {"thermal": {"conf_threshold": 0.4}}})
        self.assertEqual(config["models"]["thermal"]["iou_threshold"], 0.45)
        self.assertTrue(config["streams"]["optical"]["enabled"])

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_config({"models": {"optical": {"iou_threshold": 1.2}}})

    @patch("main.AlertManager")
    @patch("main.IntrusionEngine")
    @patch("main.OpticalDetector")
    @patch("main.ThermalDetector")
    def test_system_partial_update_preserves_independent_model_settings(
        self, thermal_detector, optical_detector, intrusion_engine, alert_manager
    ):
        from main import RailwayIntrusionSystem

        system = RailwayIntrusionSystem(config={"streams": {"optical": {"source": ""}, "thermal": {"source": ""}}})
        system.update_config({"models": {"optical": {"conf_threshold": 0.7}}})
        self.assertEqual(system.config["models"]["optical"]["iou_threshold"], 0.45)
        self.assertEqual(system.config["models"]["thermal"]["conf_threshold"], 0.25)
        self.assertEqual(system.optical_det.conf, 0.7)
        self.assertEqual(system.thermal_det.iou, 0.45)

    @patch("main.AlertManager")
    @patch("main.IntrusionEngine")
    @patch("main.StreamManager")
    @patch("main.OpticalDetector")
    @patch("main.ThermalDetector")
    def test_optical_only_and_dual_stream_lifecycle(
        self, thermal_detector, optical_detector, stream_manager, intrusion_engine, alert_manager
    ):
        from main import RailwayIntrusionSystem

        stream_manager.return_value.get_status.return_value = {}
        optical_only = RailwayIntrusionSystem(config={
            "streams": {"optical": {"source": "camera.mp4"}, "thermal": {"source": ""}}
        })
        optical_only.start_processing()
        stream_manager.return_value.start.assert_called_with(enabled_streams=["optical"])
        self.assertTrue(optical_only.state["processing_active"])
        optical_only.stop_processing()
        stream_manager.return_value.stop.assert_called_once()

        stream_manager.reset_mock()
        dual_stream = RailwayIntrusionSystem(config={
            "streams": {"optical": {"source": "optical.mp4"}, "thermal": {"source": "thermal.mp4"}}
        })
        dual_stream.start_processing()
        stream_manager.return_value.start.assert_called_with(enabled_streams=["optical", "thermal"])
