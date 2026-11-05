import logging
import time

import cv2
import yaml

from alerts.alert_manager import AlertManager
from config_runtime import deep_merge, normalize_config
from detection.optical_detector import OpticalDetector
from detection.thermal_detector import ThermalDetector
from fusion.late_fusion import LateFusion
from intrusion.intrusion_engine import IntrusionEngine
from intrusion.track_zone import TrackZone
from streams.stream_manager import StreamManager
from tracking.person_tracker import PersonTracker

logger = logging.getLogger(__name__)


class RailwayIntrusionSystem:
    """Owns the detection pipeline for one Streamlit browser session."""

    def __init__(self, config_path="config/settings.yaml", config=None):
        with open(config_path, "r") as config_file:
            file_config = yaml.safe_load(config_file) or {}
        self.config = normalize_config(deep_merge(file_config, config or {}))
        self.streams = StreamManager(self.config)
        self.optical_det = OpticalDetector(self.config)
        self.thermal_det = ThermalDetector(self.config)
        self.fusion = LateFusion(self.config)
        self.tracker = PersonTracker()
        self.zones = self._make_zones()
        self.intrusion_engine = IntrusionEngine(self.config, self.zones)
        self.alert_manager = AlertManager(self.config)
        self.state = {
            "optical_frame": None, "thermal_frame": None, "fused_frame": None,
            "tracks": [], "events": [], "status": self.streams.get_status(),
            "fps": 0.0, "latency": 0.0, "processing_active": False,
            "processed_frames": 0, "last_error": None,
        }

    def _make_zones(self):
        return [TrackZone(zone["name"], zone["polygon"])
                for zone in self.config["intrusion"]["restricted_zones"]]

    def update_config(self, changes):
        candidate = normalize_config(deep_merge(self.config, changes))
        self.config = candidate
        self.streams.config = candidate
        self.optical_det.conf = candidate["models"]["optical"]["conf_threshold"]
        self.optical_det.iou = candidate["models"]["optical"]["iou_threshold"]
        self.thermal_det.conf = candidate["models"]["thermal"]["conf_threshold"]
        self.thermal_det.iou = candidate["models"]["thermal"]["iou_threshold"]
        self.fusion.iou_threshold = candidate["fusion"]["iou_threshold"]
        self.fusion.calibration_enabled = candidate["fusion"]["calibration_enabled"]
        self.fusion.homography = candidate["fusion"]["homography_matrix"]
        self.zones = self._make_zones()
        self.intrusion_engine.zones = self.zones
        self.intrusion_engine.temporal_threshold = candidate["intrusion"]["temporal_confirmation_frames"]

    def update_stream_source(self, stream_name, source_path):
        if stream_name not in self.config["streams"]:
            raise ValueError(f"Unknown stream: {stream_name}")
        if self.state["processing_active"]:
            raise RuntimeError("Stop detection before changing a video source.")
        self.config["streams"][stream_name]["source"] = source_path or ""

    def start_processing(self):
        enabled = [
            name for name, settings in self.config["streams"].items()
            if settings.get("enabled", True) and settings.get("source")
        ]
        if not enabled:
            raise RuntimeError("Upload at least one readable video before starting detection.")
        self.streams.start(enabled_streams=enabled)
        self.state["processing_active"] = True

    def stop_processing(self):
        self.state["processing_active"] = False
        self.streams.stop()
        self.state["status"] = self.streams.get_status()

    def reset_session(self):
        self.tracker = PersonTracker()
        self.intrusion_engine.intrusion_counts.clear()
        self.intrusion_engine.active_events.clear()
        self.intrusion_engine.event_history.clear()
        self.state.update({"tracks": [], "events": [], "processed_frames": 0, "fps": 0.0,
                           "latency": 0.0, "optical_frame": None, "thermal_frame": None,
                           "fused_frame": None})

    def run_step(self):
        if not self.state["processing_active"]:
            return False
        started_at = time.perf_counter()
        optical_enabled = self.config["streams"]["optical"].get("enabled", True)
        thermal_enabled = self.config["streams"]["thermal"].get("enabled", True)
        opt_frame = self.streams.get_frame("optical") if optical_enabled else None
        thm_frame = self.streams.get_frame("thermal") if thermal_enabled else None
        self.state["status"] = self.streams.get_status()
        if opt_frame is None and thm_frame is None:
            return False
        try:
            opt_detections = self.optical_det.detect(opt_frame) if opt_frame is not None else []
            thermal_detections = self.thermal_det.detect(thm_frame) if thm_frame is not None else []
            fusion_enabled = self.config["fusion"].get("enabled", True) and opt_frame is not None and thm_frame is not None
            fused_detections = self.fusion.fuse(opt_detections, thermal_detections) if fusion_enabled else opt_detections + thermal_detections
            tracks = self.tracker.update(fused_detections)
            for event in self.intrusion_engine.process_tracks(tracks):
                source_frame = opt_frame if opt_frame is not None else thm_frame
                matching_track = next((track for track in tracks if track["track_id"] == event.person_id), None)
                if matching_track is not None:
                    self.alert_manager.trigger_alert(event, source_frame, matching_track["bbox"])
            latency = time.perf_counter() - started_at
            self.state.update({"optical_frame": opt_frame, "thermal_frame": thm_frame, "tracks": tracks,
                               "events": self.intrusion_engine.get_history(), "latency": latency,
                               "processed_frames": self.state["processed_frames"] + 1, "last_error": None,
                               "fps": 1.0 / latency if latency > 0 else 0.0})
            self.state["fused_frame"] = self._draw_fused_frame(opt_frame if opt_frame is not None else thm_frame, tracks)
            return True
        except Exception as error:
            logger.exception("Processing step failed")
            self.state["last_error"] = str(error)
            return False

    def _draw_fused_frame(self, frame, tracks):
        image = frame.copy()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"ID:{track['track_id']} {track['conf']:.2f}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for zone in self.zones:
            cv2.polylines(image, [zone.polygon], isClosed=True, color=(0, 165, 255), thickness=2)
        return image
