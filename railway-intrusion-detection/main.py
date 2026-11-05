import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from alerts.alert_manager import AlertManager
from config_runtime import BASE_DIR, deep_merge, normalize_config
from detection.optical_detector import OpticalDetector
from detection.thermal_detector import ThermalDetector
from fusion.late_fusion import LateFusion
from intrusion.intrusion_engine import IntrusionEngine
from intrusion.track_zone import TrackZone
from streams.stream_manager import StreamManager
from tracking.person_tracker import PersonTracker

logger = logging.getLogger(__name__)


class RailwayIntrusionSystem:
    """Coordinates detection, ByteTrack tracking, and zone intrusion analysis."""

    def __init__(self, config_path=None, config=None):
        file_config = {}
        target_path = Path(config_path) if config_path else BASE_DIR / "config" / "settings.yaml"
        if not target_path.exists():
            target_path = Path("config/settings.yaml")

        if target_path.exists():
            try:
                with open(target_path, "r") as config_file:
                    file_config = yaml.safe_load(config_file) or {}
            except Exception as e:
                logger.warning(f"Could not load {target_path}, falling back to defaults: {e}")

        self.config = normalize_config(deep_merge(file_config, config or {}))
        self.streams = StreamManager(self.config)
        self.optical_det = OpticalDetector(self.config)
        self.thermal_det = ThermalDetector(self.config)
        self.fusion = LateFusion(self.config)
        self.tracker = PersonTracker(max_age=30, min_hits=1, high_thresh=0.30, match_thresh=0.25)
        self.zones = self._make_zones()
        self.intrusion_engine = IntrusionEngine(self.config, self.zones)
        self.alert_manager = AlertManager(self.config)
        
        self.state = {
            "optical_frame": None,
            "thermal_frame": None,
            "fused_frame": None,
            "tracks": [],
            "events": [],
            "status": self.streams.get_status(),
            "fps": 0.0,
            "latency": 0.0,
            "processing_active": False,
            "processed_frames": 0,
            "last_detections_count": 0,
            "last_error": None,
        }

    def _make_zones(self):
        zones = []
        for zone in self.config["intrusion"]["restricted_zones"]:
            zones.append(TrackZone(zone["name"], zone["polygon"]))
        return zones

    def update_config(self, changes):
        candidate = normalize_config(deep_merge(self.config, changes))
        self.config = candidate
        self.streams.config = candidate
        
        # Hyperparameters
        opt_cfg = candidate["models"]["optical"]
        self.optical_det.conf = opt_cfg["conf_threshold"]
        self.optical_det.iou = opt_cfg["iou_threshold"]
        self.optical_det.imgsz = opt_cfg.get("imgsz", 640)
        
        thm_cfg = candidate["models"]["thermal"]
        self.thermal_det.conf = thm_cfg["conf_threshold"]
        self.thermal_det.iou = thm_cfg["iou_threshold"]
        self.thermal_det.imgsz = thm_cfg.get("imgsz", 640)
        
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
        self.tracker.reset()
        self.intrusion_engine.reset()
        self.state.update({
            "tracks": [],
            "events": [],
            "processed_frames": 0,
            "last_detections_count": 0,
            "fps": 0.0,
            "latency": 0.0,
            "optical_frame": None,
            "thermal_frame": None,
            "fused_frame": None,
        })

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
            # 1. Detection
            opt_detections = self.optical_det.detect(opt_frame) if opt_frame is not None else []
            thermal_detections = self.thermal_det.detect(thm_frame) if thm_frame is not None else []
            
            fusion_enabled = self.config["fusion"].get("enabled", True) and opt_frame is not None and thm_frame is not None
            fused_detections = (
                self.fusion.fuse(opt_detections, thermal_detections)
                if fusion_enabled
                else (opt_detections if opt_frame is not None else thermal_detections)
            )
            
            # 2. Tracking (ByteTrack multi-stage association)
            tracks = self.tracker.update(fused_detections)
            
            # 3. Intrusion Analysis
            new_events = self.intrusion_engine.process_tracks(tracks)
            
            # 4. Snapshot logging for confirmed new intrusions
            source_frame = opt_frame if opt_frame is not None else thm_frame
            for event in new_events:
                matching_track = next((t for t in tracks if t["track_id"] == event.person_id), None)
                if matching_track is not None and source_frame is not None:
                    self.alert_manager.trigger_alert(event, source_frame, matching_track["bbox"])
            
            latency = time.perf_counter() - started_at
            current_fps = 1.0 / latency if latency > 0 else 0.0
            prev_fps = self.state["fps"]
            smoothed_fps = (0.2 * current_fps + 0.8 * prev_fps) if prev_fps > 0 else current_fps

            self.state.update({
                "optical_frame": opt_frame,
                "thermal_frame": thm_frame,
                "tracks": tracks,
                "events": self.intrusion_engine.get_history(),
                "latency": latency,
                "processed_frames": self.state["processed_frames"] + 1,
                "last_detections_count": len(fused_detections),
                "last_error": None,
                "fps": smoothed_fps,
            })
            
            # 5. Render annotated output frame
            self.state["fused_frame"] = self._draw_fused_frame(source_frame, tracks)
            return True
        except Exception as error:
            logger.exception("Processing step failed")
            self.state["last_error"] = str(error)
            return False

    def _draw_fused_frame(self, frame, tracks):
        """
        Renders clear, uncluttered detection visualization:
        - Green bounding boxes for normal pedestrians outside restricted zone.
        - Red bounding boxes for confirmed track intruders.
        - Track ID and confidence score clamped within image bounds.
        - Orange polygon zone overlay with clear semi-transparent styling.
        """
        if frame is None:
            return None
            
        image = frame.copy()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        h, w = image.shape[:2]

        # Draw restricted zone boundaries
        for zone in self.zones:
            cv2.polylines(image, [zone.polygon], isClosed=True, color=(0, 165, 255), thickness=2)
            if len(zone.polygon) > 0:
                zx, zy = zone.polygon[0]
                cv2.putText(
                    image, zone.name, (max(10, zx), max(20, zy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA
                )

        # Draw person tracks
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            
            tid = track["track_id"]
            conf = track["conf"]
            
            is_intruder = self.intrusion_engine.is_person_intruding(tid)
            box_color = (0, 0, 255) if is_intruder else (0, 255, 0)
            thickness = 2

            # Bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
            
            # Label
            status_text = " [ALERT]" if is_intruder else ""
            label = f"ID:{tid} {conf:.2f}{status_text}"
            
            font_scale = 0.45
            font_thickness = 1
            (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            lbl_y1 = max(lbl_h + 4, y1 - 4)
            lbl_y0 = lbl_y1 - lbl_h - 4
            cv2.rectangle(image, (x1, lbl_y0), (min(w - 1, x1 + lbl_w + 6), lbl_y1), box_color, -1)
            cv2.putText(
                image, label, (x1 + 3, lbl_y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )
            
            # Foot point
            foot_x = (x1 + x2) // 2
            foot_y = y2
            cv2.circle(image, (foot_x, foot_y), 4, (0, 0, 255) if is_intruder else (255, 255, 0), -1)

        return image
