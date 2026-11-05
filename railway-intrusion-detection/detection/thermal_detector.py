import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config_runtime import BASE_DIR
from detection.optical_detector import select_device

logger = logging.getLogger(__name__)


class ThermalDetector:
    def __init__(self, config):
        model_cfg = config.get("models", {}).get("thermal", {})
        self.conf = float(model_cfg.get("conf_threshold", 0.35))
        self.iou = float(model_cfg.get("iou_threshold", 0.45))
        self.imgsz = int(model_cfg.get("imgsz", 960))
        
        pref_device = config.get("system", {}).get("device", "mps")
        self.device = select_device(pref_device)
        
        weights_path = Path(model_cfg.get("weights", "models/thermal/yolov8n.pt"))
        if not weights_path.exists():
            resolved = BASE_DIR / weights_path
            if resolved.exists():
                weights_path = resolved
        
        logger.info(f"Loading Thermal YOLO model: {weights_path} on {self.device}")
        try:
            self.model = YOLO(str(weights_path))
        except Exception as e:
            logger.error(f"Failed to load thermal model: {e}")
            raise

    def _preprocess(self, frame):
        """
        Ensures thermal frame is 3-channel RGB (by duplicating grayscale or enhancing contrast) 
        to satisfy YOLOv8 input expectations.
        """
        if frame is None or frame.size == 0:
            return frame
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame

    def detect(self, frame):
        """
        Runs person detection on the thermal frame.
        """
        if frame is None or frame.size == 0:
            return []
            
        processed_frame = self._preprocess(frame)
        results = self.model.predict(
            source=processed_frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[0],
            device=self.device,
            verbose=False,
        )
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    "bbox": coords,
                    "conf": conf,
                    "class": cls_id,
                })
        return detections

    def track(self, frame, tracker="bytetrack.yaml", persist=True):
        """
        Runs fused detection + ByteTrack tracking on thermal frame.
        """
        if frame is None or frame.size == 0:
            return []
            
        processed_frame = self._preprocess(frame)
        results = self.model.track(
            source=processed_frame,
            persist=persist,
            tracker=tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[0],
            device=self.device,
            verbose=False,
        )
        
        tracks = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                if track_id != -1:
                    tracks.append({
                        "track_id": track_id,
                        "bbox": coords,
                        "conf": conf,
                        "class": cls_id,
                    })
        return tracks
