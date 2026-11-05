import logging
from pathlib import Path
import torch
from ultralytics import YOLO

from config_runtime import BASE_DIR

logger = logging.getLogger(__name__)


def select_device(preferred="mps"):
    """Select the best available computation device with safe fallback."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    elif preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class OpticalDetector:
    def __init__(self, config):
        model_cfg = config.get("models", {}).get("optical", {})
        self.conf = float(model_cfg.get("conf_threshold", 0.35))
        self.iou = float(model_cfg.get("iou_threshold", 0.45))
        self.imgsz = int(model_cfg.get("imgsz", 960))
        
        pref_device = config.get("system", {}).get("device", "mps")
        self.device = select_device(pref_device)
        
        weights_path = Path(model_cfg.get("weights", "models/optical/yolov8n.pt"))
        if not weights_path.exists():
            resolved = BASE_DIR / weights_path
            if resolved.exists():
                weights_path = resolved
        
        logger.info(f"Loading Optical YOLO model: {weights_path} on {self.device}")
        try:
            self.model = YOLO(str(weights_path))
        except Exception as e:
            logger.error(f"Failed to load optical model: {e}")
            raise

    def detect(self, frame):
        """
        Runs person detection on the optical frame with COCO class 0 (person).
        Returns a list of detections: [{'bbox': [x1, y1, x2, y2], 'conf': score, 'class': 0}]
        """
        if frame is None or frame.size == 0:
            return []
            
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[0],  # Person class in COCO dataset
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
        Runs fused detection + ByteTrack tracking on frame using Ultralytics native tracking.
        Returns a list of tracked objects: [{'track_id': id, 'bbox': [x1, y1, x2, y2], 'conf': score, 'class': 0}]
        """
        if frame is None or frame.size == 0:
            return []
            
        results = self.model.track(
            source=frame,
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
                # Track ID is assigned when tracker confirms an association
                track_id = int(box.id[0]) if box.id is not None else -1
                if track_id != -1:
                    tracks.append({
                        "track_id": track_id,
                        "bbox": coords,
                        "conf": conf,
                        "class": cls_id,
                    })
        return tracks
