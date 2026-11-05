import logging
from pathlib import Path
from ultralytics import YOLO

from config_runtime import BASE_DIR

logger = logging.getLogger(__name__)

class OpticalDetector:
    def __init__(self, config):
        self.conf = config['models']['optical']['conf_threshold']
        self.iou = config['models']['optical']['iou_threshold']
        weights_path = Path(config['models']['optical']['weights'])
        
        # Support both relative to BASE_DIR and direct cwd
        if not weights_path.exists():
            resolved = BASE_DIR / weights_path
            if resolved.exists():
                weights_path = resolved
        
        logger.info(f"Loading Optical YOLO model: {weights_path}")
        try:
            self.model = YOLO(str(weights_path))
        except Exception as e:
            logger.error(f"Failed to load optical model: {e}")
            raise

    def detect(self, frame):
        """
        Runs person detection on the optical frame.
        Returns a list of detections: [{'bbox': [x1, y1, x2, y2], 'conf': score, 'class': cls}]
        """
        results = self.model.predict(
            source=frame, 
            conf=self.conf, 
            iou=self.iou, 
            classes=[0], # Person class in COCO
            verbose=False
        )
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({
                    'bbox': coords, 
                    'conf': conf, 
                    'class': int(box.cls[0])
                })
        return detections
