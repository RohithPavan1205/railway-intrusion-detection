import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class OpticalDetector:
    def __init__(self, config):
        self.conf = config['models']['optical']['conf_threshold']
        self.iou = config['models']['optical']['iou_threshold']
        weights = config['models']['optical']['weights']
        
        logger.info(f"Loading Optical YOLO model: {weights}")
        try:
            self.model = YOLO(weights)
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
