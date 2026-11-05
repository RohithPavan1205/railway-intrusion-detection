import logging
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ThermalDetector:
    def __init__(self, config):
        self.conf = config['models']['thermal']['conf_threshold']
        self.iou = config['models']['thermal']['iou_threshold']
        weights = config['models']['thermal']['weights']
        
        logger.info(f"Loading Thermal YOLO model: {weights}")
        try:
            self.model = YOLO(weights)
        except Exception as e:
            logger.error(f"Failed to load thermal model: {e}")
            raise

    def _preprocess(self, frame):
        """
        Ensures thermal frame is 3-channel RGB (by duplicating grayscale) 
        to satisfy YOLOv8 requirements.
        """
        if len(frame.shape) == 2: # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1: # Single channel
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return frame

    def detect(self, frame):
        """
        Runs person detection on the thermal frame.
        """
        processed_frame = self._preprocess(frame)
        
        results = self.model.predict(
            source=processed_frame, 
            conf=self.conf, 
            iou=self.iou, 
            classes=[0], 
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
