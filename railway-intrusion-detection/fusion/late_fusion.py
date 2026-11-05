import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)

def calculate_iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class LateFusion:
    def __init__(self, config):
        self.config = config
        self.iou_threshold = config['fusion']['iou_threshold']
        self.calibration_enabled = config['fusion']['calibration_enabled']
        self.homography = config['fusion']['homography_matrix']

    def apply_calibration(self, bbox, matrix):
        """
        Transforms a bbox using a homography matrix.
        BBox: [x1, y1, x2, y2]
        """
        if matrix is None:
            return bbox
        
        # Transform 4 corners of the bbox
        pts = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]]
        ], dtype='float32').reshape(-1, 1, 2)
        
        transformed = cv2.perspectiveTransform(pts, matrix)
        transformed = transformed.reshape(-1, 2)
        
        x_coords = transformed[:, 0]
        y_coords = transformed[:, 1]
        
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    def fuse(self, optical_dets, thermal_dets):
        """
        Fuses optical and thermal detections.
        Returns a list of fused detections.
        """
        fused = []
        used_thermal = set()
        
        # Transform thermal detections if calibration is enabled
        processed_thermal = []
        for det in thermal_dets:
            bbox = det['bbox']
            if self.calibration_enabled:
                bbox = self.apply_calibration(bbox, self.homography)
            processed_thermal.append({'bbox': bbox, 'conf': det['conf'], 'original': det})

        # Match optical to thermal
        for o_det in optical_dets:
            best_iou = 0
            best_t_idx = -1
            
            for i, t_det in enumerate(processed_thermal):
                if i in used_thermal:
                    continue
                
                iou = calculate_iou(o_det['bbox'], t_det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_t_idx = i
            
            if best_t_idx != -1 and best_iou > self.iou_threshold:
                # Fused detection: combine confidence (e.g., average or max)
                t_det = processed_thermal[best_t_idx]
                fused_conf = (o_det['conf'] + t_det['conf']) / 2
                fused.append({
                    'bbox': o_det['bbox'], # Use optical for spatial ref
                    'conf': fused_conf,
                    'sources': ['optical', 'thermal']
                })
                used_thermal.add(best_t_idx)
            else:
                # Optical only
                fused.append({
                    'bbox': o_det['bbox'],
                    'conf': o_det['conf'],
                    'sources': ['optical']
                })
                
        # Add remaining thermal detections
        for i, t_det in enumerate(processed_thermal):
            if i not in used_thermal:
                fused.append({
                    'bbox': t_det['bbox'],
                    'conf': t_det['conf'],
                    'sources': ['thermal']
                })
                
        return fused
