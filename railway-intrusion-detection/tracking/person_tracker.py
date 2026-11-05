import numpy as np
import logging
from fusion.late_fusion import calculate_iou

logger = logging.getLogger(__name__)

class Track:
    def __init__(self, track_id, bbox, conf):
        self.track_id = track_id
        self.bbox = bbox
        self.conf = conf
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        # Simple prediction: bbox stays the same
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox, conf):
        self.bbox = bbox
        self.conf = conf
        self.time_since_update = 0
        self.hits += 1

class PersonTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """
        Update tracks with new detections.
        detections: list of {'bbox': [x1, y1, x2, y2], 'conf': score}
        """
        # Predict next state for all tracks
        for t in self.tracks:
            t.predict()

        # Match detections to tracks using IoU
        assigned_dets = set()
        updated_tracks = set()

        # Sort detections by confidence (High-score first)
        sorted_dets = sorted(detections, key=lambda x: x['conf'], reverse=True)

        for t in self.tracks:
            best_iou = 0
            best_det_idx = -1
            
            for i, det in enumerate(sorted_dets):
                if i in assigned_dets:
                    continue
                
                iou = calculate_iou(t.bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx != -1 and best_iou > 0.3:
                t.update(sorted_dets[best_det_idx]['bbox'], sorted_dets[best_det_idx]['conf'])
                assigned_dets.add(best_det_idx)
                updated_tracks.add(t)

        # Create new tracks for unmatched detections
        for i, det in enumerate(sorted_dets):
            if i not in assigned_dets:
                new_track = Track(self.next_id, det['bbox'], det['conf'])
                self.tracks.append(new_track)
                self.next_id += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # Return only tracks that have been seen enough times
        return [{
            'track_id': t.track_id,
            'bbox': t.bbox,
            'conf': t.conf
        } for t in self.tracks if t.hits >= 1] # Using 1 for prototype responsiveness
