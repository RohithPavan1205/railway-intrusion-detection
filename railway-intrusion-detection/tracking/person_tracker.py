"""Robust ByteTrack and IoU tracking for persistent pedestrian tracking."""

import numpy as np
import logging
from fusion.late_fusion import calculate_iou

logger = logging.getLogger(__name__)


class Track:
    def __init__(self, track_id, bbox, conf, min_hits=1):
        self.track_id = track_id
        self.bbox = list(bbox)
        self.conf = float(conf)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.min_hits = min_hits
        # Velocity estimate [vx, vy, vw, vh]
        self.velocity = [0.0, 0.0, 0.0, 0.0]

    def predict(self):
        """Advance track state by motion prediction."""
        self.age += 1
        self.time_since_update += 1
        
        # Apply velocity prediction to bbox
        x1, y1, x2, y2 = self.bbox
        vx, vy, vw, vh = self.velocity
        w = x2 - x1
        h = y2 - y1
        new_x1 = x1 + vx
        new_y1 = y1 + vy
        new_x2 = new_x1 + w + vw
        new_y2 = new_y1 + h + vh
        self.bbox = [new_x1, new_y1, new_x2, new_y2]
        return self.bbox

    def update(self, bbox, conf):
        """Update track with newly matched detection and update velocity."""
        old_x1, old_y1, old_x2, old_y2 = self.bbox
        new_x1, new_y1, new_x2, new_y2 = bbox
        
        # Exponential moving average for velocity
        vx = new_x1 - old_x1
        vy = new_y1 - old_y1
        vw = (new_x2 - new_x1) - (old_x2 - old_x1)
        vh = (new_y2 - new_y1) - (old_y2 - old_y1)
        
        alpha = 0.5
        self.velocity = [
            alpha * vx + (1 - alpha) * self.velocity[0],
            alpha * vy + (1 - alpha) * self.velocity[1],
            alpha * vw + (1 - alpha) * self.velocity[2],
            alpha * vh + (1 - alpha) * self.velocity[3],
        ]
        
        self.bbox = list(bbox)
        self.conf = float(conf)
        self.time_since_update = 0
        self.hits += 1

    @property
    def is_confirmed(self):
        return self.hits >= self.min_hits


class PersonTracker:
    """
    ByteTrack-inspired multi-stage association tracker:
    1. Associate high-confidence detections with active tracks using IoU & motion.
    2. Associate remaining low-confidence detections with remaining tracks to handle occlusions.
    3. Initialize new tracks from unmatched detections.
    4. Evict stale tracks exceeding max_age.
    """
    def __init__(self, max_age=30, min_hits=1, high_thresh=0.30, match_thresh=0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        Update tracks with new detections.
        detections: list of {'bbox': [x1, y1, x2, y2], 'conf': score}
        """
        if not detections and not self.tracks:
            return []

        # 1. Predict next state for all tracks
        for t in self.tracks:
            t.predict()

        # Partition detections into high and low confidence
        high_dets = [d for d in detections if d['conf'] >= self.high_thresh]
        low_dets = [d for d in detections if d['conf'] < self.high_thresh]

        # First association: High-confidence detections with all tracks
        matched_tracks, unmatched_tracks, unmatched_high_dets = self._associate(
            self.tracks, high_dets, self.match_thresh
        )

        for track, det in matched_tracks:
            track.update(det['bbox'], det['conf'])

        # Second association: Low-confidence detections with remaining tracks (handles occlusion)
        matched_tracks_second, remaining_tracks, _ = self._associate(
            unmatched_tracks, low_dets, self.match_thresh * 0.8
        )

        for track, det in matched_tracks_second:
            track.update(det['bbox'], det['conf'])

        # Initialize new tracks for unmatched high-confidence detections
        for det in unmatched_high_dets:
            new_track = Track(self.next_id, det['bbox'], det['conf'], self.min_hits)
            self.tracks.append(new_track)
            self.next_id += 1

        # Remove stale tracks that have been lost for too long
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks that were updated recently
        active_output = []
        for t in self.tracks:
            if t.is_confirmed and t.time_since_update <= 1:
                active_output.append({
                    'track_id': t.track_id,
                    'bbox': t.bbox,
                    'conf': t.conf
                })
        return active_output

    def _associate(self, tracks, detections, threshold):
        """Associates detections with tracks using IoU matching."""
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        # Compute cost/IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = calculate_iou(track.bbox, det['bbox'])

        matched_tracks = []
        assigned_tracks = set()
        assigned_dets = set()

        # Match from highest IoU down
        while True:
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_iou = iou_matrix[max_idx]
            if best_iou < threshold or best_iou <= 0:
                break
            
            track_idx, det_idx = max_idx
            matched_tracks.append((tracks[track_idx], detections[det_idx]))
            assigned_tracks.add(track_idx)
            assigned_dets.add(det_idx)
            
            iou_matrix[track_idx, :] = -1
            iou_matrix[:, det_idx] = -1

        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in assigned_tracks]
        unmatched_dets = [d for j, d in enumerate(detections) if j not in assigned_dets]

        return matched_tracks, unmatched_tracks, unmatched_dets

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
