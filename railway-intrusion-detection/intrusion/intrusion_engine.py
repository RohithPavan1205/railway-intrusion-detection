import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class IntrusionEvent:
    def __init__(self, person_id, zone_name, confidence):
        self.timestamp = time.time()
        self.person_id = person_id
        self.zone_name = zone_name
        self.confidence = confidence
        self.resolved = False

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'person_id': self.person_id,
            'zone_name': self.zone_name,
            'confidence': self.confidence,
            'resolved': self.resolved
        }

class IntrusionEngine:
    def __init__(self, config, zones):
        self.config = config
        self.zones = zones # List of TrackZone objects
        self.temporal_threshold = config['intrusion']['temporal_confirmation_frames']
        self.intrusion_counts = defaultdict(int) # person_id -> frame_count
        self.active_events = {} # person_id -> IntrusionEvent
        self.event_history = []

    def process_tracks(self, tracks):
        """
        Processes tracked persons and detects intrusions.
        tracks: list of {'track_id': id, 'bbox': bbox, 'conf': conf}
        """
        current_frame_intrusions = []
        
        for t in tracks:
            pid = t['track_id']
            bbox = t['bbox']
            conf = t['conf']
            
            # Check each zone
            for zone in self.zones:
                # We use a dummy point from the TrackZone's get_foot_point method
                # (which should be called by the engine)
                from .track_zone import TrackZone
                # Re-instantiate or use a helper since get_foot_point is a method of TrackZone
                foot_point = [(bbox[0] + bbox[2]) / 2, bbox[3]]
                
                if zone.is_inside(foot_point):
                    self.intrusion_counts[pid] += 1
                    
                    # Confirm intrusion after N frames
                    if self.intrusion_counts[pid] >= self.temporal_threshold:
                        if pid not in self.active_events:
                            event = IntrusionEvent(pid, zone.name, conf)
                            self.active_events[pid] = event
                            self.event_history.append(event)
                            logger.warning(f"INTRUSION DETECTED: Person {pid} entered {zone.name}")
                            current_frame_intrusions.append(event)
                else:
                    # Reset count if they leave the zone
                    self.intrusion_counts[pid] = 0
                    # Resolve event if they leave
                    if pid in self.active_events:
                        self.active_events[pid].resolved = True
                        del self.active_events[pid]
        
        return current_frame_intrusions

    def get_history(self):
        return [e.to_dict() for e in self.event_history]
