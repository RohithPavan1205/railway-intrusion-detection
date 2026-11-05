import time
import logging
from collections import defaultdict
from intrusion.track_zone import TrackZone

logger = logging.getLogger(__name__)


class IntrusionEvent:
    def __init__(self, person_id, zone_name, confidence):
        self.timestamp = time.time()
        self.person_id = person_id
        self.zone_name = zone_name
        self.confidence = float(confidence)
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
    """
    Intrusion engine monitoring tracked persons across defined restricted zones.
    Key guarantees:
    1. Multi-zone independent counters (keyed by (person_id, zone_name)).
    2. Temporal confirmation over N consecutive frames before declaring an intrusion.
    3. Hysteresis/cooldown: No alert flooding for the same continuous intrusion event.
    4. Proper resolution when person exits, allowing re-detection if they leave and re-enter.
    """
    def __init__(self, config, zones):
        self.config = config
        self.zones = zones  # List of TrackZone objects
        self.temporal_threshold = int(config.get('intrusion', {}).get('temporal_confirmation_frames', 5))
        
        # (person_id, zone_name) -> consecutive frame count
        self.zone_counts = defaultdict(int)
        
        # (person_id, zone_name) -> IntrusionEvent (active ongoing intrusions)
        self.active_events = {}
        
        # Set of confirmed intruding track IDs in current frame
        self.confirmed_intruders = set()
        
        self.event_history = []

    def process_tracks(self, tracks):
        """
        Processes tracked persons and detects intrusions.
        tracks: list of {'track_id': id, 'bbox': bbox, 'conf': conf}
        Returns list of new IntrusionEvent instances created in this frame.
        """
        new_intrusions = []
        current_frame_active_intruders = set()
        
        # Set of track IDs currently present
        current_track_ids = {t['track_id'] for t in tracks}
        
        for t in tracks:
            pid = t['track_id']
            bbox = t['bbox']
            conf = t['conf']
            
            # Person's foot point approximation: bottom-center of bounding box
            foot_point = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
            
            for zone in self.zones:
                zone_key = (pid, zone.name)
                is_inside = zone.is_inside(foot_point)
                
                if is_inside:
                    self.zone_counts[zone_key] += 1
                    
                    # Confirm intrusion once threshold is met
                    if self.zone_counts[zone_key] >= self.temporal_threshold:
                        current_frame_active_intruders.add(pid)
                        
                        # Only fire event once per continuous intrusion episode
                        if zone_key not in self.active_events:
                            event = IntrusionEvent(pid, zone.name, conf)
                            self.active_events[zone_key] = event
                            self.event_history.append(event)
                            logger.warning(f"INTRUSION CONFIRMED: Person {pid} entered {zone.name}")
                            new_intrusions.append(event)
                else:
                    # Person has left the zone: reset count
                    self.zone_counts[zone_key] = 0
                    
                    # Resolve active event if it was confirmed
                    if zone_key in self.active_events:
                        self.active_events[zone_key].resolved = True
                        del self.active_events[zone_key]
        
        # Clean up stale track keys for people no longer present in scene
        stale_keys = [k for k in self.zone_counts if k[0] not in current_track_ids]
        for k in stale_keys:
            del self.zone_counts[k]
            if k in self.active_events:
                self.active_events[k].resolved = True
                del self.active_events[k]
                
        self.confirmed_intruders = current_frame_active_intruders
        return new_intrusions

    def is_person_intruding(self, person_id):
        """Checks if a person currently has an active confirmed intrusion."""
        return person_id in self.confirmed_intruders

    def get_history(self):
        return [e.to_dict() for e in self.event_history]

    def reset(self):
        self.zone_counts.clear()
        self.active_events.clear()
        self.confirmed_intruders.clear()
        self.event_history.clear()
