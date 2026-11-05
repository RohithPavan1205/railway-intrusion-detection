import unittest
import numpy as np
from intrusion.track_zone import TrackZone
from intrusion.intrusion_engine import IntrusionEngine, IntrusionEvent
from tracking.person_tracker import PersonTracker
from detection.optical_detector import OpticalDetector
from config_runtime import normalize_config


class TestDetectionAndTrackingAccuracy(unittest.TestCase):
    def setUp(self):
        self.config = normalize_config({
            "models": {"optical": {"conf_threshold": 0.35, "imgsz": 640}},
            "intrusion": {"temporal_confirmation_frames": 3}
        })
        self.zone1 = TrackZone("ZoneA", [[10, 10], [50, 10], [50, 50], [10, 50]])
        self.zone2 = TrackZone("ZoneB", [[100, 100], [200, 100], [200, 200], [100, 200]])

    def test_person_class_filtering_and_confidence(self):
        """Ensure detector uses COCO class 0 (person) and respects confidence threshold."""
        det = OpticalDetector(self.config)
        self.assertEqual(det.conf, 0.35)
        # Test on dummy black frame returns empty list without error
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = det.detect(dummy)
        self.assertIsInstance(dets, list)
        self.assertEqual(len(dets), 0)

    def test_tracker_id_consistency_and_motion(self):
        """Test that ByteTrack preserves person track ID across smooth motion."""
        tracker = PersonTracker(max_age=30, min_hits=2, match_thresh=0.3)
        
        # Frame 1: Detection at [100, 100, 150, 200]
        d1 = [{'bbox': [100, 100, 150, 200], 'conf': 0.85}]
        t1 = tracker.update(d1)
        
        # Frame 2: Same person moves slightly to [105, 102, 155, 202]
        d2 = [{'bbox': [105, 102, 155, 202], 'conf': 0.88}]
        t2 = tracker.update(d2)
        
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0]['track_id'], 1)

    def test_tracker_temporary_occlusion_handling(self):
        """Test that track ID is preserved across temporary occlusion."""
        tracker = PersonTracker(max_age=5, min_hits=2, match_thresh=0.3)
        
        # Establish confirmed track
        tracker.update([{'bbox': [100, 100, 150, 200], 'conf': 0.85}])
        tracker.update([{'bbox': [101, 101, 151, 201], 'conf': 0.85}])
        
        # Occluded for 1 frame
        tracker.update([])
        
        # Re-appears at overlapping location
        t_after = tracker.update([{'bbox': [102, 102, 152, 202], 'conf': 0.85}])
        self.assertEqual(len(t_after), 1)
        self.assertEqual(t_after[0]['track_id'], 1)

    def test_multi_zone_counter_independence(self):
        """Test that multiple zones maintain separate intrusion counters without resetting each other."""
        engine = IntrusionEngine(self.config, [self.zone1, self.zone2])
        
        # Track in ZoneA: bbox foot-point is [(20+40)/2, 40] = [30, 40] -> inside ZoneA [10..50], outside ZoneB
        track_in_a = [{'track_id': 10, 'bbox': [20, 10, 40, 40], 'conf': 0.9}]
        
        # Frame 1: inside ZoneA
        ev1 = engine.process_tracks(track_in_a)
        self.assertEqual(len(ev1), 0)
        self.assertEqual(engine.zone_counts[(10, "ZoneA")], 1)
        self.assertEqual(engine.zone_counts[(10, "ZoneB")], 0)
        
        # Frame 2: inside ZoneA
        ev2 = engine.process_tracks(track_in_a)
        self.assertEqual(len(ev2), 0)
        self.assertEqual(engine.zone_counts[(10, "ZoneA")], 2)
        
        # Frame 3: threshold met (3 frames) -> confirmed intrusion event
        ev3 = engine.process_tracks(track_in_a)
        self.assertEqual(len(ev3), 1)
        self.assertEqual(ev3[0].person_id, 10)
        self.assertEqual(ev3[0].zone_name, "ZoneA")
        self.assertTrue(engine.is_person_intruding(10))

    def test_no_intrusion_event_duplication(self):
        """Ensure a confirmed intrusion does not spam duplicate events while person remains in zone."""
        engine = IntrusionEngine(self.config, [self.zone1])
        track_in_a = [{'track_id': 10, 'bbox': [20, 10, 40, 40], 'conf': 0.9}]
        
        # 3 frames to trigger
        for _ in range(3):
            engine.process_tracks(track_in_a)
            
        # Frames 4-10: Person still in zone
        for _ in range(7):
            evs = engine.process_tracks(track_in_a)
            self.assertEqual(len(evs), 0, "Must not fire duplicate intrusion events while continuously inside")
            
        self.assertEqual(len(engine.event_history), 1)

    def test_person_leaving_and_reentering_zone(self):
        """Ensure exiting a zone resolves the event, and re-entering creates a new event."""
        engine = IntrusionEngine(self.config, [self.zone1])
        track_in = [{'track_id': 10, 'bbox': [20, 10, 40, 40], 'conf': 0.9}]
        track_out = [{'track_id': 10, 'bbox': [80, 80, 90, 90], 'conf': 0.9}]  # foot point [85, 90] outside
        
        # Confirm intrusion
        for _ in range(3):
            engine.process_tracks(track_in)
        self.assertTrue(engine.is_person_intruding(10))
        
        # Person exits zone
        engine.process_tracks(track_out)
        self.assertFalse(engine.is_person_intruding(10))
        
        # Person re-enters zone and stays for 3 frames
        engine.process_tracks(track_in)
        engine.process_tracks(track_in)
        new_evs = engine.process_tracks(track_in)
        
        self.assertEqual(len(new_evs), 1, "Must trigger new event upon re-entry")
        self.assertEqual(len(engine.event_history), 2)


if __name__ == "__main__":
    unittest.main()
