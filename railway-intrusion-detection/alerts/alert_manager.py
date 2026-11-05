import cv2
import os
import csv
import logging
import time

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, config):
        self.config = config
        self.save_snapshots = config['alerts']['save_snapshots']
        self.snapshot_path = config['alerts']['snapshot_path']
        self.event_log_path = config['alerts']['event_log_path']
        
        os.makedirs(self.snapshot_path, exist_ok=True)
        
        # Initialize event log file
        if not os.path.exists(self.event_log_path):
            with open(self.event_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'person_id', 'zone_name', 'confidence', 'snapshot_path'])

    def trigger_alert(self, event, frame, bbox):
        """
        Handles the alert: saves snapshot and logs to CSV.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        snapshot_filename = f"intrusion_{timestamp}_p{event.person_id}.jpg"
        snapshot_full_path = os.path.join(self.snapshot_path, snapshot_filename)
        
        if self.save_snapshots:
            # Crop and save the intruder
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                cv2.imwrite(snapshot_full_path, crop)
        
        # Log to CSV
        with open(self.event_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                event.timestamp, 
                event.person_id, 
                event.zone_name, 
                event.confidence, 
                snapshot_full_path
            ])
        
        logger.info(f"Alert triggered and logged: {snapshot_full_path}")
