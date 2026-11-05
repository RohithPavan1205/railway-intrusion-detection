import threading
import queue
import logging
import time
import cv2
from .video_capture import VideoCapture

logger = logging.getLogger(__name__)

class StreamManager:
    def __init__(self, config):
        self.config = config
        self.queues = {}
        self.status = {}
        self.running = False
        self._threads = []
        self._stop_event = threading.Event()

    def _capture_loop(self, stream_name, source_config):
        logger.info(f"Starting capture loop for {stream_name}")
        cap = VideoCapture(
            source_config['source'], 
            width=source_config['width'], 
            height=source_config['height']
        )
        
        # Initial open
        opened = cap.open()
        self.status[stream_name] = "Connected" if opened else ("Missing" if cap.is_missing else "Disconnected")
        
        while self.running and not self._stop_event.is_set():
            if cap.is_missing:
                # Don't retry local missing files
                self._stop_event.wait(5)
                continue

            if not cap.cap or not cap.cap.isOpened():
                self.status[stream_name] = "Disconnected"
                self._stop_event.wait(2)
                cap.open()
                if cap.is_missing:
                    self.status[stream_name] = "Missing"
                continue
                
            self.status[stream_name] = "Connected"
            ret, frame = cap.read()
            
            if not ret:
                if any(ext in source_config['source'] for ext in [".mp4", ".avi", ".mkv"]):
                    cap.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.status[stream_name] = "Disconnected"
                    self._stop_event.wait(2)
                    continue
            
            if self.queues[stream_name].full():
                try:
                    self.queues[stream_name].get_nowait()
                except queue.Empty:
                    pass
            
            self.queues[stream_name].put(frame)
            
        cap.release()

    def start(self, enabled_streams=None):
        if self.running:
            return
        self._stop_event.clear()
        self.running = True
        selected_streams = set(enabled_streams or self.config['streams'].keys())
        for name, conf in self.config['streams'].items():
            if name not in selected_streams:
                self.status[name] = "Disabled"
                continue
            if not conf.get('source'):
                self.status[name] = "Not configured"
                continue
            self.queues[name] = queue.Queue(maxsize=10)
            self.status[name] = "Initializing"
            t = threading.Thread(target=self._capture_loop, args=(name, conf), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self.running = False
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        for name, state in self.status.items():
            if state == "Connected":
                self.status[name] = "Stopped"

    def get_frame(self, stream_name):
        try:
            return self.queues[stream_name].get(timeout=0.1)
        except (queue.Empty, KeyError):
            return None

    def get_status(self):
        return self.status
