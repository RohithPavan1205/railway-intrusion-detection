import cv2
import logging
import os

logger = logging.getLogger(__name__)

class VideoCapture:
    def __init__(self, source, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.is_missing = False
        self.open()

    def open(self):
        # Check if it's a local file and if it exists
        if not self.source.startswith(('rtsp://', 'http://', 'https://')) and not self.source.isdigit():
            if not os.path.exists(self.source):
                logger.error(f"Video file not found: {self.source}")
                self.is_missing = True
                return False
        
        logger.info(f"Opening video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False
        
        self.is_missing = False
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
        return ret, frame

    def get_fps(self):
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info(f"Released video source: {self.source}")

    def __del__(self):
        self.release()
