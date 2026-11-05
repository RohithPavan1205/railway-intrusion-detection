import cv2
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    OpenCV video stream reader with support for native resolution reading 
    or optional resizing when requested.
    """
    def __init__(self, source, width=None, height=None):
        self.source = source
        self.target_width = width
        self.target_height = height
        self.cap = None
        self.is_missing = False
        self.native_width = 0
        self.native_height = 0
        self.open()

    def open(self):
        if not self.source:
            self.is_missing = True
            return False
            
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
        self.native_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.native_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return True

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        if ret and self.target_width and self.target_height:
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
        return ret, frame

    def get_fps(self):
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS) or 0
        return 0

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info(f"Released video source: {self.source}")

    def __del__(self):
        self.release()
