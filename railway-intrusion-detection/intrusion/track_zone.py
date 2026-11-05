import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class TrackZone:
    def __init__(self, name, polygon):
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)

    def is_inside(self, point):
        """
        Check if a point [x, y] is inside the polygon.
        """
        result = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    def get_foot_point(self, bbox):
        """
        Calculate the bottom-center point of a bounding box.
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, y2]
