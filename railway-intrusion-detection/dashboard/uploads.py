"""Video upload persistence and OpenCV metadata validation."""

import os
from pathlib import Path
from uuid import uuid4

import cv2


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}


def inspect_video(path):
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"valid": False, "error": "Supported formats are MP4, AVI, and MOV."}
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"valid": False, "error": "OpenCV could not read this video or codec."}
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            return {"valid": False, "error": "Video metadata is incomplete or invalid."}
        return {
            "valid": True, "path": str(path), "name": path.name, "size_bytes": path.stat().st_size,
            "fps": fps, "frame_count": frame_count, "width": width, "height": height,
            "duration_seconds": frame_count / fps,
        }
    finally:
        capture.release()


def save_uploaded_video(upload, directory="data/uploads"):
    suffix = Path(upload.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return {"valid": False, "error": "Supported formats are MP4, AVI, and MOV."}
    upload_dir = Path(directory)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid4().hex}{suffix}"
    path.write_bytes(upload.getbuffer())
    metadata = inspect_video(path)
    if not metadata["valid"]:
        path.unlink(missing_ok=True)
    return metadata


def pairing_warnings(optical, thermal):
    if not optical or not thermal:
        return ["Single-stream mode: fusion is unavailable."], False
    warnings = []
    if abs(optical["fps"] - thermal["fps"]) > 0.5:
        warnings.append("Frame rates differ; streams may not be synchronized.")
    if abs(optical["duration_seconds"] - thermal["duration_seconds"]) > 1.0:
        warnings.append("Durations differ by more than one second.")
    if optical["frame_count"] != thermal["frame_count"]:
        warnings.append("Frame counts differ.")
    return warnings, not warnings
