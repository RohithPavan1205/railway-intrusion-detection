"""Configuration defaults, merging, and validation for the monitoring runtime."""

from copy import deepcopy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "system": {
        "device": "mps",  # Prefer Apple Silicon MPS, fallback to CPU
        "log_level": "INFO",
    },
    "streams": {
        "optical": {"source": "", "fps": 30, "width": 640, "height": 480, "enabled": True},
        "thermal": {"source": "", "fps": 30, "width": 640, "height": 480, "enabled": True},
    },
    "models": {
        "optical": {
            "weights": "models/optical/yolov8n.pt",
            "conf_threshold": 0.20,  # Responsive default for distant/walking people
            "iou_threshold": 0.45,
            "imgsz": 640,  # Optimal speed/accuracy for real-time mobile/CCTV footage
        },
        "thermal": {
            "weights": "models/thermal/yolov8n.pt",
            "conf_threshold": 0.20,
            "iou_threshold": 0.45,
            "imgsz": 640,
        },
    },
    "tracking": {
        "tracker": "bytetrack.yaml",
        "persist": True,
        "track_buffer": 30,  # Max frames to keep lost track before removing
        "match_thresh": 0.7,
    },
    "fusion": {
        "enabled": False,
        "sync_offset_ms": 0,
        "iou_threshold": 0.30,
        "calibration_enabled": False,
        "homography_matrix": None,
    },
    "intrusion": {
        "restricted_zones": [
            {"name": "Track Zone 1", "polygon": [[100, 400], [540, 400], [600, 480], [40, 480]]}
        ],
        "temporal_confirmation_frames": 5,
        "cooldown_frames": 30,  # Prevent repeated alert flooding for continuous intrusion
    },
    "alerts": {
        "audio_enabled": False,
        "save_snapshots": True,
        "snapshot_path": "outputs/snapshots",
        "event_log_path": "outputs/events/events.csv",
    },
}


def deep_merge(base, override):
    """Return a recursive merge without mutating either input."""
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(config):
    for stream in ("optical", "thermal"):
        settings = config["streams"][stream]
        if settings["width"] <= 0 or settings["height"] <= 0:
            raise ValueError(f"{stream} stream dimensions must be positive")
        if settings["fps"] <= 0:
            raise ValueError(f"{stream} FPS must be positive")

    for stream in ("optical", "thermal"):
        settings = config["models"][stream]
        for key in ("conf_threshold", "iou_threshold"):
            if not 0 <= float(settings[key]) <= 1:
                raise ValueError(f"{stream} {key} must be between 0 and 1")

    if config["intrusion"]["temporal_confirmation_frames"] < 1:
        raise ValueError("Intrusion confirmation frames must be at least 1")
    return config


def normalize_config(config):
    return validate_config(deep_merge(DEFAULT_CONFIG, config or {}))
