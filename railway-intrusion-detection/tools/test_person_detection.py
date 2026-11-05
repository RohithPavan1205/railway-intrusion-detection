#!/usr/bin/env python3
"""
Standalone Diagnostic Tool: Railway Person Detection Pipeline Verification.
Tests YOLO inference independently of Streamlit on actual video footage.
"""

import argparse
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnostic")


def run_diagnostics(video_path, model_path, output_dir, conf_thresh=0.25, imgsz=960, sample_interval=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    logger.info(f"Model loaded successfully. Classes: {len(model.names)} (Class 0: {model.names.get(0, 'Unknown')})")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {video_path} | Resolution: {width}x{height} | Total Frames: {total_frames} | FPS: {fps:.2f}")

    diagnostics = {
        "video": str(video_path),
        "resolution": f"{width}x{height}",
        "total_frames": total_frames,
        "model": str(model_path),
        "conf_threshold": conf_thresh,
        "imgsz": imgsz,
        "frame_results": []
    }

    frames_to_test = list(range(0, min(total_frames, 300), sample_interval))
    if not frames_to_test:
        frames_to_test = [0]

    for f_idx in frames_to_test:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Run raw inference
        raw_res = model.predict(source=frame, conf=0.10, imgsz=imgsz, verbose=False)[0]
        raw_count = len(raw_res.boxes)

        # Filter to person class with configured threshold
        person_res = model.predict(source=frame, conf=conf_thresh, classes=[0], imgsz=imgsz, verbose=False)[0]
        person_boxes = person_res.boxes
        person_count = len(person_boxes)

        detected_people = []
        annotated = frame.copy()

        for b in person_boxes:
            coords = [round(float(c), 1) for c in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            detected_people.append({"bbox": coords, "conf": round(conf, 3)})
            
            # Annotate image
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lbl = f"Person {conf:.2f}"
            cv2.putText(annotated, lbl, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        save_img_path = output_dir / f"diag_frame_{f_idx:04d}.jpg"
        cv2.imwrite(str(save_img_path), annotated)

        frame_data = {
            "frame_index": f_idx,
            "raw_detections": raw_count,
            "person_detections": person_count,
            "people": detected_people,
            "saved_image": str(save_img_path)
        }
        diagnostics["frame_results"].append(frame_data)
        logger.info(f"Frame {f_idx:4d}: Raw={raw_count} | People={person_count} | Details: {detected_people}")

    cap.release()

    report_path = output_dir / "diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info(f"Diagnostic report written to: {report_path}")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Railway Person Detection Standalone Diagnostics")
    parser.add_argument("--video", type=str, required=True, help="Path to input railway video")
    parser.add_argument("--model", type=str, default="models/optical/yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--output", type=str, default="outputs/diagnostics", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    args = parser.parse_args()

    run_diagnostics(args.video, args.model, args.output, args.conf, args.imgsz)
