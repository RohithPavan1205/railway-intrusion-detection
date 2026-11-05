# RailGuard — Railway Track Intrusion & Trespassing Detection System

RailGuard is a real-time computer vision monitoring system designed to detect human trespassing and intrusions on railway tracks using YOLOv8-based person detection, multi-object tracking (ByteTrack-inspired IoU tracker), and polygon-based restricted zone geometric analysis.

The system features a minimal, high-contrast web dashboard built with Streamlit, enabling live surveillance, automated incident logging, and snapshot capture.

---

## Key Features

- **Person Detection**: YOLOv8 neural network inference for robust pedestrian and worker detection across optical (RGB) and infrared/thermal footage.
- **Persistent Tracking**: IoU-based multi-object tracking assigning persistent track IDs across successive frames.
- **Restricted Zone Monitoring**: Precise foot-point containment checks (`cv2.pointPolygonTest`) against customizable railway track polygon zones.
- **Incident Logging & Snapshots**: Automated logging of intrusion events with timestamps, track IDs, zone labels, confidence ratings, and cropped evidence snapshots.
- **Minimal Web Dashboard**: Real-time monitoring console providing:
  - Side-by-side raw input and AI bounding-box / zone overlay feeds.
  - Live throughput (FPS) and inference latency telemetry.
  - CSV export of incident logs and visual snapshot gallery.
  - Runtime parameter adjustments (confidence thresholds, confirmation frames).

---

## Architecture Overview

```
                      ┌──────────────────────┐
                      │ CCTV / Video Stream  │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │ YOLOv8 Person Model  │
                      └──────────┬───────────┘
                                 │ Detections [bbox, conf]
                                 ▼
                      ┌──────────────────────┐
                      │    Person Tracker    │
                      └──────────┬───────────┘
                                 │ Persistent Tracks [track_id, bbox]
                                 ▼
                      ┌──────────────────────┐
                      │   Intrusion Engine   │
                      │ (Foot-point in Zone) │
                      └──────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         ┌─────────────────────┐   ┌─────────────────────┐
         │ Streamlit Dashboard │   │ Alert & Snapshot DB │
         │   (Live Display)    │   │  (outputs/events/)  │
         └─────────────────────┘   └─────────────────────┘
```

---

## Project Structure

```text
railway-intrusion-detection/
├── app.py                     # Streamlit web monitoring dashboard
├── main.py                    # Pipeline engine coordinating detection, tracking, & zones
├── config_runtime.py          # Dynamic configuration merging & validation
├── config/
│   └── settings.yaml          # Base pipeline configuration (zones, thresholds, streams)
├── detection/
│   ├── optical_detector.py    # YOLOv8 detector for RGB optical video
│   └── thermal_detector.py    # YOLOv8 detector for infrared/thermal video
├── fusion/
│   └── late_fusion.py         # Multi-sensor late fusion (IoU matching & confidence boost)
├── tracking/
│   └── person_tracker.py      # IoU-based persistent track manager
├── intrusion/
│   ├── track_zone.py          # Geometric polygon zone definition and foot-point logic
│   └── intrusion_engine.py    # Temporal confirmation & intrusion state lifecycle
├── alerts/
│   └── alert_manager.py       # Event persistence (CSV) & intruder snapshot cropping
├── streams/
│   ├── video_capture.py       # Threaded OpenCV video reader
│   └── stream_manager.py      # Stream lifecycle coordinator
├── data/
│   └── samples/               # Sample optical & thermal test clips
├── outputs/
│   ├── events/                # Intrusion CSV audit logs
│   └── snapshots/             # Cropped intrusion evidence images
└── tests/                     # Comprehensive pytest test suite
```

---

## Quick Start

### 1. Prerequisites
- macOS (Apple Silicon MPS supported) or Linux / Windows (CUDA / CPU)
- Python 3.10+

### 2. Installation

Clone the repository and install the dependencies:
```bash
git clone <your-repo-url>
cd railway-intrusion-detection

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Running the Dashboard

Launch the Streamlit web interface:
```bash
streamlit run app.py
```
Open your browser at [http://localhost:8501](http://localhost:8501).

---

## Using the System

1. **Select a Video**: In the left sidebar, pick from the bundled demo presets:
   - **Daytime Demo (Optical)**: High-resolution RGB sample clip.
   - **Night-Vision Demo (Thermal)**: Infrared heat signature sample clip.
   - **Custom Upload**: Drag and drop any `.mp4`, `.avi`, or `.mov` railway CCTV file.
2. **Start Monitoring**: Click **Start** to initialize the YOLOv8 vision pipeline and start tracking.
3. **Inspect Output**:
   - **Live Feed tab**: View the live raw video and real-time bounding boxes with zone boundaries.
   - **Incident Log tab**: Inspect timestamped intrusion records, view cropped intruder snapshots, and export logs to CSV.
   - **Safety Zones tab**: Verify the geometric coordinates and vertices defining the track boundaries.
4. **Tune Parameters**: Adjust the confidence slider or confirmation frames directly from the sidebar.

---

## Running Automated Tests

Run the test suite using `pytest`:
```bash
PYTHONPATH=. pytest -v
```

All 16 unit and integration tests validate:
- Configuration merging and schema normalization.
- Polygon foot-point containment and intrusion triggers.
- Multi-stream video capture and codec metadata parsing.
- Tracker lifecycle and event logging.

---

## License

This project is developed for research and educational purposes (Academic prototype — not safety-certified for actual railway operations).
