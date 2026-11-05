# Railway Track Intrusion and Trespassing Detection System

This is a prototype system for detecting intrusions on railway tracks using late-fusion of optical and thermal video streams.

## Architecture
- **Detection**: YOLOv8n for person detection in both optical and thermal streams.
- **Fusion**: Late-fusion based on IoU matching of detections.
- **Tracking**: Person tracking with persistent IDs using a ByteTrack-inspired IoU tracker.
- **Intrusion**: Polygon-based restricted zone monitoring using foot-point detection.
- **Alerts**: Event logging to CSV and automatic snapshot capture of intruders.
- **Dashboard**: Real-time monitoring interface built with Streamlit.

## Setup Instructions

### 1. Environment
- Hardware: MacBook Air M4 (Apple Silicon)
- OS: macOS
- Python: 3.11+

### 2. Installation
```bash
cd railway-intrusion-detection
pip install -r requirements.txt
```

### 3. Data Acquisition
The system requires paired optical and thermal videos. 
1. Download the **VidLLVIP** dataset from Hugging Face: [https://huggingface.co/datasets/jianfeng0369/VidLLVIP](https://huggingface.co/datasets/jianfeng0369/VidLLVIP)
2. Extract the `dataset.zip`.
3. Run the sample setup script:
   ```bash
   python datasets/download_samples.py --path /path/to/extracted/VidLLVIP
   ```

### 4. Configuration
Modify `config/settings.yaml` to adjust:
- Video source paths.
- Model weights.
- Restricted zone polygon coordinates.
- Detection thresholds.

### 5. Execution
Launch the monitoring dashboard:
```bash
streamlit run app.py
```

## Project Structure
- `app.py`: Streamlit dashboard.
- `main.py`: Core processing pipeline.
- `detection/`: Optical and Thermal detection modules.
- `fusion/`: Late-fusion and calibration logic.
- `tracking/`: Person tracking implementation.
- `intrusion/`: Zone definition and intrusion engine.
- `alerts/`: Alert management and snapshotting.
- `streams/`: Video capture and stream management.

## Known Limitations
- **Calibration**: Currently uses uncalibrated fusion (assumes aligned streams). Calibrated fusion is supported via homography in `settings.yaml`.
- **Hardware**: Optimized for Apple Silicon (MPS).
- **Dataset**: Requires manually provided paired video streams.

**Note**: This is an academic research prototype and is NOT safety-certified for actual railway operations.
