# CAM: Driver Cognitive Attention & Eye-Gaze Analytics

A research platform and toolchain for **driver visual attention analysis, gaze-object intersection mapping, pupil dynamics, and multi-sensor data extraction** designed specifically for heterogeneous and unstructured traffic environments.

---

## 📌 Repository Overview

This repository contains AI pipelines, data extraction routines, and gaze-to-object attention mapping scripts for rider and driver visual behavior research:

```
CAM/
├── Attention Analysis/          # Core attention analytics & YOLO gaze mapping
│   ├── dreye_final.py           # End-to-end gaze attention evaluation pipeline
│   ├── dreye_yolo.py            # YOLO detection & gaze ROI intersection mapping
│   ├── Location.csv             # GPS trajectory stream for driving sessions
│   ├── general_eye_gaze.csv     # Extracted 2D/3D gaze tracking coordinate records
│   ├── results.csv              # Aggregated attention & fixation metrics
│   ├── run_log.txt              # Pipeline execution logs
│   ├── run_log_100epochs.txt    # Extended training & evaluation benchmark logs
│   └── sreevatsav-tada-p21.json # Calibration parameters
│
├── Data extraction/             # Meta Project Aria .vrs data extraction utilities
│   ├── SHOW_EYE.py              # Real-time gaze & eye camera visualizer
│   ├── analyze_vrs_profiles.py  # Profile parser and sensor integrity analyzer
│   ├── batch_pupil_detect.py    # Automated batch pupil diameter detector
│   ├── extract_all_et_images.py # Frame extraction pipeline from raw .vrs files
│   └── quick_rotate_fix.py      # Camera frame rotation and alignment helper
│
└── .gitignore                   # Repository filtering rules
```

---

## 🛠️ Modules & Features

### 1. Attention Analysis (`Attention Analysis/`)
* **Gaze-Object Mapping (`dreye_final.py`, `dreye_yolo.py`)**:
  * Associates frame-level gaze coordinates $(x, y)$ with dynamic bounding boxes detected by YOLO (e.g., `autorickshaw`, `two_wheeler`, `pedestrian`, `large_vehicle`, `road`).
  * Classifies visual focus into **on-road**, **lead vehicle**, **mirror glance**, or **off-road attention**.
* **Attention Metrics**:
  * Fixation duration (I-VT velocity threshold identification).
  * Gaze transition entropy and scanning behavior across traffic hazards.

### 2. Data Extraction & Sensor Utilities (`Data extraction/`)
* **VRS Stream Parsing (`analyze_vrs_profiles.py`)**: Analyzes Meta Project Aria multi-sensor `.vrs` records (RGB scene camera, dual eye-tracking cameras, IMU, GPS).
* **Batch Frame Extraction (`extract_all_et_images.py`)**: Synchronizes and extracts high-resolution scene video frames and corresponding eye crops.
* **Pupil Tracking (`batch_pupil_detect.py`)**: Computes pupil dilation dynamics for cognitive workload estimation.

---

## 🚀 Setup & Installation

### Prerequisites
* Python 3.8+
* PyTorch & Torchvision (CUDA enabled recommended)
* Ultralytics (YOLOv8 / YOLOv11)
* OpenCV & NumPy

```bash
# Clone the repository
git clone https://github.com/swathigiiits/CAM.git
cd CAM

# Install required dependencies
pip install opencv-python numpy pandas matplotlib ultralytics scipy
```

---

## 📊 Related Datasets & Research Assets

* **`Data50`**: Annotated YOLO segmentation and detection dataset for 5 urban traffic classes: `autorickshaw`, `large_vehicle`, `pedestrian`, `road`, and `two_wheeler`.
* **`et_data`**: Driver recordings from wearable eye-tracking glasses (Meta Project Aria) containing raw `.vrs` streams, sensor telemetry, and gaze outputs.

---

## 📬 Contact & Citation

For research collaboration or inquiries regarding this toolchain and dataset, please open an issue or reach out via GitHub ([@swathigiiits](https://github.com/swathigiiits)).
