# CAM: Driver Cognitive Attention & Eye-Gaze Analytics

A comprehensive research platform and toolchain for **driver attention analysis, visual gaze mapping, and multi-sensor data extraction** designed specifically for heterogeneous and unstructured traffic environments (motorized two-wheelers, urban traffic).

---

## 📌 Repository Overview

This repository contains the core AI pipelines, data extraction routines, and gaze-to-object attention mapping scripts used in two-wheeler rider visual behavior research:

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
│   └── sreevatsav-tada-p21.json # Aria session sensor calibration parameters
│
├── Data extraction/             # Meta Project Aria .vrs data extraction utilities
│   ├── SHOW_EYE.py              # Real-time gaze & eye camera visualizer
│   ├── analyze_vrs_profiles.py  # Profile parser and sensor integrity analyzer
│   ├── batch_pupil_detect.py    # Automated batch pupil diameter detector
│   ├── extract_all_et_images.py # Frame extraction pipeline from raw .vrs files
│   └── quick_rotate_fix.py      # Camera frame rotation and alignment helper
│
├── README_myEye2Wheeler.md      # Reference specs for the myEye2Wheeler dataset
└── .gitignore                   # Repository filtering rules
```

---

## 🛠️ Modules & Features

### 1. Attention Analysis (`Attention Analysis/`)
* **Gaze-Object Mapping (`dreye_final.py`, `dreye_yolo.py`)**:
  * Associates frame-level gaze points $(x, y)$ with dynamic bounding boxes detected by YOLO (e.g., `autorickshaw`, `two_wheeler`, `pedestrian`, `large_vehicle`, `road`).
  * Classifies visual focus into **on-road**, **lead vehicle**, **mirror glance**, or **off-road distraction**.
* **Attention Metrics**:
  * Fixation duration (I-VT velocity threshold identification).
  * Gaze transition entropy and scanning quality across traffic hazards.

### 2. Data Extraction & VRS Utilities (`Data extraction/`)
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
git clone https://github.com/swaaathi1/CAM.git
cd CAM

# Install required dependencies
pip install opencv-python numpy pandas matplotlib ultralytics scipy
```

---

## 📊 Related Datasets & Research

* **`Data50`**: Annotated YOLO segmentation and detection dataset for 5 urban traffic classes: `autorickshaw`, `large_vehicle`, `pedestrian`, `road`, and `two_wheeler`.
* **`et_data`**: Volunteer driver recordings from Meta Project Aria glasses containing raw `.vrs` files, sensor sidecars, and MPS (Machine Perception Services) gaze outputs.
* **`myEye2Wheeler`**: Benchmark two-wheeler eye-tracking dataset in urban Indian traffic (*IEEE ITSC 2024*).

---

## 📖 Citation

If you use these scripts or associated datasets in your research, please cite:
```bibtex
@inproceedings{kumar2024myeye2wheeler,
  title={myEye2Wheeler: A Two-Wheeler Indian Driver Real-World Eye-Tracking Dataset},
  author={Kumar, Bhaiya Vaibhaw and Rawat, D. and Kandalla, T. and Nagariya, A. and Vemuri, K.},
  booktitle={2024 IEEE 27th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={3773--3778},
  year={2024},
  organization={IEEE},
  doi={10.1109/ITSC58415.2024.10920038}
}
```
