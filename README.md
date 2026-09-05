# CAM: Driver Cognitive Attention & Eye-Gaze Analytics

A multimodal research platform and toolchain for **driver visual attention modeling, gaze-to-object semantic segmentation, pupil dynamics, and sensor telemetry extraction** designed specifically for dense, heterogeneous, and unstructured traffic environments.

---

## 📌 Repository Structure

```text
CAM/
├── Attention Analysis/          # Gaze attention analytics & semantic mapping pipelines
│   ├── dreye_final.py           # SegFormer + DR(EYE)VE attentiveness & 2x2 grid visualization
│   ├── dreye_yolo.py            # YOLO11s-seg + DR(EYE)VE attention mapping pipeline
│   ├── Location.csv             # GPS telemetry trajectory stream
│   └── general_eye_gaze.csv     # Extracted 2D/3D gaze tracking coordinate records
│
├── Data extraction/             # Meta Project Aria .vrs data extraction utilities
│   ├── SHOW_EYE.py              # Real-time gaze & eye camera region visualizer
│   ├── analyze_vrs_profiles.py  # Profile parser and sensor integrity analyzer
│   ├── batch_pupil_detect.py    # Automated batch pupil diameter detector
│   ├── extract_all_et_images.py # Synchronized frame extraction from raw .vrs files
│   └── quick_rotate_fix.py      # Camera frame rotation and alignment helper to extract scene data
│
├── smart-helmet-agent/          # Full-stack Smart Helmet driver attention analytics agent platform
│   ├── backend/                 # FastAPI backend (analytics, gaze homography, coaching engine)
│   ├── frontend/                # React + Vite + Tailwind interactive UI dashboard
│   └── README.md                # Full feature set & API reference
│
├── weights/                     # Pre-trained model weights
│   └── best.pt                  # Custom trained YOLO11s-seg weights (Data50 Indian traffic)
│
└── .gitignore                   # Repository filtering rules
```

---

## 🛠️ Pipelines & Core Features

### 1. Attention Analysis (`Attention Analysis/`)
* **YOLO11s-Seg Attention Mapping (`dreye_yolo.py`)**:
  * Semantic segmentation of 5 custom Indian traffic classes:
    * `0`: `autorickshaw`
    * `1`: `large_vehicle`
    * `2`: `pedestrian`
    * `3`: `road`
    * `4`: `two_wheeler`
  * Dynamic intersection mapping between gaze fixations $(x, y)$ and detected object masks.
* **SegFormer + DR(EYE)VE Pipeline (`dreye_final.py`)**:
  * Multi-panel 2x2 visualization:
    * **Top-Left:** Scene RGB + Gaze Crosshair
    * **Top-Right:** Semantic Segmentation Overlay
    * **Bottom-Left:** Predicted Saliency Map
    * **Bottom-Right:** Actual Gaze Heatmap + Real-time Metrics Panel
* **Cognitive Attention Metrics**:
  * Fixation identification using I-VT velocity thresholds and dwell time quantification.
  * Spatial scanpath analysis and transition entropy across high-density hazard areas.

### 2. Sensor Extraction & VRS Utilities (`Data extraction/`)
* **Meta Project Aria Stream Parser (`analyze_vrs_profiles.py`)**: Parses multi-stream `.vrs` records (RGB scene camera, dual infrared eye tracking sensors, IMU, GPS).
* **Batch Frame Extraction (`extract_all_et_images.py`)**: Synchronizes and extracts high-resolution scene video frames and corresponding eye camera crops.
* **Pupil Tracking (`batch_pupil_detect.py`, `SHOW_EYE.py`)**: Measures frame-level pupil detection for data quality.

### 3. Smart Helmet Analytics Agent (`smart-helmet-agent/`)
* **Full-Stack Safety Assistant**: Combines real-time gaze homography calibration, OpenCV scene understanding, and longitudinal driver risk tracking.
* **Cognitive Scoring & Coaching**:
  * Calculates composite attention scores (0–100), road vs off-road glance durations, and hazard scanning coverage.
  * Generates rule-based & LLM-powered personalized driver coaching reports.
* **Interactive Dashboard**: Responsive React/Vite UI for uploading driver sessions, streaming scene video overlays, and reviewing driver progress trends.


---

## 🚀 Setup & Installation

### Prerequisites
* Python 3.8+
* PyTorch & Torchvision (CUDA enabled recommended)
* Ultralytics (YOLO11)
* Hugging Face Transformers (SegFormer)
* OpenCV & NumPy

```bash
# Clone repository
git clone https://github.com/swaaathi1/CAM.git
cd CAM

# Install dependencies
pip install opencv-python numpy pandas matplotlib ultralytics transformers scipy tqdm
```

### Running the Attention Pipelines

```bash
# Run YOLO11s-seg Attention Mapping
python "Attention Analysis/dreye_yolo.py"

# Run SegFormer + DR(EYE)VE Attention Visualizer
python "Attention Analysis/dreye_final.py"

# Inspect Pupil and Eye Regions
python "Data extraction/SHOW_EYE.py" --image "/path/to/eye_frame.jpg"
```

---

## 📊 Associated Datasets & Assets

* **`Data50`**: Custom annotated YOLO segmentation and detection dataset for 5 Indian urban traffic classes.
* **`et_data`**: Driver recording sessions captured using Meta Project Aria glasses (raw `.vrs` files, sensor telemetry sidecars, and MPS gaze records).

---

## 📬 Contact & Citation

For questions, issues, or collaborations regarding this toolchain and dataset, please open an issue on GitHub ([@swathigiiits](https://github.com/swathigiiits)).
