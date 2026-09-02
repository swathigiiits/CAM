# myEye2Wheeler Dataset

## Overview
**myEye2Wheeler** is a first-of-its-kind, large-scale, real-world eye-tracking dataset focused on motorized two-wheeler riders navigating the highly dense, heterogeneous, and unstructured traffic conditions typical of Indian roads. The dataset contains **261,073 gaze-annotated video frames** collected from 40 participants (21 novices and 19 experienced riders) navigating a complex 3.9 km urban route in Hyderabad.

## Dataset Description
The dataset contains real-world eye-tracking and video data from motorized two-wheeler riders. 

The data was recorded using Tobii Pro Glasses 2 integrated inside standard helmets. It includes:
*   **Video Data:** High-resolution scene camera video (1920x1080 at 25 fps).
*   **Gaze Coordinates:** Synchronized frame-by-frame gaze coordinates (sampled at 50 Hz).
*   **Bounding Boxes:** Pre-computed bounding box annotations generated via YOLOv5 for vehicles and pedestrians.
*   **Gaze Targets:** Gaze target classifications for each frame, categorized as `vehicle`, `person`, or `on/off road`.
*   **Metadata:** Participant metadata (age, gender, and years of riding experience).

All personally identifiable information, including faces and vehicle license plates, has been obfuscated using the Understand AI anonymizer to ensure privacy. The dataset captures diverse traffic scenarios including signal stops, U-turns, merging traffic, and interactions with heterogeneous traffic elements (pedestrians, varying vehicle types).

## Directory Structure
The dataset is structured primarily by the rider's experience level:
```text
myEye2Wheeler/
├── Experienced/
│   ├── Participant_01_Age_Exp_Gender/
│   │   ├── video.mp4
│   │   ├── gaze_data.csv
│   │   └── ...
│   └── ...
└── Novice/
    ├── Participant_22_Age_Exp_Gender/
    │   ├── video.mp4
    │   ├── gaze_data.csv
    │   └── ...
    └── ...
```

## Data Format

### Gaze Data (`.csv`)
Each participant's folder contains a CSV file with the frame-by-frame gaze data and environmental context. The columns typically include:
*   `Participant No.`: Unique identifier for the participant.
*   `Frame No.`: The frame number corresponding to the video.
*   `Time`: Timestamp of the frame.
*   `Total Vehicles`: Number of vehicles detected in the frame via YOLOv5.
*   `Total Pedestrians`: Number of pedestrians detected in the frame via YOLOv5.
*   `Gaze Target`: The categorized object the rider is looking at (`vehicle`, `person`, or `on/off road`).

## Usage
This dataset is designed to advance research in human-computer interaction, transportation safety, and computer vision. Primary use cases include:
*   **Saliency Modeling:** Training and benchmarking predictive computational visual attention models (e.g., TASED-Net) tailored for unstructured, non-lane-based traffic environments.
*   **ADAS Development:** Informing the design of context-aware Advanced Driver Assistance Systems (ADAS) and hazard alert systems specifically tailored for the dynamic movement of two-wheelers.
*   **Behavioral Analysis:** Studying scan paths and gaze transition dynamics to understand how driving experience influences situational awareness and cognitive load.
*   **Rider Training & Policy:** Developing objective, gaze-informed metrics for rider skill assessment, simulator training, and data-driven urban infrastructure planning.

## Citation
If you use this dataset in your research, please cite the following paper:
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

## Contact
For any questions or issues regarding the dataset, please contact Bhaiya Vaibhaw Kumar at IIIT Hyderabad.
