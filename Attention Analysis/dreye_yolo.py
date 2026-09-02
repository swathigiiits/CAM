"""
YOLO SEGMENTATION + DR(EYE)VE ATTENTIVENESS PIPELINE
Replaces SegFormer with custom YOLO11s-seg weights trained on Indian traffic.

Classes: 0=autorickshaw  1=large_vehicle  2=pedestrian  3=road  4=two_wheeler

Layout:
┌─────────────────┬─────────────────┐
│   SCENE + GAZE  │  YOLO SEG       │
│   (crosshair)   │  (overlay)      │
├─────────────────┼─────────────────┤
│   PREDICTED     │  ACTUAL GAZE    │
│   SALIENCY      │  MAP (heatmap)  │
├─────────────────┴─────────────────┤
│         METRICS PANEL             │
└───────────────────────────────────┘
"""

import os
import gc
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import glob
from collections import deque, Counter
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import time
import warnings
from tqdm import tqdm
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing ultralytics YOLO
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLO (ultralytics) available ✓")
except Exception as e:
    logger.warning(f"YOLO not available: {e}")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AriaCleanConfig:
    base_path: Path = Path(r"f:\data_ai")
    subject_dir: Path = Path(r"f:\data_ai")
    csv_path: Path = Path(r"f:\data_ai\general_eye_gaze.csv")
    rgb_video_path: Path = Path(r"f:\data_ai\srivatsav-p20-tada.mp4")
    eye_video_path: Path = Path(r"f:\data_ai\with_data")
    output_dir: Path = Path(r"f:\data_ai\yolo_output")

    # Custom YOLO weights
    yolo_weights_path: Path = Path(r"f:\data_ai\weights\best.pt")

    # GPS / Speed CSV (phone Location CSV)
    speed_csv_path: Optional[Path] = Path(r"f:\data_ai\Location.csv")
    # Aria metadata JSON (contains start_time)
    metadata_json_path: Optional[Path] = Path(r"f:\data_ai\sreevatsav-tada-p21.json")

    # Camera parameters
    focal_length_px: float = 2104.8
    principal_point_x: float = 1440.0
    principal_point_y: float = 1440.0
    rgb_width: int = 2880
    rgb_height: int = 2880

    # FPS from Aria JSON (ground truth)
    rgb_fps: float = 14.984
    eye_fps: float = 29.968
    ET_RGB_RATIO: float = 2.0
    sync_offset_s: float = -0.487

    # Thresholds
    gaze_conf_threshold: float = 0.7
    depth_valid_threshold: float = 3.9
    off_task_duration_threshold: int = 3
    process_every_n_frames: int = 5   # skip frames to keep runtime reasonable

    # Pupil parameters
    pupil_min_mm: float = 2.0
    pupil_max_mm: float = 8.0
    pixel_to_mm_ratio: float = 25.0
    baseline_window_s: float = 30.0
    cognitive_load_window_s: float = 5.0

    gaze_sigma_px: int = 30
    saliency_sigma_px: int = 8
    default_speed_kmh: float = 40.0
    fisheye_mask: bool = True

    # YOLO inference size (pixels) — smaller = faster
    yolo_imgsz: int = 640

    # DR(eye)VE branch weights
    w_semantic: float = 0.35
    w_motion: float = 0.25
    w_center: float = 0.20
    w_cognitive: float = 0.20

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CLASSES AND CONSTANTS
# =============================================================================

# 5 YOLO classes + class 5 = background (no detection)
YOLO_CLASSES = {
    0: 'autorickshaw',
    1: 'large_vehicle',
    2: 'pedestrian',
    3: 'road',
    4: 'two_wheeler',
    5: 'background',
}

TASK_RELEVANCE = {
    'autorickshaw':  'on_task',
    'large_vehicle': 'on_task',
    'pedestrian':    'on_task',
    'road':          'on_task',
    'two_wheeler':   'on_task',
    'background':    'off_task',
}

SEMANTIC_SALIENCY_WEIGHTS = {
    'autorickshaw':  0.85,
    'large_vehicle': 0.85,
    'pedestrian':    0.90,
    'road':          0.70,
    'two_wheeler':   0.80,
    'background':    0.10,
}

YOLO_COLORS = {
    0: (0,  165, 255),   # autorickshaw  — orange
    1: (0,   0,  200),   # large_vehicle — red
    2: (220,  20,  60),  # pedestrian    — crimson
    3: (128,  64, 128),  # road          — purple
    4: (119,  11,  32),  # two_wheeler   — dark maroon
    5: (40,   40,  40),  # background    — dark grey
}

EPSILON = 1e-8


# =============================================================================
# FISHEYE MASK UTILITY
# =============================================================================

def make_fisheye_mask(H: int, W: int, margin: float = 0.02) -> np.ndarray:
    cy, cx = H / 2.0, W / 2.0
    radius = min(cx, cy) * (1.0 - margin)
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return dist <= radius


def apply_fisheye_mask_to_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Use cv2.bitwise_and to avoid large boolean-index temp allocations on big frames
    mask_u8 = (mask * 255).astype(np.uint8)
    return cv2.bitwise_and(frame, frame, mask=mask_u8)


# =============================================================================
# GAZE LOADER
# =============================================================================

class AriaGazeLoader:
    def __init__(self, config: AriaCleanConfig):
        self.config = config

    def load_and_project(self) -> pd.DataFrame:
        logger.info(f"Loading gaze: {self.config.csv_path}")
        df = pd.read_csv(self.config.csv_path)

        df['timestamp_s'] = df['tracking_timestamp_us'] / 1_000_000 + self.config.sync_offset_s
        df['timestamp_s'] -= df['timestamp_s'].min()

        df['yaw']   = (df['left_yaw_rads_cpf'] + df['right_yaw_rads_cpf']) / 2.0
        df['pitch'] = df['pitch_rads_cpf']

        df['gaze_x_px'] = self.config.principal_point_x + self.config.focal_length_px * np.tan(df['yaw'])
        df['gaze_y_px'] = self.config.principal_point_y - self.config.focal_length_px * np.tan(df['pitch'])
        df['gaze_x_raw'] = df['gaze_x_px'].copy()
        df['gaze_y_raw'] = df['gaze_y_px'].copy()
        df['gaze_x_px'] = np.clip(df['gaze_x_px'], 0, self.config.rgb_width  - 1)
        df['gaze_y_px'] = np.clip(df['gaze_y_px'], 0, self.config.rgb_height - 1)
        df['confidence'] = np.where(df['depth_m'] >= self.config.depth_valid_threshold, 0.5, 1.0)

        df = df.sort_values('timestamp_s').reset_index(drop=True)
        valid = (df['confidence'] >= self.config.gaze_conf_threshold).sum()
        logger.info(f"Valid gaze: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")
        return df

    def get_gaze_for_frame(self, gaze_df: pd.DataFrame, frame_timestamp_s: float,
                           window_s: float = 0.5) -> Optional[Dict]:
        mask = np.abs(gaze_df['timestamp_s'] - frame_timestamp_s) <= window_s
        candidates = gaze_df[mask]
        if candidates.empty:
            return None

        high_conf = candidates[candidates['confidence'] >= self.config.gaze_conf_threshold]
        pool = high_conf if not high_conf.empty else candidates
        idx  = (pool['timestamp_s'] - frame_timestamp_s).abs().idxmin()
        row  = pool.loc[idx]

        return {
            'gaze_x_px':   float(row['gaze_x_px']),
            'gaze_y_px':   float(row['gaze_y_px']),
            'gaze_x_raw':  float(row['gaze_x_raw']) if 'gaze_x_raw' in row else float(row['gaze_x_px']),
            'gaze_y_raw':  float(row['gaze_y_raw']) if 'gaze_y_raw' in row else float(row['gaze_y_px']),
            'depth_m':     float(row['depth_m']),
            'confidence':  float(row['confidence']),
            'timestamp_s': float(row['timestamp_s']),
        }


# =============================================================================
# PUPIL ANALYZER
# =============================================================================

class PupilAnalyzer:
    def __init__(self, config: AriaCleanConfig):
        self.config = config
        self.pupil_history = deque(maxlen=int(config.baseline_window_s * config.eye_fps))
        self.baseline_pupil_size = None
        self.baseline_std = None

    def process_eye_frame(self, eye_frame: np.ndarray) -> Dict:
        if eye_frame is None:
            return {'pupil_diameter_mm': 0.0, 'confidence': 0.0, 'is_blink': True}

        h, w = eye_frame.shape[:2]
        left_eye  = eye_frame[:, :w//2]
        right_eye = eye_frame[:, w//2:]

        def process_single(eye_img):
            if eye_img.size == 0:
                return 0.0, True
            gray  = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY) if len(eye_img.shape) == 3 else eye_img.copy()
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray  = clahe.apply(gray)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0, True
            valid = [c for c in contours if 80 < cv2.contourArea(c) < 8000]
            if not valid:
                return 0.0, True
            largest   = max(valid, key=cv2.contourArea)
            area      = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < 0.25:
                return 0.0, True
            diameter_px = 2 * np.sqrt(area / np.pi)
            diameter_mm = np.clip(diameter_px / self.config.pixel_to_mm_ratio,
                                  self.config.pupil_min_mm, self.config.pupil_max_mm)
            return diameter_mm, False

        left_diam,  left_blink  = process_single(left_eye)
        right_diam, right_blink = process_single(right_eye)

        if not left_blink and not right_blink:
            return {'pupil_diameter_mm': (left_diam + right_diam) / 2, 'confidence': 1.0, 'is_blink': False}
        elif not left_blink:
            return {'pupil_diameter_mm': left_diam, 'confidence': 0.7, 'is_blink': False}
        elif not right_blink:
            return {'pupil_diameter_mm': right_diam, 'confidence': 0.7, 'is_blink': False}
        else:
            return {'pupil_diameter_mm': 0.0, 'confidence': 0.0, 'is_blink': True}

    def compute_cognitive_load(self, pupil_diameter_mm: float, timestamp_s: float) -> Dict:
        self.pupil_history.append({'timestamp': timestamp_s, 'diameter': pupil_diameter_mm,
                                   'is_blink': pupil_diameter_mm == 0})

        if self.baseline_pupil_size is None and len(self.pupil_history) > self.config.baseline_window_s * self.config.eye_fps:
            baseline_vals = [p['diameter'] for p in self.pupil_history if p['diameter'] > 0]
            if baseline_vals:
                self.baseline_pupil_size = np.mean(baseline_vals)
                self.baseline_std        = np.std(baseline_vals)

        if self.baseline_pupil_size and self.baseline_std and self.baseline_std > 0:
            norm_dilation = np.clip(max(0, (pupil_diameter_mm - self.baseline_pupil_size) / self.baseline_std), 0, 3)
        else:
            norm_dilation = 0.0

        recent      = list(self.pupil_history)[-int(self.config.cognitive_load_window_s * self.config.eye_fps):]
        valid_diams = [p['diameter'] for p in recent if p['diameter'] > 0]

        if len(valid_diams) > 5:
            cv_raw        = np.std(valid_diams) / (np.mean(valid_diams) + 1e-8)
            cv_normalized = np.clip((cv_raw - 0.1) / 0.4, 0, 1)
        else:
            cv_normalized = 0.0

        cognitive_load = np.clip(0.5 * norm_dilation / 3.0 + 0.5 * cv_normalized, 0, 1)

        return {
            'pupil_diameter_mm': pupil_diameter_mm,
            'baseline_mm':       self.baseline_pupil_size or pupil_diameter_mm,
            'cognitive_load':    float(cognitive_load),
        }

    def compute_perclos(self, window_s: float = 30.0) -> float:
        window_n = int(window_s * self.config.eye_fps)
        recent   = list(self.pupil_history)[-window_n:]
        if not recent:
            return 0.0
        blink_count = sum(1 for p in recent if p.get('is_blink', False) or p['diameter'] == 0)
        return blink_count / len(recent)


# =============================================================================
# SPEED ESTIMATOR
# =============================================================================

class SpeedEstimator:
    def __init__(self, default_speed_kmh: float = 40.0):
        self.default_speed = default_speed_kmh
        self.speed_history = deque(maxlen=30)

    def estimate_from_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        if prev_frame is None:
            return self.default_speed
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # Downscale to 512px max before optical flow to avoid OOM on large frames
        h0, w0 = curr_gray.shape
        FLOW_MAX = 512
        scale = min(1.0, FLOW_MAX / max(h0, w0))
        if scale < 1.0:
            dsize = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
            prev_gray = cv2.resize(prev_gray, dsize, interpolation=cv2.INTER_AREA)
            curr_gray = cv2.resize(curr_gray, dsize, interpolation=cv2.INTER_AREA)
        flow      = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        h, w      = magnitude.shape
        road_region = magnitude[int(h*0.6):, :]
        if road_region.size == 0:
            return self.default_speed
        speed = np.mean(road_region) / 0.15
        self.speed_history.append(speed)
        return float(np.clip(np.mean(self.speed_history), 0, 130))


# =============================================================================
# GPS SPEED LOADER
# =============================================================================

class GPSSpeedLoader:
    def __init__(self, speed_csv_path, metadata_json_path=None):
        self._times  = None
        self._speeds = None
        self._available = False
        if speed_csv_path is None or not Path(speed_csv_path).exists():
            logger.warning("GPS CSV not found — optical-flow fallback.")
            return
        try:
            self._load(Path(speed_csv_path), metadata_json_path)
            self._available = True
            logger.info(f"GPS loaded ✓  samples={len(self._times)}  "
                        f"span={self._times[0]:.1f}–{self._times[-1]:.1f}s  "
                        f"max={self._speeds.max():.1f} km/h")
        except Exception as e:
            logger.warning(f"GPS load failed ({e}) — optical-flow fallback.")

    def _load(self, csv_path, metadata_json_path):
        df = pd.read_csv(csv_path)
        if 'speed' not in df.columns:
            raise ValueError(f"No speed column. Found: {list(df.columns)}")
        df['speed_kmh'] = df['speed'].astype(float) * 3.6
        aria_start = None
        if metadata_json_path and Path(metadata_json_path).exists():
            try:
                with open(metadata_json_path) as f:
                    meta = json.load(f)
                aria_start = float(meta['start_time'])
                logger.info(f"Aria start_time={aria_start:.2f} from JSON")
            except Exception as e:
                logger.warning(f"Could not read start_time: {e}")
        if aria_start is not None and 'time' in df.columns:
            df['aria_t'] = df['time'].astype(float) / 1e9 - aria_start
        elif 'seconds_elapsed' in df.columns:
            df['aria_t'] = df['seconds_elapsed'].astype(float)
            logger.warning("Using seconds_elapsed — timestamps may be offset.")
        else:
            raise ValueError("Need time+JSON or seconds_elapsed column.")
        df = df.sort_values('aria_t').dropna(subset=['aria_t', 'speed_kmh'])
        df = df[df['speed_kmh'] >= 0]
        self._times  = df['aria_t'].to_numpy(dtype=np.float64)
        self._speeds = df['speed_kmh'].to_numpy(dtype=np.float64)

    @property
    def available(self):
        return self._available

    def get_speed_at(self, t):
        if not self._available or t < self._times[0] or t > self._times[-1]:
            return None
        return float(np.interp(t, self._times, self._speeds))


# =============================================================================
# YOLO SEGMENTATION ENGINE  (replaces SegFormerEngine)
# =============================================================================

class YOLOSegEngine:
    """
    Wraps the custom YOLO11s-seg model.
    Converts instance-segmentation output to a per-pixel semantic mask
    compatible with the rest of the DR(eye)VE pipeline.

    Mask values:  0=autorickshaw  1=large_vehicle  2=pedestrian
                  3=road          4=two_wheeler     5=background
    """

    def __init__(self, config: AriaCleanConfig):
        self.config = config
        self.model  = None
        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        logger.info(f"Loading YOLO weights: {self.config.yolo_weights_path}")
        # Force CPU to avoid CUBLAS init errors on some GPU configurations
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        self.model = YOLO(str(self.config.yolo_weights_path))
        logger.info(f"YOLO loaded — classes: {self.model.names}")

    def segment(self, bgr_frame: np.ndarray, fisheye_mask: np.ndarray = None) -> np.ndarray:
        H, W = bgr_frame.shape[:2]
        # Default: everything is background (class 5)
        semantic_mask = np.full((H, W), 5, dtype=np.uint8)

        if self.model is None:
            return self._mock_segment(bgr_frame)

        # Mask fisheye black border so YOLO doesn't fire on black corners
        frame_in = apply_fisheye_mask_to_frame(bgr_frame, fisheye_mask) \
                   if fisheye_mask is not None else bgr_frame

        results = self.model(
            frame_in,
            imgsz=self.config.yolo_imgsz,
            verbose=False,
            conf=0.25,
            device='cpu',
        )[0]

        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()   # (N, H', W')
            classes    = results.boxes.cls.cpu().numpy().astype(int)
            confs      = results.boxes.conf.cpu().numpy()

            # Paint lowest-confidence first so highest-conf wins overlaps
            order = np.argsort(confs)
            for idx in order:
                cls_id = int(classes[idx])
                if cls_id > 4:
                    continue
                m = masks_data[idx].astype(np.float32)
                # Resize to original frame size if needed
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                semantic_mask[m > 0.5] = cls_id

        # Restore fisheye border to background
        if fisheye_mask is not None:
            semantic_mask[~fisheye_mask] = 5

        return semantic_mask

    def _mock_segment(self, bgr_frame: np.ndarray) -> np.ndarray:
        H, W = bgr_frame.shape[:2]
        mask = np.full((H, W), 5, dtype=np.uint8)
        mask[int(H * 0.5):, :] = 3   # road
        return mask


# =============================================================================
# CENTROID TRACKER
# =============================================================================

@dataclass
class TrackedObject:
    obj_id:     int
    class_id:   int
    class_name: str
    centroid:   Tuple[float, float]
    velocity:   Tuple[float, float]
    mask:       np.ndarray
    age:        int = 0


class CentroidTracker:
    # Track all YOLO classes except road (3) and background (5)
    TRACK_CLASSES = {0, 1, 2, 4}

    def __init__(self, max_distance: float = 150.0, max_age: int = 5):
        self.max_distance = max_distance
        self.max_age      = max_age
        self.next_id      = 0
        self.tracked: List[TrackedObject] = []

    def update(self, semantic_mask: np.ndarray) -> List[TrackedObject]:
        detections = self._extract_detections(semantic_mask)

        if not self.tracked:
            for det in detections:
                self.tracked.append(TrackedObject(
                    obj_id=self.next_id, class_id=det['class_id'],
                    class_name=det['class_name'], centroid=det['centroid'],
                    velocity=(0.0, 0.0), mask=det['mask']
                ))
                self.next_id += 1
            return self.tracked

        matched, unmatched_dets, unmatched_trks = self._match(detections)
        updated = []

        for det_idx, trk_idx in matched:
            old = self.tracked[trk_idx]
            det = detections[det_idx]
            vx  = det['centroid'][0] - old.centroid[0]
            vy  = det['centroid'][1] - old.centroid[1]
            updated.append(TrackedObject(
                obj_id=old.obj_id, class_id=det['class_id'],
                class_name=det['class_name'], centroid=det['centroid'],
                velocity=(vx, vy), mask=det['mask'], age=old.age + 1
            ))

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            updated.append(TrackedObject(
                obj_id=self.next_id, class_id=det['class_id'],
                class_name=det['class_name'], centroid=det['centroid'],
                velocity=(0.0, 0.0), mask=det['mask']
            ))
            self.next_id += 1

        for trk_idx in unmatched_trks:
            old = self.tracked[trk_idx]
            if old.age < self.max_age:
                updated.append(old)

        self.tracked = updated
        return self.tracked

    def _extract_detections(self, mask: np.ndarray) -> List[Dict]:
        detections = []
        H, W = mask.shape
        MIN_AREA = {0: 400, 1: 800, 2: 300, 4: 400}

        for cls_id in self.TRACK_CLASSES:
            binary   = (mask == cls_id).astype(np.uint8)
            min_area = MIN_AREA.get(cls_id, 400)
            if binary.sum() < min_area:
                continue
            num_labels, labels = cv2.connectedComponents(binary)
            for label in range(1, num_labels):
                component = (labels == label).astype(np.uint8)
                area = component.sum()
                if area < min_area:
                    continue
                ys, xs = np.where(component)
                if np.mean(ys) < H * 0.3:
                    continue
                detections.append({
                    'class_id':   cls_id,
                    'class_name': YOLO_CLASSES[cls_id],
                    'centroid':   (float(np.mean(xs)), float(np.mean(ys))),
                    'mask':       component.astype(bool),
                    'area':       area,
                })
        return detections

    def _match(self, detections):
        matched, used_trk, used_det = [], set(), set()
        for det_idx, det in enumerate(detections):
            best_dist, best_trk = self.max_distance, -1
            for trk_idx, trk in enumerate(self.tracked):
                if trk_idx in used_trk or trk.class_id != det['class_id']:
                    continue
                dist = np.hypot(det['centroid'][0] - trk.centroid[0],
                                det['centroid'][1] - trk.centroid[1])
                if dist < best_dist:
                    best_dist, best_trk = dist, trk_idx
            if best_trk >= 0:
                matched.append((det_idx, best_trk))
                used_det.add(det_idx)
                used_trk.add(best_trk)
        unmatched_dets = [i for i in range(len(detections)) if i not in used_det]
        unmatched_trks = [i for i in range(len(self.tracked)) if i not in used_trk]
        return matched, unmatched_dets, unmatched_trks


# =============================================================================
# DR(EYE)VE SALIENCY PREDICTOR
# =============================================================================

class DreyeveSaliencyPredictor:
    def __init__(self, config: AriaCleanConfig):
        self.config = config
        # Index 0-5; index 5 = background
        self._class_weight = np.array([
            SEMANTIC_SALIENCY_WEIGHTS.get(YOLO_CLASSES.get(i, 'background'), 0.1)
            for i in range(6)
        ], dtype=np.float32)
        self._prev_gray = None

    def predict(self, semantic_mask: np.ndarray, curr_frame: np.ndarray,
                cognitive_load: float = 0.0, speed_kmh: float = 40.0) -> np.ndarray:
        H, W = semantic_mask.shape

        # Semantic branch — all float32 to minimise peak RAM
        sem_map = self._class_weight[semantic_mask.clip(0, 5)].astype(np.float32)
        y_coords = np.linspace(0, 1, H, dtype=np.float32)[:, None] * np.ones((1, W), dtype=np.float32)
        x_coords = np.linspace(0, 1, W, dtype=np.float32)[None, :] * np.ones((H, 1), dtype=np.float32)
        center_decay = np.exp(-((x_coords - 0.5)**2 + (y_coords - 0.45)**2) / np.float32(0.125), dtype=np.float32)
        del x_coords, y_coords
        sem_map = sem_map * (np.float32(0.5) + np.float32(0.5) * center_decay)
        del center_decay
        sem_map = gaussian_filter(sem_map.astype(np.float64), sigma=10).astype(np.float32)

        # Motion branch
        motion_map = self._motion_branch(curr_frame, H, W)
        motion_map = gaussian_filter(motion_map.astype(np.float64), sigma=8).astype(np.float32)

        # Center / vanishing-point branch
        center_map = self._center_branch(H, W, speed_kmh, cognitive_load)
        center_map = gaussian_filter(center_map.astype(np.float64), sigma=12).astype(np.float32)

        def norm01(m):
            mn, mx = m.min(), m.max()
            return ((m - mn) / (mx - mn + np.float32(1e-8))).astype(np.float32)

        sem_map    = norm01(sem_map)
        motion_map = norm01(motion_map)
        center_map = norm01(center_map)

        saliency = (np.float32(self.config.w_semantic) * sem_map +
                    np.float32(self.config.w_motion)   * motion_map +
                    np.float32(self.config.w_center)   * center_map)
        del sem_map, motion_map, center_map
        saliency = gaussian_filter(saliency.astype(np.float64),
                                   sigma=self.config.saliency_sigma_px).astype(np.float32)

        total = saliency.sum()
        if total > 0:
            saliency /= total
        return saliency

    def _motion_branch(self, curr_frame: np.ndarray, H: int, W: int) -> np.ndarray:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # Downscale to 512px max-dim to keep optical flow memory manageable
        FLOW_MAX = 512
        scale = min(1.0, FLOW_MAX / max(H, W))
        fH, fW = max(1, int(H * scale)), max(1, int(W * scale))
        curr_small = cv2.resize(curr_gray, (fW, fH), interpolation=cv2.INTER_AREA)

        if self._prev_gray is None:
            self._prev_gray = curr_small
            return np.zeros((H, W), dtype=np.float32)

        # Resize prev to match in case frame size changed
        prev_small = cv2.resize(self._prev_gray, (fW, fH), interpolation=cv2.INTER_AREA) \
                     if self._prev_gray.shape != (fH, fW) else self._prev_gray

        flow      = cv2.calcOpticalFlowFarneback(prev_small, curr_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        upper_region = magnitude[:int(fH * 0.30), :]
        ego_motion   = np.median(upper_region) if upper_region.size > 0 else np.median(magnitude)
        magnitude    = np.maximum(0, magnitude - ego_motion * 0.8)
        if magnitude.max() > 0:
            magnitude /= magnitude.max()
        # Upsample back to full frame size
        magnitude = cv2.resize(magnitude, (W, H), interpolation=cv2.INTER_LINEAR)
        self._prev_gray = curr_small
        return magnitude.astype(np.float32)

    def _center_branch(self, H: int, W: int, speed_kmh: float, cognitive_load: float) -> np.ndarray:
        vp_y    = np.clip(0.30 + 0.10 * (speed_kmh / 100.0) + 0.05 * cognitive_load, 0.25, 0.50)
        vp_y_px = int(vp_y * H)
        speed_ratio = max(0.3, 1.0 - speed_kmh / 150.0)
        sigma_x = np.float32(W * 0.25 * speed_ratio)
        sigma_y = np.float32(H * 0.20 * speed_ratio)
        y, x    = np.mgrid[0:H, 0:W]
        x = x.astype(np.float32); y = y.astype(np.float32)
        center  = np.exp(-(((x - np.float32(W/2))**2) / (2 * sigma_x**2) +
                           ((y - np.float32(vp_y_px))**2) / (2 * sigma_y**2)))
        if center.max() > 0:
            center /= center.max()
        return center.astype(np.float32)


# =============================================================================
# DR(EYE)VE METRICS
# =============================================================================

def center_gaussian_baseline(H: int, W: int) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H * 0.40
    sx, sy = W * 0.25, H * 0.20
    baseline = np.exp(-(((x - cx)**2) / (2 * sx**2) + ((y - cy)**2) / (2 * sy**2))).astype(np.float32)
    return baseline / baseline.sum()


def compute_dkl(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    y = Y.flatten().astype(np.float64);  p = Y_hat.flatten().astype(np.float64)
    if y.sum() > 0: y /= y.sum()
    if p.sum() > 0: p /= p.sum()
    return float(np.clip(np.sum(y * np.log((y + EPSILON) / (p + EPSILON))), 0, 50))


def compute_cc(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    y = Y.flatten().astype(np.float64);  p = Y_hat.flatten().astype(np.float64)
    if y.std() < 1e-10 or p.std() < 1e-10:
        return 0.0
    cc, _ = pearsonr(y, p)
    return float(np.clip(cc, -1, 1))


def compute_ig(P: np.ndarray, Y: np.ndarray, B: np.ndarray) -> float:
    p = P.flatten().astype(np.float64)
    y = Y.flatten().astype(np.float64)
    b = B.flatten().astype(np.float64)
    for arr in [p, y, b]:
        s = arr.sum()
        if s > 0: arr /= s
    fix_locs = y > y.mean()
    if not fix_locs.any():
        return 0.0
    return float(np.mean(np.log2(p[fix_locs] + EPSILON) - np.log2(b[fix_locs] + EPSILON)))


def compute_nss(P: np.ndarray, Y: np.ndarray) -> float:
    p = P.flatten().astype(np.float64);  y = Y.flatten().astype(np.float64)
    if p.std() < EPSILON:
        return 0.0
    p_norm   = (p - p.mean()) / (p.std() + EPSILON)
    fix_locs = y > y.mean()
    if not fix_locs.any():
        return 0.0
    return float(np.mean(p_norm[fix_locs]))


def build_gaze_map(gaze_x: float, gaze_y: float, H: int, W: int, sigma: int = 60) -> np.ndarray:
    gaze_map = np.zeros((H, W), dtype=np.float32)
    ix, iy = int(round(gaze_x)), int(round(gaze_y))
    if 0 <= ix < W and 0 <= iy < H:
        gaze_map[iy, ix] = 1.0
    gaze_map = gaussian_filter(gaze_map, sigma=sigma)
    total = gaze_map.sum()
    if total > 0:
        gaze_map /= total
    return gaze_map


# =============================================================================
# ATTENTION STATE CLASSIFICATION
# =============================================================================

def classify_state(cc: float, on_task: bool, perclos: float, gaze_ratio: float,
                   cognitive_load: float, pupil_diameter: float) -> str:
    if perclos > 0.15 or (pupil_diameter < 3.0 and cognitive_load < 0.2):
        return 'DROWSY'
    if cognitive_load > 0.7 and on_task:
        return 'HIGH_COGNITIVE_LOAD'
    if gaze_ratio < 0.35 and on_task:
        return 'SCANNING_DISTANCE' if cc > 0.2 else 'SCANNING'
    if cc > 0.5 and on_task:
        return 'FOCUSED'
    if cc > 0.2 and on_task:
        return 'ATTENTIVE'
    if not on_task and cc < 0.1:
        return 'DISTRACTED'
    return 'MARGINAL'


def compute_score(cc: float, on_task: bool, perclos: float, gaze_ratio: float,
                  cognitive_load: float) -> float:
    cc_norm       = (cc + 1) / 2
    ontask_score  = 1.0 if on_task else 0.0
    perclos_score = 1.0 - perclos
    scanning_bonus = 1.0 if (gaze_ratio < 0.35 and on_task) else 0.0
    cog_score = 0.3 if cognitive_load < 0.2 else (0.5 if cognitive_load > 0.7 else 1.0)
    score = (0.30 * cc_norm + 0.25 * ontask_score + 0.15 * perclos_score +
             0.10 * scanning_bonus + 0.20 * cog_score) * 100
    return float(np.clip(score, 0, 100))


# =============================================================================
# VISUALIZATION
# =============================================================================

def colorize_mask(semantic_mask: np.ndarray, draw_boundaries: bool = True) -> np.ndarray:
    H, W = semantic_mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in YOLO_COLORS.items():
        color_mask[semantic_mask == cls_id] = color

    if draw_boundaries:
        h_edge = (semantic_mask[:-1, :] != semantic_mask[1:, :])
        v_edge = (semantic_mask[:, :-1] != semantic_mask[:, 1:])
        boundary = np.zeros((H, W), dtype=bool)
        boundary[:-1, :] |= h_edge
        boundary[1:,  :] |= h_edge
        boundary[:, :-1] |= v_edge
        boundary[:, 1:]  |= v_edge
        color_mask[boundary] = (255, 255, 255)

    return color_mask


def draw_yolo_legend(panel: np.ndarray) -> np.ndarray:
    """Draw a small YOLO class legend in the top-right corner of the seg panel."""
    out  = panel.copy()
    x0, y0, sw, sh, pad = out.shape[1] - 220, 40, 14, 14, 4
    for i, (cls_id, name) in enumerate(YOLO_CLASSES.items()):
        if cls_id == 5:  # skip background in legend
            continue
        color = YOLO_COLORS[cls_id]
        y = y0 + i * (sh + pad)
        cv2.rectangle(out, (x0, y), (x0 + sw, y + sh), color, -1)
        cv2.putText(out, name, (x0 + sw + 4, y + sh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def create_clean_visualization(
    rgb_frame:      np.ndarray,
    semantic_mask:  np.ndarray,
    gaze_x:         float,
    gaze_y:         float,
    saliency_map:   np.ndarray,
    gaze_map:       np.ndarray,
    attended_class: str,
    on_task:        bool,
    state:          str,
    score:          float,
    metrics:        Dict,
    frame_idx:      int,
    timestamp_s:    float,
    pupil_data:     Dict,
) -> np.ndarray:
    H, W       = rgb_frame.shape[:2]
    panel_size = 640
    sx, sy     = panel_size / W, panel_size / H

    # ── PANEL 1: SCENE + GAZE CROSSHAIR ───────────────────────────────────
    scene_panel = cv2.resize(rgb_frame, (panel_size, panel_size))
    gx, gy      = int(gaze_x * sx), int(gaze_y * sy)
    gaze_color  = (0, 255, 0) if on_task else (0, 0, 255)
    ch_size     = 30
    cv2.line(scene_panel, (gx - ch_size, gy), (gx + ch_size, gy), gaze_color, 3)
    cv2.line(scene_panel, (gx, gy - ch_size), (gx, gy + ch_size), gaze_color, 3)
    cv2.circle(scene_panel, (gx, gy), 8, (255, 255, 255), -1)
    cv2.circle(scene_panel, (gx, gy), 4, gaze_color, -1)
    label_bg = cv2.getTextSize(attended_class.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(scene_panel, (gx + 15, gy - 25), (gx + 15 + label_bg[0] + 10, gy - 5), (0, 0, 0), -1)
    cv2.putText(scene_panel, attended_class.upper(), (gx + 20, gy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)

    # ── PANEL 2: YOLO SEGMENTATION OVERLAY ────────────────────────────────
    color_mask  = colorize_mask(semantic_mask, draw_boundaries=True)
    seg_panel   = cv2.resize(color_mask, (panel_size, panel_size))
    scene_small = cv2.resize(rgb_frame, (panel_size, panel_size))
    seg_panel   = cv2.addWeighted(scene_small, 0.40, seg_panel, 0.60, 0)
    seg_panel   = draw_yolo_legend(seg_panel)
    cv2.circle(seg_panel, (gx, gy), 15, (255, 255, 255), 2)
    cv2.circle(seg_panel, (gx, gy), 5, gaze_color, -1)
    cv2.putText(seg_panel, "YOLO SEGMENTATION", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── PANEL 3: PREDICTED SALIENCY ───────────────────────────────────────
    sal_norm  = np.clip(saliency_map - np.percentile(saliency_map, 20), 0, None)
    sal_vis   = (sal_norm / (sal_norm.max() + 1e-8) * 255).astype(np.uint8)
    sal_panel = cv2.applyColorMap(sal_vis, cv2.COLORMAP_JET)
    sal_panel = cv2.resize(sal_panel, (panel_size, panel_size))
    cv2.putText(sal_panel, "PREDICTED SALIENCY", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── PANEL 4: ACTUAL GAZE MAP ──────────────────────────────────────────
    gaze_vis   = (gaze_map / (gaze_map.max() + 1e-8) * 255).astype(np.uint8)
    gaze_panel = cv2.applyColorMap(gaze_vis, cv2.COLORMAP_HOT)
    gaze_panel = cv2.resize(gaze_panel, (panel_size, panel_size))
    cv2.putText(gaze_panel, "ACTUAL GAZE MAP", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── METRICS PANEL (bottom) ────────────────────────────────────────────
    metrics_height, metrics_width = 220, panel_size * 2
    metrics_panel = np.full((metrics_height, metrics_width, 3), (30, 30, 40), dtype=np.uint8)

    state_colors = {
        'FOCUSED': (0, 220, 100), 'ATTENTIVE': (100, 200, 100),
        'SCANNING': (0, 180, 255), 'SCANNING_DISTANCE': (0, 200, 200),
        'HIGH_COGNITIVE_LOAD': (255, 100, 0), 'MARGINAL': (200, 200, 0),
        'DISTRACTED': (255, 50, 50), 'DROWSY': (180, 0, 255),
    }
    sc = state_colors.get(state, (200, 200, 200))

    bar_width = int((score / 100.0) * (metrics_width - 100))
    cv2.rectangle(metrics_panel, (50, 25), (50 + bar_width, 60), sc, -1)
    cv2.rectangle(metrics_panel, (50, 25), (metrics_width - 50, 60), (100, 100, 100), 2)
    score_text = f"ATTENTION SCORE: {score:.1f}/100"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(metrics_panel, score_text, (metrics_width - score_size[0] - 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    pupil_mm = pupil_data.get('pupil_diameter_mm', 0)
    cog_load = pupil_data.get('cognitive_load', 0)
    lines = [
        (f"STATE: {state}", sc),
        (f"CC: {metrics['CC']:.3f} | DKL: {metrics['DKL']:.2f} | IG: {metrics['IG']:.3f} | NSS: {metrics['NSS']:.3f}",
         (160, 200, 255)),
        (f"GAZE → {attended_class.upper()} ({'ON-TASK' if on_task else 'OFF-TASK'})",
         (0, 255, 0) if on_task else (0, 0, 255)),
        (f"PUPIL: {pupil_mm:.2f}mm | COG LOAD: {cog_load:.2f} | PERCLOS: {metrics.get('perclos', 0):.3f}",
         (255, 200, 100)),
        (f"GPS: {metrics.get('gps_speed', -1):.1f} km/h | FLOW: {metrics.get('flow_speed', 0):.1f} km/h | "
         f"FRAME: {frame_idx:04d} | TIME: {timestamp_s:.1f}s",
         (100, 255, 180) if metrics.get('gps_speed', -1) >= 0 else (180, 180, 100)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(metrics_panel, text, (50, 95 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── COMBINE ───────────────────────────────────────────────────────────
    top_row    = np.hstack([scene_panel, seg_panel])
    bottom_row = np.hstack([sal_panel, gaze_panel])
    output     = np.vstack([top_row, bottom_row, metrics_panel])
    cv2.line(output, (panel_size, 0), (panel_size, panel_size * 2), (255, 255, 255), 2)
    cv2.line(output, (0, panel_size), (panel_size * 2, panel_size), (255, 255, 255), 2)
    return output


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class AriaCleanPipeline:
    def __init__(self, config: AriaCleanConfig):
        self.config        = config
        self.gaze_loader   = AriaGazeLoader(config)
        self.yolo          = YOLOSegEngine(config)      # ← YOLO replaces SegFormer
        self.tracker       = CentroidTracker()
        self.saliency      = DreyeveSaliencyPredictor(config)
        self.pupil_analyzer = PupilAnalyzer(config)
        self.speed_estimator = SpeedEstimator(config.default_speed_kmh)
        self.gps_loader    = GPSSpeedLoader(
            speed_csv_path=config.speed_csv_path,
            metadata_json_path=config.metadata_json_path,
        )
        self.gaze_df  = None
        self.baseline = None

    def run(self):
        logger.info("=" * 65)
        logger.info("YOLO + DR(EYE)VE ATTENTIVENESS PIPELINE")
        logger.info("=" * 65)
        t0 = time.time()

        self.gaze_df = self.gaze_loader.load_and_project()

        cap_rgb = cv2.VideoCapture(str(self.config.rgb_video_path))
        if not cap_rgb.isOpened():
            logger.error(f"Cannot open RGB video: {self.config.rgb_video_path}")
            return [], {}

        actual_rgb_fps = self.config.rgb_fps
        ET_RGB_RATIO   = self.config.ET_RGB_RATIO

        # Load ET frames from folder
        et_folder      = Path(self.config.eye_video_path)
        et_frames_list = []
        eye_available  = False
        if et_folder.is_dir():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                et_frames_list += sorted(glob.glob(str(et_folder / ext)))
            if et_frames_list:
                eye_available = True
                logger.info(f"ET folder: {len(et_frames_list)} frames  ratio={ET_RGB_RATIO:.1f}")

        rgb_total = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"RGB: fps={actual_rgb_fps:.4f}  frames={rgb_total}  dur={rgb_total/actual_rgb_fps:.1f}s")
        logger.info(f"Gaze CSV: dur={self.gaze_df['timestamp_s'].max():.1f}s")
        logger.info(f"Processing every {self.config.process_every_n_frames} frame(s)")

        fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer      = None   # 4-panel video
        seg_writer      = None   # YOLO seg-only video
        SEG_FPS         = 5.0    # lower fps for seg-only output

        results    = []
        prev_frame = None
        frame_idx  = 0
        off_task_run = 0
        fisheye_mask_cache       = None
        actual_frame_size_logged = False

        pbar = tqdm(total=rgb_total, desc="Processing")

        while True:
            # Skip frames efficiently: grab() reads compressed data only (no decode)
            if frame_idx % self.config.process_every_n_frames != 0:
                grabbed = cap_rgb.grab()
                if not grabbed:
                    break
                frame_idx += 1
                pbar.update(1)
                continue

            ret_rgb, frame = cap_rgb.read()
            if not ret_rgb:
                break

            timestamp_s = frame_idx / actual_rgb_fps

            # ET frame (folder-based)
            eye_frame = None
            if eye_available and et_frames_list:
                et_idx    = min(int(frame_idx * ET_RGB_RATIO), len(et_frames_list) - 1)
                eye_frame = cv2.imread(et_frames_list[et_idx])

            # Pupil
            if eye_frame is not None:
                pupil_raw  = self.pupil_analyzer.process_eye_frame(eye_frame)
                pupil_data = self.pupil_analyzer.compute_cognitive_load(
                    pupil_raw['pupil_diameter_mm'], timestamp_s)
                pupil_data.update(pupil_raw)
                perclos = self.pupil_analyzer.compute_perclos(30.0)
            else:
                pupil_data = {'pupil_diameter_mm': 0.0, 'cognitive_load': 0.0, 'baseline_mm': 0.0}
                perclos    = 0.0

            # Gaze
            gaze = self.gaze_loader.get_gaze_for_frame(self.gaze_df, timestamp_s)
            if gaze is None or gaze['confidence'] < self.config.gaze_conf_threshold:
                frame_idx += 1
                prev_frame = frame
                pbar.update(1)
                continue

            gaze_x, gaze_y = gaze['gaze_x_px'], gaze['gaze_y_px']

            # Fisheye mask (computed once)
            if fisheye_mask_cache is None or fisheye_mask_cache.shape != frame.shape[:2]:
                fisheye_mask_cache = make_fisheye_mask(frame.shape[0], frame.shape[1])
                if not actual_frame_size_logged:
                    fH, fW = frame.shape[:2]
                    logger.info(f"Frame size: {fW}x{fH}")
                    actual_frame_size_logged = True

            fmask = fisheye_mask_cache if self.config.fisheye_mask else None

            # ── YOLO SEGMENTATION ─────────────────────────────────────────
            semantic_mask = self.yolo.segment(frame, fmask)
            orig_H, orig_W = semantic_mask.shape

            # ── DOWNSCALE to 320×320 for all downstream ops ───────────────
            # Reduces peak RAM ~4× vs 640×640 on a memory-constrained machine.
            WORK_SZ = 320
            if orig_H != WORK_SZ or orig_W != WORK_SZ:
                semantic_mask = cv2.resize(semantic_mask, (WORK_SZ, WORK_SZ),
                                           interpolation=cv2.INTER_NEAREST)
                frame_work    = cv2.resize(frame, (WORK_SZ, WORK_SZ),
                                           interpolation=cv2.INTER_AREA)
            else:
                frame_work = frame
            H, W = WORK_SZ, WORK_SZ

            # Rescale gaze to working resolution
            scale_x = WORK_SZ / self.config.rgb_width
            scale_y = WORK_SZ / self.config.rgb_height
            gaze_x  = float(np.clip(gaze['gaze_x_raw'] * scale_x, 0, W - 1))
            gaze_y  = float(np.clip(gaze['gaze_y_raw'] * scale_y, 0, H - 1))

            # Tracking
            self.tracker.update(semantic_mask)

            # Speed
            flow_speed      = self.speed_estimator.estimate_from_flow(prev_frame, frame)
            gps_speed       = self.gps_loader.get_speed_at(timestamp_s)
            estimated_speed = gps_speed if gps_speed is not None else flow_speed

            # Saliency (on working resolution)
            pred_saliency = self.saliency.predict(
                semantic_mask, frame_work,
                cognitive_load=pupil_data.get('cognitive_load', 0.0),
                speed_kmh=estimated_speed,
            )

            # Gaze map
            actual_gaze_map = build_gaze_map(gaze_x, gaze_y, H, W, self.config.gaze_sigma_px)

            # Baseline for IG
            if self.baseline is None or self.baseline.shape != (H, W):
                self.baseline = center_gaussian_baseline(H, W)

            # Downsample for metric computation (avoid minute-long pearsonr on 2880x2880)
            METRIC_H, METRIC_W = 256, 256
            sal_small  = cv2.resize(pred_saliency,    (METRIC_W, METRIC_H))
            gaze_small = cv2.resize(actual_gaze_map,  (METRIC_W, METRIC_H))
            base_small = cv2.resize(self.baseline,    (METRIC_W, METRIC_H))

            metrics = {
                'CC':         compute_cc(gaze_small, sal_small),
                'DKL':        compute_dkl(gaze_small, sal_small),
                'IG':         compute_ig(sal_small, gaze_small, base_small),
                'NSS':        compute_nss(sal_small, gaze_small),
                'perclos':    perclos,
                'speed':      estimated_speed,
                'gps_speed':  gps_speed if gps_speed is not None else -1.0,
                'flow_speed': flow_speed,
            }

            # Attended class from gaze location
            ix, iy    = int(np.clip(gaze_x, 0, W-1)), int(np.clip(gaze_y, 0, H-1))
            raw_cls   = int(semantic_mask[iy, ix])
            cls_id    = min(raw_cls, 5)
            attended_class = YOLO_CLASSES.get(cls_id, 'background')
            on_task   = TASK_RELEVANCE.get(attended_class, 'off_task') == 'on_task'

            off_task_run      = 0 if on_task else off_task_run + 1
            distraction_event = off_task_run >= self.config.off_task_duration_threshold

            gaze_ratio  = gaze_y / H
            state       = classify_state(metrics['CC'], on_task, perclos, gaze_ratio,
                                         pupil_data.get('cognitive_load', 0),
                                         pupil_data.get('pupil_diameter_mm', 0))
            attn_score  = compute_score(metrics['CC'], on_task, perclos, gaze_ratio,
                                        pupil_data.get('cognitive_load', 0))

            results.append({
                'frame_idx':        frame_idx,
                'timestamp_s':      timestamp_s,
                'gaze_x_px':        gaze_x,
                'gaze_y_px':        gaze_y,
                'attended_class':   attended_class,
                'on_task':          on_task,
                'distraction_event':distraction_event,
                'state':            state,
                'attention_score':  attn_score,
                'CC':               metrics['CC'],
                'DKL':              metrics['DKL'],
                'IG':               metrics['IG'],
                'NSS':              metrics['NSS'],
                'pupil_diameter_mm':pupil_data.get('pupil_diameter_mm', 0),
                'cognitive_load':   pupil_data.get('cognitive_load', 0),
                'perclos':          perclos,
                'estimated_speed':  estimated_speed,
            })

            vis = create_clean_visualization(
                frame_work, semantic_mask, gaze_x, gaze_y, pred_saliency,
                actual_gaze_map, attended_class, on_task, state,
                attn_score, metrics, frame_idx, timestamp_s, pupil_data,
            )

            # ── 4-PANEL VIDEO ─────────────────────────────────────────────
            if out_writer is None:
                h, w = vis.shape[:2]
                out_writer = cv2.VideoWriter(
                    str(self.config.output_dir / "yolo output for 100 epochs.mp4"),
                    fourcc, actual_rgb_fps, (w, h),
                )

            out_writer.write(vis)

            # ── SEG-ONLY VIDEO (YOLO mask panel, output at 640×640) ───────
            SEG_SZ          = 640
            color_mask_full = colorize_mask(semantic_mask, draw_boundaries=True)
            seg_only        = cv2.resize(color_mask_full, (SEG_SZ, SEG_SZ))
            scene_sm        = cv2.resize(frame_work, (SEG_SZ, SEG_SZ))
            seg_only        = cv2.addWeighted(scene_sm, 0.40, seg_only, 0.60, 0)
            del color_mask_full, scene_sm
            seg_only        = draw_yolo_legend(seg_only)
            gx_s = int(gaze_x * SEG_SZ / WORK_SZ)
            gy_s = int(gaze_y * SEG_SZ / WORK_SZ)
            gaze_color_s = (0, 255, 0) if on_task else (0, 0, 255)
            cv2.circle(seg_only, (gx_s, gy_s), 15, (255, 255, 255), 2)
            cv2.circle(seg_only, (gx_s, gy_s), 5, gaze_color_s, -1)
            cv2.putText(seg_only, "YOLO SEG - 100 EPOCHS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(seg_only, f"t={timestamp_s:.1f}s  {attended_class.upper()}",
                        (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

            if seg_writer is None:
                seg_writer = cv2.VideoWriter(
                    str(self.config.output_dir / "yolo_seg_mask_only.mp4"),
                    fourcc, SEG_FPS, (SEG_SZ, SEG_SZ),
                )

            seg_writer.write(seg_only)

            # Periodically free unreferenced objects to prevent RAM accumulation
            if frame_idx % 100 == 0:
                gc.collect()

            prev_frame  = frame_work.copy()
            frame_idx  += 1
            pbar.update(1)

        pbar.close()
        cap_rgb.release()
        if out_writer:
            out_writer.release()
        if seg_writer:
            seg_writer.release()

        self._save_results(results)
        self._print_summary(results, time.time() - t0)
        return results

    def _save_results(self, results: List[Dict]):
        out_dir = self.config.output_dir
        with open(out_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        if results:
            pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
        logger.info(f"Results saved to {out_dir}")

    def _print_summary(self, results: List[Dict], elapsed: float):
        if not results:
            return
        print("\n" + "=" * 65)
        print("YOLO ATTENTIVENESS PIPELINE - SUMMARY")
        print("=" * 65)
        print(f"\nFRAMES PROCESSED: {len(results)}")
        print(f"\nATTENTION SCORE: {np.mean([r['attention_score'] for r in results]):.1f}/100")
        print(f"\nMETRICS:")
        print(f"   CC:  {np.mean([r['CC'] for r in results]):.3f}")
        print(f"   DKL: {np.mean([r['DKL'] for r in results]):.3f}")
        print(f"   IG:  {np.mean([r['IG'] for r in results]):.3f}")
        print(f"   NSS: {np.mean([r['NSS'] for r in results]):.3f}")
        print(f"\nON-TASK: {100 * np.mean([r['on_task'] for r in results]):.1f}%")
        valid_pupils = [r['pupil_diameter_mm'] for r in results if r['pupil_diameter_mm'] > 0]
        if valid_pupils:
            print(f"PUPIL: {np.mean(valid_pupils):.2f}mm")
        print(f"\nSTATES:")
        for state, count in Counter([r['state'] for r in results]).most_common():
            print(f"   {state}: {count}")
        print(f"\nATTENDED CLASSES:")
        for cls, count in Counter([r['attended_class'] for r in results]).most_common():
            print(f"   {cls}: {count}")
        print(f"\nTime: {elapsed:.1f}s")
        print("=" * 65)
        print(f"Output 1 (4-panel): {self.config.output_dir}/yolo output for 100 epochs.mp4")
        print(f"Output 2 (seg-only): {self.config.output_dir}/yolo_seg_mask_only.mp4")
        print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO + DR(eye)VE Attentiveness Pipeline")
    parser.add_argument("--csv",       default=r"f:\data_ai\general_eye_gaze.csv")
    parser.add_argument("--video",     default=r"f:\data_ai\srivatsav-p20-tada.mp4")
    parser.add_argument("--eye-dir",   default=r"f:\data_ai\with_data")
    parser.add_argument("--weights",   default=r"f:\data_ai\weights\best.pt")
    parser.add_argument("--output",    default=r"f:\data_ai\yolo_output")
    parser.add_argument("--speed-csv", default=r"f:\data_ai\Location.csv")
    parser.add_argument("--metadata",  default=r"f:\data_ai\sreevatsav-tada-p21.json")
    parser.add_argument("--every-n",   type=int, default=5,
                        help="Process 1 of every N frames (default 5)")
    parser.add_argument("--imgsz",     type=int, default=640,
                        help="YOLO inference size in pixels (default 640)")
    args = parser.parse_args()

    config = AriaCleanConfig(
        csv_path           = Path(args.csv),
        rgb_video_path     = Path(args.video),
        eye_video_path     = Path(args.eye_dir),
        yolo_weights_path  = Path(args.weights),
        output_dir         = Path(args.output),
        speed_csv_path     = Path(args.speed_csv)   if args.speed_csv else None,
        metadata_json_path = Path(args.metadata)    if args.metadata  else None,
        process_every_n_frames = args.every_n,
        yolo_imgsz         = args.imgsz,
    )

    pipeline = AriaCleanPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
