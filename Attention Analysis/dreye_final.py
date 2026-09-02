"""
CLEAN ARIA + SEGFORMER + DR(EYE)VE ATTENTIVENESS PIPELINE
Clean 2x2 grid visualization (NO bounding boxes) with full attention modeling

Layout:
┌─────────────────┬─────────────────┐
│   SCENE + GAZE  │  SEGMENTATION   │
│   (crosshair)   │  (overlay)      │
├─────────────────┼─────────────────┤
│   PREDICTED     │  ACTUAL GAZE    │
│   SALIENCY      │  MAP (heatmap)  │
├─────────────────┴─────────────────┤
│         METRICS PANEL              │
└───────────────────────────────────┘
"""

import os
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
import json
import time
import warnings
from tqdm import tqdm
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing SegFormer
SEGFORMER_AVAILABLE = False
try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    import torch
    from PIL import Image
    SEGFORMER_AVAILABLE = True
    logger.info("SegFormer available ✓")
except Exception as e:
    logger.warning(f"SegFormer not available: {e}")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AriaCleanConfig:
    base_path: Path = Path(r"E:\aria_work")
    subject_dir: Path = Path(r"E:\aria_work\sree_dt")
    csv_path: Path = Path(r"E:\aria_work\sree_dt\sreevatsav-tada-p21.csv\general_eye_gaze.csv")
    rgb_video_path: Path = Path(r"E:\aria_work\sree_dt\srivatsav-p20-tada.mp4")
    eye_video_path: Path = Path(r"E:\aria_work\sree_dt\with_data")   # folder of ET frames
    output_dir: Path = Path(r"E:\aria_work\sree_dt\clean_output")

    # GPS / Speed CSV (phone Location CSV)
    speed_csv_path: Optional[Path] = Path(r"E:\aria_work\sree_dt\Location.csv")
    # Aria metadata JSON (contains start_time)
    metadata_json_path: Optional[Path] = Path(r"E:\aria_work\sree_dt\sreevatsav-tada-p21.json")

    # Camera parameters
    focal_length_px: float = 2104.8
    principal_point_x: float = 1440.0
    principal_point_y: float = 1440.0
    rgb_width: int = 2880
    rgb_height: int = 2880

    # FPS from Aria JSON (ground truth):
    #   rgb_camera.expected = 18236 / 1217s = 14.984 fps
    #   et_camera.expected  = 36471 / 1217s = 29.968 fps
    #   ET/RGB ratio = 36471/18236 = 2.0 exactly
    rgb_fps: float = 14.984
    eye_fps: float = 29.968
    ET_RGB_RATIO: float = 2.0   # et_frame_idx = rgb_frame_idx * 2
    sync_offset_s: float = -0.487
    
    # Thresholds
    gaze_conf_threshold: float = 0.7
    depth_valid_threshold: float = 3.9
    off_task_duration_threshold: int = 3
    process_every_n_frames: int = 1
    
    # Pupil parameters
    pupil_min_mm: float = 2.0
    pupil_max_mm: float = 8.0
    pixel_to_mm_ratio: float = 25.0
    baseline_window_s: float = 30.0
    cognitive_load_window_s: float = 5.0
    
    # Model — upgraded B0→B2 for sharper boundaries
    segformer_model_name: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    gaze_sigma_px: int = 30          # FIX: was 60 — too wide, smears actual gaze map
    saliency_sigma_px: int = 8       # FIX: was 40 — catastrophically wide, caused blob saliency
    default_speed_kmh: float = 40.0

    # Fisheye handling
    fisheye_mask: bool = True        # FIX: mask circular black border before segmentation

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

CITYSCAPES_CLASSES = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

TASK_RELEVANCE = {
    'road': 'on_task', 'sidewalk': 'off_task', 'building': 'off_task',
    'wall': 'off_task', 'fence': 'off_task', 'pole': 'on_task',
    'traffic_light': 'on_task', 'traffic_sign': 'on_task', 'vegetation': 'off_task',
    'terrain': 'on_task', 'sky': 'off_task',  # FIX: sky is not driving-relevant
    'person': 'on_task',
    'rider': 'on_task', 'car': 'on_task', 'truck': 'on_task',
    'bus': 'on_task', 'train': 'on_task', 'motorcycle': 'on_task', 'bicycle': 'on_task'
}

SEMANTIC_SALIENCY_WEIGHTS = {
    'road': 0.70, 'sidewalk': 0.05, 'building': 0.02, 'wall': 0.02,
    'fence': 0.03, 'pole': 0.30, 'traffic_light': 0.90, 'traffic_sign': 0.80,
    'vegetation': 0.02, 'terrain': 0.60, 'sky': 0.40, 'person': 0.90,
    'rider': 0.95, 'car': 0.85, 'truck': 0.85, 'bus': 0.85,
    'train': 0.85, 'motorcycle': 0.90, 'bicycle': 0.80
}

CITYSCAPES_COLORS = {
    0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70), 3: (102, 102, 156),
    4: (190, 153, 153), 5: (153, 153, 153), 6: (250, 170, 30), 7: (220, 220, 0),
    8: (107, 142, 35), 9: (152, 251, 152), 10: (135, 206, 235), 11: (220, 20, 60),
    12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70), 15: (0, 60, 100),
    16: (0, 80, 100), 17: (0, 0, 230), 18: (119, 11, 32)
}

EPSILON = 1e-8


# =============================================================================
# FISHEYE MASK UTILITY
# =============================================================================

def make_fisheye_mask(H: int, W: int, margin: float = 0.02) -> np.ndarray:
    """
    Returns a boolean mask (True = valid pixels) for a circular fisheye image.
    Pixels outside the inscribed circle are set False (black border region).
    margin: shrinks the circle slightly to avoid edge artifacts.
    """
    cy, cx = H / 2.0, W / 2.0
    radius = min(cx, cy) * (1.0 - margin)
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return dist <= radius


def apply_fisheye_mask_to_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the fisheye circle before passing to SegFormer."""
    out = frame.copy()
    out[~mask] = 0
    return out


# =============================================================================
# GAZE LOADER (with timestamp fix)
# =============================================================================

class AriaGazeLoader:
    def __init__(self, config: AriaCleanConfig):
        self.config = config
    
    def load_and_project(self) -> pd.DataFrame:
        logger.info(f"Loading gaze: {self.config.csv_path}")
        df = pd.read_csv(self.config.csv_path)
        
        df['timestamp_s'] = df['tracking_timestamp_us'] / 1_000_000 + self.config.sync_offset_s
        df['timestamp_s'] -= df['timestamp_s'].min()  # FIX: normalize to start at 0
        
        df['yaw'] = (df['left_yaw_rads_cpf'] + df['right_yaw_rads_cpf']) / 2.0
        df['pitch'] = df['pitch_rads_cpf']
        
        # Project using config dimensions (2880x2880)
        # These will be rescaled per-frame if video delivers different resolution
        df['gaze_x_px'] = self.config.principal_point_x + self.config.focal_length_px * np.tan(df['yaw'])
        df['gaze_y_px'] = self.config.principal_point_y - self.config.focal_length_px * np.tan(df['pitch'])  # minus: image-y increases downward
        # Store raw (unclipped) so we can rescale to actual frame size later
        df['gaze_x_raw'] = df['gaze_x_px'].copy()
        df['gaze_y_raw'] = df['gaze_y_px'].copy()
        df['gaze_x_px'] = np.clip(df['gaze_x_px'], 0, self.config.rgb_width  - 1)
        df['gaze_y_px'] = np.clip(df['gaze_y_px'], 0, self.config.rgb_height - 1)
        df['confidence'] = np.where(df['depth_m'] >= self.config.depth_valid_threshold, 0.5, 1.0)
        
        df = df.sort_values('timestamp_s').reset_index(drop=True)
        valid = (df['confidence'] >= self.config.gaze_conf_threshold).sum()
        logger.info(f"Valid gaze: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")
        
        return df
    
    def get_gaze_for_frame(self, gaze_df: pd.DataFrame, frame_timestamp_s: float, window_s: float = 0.5) -> Optional[Dict]:
        mask = np.abs(gaze_df['timestamp_s'] - frame_timestamp_s) <= window_s
        candidates = gaze_df[mask]
        if candidates.empty:
            return None
        
        high_conf = candidates[candidates['confidence'] >= self.config.gaze_conf_threshold]
        pool = high_conf if not high_conf.empty else candidates
        idx = (pool['timestamp_s'] - frame_timestamp_s).abs().idxmin()
        row = pool.loc[idx]
        
        return {
            'gaze_x_px':  float(row['gaze_x_px']),
            'gaze_y_px':  float(row['gaze_y_px']),
            'gaze_x_raw': float(row['gaze_x_raw']) if 'gaze_x_raw' in row else float(row['gaze_x_px']),
            'gaze_y_raw': float(row['gaze_y_raw']) if 'gaze_y_raw' in row else float(row['gaze_y_px']),
            'depth_m':    float(row['depth_m']),
            'confidence': float(row['confidence']),
            'timestamp_s':float(row['timestamp_s'])
        }


# =============================================================================
# PUPIL ANALYZER (with left/right eye separation)
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
        left_eye = eye_frame[:, :w//2]
        right_eye = eye_frame[:, w//2:]
        
        def process_single(eye_img):
            if eye_img.size == 0:
                return 0.0, True
            # Aria ET frames are grayscale IR
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY) if len(eye_img.shape) == 3 else eye_img.copy()
            # CLAHE to enhance dark pupil contrast in IR image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            # Otsu on CLAHE-enhanced frame — more robust than fixed percentile for IR
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Morphological close to fill pupil boundary gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0, True
            # Area range for Aria ET pupil in pixels
            valid = [c for c in contours if 80 < cv2.contourArea(c) < 8000]
            if not valid:
                return 0.0, True
            largest = max(valid, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            # Relaxed circularity — pupil appears elliptical at extreme gaze angles
            if circularity < 0.25:
                return 0.0, True
            diameter_px = 2 * np.sqrt(area / np.pi)
            diameter_mm = diameter_px / self.config.pixel_to_mm_ratio
            diameter_mm = np.clip(diameter_mm, self.config.pupil_min_mm, self.config.pupil_max_mm)
            return diameter_mm, False
        
        left_diam, left_blink = process_single(left_eye)
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
        self.pupil_history.append({'timestamp': timestamp_s, 'diameter': pupil_diameter_mm, 'is_blink': pupil_diameter_mm == 0})
        
        if self.baseline_pupil_size is None and len(self.pupil_history) > self.config.baseline_window_s * self.config.eye_fps:
            baseline_vals = [p['diameter'] for p in self.pupil_history if p['diameter'] > 0]
            if baseline_vals:
                self.baseline_pupil_size = np.mean(baseline_vals)
                self.baseline_std = np.std(baseline_vals)
        
        if self.baseline_pupil_size and self.baseline_std and self.baseline_std > 0:
            norm_dilation = max(0, (pupil_diameter_mm - self.baseline_pupil_size) / self.baseline_std)
            norm_dilation = np.clip(norm_dilation, 0, 3)
        else:
            norm_dilation = 0.0
        
        recent = list(self.pupil_history)[-int(self.config.cognitive_load_window_s * self.config.eye_fps):]
        valid_diams = [p['diameter'] for p in recent if p['diameter'] > 0]
        
        if len(valid_diams) > 5:
            cv_raw = np.std(valid_diams) / (np.mean(valid_diams) + 1e-8)
            cv_normalized = np.clip((cv_raw - 0.1) / 0.4, 0, 1)
        else:
            cv_normalized = 0.0
        
        cognitive_load = np.clip(0.5 * norm_dilation / 3.0 + 0.5 * cv_normalized, 0, 1)
        
        return {
            'pupil_diameter_mm': pupil_diameter_mm,
            'baseline_mm': self.baseline_pupil_size or pupil_diameter_mm,
            'cognitive_load': float(cognitive_load)
        }
    
    def compute_perclos(self, window_s: float = 30.0) -> float:
        window_n = int(window_s * self.config.eye_fps)
        recent = list(self.pupil_history)[-window_n:]
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
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        h, w = magnitude.shape
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
    """
    Loads phone Location CSV, aligns to Aria timeline via start_time from JSON.
    phone time column = nanosecond Unix timestamp.
    aria_relative_s = (phone_time_ns / 1e9) - aria_start_unix_s
    Falls back to seconds_elapsed if JSON unavailable.
    """
    def __init__(self, speed_csv_path, metadata_json_path=None):
        self._times = None
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
        df = df.sort_values('aria_t').dropna(subset=['aria_t','speed_kmh'])
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
# SEGFORMER ENGINE
# =============================================================================

class SegFormerEngine:
    def __init__(self, config: AriaCleanConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.device = 'cpu'
        if SEGFORMER_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        logger.info(f"Loading SegFormer: {self.config.segformer_model_name}")
        self.processor = SegformerImageProcessor.from_pretrained(self.config.segformer_model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.config.segformer_model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model = self.model.to(self.device)
    
    def segment(self, bgr_frame: np.ndarray, fisheye_mask: np.ndarray = None) -> np.ndarray:
        if not SEGFORMER_AVAILABLE or self.model is None:
            return self._mock_segment(bgr_frame)

        H, W = bgr_frame.shape[:2]

        # Step 1: mask fisheye black border so SegFormer doesn't see black corners
        frame_in = apply_fisheye_mask_to_frame(bgr_frame, fisheye_mask)                    if fisheye_mask is not None else bgr_frame

        # Step 2: let processor handle resize to 512 (its native size for b2-1024 model)
        # Do NOT manually resize or pass do_resize=False — processor knows the correct
        # padding/normalisation for this model
        rgb     = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs  = self.processor(images=pil_img, return_tensors="pt")
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Step 3: upsample logits back to original frame size with bilinear
        # then argmax — same as original working script
        logits    = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(H, W), mode='bilinear', align_corners=False
        )
        mask = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Step 4: re-mask fisheye border to sky (class 10 = neutral)
        if fisheye_mask is not None:
            mask[~fisheye_mask] = 10

        # Step 5: morphological cleanup
        mask = self._clean_mask(mask)

        return mask

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Per-class morphological opening to remove small isolated noise blobs.
        Opening = erosion then dilation: removes specks smaller than kernel.
        """
        cleaned = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for cls_id in range(19):
            binary = (mask == cls_id).astype(np.uint8)
            if binary.sum() < 500:
                continue
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            lost = (binary == 1) & (opened == 0)
            cleaned[lost] = 255
        sentinel = cleaned == 255
        if sentinel.any():
            from scipy.ndimage import distance_transform_edt
            _, idx = distance_transform_edt(sentinel, return_indices=True)
            cleaned[sentinel] = cleaned[idx[0][sentinel], idx[1][sentinel]]
        return cleaned
    
    def _mock_segment(self, bgr_frame: np.ndarray) -> np.ndarray:
        H, W = bgr_frame.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[:int(H*0.35), :] = 10
        mask[int(H*0.35):int(H*0.65), :] = 0
        mask[int(H*0.65):, :] = 13
        return mask


# =============================================================================
# TRACKER (Minimal - only for saliency, no visualization)
# =============================================================================

@dataclass
class TrackedObject:
    obj_id: int
    class_id: int
    class_name: str
    centroid: Tuple[float, float]
    velocity: Tuple[float, float]
    mask: np.ndarray
    age: int = 0


class CentroidTracker:
    TRACK_CLASSES = {11, 12, 13, 14, 15, 16, 17, 18}
    
    def __init__(self, max_distance: float = 150.0, max_age: int = 5):
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 0
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
            vx = det['centroid'][0] - old.centroid[0]
            vy = det['centroid'][1] - old.centroid[1]
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
        MIN_AREA = {11: 400, 12: 400, 13: 800, 14: 1200, 15: 1200, 16: 1200, 17: 500, 18: 500}
        
        for cls_id in self.TRACK_CLASSES:
            binary = (mask == cls_id).astype(np.uint8)
            min_area = MIN_AREA.get(cls_id, 600)
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
                    'class_id': cls_id,
                    'class_name': CITYSCAPES_CLASSES[cls_id],
                    'centroid': (float(np.mean(xs)), float(np.mean(ys))),
                    'mask': component.astype(bool),
                    'area': area
                })
        return detections
    
    def _match(self, detections):
        matched, unmatched_dets, unmatched_trks = [], [], []
        used_trk, used_det = set(), set()
        
        for det_idx, det in enumerate(detections):
            best_dist, best_trk = self.max_distance, -1
            for trk_idx, trk in enumerate(self.tracked):
                if trk_idx in used_trk or trk.class_id != det['class_id']:
                    continue
                dist = np.hypot(det['centroid'][0] - trk.centroid[0], det['centroid'][1] - trk.centroid[1])
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
        self._class_weight = np.array([
            SEMANTIC_SALIENCY_WEIGHTS.get(CITYSCAPES_CLASSES.get(i, 'road'), 0.1)
            for i in range(19)
        ], dtype=np.float32)
        self._prev_gray = None
    
    def predict(self, semantic_mask: np.ndarray, curr_frame: np.ndarray,
                cognitive_load: float = 0.0, speed_kmh: float = 40.0) -> np.ndarray:
        H, W = semantic_mask.shape

        # Semantic branch — FIX: filter per-branch with small sigma, not globally after combine
        sem_map = self._class_weight[semantic_mask.clip(0, 18)].astype(np.float32)
        # Suppress flat dominant classes (road covers whole bottom — dilutes saliency)
        # Apply distance-weighted decay: closer to image center = higher weight
        y_coords = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
        x_coords = np.linspace(0, 1, W)[None, :] * np.ones((H, 1))
        center_decay = np.exp(-((x_coords - 0.5) ** 2 + (y_coords - 0.45) ** 2) / (2 * 0.25 ** 2))
        sem_map = sem_map * (0.5 + 0.5 * center_decay)
        # Per-branch small sigma
        sem_map = gaussian_filter(sem_map, sigma=10)

        # Motion branch
        motion_map = self._motion_branch(curr_frame, H, W)
        motion_map = gaussian_filter(motion_map, sigma=8)

        # Center branch
        center_map = self._center_branch(H, W, speed_kmh, cognitive_load)
        center_map = gaussian_filter(center_map, sigma=12)

        # Normalize each branch to [0,1] before combining — prevents one branch dominating
        def norm01(m):
            mn, mx = m.min(), m.max()
            return (m - mn) / (mx - mn + 1e-8)

        sem_map    = norm01(sem_map)
        motion_map = norm01(motion_map)
        center_map = norm01(center_map)

        # Combine
        saliency = (self.config.w_semantic * sem_map +
                    self.config.w_motion   * motion_map +
                    self.config.w_center   * center_map)

        # FIX: final blur is tiny (was sigma=40, now sigma=config.saliency_sigma_px=8)
        saliency = gaussian_filter(saliency, sigma=self.config.saliency_sigma_px)

        total = saliency.sum()
        if total > 0:
            saliency /= total
        return saliency.astype(np.float32)
    
    def _motion_branch(self, curr_frame: np.ndarray, H: int, W: int) -> np.ndarray:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = curr_gray
            return np.zeros((H, W), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # FIX: use upper 30% of frame (sky/horizon) as ego-motion reference
        # instead of full-frame median — avoids canceling all real object motion
        upper_region = magnitude[:int(H * 0.30), :]
        ego_motion = np.median(upper_region) if upper_region.size > 0 else np.median(magnitude)

        magnitude = np.maximum(0, magnitude - ego_motion * 0.8)
        if magnitude.max() > 0:
            magnitude /= magnitude.max()

        self._prev_gray = curr_gray
        return magnitude.astype(np.float32)
    
    def _center_branch(self, H: int, W: int, speed_kmh: float, cognitive_load: float) -> np.ndarray:
        vp_y = 0.30 + 0.10 * (speed_kmh / 100.0) + 0.05 * cognitive_load
        vp_y = np.clip(vp_y, 0.25, 0.50)
        vp_y_px = int(vp_y * H)
        
        speed_ratio = max(0.3, 1.0 - speed_kmh / 150.0)
        sigma_x = W * 0.25 * speed_ratio
        sigma_y = H * 0.20 * speed_ratio
        
        y, x = np.mgrid[0:H, 0:W]
        center = np.exp(-(((x - W/2)**2) / (2 * sigma_x**2) + ((y - vp_y_px)**2) / (2 * sigma_y**2)))
        
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
    baseline = np.exp(-(((x - cx)**2) / (2 * sx**2) + ((y - cy)**2) / (2 * sy**2)))
    baseline = baseline.astype(np.float32)
    return baseline / baseline.sum()


def compute_dkl(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    y = Y.flatten().astype(np.float64)
    p = Y_hat.flatten().astype(np.float64)
    if y.sum() > 0:
        y /= y.sum()
    if p.sum() > 0:
        p /= p.sum()
    return float(np.clip(np.sum(y * np.log((y + EPSILON) / (p + EPSILON))), 0, 50))


def compute_cc(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    y = Y.flatten().astype(np.float64)
    p = Y_hat.flatten().astype(np.float64)
    if y.std() < 1e-10 or p.std() < 1e-10:
        return 0.0
    cc, _ = pearsonr(y, p)
    return float(np.clip(cc, -1, 1))


def compute_ig(P: np.ndarray, Y: np.ndarray, B: np.ndarray) -> float:
    p = P.flatten().astype(np.float64)
    y = Y.flatten().astype(np.float64)
    b = B.flatten().astype(np.float64)
    
    if p.sum() > 0:
        p /= p.sum()
    if y.sum() > 0:
        y /= y.sum()
    if b.sum() > 0:
        b /= b.sum()
    
    fix_locs = y > y.mean()
    if not fix_locs.any():
        return 0.0
    
    ig = np.mean(np.log2(p[fix_locs] + EPSILON) - np.log2(b[fix_locs] + EPSILON))
    return float(ig)


def compute_nss(P: np.ndarray, Y: np.ndarray) -> float:
    p = P.flatten().astype(np.float64)
    y = Y.flatten().astype(np.float64)
    if p.std() < EPSILON:
        return 0.0
    p_norm = (p - p.mean()) / (p.std() + EPSILON)
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


def compute_score(cc: float, on_task: bool, perclos: float, gaze_ratio: float, cognitive_load: float) -> float:
    cc_norm = (cc + 1) / 2
    ontask_score = 1.0 if on_task else 0.0
    perclos_score = 1.0 - perclos
    scanning_bonus = 1.0 if (gaze_ratio < 0.35 and on_task) else 0.0
    
    if cognitive_load < 0.2:
        cog_score = 0.3
    elif cognitive_load > 0.7:
        cog_score = 0.5
    else:
        cog_score = 1.0
    
    score = (0.30 * cc_norm + 0.25 * ontask_score + 0.15 * perclos_score +
             0.10 * scanning_bonus + 0.20 * cog_score) * 100
    return float(np.clip(score, 0, 100))


# =============================================================================
# CLEAN VISUALIZATION (2x2 grid from seg.py)
# =============================================================================

def colorize_mask(semantic_mask: np.ndarray, draw_boundaries: bool = True) -> np.ndarray:
    H, W = semantic_mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in CITYSCAPES_COLORS.items():
        color_mask[semantic_mask == cls_id] = color

    # FIX: draw hard white boundary lines between classes for visual clarity
    if draw_boundaries:
        # Detect where class label changes horizontally or vertically
        h_edge = (semantic_mask[:-1, :] != semantic_mask[1:, :])
        v_edge = (semantic_mask[:, :-1] != semantic_mask[:, 1:])
        boundary = np.zeros((H, W), dtype=bool)
        boundary[:-1, :] |= h_edge
        boundary[1:,  :] |= h_edge
        boundary[:, :-1] |= v_edge
        boundary[:, 1:]  |= v_edge
        # Draw boundaries as white lines
        color_mask[boundary] = (255, 255, 255)

    return color_mask


def create_clean_visualization(
    rgb_frame: np.ndarray,
    semantic_mask: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    saliency_map: np.ndarray,
    gaze_map: np.ndarray,
    attended_class: str,
    on_task: bool,
    state: str,
    score: float,
    metrics: Dict,
    frame_idx: int,
    timestamp_s: float,
    pupil_data: Dict,
) -> np.ndarray:
    """Clean 2x2 grid visualization - NO bounding boxes"""
    H, W = rgb_frame.shape[:2]
    panel_size = 640
    sx, sy = panel_size / W, panel_size / H
    
    # ── PANEL 1: SCENE + GAZE CROSSHAIR ───────────────────────────────
    scene_panel = cv2.resize(rgb_frame, (panel_size, panel_size))
    gx, gy = int(gaze_x * sx), int(gaze_y * sy)
    gaze_color = (0, 255, 0) if on_task else (0, 0, 255)
    
    # Crosshair
    ch_size = 30
    cv2.line(scene_panel, (gx - ch_size, gy), (gx + ch_size, gy), gaze_color, 3)
    cv2.line(scene_panel, (gx, gy - ch_size), (gx, gy + ch_size), gaze_color, 3)
    cv2.circle(scene_panel, (gx, gy), 8, (255, 255, 255), -1)
    cv2.circle(scene_panel, (gx, gy), 4, gaze_color, -1)
    
    # Class label near gaze
    label_bg = cv2.getTextSize(attended_class.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(scene_panel, (gx + 15, gy - 25), (gx + 15 + label_bg[0] + 10, gy - 5), (0, 0, 0), -1)
    cv2.putText(scene_panel, attended_class.upper(), (gx + 20, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
    
    # ── PANEL 2: SEGMENTATION OVERLAY ─────────────────────────────────
    color_mask = colorize_mask(semantic_mask, draw_boundaries=True)
    seg_panel = cv2.resize(color_mask, (panel_size, panel_size))
    scene_small = cv2.resize(rgb_frame, (panel_size, panel_size))
    # FIX: was 0.3/0.7 — colors were washed out. 0.45/0.55 keeps scene context + vivid classes
    seg_panel = cv2.addWeighted(scene_small, 0.45, seg_panel, 0.55, 0)
    cv2.circle(seg_panel, (gx, gy), 15, (255, 255, 255), 2)
    cv2.circle(seg_panel, (gx, gy), 5, gaze_color, -1)
    
    # ── PANEL 3: PREDICTED SALIENCY ───────────────────────────────────
    # Clip bottom 20% to suppress dark blue background, keep JET colormap
    sal_norm = saliency_map.copy()
    sal_norm = np.clip(sal_norm - np.percentile(sal_norm, 20), 0, None)
    sal_vis  = (sal_norm / (sal_norm.max() + 1e-8) * 255).astype(np.uint8)
    sal_panel = cv2.applyColorMap(sal_vis, cv2.COLORMAP_JET)
    sal_panel = cv2.resize(sal_panel, (panel_size, panel_size))
    cv2.putText(sal_panel, "PREDICTED SALIENCY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # ── PANEL 4: ACTUAL GAZE MAP ──────────────────────────────────────
    gaze_vis = (gaze_map / (gaze_map.max() + 1e-8) * 255).astype(np.uint8)
    gaze_panel = cv2.applyColorMap(gaze_vis, cv2.COLORMAP_HOT)
    gaze_panel = cv2.resize(gaze_panel, (panel_size, panel_size))
    cv2.putText(gaze_panel, "ACTUAL GAZE MAP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # ── METRICS PANEL (Bottom) ────────────────────────────────────────
    metrics_height, metrics_width = 220, panel_size * 2
    metrics_panel = np.zeros((metrics_height, metrics_width, 3), dtype=np.uint8)
    metrics_panel[:] = (30, 30, 40)
    
    state_colors = {
        'FOCUSED': (0, 220, 100), 'ATTENTIVE': (100, 200, 100),
        'SCANNING': (0, 180, 255), 'SCANNING_DISTANCE': (0, 200, 200),
        'HIGH_COGNITIVE_LOAD': (255, 100, 0), 'MARGINAL': (200, 200, 0),
        'DISTRACTED': (255, 50, 50), 'DROWSY': (180, 0, 255)
    }
    sc = state_colors.get(state, (200, 200, 200))
    
    # Score bar
    bar_width = int((score / 100.0) * (metrics_width - 100))
    cv2.rectangle(metrics_panel, (50, 25), (50 + bar_width, 60), sc, -1)
    cv2.rectangle(metrics_panel, (50, 25), (metrics_width - 50, 60), (100, 100, 100), 2)
    
    score_text = f"ATTENTION SCORE: {score:.1f}/100"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(metrics_panel, score_text, (metrics_width - score_size[0] - 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Metrics lines
    pupil_mm = pupil_data.get('pupil_diameter_mm', 0)
    cog_load = pupil_data.get('cognitive_load', 0)
    
    lines = [
        (f"STATE: {state}", sc),
        (f"CC: {metrics['CC']:.3f} | DKL: {metrics['DKL']:.2f} | IG: {metrics['IG']:.3f} | NSS: {metrics['NSS']:.3f}", (160, 200, 255)),
        (f"GAZE → {attended_class.upper()} ({'ON-TASK' if on_task else 'OFF-TASK'})", (0, 255, 0) if on_task else (0, 0, 255)),
        (f"PUPIL: {pupil_mm:.2f}mm | COG LOAD: {cog_load:.2f} | PERCLOS: {metrics.get('perclos', 0):.3f}", (255, 200, 100)),
        (f"GPS: {metrics.get('gps_speed',-1):.1f} km/h | FLOW: {metrics.get('flow_speed',0):.1f} km/h | FRAME: {frame_idx:04d} | TIME: {timestamp_s:.1f}s",
         (100, 255, 180) if metrics.get('gps_speed', -1) >= 0 else (180, 180, 100))
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(metrics_panel, text, (50, 95 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # ── COMBINE ────────────────────────────────────────────────────────
    top_row = np.hstack([scene_panel, seg_panel])
    bottom_row = np.hstack([sal_panel, gaze_panel])
    output = np.vstack([top_row, bottom_row, metrics_panel])
    
    # Add borders
    cv2.line(output, (panel_size, 0), (panel_size, panel_size * 2), (255, 255, 255), 2)
    cv2.line(output, (0, panel_size), (panel_size * 2, panel_size), (255, 255, 255), 2)
    
    return output


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class AriaCleanPipeline:
    def __init__(self, config: AriaCleanConfig):
        self.config = config
        self.gaze_loader = AriaGazeLoader(config)
        self.segformer = SegFormerEngine(config)
        self.tracker = CentroidTracker()
        self.saliency = DreyeveSaliencyPredictor(config)
        self.pupil_analyzer = PupilAnalyzer(config)
        self.speed_estimator = SpeedEstimator(config.default_speed_kmh)
        self.gps_loader = GPSSpeedLoader(
            speed_csv_path=config.speed_csv_path,
            metadata_json_path=config.metadata_json_path,
        )
        self.gaze_df = None
        self.baseline = None
    
    def run(self):
        logger.info("=" * 65)
        logger.info("CLEAN ARIA + DR(EYE)VE ATTENTIVENESS PIPELINE")
        logger.info("=" * 65)
        t0 = time.time()
        
        self.gaze_df = self.gaze_loader.load_and_project()
        
        cap_rgb = cv2.VideoCapture(str(self.config.rgb_video_path))

        if not cap_rgb.isOpened():
            logger.error(f"Cannot open RGB video: {self.config.rgb_video_path}")
            return [], {}

        # ── FPS: always use JSON ground truth ─────────────────────────────────
        # OpenCV reads wrong fps from MP4 headers (20fps RGB, 10fps ET).
        # JSON truth: rgb=14.984fps, et=29.968fps, ratio=2.0 exactly.
        actual_rgb_fps = self.config.rgb_fps     # 14.984 from JSON
        actual_eye_fps = self.config.eye_fps     # 29.968 from JSON
        ET_RGB_RATIO   = self.config.ET_RGB_RATIO  # 2.0

        # ── ET: load from image folder (36471 frames at 30fps) ────────────────
        et_folder = Path(self.config.eye_video_path)
        et_frames_list = []
        eye_available  = False
        if et_folder.is_dir():
            exts = ['*.jpg','*.jpeg','*.png']
            for ext in exts:
                et_frames_list += sorted(glob.glob(str(et_folder / ext)))
            if et_frames_list:
                eye_available = True
                logger.info(f"ET folder: {len(et_frames_list)} frames  ratio={ET_RGB_RATIO:.1f}")
            else:
                logger.warning(f"No images found in ET folder: {et_folder}")
        else:
            # Fall back to video file if folder path doesn't exist
            cap_eye = cv2.VideoCapture(str(self.config.eye_video_path))
            eye_available = cap_eye.isOpened()
            et_frames_list = None
            if eye_available:
                logger.info(f"ET video fallback: {cap_eye.get(cv2.CAP_PROP_FRAME_COUNT):.0f} frames")

        rgb_total = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"RGB: fps={actual_rgb_fps:.4f}  frames={rgb_total}  dur={rgb_total/actual_rgb_fps:.1f}s")
        logger.info(f"Gaze CSV: dur={self.gaze_df['timestamp_s'].max():.1f}s")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        
        results = []
        prev_frame = None
        frame_idx = 0
        off_task_run = 0
        fisheye_mask_cache = None  # computed once on first frame
        actual_frame_size_logged = False
        
        total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret_rgb, frame = cap_rgb.read()
            if not ret_rgb:
                break
            
            if frame_idx % self.config.process_every_n_frames != 0:
                frame_idx += 1
                pbar.update(1)
                prev_frame = frame
                # cap_eye.grab() only valid for video file, not folder-based ET
                if eye_available and et_frames_list is None:
                    cap_eye.grab()
                continue
            
            timestamp_s = frame_idx / actual_rgb_fps   # JSON fps=14.984
            
            # ── ET frame: folder-based, ratio-synced ─────────────────────────
            # et_frame_idx = rgb_frame_idx * 2.0 (JSON ratio)
            eye_frame = None
            if eye_available:
                if et_frames_list is not None:
                    # Load directly from sorted image folder
                    et_frame_idx = min(int(frame_idx * ET_RGB_RATIO), len(et_frames_list) - 1)
                    eye_frame = cv2.imread(et_frames_list[et_frame_idx])
                else:
                    # Video file fallback
                    et_frame_idx = min(int(frame_idx * ET_RGB_RATIO), int(cap_eye.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                    cap_eye.set(cv2.CAP_PROP_POS_FRAMES, et_frame_idx)
                    ret_eye, eye_frame = cap_eye.read()
                    if not ret_eye:
                        eye_frame = None
            
            # Process pupil
            if eye_frame is not None:
                pupil_raw = self.pupil_analyzer.process_eye_frame(eye_frame)
                pupil_data = self.pupil_analyzer.compute_cognitive_load(
                    pupil_raw['pupil_diameter_mm'], timestamp_s
                )
                pupil_data.update(pupil_raw)
                perclos = self.pupil_analyzer.compute_perclos(30.0)
            else:
                pupil_data = {'pupil_diameter_mm': 0.0, 'cognitive_load': 0.0, 'baseline_mm': 0.0}
                perclos = 0.0
            
            # Get gaze
            gaze = self.gaze_loader.get_gaze_for_frame(self.gaze_df, timestamp_s)
            if gaze is None or gaze['confidence'] < self.config.gaze_conf_threshold:
                frame_idx += 1
                pbar.update(1)
                prev_frame = frame
                continue
            
            gaze_x, gaze_y = gaze['gaze_x_px'], gaze['gaze_y_px']
            
            # Semantic segmentation
            # FIX: build fisheye mask once and reuse
            if fisheye_mask_cache is None or fisheye_mask_cache.shape != frame.shape[:2]:
                fisheye_mask_cache = make_fisheye_mask(frame.shape[0], frame.shape[1])
                if not actual_frame_size_logged:
                    fH, fW = frame.shape[:2]
                    sx = fW / self.config.rgb_width
                    sy = fH / self.config.rgb_height
                    logger.info(f"Actual frame size: {fW}x{fH}  scale: {sx:.4f}x{sy:.4f}  (config: {self.config.rgb_width}x{self.config.rgb_height})")
                    actual_frame_size_logged = True

            fmask = fisheye_mask_cache if self.config.fisheye_mask else None
            semantic_mask = self.segformer.segment(frame, fmask)
            H, W = semantic_mask.shape

            # Rescale gaze to actual frame resolution
            # Video may deliver frames smaller than config 2880x2880
            # Scale factor = actual_size / config_size
            scale_x = W / self.config.rgb_width
            scale_y = H / self.config.rgb_height
            gaze_x = float(np.clip(gaze['gaze_x_raw'] * scale_x, 0, W - 1))                      if 'gaze_x_raw' in gaze else gaze_x
            gaze_y = float(np.clip(gaze['gaze_y_raw'] * scale_y, 0, H - 1))                      if 'gaze_y_raw' in gaze else gaze_y
            
            # Tracking (for saliency only)
            tracked = self.tracker.update(semantic_mask)
            
            # Speed: GPS preferred, optical-flow as fallback
            flow_speed      = self.speed_estimator.estimate_from_flow(prev_frame, frame)
            gps_speed       = self.gps_loader.get_speed_at(timestamp_s)
            estimated_speed = gps_speed if gps_speed is not None else flow_speed
            
            # Saliency prediction
            pred_saliency = self.saliency.predict(
                semantic_mask, frame,
                cognitive_load=pupil_data.get('cognitive_load', 0.0),
                speed_kmh=estimated_speed
            )
            
            # Gaze map
            actual_gaze_map = build_gaze_map(gaze_x, gaze_y, H, W, self.config.gaze_sigma_px)
            
            # Baseline for IG
            if self.baseline is None or self.baseline.shape != (H, W):
                self.baseline = center_gaussian_baseline(H, W)
            
            # FIX: downsample to 256x256 before metrics — full res (2880x2880) is 8M elements,
            # pearsonr on that size takes minutes per frame
            METRIC_H, METRIC_W = 256, 256
            sal_small  = cv2.resize(pred_saliency,    (METRIC_W, METRIC_H))
            gaze_small = cv2.resize(actual_gaze_map,  (METRIC_W, METRIC_H))
            base_small = cv2.resize(self.baseline,    (METRIC_W, METRIC_H))

            # Compute metrics
            metrics = {
                'CC':  compute_cc(gaze_small, sal_small),
                'DKL': compute_dkl(gaze_small, sal_small),
                'IG':  compute_ig(sal_small, gaze_small, base_small),
                'NSS': compute_nss(sal_small, gaze_small),
                'perclos': perclos,
                'speed': estimated_speed,
                'gps_speed': gps_speed if gps_speed is not None else -1.0,
                'flow_speed': flow_speed
            }
            
            # Attended class
            ix, iy = int(np.clip(gaze_x, 0, W-1)), int(np.clip(gaze_y, 0, H-1))
            attended_class = CITYSCAPES_CLASSES.get(int(semantic_mask[iy, ix]), 'unknown')
            on_task = TASK_RELEVANCE.get(attended_class, 'off_task') == 'on_task'
            
            # Update off-task counter
            off_task_run = 0 if on_task else off_task_run + 1
            distraction_event = off_task_run >= self.config.off_task_duration_threshold
            
            # State classification
            gaze_ratio = gaze_y / H
            state = classify_state(
                metrics['CC'], on_task, perclos, gaze_ratio,
                pupil_data.get('cognitive_load', 0), pupil_data.get('pupil_diameter_mm', 0)
            )
            attn_score = compute_score(
                metrics['CC'], on_task, perclos, gaze_ratio,
                pupil_data.get('cognitive_load', 0)
            )
            
            # Store results
            results.append({
                'frame_idx': frame_idx, 'timestamp_s': timestamp_s,
                'gaze_x_px': gaze_x, 'gaze_y_px': gaze_y,
                'attended_class': attended_class, 'on_task': on_task,
                'distraction_event': distraction_event,
                'state': state, 'attention_score': attn_score,
                'CC': metrics['CC'], 'DKL': metrics['DKL'],
                'IG': metrics['IG'], 'NSS': metrics['NSS'],
                'pupil_diameter_mm': pupil_data.get('pupil_diameter_mm', 0),
                'cognitive_load': pupil_data.get('cognitive_load', 0),
                'perclos': perclos, 'estimated_speed': estimated_speed
            })
            
            # Visualization
            vis = create_clean_visualization(
                frame, semantic_mask, gaze_x, gaze_y, pred_saliency,
                actual_gaze_map, attended_class, on_task, state,
                attn_score, metrics, frame_idx, timestamp_s, pupil_data
            )
            
            if out is None:
                h, w = vis.shape[:2]
                out = cv2.VideoWriter(str(self.config.output_dir / "clean_attentiveness.mp4"),
                                     fourcc, actual_rgb_fps, (w, h))
            out.write(vis)
            
            prev_frame = frame.copy()
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap_rgb.release()
        if eye_available and et_frames_list is None:
            cap_eye.release()
        if out:
            out.release()
        
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
        print("CLEAN ATTENTIVENESS PIPELINE - SUMMARY")
        print("=" * 65)
        print(f"\n📊 FRAMES: {len(results)}")
        print(f"\n🎯 SCORE: {np.mean([r['attention_score'] for r in results]):.1f}/100")
        print(f"\n📈 METRICS:")
        print(f"   CC:  {np.mean([r['CC'] for r in results]):.3f}")
        print(f"   DKL: {np.mean([r['DKL'] for r in results]):.3f}")
        print(f"   IG:  {np.mean([r['IG'] for r in results]):.3f}")
        print(f"   NSS: {np.mean([r['NSS'] for r in results]):.3f}")
        print(f"\n👁️ ON-TASK: {100 * np.mean([r['on_task'] for r in results]):.1f}%")
        print(f"👤 PUPIL: {np.mean([r['pupil_diameter_mm'] for r in results if r['pupil_diameter_mm'] > 0]):.2f}mm")
        print(f"\n🧠 STATES:")
        for state, count in Counter([r['state'] for r in results]).items():
            print(f"   {state}: {count}")
        print(f"\n⏱️ Time: {elapsed:.1f}s")
        print("=" * 65)
        print(f"✅ Output: {self.config.output_dir}/clean_attentiveness.mp4")
        print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean Aria + DR(eye)VE Attentiveness Pipeline")
    parser.add_argument("--base", default=r"E:\aria_work", help="Base path")
    parser.add_argument("--csv", default=None, help="Gaze CSV path")
    parser.add_argument("--video", default=None, help="RGB video path")
    parser.add_argument("--eye-video", default=None, help="Eye video path")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--every-n", type=int, default=1, help="Process every N frames")
    
    parser.add_argument("--subject-dir", default=r"E:\aria_work\sree_dt", help="Subject folder")
    parser.add_argument("--speed-csv",   default=None, help="Phone GPS Location CSV")
    parser.add_argument("--metadata",    default=None, help="Aria metadata JSON")

    args = parser.parse_args()

    base = Path(args.base)
    subj = Path(args.subject_dir)

    # Auto-discover gaze CSV
    if args.csv:
        gaze_csv = Path(args.csv)
    else:
        candidates = [
            subj / "sreevatsav-tada-p21.csv" / "general_eye_gaze.csv",
            subj / "general_eye_gaze.csv",
        ] + list(subj.glob("*/general_eye_gaze.csv"))
        gaze_csv = next((p for p in candidates if p.exists()), candidates[0])
        logger.info(f"Gaze CSV: {gaze_csv}  exists={gaze_csv.exists()}")

    # Auto-discover GPS CSV
    if args.speed_csv:
        speed_csv = Path(args.speed_csv)
    else:
        gps_candidates = [subj/"Location.csv", subj/"location.csv"]
        speed_csv = next((p for p in gps_candidates if p.exists()), None)
        if speed_csv:
            logger.info(f"GPS CSV: {speed_csv}")
        else:
            logger.warning("GPS CSV not found — use --speed-csv to specify")

    config = AriaCleanConfig(
        base_path          = base,
        subject_dir        = subj,
        csv_path           = gaze_csv,
        rgb_video_path     = Path(args.video)     if args.video     else subj / "srivatsav-p20-tada.mp4",
        eye_video_path     = Path(args.eye_video) if args.eye_video else subj / "with_data",
        output_dir         = Path(args.output)    if args.output    else subj / "clean_output",
        speed_csv_path     = speed_csv,
        metadata_json_path = Path(args.metadata)  if args.metadata  else subj / "sreevatsav-tada-p21.json",
        process_every_n_frames = args.every_n,
    )
    
    pipeline = AriaCleanPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()