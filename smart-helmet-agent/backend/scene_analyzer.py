"""Video frame scene understanding for driving attention analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SceneObject:
    category: str
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2 normalized 0-1
    confidence: float
    timestamp: float


@dataclass
class FrameAnalysis:
    timestamp: float
    objects: List[SceneObject] = field(default_factory=list)
    road_mask_pct: float = 0.0


def _norm_bbox(x: int, y: int, w: int, h: int, fw: int, fh: int) -> Tuple[float, float, float, float]:
    return (x / fw, y / fh, (x + w) / fw, (y + h) / fh)


def analyze_video(
    video_path: str,
    duration_s: float,
    sample_count: int = 60,
) -> Tuple[List[FrameAnalysis], Dict[str, float], int, int]:
    """
    Sample video frames and detect road, vehicles, pedestrians, signs, traffic lights.
    Returns frame analyses, video metadata, width, height.
    """
    try:
        import cv2
    except ImportError:
        return [], {"fps": 30, "duration": duration_s, "frames": 0}, 1920, 1080

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], {"fps": 30, "duration": duration_s, "frames": 0}, 1920, 1080

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    vid_duration = total_frames / fps if total_frames > 0 else duration_s

    sample_times = np.linspace(0, min(vid_duration, duration_s), max(2, sample_count))
    analyses: List[FrameAnalysis] = []
    prev_gray = None

    for t in sample_times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        objects = _detect_scene_objects(frame, t, prev_gray, fw, fh)
        road_pct = _estimate_road_coverage(frame)
        analyses.append(FrameAnalysis(timestamp=float(t), objects=objects, road_mask_pct=road_pct))
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()
    meta = {"fps": fps, "duration": vid_duration, "frames": total_frames, "width": fw, "height": fh}
    return analyses, meta, fw, fh


def _estimate_road_coverage(frame) -> float:
    import cv2
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.45):, int(w * 0.1): int(w * 0.9)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Asphalt/road: low saturation, moderate value
    mask = cv2.inRange(hsv, (0, 0, 40), (180, 80, 200))
    return round(float(mask.mean()) / 255 * 100, 1)


def _detect_scene_objects(frame, timestamp: float, prev_gray, fw: int, fh: int) -> List[SceneObject]:
    import cv2

    objects: List[SceneObject] = []
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Motion-based vehicles / pedestrians
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < 400 or area > w * h * 0.4:
                continue
            aspect = bw / max(bh, 1)
            cy_norm = (y + bh / 2) / h
            if cy_norm > 0.75:
                continue
            if aspect > 1.2 and area > 2000:
                cat = "vehicles"
                conf = min(0.95, 0.5 + area / (w * h) * 5)
            elif area < 8000:
                cat = "pedestrians"
                conf = min(0.85, 0.4 + area / (w * h) * 8)
            else:
                continue
            objects.append(SceneObject(cat, _norm_bbox(x, y, bw, bh, w, h), conf, timestamp))

    # Traffic signs: rectangular contours in upper region, red/blue/yellow
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    upper = frame[0: int(h * 0.45), :]
    upper_hsv = hsv[0: int(h * 0.45), :]
    for lo, hi in [((0, 100, 100), (10, 255, 255)), ((100, 80, 80), (130, 255, 255)), ((20, 100, 100), (35, 255, 255))]:
        mask = cv2.inRange(upper_hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours[:5]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < 200 or bw * bh > w * h * 0.05:
                continue
            objects.append(SceneObject(
                "traffic_signs", _norm_bbox(x, y, bw, bh, w, int(h * 0.45)), 0.65, timestamp
            ))

    # Traffic lights: bright red/green/yellow blobs in top 20%
    top = hsv[0: int(h * 0.2), int(w * 0.25): int(w * 0.75)]
    for lo, hi, color in [
        ((0, 120, 120), (10, 255, 255), "red"),
        ((35, 80, 80), (85, 255, 255), "green"),
        ((15, 100, 100), (35, 255, 255), "yellow"),
    ]:
        mask = cv2.inRange(top, lo, hi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours[:3]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < 50:
                continue
            gx = x + int(w * 0.25)
            objects.append(SceneObject(
                "traffic_lights",
                (gx / w, y / h, (gx + bw) / w, (y + bh) / h),
                0.7, timestamp,
            ))

    # Road region as implicit object (center lower)
    objects.append(SceneObject(
        "road", (0.1, 0.35, 0.9, 0.95), 0.9, timestamp
    ))

    return objects


def nearest_frame(analyses: List[FrameAnalysis], t: float) -> Optional[FrameAnalysis]:
    if not analyses:
        return None
    return min(analyses, key=lambda a: abs(a.timestamp - t))


def point_in_bbox(nx: float, ny: float, bbox: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= nx <= x2 and y1 <= ny <= y2


def map_gaze_to_scene(
    nx: float, ny: float, frame: Optional[FrameAnalysis]
) -> str:
    """Map gaze point to scene category using detections or zones."""
    if frame and frame.objects:
        # Prefer specific objects over road
        best = None
        best_conf = 0
        for obj in frame.objects:
            if obj.category == "road":
                continue
            if point_in_bbox(nx, ny, obj.bbox) and obj.confidence > best_conf:
                best = obj.category
                best_conf = obj.confidence
        if best:
            return best
        for obj in frame.objects:
            if obj.category == "road" and point_in_bbox(nx, ny, obj.bbox):
                return "road"

    from gaze import classify_gaze_zone
    return classify_gaze_zone(nx, ny)


def build_hazard_timeline(analyses: List[FrameAnalysis]) -> List[dict]:
    """Track when hazards (vehicles/pedestrians) first appear."""
    seen: Dict[str, float] = {}
    events = []
    for fa in sorted(analyses, key=lambda a: a.timestamp):
        for obj in fa.objects:
            if obj.category not in ("vehicles", "pedestrians"):
                continue
            key = f"{obj.category}_{int(obj.bbox[0]*100)}_{int(obj.bbox[1]*100)}"
            if key not in seen:
                seen[key] = fa.timestamp
                events.append({
                    "type": obj.category,
                    "timestamp": fa.timestamp,
                    "bbox": obj.bbox,
                    "key": key,
                })
    return events


def compute_reaction_latencies(
    df,
    hazard_events: List[dict],
    video_w: int,
    video_h: int,
) -> Tuple[float, List[dict]]:
    """Time from hazard appearance to first gaze on hazard bbox."""
    if not hazard_events or len(df) == 0:
        return 0.0, []

    latencies = []
    details = []
    nx = df["gaze_x"] / video_w
    ny = df["gaze_y"] / video_h

    for ev in hazard_events:
        t0 = ev["timestamp"]
        bbox = ev["bbox"]
        reaction = None
        for i in range(len(df)):
            ts = df["timestamp"].iloc[i]
            if ts < t0:
                continue
            if ts > t0 + 3.0:
                break
            if point_in_bbox(nx.iloc[i], ny.iloc[i], bbox):
                reaction = float(ts - t0)
                break
        if reaction is not None:
            latencies.append(reaction)
            details.append({**ev, "reaction_time_s": round(reaction, 3)})

    avg = round(float(np.mean(latencies)), 3) if latencies else 0.0
    return avg, details
