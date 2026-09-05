"""Gaze CSV parsing, fixation detection, homography calibration."""
import io
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_gaze_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    required = {"timestamp", "gaze_x", "gaze_y", "pupil_size"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    df = df.dropna(subset=list(required))
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def default_homography() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def homography_from_points(
    src: List[List[float]], dst: List[List[float]]
) -> np.ndarray:
    """4-point homography: helmet gaze space -> video frame space."""
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("Homography requires exactly 4 source and 4 destination points")
    src_arr = np.array(src, dtype=np.float64)
    dst_arr = np.array(dst, dtype=np.float64)
    try:
        import cv2
        H, _ = cv2.findHomography(src_arr, dst_arr, method=0)
        if H is not None:
            return H
    except ImportError:
        pass
    # Fallback: affine from first 3 points
    A = np.zeros((6, 6))
    b = np.zeros(6)
    for i in range(3):
        A[2 * i] = [src_arr[i, 0], src_arr[i, 1], 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, src_arr[i, 0], src_arr[i, 1], 1]
        b[2 * i] = dst_arr[i, 0]
        b[2 * i + 1] = dst_arr[i, 1]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H = np.eye(3)
    H[0, 0], H[0, 1], H[0, 2] = params[0], params[1], params[2]
    H[1, 0], H[1, 1], H[1, 2] = params[3], params[4], params[5]
    return H


def apply_homography(df: pd.DataFrame, H: np.ndarray, video_w: int, video_h: int) -> pd.DataFrame:
    out = df.copy()
    pts = np.column_stack([df["gaze_x"].values, df["gaze_y"].values, np.ones(len(df))])
    transformed = (H @ pts.T).T
    w = transformed[:, 2]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    out["gaze_x"] = np.clip(transformed[:, 0] / w, 0, video_w)
    out["gaze_y"] = np.clip(transformed[:, 1] / w, 0, video_h)
    return out


def compute_fixations(df: pd.DataFrame, velocity_threshold: float = 30.0) -> List[dict]:
    fixations = []
    if len(df) < 2:
        return fixations

    velocities = []
    for i in range(1, len(df)):
        dt = df["timestamp"].iloc[i] - df["timestamp"].iloc[i - 1]
        if dt <= 0:
            velocities.append(0)
            continue
        dx = df["gaze_x"].iloc[i] - df["gaze_x"].iloc[i - 1]
        dy = df["gaze_y"].iloc[i] - df["gaze_y"].iloc[i - 1]
        velocities.append(math.sqrt(dx**2 + dy**2) / dt)

    in_fix = False
    fix_start = None
    fix_points = []

    for i in range(len(df)):
        vel = velocities[i - 1] if i > 0 else 0
        is_fix = vel < velocity_threshold
        row = df.iloc[i]
        if is_fix:
            if not in_fix:
                in_fix = True
                fix_start = row["timestamp"]
                fix_points = []
            fix_points.append((row["gaze_x"], row["gaze_y"]))
        elif in_fix:
            duration = row["timestamp"] - fix_start
            if duration > 0.1 and fix_points:
                fixations.append({
                    "start": float(fix_start),
                    "duration": float(duration),
                    "x": float(np.mean([p[0] for p in fix_points])),
                    "y": float(np.mean([p[1] for p in fix_points])),
                })
            in_fix = False

    if in_fix and fix_points:
        duration = df["timestamp"].iloc[-1] - fix_start
        if duration > 0.1:
            fixations.append({
                "start": float(fix_start),
                "duration": float(duration),
                "x": float(np.mean([p[0] for p in fix_points])),
                "y": float(np.mean([p[1] for p in fix_points])),
            })

    return fixations


def classify_gaze_zone(nx: float, ny: float) -> str:
    """Classify normalized gaze into driving zones."""
    if nx <= 0.15 and ny <= 0.25:
        return "mirror_left"
    if nx >= 0.85 and ny <= 0.25:
        return "mirror_right"
    if 0.2 <= nx <= 0.8 and ny >= 0.75:
        return "dashboard"
    if 0.1 <= nx <= 0.9 and 0.3 <= ny <= 0.8:
        return "road"
    if ny <= 0.2 and 0.35 <= nx <= 0.65:
        return "traffic_lights"
    if ny <= 0.35:
        return "traffic_signs"
    if ny <= 0.55:
        return "vehicles"
    return "distractors"
