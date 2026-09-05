"""Full attention analytics pipeline."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from gaze import apply_homography, classify_gaze_zone, compute_fixations, default_homography
from scene_analyzer import (
    analyze_video,
    build_hazard_timeline,
    compute_reaction_latencies,
    map_gaze_to_scene,
    nearest_frame,
    point_in_bbox,
)


def compute_environmental_scanning(df: pd.DataFrame, video_w: int, video_h: int) -> float:
    """Score systematic left-center-right scanning (0-100)."""
    if len(df) < 10:
        return 50.0

    nx = df["gaze_x"] / video_w
    zones = [classify_gaze_zone(nx.iloc[i], df["gaze_y"].iloc[i] / video_h) for i in range(len(df))]

    def zone_bucket(z):
        if z in ("mirror_left",):
            return "L"
        if z in ("mirror_right",):
            return "R"
        if z == "road":
            return "C"
        return "O"

    seq = [zone_bucket(z) for z in zones if zone_bucket(z) != "O"]
    if len(seq) < 5:
        return 40.0

    cycles = 0
    for i in range(len(seq) - 2):
        tri = seq[i: i + 3]
        if tri in (["L", "C", "R"], ["R", "C", "L"], ["C", "L", "C"], ["C", "R", "C"]):
            cycles += 1

    duration = float(df["timestamp"].max() - df["timestamp"].min()) or 1
    rate = cycles / (duration / 60)  # cycles per minute
    return round(min(100, 30 + rate * 15), 1)


def count_off_road_glances(df: pd.DataFrame, video_w: int, video_h: int) -> int:
    """Discrete off-road glance events (>0.4s off road)."""
    road_zone = (0.1, 0.3, 0.9, 0.8)
    nx = df["gaze_x"] / video_w
    ny = df["gaze_y"] / video_h
    glances = 0
    off_streak = 0.0
    prev_ts = None
    in_glance = False

    for i in range(len(df)):
        x, y = nx.iloc[i], ny.iloc[i]
        ts = df["timestamp"].iloc[i]
        on_road = road_zone[0] <= x <= road_zone[2] and road_zone[1] <= y <= road_zone[3]
        if not on_road:
            if prev_ts is not None:
                off_streak += ts - prev_ts
            if off_streak >= 0.4 and not in_glance:
                glances += 1
                in_glance = True
        else:
            off_streak = 0.0
            in_glance = False
        prev_ts = ts
    return glances


def compute_attention_metrics(
    df: pd.DataFrame,
    video_path: Optional[str] = None,
    homography: Optional[np.ndarray] = None,
    video_w: int = 1920,
    video_h: int = 1080,
) -> Dict[str, Any]:
    H = homography if homography is not None else default_homography()

    n_raw = len(df)
    if n_raw == 0:
        return {}

    duration = float(df["timestamp"].max() - df["timestamp"].min())

    # Resolve video dimensions before homography
    frame_analyses = []
    video_meta = {}
    scene_source = "roi_zones"
    video_path_valid = video_path and Path(video_path).exists() if video_path else False

    if video_path_valid:
        frame_analyses, video_meta, vw, vh = analyze_video(video_path, duration)
        if vw and vh:
            video_w, video_h = vw, vh
        if frame_analyses:
            scene_source = "video_cv"

    df = apply_homography(df, H, video_w, video_h)
    n = len(df)

    nx = df["gaze_x"] / video_w
    ny = df["gaze_y"] / video_h

    road_zone = (0.1, 0.3, 0.9, 0.8)
    mirror_left = (0.0, 0.0, 0.15, 0.25)
    mirror_right = (0.85, 0.0, 1.0, 0.25)
    dash_zone = (0.2, 0.75, 0.8, 1.0)

    def in_zone(x, y, zone):
        return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]

    # Scene-aware gaze classification
    scene_counts: Dict[str, int] = {
        "road": 0, "vehicles": 0, "pedestrians": 0, "traffic_signs": 0,
        "traffic_lights": 0, "mirrors": 0, "dashboard": 0, "distractors": 0,
    }

    for i in range(n):
        t = df["timestamp"].iloc[i]
        x, y = nx.iloc[i], ny.iloc[i]
        frame = nearest_frame(frame_analyses, t) if frame_analyses else None
        cat = map_gaze_to_scene(x, y, frame)
        if cat in ("mirror_left", "mirror_right"):
            cat = "mirrors"
        if cat not in scene_counts:
            cat = "distractors"
        scene_counts[cat] += 1

    scene_dist = {k: round(v / n * 100, 1) for k, v in scene_counts.items()}

    road_count = scene_counts["road"]
    mirror_count = scene_counts["mirrors"]
    dash_count = scene_counts["dashboard"]
    off_road_count = n - road_count - mirror_count - dash_count

    road_pct = scene_dist["road"]
    mirror_pct = scene_dist["mirrors"]
    dash_pct = scene_dist["dashboard"]
    off_road_pct = round(max(0, off_road_count / n * 100), 1)

    fixations = compute_fixations(df)
    avg_fix_dur = float(np.mean([f["duration"] for f in fixations])) if fixations else 0

    # Long distraction events (>2s off road)
    distraction_events = 0
    off_road_streak = 0.0
    prev_ts = None
    for i in range(n):
        x, y = nx.iloc[i], ny.iloc[i]
        ts = df["timestamp"].iloc[i]
        if not in_zone(x, y, road_zone):
            if prev_ts is not None:
                off_road_streak += ts - prev_ts
            if off_road_streak > 2.0:
                distraction_events += 1
                off_road_streak = 0.0
        else:
            off_road_streak = 0.0
        prev_ts = ts

    off_road_glances = count_off_road_glances(df, video_w, video_h)
    env_scanning = compute_environmental_scanning(df, video_w, video_h)

    velocities = []
    for i in range(1, n):
        dt = df["timestamp"].iloc[i] - df["timestamp"].iloc[i - 1]
        if dt > 0:
            dx = df["gaze_x"].iloc[i] - df["gaze_x"].iloc[i - 1]
            dy = df["gaze_y"].iloc[i] - df["gaze_y"].iloc[i - 1]
            velocities.append(math.sqrt(dx**2 + dy**2) / dt)

    gaze_stability = max(0, 100 - min(100, float(np.std(velocities)) / 5)) if velocities else 50

    x_bins = np.histogram(df["gaze_x"], bins=10)[0]
    y_bins = np.histogram(df["gaze_y"], bins=10)[0]
    x_prob = x_bins / (x_bins.sum() + 1e-9)
    y_prob = y_bins / (y_bins.sum() + 1e-9)
    entropy = float(-np.sum(x_prob * np.log2(x_prob + 1e-9)) - np.sum(y_prob * np.log2(y_prob + 1e-9)))
    attention_spread = min(100, entropy * 5)

    hazard_events = [
        {**ev, "bbox": list(ev["bbox"])} for ev in (build_hazard_timeline(frame_analyses) if frame_analyses else [])[:50]
    ]
    avg_reaction, reaction_details = compute_reaction_latencies(
        df, hazard_events, video_w, video_h
    )

    # Hazard awareness: reaction time + gaze on hazards + mirror checks
    reaction_score = max(0, 100 - avg_reaction * 80) if avg_reaction > 0 else 50
    hazard_gaze_pct = scene_dist.get("vehicles", 0) + scene_dist.get("pedestrians", 0)
    hazard_awareness = min(100, round(
        reaction_score * 0.35 + hazard_gaze_pct * 0.25 + road_pct * 0.25 + mirror_pct * 2.5 * 0.15,
        1,
    ))

    attention_score = round(min(100, max(0, (
        road_pct * 0.28 +
        gaze_stability * 0.18 +
        hazard_awareness * 0.18 +
        env_scanning * 0.12 +
        min(100, (1 - off_road_pct / 100) * 100) * 0.12 +
        min(100, max(0, 100 - distraction_events * 10)) * 0.07 +
        min(100, attention_spread) * 0.05
    ))), 1)

    risk_level = "Low" if attention_score >= 75 else ("Moderate" if attention_score >= 55 else "High")
    driver_level = "Advanced" if attention_score >= 80 else ("Intermediate" if attention_score >= 60 else "Beginner")
    attention_maturity = round(
        attention_score * 0.4 + gaze_stability * 0.2 + env_scanning * 0.2 + hazard_awareness * 0.2, 1
    )

    avg_pupil = float(df["pupil_size"].mean())

    # Fixation density heatmap
    heatmap = np.zeros((50, 50))
    for i in range(n):
        xi = int(nx.iloc[i] * 49)
        yi = int(ny.iloc[i] * 49)
        xi, yi = max(0, min(49, xi)), max(0, min(49, yi))
        heatmap[yi, xi] += 1
    heatmap = gaussian_filter(heatmap, sigma=2)
    heatmap_norm = (heatmap / (heatmap.max() + 1e-9) * 255).astype(int).tolist()

    # Temporal attention heatmap: 30 time buckets × 10 vertical zones
    bucket_count = 30
    bucket_size = max(duration / bucket_count, 0.01)
    temporal_heatmap = []
    timeline = []
    for b in range(bucket_count):
        t_start = df["timestamp"].min() + b * bucket_size
        t_end = t_start + bucket_size
        mask = (df["timestamp"] >= t_start) & (df["timestamp"] < t_end)
        subset = df[mask]
        zone_row = [0] * 10
        if len(subset) > 0:
            sx = subset["gaze_x"] / video_w
            sy = subset["gaze_y"] / video_h
            for j in range(len(subset)):
                zi = min(9, int(sy.iloc[j] * 10))
                zone_row[zi] += 1
            r = sum(in_zone(sx.iloc[j], sy.iloc[j], road_zone) for j in range(len(subset)))
            rp = round(r / len(subset) * 100, 1)
        else:
            rp = 0
        mx = max(zone_row) if zone_row else 1
        temporal_heatmap.append([round(v / mx * 100, 1) for v in zone_row])
        timeline.append({
            "t": round(t_start, 2),
            "score": rp,
            "road_pct": rp,
            "attention_score": rp,
        })

    step = max(1, n // 800)
    gaze_samples = [
        {
            "t": float(df["timestamp"].iloc[i]),
            "x": float(nx.iloc[i]),
            "y": float(ny.iloc[i]),
            "p": float(df["pupil_size"].iloc[i]),
        }
        for i in range(0, n, step)
    ]

    dangerous_glances = off_road_glances + distraction_events

    return {
        "attention_score": attention_score,
        "risk_level": risk_level,
        "driver_level": driver_level,
        "attention_maturity": attention_maturity,
        "road_attention_pct": road_pct,
        "mirror_checks_pct": mirror_pct,
        "dashboard_pct": dash_pct,
        "off_road_pct": off_road_pct,
        "off_road_glances": off_road_glances,
        "dangerous_glances": dangerous_glances,
        "gaze_stability": round(gaze_stability, 1),
        "attention_spread": round(attention_spread, 1),
        "environmental_scanning": env_scanning,
        "hazard_awareness": hazard_awareness,
        "avg_hazard_reaction_time_s": avg_reaction,
        "hazard_events": hazard_events,
        "reaction_details": [{**d, "bbox": list(d.get("bbox", []))} for d in reaction_details[:20]],
        "fixation_count": len(fixations),
        "avg_fixation_duration_ms": round(avg_fix_dur * 1000, 1),
        "distraction_events": distraction_events,
        "avg_pupil_size": round(avg_pupil, 3),
        "session_duration_s": round(duration, 2),
        "total_gaze_points": n,
        "heatmap_grid": heatmap_norm,
        "temporal_heatmap": temporal_heatmap,
        "attention_timeline": timeline,
        "scene_distribution": scene_dist,
        "scene_source": scene_source,
        "fixations": fixations[:200],
        "gaze_samples": gaze_samples,
        "video_width": video_w,
        "video_height": video_h,
        "video_meta": video_meta,
    }
