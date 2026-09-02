"""
Batch pupil detection over all ET frames in a session folder.
Finds the first frame index where BOTH eyes have a valid pupil detected,
i.e. the end of the device setup/don-time period.

Tightened detection (v2): higher circularity threshold, darkness check,
and requires N consecutive valid frames before accepting the transition,
to avoid false positives on forehead/skin blobs during setup.

Usage:
    python batch_pupil_detect.py --folder "E:\\aria_work\\sree_dt\\with_data" --fps 30 --stop-at-first-valid

Output:
    <folder>_pupil_report.csv   -- per-frame detection results
    Console summary: first valid frame index/time, % frames with valid pupils
"""
import cv2
import numpy as np
import argparse
import csv
import re
from pathlib import Path

# Eye ROIs — from show_eye.py (calibrated on one sample frame; verify per-session/per-rider)
LEFT_EYE  = (46, 67,  280, 233)   # x1,y1,x2,y2
RIGHT_EYE = (337, 102, 569, 237)

# Tightened thresholds to reject skin/eyebrow false positives during setup
MIN_AREA = 500
MAX_AREA = 5000
MIN_CIRCULARITY = 0.55       # was 0.10 — pupils are genuinely round; skin folds aren't
MAX_MEAN_INTENSITY = 60      # pupil interior should be near-black; skin blobs are lighter
CONSECUTIVE_FRAMES_REQUIRED = 5   # must hold for N frames in a row to count as real transition


def detect_pupil(eye_img):
    """Return (cx, cy, r, circularity, mean_intensity) in eye_img's local coords, or None."""
    blurred = cv2.GaussianBlur(eye_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (MIN_AREA < area < MAX_AREA):
            continue
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim ** 2 + 1e-8)
        if circ <= MIN_CIRCULARITY:
            continue
        # Darkness check: mean pixel intensity inside the contour mask
        mask = np.zeros(eye_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_val = cv2.mean(eye_img, mask=mask)[0]
        if mean_val > MAX_MEAN_INTENSITY:
            continue
        candidates.append((c, area, circ, mean_val))

    if not candidates:
        return None

    # Pick the most circular among survivors
    best_c, area, circ, mean_val = max(candidates, key=lambda x: x[2])
    M = cv2.moments(best_c)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    r = int(np.sqrt(area / np.pi))
    return (cx, cy, r, circ, mean_val)


def extract_frame_index(filename: str):
    """Pull the numeric frame index out of a filename like et_000694_<ts>.jpg or enhanced_000694.jpg."""
    m = re.search(r'(\d{5,7})', filename)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing extracted ET frames")
    parser.add_argument("--fps", type=float, default=None, help="ET stream fps, for time-based reporting")
    parser.add_argument("--pattern", default="*.jpg", help="Glob pattern for frame files (default *.jpg)")
    parser.add_argument("--stop-at-first-valid", action="store_true",
                         help="Stop scanning as soon as the first frame with both eyes valid is found")
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob(args.pattern))
    # Exclude enhanced_ duplicates (CLAHE-processed copies) and no_data_ frames —
    # only scan raw et_*.jpg frames so ordering and counts reflect true frame sequence
    files = [f for f in files if not f.name.startswith("no_data_") and not f.name.startswith("enhanced_")]

    if not files:
        print(f"[ERROR] No frames found in {folder} matching {args.pattern}")
        return

    print(f"[INFO] Found {len(files)} frames in {folder}")

    results = []
    first_valid_idx = None
    first_valid_frame_number = None
    consecutive_count = 0
    candidate_start_idx = None

    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        H, W = gray.shape[:2]

        row = {"filename": f.name, "frame_number": extract_frame_index(f.name)}
        both_valid = True

        for (x1, y1, x2, y2), label in [(LEFT_EYE, "left"), (RIGHT_EYE, "right")]:
            x2c, y2c = min(x2, W), min(y2, H)
            if x1 >= W or y1 >= H or x2c <= x1 or y2c <= y1:
                row[f"{label}_pupil"] = None
                both_valid = False
                continue
            eye_img = gray[y1:y2c, x1:x2c]
            result = detect_pupil(eye_img)
            if result:
                cx, cy, r, circ, mean_val = result
                row[f"{label}_pupil"] = 1
                row[f"{label}_diam_mm"] = round((2 * r) / 9.0, 2)
                row[f"{label}_circularity"] = round(circ, 3)
                row[f"{label}_mean_intensity"] = round(mean_val, 1)
            else:
                row[f"{label}_pupil"] = 0
                row[f"{label}_diam_mm"] = None
                row[f"{label}_circularity"] = None
                row[f"{label}_mean_intensity"] = None
                both_valid = False

        row["both_eyes_valid"] = int(both_valid)
        results.append(row)

        # Require CONSECUTIVE_FRAMES_REQUIRED in a row before accepting as the real transition
        if both_valid:
            if candidate_start_idx is None:
                candidate_start_idx = len(results) - 1
                candidate_start_frame_number = row["frame_number"]
            consecutive_count += 1
            if consecutive_count >= CONSECUTIVE_FRAMES_REQUIRED and first_valid_idx is None:
                first_valid_idx = candidate_start_idx
                first_valid_frame_number = candidate_start_frame_number
                if args.stop_at_first_valid:
                    print(f"[INFO] Confirmed transition: {CONSECUTIVE_FRAMES_REQUIRED} consecutive valid "
                          f"frames starting at index {first_valid_idx} (frame_number={first_valid_frame_number}) "
                          f"— stopping scan.")
                    break
        else:
            consecutive_count = 0
            candidate_start_idx = None

    # Write CSV report
    out_csv = folder.parent / f"{folder.name}_pupil_report.csv"
    fieldnames = ["filename", "frame_number", "left_pupil", "left_diam_mm", "left_circularity", "left_mean_intensity",
                  "right_pupil", "right_diam_mm", "right_circularity", "right_mean_intensity", "both_eyes_valid"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] Wrote {out_csv} ({len(results)} rows)")

    # Summary
    total = len(results)
    n_valid = sum(r["both_eyes_valid"] for r in results)

    print(f"\n=== SUMMARY: {folder.name} ===")
    if args.stop_at_first_valid:
        print(f"Mode: early-stop (scan halted at first valid-pupil frame)")
        print(f"Frames scanned before stopping: {total}")
    else:
        pct_valid = 100 * n_valid / total if total else 0
        print(f"Total frames scanned: {total}")
        print(f"Frames with valid pupils (both eyes): {n_valid} ({pct_valid:.1f}%)")

    if first_valid_idx is not None:
        print(f"First frame with valid pupils (both eyes): index {first_valid_idx} "
              f"(frame_number={first_valid_frame_number})")
        if args.fps:
            setup_seconds = first_valid_idx / args.fps
            print(f"Estimated setup/don-time duration: {setup_seconds:.1f} s "
                  f"(at {args.fps} fps, {first_valid_idx} frames)")
    else:
        print("No frame with valid pupils in both eyes was found.")


if __name__ == "__main__":
    main()