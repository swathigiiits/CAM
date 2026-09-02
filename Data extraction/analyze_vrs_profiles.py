#!/usr/bin/env python3
"""
Analyze Aria .vrs.json sidecar files in a folder, group sessions by
recording_profile, and produce:
  1. A per-session CSV (one row per file, with fps/quality per stream)
  2. A per-profile summary table (mean/min/max score, active streams, totals)
  3. A flag list for filename/notes/name vs recording_profile mismatches

Usage:
    python analyze_vrs_profiles.py /path/to/folder_with_sidecars

Output (written next to the input folder, or --outdir if given):
    sessions.csv
    profile_summary.csv
    mismatch_flags.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

STREAM_KEYS = [
    "et_camera", "slam_camera_1", "slam_camera_2", "rgb_camera",
    "imu_1", "imu_2", "magnetometer", "barometer", "gps", "audio",
]


def load_sidecars(folder: Path):
    files = sorted(folder.glob("*.json"))
    records = []
    for f in files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Could not read {f.name}: {e}")
            continue
        data["_sidecar_path"] = str(f)
        data["_sidecar_name"] = f.name
        records.append(data)
    return records


def session_row(rec):
    stats = rec.get("data_quality_stats", {})
    start = rec.get("start_time")
    end = rec.get("end_time")
    duration = (end - start) if (start is not None and end is not None) else None

    row = {
        "sidecar_file": rec.get("_sidecar_name"),
        "filename": rec.get("filename"),
        "name": rec.get("name"),
        "notes": rec.get("notes"),
        "recording_profile": rec.get("recording_profile"),
        "device_id": rec.get("device_id"),
        "duration_s": duration,
        "file_size_bytes": rec.get("file_size"),
        "file_size_gb": round(rec.get("file_size", 0) / 1e9, 2) if rec.get("file_size") else None,
    }

    for stream in STREAM_KEYS:
        s = stats.get(stream)
        if s is None:
            row[f"{stream}_expected"] = None
            row[f"{stream}_score"] = None
            row[f"{stream}_dropped"] = None
            row[f"{stream}_hz"] = None
            continue
        expected = s.get("expected")
        row[f"{stream}_expected"] = expected
        row[f"{stream}_score"] = s.get("score")
        row[f"{stream}_dropped"] = s.get("dropped")
        if expected is not None and duration:
            row[f"{stream}_hz"] = round(expected / duration, 2)
        else:
            row[f"{stream}_hz"] = None

    return row


def find_mismatches(row):
    """Flag sessions where filename / name / notes disagree on profile number,
    or where recording_profile doesn't match a profile number embedded in the name."""
    flags = []
    prof = row.get("recording_profile") or ""
    prof_num = "".join(ch for ch in prof if ch.isdigit())

    for field in ("filename", "name", "notes"):
        val = str(row.get(field) or "")
        digits_after_p = None
        low = val.lower()
        if "profile" in low:
            idx = low.find("profile")
            tail = val[idx + len("profile"):].strip()
            digits_after_p = "".join(ch for ch in tail if ch.isdigit())
        if digits_after_p and prof_num and digits_after_p != prof_num:
            flags.append(
                f"{row['sidecar_file']}: recording_profile={prof} but {field}='{val}' suggests profile{digits_after_p}"
            )
    return flags


def build_profile_summary(rows):
    by_profile = {}
    for row in rows:
        prof = row.get("recording_profile") or "UNKNOWN"
        by_profile.setdefault(prof, []).append(row)

    summary_rows = []
    for prof, group in sorted(by_profile.items()):
        n_sessions = len(group)
        total_duration = sum(r["duration_s"] for r in group if r["duration_s"])
        total_size_gb = sum(r["file_size_gb"] for r in group if r["file_size_gb"])

        summary_row = {
            "recording_profile": prof,
            "n_sessions": n_sessions,
            "total_duration_min": round(total_duration / 60, 1) if total_duration else None,
            "total_size_gb": round(total_size_gb, 2) if total_size_gb else None,
        }

        for stream in STREAM_KEYS:
            scores = [r[f"{stream}_score"] for r in group if r.get(f"{stream}_score") is not None]
            hz_vals = [r[f"{stream}_hz"] for r in group if r.get(f"{stream}_hz") is not None]
            n_active = len(scores)

            summary_row[f"{stream}_active_sessions"] = n_active
            if scores:
                summary_row[f"{stream}_score_mean"] = round(statistics.mean(scores), 3)
                summary_row[f"{stream}_score_min"] = round(min(scores), 3)
                summary_row[f"{stream}_score_max"] = round(max(scores), 3)
            else:
                summary_row[f"{stream}_score_mean"] = None
                summary_row[f"{stream}_score_min"] = None
                summary_row[f"{stream}_score_max"] = None

            if hz_vals:
                summary_row[f"{stream}_hz_typical"] = round(statistics.median(hz_vals), 2)
            else:
                summary_row[f"{stream}_hz_typical"] = None

        summary_rows.append(summary_row)

    return summary_rows


def write_csv(path: Path, rows, fieldnames=None):
    if not rows:
        print(f"[INFO] No rows to write for {path.name}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote {path} ({len(rows)} rows)")


def print_profile_table(summary_rows):
    print("\n=== PROFILE SUMMARY (score = mean, typical Hz shown) ===\n")
    header = ["profile", "n", "streams (Hz | mean score)"]
    print(f"{'Profile':<10}{'Sessions':<10}")
    for row in summary_rows:
        print(f"\n--- {row['recording_profile']} ({row['n_sessions']} sessions, "
              f"{row['total_duration_min']} min total, {row['total_size_gb']} GB) ---")
        for stream in STREAM_KEYS:
            active = row.get(f"{stream}_active_sessions", 0)
            if not active:
                continue
            hz = row.get(f"{stream}_hz_typical")
            mean = row.get(f"{stream}_score_mean")
            mn = row.get(f"{stream}_score_min")
            mx = row.get(f"{stream}_score_max")
            print(f"  {stream:<15} active={active:<4} hz~{hz:<8} score mean={mean} (min={mn}, max={mx})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder containing .vrs.json sidecar files")
    parser.add_argument("--outdir", type=str, default=None, help="Output folder (default: same as input)")
    args = parser.parse_args()

    folder = Path(args.folder)
    outdir = Path(args.outdir) if args.outdir else folder
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_sidecars(folder)
    print(f"[INFO] Loaded {len(records)} sidecar files from {folder}")

    rows = [session_row(r) for r in records]

    all_flags = []
    for row in rows:
        all_flags.extend(find_mismatches(row))

    summary_rows = build_profile_summary(rows)

    write_csv(outdir / "sessions.csv", rows)
    write_csv(outdir / "profile_summary.csv", summary_rows)

    if all_flags:
        flag_rows = [{"flag": f} for f in all_flags]
        write_csv(outdir / "mismatch_flags.csv", flag_rows)
        print(f"\n[WARN] {len(all_flags)} filename/notes/profile mismatches found — see mismatch_flags.csv")
    else:
        print("\n[OK] No filename/notes/profile mismatches detected.")

    print_profile_table(summary_rows)


if __name__ == "__main__":
    main()