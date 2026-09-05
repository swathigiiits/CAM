"""
Smart Helmet Attention Analytics Agent - FastAPI Backend
Full pipeline: video sync, homography, scene CV, attention metrics, coaching
"""
import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import boto3
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analytics import compute_attention_metrics
from coaching import build_coaching_insights, generate_coaching_report, llm_context
from gaze import homography_from_points, parse_gaze_csv
import storage

app = FastAPI(
    title="Smart Helmet Attention Analytics Agent",
    description="AI-powered driver attention analysis using gaze tracking & video",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage.ensure_dirs()
app.mount("/uploads", StaticFiles(directory=str(storage.UPLOADS_DIR)), name="uploads")

S3_BUCKET = os.environ.get("S3_BUCKET", "smart-helmet-sessions")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", f"us.anthropic.claude-opus-4-8")


def _sessions_db() -> dict:
    return storage.load_all()


def _save_session(session_id: str, data: dict) -> None:
    storage.save_session(session_id, data)


def _get_homography(session: dict) -> np.ndarray:
    cal = session.get("calibration")
    if cal and cal.get("matrix"):
        return np.array(cal["matrix"], dtype=np.float64)
    return np.eye(3)


def _compute_progress_extras(sessions: list) -> dict:
    """Weekly trends, consistency, risk reduction."""
    if len(sessions) < 2:
        return {
            "behavioral_consistency": 100.0,
            "risk_reduction_trend": 0.0,
            "weekly_trends": [],
            "attention_maturity_trend": [],
        }

    scores = [s["metrics"]["attention_score"] for s in sessions]
    consistency = round(max(0, 100 - float(np.std(scores)) * 2), 1)

    risk_map = {"High": 3, "Moderate": 2, "Low": 1}
    risks = [risk_map.get(s["metrics"]["risk_level"], 2) for s in sessions]
    risk_reduction = round((risks[0] - risks[-1]) / max(risks[0], 1) * 100, 1)

    weekly = {}
    for s in sessions:
        week = s["timestamp"][:10]
        weekly.setdefault(week, []).append(s["metrics"]["attention_score"])
    weekly_trends = [
        {"week": w, "avg_score": round(float(np.mean(v)), 1), "sessions": len(v)}
        for w, v in sorted(weekly.items())
    ]

    maturity_trend = [
        {"session_num": i + 1, "maturity": s["metrics"].get("attention_maturity", s["metrics"]["attention_score"])}
        for i, s in enumerate(sessions)
    ]

    return {
        "behavioral_consistency": consistency,
        "risk_reduction_trend": risk_reduction,
        "weekly_trends": weekly_trends,
        "attention_maturity_trend": maturity_trend,
    }


def _strip_metrics(metrics: dict, include_timeline: bool = False) -> dict:
    exclude = {"heatmap_grid", "gaze_samples", "fixations", "temporal_heatmap", "hazard_events", "reaction_details"}
    if not include_timeline:
        exclude.add("attention_timeline")
    return {k: v for k, v in metrics.items() if k not in exclude}


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Smart Helmet Analytics Agent Online", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "sessions": len(_sessions_db())}


@app.post("/sessions/upload")
async def upload_session(
    background_tasks: BackgroundTasks,
    gaze_csv: UploadFile = File(...),
    video: Optional[UploadFile] = File(None),
    driver_id: Optional[str] = Form(None),
    session_type: Optional[str] = Form("city"),
    weather: Optional[str] = Form("clear"),
    notes: Optional[str] = Form(""),
    speed_kmh: Optional[float] = Form(None),
    gps_lat: Optional[float] = Form(None),
    gps_lng: Optional[float] = Form(None),
):
    session_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.utcnow().isoformat()
    sdir = storage.session_dir(session_id)

    try:
        csv_content = await gaze_csv.read()
        df = parse_gaze_csv(csv_content)
        (sdir / "gaze.csv").write_bytes(csv_content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    video_path = None
    video_filename = None
    if video and video.filename:
        ext = video.filename.split(".")[-1].lower()
        video_filename = f"video.{ext}"
        video_path = str(sdir / video_filename)
        content = await video.read()
        Path(video_path).write_bytes(content)

    try:
        metrics = compute_attention_metrics(df, video_path=video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {e}")

    did = driver_id or "anonymous"
    all_sessions = _sessions_db()
    driver_sessions = [
        s for sid, s in all_sessions.items()
        if s.get("driver_id") == did and sid != session_id
    ]
    comparison = None
    if driver_sessions:
        last = sorted(driver_sessions, key=lambda x: x["timestamp"])[-1]
        comparison = last["metrics"]

    coaching = generate_coaching_report(
        metrics, {"session_type": session_type, "weather": weather}, comparison
    )
    insights = build_coaching_insights(metrics, comparison)

    session = {
        "session_id": session_id,
        "driver_id": did,
        "timestamp": timestamp,
        "session_type": session_type,
        "weather": weather,
        "notes": notes,
        "speed_kmh": speed_kmh,
        "gps": {"lat": gps_lat, "lng": gps_lng} if gps_lat is not None else None,
        "metrics": metrics,
        "coaching_report": coaching,
        "coaching_insights": insights,
        "comparison": comparison,
        "video_filename": video_filename,
        "video_url": f"/uploads/{session_id}/{video_filename}" if video_filename else None,
        "calibration": None,
        "csv_rows": len(df),
    }

    _save_session(session_id, session)

    # Sync raw files to S3
    storage._s3_upload_file(sdir / "gaze.csv", f"sessions/{session_id}/gaze.csv")
    if video_filename:
        storage._s3_upload_file(sdir / video_filename, f"sessions/{session_id}/{video_filename}")

    response = {**session}
    response["metrics"] = _strip_metrics(metrics)
    response["heatmap_grid"] = metrics["heatmap_grid"]
    response["gaze_samples"] = metrics["gaze_samples"][:200]
    response["temporal_heatmap"] = metrics.get("temporal_heatmap", [])
    return response


@app.post("/sessions/{session_id}/calibration")
async def set_calibration(session_id: str, body: dict):
    """Set 4-point homography calibration and recompute metrics."""
    sessions = _sessions_db()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    src = body.get("src_points")
    dst = body.get("dst_points")
    if not src or not dst:
        raise HTTPException(status_code=400, detail="Provide src_points and dst_points (4 each)")

    try:
        H = homography_from_points(src, dst)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = sessions[session_id]
    sdir = storage.session_dir(session_id)
    csv_path = sdir / "gaze.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail="Gaze CSV not found for session")

    df = parse_gaze_csv(csv_path.read_bytes())
    video_path = None
    if session.get("video_filename"):
        vp = sdir / session["video_filename"]
        if vp.exists():
            video_path = str(vp)

    metrics = compute_attention_metrics(df, video_path=video_path, homography=H)
    session["calibration"] = {
        "src_points": src,
        "dst_points": dst,
        "matrix": H.tolist(),
    }
    session["metrics"] = metrics
    session["coaching_report"] = generate_coaching_report(
        metrics,
        {"session_type": session.get("session_type"), "weather": session.get("weather")},
        session.get("comparison"),
    )
    session["coaching_insights"] = build_coaching_insights(metrics, session.get("comparison"))
    _save_session(session_id, session)

    return {
        "session_id": session_id,
        "calibration": session["calibration"],
        "metrics": _strip_metrics(metrics),
    }


@app.get("/sessions")
def list_sessions(driver_id: Optional[str] = None):
    sessions = list(_sessions_db().values())
    if driver_id:
        sessions = [s for s in sessions if s["driver_id"] == driver_id]
    sessions = sorted(sessions, key=lambda x: x["timestamp"], reverse=True)
    return [{**s, "metrics": _strip_metrics(s["metrics"])} for s in sessions]


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    sessions = _sessions_db()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@app.get("/sessions/{session_id}/video")
def get_session_video(session_id: str):
    sessions = _sessions_db()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    if not session.get("video_filename"):
        raise HTTPException(status_code=404, detail="No video for this session")
    path = storage.session_dir(session_id) / session["video_filename"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")
    return FileResponse(path)


@app.get("/sessions/{session_id}/heatmap")
def get_heatmap(session_id: str):
    sessions = _sessions_db()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    m = sessions[session_id]["metrics"]
    return {
        "heatmap_grid": m["heatmap_grid"],
        "temporal_heatmap": m.get("temporal_heatmap", []),
    }


@app.get("/sessions/{session_id}/timeline")
def get_timeline(session_id: str):
    sessions = _sessions_db()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"timeline": sessions[session_id]["metrics"]["attention_timeline"]}


@app.get("/drivers/{driver_id}/progress")
def driver_progress(driver_id: str):
    sessions = [s for s in _sessions_db().values() if s["driver_id"] == driver_id]
    sessions = sorted(sessions, key=lambda x: x["timestamp"])

    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found for driver")

    progress = []
    for i, s in enumerate(sessions):
        m = s["metrics"]
        progress.append({
            "session_num": i + 1,
            "session_id": s["session_id"],
            "timestamp": s["timestamp"],
            "attention_score": m["attention_score"],
            "road_attention_pct": m["road_attention_pct"],
            "distraction_events": m["distraction_events"],
            "hazard_awareness": m["hazard_awareness"],
            "gaze_stability": m["gaze_stability"],
            "avg_hazard_reaction_time_s": m.get("avg_hazard_reaction_time_s", 0),
            "environmental_scanning": m.get("environmental_scanning", 0),
            "attention_maturity": m.get("attention_maturity", m["attention_score"]),
            "dangerous_glances": m.get("dangerous_glances", 0),
            "risk_level": m["risk_level"],
            "driver_level": m["driver_level"],
        })

    latest = sessions[-1]["metrics"]
    first = sessions[0]["metrics"]
    extras = _compute_progress_extras(sessions)

    return {
        "driver_id": driver_id,
        "total_sessions": len(sessions),
        "current_level": latest["driver_level"],
        "attention_maturity": latest.get("attention_maturity", latest["attention_score"]),
        "overall_improvement": round(latest["attention_score"] - first["attention_score"], 1),
        "sessions": progress,
        **extras,
    }


@app.post("/sessions/compare")
def compare_sessions(body: dict):
    ids = body.get("session_ids", [])
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 session IDs")

    sessions = _sessions_db()
    result = []
    for sid in ids:
        if sid not in sessions:
            continue
        s = sessions[sid]
        m = s["metrics"]
        result.append({
            "session_id": sid,
            "timestamp": s["timestamp"],
            "attention_score": m["attention_score"],
            "road_attention_pct": m["road_attention_pct"],
            "distraction_events": m["distraction_events"],
            "dangerous_glances": m.get("dangerous_glances", 0),
            "hazard_awareness": m["hazard_awareness"],
            "avg_hazard_reaction_time_s": m.get("avg_hazard_reaction_time_s", 0),
            "environmental_scanning": m.get("environmental_scanning", 0),
            "gaze_stability": m["gaze_stability"],
            "risk_level": m["risk_level"],
            "driver_level": m["driver_level"],
        })

    deltas = {}
    if len(result) >= 2:
        a, b = result[0], result[-1]
        deltas = {
            "attention_score_delta": round(b["attention_score"] - a["attention_score"], 1),
            "road_focus_delta": round(b["road_attention_pct"] - a["road_attention_pct"], 1),
            "distraction_delta": a["distraction_events"] - b["distraction_events"],
            "hazard_reaction_delta": round(
                (a["avg_hazard_reaction_time_s"] or 0) - (b["avg_hazard_reaction_time_s"] or 0), 3
            ),
        }

    return {"comparison": result, "deltas": deltas}


@app.post("/sessions/{session_id}/coaching/llm")
async def llm_coaching(session_id: str, body: dict = {}):
    sessions = _sessions_db()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    question = body.get("question", "Give me a detailed coaching report.")

    prior = [
        s for sid, s in sessions.items()
        if s["driver_id"] == session["driver_id"] and sid != session_id
    ]
    prior = sorted(prior, key=lambda x: x["timestamp"])

    system_prompt = """You are an expert driving coach and attention analyst.
Analyze gaze tracking data and provide actionable, personalized coaching.
Reference specific metrics and scene distribution. Be encouraging and safety-focused."""

    user_prompt = f"""{llm_context(session, prior)}

Driver Question: {question}

Provide a detailed, actionable coaching report with specific recommendations."""

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 1200},
        )
        coaching_text = response["output"]["message"]["content"][0]["text"]
        return {
            "coaching": coaching_text,
            "insights": session.get("coaching_insights", []),
            "llm": True,
        }
    except Exception as e:
        return {
            "coaching": session.get("coaching_report", "No coaching available."),
            "insights": session.get("coaching_insights", []),
            "llm": False,
            "error": str(e),
        }


@app.get("/demo/generate-csv")
def generate_demo_csv():
    n = 1000
    t = np.linspace(0, 30, n)
    gaze_x = 960 + np.random.normal(0, 80, n)
    gaze_y = 400 + np.random.normal(0, 60, n)
    for _ in range(5):
        idx = random.randint(100, 900)
        gaze_x[idx:idx + 30] = random.uniform(200, 1700)
        gaze_y[idx:idx + 30] = random.uniform(100, 900)
    pupil = 0.75 + np.random.normal(0, 0.05, n)

    df = pd.DataFrame({
        "timestamp": np.round(t, 3),
        "gaze_x": np.round(np.clip(gaze_x, 0, 1920), 1),
        "gaze_y": np.round(np.clip(gaze_y, 0, 1080), 1),
        "pupil_size": np.round(np.clip(pupil, 0.3, 1.2), 3),
    })

    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return JSONResponse(content={"csv": buf.getvalue(), "rows": n})
