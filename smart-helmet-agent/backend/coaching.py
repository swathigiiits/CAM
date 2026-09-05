"""Coaching reports with explainable recommendations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_coaching_insights(metrics: dict, comparison: Optional[dict] = None) -> List[dict]:
    """Structured explainable coaching evidence."""
    insights = []

    def add(metric, value, evidence, recommendation, severity="info"):
        insights.append({
            "metric": metric,
            "value": value,
            "evidence": evidence,
            "recommendation": recommendation,
            "severity": severity,
        })

    road = metrics["road_attention_pct"]
    if road < 60:
        add("road_attention_pct", f"{road}%", "Gaze on road below 60% safety threshold",
            "Practice 20-second road scanning: sweep gaze left to right every 20 seconds.", "warning")
    elif road >= 80:
        add("road_attention_pct", f"{road}%", "Strong sustained road focus detected",
            "Maintain current road attention habits in complex traffic.", "success")

    dist = metrics["distraction_events"]
    if dist > 3:
        add("distraction_events", dist, f"{dist} prolonged off-road fixations (>2s)",
            "Reduce phone and dashboard interaction while moving.", "warning")
    elif dist == 0:
        add("distraction_events", 0, "No prolonged distraction events",
            "Excellent focus discipline — keep it up.", "success")

    reaction = metrics.get("avg_hazard_reaction_time_s", 0)
    if reaction > 0:
        if reaction > 0.8:
            add("avg_hazard_reaction_time_s", f"{reaction}s", "Slow hazard gaze response",
                "At intersections, actively scan for pedestrians and crossing traffic.", "warning")
        else:
            add("avg_hazard_reaction_time_s", f"{reaction}s", "Fast hazard gaze response",
                "Good hazard detection — maintain mirror checks near blind spots.", "success")

    scan = metrics.get("environmental_scanning", 50)
    if scan < 45:
        add("environmental_scanning", f"{scan}%", "Limited left-center-right scanning pattern",
            "Increase side scanning near intersections to reduce tunnel vision.", "warning")

    signs = metrics.get("scene_distribution", {}).get("traffic_signs", 0)
    if signs < 3 and road > 70:
        add("traffic_signs", f"{signs}%", "Low traffic sign monitoring despite good road focus",
            "Briefly check signs and signals at intersections and lane changes.", "warning")

    if comparison:
        delta = round(metrics["attention_score"] - comparison["attention_score"], 1)
        if delta > 0:
            add("attention_score_delta", f"+{delta}", "Score improved vs previous session",
                "Build on gains with highway or night driving practice.", "success")
        elif delta < 0:
            add("attention_score_delta", str(delta), "Score declined vs previous session",
                "Review distraction triggers from the prior stronger session.", "warning")

    return insights


def generate_coaching_report(
    metrics: dict,
    session_meta: dict,
    comparison: Optional[dict] = None,
) -> str:
    score = metrics["attention_score"]
    risk = metrics["risk_level"]
    road = metrics["road_attention_pct"]
    dist = metrics["distraction_events"]
    hazard = metrics["hazard_awareness"]
    stability = metrics["gaze_stability"]
    level = metrics["driver_level"]
    reaction = metrics.get("avg_hazard_reaction_time_s", 0)
    scanning = metrics.get("environmental_scanning", 0)
    maturity = metrics.get("attention_maturity", score)

    lines = [
        "**Session Analysis Report**",
        f"Driver Attention Level: {level} | Risk: {risk} | Score: {score}/100",
        f"Attention Maturity: {maturity}/100 | Environmental Scanning: {scanning}%",
        "",
    ]

    if comparison:
        delta = round(score - comparison["attention_score"], 1)
        road_delta = round(road - comparison["road_attention_pct"], 1)
        dist_delta = comparison["distraction_events"] - dist
        prev_reaction = comparison.get("avg_hazard_reaction_time_s", 0)
        sign = "↑" if delta > 0 else "↓"
        lines.append(
            f"Your attention score {sign} {abs(delta)} points from last session. "
            f"Road focus {'increased' if road_delta > 0 else 'decreased'} by {abs(road_delta)}%, "
            f"and prolonged distraction events {'reduced' if dist_delta > 0 else 'increased'} by {abs(dist_delta)}."
        )
        if reaction > 0 and prev_reaction > 0:
            lines.append(
                f"Hazard reaction time: {prev_reaction}s → {reaction}s "
                f"({'improved' if reaction < prev_reaction else 'slower'})."
            )
    else:
        lines.append(f"First session recorded. Attention score: {score}/100 with {road}% road attention.")

    lines.append("")
    insights = build_coaching_insights(metrics, comparison)
    for ins in insights:
        prefix = "✅" if ins["severity"] == "success" else "⚠️" if ins["severity"] == "warning" else "•"
        lines.append(f"{prefix} {ins['evidence']}")

    lines.append("")
    lines.append("**Recommendations:**")
    recs = [i["recommendation"] for i in insights if i["severity"] != "success"]
    if not recs:
        lines.append("• Strong performance — challenge yourself with complex traffic scenarios.")
    else:
        for r in recs[:4]:
            lines.append(f"• {r}")

    scene = metrics.get("scene_distribution", {})
    if scene:
        lines.append("")
        lines.append("**Scene-Aware Attention:**")
        for k, label in [
            ("road", "Road"), ("vehicles", "Vehicles"), ("traffic_signs", "Signs"),
            ("pedestrians", "Pedestrians"), ("traffic_lights", "Traffic Lights"),
            ("distractors", "Distractors"),
        ]:
            if scene.get(k, 0) > 0:
                lines.append(f"• {label}: {scene[k]}%")

    return "\n".join(lines)


def llm_context(session: dict, prior_sessions: Optional[List[dict]] = None) -> str:
    m = session["metrics"]
    parts = [
        f"Session ID: {session['session_id']}",
        f"Attention Score: {m['attention_score']}/100",
        f"Risk: {m['risk_level']} | Level: {m['driver_level']}",
        f"Road Attention: {m['road_attention_pct']}%",
        f"Gaze Stability: {m['gaze_stability']}",
        f"Hazard Awareness: {m['hazard_awareness']}",
        f"Hazard Reaction Time: {m.get('avg_hazard_reaction_time_s', 'N/A')}s",
        f"Environmental Scanning: {m.get('environmental_scanning', 'N/A')}%",
        f"Distraction Events: {m['distraction_events']}",
        f"Off-road Glances: {m.get('off_road_glances', 0)}",
        f"Mirror Checks: {m['mirror_checks_pct']}%",
        f"Scene Source: {m.get('scene_source', 'roi')}",
        f"Session Type: {session.get('session_type')} | Weather: {session.get('weather')}",
    ]
    if prior_sessions:
        parts.append("\nPrior sessions for this driver:")
        for ps in prior_sessions[-3:]:
            pm = ps["metrics"]
            parts.append(
                f"  #{ps['session_id']}: score={pm['attention_score']}, road={pm['road_attention_pct']}%"
            )
    return "\n".join(parts)
