"""
report_generator.py
-------------------
Transforms raw violation records into a structured, frontend-ready report.
"""

from datetime import datetime
from collections import Counter


def generate_report(violations: list[dict], job_id: str) -> dict:
    """
    Build a complete report dict from raw violation records.

    Args:
        violations: Output from ViolationDetector.detect_violations()
        job_id: Unique identifier for this processing job.

    Returns:
        A structured dict suitable for JSON serialization and frontend consumption.
    """
    all_types = []
    for record in violations:
        for v in record["violations"]:
            all_types.append(v["type"])

    type_counts = Counter(all_types)

    summary = {
        "total_violation_frames": len(violations),
        "total_violations": len(all_types),
        "no_helmet_count": type_counts.get("No Helmet", 0),
        "triple_riding_count": type_counts.get("Triple Riding", 0),
        "phone_usage_count": type_counts.get("Mobile Phone Usage", 0),
    }

    severity_map = {
        "No Helmet": "High",
        "Triple Riding": "High",
        "Mobile Phone Usage": "Medium",
    }

    records = []
    for v in violations:
        records.append({
            "frame_index": v["frame_index"],
            "timestamp": v["timestamp"],
            "evidence_image": v["evidence_image"],
            "violations": v["violations"],
            "max_severity": _max_severity([vio["severity"] for vio in v["violations"]]),
        })

    return {
        "job_id": job_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "records": records,
        "status": "completed",
    }


def _max_severity(severities: list[str]) -> str:
    order = {"High": 3, "Medium": 2, "Low": 1}
    return max(severities, key=lambda s: order.get(s, 0), default="Low")
