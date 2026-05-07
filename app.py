"""
app.py
------
Flask REST API for the Smart Traffic Violation Detection System (STVDS).

Endpoints:
  POST /api/upload              — Upload a video file, returns job_id
  POST /api/process/<job_id>    — Trigger detection pipeline
  GET  /api/status/<job_id>     — Poll job status (progress, errors)
  GET  /api/results/<job_id>    — Fetch cached report JSON
  GET  /api/results/<job_id>/images/<filename> — Serve annotated evidence images
  GET  /api/health              — Health check
  GET  /api/live/stream         — MJPEG live camera stream (uses webcam)
"""

import os
import json
import uuid
import threading
import time

import cv2
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

from video_processor import extract_frames, get_video_metadata
from detector import ViolationDetector
from report_generator import generate_report

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

# ── Singleton detector (loads model once) ────────────────────────────────────
detector = ViolationDetector()

# ── In-memory job status store ───────────────────────────────────────────────
_job_status: dict[str, dict] = {}   # { job_id: { status, progress, error } }
_job_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _find_video(job_id: str) -> str | None:
    for f in os.listdir(UPLOAD_FOLDER):
        if f.startswith(job_id):
            return os.path.join(UPLOAD_FOLDER, f)
    return None


def _set_status(job_id: str, status: str, progress: int = 0, error: str = ""):
    with _job_lock:
        _job_status[job_id] = {"status": status, "progress": progress, "error": error}


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "YOLOv8n", "version": "1.0.0"})


# ── Upload ───────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video field in request"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}. Use {allowed}"}), 415

    job_id   = str(uuid.uuid4())
    safe_name = f"{job_id}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(save_path)

    meta = get_video_metadata(save_path)
    _set_status(job_id, "uploaded")

    return jsonify({
        "job_id":   job_id,
        "filename": safe_name,
        "metadata": meta,
        "message":  "Upload successful. Call /api/process/<job_id> to start detection.",
    }), 201


# ── Process (async in background thread) ────────────────────────────────────

@app.route("/api/process/<job_id>", methods=["POST"])
def process_video(job_id: str):
    video_path = _find_video(job_id)
    if not video_path:
        return jsonify({"error": "Video not found for this job_id"}), 404

    with _job_lock:
        current = _job_status.get(job_id, {})
        if current.get("status") == "processing":
            return jsonify({"message": "Already processing", "job_id": job_id}), 202

    def _run():
        try:
            _set_status(job_id, "processing", 10)
            result_dir = os.path.join(RESULTS_FOLDER, job_id)
            os.makedirs(result_dir, exist_ok=True)

            _set_status(job_id, "processing", 20)
            frames = extract_frames(video_path, result_dir, sample_fps=2)

            _set_status(job_id, "processing", 50)
            violations = detector.detect_violations(frames, result_dir)

            _set_status(job_id, "processing", 85)
            report = generate_report(violations, job_id)

            report_path = os.path.join(result_dir, "report.json")
            with open(report_path, "w") as fp:
                json.dump(report, fp, indent=2)

            _set_status(job_id, "completed", 100)
        except Exception as exc:
            _set_status(job_id, "failed", 0, str(exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"message": "Processing started", "job_id": job_id}), 202


# ── Status polling ────────────────────────────────────────────────────────────

@app.route("/api/status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    with _job_lock:
        status = _job_status.get(job_id, {"status": "unknown", "progress": 0})
    return jsonify({**status, "job_id": job_id})


# ── Results ───────────────────────────────────────────────────────────────────

@app.route("/api/results/<job_id>", methods=["GET"])
def get_results(job_id: str):
    report_path = os.path.join(RESULTS_FOLDER, job_id, "report.json")
    if not os.path.exists(report_path):
        return jsonify({"error": "Results not ready yet. Poll /api/status/<job_id>"}), 404

    with open(report_path, "r") as fp:
        report = json.load(fp)

    return jsonify(report)


@app.route("/api/results/<job_id>/images/<path:filename>", methods=["GET"])
def get_evidence_image(job_id: str, filename: str):
    directory = os.path.join(RESULTS_FOLDER, job_id)
    return send_from_directory(directory, filename)


# ── Live camera MJPEG stream ─────────────────────────────────────────────────

def _generate_live_frames():
    """
    Generator that yields MJPEG frames from the default webcam,
    annotated with the YOLO live detector.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Yield a placeholder error frame
        blank = 255 * __import__("numpy").ones((360, 640, 3), dtype="uint8")
        cv2.putText(blank, "No camera available", (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)
        _, buf = cv2.imencode(".jpg", blank)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = detector.detect_frame_live(frame)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.04)   # ~25 fps cap
    finally:
        cap.release()


@app.route("/api/live/stream", methods=["GET"])
def live_stream():
    return Response(
        _generate_live_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Starting STVDS Backend on http://localhost:5000")
    # In production, use gunicorn as specified in requirements.txt:
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)

