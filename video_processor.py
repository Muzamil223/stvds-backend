"""
video_processor.py
------------------
Responsible for extracting frames from uploaded video files.
Uses OpenCV for efficient frame sampling at a configurable FPS rate.
"""

import cv2
import os
from datetime import timedelta


def extract_frames(video_path: str, output_dir: str, sample_fps: int = 2) -> list[dict]:
    """
    Extract frames from a video at a given sample rate.

    Args:
        video_path: Absolute path to the video file.
        output_dir: Directory where extracted frames will be saved.
        sample_fps: How many frames per second to sample (default: 2).

    Returns:
        List of dicts containing frame metadata:
            { index, path, timestamp }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(video_fps / sample_fps))

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    extracted = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            seconds = frame_idx / video_fps
            timestamp = str(timedelta(seconds=int(seconds)))

            filename = f"frame_{saved_idx:05d}.jpg"
            frame_path = os.path.join(frames_dir, filename)
            cv2.imwrite(frame_path, frame)

            extracted.append({
                "index": saved_idx,
                "path": frame_path,
                "timestamp": timestamp,
                "video_frame_number": frame_idx,
            })
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return extracted


def get_video_metadata(video_path: str) -> dict:
    """
    Returns basic metadata about a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_secs = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "resolution": f"{width}x{height}",
        "duration_seconds": round(duration_secs, 2),
        "duration_formatted": str(timedelta(seconds=int(duration_secs))),
    }
