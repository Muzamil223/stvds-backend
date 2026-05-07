"""
detector.py
-----------
Core detection engine using YOLOv8.

Violation Logic:
  1. Triple Riding   — 3+ persons detected in close spatial proximity (clustered).
  2. Mobile Phone    — 'cell phone' class detected near a person bounding box.
  3. No Helmet       — Person detected on/near motorcycle without helmet region.
                       NOTE: For production, use a custom YOLOv8 model trained on
                       a helmet dataset (e.g., Safety-Helmet-Dataset on Roboflow).
                       The demo uses a lightweight heuristic on the head region.

COCO class IDs used:
  0  = person
  3  = motorcycle
  67 = cell phone
"""

import cv2
import numpy as np
import os

from ultralytics import YOLO


# ── COCO class indices ──────────────────────────────────────────────────────
PERSON     = 0
BICYCLE    = 1
MOTORCYCLE = 3
CELL_PHONE = 67

# ── Visual constants ────────────────────────────────────────────────────────
RED    = (0, 0, 255)
GREEN  = (0, 200, 80)
YELLOW = (0, 200, 255)
WHITE  = (255, 255, 255)
FONT   = cv2.FONT_HERSHEY_SIMPLEX


class ViolationDetector:
    """
    Wraps a YOLOv8 model and applies traffic-violation heuristics
    frame-by-frame.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.40):
        """
        Args:
            model_path: Path to YOLOv8 weights (downloads automatically if not present).
            confidence: Minimum confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    # ── Public API ──────────────────────────────────────────────────────────

    def detect_violations(self, frames: list[dict], result_dir: str) -> list[dict]:
        """
        Run detection across all extracted frames.

        Returns:
            List of violation records, one per frame that contained violations.
        """
        violations = []

        for frame_info in frames:
            frame_path = frame_info["path"]
            timestamp  = frame_info["timestamp"]
            frame_idx  = frame_info["index"]

            img = cv2.imread(frame_path)
            if img is None:
                continue

            detections = self._run_yolo(img)
            persons     = detections["persons"]
            phones      = detections["phones"]
            motorcycles = detections["motorcycles"]

            frame_violations = []
            annotated = img.copy()

            # ── Draw baseline green boxes for all persons ──────────────────
            for p in persons:
                self._draw_box(annotated, p["bbox"], GREEN, "Person", 0.5)

            # ── Check: Triple Riding ───────────────────────────────────────
            if len(persons) >= 3:
                clusters = self._find_clusters(persons, radius=220)
                for cluster in clusters:
                    if len(cluster) >= 3:
                        frame_violations.append({
                            "type": "Triple Riding",
                            "severity": "High",
                            "detail": f"{len(cluster)} persons detected on vehicle",
                        })
                        for p in cluster:
                            self._draw_box(annotated, p["bbox"], RED, "TRIPLE RIDING", 0.55)

            # ── Check: Mobile Phone Usage ──────────────────────────────────
            seen_phone_violation = False
            for phone in phones:
                for person in persons:
                    if self._boxes_overlap_or_near(phone["bbox"], person["bbox"], slack=120):
                        if not seen_phone_violation:
                            frame_violations.append({
                                "type": "Mobile Phone Usage",
                                "severity": "Medium",
                                "detail": f"Phone detected near driver (conf: {phone['conf']:.2f})",
                            })
                            seen_phone_violation = True
                        self._draw_box(annotated, phone["bbox"], RED, "PHONE USAGE", 0.55)
                        break

            # ── Check: No Helmet ──────────────────────────────────────────
            for person in persons:
                if self._person_near_motorcycle(person, motorcycles, slack=150):
                    helmet_present = self._check_helmet_region(img, person["bbox"])
                    if not helmet_present:
                        frame_violations.append({
                            "type": "No Helmet",
                            "severity": "High",
                            "detail": "Rider detected without helmet",
                        })
                        self._draw_box(annotated, person["bbox"], RED, "NO HELMET", 0.55)

            # ── Timestamp overlay ──────────────────────────────────────────
            self._add_timestamp(annotated, timestamp)

            # ── Save annotated evidence if violations found ────────────────
            if frame_violations:
                ev_name = f"violation_{frame_idx:05d}.jpg"
                ev_path = os.path.join(result_dir, ev_name)
                cv2.imwrite(ev_path, annotated)

                violations.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "violations": frame_violations,
                    "evidence_image": ev_name,
                    "total_persons": len(persons),
                    "total_phones": len(phones),
                })

        return violations

    def detect_frame_live(self, frame: np.ndarray) -> np.ndarray:
        """
        Annotate a single frame for the live-camera stream.
        Returns the annotated frame.
        """
        detections = self._run_yolo(frame)
        persons     = detections["persons"]
        phones      = detections["phones"]
        motorcycles = detections["motorcycles"]
        annotated = frame.copy()

        for p in persons:
            self._draw_box(annotated, p["bbox"], GREEN, f"Person {p['conf']:.2f}", 0.5)

        for ph in phones:
            self._draw_box(annotated, ph["bbox"], RED, f"Phone {ph['conf']:.2f}", 0.5)

        for m in motorcycles:
            self._draw_box(annotated, m["bbox"], YELLOW, f"Moto {m['conf']:.2f}", 0.5)

        # Triple riding label
        if len(persons) >= 3:
            cv2.putText(annotated, "! TRIPLE RIDING DETECTED",
                        (10, 60), FONT, 0.9, RED, 2)

        # Live Helmet Detection
        for person in persons:
            if self._person_near_motorcycle(person, motorcycles, slack=150):
                helmet_present = self._check_helmet_region(frame, person["bbox"])
                if not helmet_present:
                    self._draw_box(annotated, person["bbox"], RED, "NO HELMET", 0.55)
                    cv2.putText(annotated, "! NO HELMET", (10, 90), FONT, 0.9, RED, 2)

        self._add_timestamp(annotated, "LIVE")
        return annotated

    # ── Internal helpers ────────────────────────────────────────────────────

    def _run_yolo(self, img: np.ndarray) -> dict:
        results = self.model(img, conf=self.confidence, verbose=False)[0]
        persons, phones, motorcycles = [], [], []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            entry = {"bbox": (x1, y1, x2, y2), "conf": conf}

            if cls == PERSON:
                persons.append(entry)
            elif cls == CELL_PHONE:
                phones.append(entry)
            elif cls in [MOTORCYCLE, BICYCLE]:
                motorcycles.append(entry)

        return {"persons": persons, "phones": phones, "motorcycles": motorcycles}

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _distance(self, bbox1, bbox2):
        c1, c2 = self._center(bbox1), self._center(bbox2)
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _boxes_overlap_or_near(self, box1, box2, slack=100):
        return self._distance(box1, box2) < slack

    def _find_clusters(self, persons, radius=220):
        """
        Simple greedy clustering — group persons within `radius` pixels of each other.
        """
        visited = [False] * len(persons)
        clusters = []

        for i, p in enumerate(persons):
            if visited[i]:
                continue
            cluster = [p]
            visited[i] = True
            for j, q in enumerate(persons):
                if not visited[j] and self._distance(p["bbox"], q["bbox"]) < radius:
                    cluster.append(q)
                    visited[j] = True
            clusters.append(cluster)

        return clusters

    def _person_near_motorcycle(self, person, motorcycles, slack=200):
        p_box = person["bbox"]
        for m in motorcycles:
            m_box = m["bbox"]
            # Check for center distance
            if self._distance(p_box, m_box) < slack:
                return True
            # Also check for significant overlap
            if self._boxes_overlap(p_box, m_box):
                return True
        return False

    def _boxes_overlap(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Check if they overlap at all
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)
        
        if ix1 < ix2 and iy1 < iy2:
            return True
        return False

    def _check_helmet_region(self, img: np.ndarray, person_bbox) -> bool:
        """
        Heuristic helmet check on the head region (top ~15% of person bbox).
        """
        x1, y1, x2, y2 = person_bbox
        # Head is at the top. Let's take a small crop.
        head_h = max(1, (y2 - y1) // 7)
        # Narrow the width a bit to focus on the center of the head
        w = x2 - x1
        cx1 = x1 + w // 4
        cx2 = x2 - w // 4
        head_region = img[y1: y1 + head_h, cx1:cx2]

        if head_region.size == 0:
            return True  # Cannot determine

        # Convert to HSV
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # We look for "non-hair-like" properties or specific helmet colors.
        # This is still a heuristic. 
        # Hair is usually very low saturation and low value (black) or low saturation and medium value (brown/blonde).
        # Helmets are often high saturation or have very distinct bright spots.
        
        # If we find ANY high saturation or very high brightness, it might be a helmet.
        # Or if it's very dark but NOT hair-textured (hard to tell).
        
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        
        mean_v = v_channel.mean()
        mean_s = s_channel.mean()
        
        # If mean saturation is high, it's likely a helmet (hair isn't saturated)
        if mean_s > 50:
            return True
            
        # If it's very bright, it's likely a helmet
        if mean_v > 160:
            return True
            
        # If it's very dark, it could be black hair OR a black helmet.
        # This is the tricky part. For a "safety" system, we might want to be strict,
        # but the user says it's NOT working (meaning it's NOT detecting violations).
        # This means it's returning True (helmet present) too often.
        
        # If it's dark (mean_v < 80) and low saturation, it's likely hair.
        if mean_v < 80 and mean_s < 30:
            return False # Likely hair -> No helmet
            
        return True # Default to helmet present to avoid false positives (harassment)

    def _draw_box(self, img, bbox, color, label, font_scale=0.5):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_bg_y = max(y1 - 22, 0)
        tw, th = cv2.getTextSize(label, FONT, font_scale, 2)[0]
        cv2.rectangle(img, (x1, label_bg_y), (x1 + tw + 4, label_bg_y + th + 6), color, -1)
        cv2.putText(img, label, (x1 + 2, label_bg_y + th + 2), FONT, font_scale, WHITE, 2)

    def _add_timestamp(self, img, timestamp: str):
        overlay_text = f"  STVDS | {timestamp}  "
        cv2.rectangle(img, (0, 0), (img.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(img, overlay_text, (8, 24), FONT, 0.65, WHITE, 1)
