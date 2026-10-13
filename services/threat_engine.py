"""Threat analysis engine for the perimeter surveillance system.

This module deliberately keeps threat analysis separate from YOLO inference.
YOLO answers: "what is visible?"  This engine answers: "how urgent is it?"
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from pipelines.inference import Detection
from utils.roi_selector import point_in_polygon
from config.settings import CRITICAL_HOURS_START, CRITICAL_HOURS_END

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"

_LEVEL_RANK = {LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3}

# Initial policy. These values are deliberately configuration-like so they can
# later move to a database and become camera/site-specific rules.
BASE_SCORES = {
    "large_gun": 100,
    "short_gun": 95,
    "matchet/knife": 80,
    "animal": 20,
    "vehicle": 15,
    "person": 5,
    "personnel": 0,
}


def level_from_score(score: int) -> str:
    if score >= 90:
        return CRITICAL
    if score >= 65:
        return HIGH
    if score >= 30:
        return MEDIUM
    return LOW


@dataclass
class ThreatAssessment:
    level: str
    score: int
    reasons: list[str] = field(default_factory=list)
    trigger_detections: list[Detection] = field(default_factory=list)
    in_roi: bool = False

    @property
    def is_alert(self) -> bool:
        return self.level != LOW


class ThreatEngine:
    """Evaluate a frame's detections using contextual surveillance rules."""

    def __init__(self, roi_polygon=None):
        self.roi_polygon = roi_polygon
        self.alert_classes = {"person", "animal", "vehicle",
                              "large_gun", "short_gun", "matchet/knife"}

    def update_roi(self, roi_polygon):
        self.roi_polygon = roi_polygon

    def update_alert_classes(self, classes: Iterable[str]):
        # Critical classes are immutable from the operator UI.
        self.alert_classes = set(classes) | {"large_gun", "short_gun", "matchet/knife"}

    def assess(self, detections: Iterable[Detection], authorization_detections: Iterable[Detection] | None = None) -> ThreatAssessment:
        detections = list(detections)
        authorization_detections = list(authorization_detections) if authorization_detections is not None else detections
        score = 0
        reasons: list[str] = []
        triggers: list[Detection] = []
        in_roi = False

        persons = [d for d in detections if d.class_name == "person"]
        vehicles = [d for d in detections if d.class_name == "vehicle"]
        personnel = [d for d in detections if d.class_name == "personnel"]
        # Use raw Model 1 personnel detections for authorization decisions.
        # Temporal confirmation may delay a personnel box, but must not cause
        # a weapon carried by an officer to trigger a false alert meanwhile.
        authorization_personnel = [d for d in authorization_detections
                                   if d.class_name == "personnel"]
        anchors = persons + vehicles + personnel + authorization_personnel

        # The local-time rule applies only when no security personnel are
        # detected. Presence is a critical event from midnight up to (but not
        # including) 06:00, subject to the operator having that class enabled.
        local_hour = datetime.now().astimezone().hour
        critical_time = (CRITICAL_HOURS_START <= local_hour < CRITICAL_HOURS_END
                         and not authorization_personnel)

        for det in detections:
            name = det.class_name

            if name in {"large_gun", "short_gun", "matchet/knife"}:
                associated = any(self._iou(det.bbox_xyxy, a.bbox_xyxy) >= 0.10
                                 for a in anchors)
                if not associated:
                    continue
                authorized = any(self._weapon_carried_by_personnel(det, p)
                                 for p in authorization_personnel)
                if authorized:
                    reasons.append(f"{name} associated with authorized personnel — alert suppressed")
                    continue
                # These classes are mandatory-alert classes and cannot be
                # disabled by operator preferences.
                score = max(score, BASE_SCORES[name])
                triggers.append(det)
                reasons.append(f"{name} associated with a detected person, vehicle, or personnel")

            elif name in {"animal", "vehicle", "person"}:
                if name not in self.alert_classes:
                    continue
                if critical_time and name in {"person", "vehicle"}:
                    score = 100
                    triggers.append(det)
                    reasons.append(f"{name} detected during critical hours (00:00–06:00)")
                    continue
                inside = self._inside_roi(det)
                if inside:
                    score = max(score, BASE_SCORES[name] + 15)
                    in_roi = True
                    triggers.append(det)
                    reasons.append(f"{name} detected inside restricted zone")

        if authorization_personnel and triggers:
            reasons.append("authorized personnel present")

        score = min(100, int(score))
        return ThreatAssessment(
            level=level_from_score(score),
            score=score,
            reasons=reasons,
            trigger_detections=triggers,
            in_roi=in_roi,
        )

    def _inside_roi(self, det: Detection) -> bool:
        if self.roi_polygon is None or len(self.roi_polygon) < 3:
            return False
        return point_in_polygon(det.center, self.roi_polygon)

    @classmethod
    def _weapon_carried_by_personnel(cls, weapon: Detection, person: Detection) -> bool:
        """Return True when the weapon is spatially associated with security personnel.

        A weapon's center inside the personnel box is the strongest signal.
        A small IoU fallback handles cases where the detector produces a
        slightly offset weapon box. This is intentionally conservative: the
        weapon must overlap the authorized-person region before an alert is
        suppressed.
        """
        x1, y1, x2, y2 = person.bbox_xyxy
        cx, cy = weapon.center
        center_inside = x1 <= cx <= x2 and y1 <= cy <= y2
        if center_inside:
            return True
        return cls._iou(weapon.bbox_xyxy, person.bbox_xyxy) >= 0.05

    @staticmethod
    def _iou(box_a: list[float], box_b: list[float]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return inter / (area_a + area_b - inter)
