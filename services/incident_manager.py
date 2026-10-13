"""Incident lifecycle management for the surveillance system.

An Incident represents one continuous threat condition rather than one
YOLO detection or one notification.  The manager is intentionally in-memory
for this phase; persistence belongs to the later database phase.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from services.threat_engine import ThreatAssessment


@dataclass
class Incident:
    incident_id: str
    started_at: float
    last_seen_at: float
    level: str
    threat_score: int
    reasons: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    in_roi: bool = False
    frame_id: int = 0
    ended_at: Optional[float] = None
    status: str = "ACTIVE"
    snapshot_url: Optional[str] = None
    video_url: Optional[str] = None
    camera_id: str = "camera_1"
    camera_name: str = "Camera 1"
    camera_location: str = ""
    camera_description: str = ""

    @property
    def duration_s(self) -> float:
        end = self.ended_at if self.ended_at is not None else self.last_seen_at
        return max(0.0, end - self.started_at)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "status": self.status,
            "level": self.level,
            "threat_score": self.threat_score,
            "reasons": list(self.reasons),
            "classes": list(self.classes),
            "in_roi": self.in_roi,
            "frame_id": self.frame_id,
            "started_at": self.started_at,
            "last_seen_at": self.last_seen_at,
            "ended_at": self.ended_at,
            "duration_s": round(self.duration_s, 2),
            "snapshot_url": self.snapshot_url,
            "video_url": self.video_url,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "camera_location": self.camera_location,
            "camera_description": self.camera_description,
        }


@dataclass
class IncidentEvent:
    type: str  # incident_created | incident_updated | incident_closed
    incident: Incident


class IncidentManager:
    """Turn continuous threat assessments into deduplicated incidents.

    A new incident is created when a threat appears.  Repeated assessments
    with the same threat signature update the active incident.  An incident
    closes after `absence_timeout_s` seconds without a matching threat.

    No database or video recording is performed here yet.
    """

    def __init__(self, absence_timeout_s: float = 5.0, history_size: int = 200,
                 camera_id: str = "camera_1", camera_name: str = "Camera 1",
                 camera_location: str = "", camera_description: str = ""):
        self.absence_timeout_s = absence_timeout_s
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.camera_location = camera_location
        self.camera_description = camera_description
        self._active: Optional[Incident] = None
        self._history: list[Incident] = []
        self._history_size = history_size
        self._counter = 0
        self._lock = threading.Lock()

    @staticmethod
    def _signature(assessment: ThreatAssessment) -> frozenset[str]:
        return frozenset(d.class_name for d in assessment.trigger_detections)

    @staticmethod
    def _signature_from_classes(classes: list[str]) -> frozenset[str]:
        weapon_classes = {"large_gun", "short_gun", "matchet/knife"}
        weapons = frozenset(c for c in classes if c in weapon_classes)
        # Weapon classes define the incident family when present, preventing
        # an additional object detection from splitting one weapon incident.
        return weapons if weapons else frozenset(classes)

    @staticmethod
    def _same_incident(active: Incident, assessment: ThreatAssessment) -> bool:
        current = IncidentManager._signature_from_classes(active.classes)
        incoming = IncidentManager._signature(assessment)
        incoming_key = IncidentManager._signature_from_classes(list(incoming))
        # Same threat family. The level may change as the event evolves.
        return bool(current and incoming_key and current == incoming_key)

    def process(
        self,
        assessment: ThreatAssessment,
        frame_id: int,
        timestamp: Optional[float] = None,
    ) -> list[IncidentEvent]:
        now = time.time() if timestamp is None else timestamp
        events: list[IncidentEvent] = []

        with self._lock:
            active = self._active

            if not assessment.is_alert:
                if active is not None and now - active.last_seen_at >= self.absence_timeout_s:
                    active.ended_at = active.last_seen_at
                    active.status = "CLOSED"
                    events.append(IncidentEvent("incident_closed", self._copy_incident(active)))
                    self._history.append(active)
                    self._trim_history()
                    self._active = None
                return events

            incoming_classes = sorted(self._signature(assessment))

            if active is not None:
                if self._same_incident(active, assessment):
                    active.last_seen_at = now
                    active.frame_id = frame_id
                    previous_level = active.level
                    active.level = assessment.level
                    active.threat_score = max(active.threat_score, assessment.score)
                    active.reasons = list(dict.fromkeys(active.reasons + assessment.reasons))
                    active.in_roi = active.in_roi or assessment.in_roi

                    # Emit an update only when the severity increases.
                    if _level_rank(assessment.level) > _level_rank(previous_level):
                        events.append(IncidentEvent("incident_updated", self._copy_incident(active)))
                    return events

                # A different threat appeared. Close the old one immediately
                # and create the new incident in the same call.
                active.ended_at = active.last_seen_at
                active.status = "CLOSED"
                events.append(IncidentEvent("incident_closed", self._copy_incident(active)))
                self._history.append(active)
                self._trim_history()
                self._active = None

            self._counter += 1
            incident = Incident(
                incident_id=f"{self.camera_id.upper()}-INC-{int(now * 1000)}-{self._counter:04d}",
                started_at=now,
                last_seen_at=now,
                level=assessment.level,
                threat_score=assessment.score,
                reasons=list(assessment.reasons),
                classes=incoming_classes,
                in_roi=assessment.in_roi,
                frame_id=frame_id,
                camera_id=self.camera_id,
                camera_name=self.camera_name,
                camera_location=self.camera_location,
                camera_description=self.camera_description,
            )
            self._active = incident
            events.append(IncidentEvent("incident_created", self._copy_incident(incident)))

        return events

    def flush(self, timestamp: Optional[float] = None) -> Optional[IncidentEvent]:
        """Close the current incident, useful during clean shutdown."""
        now = time.time() if timestamp is None else timestamp
        with self._lock:
            if self._active is None:
                return None
            self._active.ended_at = min(self._active.last_seen_at, now)
            self._active.status = "CLOSED"
            incident = self._active
            self._history.append(incident)
            self._trim_history()
            self._active = None
            return IncidentEvent("incident_closed", self._copy_incident(incident))

    def attach_media(self, incident_id: str, snapshot_url: str | None = None, video_url: str | None = None) -> bool:
        """Attach local evidence URLs to an active or historical incident."""
        with self._lock:
            if self._active is not None and self._active.incident_id == incident_id:
                if snapshot_url is not None:
                    self._active.snapshot_url = snapshot_url
                if video_url is not None:
                    self._active.video_url = video_url
                return True
            for incident in self._history:
                if incident.incident_id == incident_id:
                    if snapshot_url is not None:
                        incident.snapshot_url = snapshot_url
                    if video_url is not None:
                        incident.video_url = video_url
                    return True
        return False

    def active(self) -> Optional[Incident]:
        with self._lock:
            return self._copy_incident(self._active) if self._active else None

    def history(self) -> list[Incident]:
        with self._lock:
            return [self._copy_incident(x) for x in reversed(self._history)]

    def _trim_history(self):
        if len(self._history) > self._history_size:
            del self._history[:-self._history_size]

    @staticmethod
    def _copy_incident(incident: Incident) -> Incident:
        return Incident(
            incident_id=incident.incident_id,
            started_at=incident.started_at,
            last_seen_at=incident.last_seen_at,
            level=incident.level,
            threat_score=incident.threat_score,
            reasons=list(incident.reasons),
            classes=list(incident.classes),
            in_roi=incident.in_roi,
            frame_id=incident.frame_id,
            ended_at=incident.ended_at,
            status=incident.status,
            snapshot_url=incident.snapshot_url,
            video_url=incident.video_url,
            camera_id=incident.camera_id,
            camera_name=incident.camera_name,
            camera_location=incident.camera_location,
            camera_description=incident.camera_description,
        )


def _level_rank(level: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(level, 0)
