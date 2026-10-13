"""
Alert Pipeline
──────────────
Thread-3: Evaluates every InferenceResult against the alert ruleset.

Rules (as specified):
  1. HIGH ALERT   → any detection of a HIGH_RISK_CLASSES object (weapon/mask),
                    anywhere in the frame.
  2. MODERATE ALERT → any ROI_SENSITIVE_CLASSES object (person/animal/vehicle)
                      whose bounding-box centre is INSIDE the drawn ROI polygon.
  3. No alert     → 'personnel' class, or any class outside the ROI.

Additional features:
  • Per-alert-level cooldown timer prevents alert flooding.
  • Structured AlertEvent objects are emitted to `alert_queue` for downstream
    consumers (display banner, logging, optional webhook/notification).
  • ROI polygon membership uses OpenCV pointPolygonTest (sub-pixel accurate).
"""

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Optional

import numpy as np

from config.settings import (
    ALERT_COOLDOWN_S,
    HIGH_RISK_CLASSES,
    ROI_SENSITIVE_CLASSES,
)
from pipelines.inference import Detection, InferenceResult
from utils.logger import get_logger
from utils.roi_selector import point_in_polygon

log = get_logger(__name__)

# Alert level constants
LEVEL_HIGH     = "HIGH"
LEVEL_MODERATE = "MODERATE"
LEVEL_NONE     = "NONE"


# ─── Alert event ──────────────────────────────────────────────────────────────

@dataclass
class AlertEvent:
    level:       str                # "HIGH" | "MODERATE"
    trigger:     str                # human-readable reason
    detections:  list[Detection]    # detections that caused this alert
    frame_id:    int
    timestamp:   float
    in_roi:      bool = False       # True if triggered by ROI intrusion


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class AlertPipeline:
    """
    Consumes InferenceResult objects, evaluates rules, emits AlertEvent objects.

    Usage:
        alert_pipe = AlertPipeline(result_queue, roi_polygon)
        alert_pipe.start()
        event: AlertEvent = alert_pipe.alert_queue.get()
        alert_pipe.stop()
    """

    def __init__(
        self,
        result_queue: Queue,
        roi_polygon:  np.ndarray,   # shape (N, 2), dtype int32
    ):
        self.result_queue = result_queue
        self.roi_polygon  = roi_polygon
        self.alert_queue  = Queue(maxsize=20)

        # Last time each alert level was fired (for cooldown)
        self._last_alert: dict[str, float] = {
            LEVEL_HIGH:     0.0,
            LEVEL_MODERATE: 0.0,
        }
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(
            target=self._run, name="AlertPipeline", daemon=True
        )

    # ─── Public API ───────────────────────────────────────────────────────────
    def start(self) -> "AlertPipeline":
        log.info("Starting alert pipeline …")
        self._thread.start()
        return self

    def stop(self):
        log.info("Stopping alert pipeline …")
        self._stop_event.set()
        self._thread.join(timeout=5)
        log.info("Alert pipeline stopped.")

    def update_roi(self, roi_polygon: np.ndarray):
        """Hot-swap ROI without restarting the pipeline."""
        with self._lock:
            self.roi_polygon = roi_polygon
        log.info("ROI updated in alert pipeline.")

    # ─── Internal ─────────────────────────────────────────────────────────────
    def _cooldown_ok(self, level: str) -> bool:
        """Returns True if the cooldown for this level has elapsed."""
        now = time.time()
        with self._lock:
            if now - self._last_alert.get(level, 0.0) >= ALERT_COOLDOWN_S.get(level, 5.0):
                self._last_alert[level] = now
                return True
        return False

    def _evaluate(self, result: InferenceResult) -> Optional[AlertEvent]:
        """
        Apply the three-tier rule set.
        Returns the highest-priority AlertEvent, or None.
        """
        high_triggers:     list[Detection] = []
        moderate_triggers: list[Detection] = []

        with self._lock:
            roi = self.roi_polygon

        for det in result.detections:
            name = det.class_name

            # ── Rule 1: HIGH — risk objects anywhere ──────────────────────────
            if name in HIGH_RISK_CLASSES:
                high_triggers.append(det)

            # ── Rule 2: MODERATE — sensitive class inside ROI ─────────────────
            elif name in ROI_SENSITIVE_CLASSES:
                if point_in_polygon(det.center, roi):
                    moderate_triggers.append(det)

            # ── Rule 3: personnel / unknown → no alert ────────────────────────
            # (implicit — do nothing)

        # High trumps moderate
        if high_triggers and self._cooldown_ok(LEVEL_HIGH):
            names = list({d.class_name for d in high_triggers})
            return AlertEvent(
                level      = LEVEL_HIGH,
                trigger    = f"High-risk object detected: {', '.join(names)}",
                detections = high_triggers,
                frame_id   = result.frame_id,
                timestamp  = result.timestamp,
                in_roi     = False,
            )

        if moderate_triggers and self._cooldown_ok(LEVEL_MODERATE):
            names = list({d.class_name for d in moderate_triggers})
            return AlertEvent(
                level      = LEVEL_MODERATE,
                trigger    = f"Restricted zone intrusion: {', '.join(names)}",
                detections = moderate_triggers,
                frame_id   = result.frame_id,
                timestamp  = result.timestamp,
                in_roi     = True,
            )

        return None

    def _emit(self, event: AlertEvent):
        """Push alert to queue; log it regardless."""
        log.warning(
            f"[ALERT:{event.level}] {event.trigger} "
            f"| frame={event.frame_id} "
            f"| detections={len(event.detections)}"
        )
        if not self.alert_queue.full():
            self.alert_queue.put_nowait(event)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                result: InferenceResult = self.result_queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                event = self._evaluate(result)
            except Exception as exc:
                log.error(f"Alert evaluation error: {exc}")
                continue

            if event:
                self._emit(event)
