"""
Display Pipeline
────────────────
Runs on the MAIN thread (OpenCV imshow must stay on the thread that
created the window on most platforms).

Responsibilities:
  • Poll the inference result queue for latest annotated frame.
  • Poll the alert queue and overlay a colour-coded banner.
  • Draw bounding boxes with class labels and confidence.
  • Draw the ROI polygon overlay.
  • Compute and display rolling-average FPS.
  • Handle keyboard quit (q / ESC).
"""

import time
from collections import deque
from queue import Empty, Queue

import cv2
import numpy as np

from config.settings import (
    ALERT_BANNER_DURATION_S,
    CLASS_NAMES,
    DISPLAY_FPS_SMOOTHING,
    DISPLAY_WINDOW_NAME,
    HIGH_RISK_CLASSES,
    ROI_SENSITIVE_CLASSES,
    SHOW_DETECTIONS,
    SHOW_FPS,
    SHOW_ROI,
)
from pipelines.alert import AlertEvent, LEVEL_HIGH, LEVEL_MODERATE
from pipelines.inference import InferenceResult
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Colour palette ───────────────────────────────────────────────────────────
# BGR format

COLORS = {
    "large_gun":    (0,   0,   255),   # red
    "short_gun":    (0, 0, 220),
    "matchet/knife":      (0,   0,   200),   # dark red
    "person":    (255, 165,   0),   # orange
    "animal":    (0,   200, 100),   # green
    "vehicle":   (200, 100,   0),   # teal-ish
    "personnel": (0,   255,   0),   # bright green
    "default":   (200, 200, 200),   # grey
}

ALERT_BANNER_COLORS = {
    LEVEL_HIGH:     (0,   0,   200),   # red
    LEVEL_MODERATE: (0,  140,  255),   # orange
}

ROI_NORMAL_COLOR = (0, 255, 255)       # cyan
ROI_ALERT_COLOR  = (0,   0, 255)       # red when alert active

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2


class DisplayPipeline:
    """
    Drives the OpenCV window.  Call `run()` — it blocks until the user
    presses 'q' or ESC, or the capture pipeline dies.

    Usage:
        display = DisplayPipeline(result_queue, alert_queue, roi_polygon)
        display.run()   # blocks — call from main thread
    """

    def __init__(
        self,
        result_queue: Queue,
        alert_queue:  Queue,
        roi_polygon:  np.ndarray,
        frame_queue:  Queue | None = None,   # raw frames fallback
    ):
        self.result_queue = result_queue
        self.alert_queue  = alert_queue
        self.roi_polygon  = roi_polygon
        self.frame_queue  = frame_queue      # <-- add this

        # Rolling FPS buffer
        self._frame_times: deque = deque(maxlen=DISPLAY_FPS_SMOOTHING)

        # Active alert state
        self._active_alert:      AlertEvent | None = None
        self._alert_expire_time: float             = 0.0

        # Last rendered frame (re-displayed if no new result yet)
        self._last_canvas: np.ndarray | None = None

    # ─── Public ───────────────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        log.info(f"Display window opened: '{DISPLAY_WINDOW_NAME}'")

        while True:
            # ── Poll inference result ─────────────────────────────────────────
            try:
                result: InferenceResult = self.result_queue.get_nowait()
                canvas = self._render(result)   # bounding boxes drawn here
                self._last_canvas = canvas      # always prefer inferred frame
            except Empty:
                # No new inferred frame — only use raw as bootstrap before first inference
                if self._last_canvas is None and self.frame_queue is not None:
                    try:
                        raw = self.frame_queue.get_nowait()
                        canvas = raw.copy()
                    except Empty:
                        canvas = None
                else:
                    canvas = self._last_canvas  # re-display last inferred frame

            if canvas is None:
                # Nothing received yet — show waiting screen
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(canvas, "Waiting for stream …",
                            (50, 360), FONT, 1.2, (200, 200, 200), 2)

            # ── Poll alerts ───────────────────────────────────────────────────
            try:
                event: AlertEvent = self.alert_queue.get_nowait()
                self._active_alert      = event
                self._alert_expire_time = time.time() + ALERT_BANNER_DURATION_S
            except Empty:
                pass

            # ── Draw alert banner ─────────────────────────────────────────────
            if self._active_alert and time.time() < self._alert_expire_time:
                canvas = self._draw_alert_banner(canvas, self._active_alert)
                roi_color = ROI_ALERT_COLOR
            else:
                self._active_alert = None
                roi_color = ROI_NORMAL_COLOR

            # ── Draw ROI polygon ──────────────────────────────────────────────
            if SHOW_ROI:
                self._draw_roi(canvas, roi_color)

            # ── Draw FPS ──────────────────────────────────────────────────────
            if SHOW_FPS:
                self._draw_fps(canvas)

            cv2.imshow(DISPLAY_WINDOW_NAME, canvas)

            # ── Keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q or ESC
                log.info("Quit key pressed — shutting down display.")
                break

        cv2.destroyAllWindows()

    def update_roi(self, roi_polygon: np.ndarray):
        self.roi_polygon = roi_polygon

    # ─── Rendering helpers ────────────────────────────────────────────────────
    def _render(self, result: InferenceResult) -> np.ndarray:
        canvas = result.frame.copy()

        if SHOW_DETECTIONS:
            for det in result.detections:
                self._draw_detection(canvas, det)

        return canvas

    def _draw_detection(self, canvas: np.ndarray, det) -> None:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = COLORS.get(det.class_name, COLORS["default"])
        label = f"{det.class_name} {det.confidence:.0%}"

        # Box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, THICKNESS)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(canvas, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 4),
                    FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

        # Centre dot (used for ROI test visualisation)
        cv2.circle(canvas, det.center, 4, color, -1)

    def _draw_roi(self, canvas: np.ndarray, color: tuple) -> None:
        pts = self.roi_polygon.reshape((-1, 1, 2))

        # Semi-transparent fill
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [self.roi_polygon], color)
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

        # Border
        cv2.polylines(canvas, [pts], isClosed=True,
                      color=color, thickness=2, lineType=cv2.LINE_AA)

        # Label
        tx, ty = int(self.roi_polygon[:, 0].min()), int(self.roi_polygon[:, 1].min()) - 8
        cv2.putText(canvas, "RESTRICTED ZONE", (max(tx, 0), max(ty, 15)),
                    FONT, 0.5, color, 1, cv2.LINE_AA)

    def _draw_alert_banner(self, canvas: np.ndarray, event: AlertEvent) -> np.ndarray:
        h, w = canvas.shape[:2]
        banner_h = 60
        color    = ALERT_BANNER_COLORS.get(event.level, (100, 100, 100))

        # Filled banner strip at top
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Alert text
        text  = f"⚠  {event.level} ALERT  |  {event.trigger}"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.75, 2)
        tx    = max(10, (w - tw) // 2)
        cv2.putText(canvas, text, (tx, 40), FONT, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def _draw_fps(self, canvas: np.ndarray) -> None:
        now = time.time()
        self._frame_times.append(now)

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps     = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0
            text    = f"Display: {fps:.1f} FPS"
        else:
            text = "FPS: --"

        h, _ = canvas.shape[:2]
        cv2.putText(canvas, text, (10, h - 12),
                    FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
