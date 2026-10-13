"""
ROI Selector
────────────
Lets the operator draw a polygon on a grabbed frame at startup.
Left-click  → add vertex
Right-click → undo last vertex
Enter / 'q' → confirm and close

Returns a list of [x, y] pixel points.
"""

import cv2
import numpy as np

from config.settings import DEFAULT_ROI_POINTS
from utils.logger import get_logger

log = get_logger(__name__)


class ROISelector:
    """
    Interactive polygon-drawing tool.
    Pass a single representative frame; the operator clicks vertices.
    """

    POINT_COLOR    = (0, 255, 255)   # cyan dots
    LINE_COLOR     = (0, 255, 255)
    FILL_COLOR     = (0, 255, 0)
    FONT           = cv2.FONT_HERSHEY_SIMPLEX
    WIN            = "ROI Selector  |  Left-click: add point  |  Right-click: undo  |  ENTER: confirm"

    def __init__(self, frame: np.ndarray):
        self._base   = frame.copy()
        self._canvas = frame.copy()
        self._points: list[list[int]] = []

    # ─── Mouse callback ────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._points.append([x, y])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and self._points:
            self._points.pop()
            self._redraw()

    def _redraw(self):
        self._canvas = self._base.copy()
        pts = self._points
        n   = len(pts)

        if n == 0:
            return

        # Draw filled semi-transparent polygon once ≥3 points
        if n >= 3:
            overlay = self._canvas.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], self.FILL_COLOR)
            cv2.addWeighted(overlay, 0.25, self._canvas, 0.75, 0, self._canvas)

        # Draw edges
        for i in range(n - 1):
            cv2.line(self._canvas, tuple(pts[i]), tuple(pts[i + 1]), self.LINE_COLOR, 2)
        if n >= 3:
            cv2.line(self._canvas, tuple(pts[-1]), tuple(pts[0]), self.LINE_COLOR, 2)

        # Draw vertices
        for p in pts:
            cv2.circle(self._canvas, tuple(p), 6, self.POINT_COLOR, -1)

        # Instruction overlay
        cv2.putText(
            self._canvas,
            f"Vertices: {n}  (right-click to undo, ENTER to confirm)",
            (10, 30), self.FONT, 0.6, (255, 255, 255), 2,
        )

    # ─── Public ───────────────────────────────────────────────────────────────
    def run(self) -> list[list[int]]:
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WIN, self._on_mouse)
        self._redraw()

        log.info("ROI Selector opened. Draw your zone and press ENTER to confirm.")

        while True:
            cv2.imshow(self.WIN, self._canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, ord("q")):   # Enter or q
                break

        cv2.destroyWindow(self.WIN)

        if len(self._points) < 3:
            log.warning("ROI has fewer than 3 points — falling back to default ROI.")
            return DEFAULT_ROI_POINTS

        log.info(f"ROI confirmed with {len(self._points)} vertices: {self._points}")
        return self._points


def point_in_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    """
    Returns True if (x, y) is inside the given polygon.
    Uses OpenCV's pointPolygonTest; result > 0 means inside.
    """
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0
