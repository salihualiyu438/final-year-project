"""Motion / scene-change gating for the surveillance pipeline.

The camera is continuously captured. This component only decides whether the
current frame is worth sending through the expensive YOLO inference stage.
"""
from __future__ import annotations

import time
import cv2
import numpy as np


class MotionSceneGate:
    """Background-difference scene-change detector with a short hold period."""

    def __init__(
        self,
        enabled: bool = True,
        downscale_width: int = 320,
        pixel_threshold: int = 25,
        min_area_ratio: float = 0.003,
        blur_size: int = 5,
        background_alpha: float = 0.05,
        warmup_frames: int = 15,
        hold_seconds: float = 2.0,
        force_inference_seconds: float = 5.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.downscale_width = int(downscale_width)
        self.pixel_threshold = int(pixel_threshold)
        self.min_area_ratio = float(min_area_ratio)
        self.blur_size = max(3, int(blur_size) | 1)
        self.background_alpha = float(background_alpha)
        self.warmup_frames = int(warmup_frames)
        self.hold_seconds = float(hold_seconds)
        self.force_inference_seconds = float(force_inference_seconds)
        self.background: np.ndarray | None = None
        self.warmup = 0
        self.active_until = 0.0
        self.last_forced = 0.0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.reset()

    def reset(self) -> None:
        self.background = None
        self.warmup = 0
        self.active_until = 0.0
        self.last_forced = 0.0

    def check(self, frame: np.ndarray) -> tuple[bool, float]:
        """Return (should_run_inference, changed_pixel_ratio)."""
        now = time.time()
        if not self.enabled:
            return True, 1.0

        h, w = frame.shape[:2]
        if w > self.downscale_width:
            nh = max(1, int(h * self.downscale_width / w))
            work = cv2.resize(frame, (self.downscale_width, nh))
        else:
            work = frame

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0).astype(np.float32)

        if self.background is None:
            self.background = gray.copy()
            self.warmup = 1
            self.last_forced = now
            return True, 1.0

        diff = cv2.absdiff(gray.astype(np.uint8), self.background.astype(np.uint8))
        changed_ratio = float(np.count_nonzero(diff >= self.pixel_threshold)) / max(1, diff.size)
        cv2.accumulateWeighted(gray, self.background, self.background_alpha)

        if self.warmup < self.warmup_frames:
            self.warmup += 1
            return True, changed_ratio

        if changed_ratio >= self.min_area_ratio:
            self.active_until = now + self.hold_seconds

        force = (now - self.last_forced) >= self.force_inference_seconds
        if force:
            self.last_forced = now

        return (now <= self.active_until or force), changed_ratio
