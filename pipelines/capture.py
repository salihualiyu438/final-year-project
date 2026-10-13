"""
Hardened RTSP Capture Pipeline
──────────────────────────────
The capture path is deliberately isolated from inference and browser output.

Key protections:
  • FFmpeg open/read timeouts prevent the ~30-second OpenCV stalls previously
    observed when RTSP stopped responding.
  • TCP RTSP is preferred to match the proven IP Camera Lite configuration.
  • Reconnection is continuous rather than terminating after five failures.
  • Exponential backoff is capped to keep recovery responsive.
  • Queue(maxsize=1) keeps only the newest frame.
  • Capture resizing happens after decode and therefore reduces downstream
    CPU/memory pressure, not camera/network bandwidth.
"""

import os

# OpenCV's FFmpeg backend reads these options when a VideoCapture is opened.
# Keep this before importing/creating VideoCapture objects.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;7000000",
)

import threading
import time
from queue import Full, Queue

import cv2
import numpy as np

from config.settings import (
    CAMERA_OPEN_TIMEOUT_MS,
    CAMERA_READ_TIMEOUT_MS,
    CAMERA_RECONNECT_DELAY_S,
    CAPTURE_QUEUE_MAXSIZE,
    OPENCV_BUFFER_SIZE,
    RTSP_URL,
)
from utils.logger import get_logger

log = get_logger(__name__)


class CaptureStats:
    """Lightweight, thread-safe counters."""

    def __init__(self):
        self._lock = threading.Lock()
        self.frames_read = 0
        self.frames_dropped = 0
        self.reconnects = 0
        self.open_failures = 0
        self.read_failures = 0

    def inc_read(self):
        with self._lock:
            self.frames_read += 1

    def inc_drop(self):
        with self._lock:
            self.frames_dropped += 1

    def inc_reconnect(self):
        with self._lock:
            self.reconnects += 1

    def inc_open_failure(self):
        with self._lock:
            self.open_failures += 1

    def inc_read_failure(self):
        with self._lock:
            self.read_failures += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frames_read": self.frames_read,
                "frames_dropped": self.frames_dropped,
                "reconnects": self.reconnects,
                "open_failures": self.open_failures,
                "read_failures": self.read_failures,
            }


class CapturePipeline:
    """
    Background RTSP reader.

    The important behavioral change from the previous version is that a
    transient RTSP failure no longer causes the capture thread to permanently
    exit after a fixed number of attempts. It keeps reconnecting until stop()
    is requested.
    """

    def __init__(self, rtsp_url: str = RTSP_URL):
        self.rtsp_url = rtsp_url
        self.frame_queue = Queue(maxsize=CAPTURE_QUEUE_MAXSIZE)
        self.stats = CaptureStats()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="CapturePipeline",
            daemon=True,
        )

    def start(self) -> "CapturePipeline":
        log.info(f"Starting capture pipeline -> {self.rtsp_url}")
        self._thread.start()
        return self

    def stop(self):
        log.info("Stopping capture pipeline …")
        self._stop_event.set()
        self._thread.join(timeout=5)
        log.info("Capture pipeline stopped.")

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _open_stream(self) -> cv2.VideoCapture | None:
        """Open RTSP with bounded FFmpeg open/read timeouts."""

        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            int(CAMERA_OPEN_TIMEOUT_MS),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            int(CAMERA_READ_TIMEOUT_MS),
        ]

        try:
            cap = cv2.VideoCapture(
                self.rtsp_url,
                cv2.CAP_FFMPEG,
                params,
            )
        except (TypeError, cv2.error):
            # Compatibility fallback for OpenCV builds that don't accept
            # constructor parameters. OPENCV_FFMPEG_CAPTURE_OPTIONS above
            # still provides FFmpeg's RTSP timeout/transport options.
            log.warning(
                "OpenCV VideoCapture parameterized open is unavailable; "
                "using FFmpeg environment options fallback."
            )
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, OPENCV_BUFFER_SIZE)
        except cv2.error:
            pass

        if not cap.isOpened():
            self.stats.inc_open_failure()
            log.warning(
                f"Failed to open stream: {self.rtsp_url} "
                f"(open timeout={CAMERA_OPEN_TIMEOUT_MS} ms)"
            )
            cap.release()
            return None

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0

        log.info(
            f"Stream opened: {w}×{h} @ {fps:.1f} fps "
            f"(read timeout={CAMERA_READ_TIMEOUT_MS} ms)"
        )
        return cap

    def _push_frame(self, frame: np.ndarray):
        """Non-blocking latest-frame queue."""

        if not self.frame_queue.full():
            self.frame_queue.put_nowait(frame)
            return

        try:
            self.frame_queue.get_nowait()
            self.stats.inc_drop()
        except Exception:
            pass

        try:
            self.frame_queue.put_nowait(frame)
        except Full:
            pass

    def _run(self):
        reconnect_delay = max(0.5, float(CAMERA_RECONNECT_DELAY_S))

        while not self._stop_event.is_set():
            cap = self._open_stream()

            if cap is None:
                if self._stop_event.wait(reconnect_delay):
                    break

                reconnect_delay = min(reconnect_delay * 1.5, 10.0)
                self.stats.inc_reconnect()
                log.warning(
                    f"RTSP reconnecting in {reconnect_delay:.1f}s …"
                )
                continue

            # Successful open: reset backoff.
            reconnect_delay = max(0.5, float(CAMERA_RECONNECT_DELAY_S))
            log.info("Capture loop running …")

            while not self._stop_event.is_set():
                try:
                    ret, frame = cap.read()
                except cv2.error as exc:
                    self.stats.inc_read_failure()
                    log.warning(f"OpenCV cap.read() error: {exc}")
                    break

                if not ret or frame is None:
                    self.stats.inc_read_failure()
                    log.warning(
                        "cap.read() failed — RTSP stream may have stalled; "
                        "reconnecting."
                    )
                    break

                self.stats.inc_read()

                self._push_frame(frame)

            cap.release()
            log.info("VideoCapture released.")

            if self._stop_event.is_set():
                break

            self.stats.inc_reconnect()

            # Short, interruptible recovery delay. The FFmpeg read timeout
            # above bounds how long cap.read() can keep us here.
            if self._stop_event.wait(reconnect_delay):
                break

            reconnect_delay = min(reconnect_delay * 1.5, 10.0)
            log.warning(
                f"Capture recovery: retrying RTSP in {reconnect_delay:.1f}s …"
            )
