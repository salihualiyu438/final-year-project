"""
Phase 3 evidence capture.

The inference thread never performs JPEG encoding, video encoding, or disk I/O.
It only makes a bounded copy of frames and submits them to this background
worker. Evidence capture therefore cannot block the live-feed pipeline.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class _Frame:
    timestamp: float
    jpeg: bytes


@dataclass
class _Recording:
    incident_id: str
    started_at: float
    frames: list[_Frame]


class EvidenceManager:
    def __init__(
        self,
        root_dir,
        pre_event_s: float = 5.0,
        post_event_s: float = 5.0,
        fps: float = 10.0,
        jpeg_quality: int = 70,
    ):
        self.root_dir = os.fspath(root_dir)
        self.pre_event_s = max(0.0, float(pre_event_s))
        self.post_event_s = max(0.0, float(post_event_s))
        self.fps = max(1.0, float(fps))
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))

        os.makedirs(self.root_dir, exist_ok=True)

        self._buffer: deque[_Frame] = deque()
        self._active: dict[str, _Recording] = {}

        self._buffer_lock = threading.Lock()
        self._active_lock = threading.Lock()

        # Small bounded queue: if evidence falls behind, frames are dropped
        # rather than allowing evidence work to affect surveillance.
        self._work: queue.Queue = queue.Queue(maxsize=20)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="EvidenceWriter",
            daemon=True,
        )
        self._worker.start()

        self._last_submitted = 0.0
        self._submit_interval = 1.0 / self.fps

    # ------------------------------------------------------------------
    # Public API used by server.py
    # ------------------------------------------------------------------

    def observe_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Submit a sampled raw frame without doing encoding in the caller.

        This is deliberately non-blocking. A surveillance frame is copied only
        when the evidence worker queue has capacity.
        """
        if frame is None:
            return

        now = time.time() if timestamp is None else float(timestamp)

        if now - self._last_submitted < self._submit_interval:
            return

        self._last_submitted = now

        try:
            # The copy is bounded to one frame and prevents the capture/inference
            # pipeline from being affected by later frame mutation.
            item = ("frame", now, frame.copy())
            self._work.put_nowait(item)
        except queue.Full:
            # Evidence is best-effort; live surveillance wins.
            pass

    def handle_event(self, incident_event, annotated_frame: Optional[np.ndarray]):
        """Handle an IncidentEvent and return media URLs known at this point."""
        incident = incident_event.incident
        incident_id = incident.incident_id
        event_type = incident_event.type

        if event_type == "incident_created":
            frame_copy = annotated_frame.copy() if annotated_frame is not None else None
            self._enqueue(
                ("start", incident_id, incident.started_at, frame_copy)
            )
            return {
                "snapshot_url": f"/media/incidents/{incident_id}/snapshot.jpg",
                "video_url": f"/media/incidents/{incident_id}/incident.mp4",
            }

        if event_type == "incident_updated":
            self._enqueue(("touch", incident_id, incident.last_seen_at, None))
            return None

        if event_type == "incident_closed":
            self._enqueue(("finish", incident_id, incident.ended_at or incident.last_seen_at, None))
            # The final MP4 is produced after post_event_s, but its URL is stable.
            return {
                "video_url": f"/media/incidents/{incident_id}/incident.mp4",
            }

        return None

    def stop(self) -> None:
        self._stop.set()
        try:
            self._work.put_nowait(("stop", None, None))
        except queue.Full:
            pass
        self._worker.join(timeout=3.0)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _enqueue(self, item) -> None:
        try:
            self._work.put_nowait(item)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        pending_finish: dict[str, float] = {}

        while not self._stop.is_set():
            try:
                item = self._work.get(timeout=0.05)
            except queue.Empty:
                item = None

            if item is not None:
                kind = item[0]

                if kind == "stop":
                    break

                if kind == "frame":
                    _, timestamp, frame = item
                    self._encode_and_store_frame(timestamp, frame)

                elif kind == "start":
                    _, incident_id, timestamp, snapshot = item
                    self._start_recording(incident_id, timestamp, snapshot)

                elif kind == "touch":
                    _, incident_id, timestamp, _ = item
                    # Timestamp is represented implicitly by the presence of
                    # subsequent frames; no disk work is needed here.

                elif kind == "finish":
                    _, incident_id, timestamp, _ = item
                    pending_finish[incident_id] = float(timestamp)

            now = time.time()

            # Finalize only after post-event footage has had time to accumulate.
            for incident_id, end_time in list(pending_finish.items()):
                if now - end_time < self.post_event_s:
                    continue

                with self._active_lock:
                    recording = self._active.pop(incident_id, None)

                if recording is not None:
                    self._write_video(recording)

                pending_finish.pop(incident_id, None)

    def _encode_and_store_frame(self, timestamp: float, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return

        item = _Frame(timestamp, encoded.tobytes())

        with self._buffer_lock:
            self._buffer.append(item)
            cutoff = timestamp - self.pre_event_s
            while self._buffer and self._buffer[0].timestamp < cutoff:
                self._buffer.popleft()

        with self._active_lock:
            for recording in self._active.values():
                recording.frames.append(item)

    def _start_recording(
        self,
        incident_id: str,
        timestamp: float,
        snapshot: Optional[np.ndarray],
    ) -> None:
        with self._buffer_lock:
            pre_frames = list(self._buffer)

        with self._active_lock:
            if incident_id in self._active:
                return
            self._active[incident_id] = _Recording(
                incident_id=incident_id,
                started_at=timestamp,
                frames=pre_frames,
            )

        # Snapshot is disk I/O, so it also stays in the evidence worker.
        if snapshot is not None:
            path = os.path.join(
                self._incident_dir(incident_id),
                "snapshot.jpg",
            )
            cv2.imwrite(
                path,
                snapshot,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def _incident_dir(self, incident_id: str) -> str:
        path = os.path.join(self.root_dir, incident_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _write_video(self, recording: _Recording) -> None:
        if not recording.frames:
            return

        frames = list(recording.frames)

        first_image = cv2.imdecode(
            np.frombuffer(frames[0].jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if first_image is None:
            return

        h, w = first_image.shape[:2]
        incident_dir = self._incident_dir(recording.incident_id)
        output = os.path.join(incident_dir, "incident.mp4")
        temporary = os.path.join(incident_dir, "incident_source.mp4")

        # OpenCV's mp4v output is easy to create on Windows, but many browsers
        # do not reliably play MPEG-4 Part 2 (mp4v). First create a temporary
        # video, then convert it to browser-compatible H.264/yuv420p using
        # FFmpeg when it is available.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temporary, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            return

        try:
            for item in frames:
                image = cv2.imdecode(
                    np.frombuffer(item.jpeg, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is not None:
                    writer.write(image)
        finally:
            writer.release()

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            command = [
                ffmpeg,
                "-y",
                "-i", temporary,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output,
            ]
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0 and os.path.isfile(output):
                    try:
                        os.remove(temporary)
                    except OSError:
                        pass
                    print(
                        f"[EVIDENCE] H.264 video created for "
                        f"{recording.incident_id}"
                    )
                    return
                print(
                    f"[EVIDENCE] FFmpeg conversion failed for "
                    f"{recording.incident_id}; keeping source video."
                )
            except Exception as exc:
                print(
                    f"[EVIDENCE] FFmpeg conversion error for "
                    f"{recording.incident_id}: {exc}"
                )

        # Fallback: preserve evidence even if FFmpeg is unavailable.
        # The resulting mp4v file may not play in all browsers, but it remains
        # available for download and offline playback.
        try:
            os.replace(temporary, output)
        except OSError:
            pass
