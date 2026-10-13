"""
Inference Pipeline
──────────────────
Thread-2: Pulls frames from the capture queue, runs YOLOv8 inference,
and pushes structured result objects to the result queue.

Design decisions:
  • Instantiates the YOLO model INSIDE the thread to ensure thread-safety
    (Ultralytics docs: "always instantiate within the thread that will use them").
  • model.fuse() folds Conv+BN layers → measurable latency reduction (Medium, 2025).
  • Supports FP16 half-precision (INFERENCE_HALF=True) for ~2× speedup on GPU.
  • Frame-skip (INFERENCE_FRAME_SKIP) lets the capture thread run at camera FPS
    while inference runs at a lower rate — prevents GPU saturation on slower HW.
  • Result dataclass keeps the alert pipeline decoupled from raw YOLO internals.
"""

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue

import numpy as np

from config.settings import (
    CLASS_NAMES,
    INFERENCE_CONF,
    INFERENCE_FRAME_SKIP,
    INFERENCE_HALF,
    INFERENCE_IOU,
    INFERENCE_IMGSZ,
    MODEL_PATH,
    RESULT_QUEUE_MAXSIZE,
)
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Result dataclass ─────────────────────────────────────────────────────────

from core.detection import Detection


@dataclass
class InferenceResult:
    """Everything the downstream pipelines need for one frame."""
    frame:       np.ndarray          # annotated (or raw) frame
    detections:  list[Detection]
    frame_id:    int
    timestamp:   float               # time.time() at inference
    inference_ms: float              # wall-clock inference duration


# ─── Inference stats ──────────────────────────────────────────────────────────

class InferenceStats:
    def __init__(self):
        self._lock         = threading.Lock()
        self.frames_inferred = 0
        self.frames_skipped  = 0
        self.total_ms        = 0.0

    def record(self, ms: float):
        with self._lock:
            self.frames_inferred += 1
            self.total_ms += ms

    def inc_skip(self):
        with self._lock:
            self.frames_skipped += 1

    def avg_ms(self) -> float:
        with self._lock:
            if self.frames_inferred == 0:
                return 0.0
            return self.total_ms / self.frames_inferred


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Consumes frames from `frame_queue`, produces InferenceResult objects
    into `result_queue`.

    Usage:
        pipeline = InferencePipeline(frame_queue)
        pipeline.start()
        result: InferenceResult = pipeline.result_queue.get()
        pipeline.stop()
    """

    def __init__(self, frame_queue: Queue):
        self.frame_queue  = frame_queue
        self.result_queue = Queue(maxsize=RESULT_QUEUE_MAXSIZE)
        self.stats        = InferenceStats()
        self._stop_event  = threading.Event()
        self._frame_count = 0
        self._thread      = threading.Thread(
            target=self._run, name="InferencePipeline", daemon=True
        )

    # ─── Public API ───────────────────────────────────────────────────────────
    def start(self) -> "InferencePipeline":
        log.info(f"Starting inference pipeline  model={MODEL_PATH}  conf={INFERENCE_CONF}")
        self._thread.start()
        return self

    def stop(self):
        log.info("Stopping inference pipeline …")
        self._stop_event.set()
        self._thread.join(timeout=5)
        log.info("Inference pipeline stopped.")

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ─── Internal ─────────────────────────────────────────────────────────────
    def _load_model(self):
        """Load and optimise model inside the inference thread."""
        from ultralytics import YOLO   # deferred import keeps startup fast
        log.info("Loading YOLO model …")
        model = YOLO(MODEL_PATH)
        # model.fuse()                   # fuse Conv+BN for speed
        log.info(
            f"Model ready  |  classes={len(model.names)}  "
            f"half={'yes' if INFERENCE_HALF else 'no'}"
        )
        return model

    def _parse_results(self, yolo_results, frame: np.ndarray) -> list[Detection]:
        """Convert raw ultralytics Results → list of Detection objects."""
        detections: list[Detection] = []
        for r in yolo_results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                xyxy    = box.xyxy[0].tolist()
                det = Detection(
                    class_id   = cls_id,
                    class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    confidence = conf,
                    bbox_xyxy  = xyxy,
                )
                detections.append(det)
        return detections

    def _run(self):
        model = self._load_model()

        while not self._stop_event.is_set():
            # ── Grab latest frame (block up to 1 s) ──────────────────────────
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                continue

            self._frame_count += 1

            # ── Frame-skip logic ─────────────────────────────────────────────
            if INFERENCE_FRAME_SKIP > 1 and (self._frame_count % INFERENCE_FRAME_SKIP != 0):
                self.stats.inc_skip()
                continue

            # ── Run inference ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                yolo_results = model.predict(
                    source  = frame,
                    conf    = INFERENCE_CONF,
                    iou     = INFERENCE_IOU,
                    imgsz   = INFERENCE_IMGSZ,
                    half    = INFERENCE_HALF,
                    verbose = False,
                )
            except Exception as exc:
                log.error(f"Inference error: {exc}")
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.stats.record(elapsed_ms)

            # ── Parse detections ──────────────────────────────────────────────
            detections = self._parse_results(yolo_results, frame)

            # ── Package result ────────────────────────────────────────────────
            result = InferenceResult(
                frame        = frame,
                detections   = detections,
                frame_id     = self._frame_count,
                timestamp    = time.time(),
                inference_ms = elapsed_ms,
            )

            # Non-blocking put: if result queue is full, discard oldest result
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except Empty:
                    pass
            try:
                self.result_queue.put_nowait(result)
            except Exception:
                pass

            log.debug(
                f"Frame {self._frame_count} | "
                f"{len(detections)} detections | "
                f"{elapsed_ms:.1f} ms"
            )
