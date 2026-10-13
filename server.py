"""
Surveillance Dashboard — FastAPI Server
────────────────────────────────────────
Architecture mirrors the proven Django exam-app pattern:

  Thread 1 — CapturePipeline  : reads RTSP frames (existing code, unchanged)
  Thread 2 — InferenceThread  : pulls frames, runs YOLO, encodes JPEG,
                                 checks alerts, sends email — all in one thread
  Main     — FastAPI          : serves MJPEG from _jpeg_queue, WebSocket
                                 events from _event_queue, static files
"""

import importlib.util
import os
import queue
import threading
import time
import json
import base64
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Web Push is optional at import time so the application can still start if the
# dependency has not yet been installed. The feature reports that state through
# the API instead of breaking the surveillance pipeline.
try:
    from pywebpush import webpush, WebPushException
    _WEBPUSH_AVAILABLE = True
except Exception:
    webpush = None
    WebPushException = Exception
    _WEBPUSH_AVAILABLE = False

from auth import create_session, get_session_from_cookie, verify_password, SESSION_COOKIE, SECURE_COOKIE

# ── Load email_alert explicitly by path ──────────────────────────────────────
_ea_path = Path(__file__).parent / "utils" / "email_alert.py"
_ea_spec  = importlib.util.spec_from_file_location("email_alert", _ea_path)
_ea_mod   = importlib.util.module_from_spec(_ea_spec)
_ea_spec.loader.exec_module(_ea_mod)
send_alert_email_sync = _ea_mod.send_alert_email_sync

# ── Pipeline imports ──────────────────────────────────────────────────────────
# import sys
# sys.path.insert(0, str(Path(__file__).parent.parent / "surveillance_app"))

from ultralytics import YOLO   # pre-import in main thread — avoids signal error

from config.settings import (
    CLASS_NAMES,
    INFERENCE_CONF,
    MODEL1_INFERENCE_IMGSZ,
    MODEL2_INFERENCE_IMGSZ,
    INFERENCE_IOU,
    INFERENCE_HALF,
    INFERENCE_FRAME_SKIP,
    MODEL_PATH,
    MODEL2_PATH,
    MODEL2_INFERENCE_CONF,
    MODEL2_FRAME_INTERVAL,
    CRITICAL_HOURS_START,
    CRITICAL_HOURS_END,
    RTSP_URL,
    ALERT_COOLDOWN_S,
    EVIDENCE_ROOT_DIR,
    EVIDENCE_PRE_EVENT_S,
    EVIDENCE_POST_EVENT_S,
    EVIDENCE_FPS,
    EVIDENCE_JPEG_QUALITY,
    MOTION_GATE_ENABLED,
    FEATURE_MOTION_GATE_ENABLED,
    FEATURE_TEMPORAL_FILTER_ENABLED,
    FEATURE_CONTEXT_FILTER_ENABLED,
    MOTION_DOWNSCALE_WIDTH,
    MOTION_PIXEL_THRESHOLD,
    MOTION_MIN_AREA_RATIO,
    MOTION_BLUR_SIZE,
    MOTION_BACKGROUND_ALPHA,
    MOTION_WARMUP_FRAMES,
    MOTION_HOLD_SECONDS,
    MOTION_FORCE_INFERENCE_SECONDS,
    TEMPORAL_FILTER_ENABLED,
    TEMPORAL_TARGET_CLASSES,
    TEMPORAL_MATCH_IOU,
    TEMPORAL_REQUIRED_MATCHES,
    TEMPORAL_MAX_MISSES,
    TEMPORAL_MAX_TRACKS_PER_CLASS,
    INCIDENT_ABSENCE_TIMEOUT_S,
    ALERT_REPEAT_COOLDOWN_S,
    ALERT_SELECTABLE_CLASSES,
    ALWAYS_ALERT_CLASSES,
    DEFAULT_ALERT_CLASSES,
)
from pipelines.capture import CapturePipeline
from services.threat_engine import ThreatEngine
from services.incident_manager import IncidentManager
from services.evidence_manager import EvidenceManager
from core.motion import MotionSceneGate
from core.temporal import TemporalMatcher

BASE_DIR = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════════════════════
# Shared queues & state
# ═══════════════════════════════════════════════════════════════════════════════

# Multi-camera runtime state. Every camera has an independent capture and
# inference pipeline, so one failed stream can reconnect without stopping the other.
_event_queue = queue.Queue(maxsize=200)
_camera_lock = threading.RLock()
_camera_configs: dict[str, dict] = {}
_jpeg_queues: dict[str, queue.Queue] = {}
_roi_polygons: dict[str, np.ndarray | None] = {}
_roi_lock = threading.RLock()
_threat_engines: dict[str, ThreatEngine] = {}
_incident_managers: dict[str, IncidentManager] = {}
_evidence_managers: dict[str, EvidenceManager] = {}

CAMERA_CONFIG_FILE = BASE_DIR / "camera_config.json"

def _default_camera_configs():
    return {
        "camera_1": {
            "id":"camera_1","name":"Camera 1","location":"Main Perimeter",
            "description":"Primary surveillance camera","url":RTSP_URL,"enabled":True
        },
        "camera_2": {
            "id":"camera_2","name":"Camera 2","location":"",
            "description":"Secondary surveillance camera","url":"","enabled":False
        }
    }

def _load_camera_configs():
    data = _default_camera_configs()
    try:
        if CAMERA_CONFIG_FILE.is_file():
            saved=json.loads(CAMERA_CONFIG_FILE.read_text(encoding="utf-8"))
            for cid in data:
                if isinstance(saved.get(cid),dict): data[cid].update(saved[cid])
    except Exception as exc:
        print(f"[CAMERA] Could not load camera_config.json: {exc}")
    return data

def _save_camera_configs():
    try:
        CAMERA_CONFIG_FILE.write_text(json.dumps(_camera_configs,indent=2),encoding="utf-8")
    except Exception as exc:
        print(f"[CAMERA] Could not save camera configuration: {exc}")

_camera_configs = _load_camera_configs()

_alert_history: deque = deque(maxlen=200)
_history_lock  = threading.Lock()

# Runtime operator alert policy. Critical classes are always included.
_alert_preferences = set(DEFAULT_ALERT_CLASSES) | set(ALWAYS_ALERT_CLASSES)
_alert_preferences_lock = threading.Lock()

# Optional processing features. These are runtime switches, intentionally
# independent from the alert-class preferences above.
_feature_flags = {
    "motion_gate": bool(FEATURE_MOTION_GATE_ENABLED),
    "temporal_filter": bool(FEATURE_TEMPORAL_FILTER_ENABLED),
    "context_filter": bool(FEATURE_CONTEXT_FILTER_ENABLED),
}
_feature_flags_lock = threading.RLock()

def _get_feature_flags() -> dict[str, bool]:
    with _feature_flags_lock:
        return dict(_feature_flags)

def _set_feature_flags(values: dict) -> dict[str, bool]:
    allowed = set(_feature_flags)
    with _feature_flags_lock:
        for key in allowed:
            if key in values:
                _feature_flags[key] = bool(values[key])
        return dict(_feature_flags)


_ws_clients: list[WebSocket] = []
_ws_lock     = threading.Lock()

# Native Web Push: no paid notification provider is required. Keep the VAPID
# private key and subscriptions on disk so phone registration survives restarts.
# Never place the generated data/ directory in a public repository or share it.
_PUSH_DATA_DIR = BASE_DIR / "data"
_PUSH_DATA_DIR.mkdir(parents=True, exist_ok=True)
_VAPID_PRIVATE_FILE = _PUSH_DATA_DIR / "vapid_private.pem"
_PUSH_SUBSCRIPTIONS_FILE = _PUSH_DATA_DIR / "push_subscriptions.json"
_push_subscriptions: dict[str, dict] = {}
_push_lock = threading.RLock()

_VAPID_PRIVATE_KEY = os.getenv("SURVEILLANCE_VAPID_PRIVATE_KEY", "").strip()
_VAPID_PUBLIC_KEY = os.getenv("SURVEILLANCE_VAPID_PUBLIC_KEY", "").strip()
_VAPID_CLAIMS_EMAIL = os.getenv(
    "SURVEILLANCE_VAPID_CLAIMS_EMAIL", "mailto:admin@example.com"
).strip()
_PUSH_DISABLED_REASON = ""


def _load_push_subscriptions():
    global _push_subscriptions
    try:
        if _PUSH_SUBSCRIPTIONS_FILE.exists():
            raw = json.loads(_PUSH_SUBSCRIPTIONS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _push_subscriptions = {
                    str(endpoint): value for endpoint, value in raw.items()
                    if isinstance(value, dict) and value.get("endpoint") == endpoint
                    and isinstance(value.get("keys"), dict)
                    and value["keys"].get("p256dh") and value["keys"].get("auth")
                }
        print(f"[PUSH] Loaded {len(_push_subscriptions)} saved phone subscription(s).")
    except Exception as exc:
        print(f"[PUSH] Could not load saved subscriptions: {exc}")
        _push_subscriptions = {}


def _save_push_subscriptions():
    temp_path = _PUSH_SUBSCRIPTIONS_FILE.with_suffix(".tmp")
    temp_path.write_text(json.dumps(_push_subscriptions, indent=2), encoding="utf-8")
    temp_path.replace(_PUSH_SUBSCRIPTIONS_FILE)


def _configure_vapid():
    global _VAPID_PRIVATE_KEY, _VAPID_PUBLIC_KEY, _PUSH_DISABLED_REASON
    if not _WEBPUSH_AVAILABLE:
        _PUSH_DISABLED_REASON = "Missing pywebpush dependency. Run: pip install -r requirements_ui.txt"
        print(f"[PUSH] {_PUSH_DISABLED_REASON}")
        return
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        # Explicit environment variables take precedence. Otherwise reuse a
        # persistent PEM key, generating it only once on first startup.
        if _VAPID_PRIVATE_KEY and _VAPID_PUBLIC_KEY:
            _PUSH_DISABLED_REASON = ""
            print("[PUSH] Using VAPID credentials from environment.")
            return
        if _VAPID_PRIVATE_KEY or _VAPID_PUBLIC_KEY:
            raise ValueError("Set both SURVEILLANCE_VAPID_PRIVATE_KEY and SURVEILLANCE_VAPID_PUBLIC_KEY, or set neither.")

        if _VAPID_PRIVATE_FILE.exists():
            private_obj = serialization.load_pem_private_key(
                _VAPID_PRIVATE_FILE.read_bytes(), password=None
            )
            if not isinstance(private_obj, ec.EllipticCurvePrivateKey):
                raise ValueError("Saved VAPID key is not an elliptic-curve private key.")
            print("[PUSH] Loaded persistent VAPID key from data/vapid_private.pem.")
        else:
            private_obj = ec.generate_private_key(ec.SECP256R1())
            _VAPID_PRIVATE_FILE.write_bytes(private_obj.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ))
            try:
                os.chmod(_VAPID_PRIVATE_FILE, 0o600)
            except OSError:
                pass
            print("[PUSH] Generated persistent VAPID key at data/vapid_private.pem.")

        public_bytes = private_obj.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )
        _VAPID_PUBLIC_KEY = base64.urlsafe_b64encode(public_bytes).decode("ascii").rstrip("=")
        # pywebpush accepts a path to the PEM private-key file.
        _VAPID_PRIVATE_KEY = str(_VAPID_PRIVATE_FILE)
        _PUSH_DISABLED_REASON = ""
    except Exception as exc:
        _PUSH_DISABLED_REASON = f"Could not configure VAPID credentials: {exc}"
        _VAPID_PRIVATE_KEY = ""
        _VAPID_PUBLIC_KEY = ""
        print(f"[PUSH] {_PUSH_DISABLED_REASON}")


_load_push_subscriptions()
_configure_vapid()


def _send_owner_push_alert(level: str, trigger: str, incident_id: str | None = None):
    """Send a Web Push alert; delivery failures must never stop inference."""
    if not _WEBPUSH_AVAILABLE or not _VAPID_PRIVATE_KEY or not _VAPID_PUBLIC_KEY:
        print(f"[PUSH] Not sent: Web Push is unavailable. {_PUSH_DISABLED_REASON}")
        return
    with _push_lock:
        subscriptions = list(_push_subscriptions.values())
    if not subscriptions:
        print("[PUSH] No registered owner phones; skipping push notification.")
        return

    payload = json.dumps({
        "title": f"{level} surveillance alert",
        "body": str(trigger),
        "level": str(level),
        "incident_id": incident_id,
        "url": "/owner",
        "requireInteraction": level == "CRITICAL",
    })
    expired = []
    delivered = 0
    for subscription in subscriptions:
        try:
            webpush(
                subscription_info=subscription,
                data=payload,
                vapid_private_key=_VAPID_PRIVATE_KEY,
                vapid_claims={"sub": _VAPID_CLAIMS_EMAIL},
            )
            delivered += 1
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            print(f"[PUSH] Delivery failed (HTTP {status or 'unknown'}): {exc}")
            if status in (404, 410):
                expired.append(subscription.get("endpoint", ""))
    if expired:
        with _push_lock:
            for endpoint in expired:
                _push_subscriptions.pop(endpoint, None)
            try:
                _save_push_subscriptions()
            except Exception as exc:
                print(f"[PUSH] Could not save expired-subscription cleanup: {exc}")
    print(f"[PUSH] Alert delivery attempted for {len(subscriptions)} phone(s); accepted by push service: {delivered}.")

# ═══════════════════════════════════════════════════════════════════════════════
# Colours (BGR for OpenCV)
# ═══════════════════════════════════════════════════════════════════════════════

COLORS_BGR = {
    "person":    (0, 165, 255),
    "large_gun": (30, 30, 255),
    "short_gun": (50, 50, 255),
    "matchet/knife":   (80, 80, 255),
    "animal":    (100, 210, 50),
    "vehicle":   (220, 180, 50),
    "personnel": (50, 255, 50),
    "default":   (180, 180, 180),
}
FONT         = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS    = 2
JPEG_QUALITY = 65

# ── Contextual filtering thresholds ──────────────────────────────────────────
# Minimum IoU for a threat box to be associated with a person, vehicle, or
# security-personnel box before it can be considered an actionable threat.
WEAPON_OVERLAP_MIN_IOU = 0.10


# ═══════════════════════════════════════════════════════════════════════════════
# Bounding box helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(boxA: list, boxB: list) -> float:
    """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union


# ═══════════════════════════════════════════════════════════════════════════════
# Contextual detection filter
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_detections(detections: list) -> list:
    """Apply optional context filtering to Model 2 threat detections.

    Threat detections are retained only when they overlap a detected person,
    vehicle, or security-personnel box. Including personnel as an anchor is
    intentional: the threat engine must be able to recognize an authorized
    security officer carrying equipment and suppress the alert appropriately.
    Model 1 detections themselves are never discarded by this filter.
    """
    anchors = [d for d in detections
               if d.class_name in {"person", "vehicle", "personnel"}]
    weapons = [d for d in detections
               if d.class_name in {"large_gun", "short_gun", "matchet/knife"}]
    other_detections = [d for d in detections
                        if d.class_name not in {"large_gun", "short_gun", "matchet/knife"}]

    valid_weapons = []
    for weapon in weapons:
        if any(_iou(weapon.bbox_xyxy, anchor.bbox_xyxy) >= WEAPON_OVERLAP_MIN_IOU
               for anchor in anchors):
            valid_weapons.append(weapon)
    return other_detections + valid_weapons


# ═══════════════════════════════════════════════════════════════════════════════
# Annotation helper
# ═══════════════════════════════════════════════════════════════════════════════

def _run_model2(model, frame: np.ndarray) -> list:
    """Run Model 2 (the separately trained weapon/threat detector) on the same frame.

    Model 2 is expected to provide the trained threat classes:
      0 -> large_gun
      1 -> short_gun
      2 -> matchet/knife

    The primary model's detections are not modified. Model 2 detections are
    appended to the aggregated detection list so the existing downstream
    temporal/threat/annotation pipeline receives one combined set.
    """
    from pipelines.inference import Detection

    results = model.predict(
        source=frame,
        conf=MODEL2_INFERENCE_CONF,
        iou=INFERENCE_IOU,
        imgsz=MODEL2_INFERENCE_IMGSZ,
        half=INFERENCE_HALF,
        verbose=False,
    )

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            # Model 2 is the user's separately trained weapon/threat model.
            # Keep its semantic classes distinct from Model 1.
            name_map = {
                0: "large_gun",
                1: "short_gun",
                2: "matchet/knife",
            }
            if cls_id not in name_map:
                continue
            name = name_map[cls_id]
            detections.append(
                Detection(
                    class_id=100 + cls_id,
                    class_name=name,
                    confidence=float(box.conf[0]),
                    bbox_xyxy=box.xyxy[0].tolist(),
                )
            )
    return detections


def _annotate(frame: np.ndarray, detections: list, roi: np.ndarray | None) -> np.ndarray:
    canvas = frame.copy()

    if roi is not None and len(roi) >= 3:
        pts     = roi.reshape((-1, 1, 2))
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [roi], (0, 255, 255))
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)
        cv2.polylines(canvas, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
        tx = int(roi[:, 0].min())
        ty = max(int(roi[:, 1].min()) - 8, 15)
        cv2.putText(canvas, "RESTRICTED ZONE", (tx, ty),
                    FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Operator-facing visualization deliberately shows only authorized
    # security personnel. Other detections remain available internally for
    # temporal matching and threat analysis.
    for det in detections:
        if det.class_name != "personnel":
            continue
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = COLORS_BGR.get(det.class_name, COLORS_BGR["default"])
        label = f"{det.class_name} {det.confidence:.0%}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, THICKNESS)
        bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(canvas, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 4),
                    FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, det.center, 4, color, -1)

    return canvas


def _annotate_alert_snapshot(frame: np.ndarray, detections: list,
                             roi: np.ndarray | None) -> np.ndarray:
    """Create the incident snapshot shown to the owner.

    The live operator feed intentionally remains personnel-only. Incident
    snapshots are separate: they show the detections that actually triggered
    the alert so the owner can see the reason for the event.
    """
    canvas = frame.copy()

    if roi is not None and len(roi) >= 3:
        pts = roi.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = COLORS_BGR.get(det.class_name, COLORS_BGR["default"])
        label = f"{det.class_name} {det.confidence:.0%}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, THICKNESS)
        bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(canvas, (x1, bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 4),
                    FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, det.center, 4, color, -1)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# Motion gating and temporal confirmation
# ═══════════════════════════════════════════════════════════════════════════════

# Feature algorithms live in dedicated core components. These aliases preserve
# compatibility with the existing runtime while keeping the implementation out
# of server.py.
MotionGate = MotionSceneGate
TemporalDetectionFilter = TemporalMatcher


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Thread
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceThread(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, camera_id: str, camera_info: dict):
        super().__init__(daemon=True, name=f"InferenceThread-{camera_id}")
        self.frame_queue    = frame_queue
        self.camera_id      = camera_id
        self.camera_info    = dict(camera_info)
        self._stop_event    = threading.Event()
        self._frame_count   = 0
        self._last_alert    = {"HIGH": 0.0, "MODERATE": 0.0}
        self.total_inferred = 0
        self.total_ms       = 0.0
        self._stats_lock    = threading.Lock()
        self._motion_gate    = MotionGate()
        self._temporal_filter = TemporalDetectionFilter()
        self._motion_skipped  = 0

    def stop(self):
        self._stop_event.set()

    def avg_ms(self) -> float:
        with self._stats_lock:
            if self.total_inferred == 0:
                return 0.0
            return self.total_ms / self.total_inferred

    def run(self):
        print(
            f"[InferenceThread] Loading Model 1: {MODEL_PATH} "
            f"(imgsz={MODEL1_INFERENCE_IMGSZ})"
        )
        model = YOLO(MODEL_PATH)
        print("[InferenceThread] Model 1 ready.")

        # Model 2 is optional and loads in its own daemon thread. This prevents
        # loading the 1024px model from holding up the live dashboard startup.
        model2_path = Path(MODEL2_PATH)
        if not model2_path.is_absolute():
            model2_path = BASE_DIR / model2_path
        model2_state = {"model": None, "loading": False, "failed": False}
        model2_state_lock = threading.Lock()

        def _load_model2_background():
            with model2_state_lock:
                model2_state["loading"] = True
            try:
                print(
                    f"[InferenceThread] Loading optional Model 2: {model2_path} "
                    f"(imgsz={MODEL2_INFERENCE_IMGSZ})"
                )
                loaded_model = YOLO(str(model2_path))
                with model2_state_lock:
                    model2_state["model"] = loaded_model
                print("[InferenceThread] Model 2 ready.")
            except Exception as exc:
                with model2_state_lock:
                    model2_state["failed"] = True
                print(
                    f"[InferenceThread] Model 2 could not be loaded: {exc}. "
                    "Continuing with Model 1 only."
                )
            finally:
                with model2_state_lock:
                    model2_state["loading"] = False

        if model2_path.exists():
            threading.Thread(
                target=_load_model2_background,
                name=f"Model2Loader-{self.camera_id}",
                daemon=True,
            ).start()
        else:
            with model2_state_lock:
                model2_state["failed"] = True
            print(
                f"[InferenceThread] Model 2 not found: {model2_path} "
                "(continuing with Model 1 only)"
            )

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        frame_times   = deque(maxlen=30)

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._frame_count += 1

            # Motion/scene-change gate: capture remains continuous, while the
            # algorithm itself can be switched ON/OFF at runtime.
            self._motion_gate.set_enabled(_get_feature_flags()["motion_gate"])
            should_infer, motion_ratio = self._motion_gate.check(frame)
            if not should_infer:
                self._motion_skipped += 1
                _evidence_managers[self.camera_id].observe_frame(frame, time.time())
                ok, buf = cv2.imencode(".jpg", frame, encode_params)
                if ok:
                    if _jpeg_queues[self.camera_id].full():
                        try:
                            _jpeg_queues[self.camera_id].get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        _jpeg_queues[self.camera_id].put_nowait(buf.tobytes())
                    except queue.Full:
                        pass
                continue

            if INFERENCE_FRAME_SKIP > 1 and (self._frame_count % INFERENCE_FRAME_SKIP != 0):
                continue

            # Use the captured frame directly. Each YOLO model performs its
            # own aspect-ratio-preserving letterbox preprocessing internally:
            # Model 1 -> 640x640, Model 2 -> 1024x1024.
            # Capture keeps the source resolution and preserves bounding-box
            # coordinates for ROI and display alignment.
            inference_frame = frame

            # Stage 1: run Model 1 on the native captured frame. Ultralytics
            # letterboxes internally at 640x640; captured coordinates stay native.
            t0 = time.perf_counter()
            try:
                results = model.predict(
                    source=inference_frame,
                    conf=INFERENCE_CONF,
                    iou=INFERENCE_IOU,
                    imgsz=MODEL1_INFERENCE_IMGSZ,
                    half=INFERENCE_HALF,
                    verbose=False,
                )
            except Exception as exc:
                print(f"[InferenceThread] Model 1 predict error: {exc}")
                continue

            from pipelines.inference import Detection
            raw_detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    det = Detection(
                        class_id=cls_id,
                        class_name=CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                        confidence=float(box.conf[0]),
                        bbox_xyxy=box.xyxy[0].tolist(),
                    )
                    raw_detections.append(det)

            # Stage 2: Model 2 is only needed when Model 1 sees a person or
            # vehicle and has NOT classified anyone in the frame as personnel.
            # During 00:00–06:00, presence itself is an alert condition, so skip
            # the more expensive weapon model for that frame.
            from datetime import datetime
            now_local = datetime.now().astimezone()
            hour = now_local.hour
            critical_time = CRITICAL_HOURS_START <= hour < CRITICAL_HOURS_END
            has_personnel = any(d.class_name == "personnel" for d in raw_detections)
            has_person_or_vehicle = any(
                d.class_name in {"person", "vehicle"} for d in raw_detections
            )
            # Run Model 2 for any relevant live-object/personnel scene. At
            # critical hours, skip it only for an un-authorized person/vehicle
            # scene because the time rule itself already requires an alert.
            # If personnel are present, still run Model 2 so weapon detections
            # can be spatially associated and suppressed when carried by staff.
            model2_detections = []
            with model2_state_lock:
                model2 = model2_state["model"]
            should_run_model2 = has_person_or_vehicle or has_personnel
            critical_presence_alert = critical_time and has_person_or_vehicle and not has_personnel
            model2_interval = max(1, int(MODEL2_FRAME_INTERVAL))
            run_model2_this_frame = (self._frame_count % model2_interval == 0)
            if (model2 is not None and should_run_model2 and not critical_presence_alert
                    and run_model2_this_frame):
                try:
                    model2_detections = _run_model2(model2, inference_frame)
                except Exception as exc:
                    # A Model 2 runtime/OOM error must not kill the shared
                    # inference thread and freeze the dashboard feed. Disable
                    # the optional model for this process; Model 1 continues.
                    print(
                        f"[InferenceThread] Model 2 inference failed: {exc}. "
                        "Disabling Model 2 for this run; Model 1 will continue."
                    )
                    with model2_state_lock:
                        model2_state["model"] = None
                        model2_state["failed"] = True
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    model2_detections = []

            raw_detections.extend(model2_detections)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            with self._stats_lock:
                self.total_inferred += 1
                self.total_ms += elapsed_ms
            if model2_detections:
                for _d2 in model2_detections:
                    print(f"[MODEL2] {_d2.class_name} score={_d2.confidence:.2f}")

            # ── Apply context-aware filters ───────────────────────────────
            if _get_feature_flags()["context_filter"]:
                detections = _filter_detections(raw_detections)
            else:
                detections = raw_detections
            # Temporal confirmation is also a real algorithm, not a placeholder.
            # Its runtime status is independently controlled by the operator.
            self._temporal_filter.set_enabled(_get_feature_flags()["temporal_filter"])
            detections = self._temporal_filter.update(detections)

            # Get current ROI
            with _roi_lock:
                _roi = _roi_polygons.get(self.camera_id)
                roi = _roi.copy() if _roi is not None else None

            # Annotate once. The same annotated frame is used for the browser
            # feed, alert screenshots, and incident evidence.
            canvas = _annotate(frame, detections, roi)

            # Feed the evidence rolling buffer before incident processing so
            # the current frame is available as part of the pre-event context.
            _evidence_managers[self.camera_id].observe_frame(frame, time.time())

            # Check alerts (on filtered detections only)
            self._check_alerts(detections, roi, frame, canvas, raw_detections)
            ok, buf = cv2.imencode(".jpg", canvas, encode_params)
            if not ok:
                continue
            jpeg = buf.tobytes()

            # Push to MJPEG queue — drop oldest if full
            if _jpeg_queues[self.camera_id].full():
                try:
                    _jpeg_queues[self.camera_id].get_nowait()
                except queue.Empty:
                    pass
            try:
                _jpeg_queues[self.camera_id].put_nowait(jpeg)
            except queue.Full:
                pass

            # FPS tracking
            now = time.time()
            frame_times.append(now)
            fps = 0.0
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0

            # Broadcast stats
            counts = {name: 0 for name in (*CLASS_NAMES.values(), "large_gun", "short_gun", "matchet/knife")}
            for det in detections:
                if det.class_name in counts:
                    counts[det.class_name] += 1
            try:
                _event_queue.put_nowait({
                    "type":            "stats",
                    "fps":             round(fps, 1),
                    "inf_ms":          round(elapsed_ms, 1),
                    "total_inferred":  self.total_inferred,
                    "class_counts":    counts,
                    "detection_count": len(detections),
                })
            except queue.Full:
                pass

    def _check_alerts(self, detections: list, roi: np.ndarray | None, frame: np.ndarray, canvas: np.ndarray, raw_detections: list | None = None):
        # Threat analysis runs on every inferred frame.  Incident management is
        # independent of notification cooldowns so one continuous threat can
        # remain one incident instead of becoming repeated incidents.
        _threat_engines[self.camera_id].update_roi(roi)
        with _alert_preferences_lock:
            enabled_alert_classes = set(_alert_preferences)
        _threat_engines[self.camera_id].update_alert_classes(enabled_alert_classes)
        assessment = _threat_engines[self.camera_id].assess(
            detections, authorization_detections=raw_detections or detections
        )

        # Keep the live/operator canvas unchanged (personnel-only boxes).
        # Build a separate owner-facing snapshot only from detections that
        # actually triggered the alert.
        alert_snapshot = _annotate_alert_snapshot(
            frame, assessment.trigger_detections, roi
        )

        event_ts = time.time()
        incident_events = _incident_managers[self.camera_id].process(
            assessment,
            frame_id=self._frame_count,
            timestamp=event_ts,
        )

        # Phase 3: create/close local evidence around incident lifecycle events.
        for incident_event in incident_events:
            media = _evidence_managers[self.camera_id].handle_event(incident_event, alert_snapshot)
            if media:
                _incident_managers[self.camera_id].attach_media(
                    incident_event.incident.incident_id,
                    snapshot_url=media.get("snapshot_url"),
                    video_url=media.get("video_url"),
                )
                # Keep the lifecycle event sent over WebSocket consistent with
                # the API response by attaching the same media URLs to the
                # event's incident copy.
                incident_event.incident.snapshot_url = media.get("snapshot_url")
                incident_event.incident.video_url = media.get("video_url")

        # Publish incident lifecycle events to the existing WebSocket channel.
        for incident_event in incident_events:
            payload = {
                "type": incident_event.type,
                "incident": incident_event.incident.to_dict(),
            }
            try:
                _event_queue.put_nowait(payload)
            except queue.Full:
                pass

        if not assessment.is_alert:
            return

        now = time.time()
        level = assessment.level
        cooldown = ALERT_REPEAT_COOLDOWN_S.get(
            level,
            ALERT_COOLDOWN_S.get(level, ALERT_COOLDOWN_S.get("HIGH", 30.0)),
        )
        if now - self._last_alert.get(level, 0.0) < cooldown:
            return

        self._last_alert[level] = now
        names = sorted({d.class_name for d in assessment.trigger_detections})
        camera_label = self.camera_info.get("name") or self.camera_id
        camera_location = self.camera_info.get("location") or "Location not assigned"
        trigger = (
            f"Camera={camera_label} | Location={camera_location} | "
            f"{', '.join(names) or 'Threat'} | "
            f"score={assessment.score} | "
            f"{' ; '.join(assessment.reasons)}"
        )

        # Email screenshot also uses the alert-specific annotated image.
        # The browser live feed continues to use the personnel-only canvas.
        screenshot = alert_snapshot
        active = _incident_managers[self.camera_id].active()
        self._fire_alert(
            level,
            trigger,
            screenshot,
            threat_score=assessment.score,
            threat_reasons=assessment.reasons,
            detections=assessment.trigger_detections,
            in_roi=assessment.in_roi,
            incident_id=active.incident_id if active else None,
        )

    def _fire_alert(self, level: str, trigger: str, screenshot: np.ndarray,
                    threat_score: int = 0, threat_reasons: list[str] | None = None,
                    detections: list | None = None, in_roi: bool = False,
                    incident_id: str | None = None):
        ts  = time.time()
        rec = {
            "type":    "alert",
            "incident_id": incident_id,
            "level":   level,
            "trigger": trigger,
            "time":    time.strftime("%H:%M:%S", time.localtime(ts)),
            "ts":      ts,
            "threat_score": threat_score,
            "threat_reasons": threat_reasons or [],
            "detections": [
                {"class": d.class_name, "confidence": round(d.confidence, 4),
                 "bbox": [round(v, 2) for v in d.bbox_xyxy]}
                for d in (detections or [])
            ],
            "in_roi": in_roi,
            "camera_id": self.camera_id,
            "camera_name": self.camera_info.get("name", self.camera_id),
            "camera_location": self.camera_info.get("location", ""),
            "camera_description": self.camera_info.get("description", ""),
        }
        with _history_lock:
            _alert_history.appendleft(rec)
        try:
            _event_queue.put_nowait(rec)
        except queue.Full:
            pass

        print(f"[ALERT:{level}] {trigger}")

        # Email with screenshot — daemon thread so inference isn't blocked
        if level in {"HIGH", "CRITICAL"}:
            threading.Thread(
                target=send_alert_email_sync,
                args=(level, trigger, ts, screenshot),
                daemon=True,
            ).start()

        # Native web-push notification — independent of the email channel.
        # It is also daemonized so a slow/unavailable push service can never
        # stall the inference loop.
        if level in {"HIGH", "CRITICAL"}:
            threading.Thread(
                target=_send_owner_push_alert,
                args=(level, trigger, incident_id),
                daemon=True,
                name="OwnerPushAlert",
            ).start()


# ═══════════════════════════════════════════════════════════════════════════════
# Global camera runtime handles
# ═══════════════════════════════════════════════════════════════════════════════

_cap_pipelines: dict[str, CapturePipeline] = {}
_inf_threads: dict[str, InferenceThread] = {}
_cap_pipeline: CapturePipeline | None = None
_inf_thread: InferenceThread | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cap_pipeline, _inf_thread

    for camera_id, info in list(_camera_configs.items()):
        if not info.get("enabled") or not str(info.get("url","")).strip():
            continue

        _jpeg_queues[camera_id] = queue.Queue(maxsize=1)
        _roi_polygons[camera_id] = None
        _threat_engines[camera_id] = ThreatEngine()
        _incident_managers[camera_id] = IncidentManager(
            absence_timeout_s=INCIDENT_ABSENCE_TIMEOUT_S,
            history_size=200,
            camera_id=camera_id,
            camera_name=info.get("name",camera_id),
            camera_location=info.get("location",""),
            camera_description=info.get("description",""),
        )
        _evidence_managers[camera_id] = EvidenceManager(
            root_dir=BASE_DIR / EVIDENCE_ROOT_DIR,
            pre_event_s=EVIDENCE_PRE_EVENT_S,
            post_event_s=EVIDENCE_POST_EVENT_S,
            fps=EVIDENCE_FPS,
            jpeg_quality=EVIDENCE_JPEG_QUALITY,
        )

        cap=CapturePipeline(info["url"])
        inf=InferenceThread(cap.frame_queue,camera_id,info)
        _cap_pipelines[camera_id]=cap
        _inf_threads[camera_id]=inf
        cap.start()
        inf.start()
        print(f"[CAMERA] Started {camera_id}: {info.get('name')}")

    _cap_pipeline=_cap_pipelines.get("camera_1")
    _inf_thread=_inf_threads.get("camera_1")

    import asyncio
    asyncio.create_task(_ws_broadcast_loop())
    yield

    for cid,cap in list(_cap_pipelines.items()):
        try: cap.stop()
        except Exception as exc: print(f"[CAMERA] stop {cid}: {exc}")
    for cid,inf in list(_inf_threads.items()):
        try:
            inf.stop(); inf.join(timeout=5)
        except Exception as exc: print(f"[CAMERA] inference stop {cid}: {exc}")
    for manager in list(_evidence_managers.values()):
        try: manager.finalize_all()
        except Exception as exc: print(f"[EVIDENCE] finalize: {exc}")


app = FastAPI(title="Surveillance Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


async def _ws_broadcast_loop():
    import asyncio
    while True:
        try:
            event = _event_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.02)
            continue

        dead = []
        with _ws_lock:
            clients = list(_ws_clients)
        for ws in clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        with _ws_lock:
            for ws in dead:
                if ws in _ws_clients:
                    _ws_clients.remove(ws)


def _current_user(request: Request):
    return get_session_from_cookie(request.cookies.get(SESSION_COOKIE))


def _require_role(request: Request, role: str):
    user = _current_user(request)
    if not user or user.get("role") != role:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _require_any_role(request: Request):
    user = _current_user(request)
    if not user or user.get("role") not in {"operator", "owner"}:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


@app.get("/sw.js", include_in_schema=False)
def service_worker_script():
    """Serve the service worker at the origin root so it can control /owner.

    A script under /static/ cannot request scope / unless the server explicitly
    grants that scope with Service-Worker-Allowed. Serving it at /sw.js avoids
    the scope mismatch and keeps the service worker scoped to this application.
    """
    return FileResponse(
        BASE_DIR / "static" / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@app.get("/healthz")
async def healthz():
    """Minimal health endpoint for tunnel/deployment checks; no authentication required."""
    return {"status": "ok", "service": "surveillance"}


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse((BASE_DIR / "templates" / "login.html").read_text(encoding="utf-8"))


@app.post("/api/auth/login")
async def login(payload: dict):
    username = str(payload.get("username", "")).strip().lower()
    password = str(payload.get("password", ""))
    user = verify_password(username, password)
    if not user:
        return JSONResponse({"ok": False, "error": "Invalid username or password"}, status_code=401)
    token = create_session(user)
    response = JSONResponse({"ok": True, "user": user})
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, samesite="strict",
        secure=SECURE_COOKIE, max_age=28800, path="/"
    )
    return response


@app.post("/api/auth/logout")
async def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/api/auth/me")
async def auth_me(request: Request):
    user = _current_user(request)
    if not user:
        return JSONResponse({"authenticated": False}, status_code=401)
    return {"authenticated": True, "user": user}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not _current_user(request) or _current_user(request).get("role") != "operator":
        return RedirectResponse(url="/login")
    # Server/operator interface. Kept separate from the owner interface.
    return HTMLResponse(
        (BASE_DIR / "templates" / "dashboard.html").read_text(encoding="utf-8")
    )


@app.get("/owner", response_class=HTMLResponse)
async def owner_dashboard(request: Request):
    if not _current_user(request) or _current_user(request).get("role") != "owner":
        return RedirectResponse(url="/login")
    """Owner-facing dashboard. No operator controls are exposed here."""
    return HTMLResponse(
        (BASE_DIR / "templates" / "owner.html").read_text(encoding="utf-8")
    )


def _mjpeg_generator(camera_id: str):
    q=_jpeg_queues.get(camera_id)
    if q is None:
        frame=np.zeros((360,640,3),dtype=np.uint8)
        cv2.putText(frame,f"{camera_id}: NO SIGNAL",(150,185),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
        ok,enc=cv2.imencode(".jpg",frame)
        data=enc.tobytes() if ok else b""
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+data+b"\r\n"
            time.sleep(1)
    while True:
        try: jpeg=q.get(timeout=2.0)
        except queue.Empty: continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg+b"\r\n"


@app.get("/media/{file_path:path}")
def media_file(request: Request, file_path: str):
    _require_any_role(request)

    # Evidence is physically stored under:
    #   media/incidents/<incident_id>/<filename>
    #
    # Some existing incident records already use URLs beginning with
    # "/media/incidents/...". Since EVIDENCE_ROOT_DIR is itself
    # "media/incidents", remove the duplicated leading "incidents"
    # component before resolving the file. This keeps old and new media
    # URLs compatible.
    root = (BASE_DIR / EVIDENCE_ROOT_DIR).resolve()

    relative = Path(file_path)
    if relative.parts and relative.parts[0] == "incidents":
        relative = Path(*relative.parts[1:])

    target = (root / relative).resolve()

    if target != root and root not in target.parents:
        raise HTTPException(status_code=404, detail="Evidence not found")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Evidence not found")

    return FileResponse(target)


@app.get("/video_feed")
def video_feed(request: Request):
    _require_any_role(request)
    camera_id=request.query_params.get("camera_id","camera_1")
    if camera_id not in _camera_configs:
        raise HTTPException(status_code=404,detail="Unknown camera")
    return StreamingResponse(_mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control":"no-cache, no-store, must-revalidate","Pragma":"no-cache"})


@app.websocket("/ws/events")
async def websocket_events(ws: WebSocket):
    user = get_session_from_cookie(ws.cookies.get(SESSION_COOKIE))
    if not user or user.get("role") not in {"operator", "owner"}:
        await ws.close(code=1008, reason="Authentication required")
        return
    await ws.accept()
    with _ws_lock:
        _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


@app.get("/api/cameras")
async def get_cameras(request: Request):
    _require_any_role(request)
    out=[]
    for cid,cfg in _camera_configs.items():
        item=dict(cfg)
        cap=_cap_pipelines.get(cid)
        item["running"]=bool(cap and cap.is_alive())
        item["capture_stats"]=cap.stats.snapshot() if cap else None
        out.append(item)
    return out


@app.post("/api/cameras")
async def update_cameras(request: Request, payload: dict):
    _require_role(request,"operator")
    cameras=payload.get("cameras")
    if not isinstance(cameras,list):
        raise HTTPException(status_code=400,detail="cameras must be a list")
    new=_default_camera_configs()
    for raw in cameras:
        if not isinstance(raw,dict): continue
        cid=str(raw.get("id","")).strip()
        if cid not in new: continue
        c=new[cid]
        c["enabled"]=bool(raw.get("enabled",False))
        c["name"]=str(raw.get("name",c["name"])).strip()[:80] or c["name"]
        c["url"]=str(raw.get("url","")).strip()
        c["location"]=str(raw.get("location","")).strip()[:160]
        c["description"]=str(raw.get("description","")).strip()[:300]
        new[cid]=c
    with _camera_lock:
        _camera_configs.clear(); _camera_configs.update(new); _save_camera_configs()
    return {"ok":True,"cameras":list(_camera_configs.values()),"restart_required":True,
            "message":"Camera settings saved. Restart the server to apply stream changes."}


@app.get("/api/feature_flags")
async def get_feature_flags(request: Request):
    _require_role(request, "operator")
    return {
        "features": _get_feature_flags(),
        "available": {
            "motion_gate": "Motion / scene-change inference gating",
            "temporal_filter": "Temporal matching / multi-frame confirmation",
            "context_filter": "Context-aware false-positive filtering",
        },
    }


@app.post("/api/feature_flags")
async def set_feature_flags(request: Request, payload: dict):
    _require_role(request, "operator")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise HTTPException(status_code=400, detail="features must be an object")

    updated = _set_feature_flags(features)
    return {
        "ok": True,
        "features": updated,
        "message": "Optional processing feature settings applied.",
    }


@app.get("/api/alert_preferences")
async def get_alert_preferences(request: Request):
    _require_role(request, "operator")
    with _alert_preferences_lock:
        selected = sorted(_alert_preferences)
    return {
        "selected": selected,
        "selectable": sorted(ALERT_SELECTABLE_CLASSES),
        "always_alert": sorted(ALWAYS_ALERT_CLASSES),
    }


@app.post("/api/alert_preferences")
async def set_alert_preferences(request: Request, payload: dict):
    _require_role(request, "operator")
    requested = payload.get("selected", [])
    if not isinstance(requested, list):
        raise HTTPException(status_code=400, detail="selected must be a list")

    selected = {str(x).strip().lower() for x in requested}
    selected &= set(ALERT_SELECTABLE_CLASSES)
    # Critical classes are forcibly included and cannot be disabled.
    selected |= set(ALWAYS_ALERT_CLASSES)

    with _alert_preferences_lock:
        _alert_preferences.clear()
        _alert_preferences.update(selected)

    return {
        "ok": True,
        "selected": sorted(selected),
        "always_alert": sorted(ALWAYS_ALERT_CLASSES),
    }


@app.get("/api/owner/push/vapid-public-key")
async def owner_push_vapid_key(request: Request):
    _require_role(request, "owner")
    enabled = bool(_WEBPUSH_AVAILABLE and _VAPID_PUBLIC_KEY and _VAPID_PRIVATE_KEY)
    with _push_lock:
        subscription_count = len(_push_subscriptions)
    return {
        "enabled": enabled,
        "public_key": _VAPID_PUBLIC_KEY if enabled else "",
        "reason": _PUSH_DISABLED_REASON if not enabled else "",
        "registered_devices": subscription_count,
    }


@app.post("/api/owner/push/subscribe")
async def owner_push_subscribe(request: Request, payload: dict):
    user = _require_role(request, "owner")
    subscription = payload.get("subscription")
    if not isinstance(subscription, dict):
        raise HTTPException(status_code=400, detail="Invalid push subscription")
    endpoint = str(subscription.get("endpoint", "")).strip()
    keys = subscription.get("keys")
    if not endpoint or not isinstance(keys, dict) or not keys.get("p256dh") or not keys.get("auth"):
        raise HTTPException(status_code=400, detail="Incomplete push subscription")

    with _push_lock:
        _push_subscriptions[endpoint] = {
            "endpoint": endpoint,
            "keys": {
                "p256dh": str(keys["p256dh"]),
                "auth": str(keys["auth"]),
            },
        }
        try:
            _save_push_subscriptions()
        except Exception as exc:
            _push_subscriptions.pop(endpoint, None)
            print(f"[PUSH] Failed to persist phone subscription: {exc}")
            raise HTTPException(status_code=500, detail="Could not save phone registration on the server")
    print(f"[PUSH] Owner phone registered. Total registered devices: {len(_push_subscriptions)}")
    return {"ok": True, "subscribed": True, "user": user["username"]}


@app.post("/api/owner/push/test")
async def owner_push_test(request: Request):
    _require_role(request, "owner")
    if not (_WEBPUSH_AVAILABLE and _VAPID_PRIVATE_KEY and _VAPID_PUBLIC_KEY):
        raise HTTPException(status_code=503, detail=_PUSH_DISABLED_REASON or "Web Push is not configured")
    with _push_lock:
        if not _push_subscriptions:
            raise HTTPException(status_code=400, detail="No phone is registered. Enable phone alerts on your phone first.")
    threading.Thread(
        target=_send_owner_push_alert,
        args=("HIGH", "Test notification — your phone is connected to the surveillance alert system.", None),
        daemon=True,
        name="OwnerPushTest",
    ).start()
    return {"ok": True, "message": "Test push queued. Check the registered phone."}


@app.post("/api/owner/push/unsubscribe")
async def owner_push_unsubscribe(request: Request, payload: dict):
    _require_role(request, "owner")
    endpoint = str(payload.get("endpoint", "")).strip()
    if endpoint:
        with _push_lock:
            _push_subscriptions.pop(endpoint, None)
            try:
                _save_push_subscriptions()
            except Exception as exc:
                print(f"[PUSH] Failed to persist phone unregistration: {exc}")
                raise HTTPException(status_code=500, detail="Could not save phone unregistration")
    return {"ok": True}


@app.post("/api/roi")
async def update_roi(request: Request, payload: dict):
    _require_role(request,"operator")
    cid=str(payload.get("camera_id","camera_1"))
    pts=payload.get("points",[])
    if cid not in _camera_configs: raise HTTPException(status_code=404,detail="Unknown camera")
    if len(pts)>=3:
        with _roi_lock:
            _roi_polygons[cid]=np.array([[p["x"],p["y"]] for p in pts],dtype=np.int32)
    return {"status":"ok","points":len(pts),"camera_id":cid}


@app.post("/api/reset_roi")
async def reset_roi(request: Request, payload: dict | None = None):
    _require_role(request,"operator")
    cid=str((payload or {}).get("camera_id","camera_1"))
    if cid not in _camera_configs: raise HTTPException(status_code=404,detail="Unknown camera")
    with _roi_lock: _roi_polygons[cid]=None
    return {"status":"cleared","camera_id":cid}


@app.get("/api/stats")
async def get_stats(request: Request):
    _require_role(request,"operator")
    cameras=[]
    for cid,inf in _inf_threads.items():
        cfg=_camera_configs.get(cid,{})
        cap=_cap_pipelines.get(cid)
        cameras.append({"camera_id":cid,"camera_name":cfg.get("name",cid),
                        "camera_location":cfg.get("location",""),
                        "running":bool(cap and cap.is_alive()),
                        "inf_ms":round(inf.avg_ms(),1),
                        "total_inferred":inf.total_inferred,
                        "capture":cap.stats.snapshot() if cap else None})
    p=cameras[0] if cameras else {}
    return {"fps":0,"inf_ms":p.get("inf_ms",0),"total_inferred":p.get("total_inferred",0),
            "class_counts":{name:0 for name in (*CLASS_NAMES.values(), "large_gun", "short_gun", "matchet/knife")},
            "model":str(MODEL_PATH),"stream":RTSP_URL,
            "conf_threshold":INFERENCE_CONF,
            "imgsz":MODEL1_INFERENCE_IMGSZ,
            "model1_imgsz":MODEL1_INFERENCE_IMGSZ,
            "model2":str((BASE_DIR / MODEL2_PATH) if not Path(MODEL2_PATH).is_absolute() else Path(MODEL2_PATH)),
            "model2_imgsz":MODEL2_INFERENCE_IMGSZ,
            "cameras":cameras,
            "features":_get_feature_flags()}


@app.get("/api/alert_history")
async def get_alert_history(request: Request):
    _require_role(request, "operator")
    with _history_lock:
        return list(_alert_history)


@app.get("/api/incidents")
async def get_incidents(request: Request):
    _require_role(request,"operator")
    items=[]
    for manager in _incident_managers.values():
        items.extend(manager.history())
        active=manager.active()
        if active: items.append(active)
    items.sort(key=lambda x:x.started_at,reverse=True)
    return [x.to_dict() for x in items]


@app.get("/api/incidents/active")
async def get_active_incident(request: Request):
    _require_role(request,"operator")
    return [x.to_dict() for x in [m.active() for m in _incident_managers.values()] if x]


@app.get("/api/owner/summary")
async def get_owner_summary(request: Request):
    _require_role(request,"owner")
    active=[]; history=[]
    for manager in _incident_managers.values():
        a=manager.active()
        if a: active.append(a)
        history.extend(manager.history())
    history.sort(key=lambda x:x.started_at,reverse=True)
    return {
        "system":"ONLINE" if _cap_pipelines else "OFFLINE",
        "active_incident":active[0].to_dict() if active else None,
        "active_count":len(active),
        "recent_counts":{
            "CRITICAL":sum(x.level=="CRITICAL" for x in history),
            "HIGH":sum(x.level=="HIGH" for x in history),
            "MEDIUM":sum(x.level=="MEDIUM" for x in history)},
        "recent_incidents":[x.to_dict() for x in history[:8]],
        "cameras":list(_camera_configs.values())
    }


@app.get("/api/owner/incidents")
async def get_owner_incidents(request: Request, limit: int = 50):
    _require_role(request,"owner")
    items=[]
    for manager in _incident_managers.values():
        items.extend(manager.history())
        active=manager.active()
        if active: items.append(active)
    items.sort(key=lambda x:x.started_at,reverse=True)
    return [x.to_dict() for x in items[:max(1,min(limit,200))]]



# ──────────────────────────────────────────────────────────────────────────────
# Run the application directly with: python server.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
