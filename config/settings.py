"""
Central configuration for the surveillance system.
Edit this file to tune every pipeline parameter in one place.
"""

# ─── Camera ───────────────────────────────────────────────────────────────────
# Set this to the CURRENT RTSP address shown by IP Camera Lite.
# RTSP_URL = r"rtsp://admin:admin@192.168.0.176:554/12"
# RTSP_URL = r"C:\Users\User\Documents\final_year_project\final year project custom video dataset\security_personnels\22-05-2026\VID_20260522_133111.mp4"
# RTSP_URL = r"C:\Users\User\Documents\final_year_project\COMPRESSED_VIDEOS\20-08-2026_compressed_securities_videos\VID_20260501_140732_compressed.mp4"
RTSP_URL = r"C:\Users\User\Documents\final_year_project\final year project online videos\armed\youtube\Aggravated robbery at a convenience store at the 7900 block of Main. Houston PD _695232-23(480P).mp4"
# RTSP_URL = r"C:\Users\User\Documents\final_year_project\final year project online videos\armed\youtube\_5000 Reward___  Mt Juliet Sprint Store Robbery(1080P_HD).mp4"
# C:\Users\User\Documents\final_year_project\final year project online videos\armed\youtube\WATCH _ Guns blazing_ Armed gang rob motorist of R450 000 in attack on Gauteng_s N4(1080P_HD).mp4"
# a fixed number of failures.
CAMERA_RECONNECT_DELAY_S = 2.0

# FFmpeg/OpenCV RTSP timeouts. These prevent cap.open()/cap.read() from
# remaining blocked for ~30 seconds when the RTSP server stops responding.
CAMERA_OPEN_TIMEOUT_MS = 7000
CAMERA_READ_TIMEOUT_MS = 7000

# Kept for compatibility with code that may still import this setting.
CAMERA_RECONNECT_ATTEMPTS = 0   # 0 = retry indefinitely

# CAP_PROP_BUFFERSIZE: keep low (1–2) for live feeds to get latest frame
OPENCV_BUFFER_SIZE = 1

# ─── Capture Pipeline ─────────────────────────────────────────────────────────
# maxsize=1 → always keep the newest frame (no queue lag)
CAPTURE_QUEUE_MAXSIZE = 1

# Captured frames are kept at source resolution. Each YOLO model performs its
# own letterboxing internally at its configured inference size. No capture
# resizing is configured because it would alter the ROI coordinate system.

# ─── Inference Pipeline ───────────────────────────────────────────────────────
# MODEL_PATH = r"C:\Users\User\Documents\final_year_project\yolov8n_finetune\weights\best.pt"
MODEL_PATH = r"models\model1.pt"

# Optional Model 2: the separately trained weapon/threat detector.
# Expected Model 2 classes are: large_gun, short_gun, matchet/knife.
# The application treats Model 2 as an independent detector and continues
# running if its weight is unavailable. Put the trained Model 2 weight here.
MODEL2_PATH = r"models\model2.pt"

# Each model uses the input size corresponding to the final dataset/training
# configuration. Ultralytics performs aspect-ratio-preserving letterboxing
# internally for each model.
MODEL1_INFERENCE_IMGSZ = 640
MODEL2_INFERENCE_IMGSZ = 1024

MODEL2_INFERENCE_CONF = 0.60
# Run the expensive 1024px weapon model on every Nth eligible inference frame.
# 1 = every eligible frame; 2 = half as often, reducing latency/VRAM pressure.
MODEL2_FRAME_INTERVAL = 2
INFERENCE_CONF = 0.40
INFERENCE_IOU = 0.45

# Backward-compatible alias used by older components. Model 1 is the primary
# detector, so this remains 640 unless an older component imports it.
INFERENCE_IMGSZ = MODEL1_INFERENCE_IMGSZ
INFERENCE_HALF = False
RESULT_QUEUE_MAXSIZE = 2
INFERENCE_FRAME_SKIP = 1


# ─── Optional Processing Features ────────────────────────────────────────────
# Startup defaults. Set these to True/False to enable or disable the algorithms.
# The operator can also switch them at runtime from the dashboard.
# These flags disable the feature algorithm itself; they do NOT remove the
# underlying implementation.
FEATURE_MOTION_GATE_ENABLED = False
FEATURE_TEMPORAL_FILTER_ENABLED = True
FEATURE_CONTEXT_FILTER_ENABLED = True

# ─── Motion / Scene-Change Gating ────────────────────────────────────────────
# Capture never stops. YOLO runs only while meaningful motion/scene change is
# present, then remains active briefly so an entering object is not missed.
MOTION_GATE_ENABLED = False
MOTION_DOWNSCALE_WIDTH = 320
MOTION_PIXEL_THRESHOLD = 25
MOTION_MIN_AREA_RATIO = 0.003
MOTION_BLUR_SIZE = 5
MOTION_BACKGROUND_ALPHA = 0.05
MOTION_WARMUP_FRAMES = 15
MOTION_HOLD_SECONDS = 2.0
MOTION_FORCE_INFERENCE_SECONDS = 5.0

# Critical-hours policy uses the machine local time: [start, end).
CRITICAL_HOURS_START = 0  # 12:00 AM
CRITICAL_HOURS_END = 6    # 6:00 AM

# ─── Temporal Detection Confirmation ─────────────────────────────────────────
# A detection must match the same class in several recent inferred frames before
# it is released to the threat/incident engine.
TEMPORAL_FILTER_ENABLED = True
TEMPORAL_TARGET_CLASSES = {"person", "animal", "vehicle", "personnel", "large_gun", "short_gun", "matchet/knife"}
TEMPORAL_MATCH_IOU = 0.30
TEMPORAL_REQUIRED_MATCHES = {
    "person": 2,
    "animal": 2,
    "vehicle": 2,
    "personnel": 3,
    "large_gun": 1,
    "short_gun": 3,
    "matchet/knife": 3,
}
TEMPORAL_MAX_MISSES = 2
TEMPORAL_MAX_TRACKS_PER_CLASS = 20

# ─── Incident / Notification De-duplication ─────────────────────────────────
INCIDENT_ABSENCE_TIMEOUT_S = 8.0
ALERT_REPEAT_COOLDOWN_S = {
    "CRITICAL": 45.0,
    "HIGH": 30.0,
    "MEDIUM": 45.0,
    "LOW": 60.0,
}

# ─── Class Definitions ────────────────────────────────────────────────────────
# Model 1 (human/live-object detector) classes.
CLASS_NAMES = {
    0: "person",
    1: "animal",
    2: "vehicle",
    3: "personnel",
}

# Model 2 class IDs are mapped in server._run_model2():
#   0 = large_gun, 1 = short_gun, 2 = matchet/knife

HIGH_RISK_CLASSES = {"large_gun", "short_gun", "matchet/knife"}
ROI_SENSITIVE_CLASSES = {"person", "animal", "vehicle"}
NO_ALERT_CLASSES = {"personnel"}

# Operator-selectable non-critical alert classes. Model 2 weapon classes are
# always enabled and cannot be disabled from the dashboard.
ALERT_SELECTABLE_CLASSES = {"person", "animal", "vehicle"}
ALWAYS_ALERT_CLASSES = {"large_gun", "short_gun", "matchet/knife"}
DEFAULT_ALERT_CLASSES = set(ALERT_SELECTABLE_CLASSES)

# ─── Evidence Capture (Phase 3) ───────────────────────────────────────────────
EVIDENCE_ROOT_DIR = "media/incidents"
EVIDENCE_PRE_EVENT_S = 5.0
EVIDENCE_POST_EVENT_S = 5.0
EVIDENCE_FPS = 10.0
EVIDENCE_JPEG_QUALITY = 70

# ─── Alert Pipeline ───────────────────────────────────────────────────────────
ALERT_COOLDOWN_S = {
    "HIGH": 5.0,
    "MODERATE": 10.0,
}

# ─── ROI ──────────────────────────────────────────────────────────────────────
DEFAULT_ROI_POINTS = [
    [200, 200],
    [800, 200],
    [800, 600],
    [200, 600],
]

# ─── Display Pipeline ─────────────────────────────────────────────────────────
DISPLAY_WINDOW_NAME = "Surveillance System"
DISPLAY_FPS_SMOOTHING = 30
SHOW_FPS = True
SHOW_ROI = True
SHOW_DETECTIONS = True
ALERT_BANNER_DURATION_S = 3.0

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = "logs"
LOG_FILE = "logs/surveillance.log"
LOG_LEVEL = "INFO"
