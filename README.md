# SentinelAI — Web Dashboard

## Folder structure

```
surveillance_app/          ← your existing pipeline code (unchanged)
surveillance_ui/
  server.py                ← FastAPI server (entry point)
  config_additions.py      ← paste these lines into config/settings.py
  requirements_ui.txt      ← extra pip packages
  utils/
    email_alert.py         ← async email sender
  templates/
    dashboard.html         ← single-page dashboard
  static/
    sounds/
      alert_high.wav       ← HIGH alert beep
      alert_moderate.wav   ← MODERATE alert beep
```

## Setup

### 1. Install extra dependencies
```bash
pip install -r surveillance_ui/requirements_ui.txt
```

### 2. Add email settings to config/settings.py
Open `surveillance_app/config/settings.py` and paste the contents
of `surveillance_ui/config_additions.py` at the bottom.

For Gmail:
- Enable 2-step verification on your Google account
- Go to https://myaccount.google.com/apppasswords
- Create an App Password and paste it into EMAIL_PASSWORD

### 3. Place server.py next to your surveillance_app folder
```
your_project/
  surveillance_app/    ← existing
  surveillance_ui/     ← new
  server.py            ← or run from inside surveillance_ui/
```

### 4. Run
```bash
cd your_project
python surveillance_ui/server.py
```

Then open your browser at: http://localhost:8000

## Features

| Feature | Description |
|---|---|
| Live feed | MJPEG stream with bounding boxes drawn server-side |
| ROI editor | Click points on the video to draw a restricted zone |
| Alert banner | Flashing banner on HIGH / MODERATE alerts |
| Alert sound | Browser plays beep on every alert (no plugin needed) |
| Email alerts | HIGH alerts trigger an HTML email to all recipients |
| Alert log | Scrollable history of all alerts with timestamps |
| Class counters | Per-class detection bar chart, updates every frame |
| System stats | FPS, inference ms, total frames inferred |
| WebSocket | All real-time updates pushed over a single WS connection |

## Alert levels (from your existing pipeline)

| Level | Trigger |
|---|---|
| HIGH | `weapon` or `mask` detected anywhere in frame |
| MODERATE | `person`, `vehicle`, `animal` detected inside ROI zone |

## Notes

- The existing `pipelines/` code is **not modified** — the server just wraps it
- Email cooldown is 60 seconds (configurable in `utils/email_alert.py`)
- Sound requires the browser tab to be open; Chrome may block autoplay until
  the user first interacts with the page (click anywhere to unlock audio)


## Phase 2 — Incident Management

Phase 2 adds an in-memory `IncidentManager`. It turns repeated threat
assessments into one continuous incident instead of treating every inferred
frame or notification as a separate event.

- Creates a unique incident ID when a threat starts.
- Updates the active incident while the same threat continues.
- Closes an incident after 5 seconds without a matching threat.
- Keeps up to 200 closed incidents in memory.
- Publishes `incident_created`, `incident_updated`, and `incident_closed`
  events through the existing WebSocket event channel.
- Adds `incident_id` to alert records.
- Adds `GET /api/incidents` and `GET /api/incidents/active`.
- Does not yet write to PostgreSQL or record video. Those are later phases.

The existing RTSP capture and MJPEG live-feed pipeline is intentionally not
changed by Phase 2.

## Phase 3 — Local Evidence Capture

Phase 3 adds incident evidence without introducing cloud storage yet.

- Maintains a rolling 5-second pre-event buffer sampled at 10 FPS.
- Saves an annotated `snapshot.jpg` when an incident starts.
- Creates an `incident.mp4` containing pre-event footage, the active incident,
  and 5 seconds of post-event footage.
- Stores evidence under `media/incidents/<incident_id>/`.
- Serves local evidence through the `/media` route.
- Adds `snapshot_url` and `video_url` to incident API records and lifecycle
  WebSocket events.
- Finalizes open video writers cleanly during application shutdown.

Evidence settings are in `config/settings.py`:

```text
EVIDENCE_PRE_EVENT_S = 5.0
EVIDENCE_POST_EVENT_S = 5.0
EVIDENCE_FPS = 10.0
EVIDENCE_JPEG_QUALITY = 70
```

The evidence layer is deliberately local at this stage. Remote storage,
authentication, and mobile access will be introduced after the FastAPI and
database phases.


## Phase 6 — Secure Remote Access

Phase 6 keeps FastAPI bound to localhost by default and uses Cloudflare Tunnel
(`cloudflared`) for secure HTTPS remote access. The owner can therefore use the
separate owner web application from another network without exposing the RTSP
camera directly.

See `remote/README_PHASE6.md`.

For development, a free temporary Quick Tunnel can be started with:

    cloudflared tunnel --url http://127.0.0.1:8000

For permanent remote deployment, use a named Cloudflare Tunnel and a domain.


## Optional processing features

The operator dashboard now exposes independent runtime ON/OFF switches for:
- Motion / scene-change gating
- Temporal matching / multi-frame confirmation
- Context-aware false-positive filtering

These switches affect the processing pipeline immediately and do not require a server restart. Model 1 and Model 2 remain independent inference components. Model 2 is the separately trained weapon/threat detector with `large_gun`, `short_gun`, and `matchet/knife` classes. The three Model 2 threat classes are always alert-enabled.

Note: the uploaded application did not contain an implemented template-matching inference component, so no non-functional "template matching" switch was added.


## Model-specific inference input sizes

The application uses model-specific YOLO inference sizes matching the final
training datasets:

- Model 1: 640×640 (`MODEL1_INFERENCE_IMGSZ = 640`)
- Model 2: 1024×1024 (`MODEL2_INFERENCE_IMGSZ = 1024`)

The camera frame is passed directly to each model. Ultralytics performs the
aspect-ratio-preserving letterbox preprocessing internally for each model, so
the application does not perform a common 960×540 intermediate resize before
inference. This also keeps the returned bounding boxes aligned with the
original captured frame.

`INFERENCE_IMGSZ` remains as a backward-compatible alias for Model 1.


## Current class and preprocessing policy

- Model 1 classes: `person`, `animal`, `vehicle`, `personnel`.
- Model 2 classes: `large_gun`, `short_gun`, `matchet/knife`.
- Runtime class lists, temporal configuration, alert preferences, and dashboard labels use the same active class vocabulary.
- Capture frames remain at the source resolution. Model 1 uses `imgsz=640`; Model 2 uses `imgsz=1024`, and Ultralytics performs each model's internal letterboxing.
- Temporal matching is configured for all active detection classes: `person`, `animal`, `vehicle`, `personnel`, `large_gun`, `short_gun`, and `matchet/knife`. Person/animal/vehicle require two matches, personnel three, large_gun one, short_gun three, and matchet/knife three. A class requiring one match is confirmed on its first detection.
- Motion gating and temporal matching remain real algorithms, enabled by default and independently switchable through the feature flags.
- During configured critical hours (default 00:00–06:00 local time), a detected person/vehicle with no personnel present triggers a critical alert. Model 2 still runs when personnel are present so the threat engine can suppress a weapon alert when the weapon overlaps the security-personnel detection.


Model 2 resilience/performance: `MODEL2_FRAME_INTERVAL = 2` runs the 1024px weapon detector on every second eligible inference frame by default. Set it to `1` to run Model 2 on every eligible frame. If Model 2 cannot load or inference raises an exception (including runtime/memory errors), it is disabled for the current server run while Model 1 and the camera pipeline continue.
