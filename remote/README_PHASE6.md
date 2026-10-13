# Phase 6 — Secure Remote Access

This phase keeps the AI/inference server on the surveillance PC and adds a secure
remote access path for the owner web application.

## Architecture

    Camera
       |
       v
    Surveillance PC
       |
       +-- Capture / YOLO / Evidence
       |
       +-- FastAPI on 127.0.0.1:8000
       |
       +-- cloudflared (outbound tunnel)
                 |
                 v
             Cloudflare
                 |
                 v
          Owner's browser

The FastAPI server is deliberately bound to `127.0.0.1` by default. It is not
directly exposed to the LAN or Internet. `cloudflared` connects to the local
service and publishes it through HTTPS.

## Phase 6A — Free temporary test

Cloudflare provides Quick Tunnels for development/testing. They generate a
temporary random `https://*.trycloudflare.com` URL.

1. Install `cloudflared` on Windows:
   https://developers.cloudflare.com/tunnel/downloads/

2. Confirm installation:

       cloudflared --version

3. Start the surveillance server:

       python server.py

4. In a SECOND PowerShell window:

       cloudflared tunnel --url http://127.0.0.1:8000

5. Cloudflared will print a URL such as:

       https://random-name.trycloudflare.com

6. Open that URL on the owner's phone.

7. Log in using the Phase 5 owner account.

### Important

Quick Tunnel URLs are temporary and change when the tunnel is restarted.
They are intended for testing/development, not permanent production use.

## Phase 6B — Stable remote URL

For a permanent owner URL, use a named Cloudflare Tunnel and a domain managed
through Cloudflare.

Example target:

       https://surveillance.example.com

The tunnel maps:

       surveillance.example.com
                  |
                  v
       http://127.0.0.1:8000

Cloudflare Tunnel is available on all Cloudflare plans, but publishing an
application through a stable custom hostname requires a domain on Cloudflare.
The domain itself is the part that may cost money.

Follow the official setup instructions:
https://developers.cloudflare.com/tunnel/setup/

## HTTPS session cookie

Remote access is HTTPS, so set:

PowerShell:

       $env:SURVEILLANCE_SECURE_COOKIE="1"

Also set a persistent session secret before remote use:

       $env:SURVEILLANCE_SESSION_SECRET="PUT_A_LONG_RANDOM_SECRET_HERE"

Change the owner/operator passwords:

       $env:SURVEILLANCE_OWNER_PASSWORD="NEW_OWNER_PASSWORD"
       $env:SURVEILLANCE_OPERATOR_PASSWORD="NEW_OPERATOR_PASSWORD"

These environment variables should be set before:

       python server.py

For local HTTP development, `SURVEILLANCE_SECURE_COOKIE` can remain `0`.

## Security model

Phase 6 does NOT expose the RTSP camera directly.

The public endpoint exposes only the FastAPI web application:

- `/login`
- `/owner`
- authenticated owner APIs
- authenticated live feed
- authenticated evidence
- authenticated WebSocket events

The RTSP URL remains inside the surveillance PC.

The owner never connects directly to the iPhone camera.

## Important operational limitation

Remote live video is currently the same MJPEG feed used by the local dashboard.
It is suitable for the prototype and demonstrations, but continuous remote video
uses significant bandwidth.

The eventual production architecture should preferentially send event evidence
and use a dedicated low-latency video transport (for example WebRTC) for remote
live viewing.

## Quick test checklist

[ ] `python server.py` starts normally
[ ] `http://127.0.0.1:8000/login` works locally
[ ] `cloudflared tunnel --url http://127.0.0.1:8000` starts
[ ] HTTPS URL opens from another network
[ ] Owner can log in
[ ] Operator cannot access the owner dashboard
[ ] Owner cannot access operator controls
[ ] Owner incident timeline works
[ ] Owner WebSocket events work
[ ] Owner live feed works

Do not expose the application permanently until the Phase 6 security settings
and Phase 5 credentials have been changed from their development values.
