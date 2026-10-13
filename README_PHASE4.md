# Phase 4 — Separate Owner Web Application

This phase preserves the existing Phase 3 operator/server dashboard at `/` and adds a separate owner-facing dashboard at `/owner`.

## Owner features
- Mobile-responsive owner dashboard
- Current system status
- Active threat banner
- Recent severity counts
- Incident timeline/list
- Incident details
- Snapshot and evidence-video viewing when available
- Live camera view
- WebSocket-driven incident notifications in the browser

## New endpoints
- `GET /owner`
- `GET /api/owner/summary`
- `GET /api/owner/incidents?limit=50`

## Important
This phase is intentionally **not yet remote/publicly exposed** and does not add authentication. Those belong to later phases. Keep the surveillance PC/operator UI private while development continues.

## Run
```powershell
python server.py
```
Then open:
- Operator UI: `http://127.0.0.1:8000/`
- Owner UI: `http://127.0.0.1:8000/owner`

## Phase 5 — Authentication and role-based access control

Phase 5 adds separate authenticated roles:

- `operator` → `/` server/operator dashboard and operator APIs.
- `owner` → `/owner` owner dashboard and owner APIs.

The backend enforces these roles; hiding controls in the browser is not used as the security mechanism.

### Development accounts

Default development credentials are:

- Operator: `operator` / `operator123`
- Owner: `owner` / `owner123`

**Change these before any remote deployment.** Set:

- `SURVEILLANCE_OPERATOR_PASSWORD`
- `SURVEILLANCE_OWNER_PASSWORD`
- `SURVEILLANCE_SESSION_SECRET`

The implementation uses only Python's standard library for password verification/session signing, so no authentication SaaS or subscription is required.


## Phase 6 — Secure Remote Access

Phase 6 keeps FastAPI bound to localhost by default and uses Cloudflare Tunnel
(`cloudflared`) for secure HTTPS remote access. The owner can therefore use the
separate owner web application from another network without exposing the RTSP
camera directly.

See `remote/README_PHASE6.md`.

For development, a free temporary Quick Tunnel can be started with:

    cloudflared tunnel --url http://127.0.0.1:8000

For permanent remote deployment, use a named Cloudflare Tunnel and a domain.
