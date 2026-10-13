# Phone threat alerts (free Web Push)

This application uses native browser Web Push. It does not require an SMS gateway,
Firebase billing, OneSignal, or another paid push-notification subscription.

## 1. Install the UI dependencies

Activate the same Python virtual environment used to run `server.py`, then run
from this application's folder:

```powershell
python -m pip install -r requirements_ui.txt
```

The required packages include `pywebpush` and `cryptography`. Restart `server.py`
after installation. At startup, look for `[PUSH] Generated persistent VAPID key`
or `[PUSH] Loaded persistent VAPID key`. If `pywebpush` is missing, the server
logs the exact installation command.

## 2. Register your phone

1. Open the owner dashboard and sign in as the owner.
2. Open the dashboard using a supported browser on the phone.
3. Click **Enable phone alerts** and allow notifications when prompted.
4. Keep the `data/vapid_private.pem` and `data/push_subscriptions.json` files in
   place. They are generated/updated automatically and preserve registration
   across server restarts.

## 3. HTTPS requirement

Web Push requires a secure context. `http://localhost:8000` is suitable for a
same-computer development test, but a phone accessing the server by a LAN IP over
plain HTTP generally cannot subscribe. For phone use, expose the dashboard through
HTTPS (for example, a domain with a TLS certificate or a trusted HTTPS tunnel).
Do not expose the owner dashboard publicly without appropriate authentication and
network protection.

## 4. What happens on detection

The existing alert path sends Web Push for `HIGH` and `CRITICAL` alerts. Push is
sent asynchronously so a slow notification service does not block camera inference.
The browser service worker displays the notification and opens the owner dashboard
when tapped. Notifications can still be restricted by browser/OS settings, battery
optimization, or connectivity.

## Important security note

`data/vapid_private.pem` is a private signing key. Do not share it or commit it to a
public repository. The `.gitignore` excludes the key and subscription database.
Back up both privately if you need to preserve phone registration across machine
migration. If the VAPID key is deleted or replaced, devices may need to be
registered again.
