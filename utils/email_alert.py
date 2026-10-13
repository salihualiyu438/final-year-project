"""
Email Alert Utility — Synchronous (called from daemon thread)
─────────────────────────────────────────────────────────────
Attaches an annotated screenshot of the alert event to the email.
"""

import smtplib
import time
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import cv2
import numpy as np

# ── Credentials — fill these in ──────────────────────────────────────────────
EMAIL_SENDER      = "salihualiyu438@gmail.com"
EMAIL_PASSWORD    = "sfuf fnkz kjoj qktd"       # 16-char App Password
EMAIL_RECIPIENTS  = ["salihuabdulkareem353@gmail.com", "gomnajonah30@gmail.com"]
EMAIL_SMTP_HOST   = "smtp.gmail.com"
EMAIL_SMTP_PORT   = 587

# ── Cooldown ──────────────────────────────────────────────────────────────────
_last_email_time: float = 0.0
EMAIL_COOLDOWN_S: float = 60.0


def send_alert_email_sync(
    level: str,
    trigger: str,
    ts: float,
    screenshot: np.ndarray | None = None,
) -> None:
    """
    Send alert email with an annotated screenshot attached.
    Safe to call from any thread.
    """
    global _last_email_time

    now = time.time()
    if now - _last_email_time < EMAIL_COOLDOWN_S:
        return
    _last_email_time = now

    if not EMAIL_SENDER or not EMAIL_RECIPIENTS:
        return

    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    subject  = f"[SURVEILLANCE] {level} ALERT — {trigger}"

    html_body = f"""
    <html><body style="font-family:monospace;background:#0a0a0a;color:#eee;padding:24px;">
    <h2 style="color:#ff4444;">&#9888; {level} ALERT TRIGGERED</h2>
    <table style="border-collapse:collapse;width:100%;margin-bottom:20px;">
      <tr>
        <td style="padding:8px;color:#aaa;width:100px;">Trigger</td>
        <td style="padding:8px;color:#ff4444;font-weight:bold;">{trigger}</td>
      </tr>
      <tr>
        <td style="padding:8px;color:#aaa;">Level</td>
        <td style="padding:8px;">{level}</td>
      </tr>
      <tr>
        <td style="padding:8px;color:#aaa;">Time</td>
        <td style="padding:8px;">{time_str}</td>
      </tr>
    </table>
    {"<p style='color:#aaa;margin-bottom:8px;'>Screenshot of the event:</p><img src='cid:screenshot' style='max-width:100%;border:2px solid #ff4444;'/>" if screenshot is not None else ""}
    <p style="color:#555;margin-top:20px;font-size:11px;">
      Automated alert from your Surveillance System.
    </p>
    </body></html>
    """

    # Build message
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = ", ".join(EMAIL_RECIPIENTS)

    alt = MIMEMultipart("alternative")
    msg.attach(alt)
    alt.attach(MIMEText(html_body, "html"))

    # Attach screenshot if available
    if screenshot is not None:
        try:
            # Resize to max 1280px wide to keep email size reasonable
            h, w = screenshot.shape[:2]
            if w > 1280:
                scale = 1280 / w
                screenshot = cv2.resize(
                    screenshot,
                    (1280, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            ok, buf = cv2.imencode(
                ".jpg", screenshot,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            if ok:
                img_bytes = buf.tobytes()
                img_part  = MIMEImage(img_bytes, _subtype="jpeg")
                img_part.add_header("Content-ID", "<screenshot>")
                img_part.add_header(
                    "Content-Disposition", "inline",
                    filename=f"alert_{int(ts)}.jpg",
                )
                msg.attach(img_part)
        except Exception as e:
            print(f"[EMAIL] Screenshot attachment failed: {e}")

    # Send
    try:
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
        print(f"[EMAIL] Alert sent to {EMAIL_RECIPIENTS} with screenshot")
    except Exception as exc:
        import traceback
        print(f"[EMAIL] FAILED: {exc}")
        traceback.print_exc()
