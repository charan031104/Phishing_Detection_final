import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional

# Load environment variables from a .env file if present (non-fatal if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _build_message(to_email: str, url: str, is_phishing: bool) -> MIMEText:
    subject = "Phishing Alert" if is_phishing else "URL Safety Notification"
    verdict = "PHISHING" if is_phishing else "SAFE"
    body = (
        f"This is an automated notification about the URL you checked.\n\n"
        f"URL: {url}\n"
        f"Verdict: {verdict}\n\n"
        f"If you did not request this, you can ignore this message."
    )
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["To"] = to_email
    return msg


def send_alert_email(to_email: Optional[str], url: str, is_phishing: bool) -> None:
    """
    Send an alert email if SMTP environment variables are configured; otherwise log to console.

    Expected environment variables:
      - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
    If any are missing, the function will print a message and return without raising.
    """
    if not to_email:
        # No recipient provided; nothing to do
        return

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    # Normalize common Gmail app password format which may include spaces from UI
    if smtp_pass:
        smtp_pass = smtp_pass.replace(" ", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "noreply@example.com")

    if not smtp_host or not smtp_user or not smtp_pass:
        print(
            f"[email_sender] SMTP not configured; would have sent to {to_email}: URL={url}, phishing={is_phishing}"
        )
        return

    try:
        # Ensure is_phishing is a strict boolean (the caller might pass a str)
        is_phishing_bool = bool(is_phishing)
        msg = _build_message(to_email, url, is_phishing)
        msg["From"] = smtp_from
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_from, [to_email], msg.as_string())
        print(f"[email_sender] Alert email sent to {to_email}")
    except Exception as exc:
        print(f"[email_sender] Failed to send email to {to_email}: {exc}")
        # Do not raise, keep app resilient



