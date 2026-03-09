"""
Alert & Notification System
=============================
Handles alert generation and multi-channel notifications for detected risks.

Channels:
  1. Email alerts (SMTP — configurable)
  2. SMS alerts (Twilio simulation)
  3. Dashboard warnings (returned as structured data)
  4. Emergency services concept (demonstrated)
  5. Console/log alerts

Risk routing:
  Low Risk    → No alert
  Medium Risk → Counselor / moderator notification
  High Risk   → Emergency alert to all channels + authorities
"""

import smtplib
import datetime
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Configuration (replace with real credentials in production)
# ──────────────────────────────────────────────────────────────
ALERT_CONFIG = {
    "email": {
        "smtp_host":   "smtp.gmail.com",
        "smtp_port":   587,
        "sender":      "sentinel.ai.alerts@yourdomain.com",
        "password":    "YOUR_APP_PASSWORD",          # Use env var in production
        "high_risk":   ["crisis-team@hospital.org", "emergency@support.org"],
        "medium_risk": ["counselor@support.org",    "moderator@platform.org"],
    },
    "sms": {
        "account_sid": "YOUR_TWILIO_ACCOUNT_SID",
        "auth_token":  "YOUR_TWILIO_AUTH_TOKEN",
        "from_number": "+18005555555",
        "high_risk":   ["+1-800-555-0001", "+1-800-555-0002"],
        "medium_risk": ["+1-800-555-0003"],
    },
    "emergency": {
        "police_api":     "https://api.emergency-services.gov/alert",
        "crisis_hotline": "988",                    # US Suicide & Crisis Lifeline
        "api_key":        "YOUR_EMERGENCY_API_KEY",
    },
}

CATEGORY_ICONS = {
    "Normal Conversation":        "💬",
    "Stress / Anxiety":           "😰",
    "Depression / Sadness":       "😔",
    "Suicide Risk":               "🆘",
    "Cyberbullying":              "⚡",
    "Violence / Threats":         "🔴",
    "Trust / Relationship Issues":"💔",
}


@dataclass
class AlertRecord:
    """Structured alert record for logging and display."""
    timestamp: str
    risk_level: str
    category: str
    confidence: float
    text_snippet: str
    channels_notified: list = field(default_factory=list)
    alert_id: str = ""
    message: str = ""
    status: str = "sent"

    def to_dict(self) -> dict:
        return self.__dict__


class AlertSystem:
    """
    Multi-channel alert notification system.
    
    Usage:
        alert_system = AlertSystem(log_dir="outputs/alerts")
        record = alert_system.process_alert(
            text="I don't want to live anymore",
            category="Suicide Risk",
            risk_level="high",
            confidence=0.94,
        )
    """

    def __init__(self, log_dir: str = "outputs/alerts", simulation_mode: bool = True):
        """
        Args:
            log_dir: Directory to persist alert logs
            simulation_mode: If True, print alerts instead of sending real emails/SMS
        """
        self.log_dir = log_dir
        self.simulation_mode = simulation_mode
        self.alert_log = []
        os.makedirs(log_dir, exist_ok=True)

    # ── Main Entry Point ────────────────────────────────────────
    def process_alert(
        self,
        text: str,
        category: str,
        risk_level: str,
        confidence: float,
        user_id: str = "anonymous",
    ) -> AlertRecord:
        """
        Main method: generate and dispatch alert based on risk level.
        Returns an AlertRecord with full details.
        """
        if risk_level == "low":
            return self._no_alert_record(text, category, confidence)

        timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_id    = f"ALERT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"
        snippet     = text[:120] + ("..." if len(text) > 120 else "")
        icon        = CATEGORY_ICONS.get(category, "⚠️")
        channels    = []

        if risk_level == "high":
            message = self._build_high_risk_message(category, snippet, icon, alert_id, confidence)
            channels += self._send_high_risk_alerts(message, alert_id, category)
        elif risk_level == "medium":
            message = self._build_medium_risk_message(category, snippet, icon, alert_id, confidence)
            channels += self._send_medium_risk_alerts(message, alert_id, category)

        record = AlertRecord(
            timestamp=timestamp,
            risk_level=risk_level,
            category=category,
            confidence=confidence,
            text_snippet=snippet,
            channels_notified=channels,
            alert_id=alert_id,
            message=message,
            status="sent" if channels else "failed",
        )

        self.alert_log.append(record.to_dict())
        self._persist_log()
        return record

    # ── Message Builders ─────────────────────────────────────────
    def _build_high_risk_message(self, category, snippet, icon, alert_id, confidence) -> str:
        return f"""
{'='*65}
🚨 HIGH RISK ALERT — IMMEDIATE ACTION REQUIRED
{'='*65}
Alert ID    : {alert_id}
Timestamp   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Category    : {icon} {category}
Risk Level  : ⛔ HIGH RISK
Confidence  : {confidence*100:.1f}%

Content Snippet:
  "{snippet}"

⚠️  POSSIBLE SITUATION: Suicide, violence, abuse, or cyberbullying detected.

REQUIRED ACTIONS:
  1. Immediately notify crisis response team
  2. Attempt contact with the user within 10 minutes
  3. If unresponsive, notify emergency services
  4. Document this incident per protocol

Emergency Resources:
  📞 988 Suicide & Crisis Lifeline (call or text)
  📞 Emergency Services: 911
  🌐 Crisis Chat: https://988lifeline.org/chat/

Alert System: Sentinel AI Mental Health Alert System
{'='*65}
"""

    def _build_medium_risk_message(self, category, snippet, icon, alert_id, confidence) -> str:
        return f"""
{'─'*65}
⚠️  MEDIUM RISK ALERT — COUNSELOR REVIEW REQUIRED
{'─'*65}
Alert ID    : {alert_id}
Timestamp   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Category    : {icon} {category}
Risk Level  : 🟡 MEDIUM RISK
Confidence  : {confidence*100:.1f}%

Content Snippet:
  "{snippet}"

SITUATION: Signs of stress, anxiety, sadness, or relationship conflict detected.

RECOMMENDED ACTIONS:
  1. Review full conversation context
  2. Reach out with a supportive message within 24 hours
  3. Provide mental health resources to the user
  4. Monitor for escalation to high-risk level

Resources to Share:
  📞 988 Suicide & Crisis Lifeline
  🌐 MentalHealth.gov
  📱 Crisis Text Line: Text HOME to 741741

Alert System: Sentinel AI Mental Health Alert System
{'─'*65}
"""

    # ── Alert Dispatchers ────────────────────────────────────────
    def _send_high_risk_alerts(self, message: str, alert_id: str, category: str) -> list:
        channels = []
        channels += self._send_email(message, ALERT_CONFIG["email"]["high_risk"],
                                     subject=f"🚨 HIGH RISK ALERT [{alert_id}] — {category}")
        channels += self._send_sms(message[:300], ALERT_CONFIG["sms"]["high_risk"])
        channels += self._notify_dashboard(message, "high")
        channels += self._concept_emergency_notify(message, alert_id)
        return channels

    def _send_medium_risk_alerts(self, message: str, alert_id: str, category: str) -> list:
        channels = []
        channels += self._send_email(message, ALERT_CONFIG["email"]["medium_risk"],
                                     subject=f"⚠️ Medium Risk Alert [{alert_id}] — {category}")
        channels += self._send_sms(message[:300], ALERT_CONFIG["sms"]["medium_risk"])
        channels += self._notify_dashboard(message, "medium")
        return channels

    # ── Channel Implementations ──────────────────────────────────
    def _send_email(self, body: str, recipients: list, subject: str) -> list:
        if self.simulation_mode:
            print(f"\n📧 [EMAIL SIMULATION] → {recipients}")
            print(f"   Subject: {subject}")
            print(body)
            return ["email (simulated)"]

        try:
            msg = MIMEMultipart()
            msg["From"]    = ALERT_CONFIG["email"]["sender"]
            msg["To"]      = ", ".join(recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(ALERT_CONFIG["email"]["smtp_host"],
                              ALERT_CONFIG["email"]["smtp_port"]) as server:
                server.starttls()
                server.login(ALERT_CONFIG["email"]["sender"],
                             ALERT_CONFIG["email"]["password"])
                server.sendmail(ALERT_CONFIG["email"]["sender"], recipients, msg.as_string())
            return ["email"]
        except Exception as e:
            print(f"  ⚠️ Email failed: {e}")
            return []

    def _send_sms(self, body: str, recipients: list) -> list:
        """
        SMS via Twilio.
        Production: pip install twilio
          from twilio.rest import Client
          client = Client(account_sid, auth_token)
          client.messages.create(body=body, from_=from_number, to=recipient)
        """
        if self.simulation_mode:
            print(f"\n📱 [SMS SIMULATION] → {recipients}")
            print(f"   Message: {body[:100]}...")
            return ["sms (simulated)"]

        try:
            from twilio.rest import Client
            client = Client(ALERT_CONFIG["sms"]["account_sid"],
                            ALERT_CONFIG["sms"]["auth_token"])
            for number in recipients:
                client.messages.create(
                    body=body[:1600],
                    from_=ALERT_CONFIG["sms"]["from_number"],
                    to=number,
                )
            return ["sms"]
        except ImportError:
            print("  ℹ️ Twilio not installed. Run: pip install twilio")
            return []
        except Exception as e:
            print(f"  ⚠️ SMS failed: {e}")
            return []

    def _notify_dashboard(self, message: str, risk_level: str) -> list:
        """Write to dashboard notification queue (JSON file)."""
        notif_path = os.path.join(self.log_dir, "dashboard_notifications.json")
        notifications = []
        if os.path.exists(notif_path):
            with open(notif_path) as f:
                try:
                    notifications = json.load(f)
                except Exception:
                    notifications = []
        notifications.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "risk_level": risk_level,
            "message": message[:500],
            "read": False,
        })
        with open(notif_path, "w") as f:
            json.dump(notifications[-100:], f, indent=2)  # keep last 100
        return ["dashboard"]

    def _concept_emergency_notify(self, message: str, alert_id: str) -> list:
        """
        Conceptual demonstration of emergency services API notification.
        In production: integrate with local emergency dispatch API.
        
        Example integration flow:
          POST https://api.emergency-services.gov/alert
          Headers: Authorization: Bearer {API_KEY}
          Body: {
            "alert_id": alert_id,
            "severity": "critical",
            "category": "mental_health",
            "message": message,
            "timestamp": now,
          }
        """
        print(f"\n🚔 [EMERGENCY SERVICES CONCEPT]")
        print(f"   Would dispatch to: {ALERT_CONFIG['emergency']['police_api']}")
        print(f"   Alert ID: {alert_id}")
        print(f"   Crisis Hotline: {ALERT_CONFIG['emergency']['crisis_hotline']}")
        print(f"   ⚠️  In production: integrate with local emergency services API")
        return ["emergency_services (concept)"]

    def _no_alert_record(self, text, category, confidence) -> AlertRecord:
        return AlertRecord(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            risk_level="low",
            category=category,
            confidence=confidence,
            text_snippet=text[:120],
            channels_notified=[],
            alert_id="",
            message="No alert required — Low risk content.",
            status="no_alert",
        )

    # ── Logging ──────────────────────────────────────────────────
    def _persist_log(self):
        path = os.path.join(self.log_dir, "alert_log.json")
        with open(path, "w") as f:
            json.dump(self.alert_log[-500:], f, indent=2)

    def get_alert_summary(self) -> dict:
        from collections import Counter
        levels = [a["risk_level"] for a in self.alert_log]
        cats   = [a["category"]   for a in self.alert_log]
        return {
            "total_alerts": len(self.alert_log),
            "by_risk_level": dict(Counter(levels)),
            "by_category":   dict(Counter(cats)),
            "recent":        self.alert_log[-5:],
        }


if __name__ == "__main__":
    alerter = AlertSystem(simulation_mode=True)

    # Test high-risk alert
    print("\n" + "="*65)
    print("TEST 1: High Risk Alert")
    r = alerter.process_alert(
        text="I've written my goodbye letters and I'm ready to end it.",
        category="Suicide Risk",
        risk_level="high",
        confidence=0.94,
    )
    print(f"\nAlert ID: {r.alert_id}")
    print(f"Channels: {r.channels_notified}")

    # Test medium-risk alert
    print("\n" + "="*65)
    print("TEST 2: Medium Risk Alert")
    r2 = alerter.process_alert(
        text="The anxiety is so bad lately I can't even leave my house.",
        category="Stress / Anxiety",
        risk_level="medium",
        confidence=0.78,
    )
    print(f"\nAlert ID: {r2.alert_id}")
    print(f"Channels: {r2.channels_notified}")

    print("\nSummary:", json.dumps(alerter.get_alert_summary(), indent=2))
