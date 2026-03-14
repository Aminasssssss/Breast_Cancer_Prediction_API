"""
Alerting system for model monitoring and infrastructure issues.
"""
import os
import json
import logging
import smtplib
import requests
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from src.config import settings

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertChannel(Enum):
    EMAIL = "email"
    DISCORD = "discord"
    SLACK = "slack"
    LOG = "log"
    WEBHOOK = "webhook"


@dataclass
class Alert:
    title: str
    message: str
    severity: AlertSeverity
    channel: AlertChannel
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "channel": self.channel.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class EmailAlertSender:
    def __init__(self):
        self.host = settings.SMTP_HOST
        self.port = settings.SMTP_PORT
        self.user = settings.SMTP_USER
        self.password = settings.SMTP_PASSWORD
        self.from_email = settings.ALERT_EMAIL_FROM
        self.to_emails = settings.ALERT_EMAIL_TO
        
        self.enabled = all([self.host, self.user, self.password, self.from_email, self.to_emails])
    
    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            logger.warning(f"Email alerts disabled - would send: {alert.title}")
            return False
        
        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = self.to_emails if isinstance(self.to_emails, str) else ", ".join(self.to_emails)
            msg["Subject"] = f"[{alert.severity.value}] {alert.title}"
            
            body = f"""
            Alert: {alert.title}
            Severity: {alert.severity.value}
            Time: {alert.timestamp.isoformat()}
            
            Message:
            {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}
            
            ---
            Breast Cancer Prediction API
            Environment: {settings.APP_ENVIRONMENT}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class DiscordAlertSender:
    def __init__(self):
        self.webhook_url = settings.ALERT_DISCORD_WEBHOOK
        self.enabled = bool(self.webhook_url)
    
    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        
        try:
            color_map = {
                AlertSeverity.DEBUG: 0x808080,
                AlertSeverity.INFO: 0x00FF00,
                AlertSeverity.WARNING: 0xFFA500,
                AlertSeverity.ERROR: 0xFF0000,
                AlertSeverity.CRITICAL: 0x8B0000
            }
            
            payload = {
                "embeds": [{
                    "title": alert.title,
                    "description": alert.message,
                    "color": color_map.get(alert.severity, 0x000000),
                    "fields": [
                        {"name": "Severity", "value": alert.severity.value, "inline": True},
                        {"name": "Environment", "value": settings.APP_ENVIRONMENT, "inline": True},
                        {"name": "Timestamp", "value": alert.timestamp.isoformat(), "inline": True}
                    ],
                    "footer": {"text": "Breast Cancer Prediction API"},
                    "timestamp": alert.timestamp.isoformat()
                }]
            }
            
            if alert.metadata:
                payload["embeds"][0]["fields"].append({
                    "name": "Metadata",
                    "value": json.dumps(alert.metadata, indent=2)[:1024],
                    "inline": False
                })
            
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            
            logger.info(f"Discord alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False


class SlackAlertSender:
    def __init__(self):
        self.webhook_url = settings.ALERT_SLACK_WEBHOOK
        self.enabled = bool(self.webhook_url)
    
    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        
        try:
            color_map = {
                AlertSeverity.DEBUG: "#808080",
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffa500",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }
            
            payload = {
                "attachments": [{
                    "title": alert.title,
                    "text": alert.message,
                    "color": color_map.get(alert.severity, "#000000"),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Environment", "value": settings.APP_ENVIRONMENT, "short": True}
                    ],
                    "footer": "Breast Cancer Prediction API",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            if alert.metadata:
                payload["attachments"][0]["fields"].append({
                    "title": "Metadata",
                    "value": json.dumps(alert.metadata, indent=2),
                    "short": False
                })
            
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class WebhookAlertSender:
    def __init__(self):
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)
    
    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        
        try:
            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                timeout=5,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"Webhook alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class AlertManager:
    def __init__(self):
        self.senders = {
            AlertChannel.EMAIL: EmailAlertSender(),
            AlertChannel.DISCORD: DiscordAlertSender(),
            AlertChannel.SLACK: SlackAlertSender(),
            AlertChannel.WEBHOOK: WebhookAlertSender(),
        }
        
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        self.suppression_window_minutes = 5
        self.recent_alerts: Dict[str, datetime] = {}
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if similar alert should be suppressed"""
        key = f"{alert.channel.value}:{alert.title}:{alert.severity.value}"
        last_time = self.recent_alerts.get(key)
        
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds() / 60
            if elapsed < self.suppression_window_minutes:
                logger.debug(f"Suppressing duplicate alert: {alert.title}")
                return True
        
        self.recent_alerts[key] = datetime.now()
        return False
    
    def send(self, title: str, message: str, severity: str = "WARNING", channels: List[str] = None):
        """Send alert through configured channels"""
        try:
            severity_enum = AlertSeverity[severity.upper()]
        except KeyError:
            severity_enum = AlertSeverity.WARNING
        
        log_alert = Alert(
            title=title,
            message=message,
            severity=severity_enum,
            channel=AlertChannel.LOG
        )
        self._log_alert(log_alert)
        self._store_alert(log_alert)
        
        if channels is None:
            if severity_enum in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                channels = ["email", "discord", "slack"]
            else:
                channels = ["discord", "slack"]
        
        for channel_name in channels:
            try:
                channel = AlertChannel(channel_name.lower())
                sender = self.senders.get(channel)
                
                if sender and sender.enabled:
                    alert = Alert(
                        title=title,
                        message=message,
                        severity=severity_enum,
                        channel=channel
                    )
                    
                    if not self._is_suppressed(alert):
                        sender.send(alert)
                        self._store_alert(alert)
                        
            except ValueError:
                logger.warning(f"Unknown channel: {channel_name}")
    
    def _log_alert(self, alert: Alert):
        """Log alert with appropriate level"""
        log_func = {
            AlertSeverity.DEBUG: logger.debug,
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }.get(alert.severity, logger.info)
        
        log_func(f"ALERT [{alert.severity.value}]: {alert.title} - {alert.message}")
    
    def _store_alert(self, alert: Alert):
        """Store alert in history"""
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
    
    def get_history(self, limit: int = 100, severity: str = None) -> List[Dict[str, Any]]:
        """Get alert history"""
        alerts = self.alert_history[-limit:]
        
        if severity:
            severity_upper = severity.upper()
            alerts = [a for a in alerts if a.severity.value == severity_upper]
        
        return [a.to_dict() for a in alerts]
    
    def check_model_health(self) -> bool:
        """Check if model API is healthy"""
        try:
            response = requests.get(f"http://localhost:{settings.API_PORT}/health", timeout=5)
            is_healthy = response.status_code == 200
            
            if not is_healthy:
                self.send(
                    title="API Health Check Failed",
                    message=f"API returned status {response.status_code}",
                    severity="ERROR"
                )
            
            return is_healthy
            
        except Exception as e:
            self.send(
                title="API Unreachable",
                message=f"Cannot connect to API: {str(e)}",
                severity="CRITICAL"
            )
            return False
    
    def check_model_performance(self, current_accuracy: float, threshold: float = 0.85):
        """Check if model performance dropped below threshold"""
        if current_accuracy < threshold:
            self.send(
                title="Model Performance Degradation",
                message=f"Model accuracy dropped to {current_accuracy:.2%} (threshold: {threshold:.2%})",
                severity="ERROR",
                channels=["email", "discord", "slack"]
            )


alert_manager = AlertManager()