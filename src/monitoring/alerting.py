"""
src/monitoring/alerting.py
---------------------------
Alerting integration for fraud model monitoring.
Supports Slack webhooks and PagerDuty Events API v2.

Triggered by:
- Data drift detection (PSI > threshold)
- Model performance degradation (AUC drop)
- High error rate or latency spike
- Fraud rate anomaly (> 3σ from baseline)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SLACK_TIMEOUT_S = 10
PAGERDUTY_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# PagerDuty severity mapping
_PD_SEVERITY = {
    AlertSeverity.INFO: "info",
    AlertSeverity.WARNING: "warning",
    AlertSeverity.CRITICAL: "critical",
}

# Slack color mapping
_SLACK_COLOR = {
    AlertSeverity.INFO: "#36a64f",
    AlertSeverity.WARNING: "#ffa500",
    AlertSeverity.CRITICAL: "#ff0000",
}


@dataclass
class Alert:
    title: str
    message: str
    severity: AlertSeverity
    source: str = "fraudguard-ml"
    details: dict[str, Any] | None = None
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(tz=UTC)


class AlertingService:
    """
    Multi-channel alerting service.

    Sends structured alerts to Slack and/or PagerDuty.
    Fails silently — alerting must never crash the main application.

    Usage:
        alerter = AlertingService(
            slack_webhook="https://hooks.slack.com/services/...",
            pagerduty_key="your-routing-key",
        )
        await alerter.send(Alert(
            title="Data drift detected",
            message="Feature 'amount_zscore' PSI=0.38 exceeds threshold 0.25",
            severity=AlertSeverity.CRITICAL,
        ))
    """

    def __init__(
        self,
        slack_webhook: str | None = None,
        pagerduty_key: str | None = None,
        environment: str = "production",
    ) -> None:
        self._slack_webhook = slack_webhook
        self._pagerduty_key = pagerduty_key
        self._environment = environment

        channels = []
        if slack_webhook:
            channels.append("Slack")
        if pagerduty_key:
            channels.append("PagerDuty")
        if channels:
            logger.info("AlertingService initialized: %s", ", ".join(channels))
        else:
            logger.warning(
                "AlertingService: no channels configured (set SLACK_WEBHOOK or PAGERDUTY_KEY)"
            )

    async def send(self, alert: Alert) -> None:
        """Dispatch alert to all configured channels."""
        if self._slack_webhook:
            await self._send_slack(alert)
        if self._pagerduty_key and alert.severity == AlertSeverity.CRITICAL:
            await self._send_pagerduty(alert)

    async def send_drift_alert(
        self,
        drifted_features: list[str],
        max_psi: float,
        threshold: float,
    ) -> None:
        """Convenience: send a feature drift alert."""
        await self.send(
            Alert(
                title="🔴 Data drift detected",
                message=(
                    f"{len(drifted_features)} feature(s) drifted. "
                    f"Max PSI={max_psi:.3f} (threshold={threshold:.2f}). "
                    f"Drifted: {', '.join(drifted_features[:5])}"
                    + (" ..." if len(drifted_features) > 5 else "")
                ),
                severity=AlertSeverity.CRITICAL if max_psi > 0.40 else AlertSeverity.WARNING,
                details={
                    "drifted_features": drifted_features,
                    "max_psi": round(max_psi, 4),
                    "threshold": threshold,
                },
            )
        )

    async def send_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        drop_pct: float,
    ) -> None:
        """Convenience: send a model performance degradation alert."""
        await self.send(
            Alert(
                title="📉 Model performance degraded",
                message=(
                    f"{metric_name} dropped {drop_pct:.1f}% "
                    f"({baseline_value:.4f} → {current_value:.4f}). "
                    "Consider retraining."
                ),
                severity=AlertSeverity.CRITICAL if drop_pct > 10 else AlertSeverity.WARNING,
                details={
                    "metric": metric_name,
                    "current": round(current_value, 4),
                    "baseline": round(baseline_value, 4),
                    "drop_pct": round(drop_pct, 2),
                },
            )
        )

    async def send_fraud_rate_alert(
        self,
        current_rate: float,
        baseline_rate: float,
        z_score: float,
    ) -> None:
        """Convenience: send a fraud rate anomaly alert."""
        direction = "spike" if current_rate > baseline_rate else "drop"
        await self.send(
            Alert(
                title=f"⚠️ Fraud rate {direction} detected",
                message=(
                    f"Fraud rate {current_rate:.3%} vs baseline {baseline_rate:.3%} "
                    f"(z={z_score:.1f}σ)."
                ),
                severity=AlertSeverity.CRITICAL if abs(z_score) > 4 else AlertSeverity.WARNING,
                details={
                    "current_rate": round(current_rate, 5),
                    "baseline_rate": round(baseline_rate, 5),
                    "z_score": round(z_score, 2),
                },
            )
        )

    async def send_latency_alert(self, p99_ms: float, sla_ms: float = 200) -> None:
        """Convenience: send a latency SLA breach alert."""
        await self.send(
            Alert(
                title="🐌 Latency SLA breached",
                message=f"P99 latency {p99_ms:.0f}ms exceeds SLA {sla_ms:.0f}ms.",
                severity=AlertSeverity.WARNING,
                details={"p99_ms": p99_ms, "sla_ms": sla_ms},
            )
        )

    # ── Private: Slack ────────────────────────────────────────────────────────

    async def _send_slack(self, alert: Alert) -> None:
        """Send alert to Slack via webhook."""
        color = _SLACK_COLOR[alert.severity]
        ts = alert.timestamp.isoformat() if alert.timestamp else ""

        fields = [
            {"title": "Environment", "value": self._environment, "short": True},
            {"title": "Severity", "value": alert.severity.upper(), "short": True},
        ]
        if alert.details:
            for k, v in list(alert.details.items())[:6]:
                fields.append({"title": k, "value": str(v), "short": True})

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": fields,
                    "footer": f"FraudGuard ML | {alert.source}",
                    "ts": ts,
                }
            ]
        }

        try:
            async with httpx.AsyncClient(timeout=SLACK_TIMEOUT_S) as client:
                resp = await client.post(self._slack_webhook, json=payload)  # type: ignore[arg-type]
                if resp.status_code != 200:
                    logger.warning(
                        "Slack alert failed: status=%d body=%s",
                        resp.status_code,
                        resp.text,
                    )
                else:
                    logger.info("Slack alert sent: '%s'", alert.title)
        except Exception as exc:
            logger.warning("Slack alert error (non-fatal): %s", exc)

    # ── Private: PagerDuty ────────────────────────────────────────────────────

    async def _send_pagerduty(self, alert: Alert) -> None:
        """Trigger a PagerDuty incident via Events API v2."""
        payload = {
            "routing_key": self._pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "source": f"{self._environment}/{alert.source}",
                "severity": _PD_SEVERITY[alert.severity],
                "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                "custom_details": {
                    "message": alert.message,
                    **(alert.details or {}),
                },
            },
        }

        try:
            async with httpx.AsyncClient(timeout=SLACK_TIMEOUT_S) as client:
                resp = await client.post(PAGERDUTY_EVENTS_URL, json=payload)
                if resp.status_code not in {200, 202}:
                    logger.warning(
                        "PagerDuty alert failed: status=%d body=%s",
                        resp.status_code,
                        resp.text,
                    )
                else:
                    logger.info("PagerDuty incident triggered: '%s'", alert.title)
        except Exception as exc:
            logger.warning("PagerDuty alert error (non-fatal): %s", exc)


def build_alerting_service() -> AlertingService:
    """Build AlertingService from application settings."""
    from src.core.config import get_settings

    settings = get_settings()
    return AlertingService(
        slack_webhook=settings.mlflow.tracking_uri,  # placeholder — add SLACK_WEBHOOK to settings
        environment=settings.environment,
    )
