"""
src/monitoring/metrics_collector.py
------------------------------------
Prometheus metrics for the fraud detection service.
All metrics follow Prometheus naming conventions.
"""

from __future__ import annotations

from functools import lru_cache

from prometheus_client import Counter, Gauge, Histogram, Info


class FraudMetricsCollector:
    """
    Centralized Prometheus metrics for FraudGuard ML.

    Metrics exposed at /metrics for Prometheus scraping.
    Grafana dashboards built on top of these metrics.
    """

    def __init__(self) -> None:
        # --- Prediction metrics ---
        self.predictions_total = Counter(
            "fraudguard_predictions_total",
            "Total number of fraud predictions made",
            labelnames=["risk_tier", "environment"],
        )

        self.fraud_predictions_total = Counter(
            "fraudguard_fraud_predictions_total",
            "Total number of transactions predicted as fraud",
            labelnames=["risk_tier"],
        )

        self.prediction_latency_ms = Histogram(
            "fraudguard_prediction_latency_ms",
            "End-to-end prediction latency in milliseconds",
            buckets=[5, 10, 20, 30, 50, 75, 100, 150, 200, 500],
        )

        self.fraud_probability = Histogram(
            "fraudguard_fraud_probability",
            "Distribution of fraud probability scores",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # --- Feature store metrics ---
        self.feature_store_requests_total = Counter(
            "fraudguard_feature_store_requests_total",
            "Total feature store requests",
            labelnames=["status"],  # hit | miss | error
        )

        self.feature_store_latency_ms = Histogram(
            "fraudguard_feature_store_latency_ms",
            "Feature store retrieval latency in milliseconds",
            buckets=[1, 2, 5, 10, 20, 50],
        )

        # --- Error metrics ---
        self.errors_total = Counter(
            "fraudguard_errors_total",
            "Total application errors",
            labelnames=["error_type"],
        )

        # --- Model metrics (updated by monitoring job) ---
        self.model_auc_roc = Gauge(
            "fraudguard_model_auc_roc",
            "Current model AUC-ROC on recent evaluation window",
        )

        self.model_auc_pr = Gauge(
            "fraudguard_model_auc_pr",
            "Current model AUC-PR on recent evaluation window",
        )

        self.feature_drift_psi = Gauge(
            "fraudguard_feature_drift_psi",
            "Population Stability Index for each feature",
            labelnames=["feature_name"],
        )

        self.fraud_rate_gauge = Gauge(
            "fraudguard_fraud_rate_rolling_1h",
            "Rolling 1-hour fraud rate (predictions labeled fraud / total)",
        )

        # --- System info ---
        self.model_info = Info(
            "fraudguard_model",
            "Information about the currently loaded model",
        )

        # --- Batch metrics ---
        self.batch_size = Histogram(
            "fraudguard_batch_size",
            "Number of transactions in batch prediction requests",
            buckets=[1, 10, 50, 100, 250, 500, 1000],
        )

    def record_prediction(
        self,
        fraud_prob: float,
        risk_tier: str,
        latency_ms: int,
        environment: str = "production",
    ) -> None:
        """Record a prediction observation."""
        self.predictions_total.labels(risk_tier=risk_tier, environment=environment).inc()
        self.prediction_latency_ms.observe(latency_ms)
        self.fraud_probability.observe(fraud_prob)

        if fraud_prob >= 0.5:
            self.fraud_predictions_total.labels(risk_tier=risk_tier).inc()

    def record_error(self, error_type: str) -> None:
        """Record an application error."""
        self.errors_total.labels(error_type=error_type).inc()

    def record_feature_store_request(self, status: str, latency_ms: float) -> None:
        """Record a feature store operation."""
        self.feature_store_requests_total.labels(status=status).inc()
        self.feature_store_latency_ms.observe(latency_ms)

    def update_model_metrics(self, auc_roc: float, auc_pr: float) -> None:
        """Update model performance gauges (called by monitoring job)."""
        self.model_auc_roc.set(auc_roc)
        self.model_auc_pr.set(auc_pr)

    def update_feature_drift(self, feature_name: str, psi: float) -> None:
        """Update PSI drift score for a feature."""
        self.feature_drift_psi.labels(feature_name=feature_name).set(psi)

    def update_fraud_rate(self, rate: float) -> None:
        """Update rolling fraud rate gauge."""
        self.fraud_rate_gauge.set(rate)

    def set_model_info(self, version: str, stage: str, training_date: str) -> None:
        """Set static model metadata."""
        self.model_info.info({
            "version": version,
            "stage": stage,
            "training_date": training_date,
        })


@lru_cache(maxsize=1)
def get_metrics() -> FraudMetricsCollector:
    """Singleton metrics collector."""
    return FraudMetricsCollector()
