"""
src/monitoring/drift_detector.py
---------------------------------
Statistical drift detection for input feature distributions and concept drift.

Uses:
- Population Stability Index (PSI): standard in banking
- Kolmogorov-Smirnov test: continuous feature drift
- Jensen-Shannon divergence: symmetric, bounded [0,1]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from src.core.interfaces import BaseDriftDetector
from src.monitoring.metrics_collector import get_metrics

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

PSI_THRESHOLDS = {
    "stable": 0.10,
    "moderate": 0.20,
    "significant": 0.25,
}


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    timestamp: str
    overall_drifted: bool
    drifted_features: list[str] = field(default_factory=list)
    stable_features: list[str] = field(default_factory=list)
    feature_psi: dict[str, float] = field(default_factory=dict)
    feature_ks_statistic: dict[str, float] = field(default_factory=dict)
    feature_ks_pvalue: dict[str, float] = field(default_factory=dict)
    feature_js_divergence: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


class FraudDriftDetector(BaseDriftDetector):
    """
    Production drift detection for fraud feature distributions.

    Usage:
        detector = FraudDriftDetector()
        detector.fit_reference(training_feature_df)

        # Run periodically:
        report = detector.detect_drift(recent_predictions_df)
        if report.overall_drifted:
            trigger_alert(report)
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        ks_threshold: float = 0.05,
        n_bins: int = 10,
    ) -> None:
        self._psi_threshold = psi_threshold
        self._ks_threshold = ks_threshold
        self._n_bins = n_bins
        self._reference_data: pd.DataFrame | None = None
        self._reference_stats: dict[str, dict[str, Any]] = {}
        self._metrics = get_metrics()

    def fit_reference(self, reference_data: pd.DataFrame) -> None:
        """
        Fit reference distribution from training/historical data.
        Must be called once before detect_drift().
        """
        self._reference_data = reference_data.copy()
        self._reference_stats = {}

        for col in reference_data.select_dtypes(include=[np.number]).columns:
            series = reference_data[col].dropna()
            self._reference_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "percentiles": np.percentile(series, [10, 25, 50, 75, 90]).tolist(),
                "values": series.values,
                "bins": np.histogram_bin_edges(series, bins=self._n_bins),
            }

        logger.info(
            "Reference distribution fitted: %d samples, %d numeric features",
            len(reference_data),
            len(self._reference_stats),
        )

    def detect_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """
        Compare current feature distributions against reference.

        Returns DriftReport with per-feature and aggregate drift signals.
        """
        if self._reference_data is None:
            msg = "Reference distribution not fitted. Call fit_reference() first."
            raise RuntimeError(msg)

        report = DriftReport(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            overall_drifted=False,
        )

        numeric_cols = [
            col for col in current_data.select_dtypes(include=[np.number]).columns
            if col in self._reference_stats
        ]

        for col in numeric_cols:
            current_series = current_data[col].dropna()
            if len(current_series) < 30:
                continue

            ref_stats = self._reference_stats[col]
            ref_series = ref_stats["values"]

            psi = self._compute_psi(ref_series, current_series.values, ref_stats["bins"])
            report.feature_psi[col] = round(float(psi), 4)

            ks_stat, ks_pvalue = ks_2samp(ref_series, current_series.values)
            report.feature_ks_statistic[col] = round(float(ks_stat), 4)
            report.feature_ks_pvalue[col] = round(float(ks_pvalue), 4)

            js_div = self._compute_js_divergence(ref_series, current_series.values, ref_stats["bins"])
            report.feature_js_divergence[col] = round(float(js_div), 4)

            if psi > self._psi_threshold or ks_pvalue < self._ks_threshold:
                report.drifted_features.append(col)
                logger.warning(
                    "Drift detected in '%s': PSI=%.3f KS-p=%.4f", col, psi, ks_pvalue
                )
            else:
                report.stable_features.append(col)

            self._metrics.update_feature_drift(col, psi)

        report.overall_drifted = len(report.drifted_features) > 0
        report.summary = {
            "total_features_checked": len(numeric_cols),
            "drifted_feature_count": len(report.drifted_features),
            "stable_feature_count": len(report.stable_features),
            "drift_rate": len(report.drifted_features) / max(len(numeric_cols), 1),
            "max_psi": max(report.feature_psi.values()) if report.feature_psi else 0.0,
            "max_ks_stat": max(report.feature_ks_statistic.values()) if report.feature_ks_statistic else 0.0,
        }

        if report.overall_drifted:
            logger.warning(
                "DATA DRIFT DETECTED: %d/%d features drifted: %s",
                len(report.drifted_features), len(numeric_cols), report.drifted_features[:5],
            )
        else:
            logger.info(
                "No drift detected across %d features. Max PSI=%.3f",
                len(numeric_cols), report.summary["max_psi"],
            )

        return report

    def is_drifted(self, current_data: pd.DataFrame) -> bool:
        """Returns True if any drift detected."""
        return self.detect_drift(current_data).overall_drifted

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: np.ndarray) -> float:
        """
        Population Stability Index.
        < 0.10: stable  |  0.10-0.25: moderate  |  > 0.25: significant
        """
        eps = 1e-8
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)
        ref_pct = ref_hist / (ref_hist.sum() + eps) + eps
        cur_pct = cur_hist / (cur_hist.sum() + eps) + eps
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    @staticmethod
    def _compute_js_divergence(
        reference: np.ndarray,
        current: np.ndarray,
        bins: np.ndarray,
    ) -> float:
        """Jensen-Shannon divergence — symmetric, bounded [0, log(2)]."""
        eps = 1e-8
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)
        return float(jensenshannon(ref_hist + eps, cur_hist + eps))
