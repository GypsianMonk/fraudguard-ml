"""
tests/unit/test_drift_detector.py
-----------------------------------
Unit tests for FraudDriftDetector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import FraudDriftDetector


def make_numeric_df(n: int, means: list[float], std: float = 1.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        f"feature_{i}": rng.normal(loc=mean, scale=std, size=n)
        for i, mean in enumerate(means)
    })


@pytest.fixture
def detector():
    return FraudDriftDetector(psi_threshold=0.25, ks_threshold=0.05)


@pytest.fixture
def reference_df():
    return make_numeric_df(n=5000, means=[0.0, 1.0, -1.0, 2.5], seed=10)


@pytest.fixture
def fitted_detector(detector, reference_df):
    detector.fit_reference(reference_df)
    return detector


class TestFraudDriftDetector:

    def test_fit_reference_sets_stats(self, detector, reference_df):
        detector.fit_reference(reference_df)
        assert len(detector._reference_stats) == 4
        assert "feature_0" in detector._reference_stats

    def test_detect_without_fit_raises(self, detector, reference_df):
        with pytest.raises(RuntimeError, match="fit_reference"):
            detector.detect_drift(reference_df)

    def test_no_drift_on_identical_distribution(self, fitted_detector, reference_df):
        report = fitted_detector.detect_drift(reference_df)
        assert not report.overall_drifted

    def test_no_drift_on_similar_distribution(self, fitted_detector):
        similar = make_numeric_df(n=2000, means=[0.05, 1.05, -0.95, 2.55], seed=99)
        report = fitted_detector.detect_drift(similar)
        # Very small shift should not trigger drift
        assert not report.overall_drifted

    def test_drift_detected_on_shifted_distribution(self, fitted_detector):
        # Large shift: mean shifted by 3 standard deviations
        drifted = make_numeric_df(n=2000, means=[3.0, 4.0, 2.0, 5.5], std=1.0, seed=99)
        report = fitted_detector.detect_drift(drifted)
        assert report.overall_drifted
        assert len(report.drifted_features) > 0

    def test_report_contains_psi_scores(self, fitted_detector, reference_df):
        report = fitted_detector.detect_drift(reference_df)
        assert isinstance(report.feature_psi, dict)
        assert len(report.feature_psi) > 0
        for val in report.feature_psi.values():
            assert val >= 0.0

    def test_report_contains_ks_stats(self, fitted_detector, reference_df):
        report = fitted_detector.detect_drift(reference_df)
        assert isinstance(report.feature_ks_statistic, dict)
        for val in report.feature_ks_statistic.values():
            assert 0.0 <= val <= 1.0

    def test_report_has_summary(self, fitted_detector, reference_df):
        report = fitted_detector.detect_drift(reference_df)
        assert "total_features_checked" in report.summary
        assert "drifted_feature_count" in report.summary
        assert "max_psi" in report.summary

    def test_is_drifted_convenience_method(self, fitted_detector):
        drifted = make_numeric_df(n=2000, means=[5.0, 6.0, 4.0, 7.5], seed=99)
        assert fitted_detector.is_drifted(drifted)

    def test_is_not_drifted_on_same_data(self, fitted_detector, reference_df):
        assert not fitted_detector.is_drifted(reference_df)

    def test_psi_near_zero_for_identical(self, fitted_detector, reference_df):
        report = fitted_detector.detect_drift(reference_df)
        for feature, psi in report.feature_psi.items():
            assert psi < 0.1, f"Expected low PSI for identical data, got {psi:.4f} for {feature}"

    def test_drifted_features_subset_of_all_features(self, fitted_detector):
        drifted = make_numeric_df(n=2000, means=[5.0, 1.0, -1.0, 2.5], seed=99)
        report = fitted_detector.detect_drift(drifted)
        all_features = set(report.feature_psi.keys())
        for f in report.drifted_features:
            assert f in all_features

    def test_small_sample_skipped_gracefully(self, fitted_detector, reference_df):
        """Columns with very few samples should be skipped without error."""
        tiny_df = reference_df.head(10)
        # Should not raise, may just skip features with too few samples
        report = fitted_detector.detect_drift(tiny_df)
        assert report is not None

    def test_psi_computation_known_values(self, detector):
        """Verify PSI calculation with known expected values."""
        # Same distribution → PSI near 0
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000)
        same = rng.normal(0, 1, 5000)
        bins = np.histogram_bin_edges(ref, bins=10)
        psi_same = detector._compute_psi(ref, same, bins)
        assert psi_same < 0.05

        # Very different distribution → higher PSI
        diff = rng.normal(5, 1, 5000)
        psi_diff = detector._compute_psi(ref, diff, bins)
        assert psi_diff > psi_same
