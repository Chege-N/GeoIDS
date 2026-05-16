"""
tests/test_detector.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the AnomalyDetector, GPDThreshold, and AnomalyResult.
"""

import math
import pickle
import time
import pytest
import numpy as np

from geoidslib.detection.detector import AnomalyDetector, AnomalyResult, GPDThreshold
from geoidslib.algebra.ga_engine import GeometricAlgebraEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_engine():
    return GeometricAlgebraEngine(dim=6, p=4, q=2, max_grade=2)


@pytest.fixture
def detector(small_engine):
    return AnomalyDetector(
        ga_engine=small_engine,
        window_size=500,
        reframe_interval=20,
        forgetting_factor=0.99,
        use_isolation_forest=False,   # skip IF for speed
        target_fpr=0.05,
    )


def make_features(seed: int = 0, dim: int = 6, value: float | None = None) -> np.ndarray:
    if value is not None:
        return np.full(dim, value)
    return np.random.default_rng(seed).uniform(0, 1, size=dim)


# ---------------------------------------------------------------------------
# GPDThreshold
# ---------------------------------------------------------------------------

class TestGPDThreshold:

    def test_threshold_before_min_samples(self):
        gpd = GPDThreshold(min_samples=100, target_fpr=0.1)
        for i in range(50):
            gpd.update(float(i))
        t = gpd.threshold
        assert math.isfinite(t)
        assert t > 0

    def test_threshold_after_min_samples(self):
        gpd = GPDThreshold(min_samples=50, target_fpr=0.05, refit_every=10)
        rng = np.random.default_rng(99)
        scores = rng.exponential(scale=1.0, size=200)
        for s in scores:
            gpd.update(float(s))
        assert math.isfinite(gpd.threshold)
        assert gpd.threshold > 0

    def test_threshold_increases_with_lower_fpr(self):
        gpd_hi = GPDThreshold(min_samples=20, target_fpr=0.2)
        gpd_lo = GPDThreshold(min_samples=20, target_fpr=0.01)
        rng = np.random.default_rng(7)
        for s in rng.exponential(1.0, 100):
            gpd_hi.update(float(s))
            gpd_lo.update(float(s))
        # Stricter FPR → higher threshold
        assert gpd_lo.threshold >= gpd_hi.threshold

    def test_multiple_updates_stable(self):
        gpd = GPDThreshold(min_samples=30, target_fpr=0.05, refit_every=10)
        rng = np.random.default_rng(11)
        thresholds = []
        for s in rng.exponential(1.0, 150):
            t = gpd.update(float(s))
            thresholds.append(t)
        # Threshold should not diverge
        assert all(math.isfinite(t) for t in thresholds)
        assert all(t > 0 for t in thresholds)


# ---------------------------------------------------------------------------
# AnomalyResult
# ---------------------------------------------------------------------------

class TestAnomalyResult:

    def test_to_dict_keys(self):
        result = AnomalyResult(
            flow_id="test_flow",
            timestamp=1234567890.0,
            score=0.5,
            is_anomaly=True,
            threshold=0.3,
            blade_explanations=[{
                "blade_key": (0, 1),
                "blade_label": "e01",
                "description": "packet_count ∧ byte_count",
                "deviation": 0.4,
            }],
            top_blade_label="packet_count ∧ byte_count",
            confidence=0.8,
        )
        d = result.to_dict()
        assert d["flow_id"] == "test_flow"
        assert d["is_anomaly"] is True
        assert d["score"] == pytest.approx(0.5, rel=1e-4)
        assert "blade_explanations" in d
        assert len(d["blade_explanations"]) == 1

    def test_to_dict_no_explanations(self):
        result = AnomalyResult(
            flow_id="f", timestamp=0.0, score=0.1,
            is_anomaly=False, threshold=0.5,
        )
        d = result.to_dict()
        assert d["blade_explanations"] == []


# ---------------------------------------------------------------------------
# AnomalyDetector — core functionality
# ---------------------------------------------------------------------------

class TestAnomalyDetector:

    def test_process_flow_returns_result(self, detector):
        feat = make_features(0, dim=6)
        result = detector.process_flow(feat, flow_id="f0")
        assert isinstance(result, AnomalyResult)
        assert result.flow_id == "f0"
        assert math.isfinite(result.score)

    def test_score_nonnegative(self, detector):
        for i in range(10):
            feat = make_features(i, dim=6)
            result = detector.process_flow(feat)
            assert result.score >= 0.0

    def test_timestamp_auto(self, detector):
        before = time.time()
        result = detector.process_flow(make_features(0, dim=6))
        after = time.time()
        assert before <= result.timestamp <= after

    def test_stats_keys(self, detector):
        for i in range(5):
            detector.process_flow(make_features(i, dim=6))
        stats = detector.stats
        assert "total_flows" in stats
        assert "total_alerts" in stats
        assert "current_threshold" in stats
        assert stats["total_flows"] == 5

    def test_normal_window_fills(self, detector):
        for i in range(30):
            detector.process_flow(make_features(i, dim=6))
        assert detector.stats["normal_window_size"] > 0

    def test_force_recompute_reference(self, detector):
        for i in range(50):
            detector.process_flow(make_features(i, dim=6))
        detector.force_recompute_reference()
        assert detector._reference_mv is not None

    def test_alerts_for_extreme_outlier(self, small_engine):
        """Flows that are far from the normal cluster should eventually trigger alerts."""
        det = AnomalyDetector(
            ga_engine=small_engine,
            window_size=200,
            reframe_interval=30,
            use_isolation_forest=False,
            target_fpr=0.1,
        )
        # Feed normal flows (values ~0.5)
        rng = np.random.default_rng(42)
        for i in range(100):
            feat = rng.uniform(0.4, 0.6, size=6)
            det.process_flow(feat)
        det.force_recompute_reference()

        # Now feed extreme outliers
        alerts = 0
        for _ in range(20):
            extreme = np.ones(6)  # max value
            result = det.process_flow(extreme)
            if result.is_anomaly:
                alerts += 1

        assert alerts > 0, "Expected at least one alert for extreme outliers"

    def test_batch_processing(self, detector):
        batch = np.random.default_rng(5).uniform(0, 1, size=(10, 6))
        results = detector.process_batch(batch)
        assert len(results) == 10
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_batch_auto_flow_ids(self, detector):
        batch = np.random.default_rng(6).uniform(0, 1, size=(3, 6))
        results = detector.process_batch(batch)
        assert results[0].flow_id == "flow_0"
        assert results[2].flow_id == "flow_2"

    def test_total_flows_counter(self, detector):
        n = 15
        for i in range(n):
            detector.process_flow(make_features(i, dim=6))
        assert detector.stats["total_flows"] == n


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestDetectorPersistence:

    def test_save_load_roundtrip(self, detector, tmp_path):
        # Process some flows
        for i in range(60):
            detector.process_flow(make_features(i, dim=6))
        detector.force_recompute_reference()

        path = str(tmp_path / "state.pkl")
        detector.save_state(path)

        # Create new detector and load
        engine2 = GeometricAlgebraEngine(dim=6, p=4, q=2, max_grade=2)
        det2 = AnomalyDetector(ga_engine=engine2, use_isolation_forest=False)
        det2.load_state(path)

        assert det2._total_flows == detector._total_flows
        assert det2._reference_mv is not None

    def test_save_creates_file(self, detector, tmp_path):
        path = str(tmp_path / "state2.pkl")
        detector.process_flow(make_features(0, dim=6))
        detector.save_state(path)
        import os
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Isolation Forest integration
# ---------------------------------------------------------------------------

class TestIsolationForestGate:

    def test_with_iso_forest_enabled(self, small_engine):
        det = AnomalyDetector(
            ga_engine=small_engine,
            window_size=300,
            reframe_interval=50,
            use_isolation_forest=True,
            isolation_forest_contamination=0.1,
        )
        rng = np.random.default_rng(20)
        results = []
        for i in range(50):
            feat = rng.uniform(0, 1, size=6)
            results.append(det.process_flow(feat))
        assert len(results) == 50
        assert all(math.isfinite(r.score) for r in results)
