"""
geoidslib.detection.detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Anomaly detection engine for GeoIDS.

Pipeline
--------
1. Receive a feature vector (already extracted + normalised).
2. Embed as a SparseMultivector using GeometricAlgebraEngine.
3. Compute the blade-based anomaly score vs. the reference multiframe.
4. Apply a secondary Isolation Forest gate (for online normal-window updates).
5. Update the reference multiframe (with forgetting factor) only for normal flows.
6. Compare score to a dynamic threshold calibrated via Generalised Pareto Distribution.
7. Return an AnomalyResult (score, alert bool, blade explanation).
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import genpareto
from sklearn.ensemble import IsolationForest

from geoidslib.algebra.ga_engine import GeometricAlgebraEngine
from geoidslib.algebra.multivector import SparseMultivector

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single flow."""

    flow_id: str
    timestamp: float
    score: float
    is_anomaly: bool
    threshold: float
    blade_explanations: list[dict] = field(default_factory=list)
    top_blade_label: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "flow_id": self.flow_id,
            "timestamp": self.timestamp,
            "score": round(self.score, 6),
            "is_anomaly": self.is_anomaly,
            "threshold": round(self.threshold, 6),
            "confidence": round(self.confidence, 4),
            "top_blade": self.top_blade_label,
            "blade_explanations": [
                {
                    "blade": e.get("blade_label", ""),
                    "description": e.get("description", ""),
                    "deviation": round(e.get("deviation", 0.0), 6),
                }
                for e in self.blade_explanations
            ],
        }


class GPDThreshold:
    """
    Dynamic anomaly threshold using the Generalised Pareto Distribution (GPD).

    The GPD is fitted to the tail of the observed anomaly score distribution.
    Threshold = u + σ/ξ * ((α * n)^(-ξ) - 1)  where u is the tail threshold,
    n is the number of exceedances, and α is the target false positive rate.

    Parameters
    ----------
    tail_fraction : float
        Fraction of scores considered "tail" for GPD fitting (default 0.1).
    target_fpr : float
        Target false positive rate (default 0.01).
    min_samples : int
        Minimum samples before GPD fitting (uses empirical percentile before then).
    refit_every : int
        Re-fit GPD every N new scores.
    """

    def __init__(
        self,
        tail_fraction: float = 0.10,
        target_fpr: float = 0.01,
        min_samples: int = 200,
        refit_every: int = 100,
    ):
        self.tail_fraction = tail_fraction
        self.target_fpr = target_fpr
        self.min_samples = min_samples
        self.refit_every = refit_every

        self._scores: list[float] = []
        self._threshold: float = float("inf")
        self._gpd_params: tuple[float, float, float] | None = None  # c, loc, scale
        self._since_last_fit = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    def update(self, score: float) -> float:
        """Add a new score and return the current threshold."""
        self._scores.append(score)
        self._since_last_fit += 1

        n = len(self._scores)
        if n < self.min_samples:
            # Use empirical (1 - target_fpr) quantile
            self._threshold = float(np.percentile(self._scores, (1 - self.target_fpr) * 100))
            return self._threshold

        if self._since_last_fit >= self.refit_every:
            self._fit_gpd()
            self._since_last_fit = 0

        return self._threshold

    def _fit_gpd(self) -> None:
        arr = np.array(self._scores, dtype=float)
        u = np.percentile(arr, (1 - self.tail_fraction) * 100)
        exceedances = arr[arr > u] - u

        if len(exceedances) < 10:
            self._threshold = np.percentile(arr, (1 - self.target_fpr) * 100)
            return

        try:
            c, loc, scale = genpareto.fit(exceedances, floc=0)
            self._gpd_params = (c, loc, scale)
            n = len(arr)
            n_u = len(exceedances)
            # GPD quantile for target FPR
            if c == 0:
                q = u + scale * (-np.log(self.target_fpr * n / n_u))
            else:
                q = u + (scale / c) * ((self.target_fpr * n / n_u) ** (-c) - 1)
            self._threshold = float(np.clip(q, 0, arr.max() * 2))
            logger.debug(
                "GPD fitted: c=%.4f scale=%.4f threshold=%.6f", c, scale, self._threshold
            )
        except Exception as exc:
            logger.warning("GPD fit failed: %s — using empirical percentile", exc)
            self._threshold = float(np.percentile(arr, (1 - self.target_fpr) * 100))


class AnomalyDetector:
    """
    Main anomaly detection engine for GeoIDS.

    Parameters
    ----------
    ga_engine : GeometricAlgebraEngine
    window_size : int
        Number of recent normal flows to keep in the sliding window.
    reframe_interval : int
        Recompute the reference multiframe every N flows.
    forgetting_factor : float
        GA median forgetting factor (0 < λ ≤ 1).
    use_isolation_forest : bool
        Gate normal-window updates with Isolation Forest.
    isolation_forest_contamination : float
        IF contamination parameter.
    gpd_tail_fraction : float
        Tail fraction for GPD threshold fitting.
    target_fpr : float
        Target false positive rate.
    use_commutator_score : bool
        Use commutator-based blade distance (True) or Euclidean distance (False).
    explanation_threshold : float
        Minimum blade deviation to include in explanation.
    """

    def __init__(
        self,
        ga_engine: GeometricAlgebraEngine | None = None,
        window_size: int = 10_000,
        reframe_interval: int = 500,
        forgetting_factor: float = 0.99,
        use_isolation_forest: bool = True,
        isolation_forest_contamination: float = 0.05,
        gpd_tail_fraction: float = 0.10,
        target_fpr: float = 0.01,
        use_commutator_score: bool = True,
        explanation_threshold: float = 0.05,
    ):
        self.ga_engine = ga_engine or GeometricAlgebraEngine()
        self.window_size = window_size
        self.reframe_interval = reframe_interval
        self.forgetting_factor = forgetting_factor
        self.use_isolation_forest = use_isolation_forest
        self.use_commutator_score = use_commutator_score
        self.explanation_threshold = explanation_threshold

        # Sliding window of normal flow multivectors
        self._normal_window: collections.deque[SparseMultivector] = \
            collections.deque(maxlen=window_size)
        # Feature vectors for Isolation Forest
        self._if_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=window_size)

        # Reference multiframe (GA median of normal window)
        self._reference_mv: SparseMultivector | None = None
        self._flows_since_reframe = 0

        # Online threshold estimator
        self.threshold_estimator = GPDThreshold(
            tail_fraction=gpd_tail_fraction,
            target_fpr=target_fpr,
        )

        # Isolation Forest (lazy initialisation)
        self._iso_forest: IsolationForest | None = None
        self._iso_contamination = isolation_forest_contamination
        self._iso_refit_every = 2000
        self._iso_since_fit = 0

        self._total_flows = 0
        self._total_alerts = 0

        logger.info("AnomalyDetector initialised")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_flow(
        self,
        features: np.ndarray,
        flow_id: str = "",
        timestamp: float | None = None,
    ) -> AnomalyResult:
        """
        Process a single normalised feature vector and return an AnomalyResult.

        Parameters
        ----------
        features : np.ndarray, shape (25,)
        flow_id : str  (e.g. "src:dst:sport:dport:proto:timestamp")
        timestamp : float | None  (Unix time; uses time.time() if None)
        """
        if timestamp is None:
            timestamp = time.time()

        self._total_flows += 1

        # 1. Embed as multivector
        mv = self.ga_engine.embed(features)

        # 2. Compute anomaly score vs reference
        if self._reference_mv is None:
            score = 0.0
        else:
            score = self.ga_engine.anomaly_score(mv, self._reference_mv, self.use_commutator_score)

        # 3. Update dynamic threshold
        current_threshold = self.threshold_estimator.update(score)

        # 4. Is this an anomaly?
        is_anomaly = score > current_threshold

        # 5. Confidence (how far above threshold)
        confidence = min(score / current_threshold, 5.0) / 5.0 if current_threshold > 1e-12 else 0.0

        # 6. Update normal window (only for non-anomalous flows)
        if not is_anomaly:
            is_if_normal = self._iso_forest_predict(features)
            if is_if_normal:
                self._normal_window.append(mv)
                self._if_buffer.append(features)
                self._flows_since_reframe += 1

        # 7. Periodically retrain isolation forest and recompute reference
        self._maybe_refit_iso_forest()
        self._maybe_recompute_reference()

        # 8. Build explanation
        explanations = []
        top_label = ""
        if is_anomaly and self._reference_mv is not None:
            explanations = self.ga_engine.explain_anomaly(
                mv, self._reference_mv, self.explanation_threshold
            )
            top_label = explanations[0]["description"] if explanations else "unknown"
            self._total_alerts += 1

        result = AnomalyResult(
            flow_id=flow_id,
            timestamp=timestamp,
            score=score,
            is_anomaly=is_anomaly,
            threshold=current_threshold,
            blade_explanations=explanations,
            top_blade_label=top_label,
            confidence=confidence,
        )

        return result

    def process_batch(
        self,
        features_batch: np.ndarray,
        flow_ids: list[str] | None = None,
    ) -> list[AnomalyResult]:
        """
        Process a batch of feature vectors.

        Parameters
        ----------
        features_batch : np.ndarray, shape (n, 25)
        flow_ids : list of str | None

        Returns
        -------
        list of AnomalyResult
        """
        n = len(features_batch)
        if flow_ids is None:
            flow_ids = [f"flow_{i}" for i in range(n)]

        results = []
        for i in range(n):
            res = self.process_flow(features_batch[i], flow_id=flow_ids[i])
            results.append(res)
        return results

    # ------------------------------------------------------------------
    # Reference multiframe management
    # ------------------------------------------------------------------

    def _maybe_recompute_reference(self) -> None:
        if self._flows_since_reframe < self.reframe_interval:
            return
        if len(self._normal_window) < 10:
            return

        window_list = list(self._normal_window)
        # Use a subsample for speed in large windows
        if len(window_list) > 1000:
            indices = np.random.choice(len(window_list), 1000, replace=False)
            window_list = [window_list[i] for i in sorted(indices)]

        logger.debug("Recomputing reference multiframe from %d normal flows", len(window_list))
        self._reference_mv = self.ga_engine.compute_ga_median(
            window_list,
            forgetting_factor=self.forgetting_factor,
        )
        self._flows_since_reframe = 0

    def force_recompute_reference(self) -> None:
        """Manually trigger reference multiframe recomputation."""
        self._flows_since_reframe = self.reframe_interval
        self._maybe_recompute_reference()

    # ------------------------------------------------------------------
    # Isolation Forest gate
    # ------------------------------------------------------------------

    def _iso_forest_predict(self, features: np.ndarray) -> bool:
        """
        Returns True if the Isolation Forest classifies the flow as normal.
        If the forest is not yet fitted, returns True (optimistic).
        """
        if not self.use_isolation_forest or self._iso_forest is None:
            return True
        pred = self._iso_forest.predict(features.reshape(1, -1))
        return int(pred[0]) == 1  # 1 = inlier, -1 = outlier

    def _maybe_refit_iso_forest(self) -> None:
        if not self.use_isolation_forest:
            return
        self._iso_since_fit += 1
        if self._iso_since_fit < self._iso_refit_every:
            return
        if len(self._if_buffer) < 100:
            return

        x_data = np.vstack(list(self._if_buffer))
        self._iso_forest = IsolationForest(
            n_estimators=100,
            contamination=self._iso_contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._iso_forest.fit(x_data)
        self._iso_since_fit = 0
        logger.debug("Isolation Forest refitted on %d samples", len(x_data))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        return {
            "total_flows": self._total_flows,
            "total_alerts": self._total_alerts,
            "alert_rate": self._total_alerts / max(self._total_flows, 1),
            "normal_window_size": len(self._normal_window),
            "current_threshold": self.threshold_estimator.threshold,
            "reference_mv_blades": len(self._reference_mv.blades) if self._reference_mv else 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        import pickle
        state = {
            "reference_mv": self._reference_mv,
            "threshold_estimator": self.threshold_estimator,
            "iso_forest": self._iso_forest,
            "total_flows": self._total_flows,
            "total_alerts": self._total_alerts,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("AnomalyDetector state saved to %s", path)

    def load_state(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._reference_mv = state["reference_mv"]
        self.threshold_estimator = state["threshold_estimator"]
        self._iso_forest = state.get("iso_forest")
        self._total_flows = state.get("total_flows", 0)
        self._total_alerts = state.get("total_alerts", 0)
        logger.info("AnomalyDetector state loaded from %s", path)
