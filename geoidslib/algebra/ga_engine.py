"""
geoidslib.algebra.ga_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-level Geometric Algebra Engine for GeoIDS.

Responsibilities
----------------
* Maintain the metric signature for the CGA.
* Embed feature vectors as multivectors (with auto-generated bivector/trivector blades).
* Compute the reference multiframe (GA mean) over a sliding window.
* Provide Grassmannian gradient-descent iteration for the GA median.
* Expose a blade attribution dictionary for explainability.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from geoidslib.algebra.multivector import SparseMultivector

logger = logging.getLogger(__name__)

# Default blade dictionary for 25-dimensional CGA
# Maps blade key → human-readable description
# Indices 0-14: continuous features
# Indices 15-24: categorical / TLS features
_DEFAULT_BLADE_DICT: dict[tuple, str] = {
    # Grade-1 blades (raw features)
    (0,): "packet_count",
    (1,): "byte_count",
    (2,): "mean_packet_size",
    (3,): "std_packet_size",
    (4,): "mean_iat",
    (5,): "std_iat",
    (6,): "flow_duration",
    (7,): "packet_length_entropy",
    (8,): "iat_entropy",
    (9,): "tcp_flags_ratio",
    (10,): "bytes_per_second",
    (11,): "packets_per_second",
    (12,): "tls_handshake_duration",
    (13,): "tls_cert_size",
    (14,): "tls_num_extensions",
    (15,): "tls_sni_length",
    (16,): "tls_cipher_suite_id",
    (17,): "tls_version",
    (18,): "tls_num_san",
    (19,): "tls_cert_lifetime_days",
    (20,): "protocol",
    (21,): "dst_port_category",
    (22,): "src_ip_entropy",
    (23,): "dst_ip_class",
    (24,): "flow_direction_ratio",
    # Grade-2 blades (pairwise correlations — selected key pairs)
    (0, 1): "packet_count ∧ byte_count",
    (2, 4): "mean_packet_size ∧ mean_iat",
    (3, 5): "std_packet_size ∧ std_iat",
    (7, 8): "packet_length_entropy ∧ iat_entropy",
    (12, 14): "tls_handshake_duration ∧ tls_num_extensions",
    (13, 14): "tls_cert_size ∧ tls_num_extensions",
    (15, 16): "tls_sni_length ∧ tls_cipher_suite_id",
    (10, 11): "bytes_per_second ∧ packets_per_second",
    (6, 7): "flow_duration ∧ packet_length_entropy",
    (4, 7): "mean_iat ∧ packet_length_entropy",
    (0, 10): "packet_count ∧ bytes_per_second",
    (9, 24): "tcp_flags_ratio ∧ flow_direction_ratio",
    # Grade-3 blades (three-way interactions)
    (2, 4, 7): "mean_packet_size ∧ mean_iat ∧ entropy",
    (13, 14, 15): "cert_size ∧ extensions ∧ sni_length",
    (0, 10, 11): "packet_count ∧ bps ∧ pps",
}


class GeometricAlgebraEngine:
    """
    Central engine for Conformal Geometric Algebra operations in GeoIDS.

    Parameters
    ----------
    dim : int
        Dimension of the CGA (number of basis vectors = p + q).
        Default 25 (matching the 25-feature space).
    p : int
        Number of positive-metric (continuous) basis vectors.
    q : int
        Number of negative-metric (categorical) basis vectors.
    max_grade : int
        Maximum blade grade stored (default 3).
    sparsity_threshold : float
        Blades with |coefficient| below this are pruned.
    blade_dict : dict | None
        Human-readable blade descriptions. Uses default if None.
    """

    def __init__(
        self,
        dim: int = 25,
        p: int = 15,
        q: int = 10,
        max_grade: int = 3,
        sparsity_threshold: float = 1e-6,
        blade_dict: dict[tuple, str] | None = None,
    ):
        assert p + q == dim, "p + q must equal dim"
        self.dim = dim
        self.p = p
        self.q = q
        self.max_grade = max_grade
        self.sparsity_threshold = sparsity_threshold

        # Build metric signature: +1 for continuous, -1 for categorical
        self.metric = np.array([1.0] * p + [-1.0] * q)

        self.blade_dict: dict[tuple, str] = blade_dict or _DEFAULT_BLADE_DICT

        logger.info(
            "GeometricAlgebraEngine initialised: dim=%d, p=%d, q=%d, max_grade=%d",
            dim, p, q, max_grade,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        features: np.ndarray,
        include_bivectors: bool = True,
        include_trivectors: bool = True,
    ) -> SparseMultivector:
        """
        Embed a feature vector into the CGA as a sparse multivector.

        The grade-1 part is the raw feature vector.
        Grade-2 blades are added for selected pairwise products (outer product).
        Grade-3 blades are added for selected triplets.

        Parameters
        ----------
        features : np.ndarray, shape (dim,)
            Normalised feature vector.
        include_bivectors : bool
            Add e_ij = e_i ∧ e_j blades for pairs in the blade_dict.
        include_trivectors : bool
            Add e_ijk blades for triplets in the blade_dict.

        Returns
        -------
        SparseMultivector
        """
        assert len(features) == self.dim, f"Expected {self.dim} features, got {len(features)}"

        mv = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )

        # Grade-1 part
        for i, v in enumerate(features):
            if abs(v) >= self.sparsity_threshold:
                mv.blades[(i,)] = float(v)

        if include_bivectors and self.max_grade >= 2:
            for key in self.blade_dict:
                if len(key) == 2:
                    i, j = key
                    if i >= self.dim or j >= self.dim:
                        continue
                    val = features[i] * features[j]
                    if abs(val) >= self.sparsity_threshold:
                        mv.blades[key] = float(val)

        if include_trivectors and self.max_grade >= 3:
            for key in self.blade_dict:
                if len(key) == 3:
                    i, j, k = key
                    if i >= self.dim or j >= self.dim or k >= self.dim:
                        continue
                    val = features[i] * features[j] * features[k]
                    if abs(val) >= self.sparsity_threshold:
                        mv.blades[key] = float(val)

        return mv

    # ------------------------------------------------------------------
    # Reference multiframe (GA mean)
    # ------------------------------------------------------------------

    def compute_ga_mean(
        self, multivectors: Sequence[SparseMultivector], forgetting_factor: float = 1.0
    ) -> SparseMultivector:
        """
        Compute an arithmetic mean in the multivector space.
        (The true GA mean via Grassmannian gradient descent is in `compute_ga_median`.)

        Parameters
        ----------
        multivectors : list of SparseMultivector
        forgetting_factor : float
            When < 1, recent samples are weighted more (exponential decay).
        """
        if not multivectors:
            raise ValueError("Cannot compute mean of empty sequence")

        n = len(multivectors)
        # Accumulate with forgetting weights
        weights = [forgetting_factor ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)

        result = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )

        for mv, w in zip(multivectors, weights, strict=False):
            scaled = mv * (w / total_w)
            result = result + scaled

        return result

    def compute_ga_median(
        self,
        multivectors: Sequence[SparseMultivector],
        max_iter: int = 50,
        tol: float = 1e-4,
        forgetting_factor: float = 0.99,
    ) -> SparseMultivector:
        """
        Compute the geometric algebra Fréchet median via Grassmannian gradient descent.

        Algorithm
        ---------
        1. Initialise with the arithmetic mean.
        2. At each step, compute the weighted sum of "logarithmic maps" from the
           current estimate to each sample.
        3. Update the estimate via exponential map.
        4. Repeat until convergence.

        For the sparse representation we approximate the Riemannian ops by
        a Weiszfeld-style iteration on blade coefficients.
        """
        if not multivectors:
            raise ValueError("Empty sequence")

        n = len(multivectors)
        weights = np.array([forgetting_factor ** (n - 1 - i) for i in range(n)], dtype=float)
        weights /= weights.sum()

        # Initialise at weighted mean
        current = self.compute_ga_mean(multivectors, forgetting_factor)

        for iteration in range(max_iter):
            numerator = SparseMultivector(
                dim=self.dim,
                metric=self.metric.copy(),
                sparsity_threshold=self.sparsity_threshold,
                max_grade=self.max_grade,
            )
            denom = 0.0

            for mv, w in zip(multivectors, weights, strict=False):
                dist = current.euclidean_distance(mv)
                if dist < 1e-12:
                    continue
                inv_dist = w / dist
                numerator = numerator + mv * inv_dist
                denom += inv_dist

            if denom < 1e-12:
                break

            new_estimate = numerator * (1.0 / denom)
            delta = current.euclidean_distance(new_estimate)
            current = new_estimate

            if delta < tol:
                logger.debug("GA median converged at iteration %d (delta=%.2e)", iteration, delta)
                break

        return current

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def anomaly_score(
        self,
        flow_mv: SparseMultivector,
        reference_mv: SparseMultivector,
        use_commutator: bool = True,
    ) -> float:
        """
        Compute the blade-based anomaly score for a flow multivector.

        Score = ‖[flow, reference]‖ / (‖flow‖ · ‖reference‖) if use_commutator
              = euclidean_distance otherwise.

        Returns a non-negative float.
        """
        if use_commutator:
            return flow_mv.blade_distance(reference_mv)
        else:
            return flow_mv.euclidean_distance(reference_mv)

    def explain_anomaly(
        self,
        flow_mv: SparseMultivector,
        reference_mv: SparseMultivector,
        threshold: float = 0.1,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Return human-readable explanation of the anomaly.

        Returns
        -------
        list of dicts with keys: blade_key, deviation, description
        """
        anomalies = flow_mv.anomalous_blades(reference_mv, threshold)[:top_k]
        results = []
        for key, deviation in anomalies:
            desc = self.blade_dict.get(key, f"blade_{'_'.join(str(i) for i in key)}")
            results.append({
                "blade_key": key,
                "blade_label": "e" + "".join(str(i) for i in key) if key else "scalar",
                "deviation": deviation,
                "description": desc,
            })
        return results

    # ------------------------------------------------------------------
    # Blade coefficient vector (for ML / PCA)
    # ------------------------------------------------------------------

    def to_coefficient_vector(
        self, mv: SparseMultivector, keys: list[tuple] | None = None
    ) -> np.ndarray:
        """
        Flatten multivector blade coefficients to a dense numpy array.
        Useful for PCA / visualisation.

        Parameters
        ----------
        keys : list of blade keys to extract (in order).
               If None, uses all keys in blade_dict.
        """
        if keys is None:
            keys = sorted(self.blade_dict.keys(), key=lambda k: (len(k), k))
        return np.array([mv.blades.get(k, 0.0) for k in keys])
