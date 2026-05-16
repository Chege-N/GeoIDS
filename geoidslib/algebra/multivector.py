"""
geoidslib.algebra.multivector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sparse multivector representation in Conformal Geometric Algebra (CGA).

Design goals
------------
* Only store blades with |coefficient| > threshold  →  O(d³) memory for order-3 approx.
* Blades are identified by a *basis blade key*: a frozenset of basis-vector indices.
* All arithmetic is lazy / sparse — we never materialise the 2^d dense vector.

Mathematical background
-----------------------
For a CGA over ℝ^(p,q) the full multivector space has dimension 2^(p+q).
Here p = number of continuous features, q = number of categorical features.
We truncate at grade k=3 (scalar + vectors + bivectors + trivectors).

Notation
--------
  e_i         grade-1 blade (basis vector i)
  e_ij = e_i∧e_j  grade-2 blade (outer product)
  e_ijk        grade-3 blade
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Tuple

import numpy as np

# Type alias: a blade key is a sorted tuple of basis-vector indices
BladeKey = Tuple[int, ...]


def _blade_key(*indices: int) -> BladeKey:
    """Canonical (sorted) tuple key for a blade."""
    return tuple(sorted(set(indices)))


@dataclass
class SparseMultivector:
    """
    A sparse multivector in Conformal Geometric Algebra.

    Attributes
    ----------
    dim : int
        Number of basis vectors (= p + q).
    metric : np.ndarray
        Diagonal metric signature array, shape (dim,).
        e.g. [1]*p + [-1]*q for ℝ^(p,q).
    blades : dict
        {BladeKey: float} — non-zero blade coefficients.
    sparsity_threshold : float
        Blades with |coeff| < this value are dropped.
    max_grade : int
        Maximum blade grade stored (default 3).
    """

    dim: int
    metric: np.ndarray
    blades: Dict[BladeKey, float] = field(default_factory=dict)
    sparsity_threshold: float = 1e-9
    max_grade: int = 3

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_vector(
        cls,
        vec: np.ndarray,
        metric: np.ndarray | None = None,
        sparsity_threshold: float = 1e-9,
        max_grade: int = 3,
    ) -> "SparseMultivector":
        """
        Embed a real feature vector as a grade-1 multivector (pure vector part).

        Parameters
        ----------
        vec : array of shape (d,)
        metric : diagonal metric, defaults to Euclidean (+1, +1, …)
        """
        d = len(vec)
        if metric is None:
            metric = np.ones(d)
        mv = cls(dim=d, metric=metric, sparsity_threshold=sparsity_threshold, max_grade=max_grade)
        for i, v in enumerate(vec):
            if abs(v) >= sparsity_threshold:
                mv.blades[_blade_key(i)] = float(v)
        return mv

    @classmethod
    def scalar(cls, value: float, dim: int, metric: np.ndarray | None = None) -> "SparseMultivector":
        """Create a scalar (grade-0) multivector."""
        if metric is None:
            metric = np.ones(dim)
        mv = cls(dim=dim, metric=metric)
        if abs(value) >= mv.sparsity_threshold:
            mv.blades[()] = float(value)
        return mv

    # ------------------------------------------------------------------
    # Grade extraction
    # ------------------------------------------------------------------

    def grade_part(self, k: int) -> "SparseMultivector":
        """Return the grade-k part of this multivector."""
        mv = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )
        mv.blades = {key: coeff for key, coeff in self.blades.items() if len(key) == k}
        return mv

    def scalar_part(self) -> float:
        """Return the grade-0 (scalar) part."""
        return self.blades.get((), 0.0)

    def vector_part(self) -> np.ndarray:
        """Return grade-1 coefficients as a dense array."""
        out = np.zeros(self.dim)
        for key, coeff in self.blades.items():
            if len(key) == 1:
                out[key[0]] = coeff
        return out

    def grades_present(self) -> set:
        return {len(k) for k in self.blades}

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "SparseMultivector") -> "SparseMultivector":
        result = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )
        result.blades = dict(self.blades)
        for key, coeff in other.blades.items():
            result.blades[key] = result.blades.get(key, 0.0) + coeff
        result._prune()
        return result

    def __sub__(self, other: "SparseMultivector") -> "SparseMultivector":
        return self.__add__(other.__neg__())

    def __neg__(self) -> "SparseMultivector":
        mv = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )
        mv.blades = {k: -v for k, v in self.blades.items()}
        return mv

    def __mul__(self, scalar: float) -> "SparseMultivector":
        """Scalar multiplication."""
        mv = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )
        mv.blades = {k: v * scalar for k, v in self.blades.items()}
        mv._prune()
        return mv

    def __rmul__(self, scalar: float) -> "SparseMultivector":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "SparseMultivector":
        return self.__mul__(1.0 / scalar)

    # ------------------------------------------------------------------
    # Geometric product (sparse implementation)
    # ------------------------------------------------------------------

    def geometric_product(self, other: "SparseMultivector") -> "SparseMultivector":
        """
        Compute the geometric product A * B using the metric-aware Clifford rule:
            e_i * e_i = metric[i]
            e_i * e_j = e_ij  (i ≠ j, anti-commutes for Clifford)
        Only blades up to max_grade are retained.
        """
        result = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )

        for key_a, coeff_a in self.blades.items():
            for key_b, coeff_b in other.blades.items():
                blade_key, sign = _clifford_product(key_a, key_b, self.metric)
                if blade_key is None:
                    continue  # zero result
                grade = len(blade_key)
                if grade > self.max_grade:
                    continue
                contrib = sign * coeff_a * coeff_b
                result.blades[blade_key] = result.blades.get(blade_key, 0.0) + contrib

        result._prune()
        return result

    def outer_product(self, other: "SparseMultivector") -> "SparseMultivector":
        """
        Wedge (outer) product: A ∧ B.
        Only grade-(|A|+|B|) terms survive.
        """
        result = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )

        for key_a, coeff_a in self.blades.items():
            for key_b, coeff_b in other.blades.items():
                # Outer product only survives when no index overlap
                if set(key_a) & set(key_b):
                    continue
                merged = tuple(sorted(key_a + key_b))
                if len(merged) > self.max_grade:
                    continue
                sign = _count_swaps(key_a, key_b)
                contrib = sign * coeff_a * coeff_b
                result.blades[merged] = result.blades.get(merged, 0.0) + contrib

        result._prune()
        return result

    def inner_product(self, other: "SparseMultivector") -> "SparseMultivector":
        """Left contraction (inner product) A ⌋ B."""
        result = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )
        for key_a, coeff_a in self.blades.items():
            for key_b, coeff_b in other.blades.items():
                grade_a, grade_b = len(key_a), len(key_b)
                if grade_a > grade_b:
                    continue
                # Result grade = grade_b - grade_a
                blade_key, sign = _clifford_product(key_a, key_b, self.metric)
                if blade_key is None:
                    continue
                if len(blade_key) != grade_b - grade_a:
                    continue
                contrib = sign * coeff_a * coeff_b
                result.blades[blade_key] = result.blades.get(blade_key, 0.0) + contrib
        result._prune()
        return result

    def commutator(self, other: "SparseMultivector") -> "SparseMultivector":
        """Commutator product [A, B] = (AB - BA) / 2."""
        ab = self.geometric_product(other)
        ba = other.geometric_product(self)
        return (ab - ba) * 0.5

    def reverse(self) -> "SparseMultivector":
        """Reverse of A: reverses order of each basis blade."""
        mv = SparseMultivector(
            dim=self.dim,
            metric=self.metric.copy(),
            sparsity_threshold=self.sparsity_threshold,
            max_grade=self.max_grade,
        )
        for key, coeff in self.blades.items():
            grade = len(key)
            sign = (-1) ** (grade * (grade - 1) // 2)
            mv.blades[key] = coeff * sign
        return mv

    # ------------------------------------------------------------------
    # Norms and distances
    # ------------------------------------------------------------------

    def scalar_product(self, other: "SparseMultivector") -> float:
        """
        Scalar product ⟨A, B⟩ = ⟨A B̃⟩₀
        where B̃ is the reverse of B.
        """
        return self.geometric_product(other.reverse()).scalar_part()

    def norm_squared(self) -> float:
        return self.scalar_product(self)

    def norm(self) -> float:
        ns = self.norm_squared()
        return math.sqrt(abs(ns))

    def blade_distance(self, other: "SparseMultivector") -> float:
        """
        Blade-based anomaly distance between two multivectors.
        Uses the norm of the commutator: ‖[A, B]‖ / (‖A‖ · ‖B‖)
        Normalised to [0, 1].
        """
        comm = self.commutator(other)
        comm_norm = comm.norm()
        denom = self.norm() * other.norm()
        if denom < 1e-12:
            return 0.0
        return comm_norm / denom

    def euclidean_distance(self, other: "SparseMultivector") -> float:
        """L2 distance on blade coefficients (as sparse vectors)."""
        keys = set(self.blades) | set(other.blades)
        total = 0.0
        for k in keys:
            diff = self.blades.get(k, 0.0) - other.blades.get(k, 0.0)
            total += diff * diff
        return math.sqrt(total)

    # ------------------------------------------------------------------
    # Blade attribution (explainability)
    # ------------------------------------------------------------------

    def dominant_blades(self, top_k: int = 5) -> list[tuple[BladeKey, float]]:
        """Return the top-k blades by absolute coefficient magnitude."""
        items = sorted(self.blades.items(), key=lambda x: abs(x[1]), reverse=True)
        return items[:top_k]

    def anomalous_blades(
        self,
        reference: "SparseMultivector",
        threshold: float,
    ) -> list[tuple[BladeKey, float]]:
        """
        Return blades where |self[key] - reference[key]| > threshold.
        Used for explainability: which blade dimension triggered the alert.
        """
        anomalies = []
        all_keys = set(self.blades) | set(reference.blades)
        for key in all_keys:
            diff = abs(self.blades.get(key, 0.0) - reference.blades.get(key, 0.0))
            if diff > threshold:
                anomalies.append((key, diff))
        return sorted(anomalies, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "metric": self.metric.tolist(),
            "blades": {str(k): v for k, v in self.blades.items()},
            "max_grade": self.max_grade,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SparseMultivector":
        import ast

        mv = cls(
            dim=d["dim"],
            metric=np.array(d["metric"]),
            max_grade=d.get("max_grade", 3),
        )
        mv.blades = {tuple(ast.literal_eval(k)): v for k, v in d["blades"].items()}
        return mv

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove blades with negligible coefficients."""
        self.blades = {k: v for k, v in self.blades.items() if abs(v) >= self.sparsity_threshold}

    def __repr__(self) -> str:
        terms = []
        for key in sorted(self.blades, key=lambda k: (len(k), k)):
            coeff = self.blades[key]
            if key == ():
                terms.append(f"{coeff:.4f}")
            else:
                label = "e" + "".join(str(i) for i in key)
                terms.append(f"{coeff:.4f}·{label}")
        if not terms:
            return "0"
        return " + ".join(terms)

    def __len__(self) -> int:
        return len(self.blades)


# ---------------------------------------------------------------------------
# Low-level Clifford product helpers
# ---------------------------------------------------------------------------


def _clifford_product(
    key_a: BladeKey,
    key_b: BladeKey,
    metric: np.ndarray,
) -> tuple[BladeKey | None, float]:
    """
    Compute the geometric product of two basis blades.

    Returns
    -------
    (result_key, sign)  or  (None, 0) if the product is zero.
    """
    # Work with lists for manipulation
    result = list(key_a)
    sign = 1.0

    for idx in key_b:
        if idx in result:
            # e_i * e_i = metric[i]
            pos = result.index(idx)
            # Count swaps to bring idx to position pos
            sign *= (-1) ** (len(result) - 1 - pos)
            sign *= metric[idx]
            result.remove(idx)
        else:
            # Append and bubble into sorted position
            result.append(idx)
            pos = len(result) - 1
            while pos > 0 and result[pos] < result[pos - 1]:
                result[pos], result[pos - 1] = result[pos - 1], result[pos]
                sign *= -1
                pos -= 1

    return tuple(result), sign


def _count_swaps(key_a: BladeKey, key_b: BladeKey) -> float:
    """
    Count the number of transpositions needed to merge two sorted sequences.
    Returns the sign (+1 or -1) for the outer product.
    """
    count = 0
    for i, a in enumerate(reversed(key_a)):
        for b in key_b:
            if a > b:
                count += 1
    return (-1) ** count
