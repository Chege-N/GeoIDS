"""
tests/test_multivector.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the SparseMultivector implementation.
"""

import math

import numpy as np
import pytest

from geoidslib.algebra.multivector import SparseMultivector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metric_3d():
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def mv_e1(metric_3d):
    """Grade-1 basis vector e1."""
    mv = SparseMultivector(dim=3, metric=metric_3d)
    mv.blades[(0,)] = 1.0
    return mv


@pytest.fixture
def mv_e2(metric_3d):
    """Grade-1 basis vector e2."""
    mv = SparseMultivector(dim=3, metric=metric_3d)
    mv.blades[(1,)] = 1.0
    return mv


@pytest.fixture
def mv_e3(metric_3d):
    mv = SparseMultivector(dim=3, metric=metric_3d)
    mv.blades[(2,)] = 1.0
    return mv


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_from_vector(self):
        vec = np.array([1.0, 2.0, 3.0])
        mv = SparseMultivector.from_vector(vec)
        assert mv.blades[(0,)] == pytest.approx(1.0)
        assert mv.blades[(1,)] == pytest.approx(2.0)
        assert mv.blades[(2,)] == pytest.approx(3.0)
        assert len(mv.blades) == 3

    def test_scalar(self):
        mv = SparseMultivector.scalar(5.0, dim=3)
        assert mv.blades[()] == pytest.approx(5.0)

    def test_sparsity_threshold_pruning(self):
        vec = np.array([1.0, 1e-15, 3.0])
        mv = SparseMultivector.from_vector(vec, sparsity_threshold=1e-9)
        assert (1,) not in mv.blades  # 1e-15 < threshold → pruned

    def test_grade_part(self, mv_e1):
        mv_e1.blades[(0, 1)] = 0.5  # add bivector
        grade1 = mv_e1.grade_part(1)
        assert (0,) in grade1.blades
        assert (0, 1) not in grade1.blades

    def test_vector_part(self):
        vec = np.array([3.0, 0.0, -1.0])
        mv = SparseMultivector.from_vector(vec)
        result = mv.vector_part()
        np.testing.assert_array_almost_equal(result, vec)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:

    def test_add(self, mv_e1, mv_e2):
        result = mv_e1 + mv_e2
        assert result.blades[(0,)] == pytest.approx(1.0)
        assert result.blades[(1,)] == pytest.approx(1.0)

    def test_sub(self, mv_e1, mv_e2):
        result = mv_e1 - mv_e2
        assert result.blades[(0,)] == pytest.approx(1.0)
        assert result.blades[(1,)] == pytest.approx(-1.0)

    def test_scalar_mul(self, mv_e1):
        result = mv_e1 * 3.0
        assert result.blades[(0,)] == pytest.approx(3.0)

    def test_neg(self, mv_e1):
        result = -mv_e1
        assert result.blades[(0,)] == pytest.approx(-1.0)

    def test_add_cancels(self, mv_e1):
        result = mv_e1 + (-mv_e1)
        assert (0,) not in result.blades or abs(result.blades.get((0,), 0)) < 1e-9

    def test_div(self, mv_e1):
        result = mv_e1 / 2.0
        assert result.blades[(0,)] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Clifford geometric product
# ---------------------------------------------------------------------------

class TestGeometricProduct:

    def test_e1_squared_euclidean(self, mv_e1):
        """e1 * e1 = 1 in Euclidean metric."""
        result = mv_e1.geometric_product(mv_e1)
        assert result.scalar_part() == pytest.approx(1.0)

    def test_e1_squared_minkowski(self):
        """e1 * e1 = -1 in Minkowski metric."""
        metric = np.array([-1.0, 1.0, 1.0])
        mv = SparseMultivector(dim=3, metric=metric)
        mv.blades[(0,)] = 1.0
        result = mv.geometric_product(mv)
        assert result.scalar_part() == pytest.approx(-1.0)

    def test_e1_e2_gives_bivector(self, mv_e1, mv_e2):
        """e1 * e2 = e12 (bivector)."""
        result = mv_e1.geometric_product(mv_e2)
        assert (0, 1) in result.blades
        assert result.blades[(0, 1)] == pytest.approx(1.0)
        assert result.scalar_part() == pytest.approx(0.0)

    def test_anticommutativity(self, mv_e1, mv_e2):
        """e1 * e2 = -e2 * e1."""
        e12 = mv_e1.geometric_product(mv_e2)
        e21 = mv_e2.geometric_product(mv_e1)
        assert e12.blades.get((0, 1), 0) == pytest.approx(-e21.blades.get((0, 1), 0))

    def test_reverse(self, mv_e1, mv_e2):
        """Reverse of e12 = -e12."""
        e12 = mv_e1.geometric_product(mv_e2)
        rev = e12.reverse()
        assert rev.blades.get((0, 1), 0) == pytest.approx(-e12.blades.get((0, 1), 0))


# ---------------------------------------------------------------------------
# Outer product
# ---------------------------------------------------------------------------

class TestOuterProduct:

    def test_e1_wedge_e2(self, mv_e1, mv_e2):
        result = mv_e1.outer_product(mv_e2)
        assert (0, 1) in result.blades

    def test_e1_wedge_e1_is_zero(self, mv_e1):
        """e1 ∧ e1 = 0."""
        result = mv_e1.outer_product(mv_e1)
        assert len(result.blades) == 0 or all(abs(v) < 1e-9 for v in result.blades.values())

    def test_grade_increases(self, mv_e1, mv_e2, mv_e3):
        e12 = mv_e1.outer_product(mv_e2)
        e123 = e12.outer_product(mv_e3)
        assert (0, 1, 2) in e123.blades


# ---------------------------------------------------------------------------
# Norms and distances
# ---------------------------------------------------------------------------

class TestNorms:

    def test_norm_unit_vector(self, mv_e1):
        assert mv_e1.norm() == pytest.approx(1.0)

    def test_norm_scaled(self, mv_e1):
        scaled = mv_e1 * 3.0
        assert scaled.norm() == pytest.approx(3.0)

    def test_euclidean_distance_zero(self, mv_e1):
        assert mv_e1.euclidean_distance(mv_e1) == pytest.approx(0.0)

    def test_euclidean_distance_orthogonal(self, mv_e1, mv_e2):
        dist = mv_e1.euclidean_distance(mv_e2)
        assert dist == pytest.approx(math.sqrt(2))

    def test_blade_distance_self_zero(self, mv_e1):
        # Commutator [A, A] = 0 → distance = 0
        assert mv_e1.blade_distance(mv_e1) == pytest.approx(0.0, abs=1e-9)

    def test_scalar_product_unit(self, mv_e1):
        assert mv_e1.scalar_product(mv_e1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:

    def test_roundtrip(self):
        vec = np.array([1.5, -2.0, 0.0, 4.0, 0.1])
        metric = np.array([1.0, 1.0, -1.0, 1.0, -1.0])
        mv = SparseMultivector.from_vector(vec, metric=metric)
        d = mv.to_dict()
        mv2 = SparseMultivector.from_dict(d)
        for key in mv.blades:
            assert mv.blades[key] == pytest.approx(mv2.blades.get(key, 0.0))


# ---------------------------------------------------------------------------
# Blade attribution
# ---------------------------------------------------------------------------

class TestAttribution:

    def test_dominant_blades(self):
        mv = SparseMultivector(dim=5, metric=np.ones(5))
        mv.blades[(0,)] = 10.0
        mv.blades[(1,)] = 5.0
        mv.blades[(2,)] = 1.0
        top = mv.dominant_blades(top_k=2)
        assert top[0][0] == (0,)
        assert top[1][0] == (1,)

    def test_anomalous_blades(self):
        dim = 4
        metric = np.ones(dim)
        mv_ref = SparseMultivector(dim=dim, metric=metric)
        mv_ref.blades[(0,)] = 1.0

        mv_anom = SparseMultivector(dim=dim, metric=metric)
        mv_anom.blades[(0,)] = 5.0  # large deviation

        anomalies = mv_anom.anomalous_blades(mv_ref, threshold=2.0)
        assert len(anomalies) >= 1
        assert anomalies[0][0] == (0,)
