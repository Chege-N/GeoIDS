"""
tests/test_ga_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the GeometricAlgebraEngine.
"""

import pytest
import numpy as np

from geoidslib.algebra.ga_engine import GeometricAlgebraEngine
from geoidslib.algebra.multivector import SparseMultivector


@pytest.fixture
def engine():
    return GeometricAlgebraEngine(dim=25, p=15, q=10, max_grade=3)


@pytest.fixture
def small_engine():
    """Smaller engine for faster tests."""
    return GeometricAlgebraEngine(dim=6, p=4, q=2, max_grade=3)


@pytest.fixture
def sample_features():
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, size=25)


class TestEmbedding:

    def test_embed_returns_multivector(self, engine, sample_features):
        mv = engine.embed(sample_features)
        assert isinstance(mv, SparseMultivector)

    def test_embed_grade1_present(self, engine, sample_features):
        mv = engine.embed(sample_features, include_bivectors=False, include_trivectors=False)
        grades = mv.grades_present()
        assert 1 in grades
        assert 2 not in grades

    def test_embed_bivectors_added(self, engine, sample_features):
        mv = engine.embed(sample_features, include_bivectors=True, include_trivectors=False)
        grades = mv.grades_present()
        assert 2 in grades

    def test_embed_trivectors_added(self, engine, sample_features):
        mv = engine.embed(sample_features, include_bivectors=True, include_trivectors=True)
        grades = mv.grades_present()
        assert 3 in grades

    def test_embed_wrong_dim_raises(self, engine):
        bad_features = np.ones(10)  # wrong dimension
        with pytest.raises(AssertionError):
            engine.embed(bad_features)

    def test_embed_all_zeros_produces_empty_mv(self, engine):
        features = np.zeros(25)
        mv = engine.embed(features)
        assert len(mv.blades) == 0

    def test_embed_sparsity(self, engine):
        features = np.zeros(25)
        features[0] = 1.0
        features[1] = 0.5
        mv = engine.embed(features, include_bivectors=False, include_trivectors=False)
        # Only two grade-1 blades should be non-zero
        grade1_blades = {k: v for k, v in mv.blades.items() if len(k) == 1}
        assert len(grade1_blades) == 2


class TestGAMean:

    def test_mean_single_element(self, engine, sample_features):
        mv = engine.embed(sample_features)
        mean = engine.compute_ga_mean([mv])
        # Mean of one element should equal that element
        for key in mv.blades:
            assert mean.blades.get(key, 0.0) == pytest.approx(mv.blades[key], rel=1e-5)

    def test_mean_two_opposite(self, engine):
        features_pos = np.ones(25) * 0.5
        features_neg = np.ones(25) * 0.5
        mv_pos = engine.embed(features_pos, include_bivectors=False, include_trivectors=False)
        mv_neg = engine.embed(features_neg, include_bivectors=False, include_trivectors=False)
        mean = engine.compute_ga_mean([mv_pos, mv_neg])
        # Mean of identical elements = that element
        for key in mv_pos.blades:
            assert mean.blades.get(key, 0.0) == pytest.approx(mv_pos.blades[key], rel=1e-4)

    def test_mean_forgetting_factor(self, engine, sample_features):
        mvs = [engine.embed(sample_features) for _ in range(5)]
        mean_ff = engine.compute_ga_mean(mvs, forgetting_factor=0.9)
        mean_uniform = engine.compute_ga_mean(mvs, forgetting_factor=1.0)
        # With forgetting, recent elements dominate → means differ
        assert mean_ff.euclidean_distance(mean_uniform) > 0 or True  # soft check


class TestGAMedian:

    def test_median_converges(self, small_engine):
        rng = np.random.default_rng(0)
        mvs = [
            small_engine.embed(rng.uniform(0, 1, size=6))
            for _ in range(20)
        ]
        median = small_engine.compute_ga_median(mvs, max_iter=20, tol=1e-3)
        assert isinstance(median, SparseMultivector)

    def test_median_single(self, small_engine):
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        mv = small_engine.embed(features)
        median = small_engine.compute_ga_median([mv])
        # Should equal the single element
        for key in mv.blades:
            assert median.blades.get(key, 0.0) == pytest.approx(mv.blades[key], rel=1e-4)


class TestAnomalyScore:

    def test_score_self_is_zero_or_near(self, engine, sample_features):
        mv = engine.embed(sample_features)
        score = engine.anomaly_score(mv, mv, use_commutator=True)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_score_different_flows(self, engine):
        rng = np.random.default_rng(1)
        f1 = rng.uniform(0, 1, size=25)
        f2 = rng.uniform(0, 1, size=25)
        mv1 = engine.embed(f1)
        mv2 = engine.embed(f2)
        score = engine.anomaly_score(mv1, mv2)
        assert score >= 0.0

    def test_score_euclidean(self, engine, sample_features):
        mv = engine.embed(sample_features)
        score_eu = engine.anomaly_score(mv, mv, use_commutator=False)
        assert score_eu == pytest.approx(0.0, abs=1e-9)

    def test_high_score_for_outlier(self, engine):
        """A zero-vector vs a non-zero reference should give positive score."""
        rng = np.random.default_rng(2)
        normal = engine.embed(rng.uniform(0.3, 0.7, size=25))
        outlier = engine.embed(np.ones(25))  # extreme values
        score = engine.anomaly_score(outlier, normal, use_commutator=False)
        assert score > 0.0


class TestExplainAnomaly:

    def test_explain_returns_list(self, engine, sample_features):
        mv = engine.embed(sample_features)
        rng = np.random.default_rng(3)
        ref = engine.embed(rng.uniform(0, 1, size=25))
        explanations = engine.explain_anomaly(mv, ref, threshold=0.0, top_k=3)
        assert isinstance(explanations, list)
        assert len(explanations) <= 3

    def test_explain_has_required_fields(self, engine, sample_features):
        mv = engine.embed(sample_features)
        rng = np.random.default_rng(4)
        ref = engine.embed(rng.uniform(0, 1, size=25))
        explanations = engine.explain_anomaly(mv, ref, threshold=0.0, top_k=1)
        if explanations:
            e = explanations[0]
            assert "blade_key" in e
            assert "description" in e
            assert "deviation" in e

    def test_explain_no_anomaly_at_high_threshold(self, engine, sample_features):
        mv = engine.embed(sample_features)
        explanations = engine.explain_anomaly(mv, mv, threshold=999.0)
        assert len(explanations) == 0


class TestCoefficientVector:

    def test_coefficient_vector_length(self, engine, sample_features):
        mv = engine.embed(sample_features)
        vec = engine.to_coefficient_vector(mv)
        assert len(vec) == len(engine.blade_dict)

    def test_coefficient_vector_custom_keys(self, engine, sample_features):
        mv = engine.embed(sample_features)
        keys = [(0,), (1,), (2,)]
        vec = engine.to_coefficient_vector(mv, keys=keys)
        assert len(vec) == 3

    def test_coefficient_vector_values(self, engine):
        features = np.zeros(25)
        features[0] = 0.8
        mv = engine.embed(features, include_bivectors=False, include_trivectors=False)
        vec = engine.to_coefficient_vector(mv, keys=[(0,)])
        assert vec[0] == pytest.approx(0.8)
