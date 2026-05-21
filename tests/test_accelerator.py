"""
tests/test_accelerator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for geoidslib.algebra.accelerator.

These run in the pure-Python path (Cython extension not compiled in CI)
but verify the loader contract: it should never crash, should report
status correctly, and force_python_fallback() should work.
"""

import pytest

from geoidslib.algebra import accelerator


class TestAcceleratorLoader:

    def test_acceleration_status_returns_string(self):
        status = accelerator.acceleration_status()
        assert isinstance(status, str)
        assert len(status) > 0

    def test_is_accelerated_bool(self):
        result = accelerator.is_accelerated()
        assert isinstance(result, bool)

    def test_status_matches_is_accelerated(self):
        if accelerator.is_accelerated():
            assert "Cython" in accelerator.acceleration_status()
        else:
            assert "Python" in accelerator.acceleration_status()

    def test_force_python_fallback_no_crash(self):
        """force_python_fallback() must not raise even if already Python."""
        accelerator.force_python_fallback()
        assert not accelerator.is_accelerated()
        assert "Python" in accelerator.acceleration_status()

    def test_multivector_still_works_after_fallback(self):
        """Core GA operations must work after forcing Python fallback."""
        accelerator.force_python_fallback()
        import numpy as np

        from geoidslib.algebra.ga_engine import GeometricAlgebraEngine

        engine = GeometricAlgebraEngine(dim=6, p=4, q=2, max_grade=2)
        features = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6])
        mv = engine.embed(features)
        score = engine.anomaly_score(mv, mv)
        assert score == pytest.approx(0.0, abs=1e-9)
