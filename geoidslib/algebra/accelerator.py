"""
geoidslib/algebra/accelerator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Transparent acceleration layer.

At import time this module tries to load the compiled Cython extension
(ga_ops_cy).  If it's available, the hot-path functions from multivector.py
are monkey-patched to use the C-level implementations.

If the extension is absent, the pure-Python fallback is used silently.

Usage
-----
Import this module early in your application:

    from geoidslib.algebra.accelerator import acceleration_status
    print(acceleration_status())   # "Cython (C++)" or "Pure Python"

GeoIDS.from_config() calls this automatically.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_USING_CYTHON = False
_STATUS = "Pure Python"


def _try_load_cython() -> bool:
    """Attempt to import the compiled extension and patch hot paths."""
    global _USING_CYTHON, _STATUS

    try:
        from geoidslib.algebra import ga_ops_cy  # type: ignore[import]
        from geoidslib.algebra import multivector as mv_module

        # ── Patch SparseMultivector hot paths ──────────────────────────

        original_gp = mv_module.SparseMultivector.geometric_product
        original_op = mv_module.SparseMultivector.outer_product
        original_ed = mv_module.SparseMultivector.euclidean_distance
        original_bd = mv_module.SparseMultivector.blade_distance

        def _cy_geometric_product(self, other):
            result_blades = ga_ops_cy.cy_geometric_product(
                self.blades, other.blades, self.metric,
                self.max_grade, self.sparsity_threshold,
            )
            out = mv_module.SparseMultivector(
                dim=self.dim, metric=self.metric.copy(),
                sparsity_threshold=self.sparsity_threshold,
                max_grade=self.max_grade,
            )
            out.blades = result_blades
            return out

        def _cy_outer_product(self, other):
            result_blades = ga_ops_cy.cy_outer_product(
                self.blades, other.blades,
                self.max_grade, self.sparsity_threshold,
            )
            out = mv_module.SparseMultivector(
                dim=self.dim, metric=self.metric.copy(),
                sparsity_threshold=self.sparsity_threshold,
                max_grade=self.max_grade,
            )
            out.blades = result_blades
            return out

        def _cy_euclidean_distance(self, other):
            return ga_ops_cy.cy_euclidean_distance(self.blades, other.blades)

        def _cy_blade_distance(self, other):
            return ga_ops_cy.cy_blade_distance(
                self.blades, other.blades, self.metric,
                self.max_grade, self.sparsity_threshold,
            )

        mv_module.SparseMultivector.geometric_product = _cy_geometric_product
        mv_module.SparseMultivector.outer_product     = _cy_outer_product
        mv_module.SparseMultivector.euclidean_distance = _cy_euclidean_distance
        mv_module.SparseMultivector.blade_distance    = _cy_blade_distance

        # ── Patch GeometricAlgebraEngine batch embed ───────────────────

        from geoidslib.algebra import ga_engine as eng_module
        import numpy as np

        original_embed = eng_module.GeometricAlgebraEngine.embed

        def _cy_embed(self, features, include_bivectors=True, include_trivectors=True):
            assert len(features) == self.dim
            biv_keys = [k for k in self.blade_dict if len(k) == 2
                        and all(i < self.dim for i in k)] if include_bivectors else []
            tri_keys = [k for k in self.blade_dict if len(k) == 3
                        and all(i < self.dim for i in k)] if include_trivectors else []
            fm = np.asarray(features, dtype=np.float64).reshape(1, -1)
            batch = ga_ops_cy.cy_embed_batch(
                fm, biv_keys, tri_keys,
                self.sparsity_threshold, self.max_grade,
            )
            out = mv_module.SparseMultivector(
                dim=self.dim, metric=self.metric.copy(),
                sparsity_threshold=self.sparsity_threshold,
                max_grade=self.max_grade,
            )
            out.blades = batch[0]
            return out

        eng_module.GeometricAlgebraEngine.embed = _cy_embed

        _USING_CYTHON = True
        _STATUS = "Cython (C++)"
        logger.info("Cython GA acceleration loaded successfully.")
        return True

    except ImportError:
        logger.debug("Cython GA extension not found — using pure Python.")
        return False


# Attempt load at module import time
_try_load_cython()


def acceleration_status() -> str:
    """Return a human-readable string describing the active backend."""
    return _STATUS


def is_accelerated() -> bool:
    """Return True if the Cython extension is active."""
    return _USING_CYTHON


def force_python_fallback() -> None:
    """
    Disable Cython patches and revert to pure Python.
    Useful for testing or debugging.
    """
    global _USING_CYTHON, _STATUS
    if not _USING_CYTHON:
        return

    from geoidslib.algebra import multivector as mv_module
    from geoidslib.algebra.multivector import (
        SparseMultivector as _SV,
    )

    # Reload the original methods from a fresh class definition
    import importlib
    mod = importlib.reload(mv_module)

    _USING_CYTHON = False
    _STATUS = "Pure Python (forced)"
    logger.info("Reverted to pure-Python GA backend.")
