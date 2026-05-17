# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
geoidslib/algebra/ga_ops_cy.pyx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cython-accelerated inner loops for Geometric Algebra operations.

This module provides drop-in replacements for the hot paths in
multivector.py, giving 10–50x speedups over pure Python:

  * cy_geometric_product  — C-level loop over blade pairs
  * cy_outer_product      — grade-filtered wedge product
  * cy_embed_grade1       — fast grade-1 embedding
  * cy_blade_distance     — commutator norm (anomaly score)
  * cy_euclidean_distance — L2 on sparse blade dicts

Build
-----
  python setup_cy.py build_ext --inplace

The pure-Python fallback (multivector.py) is used automatically when
the compiled extension is not available.
"""

import cython
from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as np

# ─── Type aliases ───────────────────────────────────────────────────────────

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t   INT32


# ─── Grade-1 embedding ──────────────────────────────────────────────────────

def cy_embed_grade1(
    double[:] features,
    double    threshold,
    dict      out_blades,
) -> None:
    """
    Fast grade-1 embedding: write non-negligible feature values into out_blades.

    Parameters
    ----------
    features     : 1-D double array of normalised feature values
    threshold    : sparsity threshold — skip |v| < threshold
    out_blades   : dict to populate in-place  {(i,): float}
    """
    cdef int i
    cdef double v
    cdef int n = features.shape[0]

    for i in range(n):
        v = features[i]
        if v > threshold or v < -threshold:
            out_blades[(i,)] = v


# ─── Geometric product ───────────────────────────────────────────────────────

def cy_geometric_product(
    dict   blades_a,
    dict   blades_b,
    double[:] metric,
    int    max_grade,
    double threshold,
) -> dict:
    """
    Sparse geometric product of two multivectors.

    Returns
    -------
    dict {BladeKey: float} — the result blades (pruned).
    """
    cdef dict result = {}
    cdef list key_a_list, key_b_list, working
    cdef tuple key_a, key_b, blade_key
    cdef double coeff_a, coeff_b, sign, contrib, metric_val
    cdef int idx, pos, n, i

    for key_a, coeff_a in blades_a.items():
        for key_b, coeff_b in blades_b.items():
            # Clifford product in C-level loop
            working = list(key_a)
            sign = 1.0

            for idx in key_b:
                if idx in working:
                    pos = working.index(idx)
                    sign *= (-1.0) ** (len(working) - 1 - pos)
                    sign *= metric[idx]
                    working.remove(idx)
                else:
                    working.append(idx)
                    n = len(working) - 1
                    while n > 0 and working[n] < working[n - 1]:
                        working[n], working[n - 1] = working[n - 1], working[n]
                        sign *= -1.0
                        n -= 1

            blade_key = tuple(working)
            if len(blade_key) > max_grade:
                continue

            contrib = sign * coeff_a * coeff_b
            if blade_key in result:
                result[blade_key] = result[blade_key] + contrib
            else:
                result[blade_key] = contrib

    # Prune
    return {k: v for k, v in result.items() if v > threshold or v < -threshold}


# ─── Outer product ───────────────────────────────────────────────────────────

def cy_outer_product(
    dict   blades_a,
    dict   blades_b,
    int    max_grade,
    double threshold,
) -> dict:
    """
    Sparse outer (wedge) product A ∧ B.
    Only grade-(|A|+|B|) terms survive; skips overlapping indices.
    """
    cdef dict result = {}
    cdef tuple key_a, key_b, merged
    cdef double coeff_a, coeff_b, sign, contrib
    cdef int count, i
    cdef list la, lb

    for key_a, coeff_a in blades_a.items():
        for key_b, coeff_b in blades_b.items():
            # No index overlap → wedge is non-zero
            la = list(key_a)
            lb = list(key_b)
            overlap = False
            for idx in lb:
                if idx in la:
                    overlap = True
                    break
            if overlap:
                continue

            merged = tuple(sorted(la + lb))
            if len(merged) > max_grade:
                continue

            # Count sign-swaps
            count = 0
            for i in range(len(la) - 1, -1, -1):
                for jj in lb:
                    if la[i] > jj:
                        count += 1
            sign = 1.0 if count % 2 == 0 else -1.0
            contrib = sign * coeff_a * coeff_b

            if merged in result:
                result[merged] = result[merged] + contrib
            else:
                result[merged] = contrib

    return {k: v for k, v in result.items() if v > threshold or v < -threshold}


# ─── Norms and distances ─────────────────────────────────────────────────────

def cy_euclidean_distance(dict blades_a, dict blades_b) -> double:
    """
    L2 distance between two sparse blade dicts.
    Iterates over the union of keys.
    """
    cdef double total = 0.0
    cdef double diff
    cdef set all_keys

    all_keys = set(blades_a.keys()) | set(blades_b.keys())
    for k in all_keys:
        diff = blades_a.get(k, 0.0) - blades_b.get(k, 0.0)
        total += diff * diff
    return sqrt(total)


def cy_norm_squared(dict blades, double[:] metric) -> double:
    """
    Scalar product ⟨A, Ã⟩₀ = ‖A‖²  via reverse.
    For grade-k blade e_{i1...ik} with metric:
        ⟨e_I, ẽ_I⟩₀ = (-1)^{k(k-1)/2} * ∏ metric[iⱼ]
    """
    cdef double total = 0.0
    cdef double blade_norm_sq
    cdef int grade, i
    cdef double sign

    for key, coeff in blades.items():
        grade = len(key)
        sign = 1.0 if (grade * (grade - 1) // 2) % 2 == 0 else -1.0
        blade_norm_sq = 1.0
        for i in key:
            blade_norm_sq *= metric[i]
        total += sign * blade_norm_sq * coeff * coeff

    return total


def cy_blade_distance(
    dict   blades_a,
    dict   blades_b,
    double[:] metric,
    int    max_grade,
    double threshold,
) -> double:
    """
    Commutator-based blade distance:
      dist = ‖[A, B]‖ / (‖A‖ · ‖B‖)
    Uses cy_geometric_product internally.
    """
    # [A, B] = (AB - BA) / 2
    ab = cy_geometric_product(blades_a, blades_b, metric, max_grade, threshold)
    ba = cy_geometric_product(blades_b, blades_a, metric, max_grade, threshold)

    cdef dict comm = {}
    all_keys = set(ab.keys()) | set(ba.keys())
    for k in all_keys:
        v = 0.5 * (ab.get(k, 0.0) - ba.get(k, 0.0))
        if v > threshold or v < -threshold:
            comm[k] = v

    cdef double comm_norm = sqrt(fabs(cy_norm_squared(comm, metric)))
    cdef double norm_a    = sqrt(fabs(cy_norm_squared(blades_a, metric)))
    cdef double norm_b    = sqrt(fabs(cy_norm_squared(blades_b, metric)))
    cdef double denom     = norm_a * norm_b

    if denom < 1e-12:
        return 0.0
    return comm_norm / denom


# ─── Batch embedding ─────────────────────────────────────────────────────────

def cy_embed_batch(
    double[:, :] feature_matrix,
    list         biv_keys,
    list         tri_keys,
    double       threshold,
    int          max_grade,
) -> list:
    """
    Embed a batch of feature vectors into blade dicts.

    Parameters
    ----------
    feature_matrix : (n_flows, dim) float64 array
    biv_keys       : list of (i, j) tuples — precomputed bivector keys
    tri_keys       : list of (i, j, k) tuples
    threshold      : sparsity threshold
    max_grade      : maximum blade grade

    Returns
    -------
    list of dicts — one blade dict per flow
    """
    cdef int n = feature_matrix.shape[0]
    cdef int dim = feature_matrix.shape[1]
    cdef int row, i, j, k
    cdef double vi, vj, vk, val
    cdef dict blades
    cdef list results = []

    for row in range(n):
        blades = {}

        # Grade-1
        for i in range(dim):
            vi = feature_matrix[row, i]
            if vi > threshold or vi < -threshold:
                blades[(i,)] = vi

        # Grade-2 (bivectors)
        if max_grade >= 2:
            for (i, j) in biv_keys:
                vi = feature_matrix[row, i]
                vj = feature_matrix[row, j]
                val = vi * vj
                if val > threshold or val < -threshold:
                    blades[(i, j)] = val

        # Grade-3 (trivectors)
        if max_grade >= 3:
            for (i, j, k) in tri_keys:
                vi = feature_matrix[row, i]
                vj = feature_matrix[row, j]
                vk = feature_matrix[row, k]
                val = vi * vj * vk
                if val > threshold or val < -threshold:
                    blades[(i, j, k)] = val

        results.append(blades)

    return results
