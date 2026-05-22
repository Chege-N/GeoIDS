"""
benchmarks/bench_throughput.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Throughput benchmark for GeoIDS.

Measures:
1. Feature extraction rate (flows/s)
2. GA embedding rate (multivectors/s)
3. Anomaly detection rate (flows/s)
4. End-to-end pipeline rate (flows/s)

Usage
-----
python benchmarks/bench_throughput.py
  or
pytest benchmarks/bench_throughput.py --benchmark-sort=mean
"""

import time
import os
import numpy as np
import pytest

from geoidslib.algebra.ga_engine import GeometricAlgebraEngine
from geoidslib.detection.detector import AnomalyDetector
from geoidslib.features.extractor import FeatureExtractor, FlowRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_flows(n: int, seed: int = 0) -> list[FlowRecord]:
    rng = np.random.default_rng(seed)
    flows = []
    for _ in range(n):
        flows.append(FlowRecord(
            src_ip=f"10.0.{rng.integers(0,255)}.{rng.integers(1,254)}",
            dst_ip="8.8.8.8",
            src_port=int(rng.integers(49152, 65535)),
            dst_port=443,
            protocol=6,
            packet_count=int(rng.integers(5, 50)),
            byte_count=int(rng.integers(500, 50_000)),
            packet_sizes=[int(rng.integers(40, 1500)) for _ in range(10)],
            inter_arrival_times=[float(rng.uniform(0.001, 0.1)) for _ in range(9)],
            flow_start=0.0,
            flow_end=1.0,
            tls_version=1.3,
            tls_cert_size=2048,
            tls_num_extensions=5,
        ))
    return flows


def make_feature_matrix(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, size=(n, 25))


# ---------------------------------------------------------------------------
# Standalone benchmarks (no pytest-benchmark dependency)
# ---------------------------------------------------------------------------

def bench(name: str, fn, n_runs: int = 3, n_items: int = 10_000):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    fps = n_items / best
    print(f"  {name:<40} {fps:>12,.0f} flows/s   (best of {n_runs}: {best*1000:.1f} ms)")


def run_benchmarks():
    N = int(os.getenv("BENCH_N_FLOWS", 10_000))
    if os.getenv("CI"):
      N = min(N, 500)
    print(f"\n{'='*65}")
    print(f"  GeoIDS Throughput Benchmark  (n_flows={n_flows:,} flows)")
    print(f"{'='*65}")

    # --- Feature extraction ---
    flows = make_flows(n_flows)
    extractor = FeatureExtractor(online_normalise=False)
    bench("FeatureExtractor.extract_raw",
          lambda: [extractor.extract_raw(f) for f in flows],
          n_items=n_flows)

    extractor_norm = FeatureExtractor(online_normalise=True)
    bench("FeatureExtractor.extract (normalised)",
          lambda: [extractor_norm.extract(f) for f in flows],
          n_items=n_flows)

    # --- GA embedding ---
    engine = GeometricAlgebraEngine(dim=25, p=15, q=10, max_grade=2)
    feat_matrix = make_feature_matrix(n_flows)
    bench(
        "GAEngine.embed (grade ≤ 2)",
        lambda: [engine.embed(feat_matrix[i]) for i in range(n_flows)],
        n_items=n_flows,
    )

    engine3 = GeometricAlgebraEngine(dim=25, p=15, q=10, max_grade=3)
    bench(
        "GAEngine.embed (grade ≤ 3)",
        lambda: [engine3.embed(feat_matrix[i]) for i in range(n_flows)],
        n_items=n_flows,
    )

    # --- Anomaly scoring ---
    det = AnomalyDetector(
        ga_engine=engine,
        window_size=1000,
        reframe_interval=200,
        use_isolation_forest=False,
        target_fpr=0.05,
    )
    # Warmup
    for i in range(200):
        det.process_flow(feat_matrix[i])
    det.force_recompute_reference()

    bench(
        "AnomalyDetector.process_flow",
        lambda: [det.process_flow(feat_matrix[i % n_flows]) for i in range(n_flows)],
        n_items=n_flows,
    )

    # --- Batch processing ---
    bench(
        "AnomalyDetector.process_batch (n_flows=1000)",
        lambda: det.process_batch(feat_matrix[:1000]),
        n_runs=5,
        n_items=1000,
    )

    # --- Geometric product ---
    mv_list = [engine.embed(feat_matrix[i]) for i in range(100)]
    ref_mv = mv_list[0]
    bench(
        "SparseMultivector.geometric_product",
        lambda: [mv_list[i % 100].geometric_product(ref_mv) for i in range(n_flows)],
        n_items=n_flows,
    )

    bench(
        "SparseMultivector.commutator",
        lambda: [mv_list[i % 100].commutator(ref_mv) for i in range(n_flows)],
        n_items=n_flows,
    )

    print("\n  Target: 100,000 flows/s (with Cython/C++ acceleration)")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# pytest-benchmark variants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_engine():
    return GeometricAlgebraEngine(dim=25, p=15, q=10, max_grade=2)


@pytest.fixture(scope="module")
def shared_features():
    return make_feature_matrix(1000)


def test_bench_embed(benchmark, shared_engine, shared_features):
    benchmark(lambda: shared_engine.embed(shared_features[0]))


def test_bench_process_flow(benchmark, shared_engine):
    det = AnomalyDetector(
        ga_engine=shared_engine,
        use_isolation_forest=False,
        reframe_interval=50,
    )
    feat = np.random.default_rng(0).uniform(0, 1, size=25)
    # Warmup
    for _ in range(50):
        det.process_flow(feat)
    benchmark(lambda: det.process_flow(feat))


def test_bench_geometric_product(benchmark, shared_engine, shared_features):
    mv1 = shared_engine.embed(shared_features[0])
    mv2 = shared_engine.embed(shared_features[1])
    benchmark(lambda: mv1.geometric_product(mv2))


def test_bench_blade_distance(benchmark, shared_engine, shared_features):
    mv1 = shared_engine.embed(shared_features[0])
    mv2 = shared_engine.embed(shared_features[1])
    benchmark(lambda: mv1.blade_distance(mv2))


if __name__ == "__main__":
    run_benchmarks()
