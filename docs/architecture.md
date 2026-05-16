# GeoIDS Architecture

## Overview

GeoIDS processes network flows through four stages:

```
[Ingestion] → [Feature Extraction] → [GA Embedding] → [Detection]
```

---

## Stage 1: Ingestion

The `FlowIngester` unifies four data sources:

| Source | Driver | Throughput |
|---|---|---|
| PCAP files | dpkt | ~200 k pkt/s |
| Live capture | nfstream (dpdk) | ~10 M pkt/s |
| NetFlow v5/v9/IPFIX | nfstream | ~500 k flows/s |
| Zeek logs | built-in parser | ~1 M lines/s |

A `collections.deque(maxlen=N)` ring buffer decouples ingestion from detection.

---

## Stage 2: Feature Extraction

25 features per flow:

### Continuous (indices 0–14)
These are normalised using an online min-max scaler that tracks running min/max.

| Index | Feature | Notes |
|---|---|---|
| 0 | packet_count | Raw integer |
| 1 | byte_count | |
| 2 | mean_packet_size | |
| 3 | std_packet_size | Burstiness proxy |
| 4 | mean_iat | Inter-arrival time |
| 5 | std_iat | |
| 6 | flow_duration | |
| 7 | packet_length_entropy | Shannon entropy |
| 8 | iat_entropy | |
| 9 | tcp_flags_ratio | (SYN+FIN+RST)/packets |
| 10 | bytes_per_second | |
| 11 | packets_per_second | |
| 12 | tls_handshake_duration | |
| 13 | tls_cert_size | |
| 14 | tls_num_extensions | |

### Categorical / TLS (indices 15–24)
These use fixed normalisation mappings.

| Index | Feature | Encoding |
|---|---|---|
| 15 | tls_sni_length | raw int |
| 16 | tls_cipher_suite_id | / 65535 |
| 17 | tls_version | 1.0→0, 1.3→1 |
| 18 | tls_num_san | |
| 19 | tls_cert_lifetime_days | log(1+x) |
| 20 | protocol | TCP=0, UDP=0.5 |
| 21 | dst_port_category | well-known=0 |
| 22 | src_ip_entropy | octet entropy |
| 23 | dst_ip_class | A=0, D=1 |
| 24 | flow_direction_ratio | fwd/total |

---

## Stage 3: Geometric Algebra Embedding

### Basis and Metric

We use a Conformal Geometric Algebra over ℝ^(p,q) where p=15, q=10.

**Metric signature**: `diag(+1,...,+1, -1,...,-1)` (p positive, q negative entries).

The metric encodes the semantic distinction between continuous features (Euclidean geometry) and categorical features (hyperbolic geometry).

### Multivector Structure

For a feature vector **x** = (x₀, x₁, …, x₂₄), the multivector is:

```
M(x) = Σᵢ xᵢ eᵢ                          (grade-1: raw features)
     + Σᵢ<ⱼ xᵢxⱼ eᵢⱼ                      (grade-2: pairwise correlations)
     + Σᵢ<ⱼ<ₖ xᵢxⱼxₖ eᵢⱼₖ                (grade-3: three-way interactions)
```

Only blades in a predefined dictionary (O(d²) pairs, O(d³) triplets) are computed — not all C(25,2) = 300 bivectors or C(25,3) = 2300 trivectors.

### Sparse Storage

Blades with |coeff| < ε (default 1e-6) are discarded. In practice, each multivector stores O(50–150) non-zero blades.

Memory per multivector: ~1–4 KB (vs. 2^25 × 8 = 256 MB dense).

---

## Stage 4: Anomaly Detection

### Reference Multiframe

Every `reframe_interval` normal flows, the reference is recomputed via Grassmannian gradient descent (Weiszfeld algorithm in multivector space):

```
M_ref^(t+1) = (Σᵢ λⁿ⁻ⁱ Mᵢ / d(M_ref^(t), Mᵢ)) / (Σᵢ λⁿ⁻ⁱ / d(…))
```

where λ is the forgetting factor (default 0.99).

### Anomaly Score

```
score(M_flow) = ‖[M_flow, M_ref]‖ / (‖M_flow‖ · ‖M_ref‖)
```

The commutator `[A, B] = (AB - BA)/2` captures non-commutativity — how much the flow's multivector "disagrees" structurally with the reference.

### Dynamic Threshold

The score distribution is monitored online. The tail (top 10%) is fitted to a Generalised Pareto Distribution (GPD). The threshold is set at the GPD quantile corresponding to the target FPR (default 1%).

GPD parameters are refitted every 100 flows.

### Normal Window Gating

An Isolation Forest (IF) is trained on the last N normal-flow feature vectors. When a new flow is classified as an outlier by the IF, it is **not** added to the normal window (even if its blade score is below threshold). This prevents adversarial drift of the reference multiframe.

---

## Explainability

When a flow is flagged, the blade attribution module identifies which blades deviate most from the reference:

```python
anomalous_blades = {key: |M_flow[key] - M_ref[key]| for key in blades}
sorted_desc = sorted(anomalous_blades.items(), key=lambda x: x[1], reverse=True)
```

Each blade key maps to a human-readable description via the blade dictionary:

| Blade | Description |
|---|---|
| e₀ | packet_count |
| e₁₂ | mean_packet_size ∧ mean_iat |
| e₇₈ | packet_length_entropy ∧ iat_entropy |
| e₁₃₁₄₁₅ | cert_size ∧ extensions ∧ sni_length |

---

## Performance Notes

### Python baseline
- Feature extraction: ~500 k flows/s
- GA embedding (grade ≤ 2): ~50 k multivectors/s
- End-to-end: ~30 k flows/s

### With Cython/C++ (roadmap)
- GA operations in C++ (Eigen): target ~5 M multivectors/s
- End-to-end target: 100 k–1 M flows/s

### Scaling
For 100 Gbps networks, run multiple GeoIDS processes with flow-hash-based partitioning (e.g., 8 processes × 12.5 Gbps each).
