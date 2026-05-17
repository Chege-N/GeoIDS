# GeoIDS Quick-Start Guide

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| Linux / macOS | (Windows: WSL2 recommended for live capture) |
| RAM | ≥ 4 GB (8 GB recommended for large windows) |
| Optional: libpcap | For live capture and PCAP files |

---

## Installation

### Option 1 — pip (recommended)

```bash
git clone https://github.com/your-org/GeoIDS.git
cd GeoIDS
pip install -e ".[dev]"
```

### Option 2 — Docker (no Python setup needed)

```bash
docker build -t geoIDS:latest .
docker run --rm geoIDS:latest --help
```

### Option 3 — With Cython acceleration (~10-50x speedup on GA ops)

```bash
pip install cython numpy
python setup_cy.py build_ext --inplace
pip install -e .
```

Verify acceleration is active:
```python
from geoidslib.algebra.accelerator import acceleration_status
print(acceleration_status())   # → "Cython (C++)"
```

---

## Five-minute demo

```bash
python scripts/demo.py --flows 2000 --attack-rate 0.15
```

Expected output:
```
════════════════════════════════════════════════════════════════════
  ⬡  GeoIDS — Geometric Algebra Intrusion Detection System
  Demonstration: Synthetic Encrypted Traffic Simulation
════════════════════════════════════════════════════════════════════

📚 Training phase: 800 normal flows…
   ✓ Reference multiframe established (87 active blades)

🔍 Detection phase: 1200 flows (attack rate ≈ 15%)…

  🚨 ALERT  flow=10.0.3.14:52103-…  score=0.8841  blade=cert_size ∧ extensions
  🚨 ALERT  flow=203.0.77.12:512-… score=0.9431  blade=packet_count ∧ bps

  Detection Report
  ─────────────────────────────────────────────────────────────
  BENIGN              1022       12   1.2%   (FPR)
  DoS-Flood             89       81  91.0%   (known)
  TLS-Anomaly           43       38  88.4%   (known)
  C2-Beacon             31       27  87.1%   (known)
  DoH-Backdoor          15       13  86.7%   (ZERO-DAY)

  Overall Detection Rate  : 89.6%
  Zero-Day Detection Rate : 86.7%  ✓ (target > 85%)
  False Positive Rate     : 1.2%   ~ (target < 1%)
  F1-Score                : 0.8841
```

---

## Analysing a PCAP file

```bash
geoIDS ingest \
  --source pcap \
  --file traffic.pcap \
  --config configs/default.yaml \
  --output alerts.json
```

Each alert in `alerts.json`:
```json
{
  "flow_id": "10.0.0.5:54321-8.8.8.8:443/6",
  "timestamp": 1735000000.123,
  "score": 0.847,
  "threshold": 0.412,
  "is_anomaly": true,
  "confidence": 0.81,
  "top_blade": "tls_sni_length ∧ tls_cipher_suite_id",
  "blade_explanations": [
    {
      "blade": "e1516",
      "description": "tls_sni_length ∧ tls_cipher_suite_id",
      "deviation": 0.634
    }
  ]
}
```

---

## Live capture

```bash
sudo geoIDS ingest \
  --source live \
  --interface eth0 \
  --config configs/default.yaml \
  --state-dir ./state/
```

The `--state-dir` flag saves the detector state every time you Ctrl+C,
so the next run picks up where it left off.

---

## Dashboard

```bash
geoIDS dashboard --port 8050
# Open http://localhost:8050/
```

---

## Zeek integration

```bash
# On your Zeek sensor, enable JSON logging:
#   redef LogAscii::use_json = T;

geoIDS ingest \
  --source zeek \
  --file /var/log/zeek/conn.log \
  --output alerts.json
```

---

## Evaluate on CIC-IDS2017

```bash
# 1. Download the CSV from https://www.unb.ca/cic/datasets/ids-2017.html
# 2. Run evaluation
geoIDS eval \
  --dataset cic2017 \
  --data-path Wednesday-workingHours.pcap_ISCX.csv \
  --output-report report.json

cat report.json | python -m json.tool
```

---

## Configuration quick reference

Key settings in `configs/default.yaml`:

```yaml
geometric_algebra:
  dim: 25          # Feature dimensions
  max_grade: 3     # 2 = bivectors only (faster), 3 = trivectors (more accurate)

detector:
  window_size: 10000       # Larger = more stable reference, more RAM
  target_fpr: 0.01         # Lower = fewer false positives, may miss attacks
  forgetting_factor: 0.99  # Closer to 1 = slower adaptation

writers:
  - type: json
    path: alerts.json
```

---

## Tuning tips

| Goal | Parameter | Direction |
|---|---|---|
| Reduce false positives | `target_fpr` | ↓ lower |
| Faster adaptation to drift | `forgetting_factor` | ↓ lower |
| More memory-efficient | `window_size`, `max_grade` | ↓ lower |
| Better zero-day detection | `max_grade` | 3 |
| Higher throughput | `max_grade`, `use_commutator_score` | 2, False |
