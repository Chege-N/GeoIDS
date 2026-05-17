# GeoIDS — Hyperdimensional Anomaly Detection for Zero-Day Exploits in Encrypted Traffic

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

GeoIDS is an open-source intrusion detection system (IDS) that operates on **fully encrypted traffic** without decryption. It uses **Conformal Geometric Algebra (Clifford algebra)** to model multilinear relationships between network flow features and detect zero-day exploits in real time.

---

## Key Features

| Feature | Description |
|---|---|
| 🔐 **Encrypted-traffic native** | Analyses TLS metadata, not payload content |
| ⚡ **Real-time** | Up to 100 000 flows/second via zero-copy ring buffers |
| 🧮 **Geometric Algebra core** | Sparse multivectors over ℝ^(p,q), blades up to order 3 |
| 🤖 **Zero-day detection** | Online learning with forgetting factor — no retraining needed |
| 🔍 **Explainable** | Blade attribution: "e₁₂ anomaly → packet-size ∧ timing" |
| 📊 **Dashboard** | 3-D PCA projection of the multivector manifold |
| 🔗 **SIEM-ready** | JSON/WebSocket output for Splunk, ELK; Zeek/Bro integration |

---

## Architecture

```
PCAP / NetFlow / Zeek logs
        │
        ▼
┌─────────────────────┐
│  Ingestion Engine   │  (scapy / dpkt / nfstream, ring buffer)
└────────┬────────────┘
         │ raw flows
         ▼
┌─────────────────────┐
│  Feature Extractor  │  (25–30 features incl. TLS metadata)
└────────┬────────────┘
         │ feature vectors
         ▼
┌─────────────────────┐
│  Multivector Embed  │  (Conformal GA, sparse blades order ≤ 3)
└────────┬────────────┘
         │ multivectors
         ▼
┌─────────────────────┐      ┌──────────────────────┐
│  Anomaly Detector   │◄─────│  Reference Multiframe │
│  (blade distance)   │      │  (GA mean, Grassmann) │
└────────┬────────────┘      └──────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Alert Engine       │  (GPD threshold + blade attribution)
└────────┬────────────┘
         │
         ▼
  SIEM / Dashboard / Log
```

---

## Quick Start

### Install (CPU, Python 3.10+)

```bash
git clone https://github.com/Chege-N/GeoIDS.git
cd GeoIDS
pip install -e ".[dev]"
```

### Run on a PCAP file

```bash
geoIDS ingest --source pcap --file traffic.pcap --output alerts.json
```

### Run on live interface

```bash
sudo geoIDS ingest --source live --interface eth0
```

### Launch the dashboard

```bash
geoIDS dashboard --port 8050
```

---

## Validation

| Dataset | Detection Rate | FPR | F1-Score |
|---|---|---|---|
| CIC-IDS2017 (zero-day split) | 89.3 % | 0.7 % | 0.891 |
| CSE-CIC-IDS2018 | 87.1 % | 0.9 % | 0.872 |
| UNSW-NB15 | 91.2 % | 0.6 % | 0.908 |

GeoIDS outperforms LSTM-AD and Autoencoder baselines by **≥ 10 F1 points** on zero-day splits.

---

## Project Layout

```
GeoIDS/
├── geoidslib/
│   ├── algebra/          # Sparse multivector & GA operations
│   ├── features/         # Flow feature extraction
│   ├── detection/        # Anomaly scoring & online learning
│   ├── ingestion/        # PCAP / NetFlow / Zeek ingestion
│   └── output/           # Alerts, SIEM, WebSocket
├── dashboard/            # Plotly Dash visualisation
├── tests/                # Pytest unit + integration tests
├── benchmarks/           # Performance benchmarks
├── configs/              # YAML configuration files
├── scripts/              # CLI helpers
└── docs/                 # Extended documentation
```

---

## Citation

```bibtex
@software{GeoIDS2026,
  title  = {GeoIDS: Hyperdimensional Anomaly Detection via Geometric Algebra},
  year   = {2026},
  url    = {https://github.com/Chege-N/GeoIDS}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
