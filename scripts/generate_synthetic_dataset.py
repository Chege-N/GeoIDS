#!/usr/bin/env python3
"""
scripts/generate_synthetic_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a synthetic labelled network-flow dataset in CSV format
compatible with GeoIDS's evaluation harness.

Useful for:
  - Testing the evaluator without downloading large public datasets
  - Unit/integration test fixtures
  - Benchmarking

Output columns match CIC-IDS2017 schema (label col = "Label").

Usage
-----
  python scripts/generate_synthetic_dataset.py \
      --flows 50000 --attack-rate 0.15 --output data/synthetic.csv

  # Then evaluate:
  geoIDS eval --dataset cic2017 --data-path data/synthetic.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _normal_row(rng: np.random.Generator, ts: float) -> dict:
    pkt = int(rng.integers(5, 50))
    byt = int(rng.integers(2000, 80000))
    dur = float(rng.uniform(0.5, 10.0))
    iats = [float(rng.uniform(0.005, 0.3)) for _ in range(max(pkt - 1, 1))]
    sizes = [int(rng.integers(200, 1400)) for _ in range(pkt)]
    return {
        "Label": "BENIGN",
        "timestamp": ts,
        "Protocol": 6,
        "Destination Port": 443,
        "Total Fwd Packets": pkt,
        "Total Length of Fwd Packets": byt,
        "Flow Duration": dur,
        "Average Packet Size": float(np.mean(sizes)),
        "Packet Length Std": float(np.std(sizes) if len(sizes) > 1 else 0),
        "Flow IAT Mean": float(np.mean(iats)) if iats else 0.0,
        "Flow IAT Std": float(np.std(iats)) if len(iats) > 1 else 0.0,
        "Flow Bytes/s": byt / max(dur, 1e-6),
        "Flow Packets/s": pkt / max(dur, 1e-6),
        "Fwd Packets/s": pkt / max(dur, 1e-6),
    }


def _attack_row(rng: np.random.Generator, ts: float, attack_type: str) -> dict:
    row = _normal_row(rng, ts)
    row["Label"] = attack_type

    if attack_type in ("DoS Hulk", "DDoS"):
        row["Total Fwd Packets"] = int(rng.integers(2000, 10000))
        row["Total Length of Fwd Packets"] = int(rng.integers(80000, 500000))
        row["Flow Duration"] = float(rng.uniform(0.01, 0.5))
        row["Average Packet Size"] = float(rng.uniform(40, 80))
        row["Packet Length Std"] = float(rng.uniform(0, 5))
        row["Flow IAT Mean"] = float(rng.uniform(0.00005, 0.001))
        row["Flow IAT Std"] = float(rng.uniform(0, 0.0005))

    elif attack_type == "PortScan":
        row["Destination Port"] = int(rng.integers(1, 65535))
        row["Total Fwd Packets"] = int(rng.integers(1, 3))
        row["Total Length of Fwd Packets"] = int(rng.integers(40, 100))
        row["Flow Duration"] = float(rng.uniform(0.0001, 0.01))

    elif attack_type == "Bot":
        row["Flow IAT Mean"] = float(rng.choice([30.0, 60.0, 120.0, 300.0]))
        row["Flow IAT Std"] = float(rng.uniform(0.0001, 0.001))
        row["Total Fwd Packets"] = int(rng.integers(3, 6))
        row["Average Packet Size"] = float(rng.uniform(60, 90))

    elif attack_type in ("DoH-Backdoor", "Infiltration"):
        # Zero-day patterns
        row["Destination Port"] = 443
        row["Total Fwd Packets"] = int(rng.integers(20, 60))
        row["Average Packet Size"] = float(rng.uniform(40, 120))
        row["Flow IAT Mean"] = float(rng.uniform(0.0001, 0.001))
        row["Flow Duration"] = float(rng.uniform(0.05, 0.5))

    byt = row["Total Length of Fwd Packets"]
    dur = max(row["Flow Duration"], 1e-6)
    pkt = row["Total Fwd Packets"]
    row["Flow Bytes/s"] = byt / dur
    row["Flow Packets/s"] = pkt / dur
    row["Fwd Packets/s"] = pkt / dur

    return row


def generate(
    n_flows: int = 50_000,
    attack_rate: float = 0.15,
    output: str = "data/synthetic.csv",
    seed: int = 42,
    zero_day_fraction: float = 0.3,
) -> str:
    rng = np.random.default_rng(seed)
    t = time.time()

    # Attack types: training-phase and zero-day
    known_attacks = ["DoS Hulk", "DDoS", "PortScan", "Bot"]
    zero_day_attacks = ["DoH-Backdoor", "Infiltration"]

    rows = []
    for i in range(n_flows):
        ts = t + i * 0.01
        roll = rng.random()
        if roll < attack_rate:
            if rng.random() < zero_day_fraction:
                atype = str(rng.choice(zero_day_attacks))
            else:
                atype = str(rng.choice(known_attacks))
            rows.append(_attack_row(rng, ts, atype))
        else:
            rows.append(_normal_row(rng, ts))

    df = pd.DataFrame(rows)

    # Reorder: benign first (acts as "training week"), attacks after
    benign = df[df["Label"] == "BENIGN"]
    attacks = df[df["Label"] != "BENIGN"]
    df_out = pd.concat([benign, attacks], ignore_index=True)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output, index=False)

    print(f"Generated {n_flows:,} flows → {output}")
    vc = df_out["Label"].value_counts()
    for label, count in vc.items():
        zd = " (zero-day)" if label in zero_day_attacks else ""
        print(f"  {label:<20} {count:>6}{zd}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic GeoIDS dataset")
    parser.add_argument("--flows", type=int, default=50_000)
    parser.add_argument("--attack-rate", type=float, default=0.15)
    parser.add_argument("--output", default="data/synthetic.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-day-fraction", type=float, default=0.3)
    args = parser.parse_args()
    generate(
        n_flows=args.flows,
        attack_rate=args.attack_rate,
        output=args.output,
        seed=args.seed,
        zero_day_fraction=args.zero_day_fraction,
    )
