#!/usr/bin/env python3
"""
scripts/demo.py
~~~~~~~~~~~~~~~
GeoIDS live demonstration using synthetic network traffic.

Simulates:
  1. Normal HTTPS flows (training phase)
  2. Gradually introduces:
     a. DoS floods (high packet count, tiny packets)
     b. TLS anomalies (unusual cipher suite + SNI length)
     c. Encrypted C2 beaconing (regular inter-arrival times)
     d. Zero-day: DoH-based backdoor (DNS-over-HTTPS exfil pattern)

Prints colour-coded output and a final report.
No PCAP or live interface required.

Usage
-----
  python scripts/demo.py
  python scripts/demo.py --flows 2000 --attack-rate 0.15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make sure geoidslib is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoidslib import GeoIDS
from geoidslib.algebra.ga_engine import GeometricAlgebraEngine
from geoidslib.detection.detector import AnomalyDetector
from geoidslib.features.extractor import FeatureExtractor, FlowRecord

# ─── Synthetic flow generators ────────────────────────────────────────────────

def normal_flow(rng: np.random.Generator, t: float) -> FlowRecord:
    """Realistic HTTPS browsing flow."""
    return FlowRecord(
        src_ip=f"10.0.{rng.integers(0,50)}.{rng.integers(1,254)}",
        dst_ip=f"{rng.integers(1,254)}.{rng.integers(0,254)}.{rng.integers(0,254)}.{rng.integers(1,254)}",
        src_port=int(rng.integers(49152, 65535)),
        dst_port=443,
        protocol=6,
        packet_count=int(rng.integers(5, 40)),
        byte_count=int(rng.integers(2000, 80000)),
        packet_sizes=[int(rng.integers(200, 1400)) for _ in range(12)],
        inter_arrival_times=[float(rng.uniform(0.005, 0.2)) for _ in range(11)],
        flow_start=t,
        flow_end=t + float(rng.uniform(0.5, 8.0)),
        flag_syn=1, flag_fin=1, flag_ack=int(rng.integers(5, 40)),
        fwd_bytes=int(rng.integers(1000, 40000)),
        bwd_bytes=int(rng.integers(1000, 40000)),
        tls_version=1.3,
        tls_handshake_duration=float(rng.uniform(0.05, 0.3)),
        tls_cert_size=int(rng.integers(1500, 3500)),
        tls_num_extensions=int(rng.integers(4, 9)),
        tls_sni_length=int(rng.integers(8, 30)),
        tls_cipher_suite_id=0xC02B,
        tls_num_san=int(rng.integers(1, 5)),
        tls_cert_lifetime_days=365,
        label="BENIGN",
    )


def dos_flood(rng: np.random.Generator, t: float) -> FlowRecord:
    """SYN-flood / DoS: tiny packets, huge count, fast IAT."""
    return FlowRecord(
        src_ip=f"{rng.integers(1,254)}.{rng.integers(0,254)}.{rng.integers(0,254)}.{rng.integers(1,254)}",
        dst_ip="10.0.0.1",
        src_port=int(rng.integers(1, 1024)),
        dst_port=int(rng.choice([80, 443, 22])),
        protocol=6,
        packet_count=int(rng.integers(2000, 10000)),
        byte_count=int(rng.integers(80000, 500000)),
        packet_sizes=[int(rng.integers(40, 64)) for _ in range(20)],
        inter_arrival_times=[float(rng.uniform(0.00005, 0.0005)) for _ in range(19)],
        flow_start=t,
        flow_end=t + float(rng.uniform(0.01, 0.1)),
        flag_syn=int(rng.integers(1500, 9000)),
        flag_fin=0, flag_ack=0, flag_rst=0,
        fwd_bytes=int(rng.integers(80000, 500000)),
        bwd_bytes=0,
        label="DoS-Flood",
    )


def tls_anomaly(rng: np.random.Generator, t: float) -> FlowRecord:
    """TLS anomaly: expired cert, unknown cipher, long SNI."""
    return FlowRecord(
        src_ip=f"192.168.{rng.integers(0,10)}.{rng.integers(1,254)}",
        dst_ip=f"{rng.integers(1,254)}.0.0.1",
        src_port=int(rng.integers(49152, 65535)),
        dst_port=443,
        protocol=6,
        packet_count=int(rng.integers(8, 25)),
        byte_count=int(rng.integers(3000, 20000)),
        packet_sizes=[int(rng.integers(100, 1400)) for _ in range(10)],
        inter_arrival_times=[float(rng.uniform(0.01, 0.5)) for _ in range(9)],
        flow_start=t,
        flow_end=t + float(rng.uniform(1.0, 10.0)),
        flag_syn=1, flag_fin=1, flag_ack=int(rng.integers(5, 20)),
        fwd_bytes=int(rng.integers(1500, 10000)),
        bwd_bytes=int(rng.integers(1500, 10000)),
        tls_version=1.0,                  # Old TLS version — anomalous
        tls_cert_size=int(rng.integers(200, 400)),  # Tiny cert
        tls_num_extensions=int(rng.integers(15, 25)),  # Too many extensions
        tls_sni_length=int(rng.integers(80, 120)),   # Suspiciously long SNI
        tls_cipher_suite_id=0x0035,       # RC4 — unusual
        tls_num_san=0,
        tls_cert_lifetime_days=int(rng.integers(1, 5)),  # Near-expiry
        tls_handshake_duration=float(rng.uniform(2.0, 8.0)),  # Slow handshake
        label="TLS-Anomaly",
    )


def c2_beacon(rng: np.random.Generator, t: float) -> FlowRecord:
    """Encrypted C2 beaconing: perfectly regular IAT = machine timing."""
    interval = float(rng.choice([30.0, 60.0, 120.0, 300.0]))  # beacon period
    jitter = float(rng.uniform(0.0001, 0.001))
    return FlowRecord(
        src_ip=f"10.0.{rng.integers(0,50)}.{rng.integers(1,254)}",
        dst_ip=f"{rng.integers(1,254)}.{rng.integers(0,254)}.0.1",
        src_port=int(rng.integers(49152, 65535)),
        dst_port=443,
        protocol=6,
        packet_count=4,
        byte_count=int(rng.integers(200, 600)),
        packet_sizes=[int(rng.integers(60, 80)) for _ in range(4)],
        inter_arrival_times=[interval + jitter] * 3,  # very regular
        flow_start=t,
        flow_end=t + interval * 3 + jitter,
        flag_syn=1, flag_fin=1, flag_ack=4,
        fwd_bytes=int(rng.integers(100, 300)),
        bwd_bytes=int(rng.integers(100, 300)),
        tls_version=1.3,
        tls_cert_size=int(rng.integers(800, 1200)),
        tls_num_extensions=3,
        tls_sni_length=int(rng.integers(30, 50)),
        tls_cipher_suite_id=0xC02B,
        label="C2-Beacon",
    )


def doh_backdoor(rng: np.random.Generator, t: float) -> FlowRecord:
    """
    Zero-day: DNS-over-HTTPS data exfiltration.
    High TLS extensions, very long SNI (data-encoded), bursty small packets.
    """
    return FlowRecord(
        src_ip=f"10.0.{rng.integers(0,50)}.{rng.integers(1,254)}",
        dst_ip=f"1.1.1.{rng.integers(1,5)}",
        src_port=int(rng.integers(49152, 65535)),
        dst_port=443,
        protocol=6,
        packet_count=int(rng.integers(20, 60)),
        byte_count=int(rng.integers(1000, 5000)),
        packet_sizes=[int(rng.integers(40, 120)) for _ in range(20)],
        inter_arrival_times=[float(rng.uniform(0.0001, 0.001)) for _ in range(19)],  # bursty
        flow_start=t,
        flow_end=t + float(rng.uniform(0.05, 0.5)),
        flag_syn=1, flag_fin=1, flag_ack=int(rng.integers(20, 60)),
        fwd_bytes=int(rng.integers(800, 4000)),
        bwd_bytes=int(rng.integers(200, 1000)),
        tls_version=1.3,
        tls_cert_size=int(rng.integers(500, 800)),
        tls_num_extensions=int(rng.integers(18, 30)),  # anomalous
        tls_sni_length=int(rng.integers(100, 200)),    # very long — encoded data
        tls_cipher_suite_id=0x1301,
        tls_num_san=int(rng.integers(10, 20)),
        tls_cert_lifetime_days=int(rng.integers(30, 90)),
        label="DoH-Backdoor",  # Zero-day: not seen in training
    )


# ─── Demo runner ─────────────────────────────────────────────────────────────

ATTACK_GENERATORS = {
    "DoS-Flood":    dos_flood,
    "TLS-Anomaly":  tls_anomaly,
    "C2-Beacon":    c2_beacon,
}

ZERO_DAY_GENERATORS = {
    "DoH-Backdoor": doh_backdoor,
}

COLOURS = {
    "BENIGN":       "\033[92m",   # green
    "DoS-Flood":    "\033[91m",   # red
    "TLS-Anomaly":  "\033[93m",   # yellow
    "C2-Beacon":    "\033[95m",   # magenta
    "DoH-Backdoor": "\033[96m",   # cyan (zero-day)
    "RESET":        "\033[0m",
}


def print_header():
    print("\n" + "═" * 68)
    print("  ⬡  GeoIDS — Geometric Algebra Intrusion Detection System")
    print("  Demonstration: Synthetic Encrypted Traffic Simulation")
    print("═" * 68)


def print_progress_bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def run_demo(n_flows: int = 1500, attack_rate: float = 0.12, seed: int = 42):
    rng = np.random.default_rng(seed)
    t = time.time()

    print_header()

    # Build IDS
    engine = GeometricAlgebraEngine(dim=25, p=15, q=10, max_grade=3)
    extractor = FeatureExtractor(online_normalise=True)
    detector = AnomalyDetector(
        ga_engine=engine,
        window_size=500,
        reframe_interval=50,
        forgetting_factor=0.99,
        use_isolation_forest=True,
        target_fpr=0.02,
    )
    ids = GeoIDS(ga_engine=engine, feature_extractor=extractor,
                 anomaly_detector=detector, writers=[])

    # Training phase
    n_train = int(n_flows * 0.4)
    print(f"\n📚 Training phase: {n_train} normal flows…")
    for i in range(n_train):
        ids.process_flow_record(normal_flow(rng, t + i * 0.01))
    ids.anomaly_detector.force_recompute_reference()
    print(
        f"   ✓ Reference multiframe established "
        f"({len(ids.anomaly_detector._reference_mv.blades)} active blades)"
    )

    # Stats tracking
    stats: dict[str, dict] = {
        label: {"total": 0, "detected": 0}
        for label in ["BENIGN", "DoS-Flood", "TLS-Anomaly", "C2-Beacon", "DoH-Backdoor"]
    }
    recent_alerts = []

    # Test phase
    n_test = n_flows - n_train
    print(f"\n🔍 Detection phase: {n_test} flows (attack rate ≈ {attack_rate:.0%})…\n")

    attack_labels = list(ATTACK_GENERATORS.keys())
    zero_day_labels = list(ZERO_DAY_GENERATORS.keys())
    all_attack_labels = attack_labels + zero_day_labels

    for i in range(n_test):
        roll = rng.random()

        # Decide flow type
        if roll < attack_rate:
            # 60% known attacks, 40% zero-day
            if rng.random() < 0.6:
                label = str(rng.choice(attack_labels))
                flow = ATTACK_GENERATORS[label](rng, t + (n_train + i) * 0.01)
            else:
                label = str(rng.choice(zero_day_labels))
                flow = ZERO_DAY_GENERATORS[label](rng, t + (n_train + i) * 0.01)
        else:
            flow = normal_flow(rng, t + (n_train + i) * 0.01)
            label = "BENIGN"

        result = ids.process_flow_record(flow)
        stats[label]["total"] += 1

        if result.is_anomaly:
            stats[label]["detected"] += 1
            col = COLOURS.get(label, COLOURS["RESET"])
            recent_alerts.append({
                "i": n_train + i,
                "label": label,
                "score": result.score,
                "blade": result.top_blade_label or "—",
                "col": col,
            })

        # Print progress every 200 flows
        if (i + 1) % 200 == 0 or i == n_test - 1:
            bar = print_progress_bar(i + 1, n_test)
            alert_count = sum(s["detected"] for s in stats.values())
            print(f"\r  {bar}  alerts: {alert_count}", end="", flush=True)

    print()

    # Print last 10 alerts
    if recent_alerts:
        print(f"\n🚨 Last {min(10, len(recent_alerts))} alerts:\n")
        print(f"  {'Flow#':<7} {'Attack Type':<16} {'Score':>8}  {'Top Blade'}")
        print("  " + "─" * 62)
        for a in recent_alerts[-10:]:
            col = a["col"]
            rst = COLOURS["RESET"]
            print(f"  {a['i']:<7} {col}{a['label']:<16}{rst} {a['score']:>8.4f}  {a['blade'][:40]}")

    # Final report
    print("\n" + "═" * 68)
    print("  📊 Detection Report")
    print("═" * 68)
    print(f"  {'Category':<18} {'Total':>7} {'Detected':>9} {'DR':>7}  {'Type'}")
    print("  " + "─" * 60)

    for label in ["BENIGN", "DoS-Flood", "TLS-Anomaly", "C2-Beacon", "DoH-Backdoor"]:
        s = stats[label]
        if s["total"] == 0:
            continue
        detected = s["detected"]
        total = s["total"]
        dr = detected / total

        if label == "BENIGN":
            dr_str = f"{dr:.1%}"  # FPR for benign
            kind = "(FPR)"
            col = "\033[91m" if dr > 0.05 else "\033[92m"
        elif label == "DoH-Backdoor":
            dr_str = f"{dr:.1%}"
            kind = "(ZERO-DAY)"
            col = "\033[96m"
        else:
            dr_str = f"{dr:.1%}"
            kind = "(known)"
            col = "\033[92m" if dr >= 0.85 else "\033[93m"

        rst = COLOURS["RESET"]
        print(f"  {label:<18} {total:>7} {detected:>9} {col}{dr_str:>7}{rst}  {kind}")

    # Overall metrics
    total_attacks = sum(stats[label]["total"] for label in all_attack_labels)
    total_detected = sum(stats[label]["detected"] for label in all_attack_labels)
    zd_total = stats["DoH-Backdoor"]["total"]
    zd_detected = stats["DoH-Backdoor"]["detected"]
    benign_total = stats["BENIGN"]["total"]
    benign_fp = stats["BENIGN"]["detected"]

    overall_dr = total_detected / max(total_attacks, 1)
    overall_fpr = benign_fp / max(benign_total, 1)
    zd_dr = zd_detected / max(zd_total, 1)

    precision = total_detected / max(total_detected + benign_fp, 1)
    f1 = 2 * precision * overall_dr / max(precision + overall_dr, 1e-9)

    print("  " + "─" * 60)
    print(f"\n  Overall Detection Rate  : {overall_dr:.1%}")
    print(f"  Zero-Day Detection Rate : {zd_dr:.1%}  {'✓' if zd_dr > 0.85 else '✗'} (target > 85%)")
    print(
        f"  False Positive Rate     : {overall_fpr:.2%} "
        f"{'✓' if overall_fpr < 0.01 else '~'} (target < 1%)"
    )
    print(f"  F1-Score                : {f1:.4f}")
    print(f"\n  Detector stats: {ids.stats}")
    print("\n" + "═" * 68 + "\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoIDS Demo")
    parser.add_argument("--flows", type=int, default=1500, help="Total flows to simulate")
    parser.add_argument("--attack-rate", type=float, default=0.12, help="Fraction of attack flows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_demo(n_flows=args.flows, attack_rate=args.attack_rate, seed=args.seed)
