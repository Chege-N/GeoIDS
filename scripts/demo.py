#!/usr/bin/env python3
"""
scripts/demo.py
~~~~~~~~~~~~~~~
GeoIDS live demonstration using synthetic network traffic.

Simulates:
  1. Normal HTTPS flows (training phase) — tight, consistent parameters
  2. Gradually introduces:
     a. DoS floods      (high pkt count, tiny packets, no TLS)
     b. TLS anomalies   (old cipher, huge SNI, tiny cert)
     c. C2 beaconing    (machine-regular IAT)
     d. DoH-backdoor    (zero-day: DNS-over-HTTPS exfil)

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoidslib import GeoIDS
from geoidslib.algebra.ga_engine import GeometricAlgebraEngine
from geoidslib.detection.detector import AnomalyDetector
from geoidslib.features.extractor import FeatureExtractor, FlowRecord

# ─── Synthetic flow generators ────────────────────────────────────────────────

# Fixed destination IPs to keep dst_ip_class stable in normal traffic
_NORMAL_DST_IPS = [
    "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1",
    "13.107.42.14", "52.96.0.1", "104.16.0.1", "93.184.216.34",
]


def normal_flow(rng: np.random.Generator, t: float) -> FlowRecord:
    """
    Realistic HTTPS browsing flow.
    Key constraint: ALL normal flows use TLS 1.3, fixed cipher, Class-A
    destinations and consistent packet sizes — so the reference multiframe
    is tight and attacks stand out clearly.
    """
    pkt = int(rng.integers(8, 35))
    byt = int(rng.integers(3000, 60_000))
    dur = float(rng.uniform(0.5, 6.0))
    fwd = int(byt * rng.uniform(0.4, 0.6))
    return FlowRecord(
        src_ip=f"10.0.{rng.integers(0, 50)}.{rng.integers(1, 254)}",
        dst_ip=str(rng.choice(_NORMAL_DST_IPS)),
        src_port=int(rng.integers(49152, 65535)),
        dst_port=443,
        protocol=6,
        packet_count=pkt,
        byte_count=byt,
        packet_sizes=[int(rng.integers(400, 1400)) for _ in range(pkt)],
        inter_arrival_times=[float(rng.uniform(0.01, 0.15)) for _ in range(pkt - 1)],
        flow_start=t,
        flow_end=t + dur,
        flag_syn=1, flag_fin=1,
        flag_ack=int(rng.integers(8, 35)),
        flag_rst=0,
        fwd_bytes=fwd,
        bwd_bytes=byt - fwd,
        # TLS: always 1.3, well-known cipher, moderate SNI
        tls_version=1.3,
        tls_handshake_duration=float(rng.uniform(0.05, 0.25)),
        tls_cert_size=int(rng.integers(2000, 3500)),
        tls_num_extensions=int(rng.integers(5, 9)),
        tls_sni_length=int(rng.integers(8, 25)),
        tls_cipher_suite_id=0xC02B,          # TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
        tls_num_san=int(rng.integers(2, 5)),
        tls_cert_lifetime_days=365,
        label="BENIGN",
    )


def dos_flood(rng: np.random.Generator, t: float) -> FlowRecord:
    """SYN-flood: thousands of tiny packets, no TLS, asymmetric direction."""
    pkt = int(rng.integers(3000, 10_000))
    byt = pkt * int(rng.integers(40, 64))
    return FlowRecord(
        src_ip=f"{rng.integers(100,250)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}",
        dst_ip="10.0.0.1",
        src_port=int(rng.integers(1024, 65535)),
        dst_port=int(rng.choice([80, 443, 22, 3389])),
        protocol=6,
        packet_count=pkt,
        byte_count=byt,
        packet_sizes=[int(rng.integers(40, 64)) for _ in range(20)],
        inter_arrival_times=[float(rng.uniform(0.00005, 0.0005)) for _ in range(19)],
        flow_start=t,
        flow_end=t + float(rng.uniform(0.01, 0.3)),
        flag_syn=pkt - 2, flag_fin=0, flag_ack=0, flag_rst=0,
        fwd_bytes=byt, bwd_bytes=0,
        tls_version=0.0, tls_cert_size=0,
        tls_num_extensions=0, tls_sni_length=0,
        tls_cipher_suite_id=0, tls_num_san=0,
        tls_cert_lifetime_days=0,
        label="DoS-Flood",
    )


def tls_anomaly(rng: np.random.Generator, t: float) -> FlowRecord:
    """TLS anomaly: old TLS 1.0, unknown cipher, huge SNI, near-expiry cert."""
    pkt = int(rng.integers(8, 20))
    byt = int(rng.integers(3000, 20_000))
    fwd = int(byt * rng.uniform(0.4, 0.6))
    return FlowRecord(
        src_ip=f"192.168.{rng.integers(0, 10)}.{rng.integers(1, 254)}",
        dst_ip=f"{rng.integers(1, 254)}.0.0.1",
        src_port=int(rng.integers(49152, 65535)),
        dst_port=443,
        protocol=6,
        packet_count=pkt,
        byte_count=byt,
        packet_sizes=[int(rng.integers(100, 800)) for _ in range(pkt)],
        inter_arrival_times=[float(rng.uniform(0.01, 0.5)) for _ in range(pkt - 1)],
        flow_start=t,
        flow_end=t + float(rng.uniform(1.0, 10.0)),
        flag_syn=1, flag_fin=1,
        flag_ack=int(rng.integers(5, 20)), flag_rst=0,
        fwd_bytes=fwd, bwd_bytes=byt - fwd,
        tls_version=1.0,                          # ← old TLS
        tls_handshake_duration=float(rng.uniform(2.0, 8.0)),  # ← slow
        tls_cert_size=int(rng.integers(200, 400)),             # ← tiny cert
        tls_num_extensions=int(rng.integers(18, 28)),          # ← too many
        tls_sni_length=int(rng.integers(80, 120)),             # ← long SNI
        tls_cipher_suite_id=0x0035,               # ← RC4 (obsolete)
        tls_num_san=0,
        tls_cert_lifetime_days=int(rng.integers(1, 5)),        # ← near-expiry
        label="TLS-Anomaly",
    )


def c2_beacon(rng: np.random.Generator, t: float) -> FlowRecord:
    """Encrypted C2: machine-regular IAT (beaconing interval), tiny payloads."""
    interval = float(rng.choice([30.0, 60.0, 120.0, 300.0]))
    jitter = float(rng.uniform(0.0001, 0.001))
    pkt = 4
    byt = int(rng.integers(180, 500))
    return FlowRecord(
        src_ip=f"10.0.{rng.integers(0, 50)}.{rng.integers(1, 254)}",
        dst_ip=str(rng.choice(_NORMAL_DST_IPS)),
        src_port=int(rng.integers(49152, 65535)),
        dst_port=443,
        protocol=6,
        packet_count=pkt,
        byte_count=byt,
        packet_sizes=[int(rng.integers(60, 80)) for _ in range(pkt)],
        inter_arrival_times=[interval + jitter] * (pkt - 1),  # perfectly regular
        flow_start=t,
        flow_end=t + interval * (pkt - 1) + jitter,
        flag_syn=1, flag_fin=1, flag_ack=pkt, flag_rst=0,
        fwd_bytes=byt // 2, bwd_bytes=byt // 2,
        tls_version=1.3,
        tls_handshake_duration=float(rng.uniform(0.05, 0.15)),
        tls_cert_size=int(rng.integers(800, 1200)),
        tls_num_extensions=3,
        tls_sni_length=int(rng.integers(30, 50)),
        tls_cipher_suite_id=0xC02B,
        tls_num_san=int(rng.integers(1, 3)),
        tls_cert_lifetime_days=90,
        label="C2-Beacon",
    )


def doh_backdoor(rng: np.random.Generator, t: float) -> FlowRecord:
    """
    Zero-day: DNS-over-HTTPS data exfiltration.
    High extension count, very long SNI (data-encoded), bursty tiny packets,
    asymmetric direction (fwd >> bwd).
    """
    pkt = int(rng.integers(25, 70))
    byt = int(rng.integers(1200, 6000))
    return FlowRecord(
        src_ip=f"10.0.{rng.integers(0, 50)}.{rng.integers(1, 254)}",
        dst_ip=f"1.1.1.{rng.integers(1, 5)}",
        src_port=int(rng.integers(49152, 65535)),
        dst_port=443,
        protocol=6,
        packet_count=pkt,
        byte_count=byt,
        packet_sizes=[int(rng.integers(40, 120)) for _ in range(pkt)],
        inter_arrival_times=[float(rng.uniform(0.0001, 0.001)) for _ in range(pkt - 1)],
        flow_start=t,
        flow_end=t + float(rng.uniform(0.05, 0.5)),
        flag_syn=1, flag_fin=1,
        flag_ack=int(rng.integers(20, 60)), flag_rst=0,
        fwd_bytes=int(byt * 0.85),          # heavily asymmetric
        bwd_bytes=int(byt * 0.15),
        tls_version=1.3,
        tls_handshake_duration=float(rng.uniform(0.05, 0.2)),
        tls_cert_size=int(rng.integers(500, 800)),
        tls_num_extensions=int(rng.integers(20, 30)),   # anomalous
        tls_sni_length=int(rng.integers(110, 200)),     # very long
        tls_cipher_suite_id=0x1301,                     # TLS_AES_128_GCM_SHA256
        tls_num_san=int(rng.integers(12, 22)),
        tls_cert_lifetime_days=int(rng.integers(30, 90)),
        label="DoH-Backdoor",
    )


# ─── Demo runner ─────────────────────────────────────────────────────────────

ATTACK_GENERATORS = {
    "DoS-Flood":   dos_flood,
    "TLS-Anomaly": tls_anomaly,
    "C2-Beacon":   c2_beacon,
}
ZERO_DAY_GENERATORS = {
    "DoH-Backdoor": doh_backdoor,
}

COLOURS = {
    "BENIGN":       "\033[92m",
    "DoS-Flood":    "\033[91m",
    "TLS-Anomaly":  "\033[93m",
    "C2-Beacon":    "\033[95m",
    "DoH-Backdoor": "\033[96m",
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


def run_demo(n_flows: int = 2000, attack_rate: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    t = time.time()

    print_header()

    # ── Build IDS with tuned parameters ───────────────────────────────────────
    engine = GeometricAlgebraEngine(dim=25, p=15, q=10, max_grade=3)
    extractor = FeatureExtractor(online_normalise=True)
    detector = AnomalyDetector(
        ga_engine=engine,
        window_size=800,
        reframe_interval=100,       # recompute reference every 100 normal flows
        forgetting_factor=0.995,    # slow drift — normal traffic is stable
        use_isolation_forest=True,
        isolation_forest_contamination=0.02,
        target_fpr=0.01,            # 1% FPR target
        gpd_tail_fraction=0.05,     # tighter tail fit
        use_commutator_score=True,
        explanation_threshold=0.05,
    )
    ids = GeoIDS(
        ga_engine=engine,
        feature_extractor=extractor,
        anomaly_detector=detector,
        writers=[],
    )

    # ── Training phase ────────────────────────────────────────────────────────
    n_train = max(int(n_flows * 0.40), 300)
    print(f"\n📚 Training phase: {n_train} normal flows…")
    for i in range(n_train):
        ids.process_flow_record(normal_flow(rng, t + i * 0.01))
    ids.anomaly_detector.force_recompute_reference()
    ref_blades = len(ids.anomaly_detector._reference_mv.blades)
    thr = ids.anomaly_detector.threshold_estimator.threshold
    print(f"   ✓ Reference multiframe: {ref_blades} active blades  |  "
          f"initial threshold: {thr:.4f}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    attack_labels = list(ATTACK_GENERATORS.keys())
    zero_day_labels = list(ZERO_DAY_GENERATORS.keys())
    all_attack_labels = attack_labels + zero_day_labels
    stats: dict[str, dict] = {
        lbl: {"total": 0, "detected": 0}
        for lbl in ["BENIGN"] + all_attack_labels
    }
    recent_alerts: list[dict] = []

    # ── Detection phase ───────────────────────────────────────────────────────
    n_test = n_flows - n_train
    print(f"\n🔍 Detection phase: {n_test} flows (attack rate ≈ {attack_rate:.0%})…\n")

    for i in range(n_test):
        roll = rng.random()
        if roll < attack_rate:
            if rng.random() < 0.55:
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
            recent_alerts.append({
                "i": n_train + i,
                "label": label,
                "score": result.score,
                "blade": result.top_blade_label or "—",
                "col": COLOURS.get(label, COLOURS["RESET"]),
            })

        if (i + 1) % 200 == 0 or i == n_test - 1:
            alert_count = sum(s["detected"] for s in stats.values())
            bar = print_progress_bar(i + 1, n_test)
            thr_now = ids.anomaly_detector.threshold_estimator.threshold
            print(f"\r  {bar}  alerts: {alert_count}  thr: {thr_now:.3f}",
                  end="", flush=True)

    print()

    # ── Last 10 alerts ────────────────────────────────────────────────────────
    if recent_alerts:
        print(f"\n🚨 Last {min(10, len(recent_alerts))} alerts:\n")
        print(f"  {'Flow#':<7} {'Attack Type':<16} {'Score':>8}  {'Top Blade'}")
        print("  " + "─" * 62)
        for a in recent_alerts[-10:]:
            col, rst = a["col"], COLOURS["RESET"]
            print(f"  {a['i']:<7} {col}{a['label']:<16}{rst} "
                  f"{a['score']:>8.4f}  {a['blade'][:40]}")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  📊 Detection Report")
    print("═" * 68)
    print(f"  {'Category':<18} {'Total':>7} {'Detected':>9} {'DR':>7}  {'Type'}")
    print("  " + "─" * 60)

    for label in ["BENIGN"] + attack_labels + zero_day_labels:
        s = stats[label]
        if s["total"] == 0:
            continue
        dr = s["detected"] / s["total"]
        if label == "BENIGN":
            col = "\033[91m" if dr > 0.05 else "\033[92m"
            kind = "(FPR)"
        elif label in zero_day_labels:
            col = "\033[92m" if dr >= 0.85 else "\033[93m"
            kind = "(ZERO-DAY)"
        else:
            col = "\033[92m" if dr >= 0.80 else "\033[93m"
            kind = "(known)"
        rst = COLOURS["RESET"]
        print(f"  {label:<18} {s['total']:>7} {s['detected']:>9} "
              f"{col}{dr:>7.1%}{rst}  {kind}")

    # Overall
    total_attacks = sum(stats[lbl]["total"] for lbl in all_attack_labels)
    total_detected = sum(stats[lbl]["detected"] for lbl in all_attack_labels)
    zd_total = stats["DoH-Backdoor"]["total"]
    zd_detected = stats["DoH-Backdoor"]["detected"]
    benign_total = stats["BENIGN"]["total"]
    benign_fp = stats["BENIGN"]["detected"]

    dr_overall = total_detected / max(total_attacks, 1)
    dr_zd = zd_detected / max(zd_total, 1)
    fpr = benign_fp / max(benign_total, 1)
    precision = total_detected / max(total_detected + benign_fp, 1)
    f1 = 2 * precision * dr_overall / max(precision + dr_overall, 1e-9)

    print("  " + "─" * 60)
    print(f"\n  Overall Detection Rate  : {dr_overall:.1%}")
    zd_ok = "✓" if dr_zd >= 0.85 else "✗"
    fpr_ok = "✓" if fpr < 0.01 else "~"
    print(f"  Zero-Day Detection Rate : {dr_zd:.1%}  {zd_ok} (target > 85%)")
    print(f"  False Positive Rate     : {fpr:.2%}  {fpr_ok} (target < 1%)")
    print(f"  F1-Score                : {f1:.4f}")
    print(f"\n  Detector stats: {ids.stats}")
    print("\n" + "═" * 68 + "\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoIDS Demo")
    parser.add_argument("--flows", type=int, default=2000)
    parser.add_argument("--attack-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_demo(n_flows=args.flows, attack_rate=args.attack_rate, seed=args.seed)
