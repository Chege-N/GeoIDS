"""
geoidslib.features.extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extracts 25 normalised features from a network flow (5-tuple + metadata).

Feature dimensions (indices 0-24)
----------------------------------
Continuous (indices 0-14):
  0  packet_count
  1  byte_count
  2  mean_packet_size
  3  std_packet_size
  4  mean_iat              inter-arrival time
  5  std_iat
  6  flow_duration
  7  packet_length_entropy
  8  iat_entropy
  9  tcp_flags_ratio       (SYN+FIN+RST) / total packets
 10  bytes_per_second
 11  packets_per_second
 12  tls_handshake_duration
 13  tls_cert_size
 14  tls_num_extensions

Categorical / TLS (indices 15-24):
 15  tls_sni_length
 16  tls_cipher_suite_id   (normalised)
 17  tls_version           (1.0→0.0, 1.1→0.33, 1.2→0.67, 1.3→1.0)
 18  tls_num_san
 19  tls_cert_lifetime_days (log-normalised)
 20  protocol              (TCP→0, UDP→0.5, ICMP→1)
 21  dst_port_category     (well-known→0, registered→0.5, ephemeral→1)
 22  src_ip_entropy
 23  dst_ip_class          (A→0, B→0.33, C→0.67, D→1)
 24  flow_direction_ratio  upload/download ratio

All features are min-max normalised to [0, 1] using online statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NUM_FEATURES = 25


@dataclass
class FlowRecord:
    """
    Parsed representation of a single network flow.
    All fields are optional — missing values → 0.
    """

    # 5-tuple
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 6  # TCP=6, UDP=17, ICMP=1

    # Packet-level statistics
    packet_count: int = 0
    byte_count: int = 0
    packet_sizes: list[int] = field(default_factory=list)
    inter_arrival_times: list[float] = field(default_factory=list)
    flow_start: float = 0.0
    flow_end: float = 0.0

    # TCP flags (counts)
    flag_syn: int = 0
    flag_fin: int = 0
    flag_rst: int = 0
    flag_ack: int = 0

    # Upload / download bytes
    fwd_bytes: int = 0
    bwd_bytes: int = 0

    # TLS metadata (present only for TLS flows)
    tls_handshake_duration: float = 0.0
    tls_cert_size: int = 0
    tls_num_extensions: int = 0
    tls_sni_length: int = 0
    tls_cipher_suite_id: int = 0
    tls_version: float = 0.0  # 1.0, 1.1, 1.2, 1.3
    tls_num_san: int = 0
    tls_cert_lifetime_days: int = 0

    # Ground truth (for evaluation only)
    label: str | None = None


class OnlineNormaliser:
    """
    Incremental min-max normaliser.
    Tracks running min and max for each feature dimension.
    """

    def __init__(self, dim: int, clip: bool = True):
        self.dim = dim
        self.clip = clip
        self._min = np.full(dim, np.inf)
        self._max = np.full(dim, -np.inf)
        self._n = 0

    def update(self, vec: np.ndarray) -> None:
        np.minimum(self._min, vec, out=self._min)
        np.maximum(self._max, vec, out=self._max)
        self._n += 1

    def transform(self, vec: np.ndarray) -> np.ndarray:
        denom = self._max - self._min
        denom = np.where(denom < 1e-9, 1.0, denom)
        result = (vec - self._min) / denom
        if self.clip:
            result = np.clip(result, 0.0, 1.0)
        return result

    def fit_transform(self, vec: np.ndarray) -> np.ndarray:
        self.update(vec)
        return self.transform(vec)

    def reset(self) -> None:
        self._min = np.full(self.dim, np.inf)
        self._max = np.full(self.dim, -np.inf)
        self._n = 0


def _shannon_entropy(values: list[float]) -> float:
    """Compute normalised Shannon entropy of a list of values."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return 0.0
    arr /= arr.sum()
    ent = -np.sum(arr * np.log2(arr + 1e-12))
    return float(ent / math.log2(len(values) + 1))  # normalise to [0,1]


def _packet_size_entropy(sizes: list[int]) -> float:
    """Entropy of packet-size distribution."""
    if not sizes:
        return 0.0
    counts: dict[int, int] = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    return _shannon_entropy(list(counts.values()))


def _iat_entropy(iats: list[float]) -> float:
    """Entropy of discretised inter-arrival time distribution."""
    if not iats:
        return 0.0
    # Discretise into 20 bins
    arr = np.array(iats, dtype=float)
    bins = np.linspace(arr.min(), arr.max() + 1e-9, 21)
    hist, _ = np.histogram(arr, bins=bins)
    return _shannon_entropy(hist.tolist())


def _ip_entropy(ip_str: str) -> float:
    """Compute entropy of IP address octets (proxy for randomness)."""
    try:
        octets = [int(x) / 255.0 for x in ip_str.split(".")]
        return _shannon_entropy(octets)
    except Exception:
        return 0.0


def _dst_ip_class(ip_str: str) -> float:
    """Map IP to class A/B/C/D encoding."""
    try:
        first_octet = int(ip_str.split(".")[0])
        if first_octet < 128:
            return 0.0  # Class A
        elif first_octet < 192:
            return 0.33  # Class B
        elif first_octet < 224:
            return 0.67  # Class C
        else:
            return 1.0  # Class D (multicast)
    except Exception:
        return 0.0


def _port_category(port: int) -> float:
    """Map port to category: well-known / registered / ephemeral."""
    if port < 1024:
        return 0.0
    elif port < 49152:
        return 0.5
    else:
        return 1.0


def _tls_version_norm(version: float) -> float:
    mapping = {1.0: 0.0, 1.1: 0.33, 1.2: 0.67, 1.3: 1.0}
    return mapping.get(round(version, 1), 0.5)


def _protocol_norm(proto: int) -> float:
    mapping = {6: 0.0, 17: 0.5, 1: 1.0}
    return mapping.get(proto, 0.5)


class FeatureExtractor:
    """
    Extract and normalise a 25-dimensional feature vector from a FlowRecord.

    Parameters
    ----------
    online_normalise : bool
        Use online min-max normalisation (recommended for streaming).
    normaliser : OnlineNormaliser | None
        Pre-built normaliser (e.g. loaded from disk).
    """

    def __init__(
        self,
        online_normalise: bool = True,
        normaliser: OnlineNormaliser | None = None,
    ):
        self.online_normalise = online_normalise
        self.normaliser = normaliser or OnlineNormaliser(NUM_FEATURES)
        self._feature_names = [
            "packet_count", "byte_count", "mean_packet_size", "std_packet_size",
            "mean_iat", "std_iat", "flow_duration", "packet_length_entropy",
            "iat_entropy", "tcp_flags_ratio", "bytes_per_second", "packets_per_second",
            "tls_handshake_duration", "tls_cert_size", "tls_num_extensions",
            "tls_sni_length", "tls_cipher_suite_id", "tls_version",
            "tls_num_san", "tls_cert_lifetime_days", "protocol",
            "dst_port_category", "src_ip_entropy", "dst_ip_class",
            "flow_direction_ratio",
        ]

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def extract_raw(self, flow: FlowRecord) -> np.ndarray:
        """
        Extract the raw (unnormalised) feature vector from a FlowRecord.

        Returns
        -------
        np.ndarray of shape (25,)
        """
        sizes = flow.packet_sizes or []
        iats = flow.inter_arrival_times or []

        n = max(flow.packet_count, 1)
        duration = max(flow.flow_end - flow.flow_start, 1e-6)

        mean_ps = np.mean(sizes) if sizes else 0.0
        std_ps = np.std(sizes) if len(sizes) > 1 else 0.0
        mean_iat = np.mean(iats) if iats else 0.0
        std_iat = np.std(iats) if len(iats) > 1 else 0.0

        tcp_flags = flow.flag_syn + flow.flag_fin + flow.flag_rst
        tcp_flags_ratio = tcp_flags / n

        bps = flow.byte_count / duration
        pps = flow.packet_count / duration

        total_bw = max(flow.fwd_bytes + flow.bwd_bytes, 1)
        dir_ratio = flow.fwd_bytes / total_bw

        cert_lifetime_log = math.log1p(flow.tls_cert_lifetime_days)

        vec = np.array([
            float(flow.packet_count),
            float(flow.byte_count),
            float(mean_ps),
            float(std_ps),
            float(mean_iat),
            float(std_iat),
            float(duration),
            _packet_size_entropy(sizes),
            _iat_entropy(iats),
            float(tcp_flags_ratio),
            float(bps),
            float(pps),
            float(flow.tls_handshake_duration),
            float(flow.tls_cert_size),
            float(flow.tls_num_extensions),
            float(flow.tls_sni_length),
            float(flow.tls_cipher_suite_id) / 65535.0,  # normalise to [0,1]
            _tls_version_norm(flow.tls_version),
            float(flow.tls_num_san),
            float(cert_lifetime_log),
            _protocol_norm(flow.protocol),
            _port_category(flow.dst_port),
            _ip_entropy(flow.src_ip),
            _dst_ip_class(flow.dst_ip),
            float(dir_ratio),
        ], dtype=float)

        return vec

    def extract(self, flow: FlowRecord) -> np.ndarray:
        """
        Extract and normalise feature vector.

        Returns
        -------
        np.ndarray of shape (25,), values in [0, 1].
        """
        raw = self.extract_raw(flow)
        if self.online_normalise:
            return self.normaliser.fit_transform(raw)
        return raw

    def extract_batch(self, flows: list[FlowRecord]) -> np.ndarray:
        """
        Extract features from a list of flows.

        Returns
        -------
        np.ndarray of shape (n_flows, 25)
        """
        return np.vstack([self.extract(f) for f in flows])

    def save_normaliser(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.normaliser, f)

    def load_normaliser(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            self.normaliser = pickle.load(f)
