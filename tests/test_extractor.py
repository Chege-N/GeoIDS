"""
tests/test_extractor.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the FeatureExtractor and FlowRecord.
"""

import pytest
import numpy as np

from geoidslib.features.extractor import (
    FeatureExtractor,
    FlowRecord,
    OnlineNormaliser,
    NUM_FEATURES,
    _shannon_entropy,
    _packet_size_entropy,
    _iat_entropy,
    _ip_entropy,
    _dst_ip_class,
    _port_category,
    _tls_version_norm,
    _protocol_norm,
)


# ---------------------------------------------------------------------------
# Helper / utility function tests
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_shannon_entropy_uniform(self):
        # Uniform distribution → maximum entropy for given n
        # Our normalisation: ent / log2(n+1), so for n=4: log2(4)/log2(5) ≈ 0.861
        vals = [1.0, 1.0, 1.0, 1.0]
        ent = _shannon_entropy(vals)
        assert 0.8 <= ent <= 1.0  # high entropy for uniform

    def test_shannon_entropy_single_value(self):
        ent = _shannon_entropy([5.0])
        assert ent == pytest.approx(0.0, abs=1e-6)

    def test_shannon_entropy_empty(self):
        assert _shannon_entropy([]) == 0.0

    def test_packet_size_entropy_uniform(self):
        sizes = [100, 200, 300, 400]  # all unique → high entropy
        ent = _packet_size_entropy(sizes)
        assert 0.0 <= ent <= 1.0

    def test_iat_entropy_empty(self):
        assert _iat_entropy([]) == 0.0

    def test_iat_entropy_constant(self):
        ent = _iat_entropy([0.01, 0.01, 0.01])
        assert ent == pytest.approx(0.0, abs=1e-3)

    def test_ip_entropy_valid(self):
        ent = _ip_entropy("192.168.1.1")
        assert 0.0 <= ent <= 1.0

    def test_ip_entropy_invalid(self):
        ent = _ip_entropy("not.an.ip")
        assert ent == 0.0

    def test_dst_ip_class_a(self):
        assert _dst_ip_class("10.0.0.1") == pytest.approx(0.0)

    def test_dst_ip_class_b(self):
        assert _dst_ip_class("172.16.0.1") == pytest.approx(0.33)

    def test_dst_ip_class_c(self):
        assert _dst_ip_class("192.168.0.1") == pytest.approx(0.67)

    def test_dst_ip_class_d(self):
        assert _dst_ip_class("239.0.0.1") == pytest.approx(1.0)

    def test_port_category_well_known(self):
        assert _port_category(80) == 0.0
        assert _port_category(443) == 0.0

    def test_port_category_registered(self):
        assert _port_category(8080) == 0.5

    def test_port_category_ephemeral(self):
        assert _port_category(55000) == 1.0

    def test_tls_version_norm(self):
        assert _tls_version_norm(1.3) == pytest.approx(1.0)
        assert _tls_version_norm(1.2) == pytest.approx(0.67)
        assert _tls_version_norm(1.0) == pytest.approx(0.0)

    def test_protocol_norm(self):
        assert _protocol_norm(6) == pytest.approx(0.0)   # TCP
        assert _protocol_norm(17) == pytest.approx(0.5)  # UDP
        assert _protocol_norm(1) == pytest.approx(1.0)   # ICMP
        assert _protocol_norm(99) == pytest.approx(0.5)  # unknown → 0.5


# ---------------------------------------------------------------------------
# FlowRecord defaults
# ---------------------------------------------------------------------------

class TestFlowRecord:

    def test_defaults(self):
        flow = FlowRecord()
        assert flow.src_ip == "0.0.0.0"
        assert flow.protocol == 6
        assert flow.packet_count == 0
        assert flow.label is None

    def test_custom_fields(self):
        flow = FlowRecord(
            src_ip="10.0.0.1",
            dst_ip="8.8.8.8",
            src_port=50000,
            dst_port=443,
            protocol=6,
            packet_count=50,
            byte_count=30000,
        )
        assert flow.dst_port == 443
        assert flow.packet_count == 50


# ---------------------------------------------------------------------------
# OnlineNormaliser
# ---------------------------------------------------------------------------

class TestOnlineNormaliser:

    def test_fit_transform_single(self):
        norm = OnlineNormaliser(dim=3)
        vec = np.array([1.0, 2.0, 3.0])
        result = norm.fit_transform(vec)
        # After single sample, min == max → output is 0 (clipped) or depends on denom
        assert result.shape == (3,)

    def test_fit_transform_range(self):
        norm = OnlineNormaliser(dim=2)
        norm.fit_transform(np.array([0.0, 0.0]))
        result = norm.fit_transform(np.array([1.0, 2.0]))
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)

    def test_clip(self):
        norm = OnlineNormaliser(dim=1, clip=True)
        norm.fit_transform(np.array([0.0]))
        norm.fit_transform(np.array([1.0]))
        # value above max → clipped to 1.0
        out = norm.transform(np.array([5.0]))
        assert out[0] <= 1.0

    def test_reset(self):
        norm = OnlineNormaliser(dim=2)
        norm.update(np.array([1.0, 2.0]))
        norm.reset()
        assert norm._n == 0
        assert np.all(np.isinf(norm._min))


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(online_normalise=False)

    @pytest.fixture
    def rich_flow(self):
        return FlowRecord(
            src_ip="10.0.0.1",
            dst_ip="8.8.8.8",
            src_port=54321,
            dst_port=443,
            protocol=6,
            packet_count=20,
            byte_count=15000,
            packet_sizes=[500, 750, 1000, 200, 300] * 4,
            inter_arrival_times=[0.01, 0.02, 0.005, 0.03] * 4,
            flow_start=1_000_000.0,
            flow_end=1_000_005.0,
            flag_syn=1,
            flag_fin=1,
            flag_ack=18,
            fwd_bytes=10000,
            bwd_bytes=5000,
            tls_handshake_duration=0.35,
            tls_cert_size=2048,
            tls_num_extensions=5,
            tls_sni_length=20,
            tls_cipher_suite_id=0xC02B,
            tls_version=1.3,
            tls_num_san=3,
            tls_cert_lifetime_days=365,
        )

    def test_extract_raw_shape(self, extractor, rich_flow):
        vec = extractor.extract_raw(rich_flow)
        assert vec.shape == (NUM_FEATURES,)

    def test_extract_raw_no_nan(self, extractor, rich_flow):
        vec = extractor.extract_raw(rich_flow)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_extract_shape(self, extractor, rich_flow):
        vec = extractor.extract(rich_flow)
        assert vec.shape == (NUM_FEATURES,)

    def test_extract_with_normalisation(self, rich_flow):
        extractor = FeatureExtractor(online_normalise=True)
        # Feed several flows to warm up normaliser
        for _ in range(5):
            extractor.extract(rich_flow)
        vec = extractor.extract(rich_flow)
        assert vec.shape == (NUM_FEATURES,)
        assert not np.any(np.isnan(vec))

    def test_feature_names_count(self, extractor):
        assert len(extractor.feature_names) == NUM_FEATURES

    def test_extract_empty_flow(self, extractor):
        flow = FlowRecord()
        vec = extractor.extract_raw(flow)
        assert vec.shape == (NUM_FEATURES,)
        assert not np.any(np.isnan(vec))

    def test_extract_batch(self, extractor, rich_flow):
        flows = [rich_flow] * 5
        result = extractor.extract_batch(flows)
        assert result.shape == (5, NUM_FEATURES)

    def test_packet_count_in_feature(self, extractor, rich_flow):
        vec = extractor.extract_raw(rich_flow)
        # Index 0 = packet_count
        assert vec[0] == pytest.approx(rich_flow.packet_count)

    def test_tls_version_feature(self, extractor, rich_flow):
        vec = extractor.extract_raw(rich_flow)
        # Index 17 = tls_version_norm → TLS 1.3 → 1.0
        assert vec[17] == pytest.approx(1.0)

    def test_direction_ratio(self, extractor, rich_flow):
        vec = extractor.extract_raw(rich_flow)
        # Index 24 = flow_direction_ratio = 10000/15000
        expected = 10000 / 15000
        assert vec[24] == pytest.approx(expected, rel=1e-4)

    def test_normaliser_persistence(self, tmp_path, rich_flow):
        extractor = FeatureExtractor(online_normalise=True)
        for _ in range(3):
            extractor.extract(rich_flow)
        path = str(tmp_path / "norm.pkl")
        extractor.save_normaliser(path)

        extractor2 = FeatureExtractor(online_normalise=True)
        extractor2.load_normaliser(path)
        assert extractor2.normaliser._n == extractor.normaliser._n
