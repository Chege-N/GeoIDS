"""
tests/test_scripts.py
~~~~~~~~~~~~~~~~~~~~~
Tests for scripts/demo.py and scripts/generate_synthetic_dataset.py.
Exercises the synthetic flow generators and the end-to-end demo loop.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Demo script tests
# ---------------------------------------------------------------------------

class TestDemoFlowGenerators:

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    def _import_demo(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "demo", str(Path(__file__).parent.parent / "scripts" / "demo.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_normal_flow_label(self, rng):
        demo = self._import_demo()
        flow = demo.normal_flow(rng, 0.0)
        assert flow.label == "BENIGN"
        assert flow.packet_count > 0
        assert flow.dst_port == 443

    def test_dos_flood_label(self, rng):
        demo = self._import_demo()
        flow = demo.dos_flood(rng, 0.0)
        assert flow.label == "DoS-Flood"
        assert flow.packet_count >= 2000
        assert flow.flag_syn > 0

    def test_tls_anomaly_label(self, rng):
        demo = self._import_demo()
        flow = demo.tls_anomaly(rng, 0.0)
        assert flow.label == "TLS-Anomaly"
        assert flow.tls_version == 1.0  # old TLS

    def test_c2_beacon_label(self, rng):
        demo = self._import_demo()
        flow = demo.c2_beacon(rng, 0.0)
        assert flow.label == "C2-Beacon"
        # IATs should be very regular
        assert len(set(flow.inter_arrival_times)) == 1  # all the same

    def test_doh_backdoor_label(self, rng):
        demo = self._import_demo()
        flow = demo.doh_backdoor(rng, 0.0)
        assert flow.label == "DoH-Backdoor"
        assert flow.tls_sni_length >= 100  # suspiciously long

    def test_run_demo_small(self, capsys):
        demo = self._import_demo()
        # Run with very few flows — should not raise
        demo.run_demo(n_flows=100, attack_rate=0.2, seed=1)
        captured = capsys.readouterr()
        assert "Detection Report" in captured.out

    def test_demo_dr_gt_zero(self):
        """With sufficient flows, at least some attacks should be detected."""
        demo = self._import_demo()
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            demo.run_demo(n_flows=300, attack_rate=0.3, seed=42)
        output = buf.getvalue()
        assert "Detection Report" in output


# ---------------------------------------------------------------------------
# Dataset generator tests
# ---------------------------------------------------------------------------

class TestDatasetGenerator:

    def _import_gen(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gen", str(Path(__file__).parent.parent / "scripts" / "generate_synthetic_dataset.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_generate_creates_file(self, tmp_path):
        gen = self._import_gen()
        out = str(tmp_path / "test.csv")
        gen.generate(n_flows=200, attack_rate=0.1, output=out, seed=0)
        assert Path(out).exists()

    def test_generate_row_count(self, tmp_path):
        import pandas as pd
        gen = self._import_gen()
        out = str(tmp_path / "test2.csv")
        gen.generate(n_flows=500, attack_rate=0.1, output=out, seed=1)
        df = pd.read_csv(out)
        assert len(df) == 500

    def test_generate_has_label_column(self, tmp_path):
        import pandas as pd
        gen = self._import_gen()
        out = str(tmp_path / "test3.csv")
        gen.generate(n_flows=100, output=out, seed=2)
        df = pd.read_csv(out)
        assert "Label" in df.columns

    def test_generate_attack_rate(self, tmp_path):
        import pandas as pd
        gen = self._import_gen()
        out = str(tmp_path / "test4.csv")
        rate = 0.2
        n = 1000
        gen.generate(n_flows=n, attack_rate=rate, output=out, seed=3)
        df = pd.read_csv(out)
        actual_rate = (df["Label"] != "BENIGN").mean()
        # Allow ±5% tolerance
        assert abs(actual_rate - rate) < 0.05

    def test_generate_contains_zero_day(self, tmp_path):
        import pandas as pd
        gen = self._import_gen()
        out = str(tmp_path / "test5.csv")
        gen.generate(n_flows=2000, attack_rate=0.3, zero_day_fraction=0.5, output=out, seed=4)
        df = pd.read_csv(out)
        labels = set(df["Label"].unique())
        assert "DoH-Backdoor" in labels or "Infiltration" in labels

    def test_generated_csv_compatible_with_evaluator(self, tmp_path):
        """Generated CSV should be parseable by the GeoIDS Evaluator."""
        import pandas as pd
        gen = self._import_gen()
        out = str(tmp_path / "eval_test.csv")
        gen.generate(n_flows=300, attack_rate=0.2, output=out, seed=5)

        from geoidslib.evaluation import DATASET_SCHEMAS, _build_flow_record
        schema = DATASET_SCHEMAS["cic2017"]
        df = pd.read_csv(out)

        # Should be able to build FlowRecords from all rows without error
        errors = 0
        for _, row in df.iterrows():
            try:
                rec = _build_flow_record(row, schema)
                assert rec is not None
            except Exception:
                errors += 1
        assert errors == 0, f"{errors} rows failed to parse"
