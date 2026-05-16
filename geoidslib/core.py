"""
geoidslib.core
~~~~~~~~~~~~~~
High-level GeoIDS façade that wires together all components.

Usage
-----
from geoidslib import GeoIDS

ids = GeoIDS.from_config("configs/default.yaml")
for flow in ids.run(source="pcap", file="traffic.pcap"):
    if flow.is_anomaly:
        print(flow.top_blade_label)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

from geoidslib.algebra.ga_engine import GeometricAlgebraEngine
from geoidslib.detection.detector import AnomalyDetector, AnomalyResult
from geoidslib.features.extractor import FeatureExtractor, FlowRecord
from geoidslib.ingestion.ingester import FlowIngester
from geoidslib.output.alert_writer import BaseWriter, ConsoleWriter, JSONFileWriter

logger = logging.getLogger(__name__)


class GeoIDS:
    """
    High-level façade for the GeoIDS intrusion detection system.

    Parameters
    ----------
    ga_engine : GeometricAlgebraEngine | None
    feature_extractor : FeatureExtractor | None
    anomaly_detector : AnomalyDetector | None
    flow_ingester : FlowIngester | None
    writers : list[BaseWriter] | None   Output backends.
    """

    def __init__(
        self,
        ga_engine: Optional[GeometricAlgebraEngine] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        flow_ingester: Optional[FlowIngester] = None,
        writers: Optional[List[BaseWriter]] = None,
    ):
        self.ga_engine = ga_engine or GeometricAlgebraEngine()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.anomaly_detector = anomaly_detector or AnomalyDetector(ga_engine=self.ga_engine)
        self.flow_ingester = flow_ingester or FlowIngester()
        self.writers: List[BaseWriter] = writers or [ConsoleWriter()]

        self._total_flows = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Factory: from config file
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> "GeoIDS":
        """
        Build a GeoIDS instance from a YAML configuration file.

        Parameters
        ----------
        config_path : str  Path to YAML file (see configs/default.yaml).
        """
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        ga_cfg = cfg.get("geometric_algebra", {})
        ga_engine = GeometricAlgebraEngine(
            dim=ga_cfg.get("dim", 25),
            p=ga_cfg.get("p", 15),
            q=ga_cfg.get("q", 10),
            max_grade=ga_cfg.get("max_grade", 3),
            sparsity_threshold=ga_cfg.get("sparsity_threshold", 1e-6),
        )

        det_cfg = cfg.get("detector", {})
        detector = AnomalyDetector(
            ga_engine=ga_engine,
            window_size=det_cfg.get("window_size", 10_000),
            reframe_interval=det_cfg.get("reframe_interval", 500),
            forgetting_factor=det_cfg.get("forgetting_factor", 0.99),
            target_fpr=det_cfg.get("target_fpr", 0.01),
        )

        writers: List[BaseWriter] = []
        for w_cfg in cfg.get("writers", [{"type": "console"}]):
            w_type = w_cfg.get("type", "console")
            if w_type == "console":
                writers.append(ConsoleWriter(alerts_only=w_cfg.get("alerts_only", True)))
            elif w_type == "json":
                writers.append(JSONFileWriter(
                    path=w_cfg["path"],
                    alerts_only=w_cfg.get("alerts_only", True),
                ))

        if not writers:
            writers = [ConsoleWriter()]

        return cls(
            ga_engine=ga_engine,
            anomaly_detector=detector,
            writers=writers,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        source: str,
        file: Optional[str] = None,
        interface: Optional[str] = None,
        max_flows: Optional[int] = None,
        **kwargs,
    ) -> Generator[AnomalyResult, None, None]:
        """
        Run the full IDS pipeline and yield AnomalyResult for each flow.

        Parameters
        ----------
        source : str    "pcap" | "netflow" | "live" | "zeek"
        file : str | None
        interface : str | None
        max_flows : int | None   Stop after this many flows.
        """
        logger.info("GeoIDS starting: source=%s file=%s interface=%s", source, file, interface)
        self._start_time = time.time()

        for flow in self.flow_ingester.ingest(source=source, file=file, interface=interface, **kwargs):
            features = self.feature_extractor.extract(flow)
            flow_id = f"{flow.src_ip}:{flow.src_port}-{flow.dst_ip}:{flow.dst_port}/{flow.protocol}"
            result = self.anomaly_detector.process_flow(features, flow_id=flow_id)

            for writer in self.writers:
                writer.write(result)

            self._total_flows += 1
            yield result

            if max_flows and self._total_flows >= max_flows:
                logger.info("max_flows=%d reached, stopping", max_flows)
                break

        self._cleanup()

    def process_flow_record(self, flow: FlowRecord) -> AnomalyResult:
        """Process a single pre-built FlowRecord."""
        features = self.feature_extractor.extract(flow)
        flow_id = f"{flow.src_ip}:{flow.src_port}-{flow.dst_ip}:{flow.dst_port}/{flow.protocol}"
        result = self.anomaly_detector.process_flow(features, flow_id=flow_id)
        for writer in self.writers:
            writer.write(result)
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        elapsed = time.time() - self._start_time
        fps = self._total_flows / max(elapsed, 0.001)
        return {
            "total_flows": self._total_flows,
            "elapsed_s": round(elapsed, 2),
            "flows_per_second": round(fps, 1),
            **self.anomaly_detector.stats,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save detector state and normaliser."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.anomaly_detector.save_state(str(d / "detector_state.pkl"))
        self.feature_extractor.save_normaliser(str(d / "normaliser.pkl"))
        logger.info("GeoIDS state saved to %s", directory)

    def load(self, directory: str) -> None:
        d = Path(directory)
        self.anomaly_detector.load_state(str(d / "detector_state.pkl"))
        self.feature_extractor.load_normaliser(str(d / "normaliser.pkl"))
        logger.info("GeoIDS state loaded from %s", directory)

    def _cleanup(self) -> None:
        for writer in self.writers:
            writer.close()
        logger.info("GeoIDS shutdown. Total flows: %d", self._total_flows)
