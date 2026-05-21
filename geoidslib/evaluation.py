"""
geoidslib.evaluation
~~~~~~~~~~~~~~~~~~~~~
Evaluation harness for GeoIDS against public IDS datasets.

Supported datasets
------------------
* CIC-IDS2017
* CSE-CIC-IDS2018
* UNSW-NB15

The evaluator reads a labelled CSV (flows with ground-truth labels),
runs GeoIDS in streaming mode, and produces a classification report
with special emphasis on zero-day (unseen-attack) detection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)

from geoidslib.core import GeoIDS
from geoidslib.features.extractor import FlowRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset schemas
# ---------------------------------------------------------------------------

# Column mappings: dataset_name → {our_field: csv_column}
DATASET_SCHEMAS = {
    "cic2017": {
        "label_col": "Label",
        "normal_label": "BENIGN",
        "packet_count": "Total Fwd Packets",
        "byte_count": "Total Length of Fwd Packets",
        "flow_duration": "Flow Duration",
        "protocol": "Protocol",
        "dst_port": "Destination Port",
        "mean_packet_size": "Average Packet Size",
        "std_packet_size": "Packet Length Std",
        "mean_iat": "Flow IAT Mean",
        "std_iat": "Flow IAT Std",
        "bytes_per_second": "Flow Bytes/s",
        "packets_per_second": "Flow Packets/s",
    },
    "cic2018": {
        "label_col": "Label",
        "normal_label": "Benign",
        "packet_count": "Total Fwd Packets",
        "byte_count": "Total Length of Fwd Pkts",
        "flow_duration": "Flow Duration",
        "protocol": "Protocol",
        "dst_port": "Dst Port",
    },
    "unswnb15": {
        "label_col": "label",
        "normal_label": "0",  # 0 = normal, 1 = attack
        "packet_count": "spkts",
        "byte_count": "sbytes",
        "flow_duration": "dur",
        "protocol": "proto",
        "dst_port": "dport",
        "src_ip": "srcip",
        "dst_ip": "dstip",
    },
    "custom": {
        "label_col": "label",
        "normal_label": "normal",
    },
}


def _build_flow_record(row: pd.Series, schema: dict) -> FlowRecord:
    """Convert a dataframe row to a FlowRecord using the dataset schema."""

    def _get(field: str, default=0) -> float:
        col = schema.get(field)
        if col and col in row.index:
            val = row[col]
            try:
                return float(val)
            except Exception:
                return default
        return default

    return FlowRecord(
        src_ip=str(row.get(schema.get("src_ip", ""), "0.0.0.0")),
        dst_ip=str(row.get(schema.get("dst_ip", ""), "0.0.0.0")),
        src_port=int(_get("src_port")),
        dst_port=int(_get("dst_port")),
        protocol=int(_get("protocol", 6)),
        packet_count=int(_get("packet_count", 10)),
        byte_count=int(_get("byte_count", 1000)),
        flow_start=0.0,
        flow_end=max(_get("flow_duration", 1.0), 1e-6),
        fwd_bytes=int(_get("byte_count", 1000)),
    )


class Evaluator:
    """
    Evaluation harness for GeoIDS.

    Parameters
    ----------
    dataset : str    One of the DATASET_SCHEMAS keys.
    config_path : str  Path to GeoIDS YAML config.
    """

    def __init__(self, dataset: str = "cic2017", config_path: str = "configs/default.yaml"):
        self.dataset = dataset
        self.schema = DATASET_SCHEMAS.get(dataset, DATASET_SCHEMAS["custom"])
        self.config_path = config_path

    def run(
        self,
        data_path: str,
        train_weeks: int = 1,
        zero_day_attacks: list[str] | None = None,
        chunk_size: int = 50_000,
    ) -> dict:
        """
        Run full evaluation.

        Parameters
        ----------
        data_path : str       CSV file path.
        train_weeks : int     Use first N weeks (days) for training.
        zero_day_attacks : list of str | None
            Attack labels considered "zero-day" (test-only).
            If None, all attack types not seen in training are zero-day.
        chunk_size : int      Rows per CSV chunk.

        Returns
        -------
        dict  Evaluation report.
        """
        cfg = Path(self.config_path)
        ids = GeoIDS.from_config(str(cfg)) if cfg.exists() else GeoIDS(writers=[])

        # Silence alert writers for evaluation
        ids.writers = []

        schema = self.schema
        label_col = schema["label_col"]
        normal_label = schema["normal_label"]

        logger.info("Loading dataset: %s", data_path)
        df = pd.read_csv(data_path, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=[label_col])

        # Clean infinite/NaN numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        total = len(df)
        split = max(1, int(total * (train_weeks / 7)))
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

        # Identify zero-day attack labels (in test but not in train)
        train_labels = set(train_df[label_col].unique())
        test_labels = set(test_df[label_col].unique())

        if zero_day_attacks is None:
            zero_day_attacks = [
                label for label in test_labels
                if label != normal_label and label not in train_labels
            ]

        logger.info("Zero-day attack types: %s", zero_day_attacks)

        # Training pass
        logger.info("Training phase…")
        for _, row in train_df.iterrows():
            flow = _build_flow_record(row, schema)
            flow.label = str(row[label_col])
            ids.process_flow_record(flow)

        ids.anomaly_detector.force_recompute_reference()

        # Evaluation pass
        logger.info("Evaluation phase…")
        y_true: list[int] = []
        y_pred: list[int] = []
        y_true_zd: list[int] = []
        y_pred_zd: list[int] = []
        scores: list[float] = []

        for _, row in test_df.iterrows():
            flow = _build_flow_record(row, schema)
            label = str(row[label_col])
            flow.label = label
            result = ids.process_flow_record(flow)

            is_attack = int(label != normal_label)
            is_detected = int(result.is_anomaly)

            y_true.append(is_attack)
            y_pred.append(is_detected)
            scores.append(result.score)

            # Zero-day specific
            if label in zero_day_attacks or label == normal_label:
                y_true_zd.append(is_attack)
                y_pred_zd.append(is_detected)

        # Metrics
        report = self._compute_metrics(y_true, y_pred, scores, y_true_zd, y_pred_zd)
        report["dataset"] = self.dataset
        report["total_flows"] = total
        report["train_flows"] = len(train_df)
        report["test_flows"] = len(test_df)
        report["zero_day_attack_types"] = zero_day_attacks
        report["ids_stats"] = ids.stats

        return report

    def _compute_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        scores: list[float],
        y_true_zd: list[int],
        y_pred_zd: list[int],
    ) -> dict:
        y_t = np.array(y_true)
        y_p = np.array(y_pred)

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        fpr = fp / max(fp + tn, 1)
        detection_rate = tp / max(tp + fn, 1)  # recall
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * detection_rate / max(precision + detection_rate, 1e-9)

        try:
            auc = roc_auc_score(y_t, scores)
        except Exception:
            auc = 0.0

        result = {
            "detection_rate": round(detection_rate, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }

        # Zero-day specific
        if y_true_zd:
            y_tz = np.array(y_true_zd)
            y_pz = np.array(y_pred_zd)
            cm_zd = confusion_matrix(y_tz, y_pz, labels=[0, 1])
            tn_z, fp_z, fn_z, tp_z = cm_zd.ravel() if cm_zd.shape == (2, 2) else (0, 0, 0, 0)
            dr_zd = tp_z / max(tp_z + fn_z, 1)
            fpr_zd = fp_z / max(fp_z + tn_z, 1)
            pr_zd = tp_z / max(tp_z + fp_z, 1)
            f1_zd = 2 * pr_zd * dr_zd / max(pr_zd + dr_zd, 1e-9)
            result["zero_day"] = {
                "detection_rate": round(dr_zd, 4),
                "fpr": round(fpr_zd, 4),
                "f1": round(f1_zd, 4),
            }

        return result
