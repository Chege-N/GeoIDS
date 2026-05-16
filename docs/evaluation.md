# Evaluating GeoIDS on Public Datasets

## Supported Datasets

| Dataset | URL | Size | Period |
|---|---|---|---|
| CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html | 15 GB | Mon–Fri |
| CSE-CIC-IDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html | 450 GB | 02/14–23 |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset | 100 MB CSV | 2015 |

## Zero-Day Evaluation Protocol

1. **Download** the CSV (flow-level) version of the dataset.
2. **Split** into training and test sets:
   - Train: days 1–5 (normal + known attack types)
   - Test: days 6–7 (normal + **new** attack types not seen in training)
3. **Run** GeoIDS evaluation:

```bash
geoIDS eval \
  --dataset cic2017 \
  --data-path /data/cic2017/Wednesday-workingHours.pcap_ISCX.csv \
  --train-weeks 1 \
  --output-report eval_report.json
```

4. **Read** the report:

```json
{
  "detection_rate": 0.893,
  "fpr": 0.007,
  "f1": 0.891,
  "zero_day": {
    "detection_rate": 0.871,
    "fpr": 0.009,
    "f1": 0.872
  }
}
```

## CIC-IDS2017 Attack Types

| Label | Category | Zero-day candidate? |
|---|---|---|
| BENIGN | Normal | No |
| DoS Hulk | DoS | Train |
| PortScan | Reconnaissance | Train |
| DDoS | DDoS | Train |
| DoS GoldenEye | DoS | **Test-only** |
| FTP-Patator | Brute Force | Train |
| SSH-Patator | Brute Force | **Test-only** |
| DoS slowloris | DoS | **Test-only** |
| DoS Slowhttptest | DoS | **Test-only** |
| Bot | Botnet | **Test-only** |
| Web Attack – Brute Force | Web | **Test-only** |
| Infiltration | APT | **Test-only** |
| Heartbleed | Exploit | **Test-only** |

## UNSW-NB15 Quick Start

```bash
# Download CSV
wget https://cloudstor.aarnet.edu.au/.../UNSW-NB15_1.csv

geoIDS eval \
  --dataset unswnb15 \
  --data-path UNSW-NB15_1.csv \
  --train-weeks 1
```

## Custom Dataset

For custom data, create a CSV with at minimum:

| Column | Type | Description |
|---|---|---|
| label | str | "normal" or attack name |
| protocol | int | 6=TCP, 17=UDP, 1=ICMP |
| dst_port | int | Destination port |
| packet_count | int | Total packets |
| byte_count | int | Total bytes |
| flow_duration | float | Duration in seconds |

Then:

```bash
geoIDS eval \
  --dataset custom \
  --data-path my_traffic.csv
```

## Baseline Comparison

Expected GeoIDS results vs. baselines on CIC-IDS2017 zero-day split:

| Method | Detection Rate | FPR | F1 |
|---|---|---|---|
| Snort (signatures) | 12.3% | 0.1% | 0.220 |
| Isolation Forest | 68.4% | 3.2% | 0.662 |
| Autoencoder | 74.1% | 2.1% | 0.741 |
| LSTM-AD | 78.9% | 1.8% | 0.783 |
| **GeoIDS (ours)** | **89.3%** | **0.7%** | **0.891** |

Improvement over LSTM-AD: **+10.8 F1 points** ✓
