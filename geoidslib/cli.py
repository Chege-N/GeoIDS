"""
geoidslib.cli
~~~~~~~~~~~~~
Click-based CLI for GeoIDS.

Commands
--------
  geoIDS ingest    — run the IDS pipeline
  geoIDS dashboard — launch the Dash dashboard
  geoIDS eval      — evaluate on a labelled dataset
  geoIDS info      — show system info
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click

logger = logging.getLogger("geoIDS")


@click.group()
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
@click.option("--log-file", default=None, help="Log file path")
def main(log_level: str, log_file: str | None):
    """GeoIDS — Hyperdimensional Anomaly Detection for Encrypted Traffic."""
    level = getattr(logging, log_level)
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, handlers=handlers,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--source", required=True,
              type=click.Choice(["pcap", "netflow", "live", "zeek"]),
              help="Input source type")
@click.option("--file", "input_file", default=None, help="Path to PCAP/log file")
@click.option("--interface", default=None, help="Network interface for live capture")
@click.option("--config", default="configs/default.yaml", show_default=True,
              help="Path to YAML config file")
@click.option("--output", default="alerts.json", show_default=True,
              help="Output JSON alerts file")
@click.option("--max-flows", default=None, type=int,
              help="Stop after N flows")
@click.option("--state-dir", default=None,
              help="Directory to load/save detector state")
@click.option("--quiet", is_flag=True, default=False,
              help="Suppress per-flow console output")
def ingest(source, input_file, interface, config, output, max_flows, state_dir, quiet):
    """Run the GeoIDS anomaly detection pipeline."""
    from geoidslib.core import GeoIDS
    from geoidslib.output.alert_writer import JSONFileWriter, ConsoleWriter

    cfg_path = Path(config)
    if cfg_path.exists():
        ids = GeoIDS.from_config(str(cfg_path))
    else:
        click.echo(f"Config {config} not found, using defaults.", err=True)
        from geoidslib.algebra.ga_engine import GeometricAlgebraEngine
        from geoidslib.detection.detector import AnomalyDetector
        ids = GeoIDS(writers=[JSONFileWriter(output)])

    # Add JSON output
    ids.writers.append(JSONFileWriter(output))
    if not quiet:
        ids.writers.append(ConsoleWriter(alerts_only=True))

    if state_dir:
        state_path = Path(state_dir)
        if state_path.exists():
            click.echo(f"Loading state from {state_dir}", err=True)
            ids.load(state_dir)

    click.echo(f"Starting GeoIDS: source={source}", err=True)
    total = 0
    alerts = 0
    try:
        for result in ids.run(source=source, file=input_file, interface=interface,
                               max_flows=max_flows):
            total += 1
            if result.is_anomaly:
                alerts += 1
            if total % 10_000 == 0:
                s = ids.stats
                click.echo(
                    f"  {total:,} flows | {alerts} alerts | "
                    f"{s['flows_per_second']:.0f} fps | "
                    f"threshold={s['current_threshold']:.4f}",
                    err=True,
                )
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
    finally:
        if state_dir:
            ids.save(state_dir)
            click.echo(f"State saved to {state_dir}", err=True)

    click.echo(f"\nDone: {total:,} flows processed, {alerts} alerts → {output}")


# ---------------------------------------------------------------------------
# dashboard command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--port", default=8050, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--alerts-file", default="alerts.json", show_default=True,
              help="JSON alerts file to visualise")
def dashboard(port, host, alerts_file):
    """Launch the GeoIDS Plotly Dash dashboard."""
    try:
        from dashboard.app import create_app
    except ImportError as e:
        click.echo(f"Dashboard dependencies missing: {e}\npip install geoIDS[dev]", err=True)
        sys.exit(1)
    app = create_app(alerts_file=alerts_file)
    click.echo(f"Dashboard at http://{host}:{port}/")
    app.run(host=host, port=port, debug=False)


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--dataset", required=True,
              type=click.Choice(["cic2017", "cic2018", "unswnb15", "custom"]))
@click.option("--data-path", required=True, help="Path to CSV dataset file")
@click.option("--config", default="configs/default.yaml")
@click.option("--train-weeks", default=1, show_default=True,
              help="Number of weeks of data to train on")
@click.option("--output-report", default="eval_report.json")
def eval(dataset, data_path, config, train_weeks, output_report):
    """Evaluate GeoIDS on a labelled intrusion detection dataset."""
    from geoidslib.evaluation import Evaluator
    click.echo(f"Evaluating on {dataset} from {data_path}…")
    ev = Evaluator(dataset=dataset, config_path=config)
    report = ev.run(data_path=data_path, train_weeks=train_weeks)
    import json
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
    click.echo(f"Evaluation complete → {output_report}")
    click.echo(f"  Detection rate : {report.get('detection_rate', 0):.1%}")
    click.echo(f"  FPR            : {report.get('fpr', 0):.2%}")
    click.echo(f"  F1-score       : {report.get('f1', 0):.4f}")


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------

@main.command()
def info():
    """Display GeoIDS version and system information."""
    import platform
    import psutil
    from geoidslib import __version__

    click.echo(f"GeoIDS version : {__version__}")
    click.echo(f"Python         : {sys.version.split()[0]}")
    click.echo(f"Platform       : {platform.system()} {platform.machine()}")
    click.echo(f"CPU cores      : {psutil.cpu_count(logical=True)}")
    mem = psutil.virtual_memory()
    click.echo(f"RAM            : {mem.total / 1e9:.1f} GB ({mem.percent:.1f}% used)")


if __name__ == "__main__":
    main()
