"""
geoidslib.output.alert_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Alert output backends for GeoIDS:

* JSONFileWriter  — append newline-delimited JSON to a log file
* WebSocketWriter — push JSON alerts over WebSocket (async)
* SIEMWriter      — format alerts for Splunk / ELK (CEF or JSON)
* ConsoleWriter   — rich terminal output for development
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from geoidslib.detection.detector import AnomalyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseWriter(ABC):
    @abstractmethod
    def write(self, result: AnomalyResult) -> None: ...

    def write_batch(self, results: list[AnomalyResult]) -> None:
        for r in results:
            self.write(r)

    @abstractmethod
    def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# JSON file writer
# ---------------------------------------------------------------------------

class JSONFileWriter(BaseWriter):
    """
    Write alerts (and optionally all flow scores) to a NDJSON file.

    Parameters
    ----------
    path : str | Path   Output file path.
    alerts_only : bool  If True, only write flows with is_anomaly=True.
    """

    def __init__(self, path: str, alerts_only: bool = True):
        self.path = Path(path)
        self.alerts_only = alerts_only
        self._fh = self.path.open("a", encoding="utf-8")
        logger.info("JSONFileWriter opened: %s", self.path)

    def write(self, result: AnomalyResult) -> None:
        if self.alerts_only and not result.is_anomaly:
            return
        line = json.dumps(result.to_dict(), ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Console writer (development)
# ---------------------------------------------------------------------------

class ConsoleWriter(BaseWriter):
    """Colourised terminal output using rich if available."""

    def __init__(self, alerts_only: bool = True, verbose: bool = False):
        self.alerts_only = alerts_only
        self.verbose = verbose
        try:
            from rich.console import Console
            #from rich.table import Table
            self._rich_console = Console()
            self._use_rich = True
        except ImportError:
            self._use_rich = False

    def write(self, result: AnomalyResult) -> None:
        if self.alerts_only and not result.is_anomaly:
            return

        if self._use_rich:
            self._write_rich(result)
        else:
            self._write_plain(result)

    def _write_rich(self, result: AnomalyResult) -> None:

        colour = "bold red" if result.is_anomaly else "green"
        label = "🚨 ALERT" if result.is_anomaly else "✅ NORMAL"
        msg = (
            f"[{colour}]{label}[/{colour}]  "
            f"flow={result.flow_id}  "
            f"score={result.score:.4f}  "
            f"threshold={result.threshold:.4f}  "
            f"confidence={result.confidence:.2%}"
        )
        if result.is_anomaly and result.top_blade_label:
            msg += f"\n  → Anomalous feature: [yellow]{result.top_blade_label}[/yellow]"
        self._rich_console.print(msg)

    def _write_plain(self, result: AnomalyResult) -> None:
        tag = "[ALERT]" if result.is_anomaly else "[NORMAL]"
        print(
            f"{tag} flow={result.flow_id} score={result.score:.4f} "
            f"threshold={result.threshold:.4f} confidence={result.confidence:.2%}",
            file=sys.stderr if result.is_anomaly else sys.stdout,
        )


# ---------------------------------------------------------------------------
# SIEM writer (Splunk / ELK via CEF or JSON)
# ---------------------------------------------------------------------------

class SIEMWriter(BaseWriter):
    """
    Format and send alerts to a SIEM.

    Supports:
    - Splunk HTTP Event Collector (HEC)
    - ELK via Logstash HTTP input
    - Generic JSON over HTTP POST

    Parameters
    ----------
    endpoint : str   Full URL (e.g. https://splunk:8088/services/collector)
    token : str      Auth token (Splunk HEC token or Bearer token)
    format : str     "splunk_hec" | "elk" | "cef"
    """

    def __init__(
        self,
        endpoint: str,
        token: str = "",
        format: str = "splunk_hec",
        batch_size: int = 50,
        timeout_s: float = 5.0,
    ):
        self.endpoint = endpoint
        self.token = token
        self.format = format
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self._pending: list[AnomalyResult] = []

    def write(self, result: AnomalyResult) -> None:
        if not result.is_anomaly:
            return
        self._pending.append(result)
        if len(self._pending) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        try:
            import urllib.request
            payload = self._format_batch(self._pending)
            req = urllib.request.Request(
                self.endpoint,
                data=payload.encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": (
                        f"Splunk {self.token}" if self.format == "splunk_hec" 
                        else f"Bearer {self.token}",
                     )
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                logger.debug("SIEM response: %d", resp.status)
        except Exception as exc:
            logger.warning("SIEM send failed: %s", exc)
        finally:
            self._pending.clear()

    def _format_batch(self, results: list[AnomalyResult]) -> str:
        if self.format == "splunk_hec":
            events = [
                {"time": r.timestamp, "event": r.to_dict(), "sourcetype": "geoIDS_alert"}
                for r in results
            ]
            return "\n".join(json.dumps(e) for e in events)
        elif self.format == "cef":
            lines = []
            for r in results:
                cef = (
                    f"CEF:0|GeoIDS|GeoIDS|1.0|anomaly|Blade Anomaly Detected|"
                    f"{int(r.confidence * 10)}|"
                    f"src={r.flow_id} "
                    f"outcome={r.score:.4f} "
                    f"reason={r.top_blade_label}"
                )
                lines.append(cef)
            return "\n".join(lines)
        else:
            return json.dumps([r.to_dict() for r in results])

    def close(self) -> None:
        self._flush()


# ---------------------------------------------------------------------------
# WebSocket writer (async)
# ---------------------------------------------------------------------------

class WebSocketWriter(BaseWriter):
    """
    Push alerts to connected WebSocket clients.
    Run `start_server()` as an asyncio task.

    Parameters
    ----------
    host : str   Bind host (default "0.0.0.0")
    port : int   Bind port (default 9001)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9001):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    async def start_server(self) -> None:
        try:
            import websockets
        except ImportError as err:
            raise ImportError("websockets is required: pip install websockets") from err

        self._loop = asyncio.get_event_loop()

        async def handler(ws, path=None):
            self._clients.add(ws)
            logger.info("WS client connected: %s", ws.remote_address)
            try:
                await ws.wait_closed()
            finally:
                self._clients.discard(ws)

        self._server = await websockets.serve(handler, self.host, self.port)
        logger.info("WebSocket server started on ws://%s:%d", self.host, self.port)

        while True:
            msg = await self._queue.get()
            if self._clients:
                import websockets.exceptions
                dead = set()
                for ws in list(self._clients):
                    try:
                        await ws.send(msg)
                    except Exception:
                        dead.add(ws)
                self._clients -= dead

    def write(self, result: AnomalyResult) -> None:
        if not result.is_anomaly:
            return
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._queue.put(json.dumps(result.to_dict())),
                self._loop,
            )
