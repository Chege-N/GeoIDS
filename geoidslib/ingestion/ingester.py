"""
geoidslib.ingestion.ingester
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Network flow ingestion engine supporting:
  * PCAP files (via scapy / dpkt)
  * Live packet capture (via scapy)
  * NetFlow v5/v9 / IPFIX (via nfstream)
  * Zeek / Bro log files (TSV/JSON)

All sources produce FlowRecord objects consumed by the AnomalyDetector.

Ring-buffer design
------------------
A collections.deque (capacity = ring_buffer_size) acts as a lock-free
single-producer/single-consumer ring buffer between the ingest thread
and the detection thread (Python GIL ensures safe access for deque ops).
For production use, replace with a multiprocessing.Queue or mmap ring.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Generator, Iterator
from pathlib import Path

from geoidslib.features.extractor import FlowRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseIngester(ABC):
    """Abstract base class for all ingesters."""

    def __init__(self, ring_buffer_size: int = 100_000):
        self._buffer: deque[FlowRecord] = deque(maxlen=ring_buffer_size)
        self._total_flows: int = 0
        self._running: bool = False

    @abstractmethod
    def ingest(self, **kwargs) -> Generator[FlowRecord, None, None]:
        """Yield FlowRecord objects."""
        ...

    @property
    def stats(self) -> dict:
        return {
            "total_flows": self._total_flows,
            "buffer_size": len(self._buffer),
        }


# ---------------------------------------------------------------------------
# PCAP ingester
# ---------------------------------------------------------------------------

def _parse_tcp_flags(flags: int) -> tuple[int, int, int, int]:
    """Extract SYN, FIN, RST, ACK counts from TCP flags bitmask."""
    syn = 1 if flags & 0x02 else 0
    fin = 1 if flags & 0x01 else 0
    rst = 1 if flags & 0x04 else 0
    ack = 1 if flags & 0x10 else 0
    return syn, fin, rst, ack


class PcapIngester(BaseIngester):
    """
    Ingest flows from a PCAP file.

    Assembles 5-tuple flows by grouping packets with the same
    (src_ip, dst_ip, src_port, dst_port, protocol) key.
    Flows are emitted when idle for more than `flow_timeout` seconds
    or when the PCAP ends.

    Parameters
    ----------
    flow_timeout : float
        Seconds of inactivity before a flow is closed (default 60).
    max_packets : int | None
        Cap on packets to read (for testing).
    """

    def __init__(
        self,
        flow_timeout: float = 60.0,
        max_packets: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flow_timeout = flow_timeout
        self.max_packets = max_packets

    def ingest(self, path: str, **_) -> Generator[FlowRecord, None, None]:  # type: ignore[override]
        """
        Yield FlowRecord objects from a PCAP file.

        Parameters
        ----------
        path : str  Path to .pcap / .pcapng file.
        """
        try:
            import dpkt
        except ImportError as err:
            raise ImportError("dpkt is required for PCAP ingestion: pip install dpkt") from err

        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"PCAP file not found: {path}")

        # active_flows: flow_key → FlowRecord (in-progress)
        active_flows: dict[tuple, FlowRecord] = {}
        # last_seen: flow_key → last packet timestamp
        last_seen: dict[tuple, float] = {}

        packets_read = 0

        def _flush_idle(current_ts: float) -> Iterator[FlowRecord]:
            expired = [
                k for k, ts in last_seen.items()
                if current_ts - ts > self.flow_timeout
            ]
            for k in expired:
                yield active_flows.pop(k)
                del last_seen[k]

        with open(path, "rb") as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except Exception:
                # Try pcapng
                f.seek(0)
                try:
                    pcap = dpkt.pcapng.Reader(f)
                except Exception as exc:
                    raise ValueError(f"Cannot open PCAP: {exc}") from exc

            for ts, buf in pcap:
                if self.max_packets and packets_read >= self.max_packets:
                    break
                packets_read += 1

                # Flush idle flows
                yield from _flush_idle(ts)

                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except Exception:
                    continue

                if not isinstance(eth.data, dpkt.ip.IP):
                    continue

                ip = eth.data
                src_ip = _ip_to_str(ip.src)
                dst_ip = _ip_to_str(ip.dst)
                proto = ip.p
                pkt_len = len(buf)

                src_port, dst_port = 0, 0
                flags = 0

                if proto == 6 and isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    src_port, dst_port = tcp.sport, tcp.dport
                    flags = tcp.flags
                elif proto == 17 and isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    src_port, dst_port = udp.sport, udp.dport

                key = (src_ip, dst_ip, src_port, dst_port, proto)

                if key not in active_flows:
                    active_flows[key] = FlowRecord(
                        src_ip=src_ip, dst_ip=dst_ip,
                        src_port=src_port, dst_port=dst_port,
                        protocol=proto,
                        flow_start=ts,
                    )

                flow = active_flows[key]
                flow.packet_count += 1
                flow.byte_count += pkt_len
                flow.packet_sizes.append(pkt_len)
                flow.flow_end = ts

                if last_seen.get(key):
                    flow.inter_arrival_times.append(ts - last_seen[key])
                last_seen[key] = ts

                syn, fin, rst, ack = _parse_tcp_flags(flags)
                flow.flag_syn += syn
                flow.flag_fin += fin
                flow.flag_rst += rst
                flow.flag_ack += ack

                # Approximate upload/download by direction
                flow.fwd_bytes += pkt_len

        # Emit remaining active flows
        for flow in active_flows.values():
            yield flow
            self._total_flows += 1

        logger.info(
            "PCAP ingestion complete: %d packets, %d flows", 
            packets_read, self._total_flows
        )


def _ip_to_str(ip_bytes: bytes) -> str:
    """Convert 4-byte IP to dotted-decimal string."""
    if len(ip_bytes) == 4:
        return ".".join(str(b) for b in ip_bytes)
    return "0.0.0.0"


# ---------------------------------------------------------------------------
# NetFlow / IPFIX ingester
# ---------------------------------------------------------------------------

class NetFlowIngester(BaseIngester):
    """
    Ingest flows from nfstream (NetFlow v5/v9, IPFIX, or live capture).

    Parameters
    ----------
    active_timeout : int   seconds (default 120)
    idle_timeout   : int   seconds (default 60)
    """

    def __init__(
        self,
        active_timeout: int = 120,
        idle_timeout: int = 60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.active_timeout = active_timeout
        self.idle_timeout = idle_timeout

    def ingest(  # type: ignore[override]
        self,
        source: str,
        is_online: bool = False,
        **_,
    ) -> Generator[FlowRecord, None, None]:
        """
        Parameters
        ----------
        source : str
            Path to PCAP file, or network interface name for live capture.
        is_online : bool
            True for live capture.
        """
        try:
            from nfstream import NFStreamer
        except ImportError as err:
            raise ImportError("nfstream is required: pip install nfstream") from err

        streamer = NFStreamer(
            source=source,
            decode_tunnels=True,
            statistical_analysis=True,
            splt_analysis=10,
            n_dissections=20,
            active_timeout=self.active_timeout,
            idle_timeout=self.idle_timeout,
        )

        for nf_flow in streamer:
            rec = FlowRecord(
                src_ip=nf_flow.src_ip,
                dst_ip=nf_flow.dst_ip,
                src_port=nf_flow.src_port,
                dst_port=nf_flow.dst_port,
                protocol=nf_flow.protocol,
                packet_count=nf_flow.bidirectional_packets,
                byte_count=nf_flow.bidirectional_bytes,
                flow_start=nf_flow.bidirectional_first_seen_ms / 1000.0,
                flow_end=nf_flow.bidirectional_last_seen_ms / 1000.0,
                fwd_bytes=nf_flow.src2dst_bytes,
                bwd_bytes=nf_flow.dst2src_bytes,
            )
            # nfstream gives statistical arrays
            if hasattr(nf_flow, "bidirectional_mean_ps"):
                rec.packet_sizes = []  # we compute from stats
            self._total_flows += 1
            yield rec


# ---------------------------------------------------------------------------
# Zeek log ingester
# ---------------------------------------------------------------------------

class ZeekIngester(BaseIngester):
    """
    Ingest flows from Zeek (Bro) conn.log or ssl.log files.

    Supports:
    - TSV format (default Zeek output)
    - JSON format (via json-logs policy)

    Parameters
    ----------
    log_type : str   "conn" | "ssl" (default "conn")
    """

    def __init__(self, log_type: str = "conn", **kwargs):
        super().__init__(**kwargs)
        self.log_type = log_type

    def ingest(self, path: str, **_) -> Generator[FlowRecord, None, None]:  # type: ignore[override]
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Zeek log not found: {path}")

        with open(path_obj) as f:
            first = f.readline()

        # Detect format
        if first.startswith("{"):
            yield from self._ingest_json(path)
        else:
            yield from self._ingest_tsv(path)

    def _ingest_tsv(self, path: str) -> Generator[FlowRecord, None, None]:
        fields: list[str] = []
        separator = "\t"

        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#separator"):
                    sep_hex = line.split()[-1]
                    separator = bytes.fromhex(sep_hex.replace("\\x", "")).decode()
                elif line.startswith("#fields"):
                    fields = line.split(separator)[1:]
                elif line.startswith("#"):
                    continue
                else:
                    parts = line.split(separator)
                    if len(parts) != len(fields):
                        continue
                    row = dict(zip(fields, parts, strict=False))
                    rec = self._parse_conn_row(row)
                    if rec:
                        self._total_flows += 1
                        yield rec

    def _ingest_json(self, path: str) -> Generator[FlowRecord, None, None]:
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    rec = self._parse_conn_row(row)
                    if rec:
                        self._total_flows += 1
                        yield rec
                except json.JSONDecodeError:
                    continue

    def _parse_conn_row(self, row: dict) -> FlowRecord | None:
        def _float(k: str, default: float = 0.0) -> float:
            try:
                return float(row.get(k, default))
            except (ValueError, TypeError):
                return default

        def _int(k: str, default: int = 0) -> int:
            try:
                return int(float(row.get(k, default)))
            except (ValueError, TypeError):
                return default

        proto_str = str(row.get("proto", "tcp")).lower()
        proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
        proto = proto_map.get(proto_str, 6)

        ts = _float("ts")
        duration = _float("duration", 0.0)

        rec = FlowRecord(
            src_ip=str(row.get("id.orig_h", "0.0.0.0")),
            dst_ip=str(row.get("id.resp_h", "0.0.0.0")),
            src_port=_int("id.orig_p"),
            dst_port=_int("id.resp_p"),
            protocol=proto,
            packet_count=_int("orig_pkts") + _int("resp_pkts"),
            byte_count=_int("orig_bytes") + _int("resp_bytes"),
            flow_start=ts,
            flow_end=ts + duration,
            fwd_bytes=_int("orig_bytes"),
            bwd_bytes=_int("resp_bytes"),
        )
        return rec


# ---------------------------------------------------------------------------
# FlowIngester — unified façade
# ---------------------------------------------------------------------------

class FlowIngester:
    """
    Unified flow ingestion façade.

    Usage
    -----
    ingester = FlowIngester()
    for flow in ingester.ingest(source="pcap", file="traffic.pcap"):
        process(flow)
    """

    def __init__(self, ring_buffer_size: int = 100_000):
        self._ring_buffer_size = ring_buffer_size
        self._pcap = PcapIngester(ring_buffer_size=ring_buffer_size)
        self._netflow = NetFlowIngester(ring_buffer_size=ring_buffer_size)
        self._zeek = ZeekIngester(ring_buffer_size=ring_buffer_size)

    def ingest(
        self,
        source: str,
        file: str | None = None,
        interface: str | None = None,
        callback: Callable[[FlowRecord], None] | None = None,
        **kwargs,
    ) -> Generator[FlowRecord, None, None]:
        """
        Unified ingest method.

        Parameters
        ----------
        source : str    "pcap" | "netflow" | "live" | "zeek"
        file : str      Path to PCAP or log file.
        interface : str Network interface for live capture.
        callback : callable | None  Called for each flow (alternative to iteration).
        """
        if source == "pcap":
            if not file:
                raise ValueError("file path required for pcap source")
            gen = self._pcap.ingest(path=file, **kwargs)
        elif source == "netflow":
            if not file:
                raise ValueError("file path required for netflow source")
            gen = self._netflow.ingest(source=file, **kwargs)
        elif source == "live":
            iface = interface or "eth0"
            gen = self._netflow.ingest(source=iface, is_online=True, **kwargs)
        elif source == "zeek":
            if not file:
                raise ValueError("file path required for zeek source")
            gen = self._zeek.ingest(path=file, **kwargs)
        else:
            raise ValueError(f"Unknown source: {source}. Use pcap|netflow|live|zeek")

        for flow in gen:
            if callback:
                callback(flow)
            yield flow

    @property
    def stats(self) -> dict:
        return {
            "pcap": self._pcap.stats,
            "netflow": self._netflow.stats,
            "zeek": self._zeek.stats,
        }
