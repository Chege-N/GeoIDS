# ─────────────────────────────────────────────────────────────────────────────
# GeoIDS — Production Docker Image
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build:
#   builder  — compiles Cython extension + installs Python deps
#   runtime  — lean final image, no build tools
#
# Usage
# -----
#   docker build -t geoIDS:latest .
#   docker run --rm -v $(pwd)/traffic.pcap:/data/traffic.pcap \
#              geoIDS:latest ingest --source pcap --file /data/traffic.pcap
#
#   # Live capture (requires NET_ADMIN capability):
#   docker run --rm --cap-add NET_ADMIN --network host geoIDS:latest \
#              ingest --source live --interface eth0
#
#   # Dashboard:
#   docker run -p 8050:8050 geoIDS:latest dashboard --host 0.0.0.0
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System deps for dpkt, nfstream, Cython build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpcap-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only files needed for pip install
COPY pyproject.toml ./
COPY geoidslib/ ./geoidslib/
COPY setup_cy.py ./

# Install build tools and dependencies
RUN pip install --no-cache-dir --upgrade pip wheel cython numpy

# Attempt Cython build (non-fatal if it fails — pure Python fallback)
RUN python setup_cy.py build_ext --inplace || echo "Cython build failed — will use pure Python"

# Install full package
RUN pip install --no-cache-dir \
    numpy scipy scikit-learn \
    scapy dpkt nfstream \
    click pydantic pyyaml \
    websockets plotly dash dash-cytoscape pandas \
    rich loguru orjson psutil

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap0.8 \
    tshark \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash geoIDS
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source (includes compiled .so if built)
COPY --from=builder /build/geoidslib ./geoidslib
COPY --from=builder /build/setup_cy.py ./
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/
COPY docs/ ./docs/

# Mount points
RUN mkdir -p /data /var/log/geoIDS && chown -R geoIDS:geoIDS /app /data /var/log/geoIDS

USER geoIDS

# Set PYTHONPATH so geoidslib is importable
ENV PYTHONPATH=/app
ENV GEOIDSIDS_CONFIG=/app/configs/default.yaml

# Health check: verify import succeeds
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from geoidslib import GeoIDS; print('ok')" || exit 1

# Default: show help
ENTRYPOINT ["python", "-m", "geoidslib.cli"]
CMD ["--help"]

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="GeoIDS"
LABEL org.opencontainers.image.description="Hyperdimensional Anomaly Detection via Geometric Algebra"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/Chege-N/GeoIDS"
