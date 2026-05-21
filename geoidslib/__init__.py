"""
GeoIDS — Hyperdimensional Anomaly Detection via Conformal Geometric Algebra.

Public API surface:
    GeoIDS          — high-level façade
    SparseMultivector
    FeatureExtractor
    AnomalyDetector
    FlowIngester
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geoIDS")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

from geoidslib.algebra.ga_engine import GeometricAlgebraEngine
from geoidslib.algebra.multivector import SparseMultivector
from geoidslib.core import GeoIDS
from geoidslib.detection.detector import AnomalyDetector
from geoidslib.features.extractor import FeatureExtractor
from geoidslib.ingestion.ingester import FlowIngester

__all__ = [
    "__version__",
    "GeoIDS",
    "SparseMultivector",
    "GeometricAlgebraEngine",
    "FeatureExtractor",
    "AnomalyDetector",
    "FlowIngester",
]
