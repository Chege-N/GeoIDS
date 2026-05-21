"""
setup_cy.py
~~~~~~~~~~~
Build the Cython extension module for accelerated GA operations.

Usage
-----
  python setup_cy.py build_ext --inplace

After building, GeoIDS automatically uses the C extension for hot paths
(geometric product, outer product, embedding, distances).  The pure-Python
fallback (multivector.py) is used if the extension is not present.

Requirements
------------
  pip install cython numpy

Optional C++ acceleration (for Eigen-backed operations):
  pip install pybind11
"""

import sys

from setuptools import Extension, setup

try:
    import numpy as np
    numpy_include = np.get_include()
except ImportError:
    print("NumPy not found — cannot build Cython extension.")
    sys.exit(1)

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("Cython not installed — skipping .pyx compilation.")
    print("Install with: pip install cython")

# ---------------------------------------------------------------------------
# Extension definition
# ---------------------------------------------------------------------------

extensions = []

if HAS_CYTHON:
    ga_ops_ext = Extension(
        name="geoidslib.algebra.ga_ops_cy",
        sources=["geoidslib/algebra/ga_ops_cy.pyx"],
        include_dirs=[numpy_include],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            "-std=c++17",
        ],
        extra_link_args=[],
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    extensions = cythonize(
        [ga_ops_ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "embedsignature": True,
        },
        annotate=True,  # generates .html annotation files
    )

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup(
    name="geoIDS-cy",
    version="0.1.0",
    description="Cython extension for GeoIDS geometric algebra hot paths",
    ext_modules=extensions,
    python_requires=">=3.10",
    install_requires=["numpy>=1.26", "cython>=3.0"],
    zip_safe=False,
)
