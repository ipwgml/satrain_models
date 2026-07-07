"""
Minimal setup.py for C++ extension - only needed for build configuration.
The actual package metadata is in pyproject.toml.

The C++ extension is optional: if compilation fails (e.g. no C++ compiler
available) the package is installed without it and satrain_models falls back
to the pure-Python BMCI implementation. Set the environment variable
SATRAIN_MODELS_NO_EXT=1 to skip building the extension entirely.
"""
import os

from setuptools import setup

SKIP_EXT = os.environ.get("SATRAIN_MODELS_NO_EXT", "0").lower() in ("1", "true", "yes")

ext_modules = []
if not SKIP_EXT:
    try:
        from setuptools import Extension
        import pybind11
        import numpy as np

        ext_modules = [
            Extension(
                "satrain_models.bmci_c",
                sources=["satrain_models/bmci_c.cpp"],
                include_dirs=[
                    pybind11.get_include(),
                    np.get_include(),
                ],
                language="c++",
                extra_compile_args=["-std=c++14", "-fopenmp", "-O3"],
                extra_link_args=["-fopenmp"],
                # Compilation failures are non-fatal; the package falls back
                # to the pure-Python BMCI implementation.
                optional=True,
            ),
        ]
    except ImportError as exc:
        print(f"Skipping BMCI C++ extension (missing build dependency: {exc}).")

# Minimal setup - everything else is in pyproject.toml
setup(ext_modules=ext_modules, zip_safe=False)
