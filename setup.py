"""
Minimal setup.py for C++ extension - only needed for build configuration.
The actual package metadata is in pyproject.toml.
"""
from setuptools import setup, Extension
import pybind11
import numpy as np

# Define the C++ extension
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
    ),
]

# Minimal setup - everything else is in pyproject.toml
setup(ext_modules=ext_modules, zip_safe=False)