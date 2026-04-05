"""
Setup configuration for the pfa_tqft package.

Install in development mode:
    pip install -e .

Install normally:
    pip install .
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name             = "pfa_tqft",
    version          = "1.0.0",
    description      = (
        "Phase-Fidelity-Aware Truncated Quantum Fourier Transform "
        "for Scalable Phase Estimation on NISQ Hardware"
    ),
    long_description      = long_description,
    long_description_content_type = "text/markdown",
    author            = "Akoramurthy S, Surendiran B",
    author_email      = "akoramurthy@ece.nitpy.ac.in",
    url               = "https://github.com/nit-ece-qamp/pfa-tqft",
    license           = "MIT",
    packages          = find_packages(exclude=["tests*", "experiments*", "notebooks*"]),
    python_requires   = ">=3.9",
    install_requires  = [
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "pillow>=9.0",
    ],
    extras_require    = {
        "dev": ["pytest>=7.0", "pytest-cov", "black", "isort"],
        "nb":  ["jupyter", "ipykernel", "ipywidgets"],
    },
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords = (
        "quantum computing NISQ phase estimation "
        "quantum Fourier transform truncation"
    ),
    project_urls = {
        "Paper" : "https://quantum-journal.org",
        "GitHub": "https://github.com/nit-ece-qamp/pfa-tqft",
    },
)
