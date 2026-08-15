"""Ariel Data Challenge 2025 - exoplanet transmission spectra recovery.

A clean, re-structured implementation of the 63rd-place / Bronze-medal solution
for the NeurIPS 2025 Ariel Data Challenge. The package recovers the ground-truth
spectrum of exoplanets from noisy simulated ESA/Ariel telescope observations and
is scored with a Gaussian Log-Likelihood (GLL) metric.

Sub-modules are imported explicitly to keep the package root dependency-light:
``from ariel2025.inference import run_nn_pipeline``.
"""

from .config import Config, build_config

__version__ = "1.0.0"

__all__ = ["Config", "build_config", "__version__"]
