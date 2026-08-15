"""Physically-motivated transit depth estimation.

The preprocessed white light curve shows a characteristic transit dip. Rather
than training a black-box regressor directly on the raw flux, we recover a
single scalar *transit depth* with an explicit physical model:

1. Locate the ingress/egress phases on a smoothed white light curve.
2. Model the out-of-transit baseline as a low-order polynomial of time.
3. Search (Nelder-Mead) for the multiplicative depth ``s`` that best joins the
   three segments (out-of-transit / in-transit / out-of-transit) into a smooth
   curve.

The recovered depth is scaled by an empirically calibrated factor to match the
absolute emission level of the ground truth spectra.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from tqdm import tqdm

from .config import Config


class TransitModel:
    """Estimates the transit depth of a white-light curve via optimization."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Phase detection
    # ------------------------------------------------------------------
    def _phase_detector(self, signal: np.ndarray):
        """Return (phase1, phase2): the sharpest drop and rise around the transit."""
        search_slice = self.cfg.transit.phase_detection_slice
        min_index = int(np.argmin(signal[search_slice])) + search_slice.start

        signal1, signal2 = signal[:min_index], signal[min_index:]

        grad1 = np.gradient(signal1)
        grad2 = np.gradient(signal2)
        grad1 /= grad1.max()
        grad2 /= grad2.max()

        phase1 = int(np.argmin(grad1))
        phase2 = int(np.argmax(grad2)) + min_index
        return phase1, phase2

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------
    def _objective_function(self, s: float, signal: np.ndarray, phase1: int, phase2: int) -> float:
        """Fit a polynomial baseline and measure the residual after rescaling
        the in-transit segment by ``(1 + s)``."""
        delta = self.cfg.transit.optimization_delta
        power = self.cfg.transit.polynomial_degree

        if (
            phase1 - delta <= 0
            or phase2 + delta >= len(signal)
            or phase2 - delta - (phase1 + delta) < 5
        ):
            delta = 2

        y = np.concatenate(
            [
                signal[: phase1 - delta],
                signal[phase1 + delta : phase2 - delta] * (1 + s),
                signal[phase2 + delta :],
            ]
        )
        x = np.arange(len(y))

        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.abs(poly(x) - y).mean()
        return error

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, single_preprocessed_signal: np.ndarray) -> float:
        """Estimate the transit depth for a single preprocessed observation."""
        signal_1d = single_preprocessed_signal[:, 1:].mean(axis=1)
        signal_1d = savgol_filter(
            signal_1d, self.cfg.transit.predict_savgol_window, 2
        )

        phase1, phase2 = self._phase_detector(signal_1d)
        delta = self.cfg.transit.optimization_delta
        phase1 = max(delta, phase1)
        phase2 = min(len(signal_1d) - delta - 1, phase2)

        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d, phase1, phase2),
            method="Nelder-Mead",
        )
        return float(result.x[0])

    def predict_all(self, preprocessed_signals: np.ndarray) -> np.ndarray:
        """Estimate transit depths for a batch of observations.

        Returns an array of depth values scaled by the empirically calibrated
        ``cfg.scale`` factor.
        """
        predictions = [
            self.predict(signal) for signal in tqdm(preprocessed_signals, desc="Transit depths")
        ]
        return np.array(predictions) * self.cfg.scale
