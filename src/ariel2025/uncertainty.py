"""Per-target uncertainty (sigma) estimation.

The GLL metric rewards well-calibrated uncertainties as much as accurate means.
Instead of a constant sigma for every target, we estimate a per-planet noise
level from the out-of-transit vs in-transit variance of the preprocessed light
curves, and map it onto a *soft multiplier* of the base sigma:

    k_planet = sqrt( sigma_rel_planet / median(sigma_rel) )

with the multiplier gently clipped so the estimate stays conservative.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .calibration import _phase_detector_signal
from .config import Config

EPS = 1e-12


def _sigma_multipliers(preprocessed_data: np.ndarray, cfg: Config, channel: str) -> np.ndarray:
    """Compute per-planet relative-noise multipliers (FGS1 or AIRS white curves)."""
    delta = cfg.transit.optimization_delta
    sig_rel: list = []

    for single in preprocessed_data:
        if channel == "FGS1":
            # Use the AIRS white curve to locate phases, then measure on FGS1.
            white = savgol_filter(single[:, 1:].mean(axis=1), cfg.transit.savgol_window, 2)
            target = single[:, 0]
        else:
            # Phase detection uses the smoothed white curve, while the
            # variance is measured on the unsmoothed curve.
            raw_white = np.nanmean(single[:, 1:], axis=1)
            white = savgol_filter(raw_white, cfg.transit.savgol_window, 2)
            target = raw_white

        p1, p2 = _phase_detector_signal(white, cfg)
        p1 = max(delta, p1)
        p2 = min(len(white) - delta - 1, p2)

        oot_left = target[: p1 - delta] if p1 - delta > 0 else np.empty(0, target.dtype)
        oot_right = target[p2 + delta :] if (p2 + delta) < target.size else np.empty(0, target.dtype)
        oot = np.concatenate([oot_left, oot_right]) if (oot_left.size + oot_right.size) else oot_left
        inn = target[p1 + delta : max(p1 + delta, p2 - delta)]

        if oot.size == 0 or inn.size == 0:
            sig_rel.append(np.nan)
            continue

        n_oot, n_in = len(oot), len(inn)
        var_oot = np.nanvar(oot, ddof=1)
        var_in = np.nanvar(inn, ddof=1)
        oot_mean = float(np.nanmean(oot)) if np.isfinite(np.nanmean(oot)) else float(np.nanmean(target))
        sigma_rel = np.sqrt(var_oot / max(n_oot, 1) + var_in / max(n_in, 1)) / max(oot_mean, EPS)
        sig_rel.append(sigma_rel)

    s = np.asarray(sig_rel, dtype=float)
    mask = np.isfinite(s) & (s > 0)
    med = float(np.nanmedian(s[mask])) if mask.any() else 1.0

    k = np.ones_like(s)
    if med > 0 and np.isfinite(med):
        k[mask] = np.sqrt(s[mask] / med)
    return k


def estimate_sigma_fgs(preprocessed_data: np.ndarray, cfg: Config) -> np.ndarray:
    """Per-planet sigma for the FGS1 channel (shape ``(n_planets,)``)."""
    k = _sigma_multipliers(preprocessed_data, cfg, channel="FGS1")
    k = np.clip(k, 0.8, 1.25)
    return k * cfg.sigma_base


def estimate_sigma_air(preprocessed_data: np.ndarray, cfg: Config) -> np.ndarray:
    """Per-planet sigma for all AIRS-CH0 channels (shape ``(n_planets,)``)."""
    k = _sigma_multipliers(preprocessed_data, cfg, channel="AIRS")
    k = np.clip(k, 0.90, 1.20)
    return k * cfg.sigma_base
