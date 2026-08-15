"""Raw telescope signal calibration and light-curve preprocessing.

The Ariel mission returns raw uint16 ``images`` per instrument, together with
calibration frames (dark, dead/hot pixels, flat field, linearity polynomial).
This module restores the physical flux and compresses the 10^4-10^5 time frames
into a compact light curve per instrument:

    ADC restore -> non-neg clamp -> linearity correction -> dark subtraction
    (per-integration pattern) -> flat field -> ROI crop -> CDS -> binning
    -> outlier clipping -> variance weighting (AIRS).

The output of :meth:`SignalCalibrator.process_all` is a stack of shape
``(n_planets, n_bins, n_instrument_columns)`` where column 0 is FGS1 and
columns 1..282 are the AIRS-CH0 spectral bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip

from .config import Config


def apply_linear_corr(linear_corr: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Apply the per-pixel linearity-correction polynomial to a signal.

    ``linear_corr`` has shape ``(degree, X, Y)``; the polynomial is evaluated
    with Horner's method in float64 for numerical stability.
    """
    coeffs = np.flip(linear_corr, axis=0)  # highest degree first
    x = signal.astype(np.float64, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    out[...] = coeffs[0]  # broadcast (X, Y) -> (T, X, Y)
    for k in range(1, coeffs.shape[0]):
        np.multiply(out, x, out=out)
        out += coeffs[k]
    return out.astype(signal.dtype, copy=False)


class SignalCalibrator:
    """Loads, calibrates and preprocesses the raw signal files of a dataset."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.adc_info = pd.read_csv(cfg.data_path / "adc_info.csv")
        star_info = pd.read_csv(
            cfg.data_path / f"{cfg.dataset}_star_info.csv", index_col="planet_id"
        )
        self.planet_ids = star_info.index.astype(int)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _calibrate_single_signal(self, planet_id: int, sensor: str) -> np.ndarray:
        """Calibrate a single (planet, sensor) observation into physical flux.

        Masking policy: DEAD pixels are masked (set to NaN), HOT pixels are kept
        in the data (HOT-KEEP), consistent with the winning configuration.
        """
        sensor_cfg = self.cfg.sensor_config[sensor]
        base = f"{self.cfg.data_path}/{self.cfg.dataset}/{planet_id}"

        signal = pd.read_parquet(f"{base}/{sensor}_signal_0.parquet").to_numpy()
        dark = pd.read_parquet(f"{base}/{sensor}_calibration_0/dark.parquet").to_numpy()
        dead = pd.read_parquet(f"{base}/{sensor}_calibration_0/dead.parquet").to_numpy()
        flat = pd.read_parquet(f"{base}/{sensor}_calibration_0/flat.parquet").to_numpy()
        linear_corr = (
            pd.read_parquet(f"{base}/{sensor}_calibration_0/linear_corr.parquet")
            .values.astype(np.float64)
            .reshape(sensor_cfg.linear_corr_shape)
        )

        # --- reshape & restore ADC dynamic range ---
        signal = signal.reshape(sensor_cfg.raw_shape)
        gain = self.adc_info[f"{sensor}_adc_gain"].iloc[0]
        offset = self.adc_info[f"{sensor}_adc_offset"].iloc[0]
        signal = signal / gain + offset

        hot = sigma_clip(dark, sigma=5, maxiters=5).mask  # monitoring only

        # --- spatial crop ---
        if sensor == "AIRS-CH0":
            signal = signal[:, :, self.cfg.cut_inf : self.cfg.cut_sup]
            linear_corr = linear_corr[:, :, self.cfg.cut_inf : self.cfg.cut_sup]
            dark = dark[:, self.cfg.cut_inf : self.cfg.cut_sup]
            dead = dead[:, self.cfg.cut_inf : self.cfg.cut_sup]
            flat = flat[:, self.cfg.cut_inf : self.cfg.cut_sup]
            hot = hot[:, self.cfg.cut_inf : self.cfg.cut_sup]

        if sensor == "FGS1":
            y0, y1, x0, x1 = 10, 22, 10, 22
            signal = signal[:, y0:y1, x0:x1]
            dark = dark[y0:y1, x0:x1]
            dead = dead[y0:y1, x0:x1]
            flat = flat[y0:y1, x0:x1]
            linear_corr = linear_corr[:, y0:y1, x0:x1]
            hot = hot[y0:y1, x0:x1]

        # --- non-negative clamp before linearity correction ---
        np.maximum(signal, 0, out=signal)

        # --- linearity correction ---
        if sensor == "FGS1":
            signal = apply_linear_corr(linear_corr, signal)
        elif sensor == "AIRS-CH0":
            sl = (slice(None), slice(10, 22), slice(None))
            signal[sl] = apply_linear_corr(linear_corr[:, 10:22, :], signal[sl])
        else:
            signal = apply_linear_corr(linear_corr, signal)

        # --- dark subtraction respecting the alternating integration pattern ---
        base_dt, increment = sensor_cfg.dt_pattern
        signal[::2] -= dark * base_dt
        signal[1::2] -= dark * (base_dt + increment)

        # --- flat field (HOT-KEEP: dead/invalid pixels only) ---
        if sensor == "FGS1":
            flat_roi = flat.astype(signal.dtype, copy=False).copy()
            bad = (dead) | ~np.isfinite(flat_roi) | (flat_roi == 0)
            flat_roi[bad] = np.nan
            signal /= flat_roi
        elif sensor == "AIRS-CH0":
            y0, y1 = 10, 22
            flat_roi = flat[y0:y1, :].astype(signal.dtype, copy=False).copy()
            bad = (dead[y0:y1, :]) | ~np.isfinite(flat_roi) | (flat_roi == 0)
            flat_roi[bad] = np.nan
            signal[:, y0:y1, :] /= flat_roi
        else:
            flat2 = flat.astype(signal.dtype, copy=False).copy()
            bad2 = (dead) | ~np.isfinite(flat2) | (flat2 == 0)
            flat2[bad2] = np.nan
            signal /= flat2

        if getattr(self.cfg, "log_hot_stats", False):
            self._log_hot_stats(planet_id, sensor, hot, dead)

        return signal

    def _log_hot_stats(self, planet_id: int, sensor: str, hot: np.ndarray, dead: np.ndarray) -> None:
        if not hasattr(self, "stats"):
            self.stats = []
        self.stats.append(
            {
                "planet_id": int(planet_id),
                "sensor": sensor,
                "hot_frac": float(np.mean(hot)),
                "dead_frac": float(np.mean(dead)),
            }
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess_calibrated_signal(self, calibrated: np.ndarray, sensor: str) -> np.ndarray:
        """Compress a calibrated signal cube into a binned light curve.

        Steps: ROI mean -> correlated double sampling (odd/even frames) ->
        binning -> robust clipping -> per-wavelength variance weighting.
        """
        sensor_cfg = self.cfg.sensor_config[sensor]
        binning = sensor_cfg.binning

        if sensor == "AIRS-CH0":
            signal_roi = calibrated[:, 10:22, :]
        else:  # FGS1
            signal_roi = calibrated[:, 10:22, 10:22].reshape(calibrated.shape[0], -1)

        mean_signal = np.nanmean(signal_roi, axis=1)
        cds_signal = mean_signal[1::2] - mean_signal[0::2]

        n_bins = cds_signal.shape[0] // binning
        binned = np.array(
            [cds_signal[j * binning : (j + 1) * binning].mean(axis=0) for j in range(n_bins)]
        )

        if sensor == "AIRS-CH0":
            q_lo = np.nanpercentile(binned, 5.0, axis=1, keepdims=True)
            q_hi = np.nanpercentile(binned, 95.0, axis=1, keepdims=True)
            np.clip(binned, q_lo, q_hi, out=binned)

            # Weight wavelengths by inverse variance to suppress noisy spectral bins.
            var = np.nanvar(binned, axis=0, ddof=1)
            med = np.nanmedian(var)
            safe_var = np.where(
                ~np.isfinite(var) | (var <= 0), med if (np.isfinite(med) and med > 0) else 1.0, var
            )
            w = 1.0 / safe_var
            lo, hi = np.nanpercentile(w, 5.0), np.nanpercentile(w, 95.0)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                w = np.clip(w, lo, hi)
            s = np.nansum(w)
            if np.isfinite(s) and s > 0:
                w = w * (binned.shape[1] / s)
            else:
                w = np.ones_like(w)
            binned *= w[None, :]
        else:  # FGS1
            binned = binned.reshape(binned.shape[0], 1)

        return binned

    # ------------------------------------------------------------------
    # Parallel entry points
    # ------------------------------------------------------------------
    def _process_planet_sensor(self, args: dict) -> np.ndarray:
        calibrated = self._calibrate_single_signal(args["planet_id"], args["sensor"])
        return self._preprocess_calibrated_signal(calibrated, args["sensor"])

    def process_all(self) -> np.ndarray:
        """Preprocess every planet for both instruments.

        Returns a stacked array of shape ``(n_planets, n_bins, 283)`` where
        column 0 is the FGS1 white light curve and the rest are AIRS bins.
        """
        try:
            from pqdm.threads import pqdm

            def run(args_list: list) -> list:
                return pqdm(args_list, self._process_planet_sensor, n_jobs=self.cfg.transit.n_jobs)

        except ImportError:
            from multiprocessing import Pool

            def run(args_list: list) -> list:
                with Pool(self.cfg.transit.n_jobs) as pool:
                    return pool.map(self._process_planet_sensor, args_list)

        args_fgs1 = [dict(planet_id=p, sensor="FGS1") for p in self.planet_ids]
        args_airs = [dict(planet_id=p, sensor="AIRS-CH0") for p in self.planet_ids]

        fgs1 = run(args_fgs1)
        airs = run(args_airs)

        return np.concatenate([np.stack(fgs1), np.stack(airs)], axis=2)


def _phase_detector_signal(signal: np.ndarray, cfg: Config):
    sl = cfg.transit.phase_detection_slice
    min_idx = int(np.argmin(signal[sl])) + sl.start
    s1, s2 = signal[:min_idx], signal[min_idx:]

    if s1.size < 3 or s2.size < 3:
        return 0, len(signal) - 1

    g1, g2 = np.gradient(s1), np.gradient(s2)
    g1_max, g2_max = (np.max(g1) if np.size(g1) else 0.0), (np.max(g2) if np.size(g2) else 0.0)
    if g1_max != 0:
        g1 /= g1_max
    if g2_max != 0:
        g2 /= g2_max

    phase1 = int(np.argmin(g1))
    phase2 = int(np.argmax(g2)) + min_idx
    return phase1, phase2
