"""Feature engineering for the gradient-boosting ensemble.

Beyond the transit-depth scalar recovered by :class:`ariel2025.transit.TransitModel`,
the XGBoost pipeline consumes a rich set of hand-crafted features extracted from
the raw light curves and the stellar-planetary metadata:

* aperture-photometry light curves for FGS1 and AIRS-CH0,
* depth measures, SNR, detrended statistics and noise autocorrelation,
* physics-informed transit quantities (impact parameter, duration, ingress,
  equilibrium temperature, gravity proxy, ...),
* pairwise polynomial interactions of the meta-features.

Each planet may have one or two observations; per-observation statistics are
aggregated with ``mean / std / min / max``.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
from joblib import Parallel, delayed
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm


def extract_light_curve_with_aperture(signal_df: pl.DataFrame, instrument: str) -> np.ndarray:
    """Sum the flux inside an aperture centered on the star's PSF."""
    if instrument == "FGS1":
        image_shape = (32, 32)
        aperture_radius = 3
    else:  # AIRS-CH0
        image_shape = (32, 356)
        aperture_height = 4

    try:
        images = signal_df.to_numpy(zero_copy_only=True).reshape(-1, *image_shape)
    except Exception:
        images = signal_df.to_numpy().reshape(-1, *image_shape)

    median_frame = np.median(images, axis=0)

    if instrument == "FGS1":
        center_y, center_x = np.unravel_index(np.argmax(median_frame), median_frame.shape)
        y_start = max(0, center_y - aperture_radius)
        y_end = min(image_shape[0], center_y + aperture_radius + 1)
        x_start = max(0, center_x - aperture_radius)
        x_end = min(image_shape[1], center_x + aperture_radius + 1)
        aperture_flux = images[:, y_start:y_end, x_start:x_end].sum(axis=(1, 2))
    else:
        vertical_profile = median_frame.sum(axis=1)
        center_y = int(np.argmax(vertical_profile))
        y_start = max(0, center_y - aperture_height)
        y_end = min(image_shape[0], center_y + aperture_height + 1)
        aperture_flux = images[:, y_start:y_end, :].sum(axis=(1, 2))

    return aperture_flux.astype(np.float32)


def _process_single_planet(planet_id, dataset: str, instrument: str, data_path: str):
    """Return the CDS net signals of all observations of one planet."""
    planet_signals: List[np.ndarray] = []
    for obs_count in range(2):
        path = f"{data_path}/{dataset}/{int(planet_id)}/{instrument}_signal_{obs_count}.parquet"
        if os.path.exists(path):
            signal_df = pl.read_parquet(path)
            aperture_flux = extract_light_curve_with_aperture(signal_df, instrument)
            net_signal = aperture_flux[1::2] - aperture_flux[0::2]
            planet_signals.append(net_signal)
    return planet_id, planet_signals


def load_all_observations(
    dataset: str, planet_ids, instrument: str, data_path: str, n_jobs: int = -1
) -> Dict[int, List[np.ndarray]]:
    """Load CDS light curves for every planet of ``instrument`` in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_planet)(pid, dataset, instrument, data_path)
        for pid in tqdm(planet_ids, desc=f"Loading {instrument} observations")
    )
    return {planet_id: signals for planet_id, signals in results if signals}


def autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    """Lag-1 autocorrelation of a (detrended) time series."""
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def get_detrended_features(raw_data: np.ndarray, transit_slice: slice) -> Dict[str, float]:
    """Robust detrending of the light curve and statistics of the residual transit."""
    time_axis = np.arange(raw_data.shape[0])
    out_of_transit_mask = np.ones_like(time_axis, dtype=bool)
    out_of_transit_mask[transit_slice] = False

    coeffs = np.polyfit(time_axis[out_of_transit_mask], raw_data[out_of_transit_mask], 2)
    poly_fit = np.poly1d(coeffs)
    trend = poly_fit(time_axis).clip(1e-6)
    normalized_data = raw_data / trend
    detrended_transit = normalized_data[transit_slice]
    return {
        "detrended_std": float(detrended_transit.std()),
        "detrended_skew": float(scipy.stats.skew(detrended_transit)),
        "detrended_kurtosis": float(scipy.stats.kurtosis(detrended_transit)),
    }


def add_physics_and_interaction_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-informed transit and planet properties."""
    df = features_df.copy()
    transit_depth_proxy = df["fgs_depth_mean"].clip(0)
    df["planet_star_radius_ratio"] = np.sqrt(transit_depth_proxy)

    b = df["impact_parameter_b"]
    p = df["planet_star_radius_ratio"]

    arg1 = (np.sqrt(((1 + p) ** 2 - b**2).clip(0)) / df["sma"]).clip(-1, 1)
    df["transit_duration_est"] = (df["P"] / np.pi) * np.arcsin(arg1)

    arg2 = (np.sqrt(((1 - p) ** 2 - b**2).clip(0)) / df["sma"]).clip(-1, 1)
    t_flat_est = (df["P"] / np.pi) * np.arcsin(arg2)
    df["ingress_egress_duration_est"] = (df["transit_duration_est"] - t_flat_est) / 2.0
    df["ingress_duration_ratio"] = (df["ingress_egress_duration_est"] / df["transit_duration_est"]).fillna(0)

    df["planet_eq_temp_est"] = df["Ts"] * np.sqrt(1 / (2 * df["sma"]))
    df["planet_gravity_proxy"] = df["Mp"] / (p * df["Rs"]) ** 2
    df["eclipsed_light_proxy"] = df["fgs_depth_mean"] * (df["Rs"] ** 2)
    df["duration_period_ratio"] = df["transit_duration_est"] / df["P"]
    return df


def extract_timeseries_features(f_raw: np.ndarray, a_raw: np.ndarray) -> Dict[str, float]:
    """Extract time-series features from FGS1/AIRS raw light curves."""
    features: Dict[str, float] = {}

    # --- FGS1 ---
    fgs_transit_slice = slice(23500, 44000)
    fgs_out_of_transit_mask = np.ones(len(f_raw), dtype=bool)
    fgs_out_of_transit_mask[fgs_transit_slice] = False
    fgs_unobscured_mean = np.mean(f_raw[fgs_out_of_transit_mask])
    fgs_transit = f_raw[fgs_transit_slice]
    fgs_depth = (fgs_unobscured_mean - np.mean(fgs_transit)) / fgs_unobscured_mean
    features["fgs_depth"] = fgs_depth
    for i in range(5):
        features[f"fgs_slice_{i+1}"] = (
            fgs_unobscured_mean - np.mean(fgs_transit[i * 4100 : (i + 1) * 4100])
        ) / fgs_unobscured_mean
    features["fgs_transit_std"] = fgs_transit.std()
    features["fgs_snr"] = fgs_depth / (fgs_transit.std() + 1e-6)
    features.update({f"fgs_{k}": v for k, v in get_detrended_features(f_raw, fgs_transit_slice).items()})
    features["fgs_noise_autocorr"] = autocorrelation(f_raw[fgs_out_of_transit_mask])

    # --- AIRS-CH0 ---
    airs_transit_slice = slice(1950, 3700)
    airs_out_of_transit_mask = np.ones(len(a_raw), dtype=bool)
    airs_out_of_transit_mask[airs_transit_slice] = False
    airs_unobscured_mean = np.mean(a_raw[airs_out_of_transit_mask])
    airs_transit = a_raw[airs_transit_slice]
    airs_depth = (airs_unobscured_mean - np.mean(airs_transit)) / airs_unobscured_mean
    features["airs_depth"] = airs_depth
    slice_len = len(airs_transit) // 5
    for i in range(5):
        features[f"airs_slice_{i+1}"] = (
            airs_unobscured_mean - np.mean(airs_transit[i * slice_len : (i + 1) * slice_len])
        ) / airs_unobscured_mean
    features["airs_transit_std"] = airs_transit.std()
    features["airs_snr"] = airs_depth / (airs_transit.std() + 1e-6)
    features.update({f"airs_{k}": v for k, v in get_detrended_features(a_raw, airs_transit_slice).items()})
    features["airs_noise_autocorr"] = autocorrelation(a_raw[airs_out_of_transit_mask])

    features["depth_ratio_fgs_airs"] = fgs_depth / (airs_depth + 1e-6)
    return features


def combined_feature_engineering(
    fgs_signals: Dict[int, List[np.ndarray]],
    airs_signals: Dict[int, List[np.ndarray]],
    star_info_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature matrix for all planets (aggregated over visits)."""
    all_planet_features_list: List[Dict] = []
    for planet_id in tqdm(star_info_df.index, desc="Feature engineering"):
        num_obs = len(fgs_signals.get(planet_id, []))
        if num_obs == 0:
            continue
        obs_features_list = [
            extract_timeseries_features(fgs_signals[planet_id][i], airs_signals[planet_id][i])
            for i in range(num_obs)
        ]
        obs_features_df = pd.DataFrame(obs_features_list)
        aggregated_feats = {
            f"{col}_{agg}": obs_features_df[col].agg(agg)
            for col in obs_features_df.columns
            for agg in ["mean", "std", "min", "max"]
        }
        aggregated_feats["planet_id"] = planet_id
        aggregated_feats["num_observations"] = num_obs
        all_planet_features_list.append(aggregated_feats)

    features_df = pd.DataFrame(all_planet_features_list).set_index("planet_id")

    meta_df = star_info_df.copy().fillna(star_info_df.median())
    meta_df["impact_parameter_b"] = meta_df["sma"] * np.cos(np.deg2rad(meta_df["i"]))
    meta_df["rho_star_proxy"] = meta_df["Ms"] / (meta_df["Rs"] ** 3)

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_cols = ["Rs", "Ts", "Mp", "P", "impact_parameter_b", "rho_star_proxy"]
    poly_features = poly.fit_transform(meta_df[poly_cols])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(poly_cols), index=meta_df.index)

    combined = pd.concat([features_df, meta_df, poly_df], axis=1)
    final_features_df = combined.loc[:, ~combined.columns.duplicated()]
    final_features_df = add_physics_and_interaction_features(final_features_df)

    print(f"Created {final_features_df.shape[1]} features in total.")
    return final_features_df.fillna(0).replace([np.inf, -np.inf], 0)
