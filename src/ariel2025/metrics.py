"""Competition metric: Gaussian Log-Likelihood (GLL).

The Ariel Data Challenge 2025 is scored with the sum over wavelengths and
targets of the Gaussian log-likelihood of the ground truth spectrum under a
predicted 1-D Gaussian (mean + uncertainty) per wavelength point. The raw GLL
is normalized between a *naive* baseline (train-set mean/sigma) and an *ideal*
prediction (perfect mean, noise floor of 10 ppm on AIRS / 1 ppm on FGS1).

The final score is clipped to the interval [0, 1]; higher is better.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pandas.api.types
import scipy.stats

# FGS1 is a single spectral point whose importance is doubled by the metric.
FGS1_RELATIVE_WEIGHT = 57.846

# Instrument noise floors used for the "ideal" prediction reference.
FGS1_SIGMA_TRUE = 1e-6
AIRS_SIGMA_TRUE = 1e-5

# Sanity floor for predicted uncertainties (prevents log(0)).
SIGMA_FLOOR = 10 ** -15


class ParticipantVisibleError(Exception):
    """Raised when a submission violates the competition format."""


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    naive_mean: float,
    naive_sigma: float,
    fgs_sigma_true: float = FGS1_SIGMA_TRUE,
    airs_sigma_true: float = AIRS_SIGMA_TRUE,
    fgs_weight: float = 1.0,
) -> float:
    """Official competition scoring function (Kaggle/NeurIPS reference).

    Parameters
    ----------
    solution:
        Ground-truth spectra, shape (n_samples, n_wavelengths).
    submission:
        Predicted spectra followed by predicted uncertainties, shape
        (n_samples, 2 * n_wavelengths).
    row_id_column_name:
        Name of the planet-id column present in both frames.
    naive_mean, naive_sigma:
        Mean / std of the training labels, used to anchor the normalization.
    fgs_sigma_true, airs_sigma_true:
        Instrument noise floors for the ideal prediction.
    fgs_weight:
        Extra weight given to the FGS1 channel.
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError("Negative values in the submission")
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f"Submission column {col} must be a number")

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError("Wrong number of columns in the submission")

    y_pred = submission.iloc[:, :n_wavelengths].values
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=SIGMA_FLOOR, a_max=None)
    sigma_true = np.append(np.array([fgs_sigma_true]), np.ones(n_wavelengths - 1) * airs_sigma_true)
    y_true = solution.values

    gll_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    gll_true = scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true))
    gll_mean = scipy.stats.norm.logpdf(
        y_true, loc=naive_mean * np.ones_like(y_true), scale=naive_sigma * np.ones_like(y_true)
    )

    ind_scores = (gll_pred - gll_mean) / (gll_true - gll_mean)
    weights = np.append(np.array([fgs_weight]), np.ones(len(solution.columns) - 1))
    weights = weights * np.ones_like(ind_scores)
    submit_score = np.average(ind_scores, weights=weights)
    return float(np.clip(submit_score, 0.0, 1.0))


def gaussian_log_likelihood(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_pred: np.ndarray,
    naive_mean: float,
    naive_sigma: float,
) -> float:
    """Fast array-based GLL score used for XGBoost OOF calibration searches.

    ``n_wavelengths`` determines the reference: 283 -> full spectrum with the
    weighted FGS1 channel; 1 -> FGS1 only; otherwise AIRS-only.
    """
    y_true, y_pred, sigma_pred = map(np.asarray, (y_true, y_pred, sigma_pred))
    sigma_pred = np.clip(sigma_pred, SIGMA_FLOOR, None)
    n_wavelengths = y_true.shape[1]

    if n_wavelengths == 283:
        sigma_true = np.append(np.array([FGS1_SIGMA_TRUE]), np.ones(n_wavelengths - 1) * AIRS_SIGMA_TRUE)
        weights = np.append(np.array([FGS1_RELATIVE_WEIGHT]), np.ones(n_wavelengths - 1))
    elif n_wavelengths == 1:
        sigma_true = np.ones(n_wavelengths) * FGS1_SIGMA_TRUE
        weights = np.ones(n_wavelengths)
    else:
        sigma_true = np.ones(n_wavelengths) * AIRS_SIGMA_TRUE
        weights = np.ones(n_wavelengths)

    sigma_true = np.tile(sigma_true, (y_true.shape[0], 1))
    weights = np.tile(weights, (y_true.shape[0], 1))

    gll_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    gll_true = scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true)
    gll_mean = scipy.stats.norm.logpdf(y_true, loc=naive_mean, scale=naive_sigma)

    ind_scores = (gll_pred - gll_mean) / (gll_true - gll_mean + 1e-9)
    final_score = np.average(ind_scores, weights=weights)
    return float(np.clip(final_score, 0.0, 1.0))
