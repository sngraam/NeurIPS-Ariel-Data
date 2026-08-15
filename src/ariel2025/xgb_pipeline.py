"""XGBoost ensemble that predicts spectra and uncertainties from features.

The gradient-boosting stage complements the residual MLPs: it uses the rich
hand-crafted feature matrix (see :mod:`ariel2025.features`) and predicts the
full 283-point spectrum in two heads:

1. A **mean model** (``XGB_PARAMS_MEAN``) that regresses the spectrum directly.
2. A **sigma model** (``XGB_PARAMS_SIGMA``) that regresses the absolute error of
   the mean model on each fold, i.e. a learned heteroscedastic uncertainty.

Both models are trained under 5-fold cross-validation; the out-of-fold
uncertainties are then *calibrated* by a grid search over per-instrument
scaling/additive factors using the official GLL metric. The final prediction
averages the fold models and applies the calibration.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor

from .config import Config
from .features import combined_feature_engineering, load_all_observations
from .metrics import gaussian_log_likelihood


def find_best_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_raw: np.ndarray,
    naive_mean: float,
    naive_sigma: float,
    instrument_name: str,
    cfg: Config,
) -> dict:
    """Grid-search the scaling/additive factors that maximize the GLL score."""
    print(f"\n--- [Validation] Searching calibration factors for {instrument_name}... ---")
    best_score, best_scaling, best_additive = -1.0, 1.0, 0.0

    for scaling in cfg.xgb.calibration_scaling_factors:
        for additive in cfg.xgb.calibration_additive_factors:
            sigma_calibrated = (sigma_raw * scaling) + additive
            score = gaussian_log_likelihood(y_true, y_pred, sigma_calibrated, naive_mean, naive_sigma)
            if score > best_score:
                best_score, best_scaling, best_additive = score, scaling, additive

    print(
        f"--- Best factors for {instrument_name}: Scale={best_scaling}, "
        f"Add={best_additive} (Best Score: {best_score:.4f}) ---"
    )
    return {"scaling": best_scaling, "additive": best_additive}


def train(cfg: Config, debug: bool = False, n_jobs: int = -1) -> Path:
    """Train the 5-fold XGBoost mean/sigma ensemble and save all artifacts.

    Returns the directory containing the trained models.
    """
    output_path = Path(cfg.output_dir) / "xgb_models"
    output_path.mkdir(parents=True, exist_ok=True)

    train_labels_df = pd.read_csv(cfg.data_path / "train.csv", index_col="planet_id")
    train_star_info_df = pd.read_csv(
        cfg.data_path / "train_star_info.csv", index_col="planet_id"
    ).loc[train_labels_df.index]

    if debug:
        train_labels_df = train_labels_df.head(100)
        train_star_info_df = train_star_info_df.head(100)

    fgs_signals_train = load_all_observations("train", train_labels_df.index, "FGS1", str(cfg.data_path), n_jobs)
    airs_signals_train = load_all_observations("train", train_labels_df.index, "AIRS-CH0", str(cfg.data_path), n_jobs)

    train_features_df = combined_feature_engineering(fgs_signals_train, airs_signals_train, train_star_info_df)
    train_labels = train_labels_df.reindex(train_features_df.index).values
    naive_mu_train, naive_sigma_train = float(np.mean(train_labels)), float(np.std(train_labels))

    # --- Save the feature column ordering used during training ---
    with open(output_path / "feature_columns.pkl", "wb") as f:
        pickle.dump(train_features_df.columns.tolist(), f)
    with open(output_path / "naive_stats.pkl", "wb") as f:
        pickle.dump({"mean": naive_mu_train, "std": naive_sigma_train}, f)

    # --- Stratified 5-fold cross-validation ---
    stratify_col = pd.cut(train_features_df["fgs_depth_mean"], bins=10, labels=False)
    skf = StratifiedKFold(n_splits=cfg.xgb.n_folds, shuffle=True, random_state=cfg.xgb.random_state)

    oof_mu = np.zeros_like(train_labels)
    oof_sigma = np.zeros_like(train_labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features_df, stratify_col)):
        print("\n" + "*" * 20 + f" FOLD {fold + 1}/{cfg.xgb.n_folds} " + "*" * 20)
        X_train_f, X_val_f = train_features_df.iloc[train_idx], train_features_df.iloc[val_idx]
        y_train_f, y_val_f = train_labels[train_idx], train_labels[val_idx]

        # Stage 1: mean model
        model_mean = MultiOutputRegressor(
            xgb.XGBRegressor(**cfg.xgb.xgb_mean_params, n_jobs=n_jobs), n_jobs=-1
        )
        model_mean.fit(X_train_f, y_train_f)

        # Stage 2: sigma model on the absolute residuals
        y_pred_mu_val = model_mean.predict(X_val_f)
        y_target_sigma_val = np.abs(y_val_f - y_pred_mu_val)
        model_sigma = MultiOutputRegressor(
            xgb.XGBRegressor(**cfg.xgb.xgb_sigma_params, n_jobs=n_jobs), n_jobs=-1
        )
        model_sigma.fit(X_val_f, y_target_sigma_val)

        oof_mu[val_idx] = y_pred_mu_val
        oof_sigma[val_idx] = model_sigma.predict(X_val_f)

        with open(output_path / f"model_mean_fold_{fold}.pkl", "wb") as f:
            pickle.dump(model_mean, f)
        with open(output_path / f"model_sigma_fold_{fold}.pkl", "wb") as f:
            pickle.dump(model_sigma, f)

    # --- OOF validation and calibration ---
    print("\n" + "=" * 50 + "\n===== FINAL OOF VALIDATION & CALIBRATION =====\n" + "=" * 50)
    oof_mu_clipped = oof_mu.clip(0, None)
    oof_sigma_clipped = oof_sigma.clip(1e-10, None)

    y_true_fgs1, y_true_airs = train_labels[:, :1], train_labels[:, 1:]
    oof_mu_fgs1, oof_mu_airs = oof_mu_clipped[:, :1], oof_mu_clipped[:, 1:]
    oof_sigma_fgs1, oof_sigma_airs = oof_sigma_clipped[:, :1], oof_sigma_clipped[:, 1:]

    best_fgs1 = find_best_calibration(
        y_true_fgs1, oof_mu_fgs1, oof_sigma_fgs1, naive_mu_train, naive_sigma_train, "FGS1", cfg
    )
    best_airs = find_best_calibration(
        y_true_airs, oof_mu_airs, oof_sigma_airs, naive_mu_train, naive_sigma_train, "AIRS", cfg
    )

    oof_sigma_cal_fgs1 = (oof_sigma_fgs1 * best_fgs1["scaling"]) + best_fgs1["additive"]
    oof_sigma_cal_airs = (oof_sigma_airs * best_airs["scaling"]) + best_airs["additive"]
    oof_sigma_cal_total = np.hstack([oof_sigma_cal_fgs1, oof_sigma_cal_airs])

    final_cv_score = gaussian_log_likelihood(
        train_labels, oof_mu_clipped, oof_sigma_cal_total, naive_mu_train, naive_sigma_train
    )
    print(f"\n--- Final Combined OOF CV Score: {final_cv_score:.5f} ---")

    calibration_params = {"fgs1": best_fgs1, "airs": best_airs}
    with open(output_path / "calibration_params.pkl", "wb") as f:
        pickle.dump(calibration_params, f)

    print("\nTraining complete. All models saved to:", output_path)
    return output_path


def predict(cfg: Config, artifacts_dir: Optional[Path] = None, n_jobs: int = -1) -> pd.DataFrame:
    """Run the XGBoost pipeline on the test set and return the submission frame."""
    artifacts_dir = Path(artifacts_dir or cfg.xgb_artifacts_dir or (cfg.output_dir / "xgb_models"))

    sample_submission = pd.read_csv(cfg.data_path / "sample_submission.csv", index_col="planet_id")
    test_star_info_df = pd.read_csv(cfg.data_path / "test_star_info.csv", index_col="planet_id")

    fgs_signals_test = load_all_observations("test", test_star_info_df.index, "FGS1", str(cfg.data_path), n_jobs)
    airs_signals_test = load_all_observations("test", test_star_info_df.index, "AIRS-CH0", str(cfg.data_path), n_jobs)
    test_features_df = combined_feature_engineering(fgs_signals_test, airs_signals_test, test_star_info_df)

    print("Loading saved models and parameters...")
    with open(artifacts_dir / "feature_columns.pkl", "rb") as f:
        train_cols = pickle.load(f)
    with open(artifacts_dir / "calibration_params.pkl", "rb") as f:
        calibration_params = pickle.load(f)

    test_features_df = test_features_df.reindex(columns=train_cols).fillna(0)

    # --- Ensemble predictions across folds ---
    all_mu_preds, all_sigma_preds = [], []
    for fold in range(cfg.xgb.n_folds):
        with open(artifacts_dir / f"model_mean_fold_{fold}.pkl", "rb") as f:
            model_mean = pickle.load(f)
        with open(artifacts_dir / f"model_sigma_fold_{fold}.pkl", "rb") as f:
            model_sigma = pickle.load(f)
        all_mu_preds.append(model_mean.predict(test_features_df))
        all_sigma_preds.append(model_sigma.predict(test_features_df))

    y_pred_test = np.mean(all_mu_preds, axis=0).clip(0, None)
    sigma_raw_test = np.mean(all_sigma_preds, axis=0).clip(1e-10, None)

    # --- Apply per-instrument calibration ---
    sigma_pred_fgs1 = (sigma_raw_test[:, :1] * calibration_params["fgs1"]["scaling"]) + calibration_params["fgs1"]["additive"]
    sigma_pred_airs = (sigma_raw_test[:, 1:] * calibration_params["airs"]["scaling"]) + calibration_params["airs"]["additive"]
    sigma_pred_test = np.hstack([sigma_pred_fgs1, sigma_pred_airs])

    pred_df = pd.DataFrame(y_pred_test, index=sample_submission.index, columns=sample_submission.columns[:283])
    sigma_df = pd.DataFrame(sigma_pred_test, index=sample_submission.index, columns=sample_submission.columns[283:])
    submission_df = pd.concat([pred_df, sigma_df], axis=1)

    submission_df.to_csv(cfg.output_dir / "submission_xgb.csv")
    print(f"\n'submission_xgb.csv' created successfully ({submission_df.shape[0]} rows).")
    return submission_df
