"""End-to-end neural-network inference.

This module reproduces the winning inference flow:

1. Calibrate & preprocess the raw signals (see :mod:`ariel2025.calibration`).
2. Recover per-planet transit depths with the physical transit model
   (see :mod:`ariel2025.transit`).
3. Predict the FGS1 spectrum point with a small residual MLP.
4. Predict the full AIRS spectrum by averaging several 10-fold CV ensembles of
   :class:`ariel2025.spectra_models.ResNetMLP2` (each trained on a different
   feature set / architecture).
5. Estimate per-planet uncertainties from the light-curve noise.
6. Assemble the submission, then blend it with the XGBoost submission.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .calibration import SignalCalibrator
from .config import Config
from .spectra_models import ResNetMLP, load_cv_models_and_scalers
from .submission import SubmissionGenerator
from .transit import TransitModel
from .uncertainty import estimate_sigma_air, estimate_sigma_fgs


def build_input_frame(
    cfg: Config, star_info: pd.DataFrame, transit_depths: np.ndarray
) -> pd.DataFrame:
    """Merge star metadata with the recovered transit depth.

    ``transit_depth`` is stored in parts-per-ten-thousand (matching the scale
    used to train the residual MLPs), so it is multiplied by 10,000 here.
    """
    input_df = star_info.copy()
    input_df.insert(0, "transit_depth", (transit_depths * 10000).to_numpy())
    return input_df


def predict_fgs_from_checkpoint(
    cfg: Config, X_tensor: torch.Tensor, checkpoint_path: str, num_blocks: int = 80, dropout: float = 0.2
) -> np.ndarray:
    """Predict the single FGS1 spectrum point from the trained MLP checkpoint."""
    model = ResNetMLP(input_dim=len(cfg.spectrum_model.features), num_blocks=num_blocks, dropout_rate=dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    return preds / 10000


def predict_airs_ensemble(
    cfg: Config,
    input_df: pd.DataFrame,
    cv_dir: str,
    features: list,
    num_blocks: int = 35,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    n_folds: int = 10,
) -> np.ndarray:
    """Average the predictions of a 10-fold CV ensemble of AIRS residual MLPs."""
    models, scaler_X, scaler_y = load_cv_models_and_scalers(
        cv_dir,
        cfg,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        dropout_rate=dropout,
        n_folds=n_folds,
    )

    X = input_df[features].values.astype(np.float64)
    X_scaled = scaler_X.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float64)

    with torch.no_grad():
        preds_scaled = np.mean([model(X_tensor).numpy() for model in models], axis=0)
    return scaler_y.inverse_transform(preds_scaled)


def run_nn_pipeline(cfg: Config) -> pd.DataFrame:
    """Execute the complete neural-network inference flow and return the submission."""
    calibrator = SignalCalibrator(cfg)
    preprocessed_data = calibrator.process_all()

    # --- 1. physical transit depth ---
    transit_model = TransitModel(cfg)
    transit_depths = transit_model.predict_all(preprocessed_data)

    # --- 2. per-planet uncertainties ---
    sigma_fgs = estimate_sigma_fgs(preprocessed_data, cfg)
    sigma_air = estimate_sigma_air(preprocessed_data, cfg)

    # --- 3. meta-features for the residual MLPs ---
    star_info = pd.read_csv(cfg.data_path / f"{cfg.dataset}_star_info.csv", index_col="planet_id")
    star_info.index = star_info.index.astype(int)
    input_df = build_input_frame(cfg, star_info, transit_depths)

    # --- 4. FGS1 spectrum ---
    if cfg.fgs_checkpoint is None:
        raise ValueError("cfg.fgs_checkpoint must point to the trained FGS1 model")
    X_fgs = torch.tensor(
        input_df[cfg.spectrum_model.features].values.astype("float32"), dtype=torch.float32
    )
    fgs_preds = predict_fgs_from_checkpoint(cfg, X_fgs, str(cfg.fgs_checkpoint))

    # --- 5. AIRS spectrum (averaged over the configured CV ensembles) ---
    airs_ensembles: list = []
    for spec in cfg.airs_cv_ensembles:
        airs_ensembles.append(
            predict_airs_ensemble(cfg, input_df, **spec)
        )
    airs_preds = np.mean(airs_ensembles, axis=0)

    # --- 6. assemble the NN submission ---
    generator = SubmissionGenerator(cfg)
    return generator.create(
        fgs_preds, airs_preds, sigma_fgs=sigma_fgs, sigma_air=sigma_air, filename="submission_nn.csv"
    )
