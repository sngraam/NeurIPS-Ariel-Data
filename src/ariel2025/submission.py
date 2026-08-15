"""Build the final submission file from spectrum and uncertainty predictions.

The competition submission has 567 columns per planet: 283 spectrum values
(``wl_...``) followed by 283 uncertainties (``sigma_...``). The first
wavelength (index 0) belongs to FGS1; wavelengths 1..282 belong to AIRS-CH0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import Config

N_WAVELENGTHS = 283


class SubmissionGenerator:
    """Assembles a submission DataFrame from predicted spectra and sigmas."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sample_submission = pd.read_csv(
            cfg.data_path / "sample_submission.csv", index_col="planet_id"
        )

    def create(
        self,
        fgs_preds: np.ndarray,
        airs_preds: np.ndarray,
        sigma_fgs: Optional[np.ndarray] = None,
        sigma_air: Optional[np.ndarray] = None,
        filename: str = "submission.csv",
    ) -> pd.DataFrame:
        """Build the submission.

        Parameters
        ----------
        fgs_preds:
            FGS1 spectrum predictions, shape ``(n_planets, 1)``.
        airs_preds:
            AIRS spectrum predictions, shape ``(n_planets, 282)``.
        sigma_fgs:
            Optional per-planet FGS1 uncertainty, shape ``(n_planets,)``.
        sigma_air:
            Optional per-planet AIRS uncertainty, shape ``(n_planets,)``.
        """
        n_mu = self.sample_submission.shape[1] // 2

        mu = np.concatenate([np.asarray(fgs_preds).reshape(-1, 1), np.asarray(airs_preds).reshape(-1, 282)], axis=1)
        mu = np.clip(mu, 0, None)

        sigmas = np.full((mu.shape[0], n_mu), self.cfg.sigma_base, dtype=float)
        if sigma_fgs is not None:
            sigmas[:, 0] = np.clip(np.asarray(sigma_fgs, dtype=float).reshape(-1), 1e-6, 0.1)
        if sigma_air is not None:
            sigmas[:, 1:] = np.clip(np.asarray(sigma_air, dtype=float).reshape(-1, 1), 1e-6, 0.1)

        submission_df = pd.DataFrame(
            np.concatenate([mu, sigmas], axis=1),
            columns=self.sample_submission.columns,
            index=self.sample_submission.index,
        )
        submission_df.to_csv(self.cfg.output_dir / filename)
        return submission_df
