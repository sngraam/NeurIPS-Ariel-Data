"""Weighted blending of the neural-network and gradient-boosting submissions.

Two architectures excel on different aspects of the task, so averaging their
predictions (with channel-dependent weights) boosts the score. This module
implements the two blending stages used for the final submission:

1. Model-level blend: ``wl = 0.2 * xgb + 0.8 * nn``, ``sigma = 0.4 * xgb + 0.6 * nn``.
2. Best-of blend: a heavier weight on the higher-scoring candidate file.
"""

from __future__ import annotations

import pandas as pd


def blend_two(
    path_a: str,
    path_b: str,
    wl_weight_a: float = 0.2,
    sigma_weight_a: float = 0.4,
    output_path: str = "blend.csv",
) -> pd.DataFrame:
    """Weighted blend of two submission CSVs.

    Parameters
    ----------
    path_a, path_b:
        Paths to the two submissions.
    wl_weight_a:
        Weight of file A for the spectrum (``wl_*``) columns.
    sigma_weight_a:
        Weight of file A for the uncertainty (``sigma_*``) columns.
    """
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    assert list(df_a.columns) == list(df_b.columns), "Submission columns must match"

    wl_cols = [c for c in df_a.columns if c.startswith("wl_")]
    sigma_cols = [c for c in df_a.columns if c.startswith("sigma_")]

    blend = df_a.copy()
    blend[wl_cols] = wl_weight_a * df_a[wl_cols] + (1 - wl_weight_a) * df_b[wl_cols]
    blend[sigma_cols] = sigma_weight_a * df_a[sigma_cols] + (1 - sigma_weight_a) * df_b[sigma_cols]
    blend.to_csv(output_path, index=False)
    return blend


def final_blend(
    high_path: str,
    low_path: str,
    wl_weight_high: float = 0.75,
    sigma_weight_high: float = 0.75,
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Blend the higher- and lower-scoring candidates for the final submission."""
    return blend_two(
        high_path,
        low_path,
        wl_weight_a=wl_weight_high,
        sigma_weight_a=sigma_weight_high,
        output_path=output_path,
    )
