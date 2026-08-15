"""Central configuration for the Ariel Data Challenge 2025 pipeline.

All tunable hyper-parameters used by the calibration, transit-model and
machine-learning stages live here so the whole pipeline can be configured
from a single place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SensorConfig:
    """Instrument-specific parameters used during calibration/preprocessing."""

    raw_shape: List[int] = field(default_factory=list)
    linear_corr_shape: tuple = ()
    dt_pattern: tuple = (0.1, 0.1)  # (base integration time, increment for odd frames)
    binning: int = 1
    # FGS1 (0.60-0.80 um) and AIRS-CH0 (1.95-3.90 um) each contribute a single
    # column to the final preprocessed feature stack:
    n_columns: int = 1


@dataclass
class TransitConfig:
    """Parameters for the physically-motivated transit depth model."""

    phase_detection_slice: slice = field(default_factory=lambda: slice(30, 140))
    optimization_delta: int = 11
    polynomial_degree: int = 3
    savgol_window: int = 20  # smoothing window for phase detection / sigma estimation
    predict_savgol_window: int = 23  # smoothing window used inside TransitModel.predict
    n_jobs: int = 3


@dataclass
class SpectrumModelConfig:
    """Hyper-parameters for the residual MLP that maps meta-features to spectra."""

    features: List[str] = field(default_factory=lambda: ["transit_depth", "Rs", "i"])
    hidden_dim: int = 256
    num_blocks: int = 35
    dropout_rate: float = 0.1
    n_cv_folds: int = 10


@dataclass
class XGBConfig:
    """Hyper-parameters for the XGBoost mean/sigma ensemble."""

    n_folds: int = 5
    random_state: int = 42
    xgb_mean_params: dict = field(
        default_factory=lambda: {
            "objective": "reg:squarederror",
            "n_estimators": 1200,
            "learning_rate": 0.02,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "tree_method": "hist",
        }
    )
    xgb_sigma_params: dict = field(
        default_factory=lambda: {
            "objective": "reg:squarederror",
            "n_estimators": 600,
            "learning_rate": 0.025,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "tree_method": "hist",
        }
    )
    calibration_scaling_factors: List[float] = field(
        default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2]
    )
    calibration_additive_factors: List[float] = field(
        default_factory=lambda: [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    )


@dataclass
class BlendConfig:
    """Weights used when blending the NN and XGBoost submissions."""

    xgb_wl_weight: float = 0.2
    xgb_sigma_weight: float = 0.4
    high_wl_weight: float = 0.75  # weight of the higher-scoring blend in the final mix
    high_sigma_weight: float = 0.75


@dataclass
class Config:
    """Top-level configuration container for the full pipeline."""

    data_path: Path = Path("/kaggle/input/ariel-data-challenge-2025")
    dataset: str = "test"  # "train" or "test"
    output_dir: Path = Path("./outputs")

    # Physical scaling factors tuned to match the emission level of the targets
    # (empirically calibrated on the public leaderboard).
    scale: float = 0.96
    sigma_base: float = 0.00055

    # Wavelength cropping for the AIRS-CH0 spectral axis (keep [CUT_INF, CUT_SUP)).
    cut_inf: int = 39
    cut_sup: int = 321

    # Per-instrument sensor parameters.
    sensor_config: dict = field(
        default_factory=lambda: {
            "AIRS-CH0": SensorConfig(
                raw_shape=[11250, 32, 356],
                linear_corr_shape=(6, 32, 356),
                dt_pattern=(0.1, 4.5),
                binning=30,
                n_columns=282,
            ),
            "FGS1": SensorConfig(
                raw_shape=[135000, 32, 32],
                linear_corr_shape=(6, 32, 32),
                dt_pattern=(0.1, 0.1),
                binning=360,
                n_columns=1,
            ),
        }
    )

    transit: TransitConfig = field(default_factory=TransitConfig)
    spectrum_model: SpectrumModelConfig = field(default_factory=SpectrumModelConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)

    # Paths to trained NN checkpoints / scalers (from Kaggle datasets or local runs).
    fgs_checkpoint: Optional[Path] = None
    airs_checkpoint: Optional[Path] = None
    airs_cv_dir: Optional[Path] = None
    airs_cv_dir_2: Optional[Path] = None
    airs_cv_dir_3: Optional[Path] = None
    xgb_artifacts_dir: Optional[Path] = None

    # AIRS CV ensembles used in the NN pipeline. Each entry is passed as kwargs
    # to :func:`ariel2025.inference.predict_airs_ensemble`:
    #   dict(cv_dir=..., features=[...], num_blocks=..., dropout=...)
    airs_cv_ensembles: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def build_config(dataset: str = "test", **overrides) -> Config:
    """Instantiate a :class:`Config`, optionally overriding any field.

    Path-like fields passed as strings are converted to :class:`pathlib.Path`.
    """
    cfg = Config(dataset=dataset)
    path_fields = {
        "data_path",
        "output_dir",
        "fgs_checkpoint",
        "airs_checkpoint",
        "airs_cv_dir",
        "airs_cv_dir_2",
        "airs_cv_dir_3",
        "xgb_artifacts_dir",
    }
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config field: {key}")
        if key in path_fields and value is not None:
            value = Path(value)
        setattr(cfg, key, value)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg
