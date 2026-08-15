"""Residual MLP models that map stellar-planetary meta-features to spectra.

Two complementary models are used:

* :class:`ResNetMLP`  - predicts the single FGS1 (visible, 0.60-0.80 um) point.
* :class:`ResNetMLP2` - predicts the full AIRS-CH0 (1.95-3.90 um) spectrum.

Both consume the same compact feature vector
``[transit_depth, Rs, i]`` (optionally extended) and produce spectra in
units of parts-per-ten-thousand, which are then rescaled. An ensemble of
10-fold cross-validated :class:`ResNetMLP2` checkpoints is averaged for the
AIRS spectrum (see :func:`load_cv_models_and_scalers`).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import joblib
import torch
import torch.nn as nn

from .config import Config


class ResidualBlock(nn.Module):
    """Residual MLP block with batch normalization."""

    def __init__(self, dim: int, p: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return self.relu(out + identity)


class ResNetMLP(nn.Module):
    """Small residual MLP used for the single-point FGS1 spectrum."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, p=dropout_rate) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class ResidualBlock2(nn.Module):
    """Residual MLP block without batch normalization (used by the AIRS ensemble)."""

    def __init__(self, dim: int, p: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + identity)


class ResNetMLP2(nn.Module):
    """Residual MLP that predicts the full AIRS-CH0 spectrum (282 bins)."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 282,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock2(hidden_dim, p=dropout_rate) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


# ----------------------------------------------------------------------------
# Optional attention modules (explored but not part of the final submission).
# ----------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=0, keepdim=True)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class AttentionBlock(nn.Module):
    """Multi-head self-attention with residual connection."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        out = self.norm(x + self.dropout(attn_out))
        return out.squeeze(1)


# ----------------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------------
def load_cv_models_and_scalers(
    directory,
    cfg: Config,
    num_blocks: int = 35,
    hidden_dim: int = 256,
    dropout_rate: float = 0.1,
    n_folds: int = 10,
) -> Tuple[List[nn.Module], object, object]:
    """Load the cross-validated AIRS ensemble and its StandardScalers.

    Returns
    -------
    all_models:
        List of ``n_folds`` trained :class:`ResNetMLP2` checkpoints.
    scaler_X:
        Feature scaler used to transform the inputs.
    scaler_y:
        Label scaler whose ``inverse_transform`` restores the spectrum scale.
    """
    scaler_X = joblib.load(os.path.join(directory, "scaler_X.joblib"))
    scaler_y = joblib.load(os.path.join(directory, "scaler_y.joblib"))

    model_params = dict(
        input_dim=len(cfg.spectrum_model.features),
        hidden_dim=hidden_dim,
        output_dim=282,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )

    all_models = []
    for fold in range(1, n_folds + 1):
        model = ResNetMLP2(**model_params).double()
        model_path = os.path.join(directory, f"best_model_airs_cv_fold{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        all_models.append(model)
    return all_models, scaler_X, scaler_y
