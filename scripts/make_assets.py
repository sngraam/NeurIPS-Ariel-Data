#!/usr/bin/env python3
"""Generate the ML diagram assets used in the README.

Produces the following PNGs under ``assets/``:

* ``pipeline.png``          - end-to-end pipeline architecture
* ``calibration_flow.png``  - signal calibration & preprocessing chain
* ``nn_architecture.png``   - ResNetMLP2 residual-network diagram
* ``transit_lightcurve.png``- synthetic transit light curve with phase detection
* ``blending.png``          - ensemble submission blending

Run with:  python scripts/make_assets.py
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------------
# Style constants
# ----------------------------------------------------------------------------
OUT = "assets"
C_MAIN = "#1f77b4"    # blue
C_ALT = "#2ca02c"     # green
C_ACC = "#d62728"     # red
C_GRAY = "#4d4d4d"
C_BG = "#ffffff"
C_SIGNAL = "#9467bd"
C_FINAL = "#17becf"

BOX_FC = "#eef4fb"
BOX_FC_ALT = "#eef9ee"
BOX_EC = C_MAIN
BOX_EC_ALT = C_ALT


def _box(ax, x, y, w, h, text, fc=BOX_FC, ec=BOX_EC, fontsize=9, weight="bold"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=C_GRAY,
        zorder=3,
    )


def _arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            zorder=1,
        )
    )


def _new_ax(fig_w, fig_h, xlim=(0, 10), ylim=(0, 10)):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    return fig, ax


def _title(ax, text, y=9.6, fontsize=13):
    ax.text(
        5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="#222222",
    )


# ----------------------------------------------------------------------------
# Diagram 1: end-to-end pipeline
# ----------------------------------------------------------------------------
def pipeline():
    fig, ax = _new_ax(12, 6.8)
    _title(ax, "Ariel Data Challenge 2025 - Winning Pipeline", fontsize=14)

    # Row 1: signal processing
    _box(ax, 1.0, 7.6, 1.6, 1.1, "Raw signals +\ncalibration frames")
    _box(ax, 3.4, 7.6, 1.9, 1.1, "Calibration &\npreprocessing", fc=BOX_FC_ALT, ec=BOX_EC_ALT)
    _box(ax, 5.8, 7.6, 1.6, 1.1, "Light curves\n(n_bins x 283)")
    _box(ax, 8.2, 6.4, 1.9, 1.1, "TransitModel\ndepth recovery", fc=BOX_FC_ALT, ec=BOX_EC_ALT)
    _arrow(ax, 1.8, 7.6, 2.45, 7.6)
    _arrow(ax, 4.35, 7.6, 5.0, 7.6)
    _arrow(ax, 6.6, 7.6, 7.25, 6.95)

    # Row 2: shared feature space
    _box(ax, 4.5, 5.2, 3.0, 1.2, "Shared features\n[transit_depth, Rs, i, stats]", fc="#f6f1ff", ec=C_SIGNAL)
    _arrow(ax, 8.2, 5.85, 5.9, 5.75, color=C_SIGNAL)
    ax.text(7.15, 6.0, "transit depth", fontsize=8, color=C_SIGNAL, ha="center")

    # Row 3: predictors
    _box(ax, 1.2, 3.2, 1.8, 1.3, "ResNetMLP\nFGS1 point", fc=BOX_FC_ALT, ec=BOX_EC_ALT)
    _box(ax, 3.9, 3.2, 2.2, 1.3, "ResNetMLP2 ensemble\n(AIRS, 10-fold CV)", fc=BOX_FC_ALT, ec=BOX_EC_ALT)
    _box(ax, 6.7, 3.2, 2.0, 1.3, "XGBoost ensemble\n(mean + sigma)", fc="#fff4e6", ec="#e08a1e")
    _box(ax, 9.2, 3.2, 1.5, 1.3, "Sigma\nestimation", fc="#fff4e6", ec="#e08a1e")

    _arrow(ax, 4.5, 4.6, 1.9, 3.85, color=C_MAIN)
    _arrow(ax, 4.5, 4.6, 3.9, 3.85, color=C_MAIN)
    _arrow(ax, 4.5, 4.6, 6.4, 3.85, color="#e08a1e")
    _arrow(ax, 5.9, 4.6, 9.0, 3.85, color="#e08a1e")

    # Row 4: submission & blend
    _box(ax, 3.2, 1.2, 2.6, 1.1, "SubmissionGenerator\n(mean + uncertainty)")
    _box(ax, 7.6, 1.2, 2.2, 1.1, "Weighted blend", fc="#e0f7fa", ec=C_FINAL)
    _arrow(ax, 1.7, 2.55, 2.8, 1.75)
    _arrow(ax, 3.9, 2.55, 3.2, 1.75)
    _arrow(ax, 6.7, 2.55, 4.0, 1.75)
    _arrow(ax, 9.0, 2.55, 4.2, 1.75)
    _arrow(ax, 4.5, 1.2, 6.5, 1.2, color=C_FINAL, lw=2.0)

    fig.text(0.5, 0.05, "Final submission  (official GLL score ~ 0.40, 63rd / 428 teams)",
             ha="center", fontsize=10, fontweight="bold", color=C_MAIN)
    fig.savefig(f"{OUT}/pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/pipeline.png")


# ----------------------------------------------------------------------------
# Diagram 2: calibration chain
# ----------------------------------------------------------------------------
def calibration_flow():
    fig, ax = _new_ax(10, 6.0, ylim=(0, 10))
    _title(ax, "Signal Calibration & Preprocessing Chain", fontsize=13)

    steps = [
        "raw uint16\nframes",
        "ADC restore\nsignal / gain + offset",
        "non-negative\nclamp",
        "linearity\ncorrection",
        "dark subtraction\n(dt / dt + incr)",
        "flat field\n(HOT-KEEP)",
        "ROI crop\n+ CDS",
        "binning\n(AIRS x30, FGS x360)",
        "outlier clip\n+ variance weights",
        "light curve\n(n_bins, 283)",
    ]
    n = len(steps)
    x0, x1 = 0.55, 9.45
    box_w = 0.8
    step = (x1 - x0) / (n - 1)
    y = 4.4
    for i, label in enumerate(steps):
        x = x0 + i * step
        alt = i in (1, 4, 5, 8)
        _box(
            ax,
            x,
            y,
            box_w,
            2.2,
            label,
            fc=BOX_FC_ALT if alt else BOX_FC,
            ec=BOX_EC_ALT if alt else BOX_EC,
            fontsize=7.5,
        )
        if i < n - 1:
            _arrow(ax, x + box_w / 2 + 0.02, y, x + step - box_w / 2 - 0.02, y)

    ax.text(5, 1.2, "Calibration frames used: dark, dead, flat, linearity polynomial",
            ha="center", fontsize=9, style="italic", color=C_GRAY)
    fig.savefig(f"{OUT}/calibration_flow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/calibration_flow.png")


# ----------------------------------------------------------------------------
# Diagram 3: ResNetMLP2 architecture
# ----------------------------------------------------------------------------
def nn_architecture():
    fig, ax = _new_ax(12, 5.2)
    _title(ax, "ResNetMLP2 - Residual MLP for the AIRS spectrum", fontsize=13)

    _box(ax, 0.8, 5.0, 1.3, 1.0, "Input\n3 features")
    _box(ax, 2.6, 5.0, 1.4, 1.0, "Linear\n3 -> 256", fc=BOX_FC_ALT, ec=BOX_EC_ALT)
    _box(ax, 4.9, 5.0, 1.6, 1.0, "ResidualBlock x35\n(256 -> 256)", fc=BOX_FC_ALT, ec=BOX_EC_ALT)
    _box(ax, 7.3, 5.0, 1.5, 1.0, "Linear\n256 -> 282", fc=BOX_FC_ALT, ec=BOX_EC_ALT)
    _box(ax, 9.2, 5.0, 1.4, 1.0, "AIRS\nspectrum", fc="#f6f1ff", ec=C_SIGNAL)

    for a, b in [(1.45, 1.9), (3.3, 4.1), (5.7, 6.55), (8.05, 8.5)]:
        _arrow(ax, a, 5.0, b, 5.0)

    # Residual block detail
    _box(ax, 4.9, 1.6, 4.6, 1.5, "ResidualBlock:  Linear -> ReLU -> Dropout -> Linear  (+) identity  -> ReLU", fontsize=8)
    ax.text(7.0, 0.6, "output = ReLU( f(x) + x )", ha="center", fontsize=8.5, style="italic", color=C_GRAY)
    _arrow(ax, 6.6, 1.6, 5.2, 4.0, color=C_SIGNAL, lw=1.2)
    _arrow(ax, 7.8, 1.6, 6.3, 4.0, color=C_SIGNAL, lw=1.2)
    ax.text(6.4, 2.6, "skip connection", fontsize=7.5, color=C_SIGNAL, ha="center")

    ax.text(1.0, 3.4, "Doubled in float64 for stability", fontsize=8, color=C_GRAY, style="italic")
    fig.savefig(f"{OUT}/nn_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/nn_architecture.png")


# ----------------------------------------------------------------------------
# Diagram 4: synthetic transit light curve
# ----------------------------------------------------------------------------
def transit_lightcurve():
    rng = np.random.default_rng(7)
    n = 320
    t = np.arange(n)
    baseline = 1.0 + 0.003 * t / n + 0.0004 * (t / n) ** 2

    # transit dip centered at t=160
    mu, sigma = 160, 26
    dip = 0.0045 * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    noise = rng.normal(0, 0.00035, n)
    y = baseline - dip + noise
    y_smooth = np.convolve(y, np.ones(7) / 7, mode="same")

    # detected phases (from the smoothed curve)
    g = np.gradient(y_smooth)
    min_idx = int(np.argmin(y_smooth[60:260])) + 60
    phase1 = int(np.argmin(g[:min_idx]))
    phase2 = int(np.argmax(g[min_idx:])) + min_idx

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.scatter(t, y, s=6, alpha=0.5, color="#9db8d2", label="binned white-light curve (noisy)")
    ax.plot(t, y_smooth, color=C_MAIN, lw=2.2, label="smoothed curve (Savitzky-Golay)")

    ax.axvline(phase1, color=C_ACC, ls="--", lw=1.8)
    ax.axvline(phase2, color=C_ACC, ls="--", lw=1.8)
    ax.axvline(min_idx, color=C_ALT, ls=":", lw=1.6)
    ax.annotate("phase 1\n(ingress)", (phase1, 1.0012), color=C_ACC, fontsize=9, ha="center")
    ax.annotate("phase 2\n(egress)", (phase2, 1.0012), color=C_ACC, fontsize=9, ha="center")
    ax.annotate("transit minimum", (min_idx, 0.9972), color=C_ALT, fontsize=9, ha="center")

    # baseline polynomial
    ax.plot(t, baseline, color=C_GRAY, lw=1.4, ls="-", alpha=0.9, label="polynomial out-of-transit baseline")

    ax.set_xlabel("time (binned index)")
    ax.set_ylabel("relative flux")
    ax.set_title("Transit depth recovery: polynomial baseline + optimized depth scaling")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_ylim(0.9945, 1.0045)
    fig.tight_layout()
    fig.savefig(f"{OUT}/transit_lightcurve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/transit_lightcurve.png")


# ----------------------------------------------------------------------------
# Diagram 5: blending
# ----------------------------------------------------------------------------
def blending():
    fig, ax = _new_ax(10, 4.6, ylim=(0, 10))
    _title(ax, "Ensemble Submission Blending", fontsize=13)

    _box(ax, 2.0, 7.0, 2.4, 1.2, "NN submission\n(ResNetMLP ensembles)", fc=BOX_FC_ALT, ec=BOX_EC_ALT)
    _box(ax, 2.0, 4.0, 2.4, 1.2, "XGBoost submission\n(mean + sigma ensemble)", fc="#fff4e6", ec="#e08a1e")
    _box(ax, 8.0, 5.5, 2.6, 1.4, "Final blend\nwl = 0.8 x NN + 0.2 x XGB\nsigma = 0.6 x NN + 0.4 x XGB", fc="#e0f7fa", ec=C_FINAL, fontsize=8.5)

    _arrow(ax, 3.2, 6.4, 6.7, 6.0, color=C_MAIN)
    _arrow(ax, 3.2, 4.0, 6.7, 5.0, color="#e08a1e")

    ax.text(5, 2.2, "then: best-of blend (0.75 / 0.25) between the two candidate blends",
            ha="center", fontsize=9, style="italic", color=C_GRAY)
    fig.savefig(f"{OUT}/blending.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/blending.png")


if __name__ == "__main__":
    pipeline()
    calibration_flow()
    nn_architecture()
    transit_lightcurve()
    blending()
    print("All assets written.")
