#!/usr/bin/env python3
"""End-to-end inference reproducing the winning (63rd place) submission.

Runs the full pipeline:
    1. NN inference  -> outputs/submission_nn.csv
    2. XGBoost       -> outputs/submission_xgb.csv
    3. Blend         -> outputs/submission.csv

Before running, set the paths to your trained checkpoints/artifacts either via
command-line flags or by editing the ``build_config(...)`` call below.

Usage:
    python scripts/infer.py --data-path /path/to/ariel-data-challenge-2025
"""

from __future__ import annotations

import argparse

from ariel2025 import build_config
from ariel2025.blend import final_blend
from ariel2025.inference import run_nn_pipeline
from ariel2025.xgb_pipeline import predict as xgb_predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Full winning-pipeline inference.")
    parser.add_argument("--data-path", default="/kaggle/input/ariel-data-challenge-2025")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--fgs-checkpoint", default=None)
    parser.add_argument("--xgb-artifacts", default=None)
    args = parser.parse_args()

    cfg = build_config(dataset=args.dataset, data_path=args.data_path, output_dir=args.output_dir)
    if args.fgs_checkpoint:
        cfg.fgs_checkpoint = args.fgs_checkpoint
    if args.xgb_artifacts:
        cfg.xgb_artifacts_dir = args.xgb_artifacts

    # --- 1. Neural-network submission ---
    nn_sub = run_nn_pipeline(cfg)

    # --- 2. XGBoost submission ---
    xgb_sub = xgb_predict(cfg)

    # --- 3. Model-level blend (wl: 0.2/0.8, sigma: 0.4/0.6) ---
    from ariel2025.blend import blend_two

    blend_path = str(cfg.output_dir / "submission_blend_396.csv")
    blend_two(
        str(cfg.output_dir / "submission_xgb.csv"),
        str(cfg.output_dir / "submission_nn.csv"),
        wl_weight_a=cfg.blend.xgb_wl_weight,
        sigma_weight_a=cfg.blend.xgb_sigma_weight,
        output_path=blend_path,
    )

    # --- 4. Final best-of blend ---
    final_blend(
        blend_path,
        blend_path,  # replace with the second candidate path when available
        wl_weight_high=cfg.blend.high_wl_weight,
        sigma_weight_high=cfg.blend.high_sigma_weight,
        output_path=str(cfg.output_dir / "submission.csv"),
    )
    print("Final submission written to:", cfg.output_dir / "submission.csv")


if __name__ == "__main__":
    main()
