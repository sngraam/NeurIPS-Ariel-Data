#!/usr/bin/env python3
"""Train the XGBoost mean/sigma ensemble and calibrate the uncertainties.

Usage:
    python scripts/train_xgb.py --data-path /path/to/ariel-data-challenge-2025 \
        --output-dir ./outputs --debug
"""

from __future__ import annotations

import argparse

from ariel2025 import build_config
from ariel2025.xgb_pipeline import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost ensemble.")
    parser.add_argument("--data-path", default="/kaggle/input/ariel-data-challenge-2025")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--debug", action="store_true", help="Use a small subset for a fast run.")
    args = parser.parse_args()

    cfg = build_config(dataset="train", data_path=args.data_path, output_dir=args.output_dir)
    train(cfg, debug=args.debug)


if __name__ == "__main__":
    main()
