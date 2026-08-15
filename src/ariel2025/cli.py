"""Command-line entry point for the pipeline (``ariel-infer``)."""

from __future__ import annotations

import argparse

from .config import build_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ariel-infer",
        description="Run the Ariel Data Challenge 2025 inference pipeline.",
    )
    parser.add_argument("--data-path", default="/kaggle/input/ariel-data-challenge-2025", help="Dataset root.")
    parser.add_argument("--dataset", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", default="./outputs", help="Where to write submissions.")
    parser.add_argument("--fgs-checkpoint", default=None, help="Path to the trained FGS1 MLP checkpoint.")
    args = parser.parse_args()

    cfg = build_config(dataset=args.dataset, data_path=args.data_path, output_dir=args.output_dir)
    if args.fgs_checkpoint:
        cfg.fgs_checkpoint = args.fgs_checkpoint

    from .inference import run_nn_pipeline

    submission = run_nn_pipeline(cfg)
    print(f"NN submission written: {len(submission)} planets")


if __name__ == "__main__":
    main()
