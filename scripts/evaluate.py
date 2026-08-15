#!/usr/bin/env python3
"""Evaluate a submission CSV against ground-truth labels with the official GLL metric.

Usage:
    python scripts/evaluate.py --solution /path/to/train.csv \
        --submission /path/to/submission.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from ariel2025.metrics import score


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the official GLL score of a submission.")
    parser.add_argument("--solution", required=True, help="Ground-truth CSV (train.csv).")
    parser.add_argument("--submission", required=True, help="Submission CSV.")
    parser.add_argument("--row-id", default="planet_id")
    args = parser.parse_args()

    solution = pd.read_csv(args.solution)
    submission = pd.read_csv(args.submission)

    # Align rows by planet_id (submissions can be out of order).
    solution = solution.set_index(args.row_id).sort_index()
    submission = submission.set_index(args.row_id).sort_index().reindex(solution.index)

    n_wl = solution.shape[1]
    wl_cols = [c for c in submission.columns if c.startswith("wl_")]
    sigma_cols = [c for c in submission.columns if c.startswith("sigma_")]
    submission = submission[wl_cols + sigma_cols]
    submission.columns = list(solution.columns) + [f"sigma_{c}" for c in solution.columns]

    naive_mean = float(solution.values.mean())
    naive_sigma = float(solution.values.std())

    result = score(
        solution.reset_index(),
        submission.reset_index(),
        args.row_id,
        naive_mean,
        naive_sigma,
    )
    print(f"\nOfficial GLL score: {result:.5f}")


if __name__ == "__main__":
    main()
