#!/usr/bin/env python
"""Regenerate the benchmark markdown report from a saved metrics file.

The benchmark run writes a metrics json that holds the per model results, the
optional logistic feature importance, and the calibration figure path. This
standalone script reads that file and rebuilds the markdown report without having
to retrain anything. It is handy when the report template changes or when you
want to rebuild the report on a different machine.

Run from the repository root.
    python scripts/generate_report.py --input results/metrics.json --output results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.evaluation.report import generate_report


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the report regeneration."""
    parser = argparse.ArgumentParser(
        description="Regenerate the benchmark report from a saved metrics json."
    )
    parser.add_argument(
        "--input",
        default="results/metrics.json",
        help="Path to the metrics json written by run_benchmark.",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Directory where the markdown report is written.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the saved metrics and regenerate the report."""
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"metrics file {args.input} not found. Run scripts/run_benchmark.py first."
        )

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    feature_importance = payload.get("feature_importance")
    calibration_png = payload.get("calibration_png")

    # The feature importance is stored as a list of two element lists in json.
    # Convert each back into a tuple for the report builder.
    if feature_importance:
        feature_importance = [(str(name), float(weight)) for name, weight in feature_importance]

    report_path = generate_report(
        results,
        args.output,
        calibration_png=calibration_png,
        feature_importance=feature_importance,
    )
    print(f"wrote {report_path}.")


if __name__ == "__main__":
    main()
