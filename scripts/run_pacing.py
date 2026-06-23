#!/usr/bin/env python
"""Compare budget pacing controllers on a simulated daily traffic curve.

This script runs three pacers against the same synthetic diurnal traffic curve
and the same daily budget. The PID pacer tracks an ideal spend trajectory with a
feedback loop. The proportional pacer aims for a flat per slot spend with no
feedback. The ASAP pacer spends as fast as possible and exhausts its budget early
which is the failure mode smooth pacing is meant to avoid.

The script plots the cumulative spend of each pacer against the ideal pacing line
and saves the figure to results/pacing_curve.png. It writes a summary report to
results/pacing_report.md and prints a table of headline metrics.

Why smooth pacing matters. Spending a budget evenly across the day avoids early
exhaustion, keeps cost per mille stable because the campaign is not forced to win
every impression in a short window, and gives broad time of day coverage so the
campaign reaches users throughout the whole day rather than only in the morning.

Run from the repository root.
    python scripts/run_pacing.py
"""

from __future__ import annotations

import argparse
import os
import sys

# Insert the repository root onto sys.path so that "import src" works when this
# script is run directly from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

# Select a non interactive backend before importing pyplot so the plot works on a
# headless machine.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 import after backend selection
import numpy as np  # noqa: E402

from src.pacing.controller import AsapPacer, PIDPacer, ThrottlePacer  # noqa: E402
from src.pacing.simulator import run_pacing, simulate_traffic  # noqa: E402

SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse the command line flags for the pacing run."""
    parser = argparse.ArgumentParser(
        description="Compare budget pacing controllers on simulated traffic."
    )
    parser.add_argument(
        "--n-slots",
        type=int,
        default=24 * 60,
        help="Number of time slots in the simulated day.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Daily budget in spend units. Defaults to half of total traffic.",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Directory where the pacing plot and report are written.",
    )
    return parser.parse_args()


def build_pacers(budget: float, n_slots: int) -> dict:
    """Build the pacers to compare keyed by display name.

    The PID gains are modest so the loop is stable. The ASAP and proportional
    pacers take no gains. The simulator resets each pacer with the real budget,
    slot count, and traffic curve before stepping.
    """
    return {
        "PID": PIDPacer(kp=0.5, ki=0.05, kd=0.1, total_budget=budget, n_slots=n_slots),
        "Proportional": ThrottlePacer(total_budget=budget, n_slots=n_slots),
        "ASAP": AsapPacer(total_budget=budget, n_slots=n_slots),
    }


def plot_pacing(runs: dict, ideal_cumulative: np.ndarray, out_path: str) -> str:
    """Plot the cumulative spend of every pacer against the ideal line.

    The runs argument maps a pacer name to its run_pacing result dict. The ideal
    cumulative curve is drawn as a dashed reference. The figure is saved to
    out_path which is returned.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    n_slots = len(ideal_cumulative)
    hours = np.arange(n_slots) / float(n_slots) * 24.0

    ax.plot(hours, ideal_cumulative, linestyle="--", color="gray", label="Ideal")
    for name, result in runs.items():
        ax.plot(hours, result["cumulative"], label=name)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Cumulative spend")
    ax.set_title("Cumulative budget spend over the day")
    ax.set_xlim(0.0, 24.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_report(runs: dict, out_dir: str, plot_path: str, budget: float) -> str:
    """Write the pacing markdown report and return its path.

    The prose follows the project style rules. No em dashes. No semicolons. No
    mid sentence colons.
    """
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "pacing_report.md")

    lines = []
    lines.append("# Budget Pacing Report")
    lines.append("")
    lines.append(
        "This report compares three budget pacing controllers on one simulated "
        "diurnal traffic curve and one daily budget. The goal of a pacer is to "
        "spend the budget smoothly across the day so the campaign avoids early "
        "exhaustion, keeps cost per mille stable, and reaches users throughout "
        "the whole day."
    )
    lines.append("")
    lines.append(f"Daily budget. {budget:.1f} spend units")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "Budget utilization is the share of the budget that was spent. Smoothness "
        "error is the normalized root mean squared error of the realized "
        "cumulative spend against the ideal traffic proportional spend, so lower "
        "is smoother. Exhaustion hour is when the campaign first spent almost all "
        "of its budget, so a later hour means the spend lasted across the day."
    )
    lines.append("")
    lines.append("| Pacer | Budget utilization | Smoothness error | Exhaustion hour |")
    lines.append("| --- | --- | --- | --- |")
    for name, result in runs.items():
        n_slots = len(result["cumulative"])
        exhaustion_hour = result["exhaustion_slot"] / float(n_slots) * 24.0
        lines.append(
            f"| {name} "
            f"| {result['budget_utilization']:.4f} "
            f"| {result['smoothness_norm_rmse']:.4f} "
            f"| {exhaustion_hour:.2f} |"
        )
    lines.append("")
    lines.append("## Cumulative Spend")
    lines.append("")
    rel = os.path.relpath(plot_path, out_dir)
    lines.append(
        "The figure overlays the cumulative spend of each pacer against the ideal "
        "pacing line. A curve that hugs the ideal line spent in step with demand."
    )
    lines.append("")
    lines.append(f"![Cumulative spend]({rel})")
    lines.append("")
    lines.append("## Why Smooth Pacing Matters")
    lines.append("")
    lines.append(
        "The ASAP pacer wins every eligible impression until the budget is gone, "
        "so it exhausts the budget early and goes dark for the rest of the day. "
        "That concentrates spend in a narrow window, drives up cost per mille, and "
        "leaves most of the day uncovered. The proportional pacer targets a flat "
        "per slot spend and does better, but with no feedback it drifts when "
        "traffic is uneven. The PID pacer tracks the ideal trajectory with a "
        "feedback loop, so it corrects drift and lands closest to the smooth "
        "ideal spend curve."
    )
    lines.append("")

    content = "\n".join(lines).rstrip() + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    return report_path


def print_summary(runs: dict) -> None:
    """Print a simple aligned summary table of the pacing metrics."""
    print()
    print(f"{'Pacer':<14}{'Utilization':<14}{'Smoothness':<14}{'Exhaust hr':<12}")
    print("-" * 54)
    for name, result in runs.items():
        n_slots = len(result["cumulative"])
        exhaustion_hour = result["exhaustion_slot"] / float(n_slots) * 24.0
        print(
            f"{name:<14}"
            f"{result['budget_utilization']:<14.4f}"
            f"{result['smoothness_norm_rmse']:<14.4f}"
            f"{exhaustion_hour:<12.2f}"
        )
    print()


def main() -> None:
    """Run the pacing comparison end to end."""
    args = parse_args()

    traffic = simulate_traffic(n_slots=args.n_slots, seed=SEED)
    total_traffic = float(np.sum(traffic))

    # A budget that is roughly half of total eligible traffic makes pacing matter.
    # With this budget the ASAP pacer clearly exhausts early while a smooth pacer
    # can stretch spend across the day.
    budget = args.budget if args.budget is not None else 0.5 * total_traffic
    print(
        f"simulated {len(traffic)} slots, total traffic {total_traffic:.0f}, "
        f"budget {budget:.0f}."
    )

    pacers = build_pacers(budget, args.n_slots)
    runs = {}
    ideal_cumulative = None
    for name, pacer in pacers.items():
        result = run_pacing(pacer, traffic, budget)
        runs[name] = result
        if ideal_cumulative is None:
            ideal_cumulative = result["ideal_cumulative"]

    os.makedirs(args.output, exist_ok=True)
    plot_path = os.path.join(args.output, "pacing_curve.png")
    plot_pacing(runs, ideal_cumulative, plot_path)
    print(f"wrote {plot_path}.")

    report_path = write_report(runs, args.output, plot_path, budget)
    print(f"wrote {report_path}.")

    print_summary(runs)
    print("pacing run complete.")


if __name__ == "__main__":
    main()
