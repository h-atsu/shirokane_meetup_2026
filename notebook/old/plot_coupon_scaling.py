"""uv run --with matplotlib python notebook/plot_coupon_scaling.py

CSV を読み、中央値と seed 間の範囲を描く。未達の時間は到達時間ではなく打切り時の実測時間。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[Path(__file__).parent / "results" / "coupon_soft_scaling.csv"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "coupon_scaling.png",
    )
    args = parser.parse_args()
    grouped = defaultdict(list)
    coupons, budgets, targets = set(), set(), set()
    for path in args.inputs:
        with path.open() as file:
            for row in csv.DictReader(file):
                if row["method"] != "LP":
                    grouped[row["method"], int(row["users"])].append(row)
                    coupons.add(int(row["coupons"]))
                    budgets.add(float(row["time_limit_s"]))
                    targets.add(float(row["target_gap"]))
    if len(coupons) != 1 or len(budgets) != 1 or len(targets) != 1:
        parser.error(
            "a scaling plot requires one coupon count, time limit and target gap"
        )
    coupon_count = next(iter(coupons))
    requested_seconds = next(iter(budgets))
    target_pct = 100 * next(iter(targets))
    colors = {"Lagrangian-soft": "#117a65", "MIP": "#c75b12", "MIP-native": "#4666b0"}
    labels = {
        "Lagrangian-soft": "Lagrangian (soft constraints)",
        "MIP": "MIP (PuLP / HiGHS)",
        "MIP-native": "MIP (native HiGHS)",
    }
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), layout="constrained")
    for method in sorted({key[0] for key in grouped}):
        sizes = sorted(n for m, n in grouped if m == method)
        color = colors[method]
        for ax, field, scale in zip(
            axes, ["solve_s", "build_s", "gap", "peak_rss_mb"], [1, 1, 100, 1]
        ):
            medians, lows, highs = [], [], []
            for n in sizes:
                values = [
                    float(r[field]) * scale for r in grouped[method, n] if r[field]
                ]
                values = np.maximum(values, 1e-5)
                medians.append(np.median(values) if len(values) else np.nan)
                lows.append(np.min(values) if len(values) else np.nan)
                highs.append(np.max(values) if len(values) else np.nan)
            ax.plot(sizes, medians, "o-", color=color, label=labels[method])
            ax.fill_between(sizes, lows, highs, color=color, alpha=0.12)
            for n, y in zip(sizes, medians):
                if any(r["reached"] != "True" for r in grouped[method, n]):
                    ax.scatter([n], [y], marker="x", s=100, color=color, zorder=5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"Users ({coupon_count} coupon types)")
            ax.grid(alpha=0.2, which="both")
    for ax, label in zip(
        axes,
        [
            "Algorithm / HiGHS run time [s]",
            "Model build + solver transfer [s]",
            "Certified gap [%]",
            "Peak process RSS [MiB]",
        ],
    ):
        ax.set_ylabel(label)
    axes[2].axhline(target_pct, linestyle="--", color="gray", linewidth=1)
    axes[2].text(
        0.02, 0.94, f"Target: {target_pct:g}%", transform=axes[2].transAxes, va="top"
    )
    axes[2].text(
        0.03,
        0.16,
        "0% gaps shown at 1e-5%\nMissing MIP points: external stop",
        transform=axes[2].transAxes,
        fontsize=7,
        va="bottom",
    )
    axes[3].text(
        0.03,
        0.95,
        "MIP RSS unavailable after external stop",
        transform=axes[3].transAxes,
        fontsize=7,
        va="top",
    )
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Coupon allocation scaling | median and min-max across seeds\n"
        f"{requested_seconds:g} s requested solve limit (can be exceeded); "
        "x = target not reached in at least one seed",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    print(args.output)


if __name__ == "__main__":
    main()
