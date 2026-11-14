"""クーポン数・予算の厳しさ・結合制約数を変える探索。選定用 seed=0 の結果を保存。

uv run python notebook/coupon_parameter_sweep.py
探索でよかった条件は、独立の seed=1,2 でも再検証する。
"""

import csv
import subprocess
import sys
from pathlib import Path


def main():
    directory = Path(__file__).parent
    configurations = [
        # users, coupons, segments, tightness, penalty_scale
        (10000, 2, 8, 0.45, 1),
        (10000, 10, 8, 0.45, 1),
        (10000, 20, 8, 0.45, 1),
        (10000, 5, 24, 0.45, 1),
        (10000, 5, 8, 0.25, 1),
        (10000, 5, 8, 0.70, 1),
        (10000, 5, 8, 0.45, 2),
    ]
    for users, coupons, segments, tightness, penalty_scale in configurations:
        name = (
            f"soft_screen_n{users}_k{coupons}_s{segments}_t{tightness}_p{penalty_scale}"
        )
        destination = directory / "results" / f"{name}.csv"
        if destination.exists() and destination.with_suffix(".json").exists():
            with destination.open() as file:
                if len(list(csv.DictReader(file))) == 2:
                    print(f"{name}: already complete", flush=True)
                    continue
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(directory / "coupon_scaling.py"),
                "--sizes",
                str(users),
                "--coupons",
                str(coupons),
                "--segments",
                str(segments),
                "--tightness",
                str(tightness),
                "--penalty-scale",
                str(penalty_scale),
                "--seeds",
                "0",
                "--seconds",
                "20",
                "--wall-limit",
                "75",
                "--gap",
                "0.001",
                "--methods",
                "LR",
                "NATIVE",
                "--lp-max-users",
                "0",
                "--output",
                str(destination),
                "--log-file",
                str(destination.with_suffix(".highs.log")),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
