"""5分比較用の設定をまとめた再現エントリーポイント。

uv run python notebook/coupon_five_minute_demo.py
uv run python notebook/coupon_five_minute_demo.py --users 100000 --rerun
uv run python notebook/coupon_five_minute_demo.py --users 10000 100000 --coupons 10

同条件で完了したCSVがある場合は再利用する。再測定は --rerun を付ける。
ソフト制約版なので任意の配布から実行可能解は作れる。調べるのは、汎用MIPが
incumbentを返すまでの時間と、分解法が目的値ギャップ0.1%に達するまでの時間。
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--coupons", type=int, default=5)
    parser.add_argument("--penalty-scale", type=float, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).parent
    for users in args.users:
        prefix = (
            directory
            / "results"
            / f"five_minute_n{users}_k{args.coupons}_p{args.penalty_scale:g}"
        )
        if args.seed:
            prefix = Path(str(prefix) + f"_seed{args.seed}")
        output = Path(str(prefix) + ".csv")
        if output.exists() and not args.rerun:
            with output.open() as file:
                records = list(csv.DictReader(file))
            if (
                len(records) == 2
                and {r["method"] for r in records} == {"Lagrangian-soft", "MIP-native"}
                and all(
                    r["model"] == "soft"
                    and int(r["seed"]) == args.seed
                    and float(r["time_limit_s"]) == args.seconds
                    and float(r["target_gap"]) == 0.001
                    and r["status"] != "External wall timeout"
                    for r in records
                )
            ):
                print(f"Reuse completed measurement: {output}", flush=True)
                continue
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(directory / "coupon_scaling.py"),
                "--sizes",
                str(users),
                "--coupons",
                str(args.coupons),
                "--segments",
                "8",
                "--tightness",
                "0.45",
                "--penalty-scale",
                str(args.penalty_scale),
                "--seeds",
                str(args.seed),
                "--seconds",
                str(args.seconds),
                "--wall-limit",
                str(args.seconds + 5),
                "--iterations",
                "20000",
                "--gap",
                "0.001",
                "--methods",
                "LR",
                "NATIVE",
                "--lp-max-users",
                "0",
                "--output",
                str(output),
                "--log-file",
                str(prefix) + ".highs.log",
                "--events-file",
                str(prefix) + ".events.jsonl",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
