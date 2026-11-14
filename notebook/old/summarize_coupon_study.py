"""保存した実測CSVから、日本語の比較表を生成する。"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def load(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def number(rows, field):
    values = [float(r[field]) for r in rows if r.get(field)]
    return median(values) if values else None


def fmt(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def main():
    directory = Path(__file__).parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scaling", type=Path, default=directory / "coupon_soft_scaling.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=directory / "coupon_soft_report.md"
    )
    parser.add_argument("--hardware", default="See experimental metadata")
    args = parser.parse_args()
    rows = load(args.scaling)
    groups = defaultdict(list)
    for row in rows:
        if row.get("model") != "soft":
            raise ValueError("hard and soft models must not be mixed")
        groups[int(row["users"]), row["method"]].append(row)
    settings = rows[0]
    seeds = sorted({int(r["seed"]) for r in rows})
    target = 100 * float(settings["target_gap"])
    text = [
        "# ソフト制約付きクーポン配布：実測結果",
        "",
        f"測定環境: {args.hardware}。各測定は独立プロセス、HiGHS は1スレッド。",
        (
            f"クーポン{settings['coupons']}種類、重複属性{settings['segments']}個、"
            f"予算比率{settings['tightness']}、ペナルティ倍率{settings['penalty_scale']}、seed={seeds}。"
        ),
        f"両手法の目標ギャップは{target:g}%。分解法は修復を一切行わず、超過ペナルティを含む目的値を評価する。",
        "",
        "## 人数を増やした場合",
        "",
        "時間は seed 間の中央値。MIP は疎行列を直接 HiGHS に渡しており、PuLP の構築コストを含まない。",
        "構築時間と求解時間は別計測。`≥` は外部打切りであり、目標精度への到達時間ではない。",
        "",
        "| 人数 | バイナリ変数 | LR準備(s) | LR求解(s) | MIP構築(s) | HiGHS求解(s) | LR達成数 | MIP達成数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted({n for n, _ in groups}):
        lr = groups[n, "Lagrangian-soft"]
        mip = groups[n, "MIP-native"]
        censored = any(r["status"] == "External wall timeout" for r in mip)
        text.append(
            f"| {n:,} | {n * int(settings['coupons']):,} | {fmt(number(lr, 'build_s'), 4)} "
            f"| {fmt(number(lr, 'solve_s'))} | {fmt(number(mip, 'build_s'), 4)} "
            f"| {'≥' if censored else ''}{fmt(number(mip, 'solve_s'))} "
            f"| {sum(r['reached'] == 'True' for r in lr)}/{len(lr)} "
            f"| {sum(r['reached'] == 'True' for r in mip)}/{len(mip)} |"
        )
    text += [
        "",
        "LR の各 seed の目的値・上界・ギャップ:",
        "",
        "| 人数 | seed | ペナルティ控除後の目的値 | 双対上界 | ギャップ(%) | 反復数 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        (r for r in rows if r["method"] == "Lagrangian-soft"),
        key=lambda r: (int(r["users"]), int(r["seed"])),
    ):
        text.append(
            f"| {int(row['users']):,} | {row['seed']} | {fmt(float(row['lower']), 2)} "
            f"| {fmt(float(row['upper']), 2)} | {100 * float(row['gap']):.4f} | {row['iterations']} |"
        )
    text += [
        "",
        "![スケーリング実測](coupon_scaling.png)",
        "",
        "## 条件探索",
        "",
        "以下は seed=0 の探索結果。人数系列と混ぜて中央値を取らず、変更した条件をすべて掲載する。",
        "券種数・属性数・予算比率・ペナルティ倍率以外は同じ生成方式を使用する。",
        "",
        "| 人数 | 券種 | 属性 | 予算比率 | ペナルティ倍率 | LR求解(s) | HiGHS求解(s) | LR gap(%) | MIP gap(%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path in sorted(directory.glob("soft_screen_*.csv")):
        if "history" in path.name:
            continue
        case = load(path)
        lr = next((r for r in case if r["method"] == "Lagrangian-soft"), None)
        mip = next((r for r in case if r["method"] == "MIP-native"), None)
        if lr is None or mip is None:
            continue
        censored = mip["status"] == "External wall timeout"
        text.append(
            f"| {int(lr['users']):,} | {lr['coupons']} | {lr['segments']} "
            f"| {lr['tightness']} | {lr['penalty_scale']} | {fmt(float(lr['solve_s']))} "
            f"| {'≥' if censored else ''}{fmt(float(mip['solve_s'])) if mip['solve_s'] else '—'} "
            f"| {100 * float(lr['gap']):.4f} "
            f"| {fmt(100 * float(mip['gap']), 4) if mip['gap'] else '未達'} |"
        )
    text += [
        "",
        "## 解釈",
        "",
        "- 比較対象は同一のソフト制約付きモデル。ハード制約版の過去の数値は使用していない。",
        (
            "- 超過した分はスラックで受け、目的関数からペナルティを差し引く。"
            "予算を守る解を修復で作る実験ではない。"
        ),
        (
            "- 予算超過率は `max_overrun_pct`、ペナルティ額は `penalty_cost` に保存。"
            "ソフト化によって許容した超過を隠さず、目的値と合わせて確認する。"
        ),
        "- 目標ギャップは最適性の保証幅であり、MIPとLRの目的値が完全に等しいことを要求していない。",
        (
            "- 求解上限は20秒を指定。停止が遅れることがあるため、別途プロセス全体に外部上限を設けた。"
            "外部打切りでは最終 incumbent や最終境界を回収できず、CSV は欠測としている。"
        ),
        (
            "- 外部打切りの値を、MIPが最適に到達した時間として速度倍率を計算してはいけない。"
            "到達までに必要な時間の下限としてのみ解釈する。"
        ),
        (
            "- 人数に対して共有制約数が固定なので、LRの双対変数数は増えない。"
            "今回の反復数もCSVに保存している。一般に反復数が一定とは保証しない。"
        ),
        "- 合成データと HiGHS 1スレッドの今回の環境での実測。別のソルバーや係数では優劣が変わりうる。",
        "",
        "## 再現コマンド",
        "",
        "```bash",
    ]
    metadata_path = args.scaling.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        text.append("uv run python " + " ".join(metadata["command"]))
        text += [
            "```",
            "",
            (
                f"環境: `{metadata['platform']}` / Python `{metadata['python'].split()[0]}` / "
                f"NumPy `{metadata['numpy']}` / PuLP `{metadata['pulp']}` / HiGHS `{metadata['highs']}`。"
            ),
        ]
    else:
        text += ["# 実験実行中。完了後にメタデータJSONを参照。", "```"]
    args.output.write_text("\n".join(text) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
