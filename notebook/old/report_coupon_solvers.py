"""横比較CSVと生ログから、欠測と候補解を区別したレポートを作る。"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

from coupon_lagrangian import save_csv
from coupon_solver_comparison import cbc_log_metrics


def load(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def value(row, key):
    item = row.get(key)
    return float(item) if item not in (None, "") else None


def formatted(number, digits=3):
    return (
        "—" if number is None or not math.isfinite(number) else f"{number:,.{digits}f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "results" / "solver_comparison_n100000.csv",
    )
    args = parser.parse_args()
    rows = load(args.input)
    directory = args.input.parent
    cbc_retry_used = False
    for row in rows:
        method = row["method"]
        prefix = directory / f"{args.input.stem}_{method.lower()}"
        retry_path = directory / "solver_comparison_n100000_cbc_wait.csv"
        if method == "CBC" and retry_path.exists():
            retry = next((r for r in load(retry_path) if r["method"] == "CBC"), None)
            if (
                retry
                and retry.get("validated") == "True"
                and all(
                    str(retry[k]) == str(row[k])
                    for k in ("users", "coupons", "segments", "seed")
                )
                and float(retry["penalty_scale"]) == float(row["penalty_scale"])
            ):
                row["initial_run_status"] = row["status"]
                row["initial_run_solve_s"] = row["solve_s"]
                row.update(retry)
                row["result_source"] = retry_path.name
                prefix = directory / f"{retry_path.stem}_cbc"
                cbc_retry_used = True
        completed = Path(str(prefix) + ".json")
        if (
            row["status"] == "External timeout"
            and completed.exists()
            and completed.stat().st_size
        ):
            saved = json.loads(completed.read_text())
            if saved.get("validated"):
                row["raw_controller_status"] = row["status"]
                row.update(saved)
                row["result_source"] = (
                    "validated worker result saved before cleanup interruption"
                )
        if method == "CBC":
            metrics = cbc_log_metrics(Path(str(prefix) + ".solver.log"))
            row.update(
                candidate_s=metrics["candidate_s"],
                candidate_profit=metrics["best_candidate_profit"],
                candidate_source="CBC feasibility pump / mini branch-and-bound log",
            )
            if value(row, "lower") is not None and metrics["absolute_gap"] is not None:
                row["raw_parsed_upper"] = row.get("upper")
                row["upper"] = math.nextafter(
                    value(row, "lower")
                    + metrics["absolute_gap"]
                    + metrics["absolute_gap_rounding"],
                    math.inf,
                )
                row["upper_source"] = (
                    "validated incumbent plus printed absolute gap, rounded outward"
                )
                row["solver_reported_gap"] = metrics["solver_reported_gap"]
        if method == "SCIP":
            # BESTSOLFOUND can be raised before getPrimalbound() is refreshed.
            # Preserve that raw callback value and recover the exact trivial bound from the solver log.
            log = Path(str(prefix) + ".worker.log").read_text(errors="replace")
            trivial = re.search(
                r"feasible solution found by trivial heuristic after ([\d.]+) seconds, objective value ([-+\deE.]+)",
                log,
            )
            if trivial and (
                value(row, "lower") is None or value(row, "lower") <= -1e19
            ):
                row["raw_callback_lower"] = row.get("lower")
                row["lower"] = float(trivial[2])
                row["lower_source"] = (
                    "SCIP trivial-heuristic log; BESTSOLFOUND callback clock is retained"
                )
                row["has_incumbent"] = True
            if value(row, "upper") is not None and value(row, "upper") >= 1e19:
                row["upper"] = float("inf")
    previous = directory / "five_minute_n100000_k5_p2.csv"
    if previous.exists():
        baseline = next(r for r in load(previous) if r["method"] == "MIP-native")
        if all(
            str(baseline[k]) == str(rows[0][k])
            for k in ("users", "coupons", "segments", "seed")
        ) and float(baseline["penalty_scale"]) == float(rows[0]["penalty_scale"]):
            baseline.update(method="HiGHS (previous run)", source=str(previous.name))
            rows.insert(0, baseline)
    for row in rows:
        lower, upper = value(row, "lower"), value(row, "upper")
        row["gap"] = (
            max(0, upper - lower) / max(1, abs(lower))
            if lower is not None and upper is not None and math.isfinite(upper)
            else None
        )
    reviewed = args.input.with_name(args.input.stem + "_reviewed.csv")
    save_csv(reviewed, rows)
    lines = [
        "# CBC・SCIP・descartes-mip の横比較",
        "",
        "10万人×5種類、共有リソース14個、予算比率0.45、ペナルティ倍率2、seed=0の同じソフト制約付き問題。",
        "Apple M2 / 8 GiB、各ソルバー1スレッド、初期解なし、目標ギャップ0.1%。",
        "HiGHS は前回の同条件の正常終了ログを再利用した。CBCは、解の出力まで待った再測定がある場合はその結果を採用する。",
        "",
        "| 手法 | モデル作成・登録(s) | 求解API(s) | 最初の解・候補 | 最終値／ログ候補 | gap(%) | 終了状態 |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        method = row["method"]
        lower = value(row, "lower")
        first = formatted(value(row, "first_incumbent_s"))
        objective = formatted(lower, 2)
        if method == "CBC":
            first = formatted(value(row, "candidate_s")) + "（候補）"
            if row.get("validated") == "True":
                objective = formatted(lower, 2) + "（割当検証済み）"
            else:
                objective = (
                    formatted(value(row, "candidate_profit"), 2) + "（ログのみ）"
                )
        elif method == "DESCARTES" and row.get("phase") == "decomposition":
            first = "なし（自動分解中、0反復）"
        elif method == "NUMPY":
            first = "求解時間内に目標精度へ到達"
        elif method.startswith("HiGHS"):
            first = "なし（最終状態で確認）"
        gap_text = formatted(
            100 * value(row, "gap") if value(row, "gap") is not None else None, 6
        )
        if method == "SCIP" and lower == 0 and value(row, "upper") > 0:
            gap_text = "∞（目的値0）"
        lines.append(
            f"| {method} | {formatted(value(row, 'build_s'))} | {formatted(value(row, 'solve_s'))} "
            f"| {first} | {objective} | {gap_text} "
            f"| {row['status']} |"
        )
    lines += [
        "",
        "## 結論と読み方",
        "",
        (
            "- **『すべての汎用MIPが5分間実行可能解を得られない』とは言えない。**"
            "SCIPは早期に目的値0の実行可能解を得ており、CBCも約240秒で実行可能候補をログに記録した。"
        ),
        (
            "- CBCは初回は303秒で外部打切り、最終割当未回収だった。"
            + (
                "追加再測定では外部期限を無効にして出力まで待ち、最終割当と目的値を独立検証した。"
                if cbc_retry_used
                else "ログ候補と検証済み返却解を区別し、候補を検証済み境界として扱わない。"
            )
        ),
        (
            "- SCIPの早い実行可能解は目的値0であり、良い配布が得られたことを意味しない。"
            "300秒で求解を終了し、最終値0・双対境界222,719,315を保存済み。空配布の割当を独立検証した。"
        ),
        (
            "- 専用NumPy分解は元のペナルティ付き目的値を再計算して検証しており、"
            "返却する上界とのギャップも併記した。"
        ),
        (
            "- descartes-mipは専用NumPy実装とは異なる。自動分解中の打切りは、"
            "ラグランジュ双対の収束が遅いという結果ではなく、反復に入る前の前処理の問題。"
        ),
        "",
        "## descartes-mip v0.2.1",
        "",
        (
            "全体原資・枚数・属性予算を `Model.relax()` でL1ペナルティ付き制約に指定した。"
            "明示的スラックは不要。制約を `a/B <= 1`、ペナルティを `rho*B` とし、元の問題と同値にしている。"
        ),
        (
            "`parallel=False`、HiGHS各1スレッド、既定Polyak係数1.0。ライブラリのアルゴリズムは変更せず、"
            "`analyze()` に時間測定だけのラッパーを付けて前処理段階を記録した。"
        ),
        (
            "[API](https://github.com/h-atsu/descartes-mip/blob/main/docs/api.md)では永続HiGHSモデルによる部分問題求解を提供する。"
            "[analyzer.py](https://github.com/h-atsu/descartes-mip/blob/main/src/descartes_mip/decomposition/analyzer.py)の"
            "`_public_subproblem()` は各部分問題について全体の目的項・緩和制約の項を再走査する。"
            "このモデルではユーザー数だけ部分問題ができるため、少なくとも目的項抽出だけでも `O(N^2 K)` の走査になる。"
            "これはコードからの計算量の推論であり、個々の関数のCPUプロファイルではない。"
        ),
        "",
        "## 計測上の区別",
        "",
        (
            "- CBCは `-F` の最小化MPSとして出力し、値と境界の符号を元の最大化目的に戻した。"
            "小規模ではCBC・SCIP・descartes-mipの目的値を独立に再計算し、同値性を確認した。"
        ),
        (
            "- CBCの構築時間はPuLP作成とMPS出力、求解API時間はMPS読込みを含む子プロセス全体。"
            "SCIPはPySCIPOptでの直接モデル作成後の `optimize()`。"
        ),
        (
            "- descartes-mipのモデル作成時間は公開Model APIの作成まで。求解API時間は `Model.solve()` 全体で、"
            "スナップショット・自動分解・部分問題コンパイル・反復を含む。専用NumPyの反復時間と区別する。"
        ),
        (
            "- 初回横比較のCBC・SCIP・descartes-mipには求解開始から303秒の外部期限を適用。CBC再測定では外部期限を無効にした。"
            "SCIPは300.51秒で求解を終えて検証済みJSONを保存しており、その後のワーカー終了処理中に外部期限となった。"
            "生のコントローラCSVは外部打切りだが、解釈用CSVでは保存済みの検証結果を採用した。"
            "モデル作成時間は別計測。結果未回収の外部打切りを『実行可能解なし』とは解釈しない。"
        ),
        (
            "- CBC最終ログのLower boundは有効桁が少ないため、その丸め値を厳密な上界として扱わない。"
            "解釈用CSVでは検証済み目的値と `integer gap` の表示から上界を外向きに丸めて復元した。"
            "元のパース値は `raw_parsed_upper` に保存している。"
        ),
        (
            "- SCIPの最初のBESTSOLFOUNDイベントでは `getPrimalbound()` がまだ無限大の番兵値を返したため、"
            "最終目的値・境界には保存済みの検証済みJSONを使用した。生イベントとコントローラCSVも保存した。"
            "計測コードは現在、イベントで `getBestSol()` の目的値を読むよう修正済み。"
        ),
        (
            "- ソフト制約版は空配布が常に実行可能。今回の比較は汎用ソルバーのコールドスタートと解の品質の比較であり、"
            "実行可能性そのものが難しいという主張ではない。"
        ),
        "",
        "## バージョンと再現",
        "",
        (
            "CBC: cbcbox 2.929 / エンジン表示 Devel (2026-03-23)、SCIP 10.0.2 / PySCIPOpt 6.3.0、"
            "descartes-mip 0.2.1、HiGHS 1.15.1、PuLP 4.2.0。"
        ),
        "",
        "```bash",
        "uv run python notebook/coupon_solver_comparison.py --users 100000 --seconds 300",
        "uv run python notebook/report_coupon_solvers.py",
        "```",
        "",
        (
            f"[生CSV]({args.input.name}) / [解釈用CSV]({reviewed.name})。"
            "同じ接頭辞の `.solver.log`・`.worker.log`・`.events.jsonl` が測定根拠。"
        ),
    ]
    if cbc_retry_used:
        cbc_row = next(r for r in rows if r["method"] == "CBC")
        lines += [
            "",
            "## CBCの出力待ち再測定",
            "",
            (
                "CBC内部の求解時間設定は300秒のまま、外部の303秒打切りを無効にした。"
                "保存した最終割当から整数性・ユーザーごとの選択制約・元のペナルティ付き目的値を検証した。"
            ),
            "",
            (
                f"[再測定CSV]({cbc_row['result_source']}) / "
                f"[CBC最終解]({Path(cbc_row['solution_file']).name}) / "
                f"[圧縮割当]({Path(cbc_row['allocation_file']).name})。"
            ),
            "",
            "```bash",
            "uv run python notebook/coupon_solver_comparison.py --users 100000 --methods CBC --seconds 300 --wait-for-result --output notebook/results/solver_comparison_n100000_cbc_wait.csv",
            "```",
        ]
    output = args.input.with_suffix(".md")
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
