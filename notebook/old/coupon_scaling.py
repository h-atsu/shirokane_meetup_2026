"""各測定を独立プロセスで実行し、時間・ピークRSS・同じ精度での到達状況を比較。

uv run python notebook/coupon_scaling.py --sizes 1000 10000 50000 100000 \
    --seeds 0 1 2 --seconds 20 --gap 0.001

gap は比率（0.001 = 0.1%）。時間制限はソルバー/LRの求解部分。
求解時間 solve_s とモデル構築時間 build_s を分けて比較する。data_s は共通のデータ生成。
MIP と LR には同じ精度を指定し、他方の解・境界を渡さない。
各手法は1プロセスずつ順番に実行し、プロセス間のCPU競合を避ける。
"""

import argparse
import csv
import json
import platform
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import highspy
import numpy as np
import pulp
from coupon_lagrangian import (
    allocation_details,
    direct,
    evaluate,
    generate,
    lagrangian,
    save_csv,
)


def native_mip(data, seconds, target_gap, log_file=None, events_file=None):
    """PuLP の構築コストを除いた対照実験。同じモデルを疎行列で HiGHS に渡す。"""
    start = perf_counter()
    n, k = data.value.shape
    binaries = n * k
    resources = len(data.budget)
    columns = binaries + resources
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    if log_file:
        model.setOptionValue("output_flag", True)
        model.setOptionValue("log_to_console", False)
        model.setOptionValue("log_file", str(log_file))
    model.setOptionValue("threads", 1)
    model.setOptionValue("random_seed", 0)
    model.setOptionValue("mip_rel_gap", target_gap)
    model.setOptionValue("time_limit", seconds)
    empty = np.empty(0, dtype=np.int32)
    model.addCols(
        columns,
        np.r_[data.value.ravel(), -data.penalty],
        np.zeros(columns),
        np.r_[np.ones(binaries), np.full(resources, highspy.kHighsInf)],
        0,
        empty,
        empty,
        np.empty(0),
    )
    model.changeObjectiveSense(highspy.ObjSense.kMaximize)
    model.changeColsIntegrality(
        binaries,
        np.arange(binaries, dtype=np.int32),
        np.full(binaries, highspy.HighsVarType.kInteger, dtype=np.uint8),
    )
    indices = [np.arange(binaries, dtype=np.int32)]
    values = [np.ones(binaries)]
    starts = list(range(0, binaries, k))
    offset = binaries
    for m in range(len(data.budget)):
        coefficients = data.resource[:, :, m].ravel()
        selected = np.flatnonzero(coefficients)
        starts.append(offset)
        indices.append(np.r_[selected, binaries + m].astype(np.int32))
        values.append(np.r_[coefficients[selected], -1.0])
        offset += len(selected) + 1
    model.addRows(
        n + len(data.budget),
        np.full(n + len(data.budget), -highspy.kHighsInf),
        np.concatenate([np.ones(n), data.budget]),
        offset,
        np.asarray(starts, dtype=np.int32),
        np.concatenate(indices),
        np.concatenate(values),
    )
    build_s = perf_counter() - start
    solve_start = perf_counter()
    incumbent_times = []

    def publish(event):
        print(json.dumps(event), file=sys.stderr, flush=True)
        if events_file:
            with events_file.open("a") as file:
                file.write(json.dumps(event) + "\n")

    if events_file:
        events_file.parent.mkdir(parents=True, exist_ok=True)
        with events_file.open("w") as file:
            file.write("")

    def on_incumbent(event):
        elapsed = perf_counter() - solve_start
        incumbent_times.append(elapsed)
        publish(
            {
                "event": "incumbent",
                "seconds": elapsed,
                "objective": float(event.data_out.objective_function_value),
                "first_incumbent_s": incumbent_times[0],
                "incumbent_count": len(incumbent_times),
            }
        )

    model.cbMipImprovingSolution.subscribe(on_incumbent)
    # 外部打切りでも構築完了までの時間を残せるよう、親に通知する。
    publish({"event": "solve_start", "build_s": build_s, "solve_start": solve_start})
    model.run()
    solve_s = perf_counter() - solve_start
    extract_start = perf_counter()
    info = model.getInfo()
    lower = None
    choice = None
    if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
        x = np.asarray(model.getSolution().col_value[:binaries]).reshape(n, k)
        assert np.all(np.abs(x - np.rint(x)) < 1e-5)
        assert np.all(np.rint(x).sum(axis=1) <= 1)
        choice = np.where(x.max(axis=1) > 0.5, x.argmax(axis=1), -1)
        lower, _ = evaluate(data, choice)
    details = allocation_details(data, choice)
    publish(
        {
            "event": "solve_end",
            "solve_s": solve_s,
            "status": model.modelStatusToString(model.getModelStatus()),
            "has_incumbent": lower is not None,
            "incumbent_count": len(incumbent_times),
            "first_incumbent_s": incumbent_times[0] if incumbent_times else None,
        }
    )
    return {
        "method": "MIP-native",
        "lower": lower,
        "upper": float(info.mip_dual_bound),
        "status": model.modelStatusToString(model.getModelStatus()),
        "model_s": build_s,
        "transfer_s": 0.0,
        "build_s": build_s,
        "solve_s": solve_s,
        "extract_s": perf_counter() - extract_start,
        "highs_run_s": model.getRunTime(),
        "total_s": perf_counter() - start,
        "has_incumbent": lower is not None,
        "first_incumbent_s": incumbent_times[0] if incumbent_times else None,
        "incumbent_count": len(incumbent_times),
        **details,
    }


def worker(args):
    start = perf_counter()
    data = generate(
        args.users,
        args.coupons,
        args.segments,
        args.tightness,
        args.seed,
        args.penalty_scale,
    )
    data_s = perf_counter() - start
    history = []
    if args.worker == "LR":
        result, history = lagrangian(
            data, args.seconds, args.iterations, target_gap=args.gap
        )
    elif args.worker == "NATIVE":
        result = native_mip(
            data, args.seconds, args.gap, args.log_file, args.events_file
        )
    else:
        result = direct(
            data, args.seconds, relax=args.worker == "LP", target_gap=args.gap
        )
    lower, upper = result["lower"], result["upper"]
    gap = (
        None
        if lower is None or not np.isfinite(upper)
        else max(0, upper - lower) / max(1, abs(lower))
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(
        users=args.users,
        seed=args.seed,
        coupons=args.coupons,
        segments=args.segments,
        tightness=args.tightness,
        time_limit_s=args.seconds,
        target_gap=args.gap,
        gap=gap,
        reached=gap is not None and gap <= args.gap + 1e-9,
        data_s=data_s,
        process_s=perf_counter() - start,
        peak_rss_mb=rss / (1024**2 if sys.platform == "darwin" else 1024),
        iterations=len(history) if history else None,
        model="soft",
        penalty_scale=args.penalty_scale,
    )
    print(json.dumps({"result": result, "history": history}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1000, 10000, 50000, 100000]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--gap", type=float, default=0.001)
    parser.add_argument("--coupons", type=int, default=5)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--tightness", type=float, default=0.45)
    parser.add_argument("--penalty-scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--events-file",
        type=Path,
        default=None,
        help="NATIVE の incumbent 到達時刻をJSONLに記録（単一測定用）",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="既存CSVの同一条件の測定を引き継ぎ、未測定分のみ実行",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="NATIVE の HiGHS ログ保存先（単一測定時に指定）",
    )
    parser.add_argument(
        "--wall-limit",
        type=float,
        default=None,
        help="子プロセス全体の打切り秒数。通常は seconds + 180",
    )
    parser.add_argument("--lp-max-users", type=int, default=10000)
    parser.add_argument(
        "--methods", nargs="+", choices=["LR", "MIP", "NATIVE"], default=["LR", "MIP"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "coupon_soft_scaling.csv",
    )
    parser.add_argument(
        "--worker", choices=["LR", "MIP", "LP", "NATIVE"], help=argparse.SUPPRESS
    )
    parser.add_argument("--users", type=int, default=1000, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds <= 0 or args.iterations < 1 or args.gap < 0:
        parser.error("positive seconds/iterations and nonnegative gap required")
    if args.worker:
        worker(args)
        return
    records, histories = [], []
    if args.resume_from:
        with args.resume_from.open() as file:
            records = list(csv.DictReader(file))
        history_path = args.resume_from.with_name(
            args.resume_from.stem + "_history.csv"
        )
        if history_path.exists():
            with history_path.open() as file:
                histories = list(csv.DictReader(file))
        for record in records:
            if (
                record.get("model") != "soft"
                or int(record["coupons"]) != args.coupons
                or int(record["segments"]) != args.segments
                or float(record["tightness"]) != args.tightness
                or float(record["penalty_scale"]) != args.penalty_scale
                or float(record["target_gap"]) != args.gap
                or float(record["time_limit_s"]) != args.seconds
            ):
                parser.error("resume CSV contains incompatible experimental settings")
    existing = {(row["method"], int(row["users"]), int(row["seed"])) for row in records}
    for n in args.sizes:
        for seed in args.seeds:
            methods = args.methods + (["LP"] if n <= args.lp_max_users else [])
            for method in methods:
                method_name = {"LR": "Lagrangian-soft", "NATIVE": "MIP-native"}.get(
                    method, method
                )
                if (method_name, n, seed) in existing:
                    print(
                        f"users={n:,} seed={seed} {method}: already measured",
                        flush=True,
                    )
                    continue
                print(f"users={n:,} seed={seed} {method}: running", flush=True)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    method,
                    "--users",
                    str(n),
                    "--seed",
                    str(seed),
                    "--seconds",
                    str(args.seconds),
                    "--gap",
                    str(args.gap),
                    "--coupons",
                    str(args.coupons),
                    "--segments",
                    str(args.segments),
                    "--tightness",
                    str(args.tightness),
                    "--iterations",
                    str(args.iterations),
                    "--penalty-scale",
                    str(args.penalty_scale),
                ]
                if args.log_file:
                    args.log_file.parent.mkdir(parents=True, exist_ok=True)
                    command.extend(["--log-file", str(args.log_file)])
                if args.events_file:
                    command.extend(["--events-file", str(args.events_file)])
                wall_limit = args.wall_limit or args.seconds + 180
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=wall_limit,
                    )
                    payload = json.loads(completed.stdout)
                except subprocess.TimeoutExpired as error:
                    # ソルバー内の time_limit が長く超過する場合も、測定を失わない。
                    events = error.stderr or b""
                    if isinstance(events, bytes):
                        events = events.decode(errors="replace")
                    timing = {}
                    for line in events.splitlines():
                        try:
                            timing.update(json.loads(line))
                        except (ValueError, TypeError):
                            pass
                    built = timing.get("build_s")
                    solve_elapsed = (
                        perf_counter() - timing["solve_start"]
                        if "solve_start" in timing
                        else None
                    )
                    payload = {
                        "result": {
                            "method": {
                                "NATIVE": "MIP-native",
                                "LR": "Lagrangian-soft",
                            }.get(method, method),
                            "lower": None,
                            "upper": float("inf"),
                            "status": "External wall timeout",
                            "model_s": built,
                            "transfer_s": 0.0 if built is not None else None,
                            "build_s": built,
                            "solve_s": solve_elapsed,
                            "extract_s": None,
                            "highs_run_s": None,
                            "total_s": wall_limit,
                            "users": n,
                            "seed": seed,
                            "coupons": args.coupons,
                            "segments": args.segments,
                            "tightness": args.tightness,
                            "time_limit_s": args.seconds,
                            "target_gap": args.gap,
                            "gap": None,
                            "reached": False,
                            "data_s": None,
                            "process_s": wall_limit,
                            "peak_rss_mb": None,
                            "iterations": None,
                            "model": "soft",
                            "penalty_scale": args.penalty_scale,
                        },
                        "history": [],
                    }
                record = payload["result"]
                records.append(record)
                histories.extend(
                    {"users": n, "seed": seed, **row} for row in payload["history"]
                )
                save_csv(args.output, records)
                if histories:
                    save_csv(
                        args.output.with_name(args.output.stem + "_history.csv"),
                        histories,
                    )
                print(
                    f"  build={record['build_s']}s solve={record['solve_s']}s "
                    f"total={record['total_s']:.3f}s gap={record['gap']} "
                    f"RSS={record['peak_rss_mb']}MB {record['status']}",
                    flush=True,
                )
    save_csv(args.output, records)
    metadata = {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pulp": pulp.__version__,
        "highs": highspy.Highs().version(),
        "threads": 1,
        "command": sys.argv,
        "time_budget": "solve only; report build_s and solve_s separately",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
