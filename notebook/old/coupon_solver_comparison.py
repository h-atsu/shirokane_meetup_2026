"""CBC / SCIP / descartes-mip / NumPy LR を同一インスタンスで比較。

uv run python notebook/coupon_solver_comparison.py --users 100000 --seconds 300
各測定は別プロセスで逐次実行。求解開始から seconds+3 秒でプロセス群を停止する。
descartes の solve 時間は自動分解・部分問題コンパイルを含む。ライブラリは変更しない。
"""

import argparse
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from coupon_lagrangian import (
    allocation_details,
    direct,
    evaluate,
    generate,
    lagrangian,
    save_csv,
)


def emit(args, **event):
    event["at"] = time.perf_counter()
    with args.events.open("a") as file:
        file.write(json.dumps(event) + "\n")


def finish(args, result):
    result.update(
        method=args.worker,
        users=args.users,
        coupons=args.coupons,
        segments=args.segments,
        seed=args.seed,
        penalty_scale=args.penalty_scale,
        profile=args.profile,
        target_gap=args.gap,
        time_limit_s=args.seconds,
    )
    lb, ub = result.get("lower"), result.get("upper")
    result["gap"] = (
        max(0, ub - lb) / max(1, abs(lb))
        if lb is not None and ub is not None and math.isfinite(ub)
        else None
    )
    args.result.write_text(json.dumps(result, indent=2) + "\n")


def scip(data, args):
    from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model, quicksum

    start = time.perf_counter()
    model = Model("coupon_soft")
    model.setLogfile(str(args.engine_log))
    model.setRealParam("limits/time", args.seconds)
    model.setRealParam("limits/gap", args.gap)
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", 0)
    n, k = data.value.shape
    x = [
        [
            model.addVar(f"x_{i}_{j}", vtype="B", obj=float(data.value[i, j]))
            for j in range(k)
        ]
        for i in range(n)
    ]
    slack = [
        model.addVar(f"slack_{m}", lb=0, obj=-float(p))
        for m, p in enumerate(data.penalty)
    ]
    model.setMaximize()
    for row in x:
        model.addCons(quicksum(row) <= 1)
    for m, budget in enumerate(data.budget):
        rows, cols = np.nonzero(data.resource[:, :, m])
        model.addCons(
            quicksum(
                float(data.resource[i, j, m]) * x[i][j]
                for i, j in zip(rows.tolist(), cols.tolist())
            )
            - slack[m]
            <= float(budget)
        )
    first = []
    solve_start = time.perf_counter()

    class Incumbent(Eventhdlr):
        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexec(self, event):
            elapsed = time.perf_counter() - solve_start
            if not first:
                first.append(elapsed)
            solution = self.model.getBestSol()
            objective = (
                float(self.model.getSolObjVal(solution))
                if solution is not None
                else None
            )
            dual = float(self.model.getDualbound())
            emit(
                args,
                kind="incumbent",
                seconds=elapsed,
                first_incumbent_s=first[0],
                lower=objective,
                upper=float("inf") if self.model.isInfinity(abs(dual)) else dual,
            )

    handler = Incumbent()
    model.includeEventhdlr(handler, "incumbent", "record first feasible solution")
    build_s = time.perf_counter() - start
    solve_start = time.perf_counter()
    emit(args, kind="phase", phase="solve", build_s=build_s, solve_start=solve_start)
    model.optimize()
    solve_s = time.perf_counter() - solve_start
    choice, lower = None, None
    if model.getNSols():
        sol = model.getBestSol()
        values = np.array([[model.getSolVal(sol, v) for v in row] for row in x])
        assert np.all(np.abs(values - np.rint(values)) < 1e-5)
        assert np.all(np.rint(values).sum(axis=1) <= 1)
        choice = np.where(values.max(axis=1) > 0.5, values.argmax(axis=1), -1)
        lower, _ = evaluate(data, choice)
        assert abs(lower - model.getObjVal()) < 0.1
    return dict(
        status=str(model.getStatus()),
        build_s=build_s,
        solve_s=solve_s,
        lower=lower,
        upper=float(model.getDualbound()),
        has_incumbent=lower is not None,
        first_incumbent_s=first[0] if first else None,
        validated=choice is not None,
        version=f"SCIP {model.getMajorVersion()}.{model.getMinorVersion()}.{model.getTechVersion()}",
        **allocation_details(data, choice),
    )


def cbc_log_metrics(path):
    text = path.read_text(errors="replace") if path.exists() else ""
    times = re.findall(
        r"Cbc00(?:04|12|16)I Integer solution of ([-+\deE.]+).*?\(([\d.]+) seconds\)",
        text,
    )
    upper = re.findall(r"Upper bound:\s*([-+\deE.]+)", text)
    lower_bound = re.findall(r"Lower bound:\s*([-+\deE.]+)", text)
    if lower_bound:
        # The bound may have only six significant digits. Round outward.
        quantum = 10.0 ** Decimal(lower_bound[-1]).as_tuple().exponent
        upper = [str(-float(lower_bound[-1]) + quantum)]
    if not upper:
        partial = re.findall(
            r"best objective [-+\deE.]+ \(best possible ([-+\deE.]+)\)", text
        )
        upper = [str(-float(partial[-1]))] if partial else []
    candidates = re.findall(
        r"Cbc0038I Pass[^\n]*\(([\d.]+) seconds\)[^\n]*\nCbc0038I Solution found of ([-+\deE.]+)",
        text,
    )
    profits = [
        -float(value)
        for value in re.findall(
            r"(?:Solution found of|Rounding solution of) ([-+\deE.]+)", text
        )
    ]
    profits += [
        -float(value)
        for value in re.findall(
            r"Mini branch and bound improved solution from [-+\deE.]+ to ([-+\deE.]+)",
            text,
        )
    ]
    integer_gap = re.findall(r"Exiting as integer gap of ([-+\deE.]+) less", text)
    reported_gap = re.findall(r"^Gap:\s*([-+\deE.]+)", text, flags=re.MULTILINE)
    return {
        "first_incumbent_s": float(times[0][1]) if times else None,
        "lower": -float(times[-1][0]) if times else None,
        "upper": float(upper[-1]) if upper else float("inf"),
        "candidate_s": float(candidates[0][0]) if candidates else None,
        "candidate_profit": -float(candidates[0][1]) if candidates else None,
        "best_candidate_profit": max(profits) if profits else None,
        "absolute_gap": float(integer_gap[-1]) if integer_gap else None,
        "absolute_gap_rounding": 10.0 ** Decimal(integer_gap[-1]).as_tuple().exponent
        if integer_gap
        else None,
        "solver_reported_gap": float(reported_gap[-1]) if reported_gap else None,
    }


def cbc(data, args):
    import pulp

    start = time.perf_counter()
    n, k = data.value.shape
    # CBC's MPS command-line maximize handling differs across builds. Export -F
    # as a standard minimization MPS, and convert all objectives/bounds back to F.
    model = pulp.LpProblem("coupon_soft", pulp.LpMinimize)
    x = [
        [model.add_variable(f"x_{i}_{j}", 0, 1, cat="Binary") for j in range(k)]
        for i in range(n)
    ]
    slack = [model.add_variable(f"slack_{m}", 0) for m in range(len(data.budget))]
    model += -pulp.lpSum(
        float(data.value[i, j]) * x[i][j] for i in range(n) for j in range(k)
    ) + pulp.lpSum(float(p) * slack[m] for m, p in enumerate(data.penalty))
    for row in x:
        model += pulp.lpSum(row) <= 1
    for m, budget in enumerate(data.budget):
        rows, cols = np.nonzero(data.resource[:, :, m])
        model += pulp.lpSum(
            float(data.resource[i, j, m]) * x[i][j]
            for i, j in zip(rows.tolist(), cols.tolist())
        ) - slack[m] <= float(budget)
    model_s = time.perf_counter() - start
    with tempfile.TemporaryDirectory(prefix="coupon-cbc-") as folder:
        mps, solution = Path(folder) / "model.mps", Path(folder) / "model.sol"
        model.writeMPS(str(mps), rename=False)
        build_s = time.perf_counter() - start
        command = [
            pulp.COIN_CMD().path,
            str(mps),
            "-timeMode",
            "elapsed",
            "-seconds",
            str(args.seconds),
            "-ratio",
            str(args.gap),
            "-threads",
            "1",
            "-randomSeed",
            "1",
            "-randomCbcSeed",
            "1",
            "-solve",
            "-solution",
            str(solution),
        ]
        solve_start = time.perf_counter()
        emit(
            args,
            kind="phase",
            phase="solve",
            model_s=model_s,
            build_s=build_s,
            solve_start=solve_start,
        )
        with args.engine_log.open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        solve_s = time.perf_counter() - solve_start
        metrics = cbc_log_metrics(args.engine_log)
        choice, lower, status = None, None, f"exit {code}"
        solution_file, allocation_file = None, None
        if solution.exists():
            lines = solution.read_text().splitlines()
            status = lines[0]
            assert not status.lower().startswith(("unbounded", "infeasible")), status
            # CBC may write an LP point with "no integer solution". Do not count it as incumbent.
            if (
                "objective value" in status.lower()
                and "no integer solution" not in status.lower()
            ):
                values = np.zeros((n, k))
                for line in lines[1:]:
                    fields = line.split()
                    if fields and fields[0] == "**":
                        fields = fields[1:]
                    if len(fields) >= 3 and fields[1].startswith("x_"):
                        _, i, j = fields[1].split("_")
                        values[int(i), int(j)] = float(fields[2])
                assert np.all(np.abs(values - np.rint(values)) < 1e-5)
                assert np.all(np.rint(values).sum(axis=1) <= 1)
                choice = np.where(values.max(axis=1) > 0.5, values.argmax(axis=1), -1)
                lower, used = evaluate(data, choice)
                reported = -float(status.rsplit("objective value", 1)[1])
                assert abs(lower - reported) < 0.1
                solution_file = args.result.with_suffix(".sol")
                allocation_file = args.result.with_suffix(".npz")
                shutil.copyfile(solution, solution_file)
                np.savez_compressed(
                    allocation_file,
                    choice=choice,
                    usage=used,
                    budget=data.budget,
                    penalty=data.penalty,
                )
        if lower is not None and metrics["absolute_gap"] is not None:
            metrics["upper"] = math.nextafter(
                lower + metrics["absolute_gap"] + metrics["absolute_gap_rounding"],
                math.inf,
            )
        if status.lower().startswith("optimal") and "gap" not in status.lower():
            metrics["upper"] = lower
        return dict(
            status=status,
            model_s=model_s,
            build_s=build_s,
            solve_s=solve_s,
            lower=lower,
            upper=metrics["upper"],
            solver_reported_gap=metrics["solver_reported_gap"],
            has_incumbent=lower is not None,
            first_incumbent_s=metrics["first_incumbent_s"],
            candidate_s=metrics["candidate_s"],
            candidate_profit=metrics["candidate_profit"],
            validated=choice is not None,
            solution_file=str(solution_file) if solution_file else None,
            allocation_file=str(allocation_file) if allocation_file else None,
            **allocation_details(data, choice),
        )


def descartes(data, args):
    import descartes_mip.algorithms.lagrangian as algorithm
    from descartes_mip import Model, VariableType, quicksum

    start = time.perf_counter()
    model = Model("coupon_soft")
    n, k = data.value.shape
    x = [
        [model.add_var(f"x_{i}_{j}", vartype=VariableType.BINARY) for j in range(k)]
        for i in range(n)
    ]
    model.maximize(
        quicksum(float(data.value[i, j]) * x[i][j] for i in range(n) for j in range(k))
    )
    for row in x:
        model.add_constraint(quicksum(row) <= 1)
    for m, budget in enumerate(data.budget):
        rows, cols = np.nonzero(data.resource[:, :, m])
        constraint = model.add_constraint(
            quicksum(
                float(data.resource[i, j, m] / budget) * x[i][j]
                for i, j in zip(rows.tolist(), cols.tolist())
            )
            <= 1
        )
        model.relax(constraint, penalty=float(data.penalty[m] * budget))
    build_s = time.perf_counter() - start
    original_analyze = algorithm.analyze
    decomposition_s = []

    def timed_analyze(*positional, **keywords):
        phase_start = time.perf_counter()
        emit(args, kind="phase", phase="decomposition")
        result = original_analyze(*positional, **keywords)
        decomposition_s.append(time.perf_counter() - phase_start)
        emit(
            args,
            kind="phase",
            phase="subproblem_setup",
            decompose_s=decomposition_s[-1],
            subproblems=len(result.subproblems),
        )
        return result

    # Timing-only wrapper; decomposition and solve algorithms are unchanged.
    algorithm.analyze = timed_analyze
    first = []
    solve_start = time.perf_counter()
    emit(args, kind="phase", phase="solve", build_s=build_s, solve_start=solve_start)

    def callback(record):
        elapsed = time.perf_counter() - solve_start
        if not first:
            first.append(elapsed)
        emit(
            args,
            kind="iteration",
            iteration=record.iteration,
            seconds=elapsed,
            lower=record.best_primal_bound,
            upper=record.best_dual_bound,
            first_incumbent_s=first[0],
        )
        return (record.best_dual_bound - record.best_primal_bound) / max(
            1, abs(record.best_primal_bound)
        ) <= args.gap

    try:
        result = model.solve(
            time_limit=args.seconds,
            max_iterations=20000,
            highs_options={"threads": 1, "output_flag": False},
            callback=callback,
        )
    finally:
        algorithm.analyze = original_analyze
    solve_s = time.perf_counter() - solve_start
    choice, lower = None, None
    if result.soft_feasible:
        values = np.array([[result.value(v) for v in row] for row in x])
        assert np.all(np.abs(values - np.rint(values)) < 1e-5)
        assert np.all(np.rint(values).sum(axis=1) <= 1)
        choice = np.where(values.max(axis=1) > 0.5, values.argmax(axis=1), -1)
        lower, _ = evaluate(data, choice)
        assert abs(lower - result.best_primal_bound) < 0.1
    return dict(
        status=result.termination_reason,
        build_s=build_s,
        solve_s=solve_s,
        decompose_s=decomposition_s[0] if decomposition_s else None,
        iterations=result.iterations,
        subproblems=len(result.decomposition.subproblems),
        lower=lower,
        upper=result.best_dual_bound,
        first_incumbent_s=first[0] if first else None,
        has_incumbent=lower is not None,
        validated=choice is not None,
        **allocation_details(data, choice),
    )


def worker(args):
    emit(args, kind="phase", phase="build")
    data = generate(
        args.users,
        args.coupons,
        args.segments,
        0.45,
        args.seed,
        args.penalty_scale,
        profile=args.profile,
    )
    if args.worker == "CBC":
        result = cbc(data, args)
    elif args.worker == "SCIP":
        result = scip(data, args)
    elif args.worker == "DESCARTES":
        result = descartes(data, args)
    elif args.worker == "HIGHS":
        result = direct(
            data,
            seconds=args.seconds,
            threads=1,
            msg=True,
            target_gap=args.gap,
        )
    else:
        result, _ = lagrangian(data, args.seconds, 20000, target_gap=args.gap)
    finish(args, result)


def events(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=100000)
    parser.add_argument("--coupons", type=int, default=5)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--penalty-scale", type=float, default=2)
    parser.add_argument(
        "--profile",
        choices=[
            "structured",
            "hard-correlated",
            "hard-correlated-raw",
            "hard-correlated-unit-budget",
        ],
        default="structured",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--gap", type=float, default=0.001)
    parser.add_argument(
        "--wait-for-result",
        action="store_true",
        help="外部期限を無効にし、ソルバー終了・解の出力まで待つ",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["CBC", "SCIP", "DESCARTES", "NUMPY"],
        choices=["CBC", "SCIP", "HIGHS", "DESCARTES", "NUMPY"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--worker",
        choices=["CBC", "SCIP", "HIGHS", "DESCARTES", "NUMPY"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--events", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--engine-log", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.users, args.coupons, args.segments, args.seconds, args.penalty_scale)
        <= 0
        or args.gap < 0
    ):
        parser.error(
            "sizes, seconds and penalty must be positive; gap must be nonnegative"
        )
    if args.output is None:
        args.output = (
            Path(__file__).parent / "results" / f"solver_comparison_n{args.users}.csv"
        )
    if args.worker:
        worker(args)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for method in args.methods:
        prefix = args.output.with_name(args.output.stem + "_" + method.lower())
        event_path, result_path = (
            Path(str(prefix) + ".events.jsonl"),
            Path(str(prefix) + ".json"),
        )
        engine_log = Path(str(prefix) + ".solver.log")
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            method,
            "--users",
            str(args.users),
            "--coupons",
            str(args.coupons),
            "--segments",
            str(args.segments),
            "--penalty-scale",
            str(args.penalty_scale),
            "--profile",
            args.profile,
            "--seed",
            str(args.seed),
            "--seconds",
            str(args.seconds),
            "--gap",
            str(args.gap),
            "--events",
            str(event_path),
            "--result",
            str(result_path),
            "--engine-log",
            str(engine_log),
        ]
        # Each output is a new measurement; no stale event/result is used.
        event_path.write_text("")
        result_path.write_text("")
        started = time.perf_counter()
        print(
            f"{method}: users={args.users:,}, coupons={args.coupons}, running",
            flush=True,
        )
        timed_out = False
        with Path(str(prefix) + ".worker.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                observed = events(event_path)
                solve = next((e for e in observed if e.get("phase") == "solve"), None)
                deadline = (
                    solve["solve_start"] + args.seconds + 3
                    if solve
                    else started + args.seconds + 3
                )
                if not args.wait_for_result and time.perf_counter() > deadline:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(0.25)
        observed = events(event_path)
        # A validated result may already exist while native-model destruction
        # continues. Keep it even if the worker cleanup exceeded the deadline.
        if result_path.exists() and result_path.stat().st_size:
            record = json.loads(result_path.read_text())
            if timed_out:
                record["worker_cleanup_interrupted"] = True
        else:
            solve = next((e for e in observed if e.get("phase") == "solve"), None)
            partial = [e for e in observed if e["kind"] in ("incumbent", "iteration")]
            last = partial[-1] if partial else {}
            metric = cbc_log_metrics(engine_log) if method == "CBC" else {}
            record = {
                "method": method,
                "users": args.users,
                "coupons": args.coupons,
                "segments": args.segments,
                "seed": args.seed,
                "penalty_scale": args.penalty_scale,
                "profile": args.profile,
                "target_gap": args.gap,
                "time_limit_s": args.seconds,
                "status": "External timeout"
                if timed_out
                else f"Worker error {process.returncode}",
                "phase": observed[-1].get("phase", "solve") if observed else "startup",
                "build_s": solve.get("build_s") if solve else None,
                "solve_s": time.perf_counter() - solve["solve_start"]
                if solve
                else None,
                "lower": last.get("lower", metric.get("lower")),
                "upper": last.get("upper", metric.get("upper")),
                "first_incumbent_s": last.get(
                    "first_incumbent_s", metric.get("first_incumbent_s")
                ),
                "validated": False,
                "iterations": last.get("iteration", -1) + 1
                if method == "DESCARTES"
                else None,
            }
            lb, ub = record["lower"], record["upper"]
            record["gap"] = (
                max(0, ub - lb) / max(1, abs(lb))
                if lb is not None and ub is not None
                else None
            )
        record["process_s"] = time.perf_counter() - started
        record["external_deadline_enabled"] = not args.wait_for_result
        records.append(record)
        save_csv(args.output, records)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("descartes-mip", "pulp", "pyscipopt", "highspy", "cbcbox")
    }
    args.output.with_suffix(".versions.json").write_text(
        json.dumps(versions, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
