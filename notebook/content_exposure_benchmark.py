# %% [markdown]
# # コンテンツ配信の最適化: MIP vs. ラグランジュ分解
#
# VS Code / Jupyter の `# %%` セルとしても、CLI スクリプトとしても実行できる。
#
# * 元問題: ユーザー別の 0-1 ナップサック + コンテンツ別のソフト最低配信回数
# * MIP: 共通の疎行列/MPS を HiGHS・SCIP・CBC で解く
# * LR: 最低配信回数制約を緩和し、NumPy のバッチ型 DP でユーザーごとに厳密求解
#
# 各 LR 反復の x は容量制約を満たす。s=max(L-A(x), 0) とすれば元の
# ソフト制約付き問題で実行可能なので、修復は不要。LR の lower は不足ペナルティ
# 控除後の実行可能値、upper はラグランジュ双対上界である。
#
# 実行例:
#
#     uv run python notebook/content_exposure_benchmark.py \
#       --sizes 1000 5000 20000 --seeds 0 1 2 --seconds 15
#
# 大規模インスタンスで LR だけを実行:
#
#     uv run python notebook/content_exposure_benchmark.py \
#       --sizes 500000 --seeds 0 --methods LR --seconds 120 --iterations 100

# %%
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import highspy
import numpy as np


@dataclass
class Instance:
    score: np.ndarray  # [user, content], 関心度・親和性を表す配信スコア
    weight: np.ndarray  # [content], 表示スペース
    capacity: np.ndarray  # [user]
    minimum: np.ndarray  # [content], 最低配信回数 L_k
    penalty: float  # 全コンテンツ共通の不足1回あたりペナルティ rho


def _validate(data: Instance) -> None:
    n, k = data.score.shape
    assert data.weight.shape == (k,)
    assert data.capacity.shape == (n,)
    assert data.minimum.shape == (k,)
    assert np.all(data.weight > 0)
    assert np.all(data.capacity > 0)
    assert np.all(data.minimum > 0)
    assert data.penalty > 0


# %% [markdown]
# ## NumPy だけで解くユーザー別 0-1 ナップサック
#
# ユーザー軸をベクトル化し、アイテムと容量だけ Python ループにする。
# `batch_size` により、ユーザー数が増えても作業メモリを一定に抑える。


def numpy_knapsack(
    score: np.ndarray,
    weight: np.ndarray,
    capacity: np.ndarray,
    batch_size: int = 50_000,
) -> tuple[np.ndarray, float]:
    """全ユーザーの 0-1 ナップサックを厳密に解く。

    Returns
    -------
    selected:
        shape=(users, contents) の bool 配列。
    objective:
        部分問題の目的値合計。何も選ばない選択肢を含む。
    """
    score = np.asarray(score, dtype=np.float64)
    weight = np.asarray(weight)
    capacity = np.asarray(capacity)
    n, k = score.shape
    if weight.shape == (k,):
        weight = np.broadcast_to(weight, (n, k))
    if weight.shape != (n, k) or capacity.shape != (n,):
        raise ValueError("incompatible knapsack array shapes")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    c_max = int(capacity.max())
    selected = np.zeros((n, k), dtype=bool)
    total = 0.0
    cap_axis = np.arange(c_max + 1, dtype=np.int16)[None, :]

    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        size = hi - lo
        rows = np.arange(size)
        dp = np.zeros((size, c_max + 1), dtype=np.float64)
        take = np.zeros((k, size, c_max + 1), dtype=bool)
        wb = weight[lo:hi]
        sb = score[lo:hi]

        for item in range(k):
            source = cap_axis - wb[:, item, None]
            feasible = source >= 0
            source = np.maximum(source, 0)
            candidate = dp[rows[:, None], source] + sb[:, item, None]
            candidate[~feasible] = -np.inf
            improve = candidate > dp + 1e-12
            take[item] = improve
            dp = np.where(improve, candidate, dp)

        remaining = capacity[lo:hi].astype(np.int64, copy=True)
        for item in range(k - 1, -1, -1):
            chosen = take[item, rows, remaining]
            selected[lo:hi, item] = chosen
            remaining -= chosen * wb[:, item]
        total += float(dp[rows, capacity[lo:hi]].sum())
    return selected, total


def delivery_count(selected: np.ndarray) -> np.ndarray:
    """コンテンツごとの配信回数 A_k(x) を返す。"""
    return selected.sum(axis=0, dtype=np.float64)


def evaluate(data: Instance, selected: np.ndarray) -> tuple[float, np.ndarray]:
    """容量制約を検査し、不足ペナルティ込みの実行可能値を返す。"""
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != data.score.shape:
        raise ValueError("selected has an incompatible shape")
    used_space = (selected * data.weight[None, :]).sum(axis=1)
    if np.any(used_space > data.capacity):
        raise AssertionError("a user capacity constraint is violated")
    count = delivery_count(selected)
    total_score = float(data.score[selected].sum())
    penalty_cost = data.penalty * float(np.maximum(data.minimum - count, 0).sum())
    return total_score - penalty_cost, count


# %% [markdown]
# ## 合成インスタンス
#
# compact / standard / rich の表示レイアウトごとにユーザー容量を変える。
# コンテンツの表示サイズは全ユーザーで共通とし、配信スコアはコンテンツの魅力、
# ユーザーセグメントとの親和性、個別ノイズから生成する。
#
# `tightness` は、制約なし解の総配信数に対する全コンテンツの最低配信回数合計の
# 比率。各コンテンツへの配分は販促上の優先度に比例させる。


def generate(
    users: int = 1_000,
    contents: int = 20,
    tightness: float = 0.9,
    penalty: float = 10.0,
    seed: int = 0,
    batch_size: int = 50_000,
) -> Instance:
    if users < 1 or contents < 1:
        raise ValueError("users and contents must be positive")
    if not 0 < tightness <= 1.2 or penalty <= 0:
        raise ValueError("require 0 < tightness <= 1.2 and penalty > 0")
    rng = np.random.default_rng(seed)
    layout = rng.choice(3, size=users, p=[0.35, 0.45, 0.20])
    base_capacity = np.array([10, 14, 18], dtype=np.int16)
    capacity = np.clip(
        base_capacity[layout] + rng.integers(-2, 3, size=users), 8, 20
    ).astype(np.int16)
    weight = rng.choice(
        np.arange(1, 6, dtype=np.int16),
        size=contents,
        p=[0.25, 0.28, 0.23, 0.15, 0.09],
    )

    # コンテンツ魅力とセグメント適合度に個別ノイズを加えた配信スコア。
    segments = 8
    user_segment = rng.integers(0, segments, size=users)
    appeal = rng.normal(0.0, 2.0, size=contents)
    affinity = rng.normal(0.0, 3.0, size=(segments, contents))
    score = (
        8.0
        + appeal[None, :]
        + affinity[user_segment]
        + rng.normal(0.0, 3.0, size=(users, contents))
    )
    score = np.maximum(score, 0.25).astype(np.float32)

    # 制約なしの配信数を基準に、販促上の優先度に沿って目標を設定。
    unconstrained, _ = numpy_knapsack(score, weight, capacity, batch_size)
    total_selected = int(unconstrained.sum())
    priority = rng.lognormal(mean=0.0, sigma=0.25, size=contents)
    priority /= priority.sum()
    minimum = np.maximum(1.0, np.rint(tightness * total_selected * priority))
    data = Instance(score, weight, capacity, minimum, float(penalty))
    _validate(data)
    return data


# %% [markdown]
# ## ラグランジュ分解
#
# $$U(\lambda)=-\sum_k\lambda_kL_k+\sum_i\max_{x_i\in X_i}
# \sum_k(v_{ik}+\lambda_k)x_{ik}.$$
#
# $0\le\lambda_k\le\rho$。劣勾配は $A_k(x)-L_k$ なので、
# $\lambda\leftarrow\operatorname{clip}(\lambda-\eta g,0,\rho)$ と更新する。


def lagrangian(
    data: Instance,
    seconds: float = 15.0,
    iterations: int = 200,
    target_gap: float = 0.001,
    batch_size: int = 50_000,
) -> tuple[dict, list[dict]]:
    start = perf_counter()
    contents = len(data.minimum)
    multiplier = np.zeros(contents, dtype=np.float64)
    empty = np.zeros_like(data.score, dtype=bool)
    best_lower, _ = evaluate(data, empty)
    best_upper = float("inf")
    best_selected = empty
    history: list[dict] = []
    theta = 1.8
    stalled = 0
    patience = 10
    build_s = perf_counter() - start
    solve_start = perf_counter()

    for iteration in range(iterations):
        if iteration and perf_counter() - solve_start >= seconds:
            break
        bonus = multiplier[None, :]
        selected, local_value = numpy_knapsack(
            data.score + bonus, data.weight, data.capacity, batch_size
        )
        dual = float(-(multiplier @ data.minimum) + local_value)
        previous_best_upper = best_upper
        best_upper = min(best_upper, dual)
        tolerance = 1e-6 * max(1.0, abs(best_upper))
        if (
            not np.isfinite(previous_best_upper)
            or dual < previous_best_upper - tolerance
        ):
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                theta = max(0.01, 0.9 * theta)
                stalled = 0
        primal, count = evaluate(data, selected)
        if primal > best_lower:
            best_lower = primal
            best_selected = selected.copy()

        relative_gap = max(0.0, best_upper - best_lower) / max(1.0, abs(best_lower))
        shortage = np.maximum(data.minimum - count, 0)
        history.append(
            {
                "iteration": iteration,
                "seconds": perf_counter() - solve_start,
                "current_lower": primal,
                "best_lower": best_lower,
                "current_upper": dual,
                "best_upper": best_upper,
                "gap": relative_gap,
                "theta": theta,
                "shortage": float(shortage.sum()),
                "max_shortage_pct": float(100 * (shortage / data.minimum).max()),
            }
        )
        if relative_gap <= target_gap:
            break

        gradient = count - data.minimum
        blocked = ((multiplier <= 0) & (gradient > 0)) | (
            (multiplier >= data.penalty) & (gradient < 0)
        )
        projected = np.where(blocked, 0.0, gradient)
        norm_squared = float(projected @ projected)
        if norm_squared <= 1e-20:
            break
        # Polyak 型。双対上界が一定期間改善しなければ係数を下げる。
        step = theta * max(dual - best_lower, 0.0) / norm_squared
        multiplier = np.clip(multiplier - step * gradient, 0.0, data.penalty)

    objective, count = evaluate(data, best_selected)
    solve_s = perf_counter() - solve_start
    result = {
        "method": "LR-NumPy",
        "status": "target gap"
        if history and history[-1]["gap"] <= target_gap
        else "limit",
        "lower": objective,
        "upper": best_upper,
        "gap": max(0.0, best_upper - objective) / max(1.0, abs(objective)),
        "build_s": build_s,
        "solve_s": solve_s,
        "extract_s": 0.0,
        "total_s": perf_counter() - start,
        "iterations": len(history),
        "final_theta": theta,
        "total_score": float(data.score[best_selected].sum()),
        "penalty_cost": data.penalty * float(np.maximum(data.minimum - count, 0).sum()),
        "total_shortage": float(np.maximum(data.minimum - count, 0).sum()),
        "max_shortage_pct": float(
            100 * (np.maximum(data.minimum - count, 0) / data.minimum).max()
        ),
    }
    return result, history


# %% [markdown]
# ## HiGHS への直接 MIP 登録


def _build_highs_model(
    data: Instance,
    seconds: float = 15.0,
    target_gap: float = 0.001,
    threads: int = 1,
) -> tuple[highspy.Highs, float]:
    start = perf_counter()
    n, k = data.score.shape
    binaries = n * k
    columns = binaries + k
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("threads", threads)
    model.setOptionValue("random_seed", 0)
    model.setOptionValue("mip_rel_gap", target_gap)
    model.setOptionValue("time_limit", seconds)

    empty_i = np.empty(0, dtype=np.int32)
    model.addCols(
        columns,
        np.r_[data.score.ravel(), np.full(k, -data.penalty)],
        np.zeros(columns),
        np.r_[np.ones(binaries), np.full(k, highspy.kHighsInf)],
        0,
        empty_i,
        empty_i,
        np.empty(0),
    )
    model.changeObjectiveSense(highspy.ObjSense.kMaximize)
    model.changeColsIntegrality(
        binaries,
        np.arange(binaries, dtype=np.int32),
        np.full(binaries, highspy.HighsVarType.kInteger, dtype=np.uint8),
    )

    # CSR 行列: 前半がユーザー容量、後半がコンテンツ最低配信回数。
    nnz = 2 * binaries + k
    indices = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.float64)
    indices[:binaries] = np.arange(binaries, dtype=np.int32)
    values[:binaries] = np.tile(data.weight, n)
    starts = np.empty(n + k, dtype=np.int64)
    starts[:n] = np.arange(0, binaries, k, dtype=np.int64)

    cursor = binaries
    for content in range(k):
        starts[n + content] = cursor
        chosen = np.arange(content, binaries, k, dtype=np.int32)
        indices[cursor : cursor + n] = chosen
        values[cursor : cursor + n] = 1.0
        cursor += n
        indices[cursor] = binaries + content
        values[cursor] = 1.0
        cursor += 1
    assert cursor == nnz
    model.addRows(
        n + k,
        np.r_[np.full(n, -highspy.kHighsInf), data.minimum],
        np.r_[data.capacity, np.full(k, highspy.kHighsInf)],
        nnz,
        starts,
        indices,
        values,
    )
    return model, perf_counter() - start


def _allocation_metrics(data: Instance, selected: np.ndarray | None) -> dict:
    if selected is None:
        return {
            "lower": None,
            "total_score": None,
            "penalty_cost": None,
            "total_shortage": None,
            "max_shortage_pct": None,
        }
    lower, count = evaluate(data, selected)
    shortage = np.maximum(data.minimum - count, 0)
    return {
        "lower": lower,
        "total_score": float(data.score[selected].sum()),
        "penalty_cost": data.penalty * float(shortage.sum()),
        "total_shortage": float(shortage.sum()),
        "max_shortage_pct": float(100 * (shortage / data.minimum).max()),
    }


def native_mip(
    data: Instance,
    seconds: float = 15.0,
    target_gap: float = 0.001,
    threads: int = 1,
) -> dict:
    start = perf_counter()
    n, k = data.score.shape
    binaries = n * k
    model, build_s = _build_highs_model(data, seconds, target_gap, threads)
    # 親プロセスが外部タイムアウトしても構築時間を回収できる。
    print(
        json.dumps({"event": "solve_start", "build_s": build_s}),
        file=sys.stderr,
        flush=True,
    )
    solve_start = perf_counter()
    model.run()
    solve_s = perf_counter() - solve_start
    extract_start = perf_counter()
    info = model.getInfo()
    solution = model.getSolution()
    selected = None
    if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
        x = np.asarray(solution.col_value[:binaries]).reshape(n, k)
        if np.all(np.abs(x - np.rint(x)) <= 1e-5):
            selected = x > 0.5
    metrics = _allocation_metrics(data, selected)
    lower = metrics["lower"]
    upper = float(info.mip_dual_bound)
    gap = (
        None
        if lower is None or not np.isfinite(upper)
        else max(0.0, upper - lower) / max(1.0, abs(lower))
    )
    return {
        "method": "MIP-HiGHS",
        "status": model.modelStatusToString(model.getModelStatus()),
        "lower": lower,
        "upper": upper,
        "gap": gap,
        "build_s": build_s,
        "solve_s": solve_s,
        "extract_s": perf_counter() - extract_start,
        "total_s": perf_counter() - start,
        "iterations": None,
        **metrics,
    }


def scip_mip(
    data: Instance,
    seconds: float = 15.0,
    target_gap: float = 0.001,
    threads: int = 1,
) -> dict:
    """HiGHS で作った共通 MPS を SCIP に読ませて求解する。"""
    from pyscipopt import Model

    started = perf_counter()
    n, k = data.score.shape
    binaries = n * k
    with tempfile.TemporaryDirectory(prefix="content-delivery-scip-") as folder:
        path = Path(folder) / "model.mps"
        source, matrix_s = _build_highs_model(data, seconds, target_gap, threads)
        source.writeModel(str(path))
        export_s = perf_counter() - started - matrix_s
        model = Model("content_delivery")
        model.hideOutput()
        model.setRealParam("limits/time", seconds)
        model.setRealParam("limits/gap", target_gap)
        model.setIntParam("parallel/maxnthreads", threads)
        model.setIntParam("randomization/randomseedshift", 0)
        model.readProblem(str(path))
        build_s = perf_counter() - started
        print(
            json.dumps({"event": "solve_start", "build_s": build_s}),
            file=sys.stderr,
            flush=True,
        )
        solve_start = perf_counter()
        model.optimize()
        solve_s = perf_counter() - solve_start
        extract_start = perf_counter()
        selected = None
        if model.getNSols() > 0:
            solution = model.getBestSol()
            values = np.zeros(binaries, dtype=np.float64)
            seen = 0
            for variable in model.getVars():
                match = re.fullmatch(r"c(\d+)", variable.name)
                if match and int(match.group(1)) < binaries:
                    values[int(match.group(1))] = model.getSolVal(solution, variable)
                    seen += 1
            if seen == binaries and np.all(np.abs(values - np.rint(values)) <= 1e-5):
                selected = values.reshape(n, k) > 0.5
        metrics = _allocation_metrics(data, selected)
        lower = metrics["lower"]
        upper = float(model.getDualbound())
        gap = (
            None
            if lower is None or not np.isfinite(upper)
            else max(0.0, upper - lower) / max(1.0, abs(lower))
        )
        return {
            "method": "MIP-SCIP",
            "status": str(model.getStatus()),
            "lower": lower,
            "upper": upper,
            "gap": gap,
            "matrix_s": matrix_s,
            "export_s": export_s,
            "build_s": build_s,
            "solve_s": solve_s,
            "extract_s": perf_counter() - extract_start,
            "total_s": perf_counter() - started,
            "iterations": None,
            **metrics,
        }


def _cbc_bound(log: str, lower: float | None, optimal: bool) -> float:
    if optimal and lower is not None:
        return lower
    raw: list[float] = []
    for pattern in (
        r"Upper bound:\s*([-+\deE.]+)",
        r"Lower bound:\s*([-+\deE.]+)",
        r"best possible\s+([-+\deE.]+)",
    ):
        raw.extend(float(value) for value in re.findall(pattern, log))
    if lower is None:
        return float("inf")
    # CBC/MPS のログは内部最小化の符号を出すビルドがある。
    candidates = [
        value for value in raw + [-value for value in raw] if value >= lower - 1e-5
    ]
    return min(candidates) if candidates else float("inf")


def cbc_mip(
    data: Instance,
    seconds: float = 15.0,
    target_gap: float = 0.001,
    threads: int = 1,
) -> dict:
    """HiGHS で作った共通 MPS を CBC CLI に渡して求解する。"""
    import pulp

    started = perf_counter()
    n, k = data.score.shape
    binaries = n * k
    with tempfile.TemporaryDirectory(prefix="content-delivery-cbc-") as folder:
        path = Path(folder)
        mps = path / "model.mps"
        solution_path = path / "model.sol"
        source, matrix_s = _build_highs_model(data, seconds, target_gap, threads)
        source.writeModel(str(mps))
        export_s = perf_counter() - started - matrix_s
        build_s = perf_counter() - started
        command = [
            pulp.COIN_CMD().path,
            str(mps),
            "-max",
            "-timeMode",
            "elapsed",
            "-seconds",
            str(seconds),
            "-ratio",
            str(target_gap),
            "-threads",
            str(threads),
            "-randomSeed",
            "1",
            "-randomCbcSeed",
            "1",
            "-solve",
            "-solution",
            str(solution_path),
        ]
        print(
            json.dumps({"event": "solve_start", "build_s": build_s}),
            file=sys.stderr,
            flush=True,
        )
        solve_start = perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        solve_s = perf_counter() - solve_start
        extract_start = perf_counter()
        log = completed.stdout + "\n" + completed.stderr
        selected = None
        status = f"exit {completed.returncode}"
        if solution_path.exists():
            lines = solution_path.read_text(errors="replace").splitlines()
            if lines:
                status = lines[0]
            values = np.zeros(binaries, dtype=np.float64)
            seen = 0
            for line in lines[1:]:
                fields = line.split()
                if fields and fields[0] == "**":
                    fields = fields[1:]
                if len(fields) < 3:
                    continue
                match = re.fullmatch(r"c(\d+)", fields[1])
                if match and int(match.group(1)) < binaries:
                    values[int(match.group(1))] = float(fields[2])
                    seen += 1
            if seen and np.all(np.abs(values - np.rint(values)) <= 1e-5):
                candidate = values.reshape(n, k) > 0.5
                try:
                    evaluate(data, candidate)
                    selected = candidate
                except AssertionError:
                    selected = None
        metrics = _allocation_metrics(data, selected)
        lower = metrics["lower"]
        optimal = status.lower().startswith("optimal") and "gap" not in status.lower()
        upper = _cbc_bound(log, lower, optimal)
        gap = (
            None
            if lower is None or not math.isfinite(upper)
            else max(0.0, upper - lower) / max(1.0, abs(lower))
        )
        return {
            "method": "MIP-CBC",
            "status": status,
            "lower": lower,
            "upper": upper,
            "gap": gap,
            "matrix_s": matrix_s,
            "export_s": export_s,
            "build_s": build_s,
            "solve_s": solve_s,
            "extract_s": perf_counter() - extract_start,
            "total_s": perf_counter() - started,
            "iterations": None,
            "solver_log_tail": " | ".join(log.splitlines()[-3:]),
            **metrics,
        }


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _worker(args: argparse.Namespace) -> None:
    started = perf_counter()
    data = generate(
        users=args.users,
        contents=args.contents,
        tightness=args.tightness,
        penalty=args.penalty,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    data_s = perf_counter() - started
    history: list[dict] = []
    if args.worker == "LR":
        result, history = lagrangian(
            data,
            seconds=args.seconds,
            iterations=args.iterations,
            target_gap=args.gap,
            batch_size=args.batch_size,
        )
    elif args.worker == "HIGHS":
        result = native_mip(
            data, seconds=args.seconds, target_gap=args.gap, threads=args.threads
        )
    elif args.worker == "SCIP":
        result = scip_mip(
            data, seconds=args.seconds, target_gap=args.gap, threads=args.threads
        )
    else:
        result = cbc_mip(
            data, seconds=args.seconds, target_gap=args.gap, threads=args.threads
        )
    result.update(
        users=args.users,
        binaries=args.users * args.contents,
        contents=args.contents,
        seed=args.seed,
        tightness=args.tightness,
        penalty=args.penalty,
        target_gap=args.gap,
        time_limit_s=args.seconds,
        threads=args.threads,
        data_s=data_s,
        process_s=perf_counter() - started,
        peak_rss_mb=_peak_rss_mb(),
        reached=result["gap"] is not None and result["gap"] <= args.gap + 1e-12,
        capacity_mean=float(data.capacity.mean()),
        weight_mean=float(data.weight.mean()),
        score_weight_corr=float(
            np.corrcoef(data.score.mean(axis=0), data.weight)[0, 1]
        ),
        target_max_min_ratio=float(data.minimum.max() / data.minimum.min()),
    )
    print(json.dumps({"result": result, "history": history}))


def _timeout_record(
    args: argparse.Namespace, method: str, users: int, seed: int, stderr: str
) -> dict:
    build_s = None
    for line in stderr.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if event.get("event") == "solve_start":
            build_s = event.get("build_s")
    return {
        "method": {
            "LR": "LR-NumPy",
            "HIGHS": "MIP-HiGHS",
            "SCIP": "MIP-SCIP",
            "CBC": "MIP-CBC",
        }[method],
        "status": "external timeout",
        "lower": None,
        "upper": None,
        "gap": None,
        "build_s": build_s,
        "solve_s": None,
        "extract_s": None,
        "total_s": None,
        "iterations": None,
        "users": users,
        "binaries": users * args.contents,
        "contents": args.contents,
        "seed": seed,
        "tightness": args.tightness,
        "penalty": args.penalty,
        "target_gap": args.gap,
        "time_limit_s": args.seconds,
        "threads": args.threads,
        "reached": False,
    }


def _print_summary(rows: list[dict]) -> None:
    print("\nmethod       users     seed  build(s)  solve(s)       gap  status")
    for row in rows:
        build = "-" if row.get("build_s") is None else f"{float(row['build_s']):8.3f}"
        solve = "-" if row.get("solve_s") is None else f"{float(row['solve_s']):8.3f}"
        gap = "-" if row.get("gap") is None else f"{100 * float(row['gap']):8.3f}%"
        print(
            f"{row['method']:<12} {int(row['users']):>8,} {int(row['seed']):>8} "
            f"{build:>9} {solve:>9} {gap:>10}  {row['status']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000, 5_000, 20_000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["LR", "HIGHS", "SCIP", "CBC"],
        default=["LR", "HIGHS", "SCIP", "CBC"],
    )
    parser.add_argument("--contents", type=int, default=20)
    parser.add_argument("--tightness", type=float, default=0.9)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wall-limit", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--gap", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "content_delivery_benchmark.csv",
    )
    parser.add_argument(
        "--worker",
        choices=["LR", "HIGHS", "SCIP", "CBC"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--users", type=int, default=1_000, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.seconds <= 0
        or args.iterations < 1
        or args.gap < 0
        or args.batch_size < 1
        or args.threads < 1
        or args.contents < 1
        or args.penalty <= 0
        or any(users < 1 for users in args.sizes)
    ):
        parser.error(
            "sizes, contents, penalty, seconds, iterations, batch-size, and threads "
            "must be positive; gap must be nonnegative"
        )
    if args.worker:
        _worker(args)
        return

    rows: list[dict] = []
    histories: list[dict] = []
    for users in args.sizes:
        for seed in args.seeds:
            for method in args.methods:
                print(f"users={users:,} seed={seed} {method}: running", flush=True)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    method,
                    "--users",
                    str(users),
                    "--seed",
                    str(seed),
                    "--contents",
                    str(args.contents),
                    "--tightness",
                    str(args.tightness),
                    "--penalty",
                    str(args.penalty),
                    "--seconds",
                    str(args.seconds),
                    "--threads",
                    str(args.threads),
                    "--iterations",
                    str(args.iterations),
                    "--gap",
                    str(args.gap),
                    "--batch-size",
                    str(args.batch_size),
                ]
                wall_limit = args.wall_limit or args.seconds + 60.0
                try:
                    completed = subprocess.run(
                        command,
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=wall_limit,
                    )
                    payload = json.loads(completed.stdout)
                    row = payload["result"]
                    for item in payload["history"]:
                        histories.append({"users": users, "seed": seed, **item})
                except subprocess.TimeoutExpired as error:
                    stderr = error.stderr or ""
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="replace")
                    row = _timeout_record(args, method, users, seed, stderr)
                rows.append(row)
                _save_csv(args.output, rows)
                if histories:
                    _save_csv(
                        args.output.with_name(args.output.stem + "_history.csv"),
                        histories,
                    )
    _print_summary(rows)
    print(f"\nresults: {args.output}")


if __name__ == "__main__":
    main()
