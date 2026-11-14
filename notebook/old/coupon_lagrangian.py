# %% [markdown]
# # ソフト制約付きクーポン配布: 直接 MIP vs ラグランジュ分解
# VS Code / Jupyter のセルとしても、Python スクリプトとしても実行可能。
#
# max sum(v*x) - sum(rho*s), sum_k x[i,k] <= 1, sum(a*x) <= B+s, s >= 0
# 配布しない選択肢を含める。予算・券種別枚数・属性別予算をソフト化して緩和すると
# U(lambda) = lambda @ B + sum_i max(0, max_k(v[i,k] - a[i,k] @ lambda))。
# 0 <= lambda <= rho。U は **上界**。超過ペナルティを差し引いた利益が **下界**。
# 各ユーザーの選択制約を満たす任意の x は、s=max(0, sum(a*x)-B) とすれば実行可能。
# 実行可能解への修復は不要。詳しい定式化はリポジトリの README.md を参照。
# 双対最適値はこの定式化の LP 緩和値と一致するが、有限反復では一致しない。
#
# 実行例（各手法の求解時間上限。MIP/LP のモデル構築時間は別計測）:
# uv run python notebook/coupon_lagrangian.py
# uv run python notebook/coupon_lagrangian.py --users 100000 --seconds 60 --iterations 1000
# uv run python notebook/coupon_lagrangian.py --users 20000 --segments 20 --seeds 0 1 2
# 大規模時のメモリは O(users * coupons * resources)。まず小規模で確認する。

# %%
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import highspy
import numpy as np
import pulp


@dataclass
class Instance:
    value: np.ndarray  # [user, coupon], 増分利益
    resource: np.ndarray  # [user, coupon, resource]
    budget: np.ndarray
    penalty: np.ndarray  # 資源1単位の超過に対するペナルティ rho[m]


def generate(
    users=3000,
    coupons=5,
    segments=8,
    tightness=0.45,
    seed=0,
    penalty_scale=1.0,
    profile="structured",
):
    """同一 seed で再現可能な合成データ。

    ``hard-correlated`` は、多次元 multiple-choice knapsack の難しい例を
    狙った実験用 profile。利益と資源消費を強く相関させ、各選択肢が少数の
    重複予算だけを消費する。制約は従来どおり全てソフトなので修復は不要。
    """
    if (
        users < 1
        or coupons < 1
        or segments < 1
        or not 0 < tightness < 1
        or penalty_scale <= 0
        or profile
        not in {
            "structured",
            "hard-correlated",
            "hard-correlated-raw",
            "hard-correlated-unit-budget",
        }
    ):
        raise ValueError(
            "positive sizes, 0 < tightness < 1, and a known profile are required"
        )
    rng = np.random.default_rng(seed)
    if profile in {
        "hard-correlated",
        "hard-correlated-raw",
        "hard-correlated-unit-budget",
    }:
        # 全体原資 + 多数の疎な重複 campaign budgets。平均8本を消費する。
        # strongly-correlated knapsack と同様に利益を資源消費量へ近づけ、
        # 支配される選択肢や明白な枝を減らす。小さなノイズで完全な対称性だけ除く。
        density = min(1.0, 8.0 / segments)
        campaign = rng.integers(1, 11, size=(users, coupons, segments)).astype(
            np.float32
        )
        campaign *= rng.random((users, coupons, segments)) < density
        # 稀に全 campaign 係数が0となる選択肢にも、必ず一つ資源を持たせる。
        empty_i, empty_k = np.nonzero(campaign.sum(axis=2) == 0)
        if len(empty_i):
            forced = rng.integers(0, segments, size=len(empty_i))
            campaign[empty_i, empty_k, forced] = rng.integers(1, 11, size=len(empty_i))
        total_cost = rng.integers(40, 161, size=(users, coupons)).astype(np.float32)
        resource = np.empty((users, coupons, 1 + segments), dtype=np.float32)
        resource[:, :, 0] = total_cost
        resource[:, :, 1:] = campaign
        # 限界利益/資源比を揃えた上で、ユーザー・券種固有の小さな差を加える。
        correlated = total_cost + 18.0 * campaign.sum(axis=2)
        value = np.maximum(
            1,
            np.rint(correlated + rng.normal(0, 12, size=(users, coupons))),
        )
        budget = np.maximum(
            1, np.floor(tightness * resource.mean(axis=1).sum(axis=0))
        ).astype(np.float32)
        # 超過は限界利益より少し割高。ほぼハードだが、全割当は実行可能。
        penalty = penalty_scale * np.r_[1.08, np.full(segments, 19.5)]
        if profile in {"hard-correlated", "hard-correlated-unit-budget"}:
            # 全ソルバーへ同じ数値スケーリングを適用する。z_m=s_m/d_m と置けば
            # a*x <= B+s, -rho*s は (a/d)*x <= B/d+z, -(rho*d)*z と同値。
            # 標準profileは行の最大係数を1にし、巨大な右辺と極小係数をともに避ける。
            scale = (
                budget.copy()
                if profile == "hard-correlated-unit-budget"
                else resource.max(axis=(0, 1))
            )
            resource /= scale[None, None, :]
            budget /= scale
            penalty *= scale
        return Instance(value.astype(float), resource, budget, penalty)

    rates = np.linspace(0.05, 0.25, coupons)
    basket = rng.integers(1000, 10001, size=(users, 1))
    redemption = rng.uniform(0.1, 0.8, size=(users, coupons))
    cost = np.maximum(1, np.rint(basket * rates * redemption))
    # 利益と原資が相関しつつ、ユーザーごとに好ましい券種が異なる。
    value = np.maximum(1, np.rint(cost * rng.uniform(0.8, 1.6, cost.shape)))
    a = np.zeros((users, coupons, 1 + coupons + segments), dtype=np.float64)
    a[:, :, 0] = cost
    for k in range(coupons):
        a[:, k, 1 + k] = 1
    # 重複する属性セグメント。1人が複数の属性別予算を消費する。
    membership = rng.random((users, segments)) < 0.35
    a[:, :, 1 + coupons :] = cost[:, :, None] * membership[:, None, :]
    # 全員にランダムな券種を1枚配る際の期待消費量の一定割合。
    budget = np.maximum(1, np.floor(tightness * a.mean(axis=1).sum(axis=0)))
    # 全体原資: 0.8/超過1円、券種別枚数: 150/超過1枚、属性別原資: 0.15/超過1円。
    penalty = (
        penalty_scale * np.r_[0.8, np.full(coupons, 150.0), np.full(segments, 0.15)]
    )
    return Instance(value.astype(float), a, budget, penalty)


def evaluate(data, choice):
    """choice=-1 は配布なし。超過を許し、そのペナルティを引いた目的値を返す。"""
    n, k = data.value.shape
    choice = np.asarray(choice)
    assert choice.shape == (n,)
    assert np.all(choice == choice.astype(int))
    assert np.all((choice >= -1) & (choice < k))
    rows = np.flatnonzero(choice >= 0)
    used = data.resource[rows, choice[rows]].sum(axis=0)
    slack = np.maximum(used - data.budget, 0)
    gross = float(data.value[rows, choice[rows]].sum())
    return gross - float(data.penalty @ slack), used


def allocation_details(data, choice):
    if choice is None:
        return {
            "gross_profit": None,
            "penalty_cost": None,
            "max_overrun_pct": None,
            "allocated_users": None,
        }
    objective, used = evaluate(data, choice)
    cost = float(data.penalty @ np.maximum(used - data.budget, 0))
    return {
        "gross_profit": objective + cost,
        "penalty_cost": cost,
        "max_overrun_pct": float(100 * np.maximum(used / data.budget - 1, 0).max()),
        "allocated_users": int(np.count_nonzero(choice >= 0)),
    }


# %% [markdown]
# 直接法: 同じ係数で PuLP モデルを作り、HiGHS で MIP / LP 緩和を解く。
# LP の目的値は最適まで解けた場合のみ上界として採用する。
# 時間制限時にソルバーが返す値を実行可能解と取り違えないよう、整数性と選択制約を検査。


# %%
def direct(data, seconds=10, relax=False, threads=1, msg=False, target_gap=0.0):
    start = perf_counter()
    n, k = data.value.shape
    model = pulp.LpProblem("coupon", pulp.LpMaximize)
    x = [
        [
            model.add_variable(
                f"x_{i}_{j}", 0, 1, cat="Continuous" if relax else "Binary"
            )
            for j in range(k)
        ]
        for i in range(n)
    ]
    slack = [model.add_variable(f"slack_{m}", 0) for m in range(len(data.budget))]
    model += pulp.lpSum(
        float(data.value[i, j]) * x[i][j] for i in range(n) for j in range(k)
    ) - pulp.lpSum(float(data.penalty[m]) * slack[m] for m in range(len(slack)))
    for i in range(n):
        model += pulp.lpSum(x[i]) <= 1
    for m, budget in enumerate(data.budget):
        rows, cols = np.nonzero(data.resource[:, :, m])
        model += (
            pulp.lpSum(
                float(data.resource[i, j, m]) * x[i][j]
                for i, j in zip(rows.tolist(), cols.tolist())
            )
            <= float(budget) + slack[m]
        )
    model_seconds = perf_counter() - start
    solver = pulp.HiGHS(
        timeLimit=seconds,
        gapRel=target_gap,
        threads=threads,
        msg=msg,
        random_seed=0,
    )
    # model.solve() は転送と解の取り出しも含むため、各段階を明示的に計測する。
    transfer_start = perf_counter()
    solver.createAndConfigureSolver(model)
    solver.buildSolverModel(model)
    transfer_seconds = perf_counter() - transfer_start
    solve_start = perf_counter()
    solver.callSolver(model)
    solve_seconds = perf_counter() - solve_start
    extract_start = perf_counter()
    solver.findSolutionValues(model)
    highs = model.solverModel
    info = highs.getInfo()
    optimal = highs.getModelStatus() == highspy.HighsModelStatus.kOptimal
    # PuLP 4.2 は HiGHS の目的方向を maximize に設定する。符号反転は不要。
    upper = (
        float(highs.getObjectiveValue())
        if relax and optimal
        else (float(info.mip_dual_bound) if not relax else float("inf"))
    )
    lower = None
    choice = None
    if (
        not relax
        and info.primal_solution_status
        == highspy.SolutionStatus.kSolutionStatusFeasible
    ):
        values = np.array([[v.varValue for v in row] for row in x], dtype=float)
        if np.all(np.isfinite(values)) and np.all(
            np.abs(values - np.rint(values)) < 1e-5
        ):
            assert np.all(np.rint(values).sum(axis=1) <= 1)
            choice = np.where(values.max(axis=1) > 0.5, values.argmax(axis=1), -1)
            lower, _ = evaluate(data, choice)
    details = allocation_details(data, choice)
    return {
        "method": "LP" if relax else "MIP",
        "lower": lower,
        "upper": upper,
        "status": highs.modelStatusToString(highs.getModelStatus()),
        "model_s": model_seconds,
        "transfer_s": transfer_seconds,
        "build_s": model_seconds + transfer_seconds,
        "solve_s": solve_seconds,
        "extract_s": perf_counter() - extract_start,
        "highs_run_s": highs.getRunTime(),
        "total_s": perf_counter() - start,
        **details,
    }


# %% [markdown]
# 分解法: 正規化した資源 b[i,k,m] = a[i,k,m] / B[m] を用いる。
# U(p) = sum(p) + sum_i max(0, max_k(v[i,k] - b[i,k] @ p))。
# g = 1 - sum(b*x) は劣勾配。p <- clip(p - step*g, 0, rho*B) で上界を最小化。
# Polyak 型のステップに既知の実行可能利益を用いる（厳密な双対収束の保証はしない）。
# 各反復で得た x に対して超過ペナルティを計算するだけで実行可能目的値を得る。


# %%
def lagrangian(data, seconds=10, iterations=500, target_gap=0.0):
    start = perf_counter()
    b = data.resource / data.budget
    n, _ = data.value.shape
    prices = np.zeros(len(data.budget))
    cap = data.penalty * data.budget
    rows = np.arange(n)
    best_choice = np.full(n, -1, dtype=int)  # 空配布は常に実行可能
    lower, upper = 0.0, float("inf")
    history = []
    build_seconds = perf_counter() - start
    solve_start = perf_counter()
    for t in range(iterations):
        if t and perf_counter() - solve_start >= seconds:
            break
        adjusted = data.value - np.einsum("ikm,m->ik", b, prices, optimize=False)
        choice = adjusted.argmax(axis=1)
        chosen_value = adjusted[rows, choice]
        choice = np.where(chosen_value > 0, choice, -1)
        dual = float(prices.sum() + np.maximum(chosen_value, 0).sum())
        upper = min(upper, dual)
        active = np.flatnonzero(choice >= 0)
        raw_usage = data.resource[active, choice[active]].sum(axis=0)
        usage = raw_usage / data.budget
        gross = float(data.value[active, choice[active]].sum())
        penalty = float(data.penalty @ np.maximum(raw_usage - data.budget, 0))
        value = gross - penalty
        if value > lower:
            lower, best_choice = value, choice.copy()
        gradient = 1 - usage
        # 射影で動けない座標をステップ長の分母から除外。
        inactive = ((prices <= 0) & (gradient > 0)) | ((prices >= cap) & (gradient < 0))
        projected = np.where(inactive, 0, gradient)
        norm = float(projected @ projected)
        history.append(
            {
                "iteration": t,
                "seconds": perf_counter() - start,
                "lower": lower,
                "upper": upper,
                "current_objective": value,
                "gross_profit": gross,
                "penalty": penalty,
                "relaxed_violation": float(np.maximum(usage - 1, 0).max()),
            }
        )
        if norm < 1e-16 or upper - lower <= max(1e-6, target_gap * max(1, abs(lower))):
            break
        alpha = 1.5 / (1 + t / 50)
        prices = np.clip(
            prices - alpha * max(0, dual - lower) / norm * gradient, 0, cap
        )
    solve_seconds = perf_counter() - solve_start
    extract_start = perf_counter()
    value, _ = evaluate(data, best_choice)
    assert np.isclose(value, lower)
    details = allocation_details(data, best_choice)
    return {
        "method": "Lagrangian-soft",
        "lower": lower,
        "upper": upper,
        "status": f"{len(history)} iterations",
        "model_s": build_seconds,
        "transfer_s": 0.0,
        "build_s": build_seconds,
        "solve_s": solve_seconds,
        "extract_s": perf_counter() - extract_start,
        "highs_run_s": None,
        "total_s": perf_counter() - start,
        **details,
    }, history


# %%
def compare(
    users=3000,
    coupons=5,
    segments=8,
    tightness=0.45,
    seed=0,
    seconds=10,
    iterations=500,
    threads=1,
    msg=False,
    penalty_scale=1.0,
    target_gap=0.001,
    profile="structured",
):
    data = generate(
        users, coupons, segments, tightness, seed, penalty_scale, profile=profile
    )
    print(
        f"\nseed={seed}, users={users}, binaries={users * coupons:,}, "
        f"resources={len(data.budget)}, budget ratio={tightness}"
    )
    # 先に LR を実行し、MIP/LP の解や境界は一切渡さない。
    lr, history = lagrangian(data, seconds, iterations, target_gap=target_gap)
    mip = direct(data, seconds, threads=threads, msg=msg, target_gap=target_gap)
    lp = direct(data, seconds, relax=True, threads=threads, msg=msg)
    results = [mip, lp, lr]
    print(
        f"{'method':<20} {'feasible profit':>16} {'upper bound':>16} "
        f"{'gap %':>10} {'build s':>9} {'solve s':>9} {'total s':>9}"
    )
    for r in results:
        lb, ub = r["lower"], r["upper"]
        gap = (
            None
            if lb is None or not np.isfinite(ub)
            else 100 * max(0, ub - lb) / max(1, abs(lb))
        )
        r.update(
            seed=seed,
            users=users,
            coupons=coupons,
            segments=segments,
            tightness=tightness,
            gap_pct=gap,
            penalty_scale=penalty_scale,
            profile=profile,
        )

        def fmt(v):
            return "—" if v is None else f"{v:,.2f}"

        print(
            f"{r['method']:<20} {fmt(lb):>16} {fmt(ub):>16} {fmt(gap):>10} "
            f"{r['build_s']:9.3f} {r['solve_s']:9.3f} {r['total_s']:9.3f}  {r['status']}"
        )
    if np.isfinite(lp["upper"]):
        assert lr["lower"] <= lp["upper"] + 1e-4
        assert lr["upper"] >= lp["upper"] - 1e-4
    print(
        "gap=(upper-lower)/max(1,abs(lower)); LP の利益は整数実行可能解ではありません。"
    )
    return results, history


def save_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        fields = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


# %% [markdown]
# セル実行時は compare(users=3000, seconds=10) 等で実験できる。
# 難易度を探すには users / segments / tightness / seed を変えて比較する。
# MIP が最適まで解けるケースもそのまま報告する。
# seconds は各手法の求解予算。LR は反復単位なので多少超過しうる。
# MIP/LP は PuLP→HiGHS の変換等で超過しうる。比較時は total_s も参照。


# %%
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=3000)
    parser.add_argument("--coupons", type=int, default=5)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--tightness", type=float, default=0.45)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--penalty-scale", type=float, default=1.0)
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
    parser.add_argument("--gap", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="比較表CSV。別ファイルにLR履歴も保存"
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.iterations < 1 or args.threads < 1:
        parser.error("seconds, iterations and threads must be positive")
    results, histories = [], []
    for seed in args.seeds:
        rows, history = compare(
            args.users,
            args.coupons,
            args.segments,
            args.tightness,
            seed,
            args.seconds,
            args.iterations,
            args.threads,
            args.solver_log,
            args.penalty_scale,
            args.gap,
            args.profile,
        )
        results.extend(rows)
        histories.extend({"seed": seed, **row} for row in history)
    if args.output:
        save_csv(args.output, results)
        save_csv(args.output.with_name(args.output.stem + "_history.csv"), histories)


if __name__ == "__main__":
    # Jupyter / Interactive Window の引数を argparse に渡さない。
    if "get_ipython" not in globals():
        main()
