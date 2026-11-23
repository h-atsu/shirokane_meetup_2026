# %%
import math
import re
from pathlib import Path

import plotly.express as px
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PULP_RUN_DIR = ROOT / "outputs" / "Default" / "20261123_234319"
LAGRANGIAN_RUN_DIR = ROOT / "outputs" / "Default" / "20261124_042026"

LAGRANGIAN_PATTERN = re.compile(
    r"iteration=\d+ elapsed=(?P<time>[-+\d.eE]+).*"
    r"best_lower=(?P<primal>[-+\d.eE]+).*"
    r"best_upper=(?P<dual>[-+\d.eE]+)"
)


def _records_to_frame(
    records: list[tuple[float, float, str]],
    model: str,
) -> pl.DataFrame:
    if not records:
        raise ValueError(f"No progress records found for {model}")
    return pl.DataFrame(
        records,
        schema=["time", "objective", "problem"],
        orient="row",
    ).with_columns(pl.lit(model).alias("model"))


def load_lagrangian_progress(path: Path) -> pl.DataFrame:
    """ラグランジュ分解の最良主問題値・双対問題値を読み込む。"""
    records: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LAGRANGIAN_PATTERN.search(line)
        if match is None:
            continue
        elapsed = float(match["time"])
        records.extend(
            [
                (elapsed, float(match["primal"]), "主問題"),
                (elapsed, float(match["dual"]), "双対問題"),
            ]
        )
    return _records_to_frame(records, "Lagrangian")


def load_highs_progress(path: Path) -> pl.DataFrame:
    """HiGHSの進捗表からBestSol・BestBoundを読み込む。"""
    records: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[-1].endswith("s"):
            continue

        source_offset = int(fields[0].isalpha() and len(fields[0]) == 1)
        if len(fields) < source_offset + 12:
            continue
        if not fields[source_offset + 3].endswith("%"):
            continue

        try:
            elapsed = float(fields[-1][:-1])
            dual = float(fields[source_offset + 4])
            primal = float(fields[source_offset + 5])
        except ValueError:
            continue

        if math.isfinite(primal):
            records.append((elapsed, primal, "主問題"))
        if math.isfinite(dual):
            records.append((elapsed, dual, "双対問題"))
    return _records_to_frame(records, "PuLP / HiGHS")


def compact_progress(progress: pl.DataFrame) -> pl.DataFrame:
    """目的関数値が変化した点と各系列の最終点だけを残す。"""
    keys = ["model", "problem"]
    return (
        progress.sort([*keys, "time"])
        .with_columns(
            (pl.col("objective") != pl.col("objective").shift().over(keys)).alias(
                "changed"
            ),
            (pl.col("time") == pl.col("time").max().over(keys)).alias("last"),
        )
        .filter(pl.col("changed").fill_null(True) | pl.col("last"))
        .drop("changed", "last")
        .sort("time")
    )


# %%
raw_progress = pl.concat(
    [
        load_highs_progress(PULP_RUN_DIR / "solver.log"),
        load_lagrangian_progress(LAGRANGIAN_RUN_DIR / "run.log"),
    ]
)
progress = compact_progress(raw_progress)
time_range = (
    0.0,
    float(raw_progress.select(pl.col("time").max()).item()) * 1.05,
)

figure = px.line(
    progress,
    x="time",
    y="objective",
    color="model",
    line_dash="problem",
    line_dash_map={"主問題": "solid", "双対問題": "dash"},
    category_orders={
        "model": ["PuLP / HiGHS", "Lagrangian"],
        "problem": ["主問題", "双対問題"],
    },
    labels={
        "time": "経過時間（秒）",
        "objective": "目的関数値",
        "model": "モデル",
        "problem": "問題",
    },
    title=(
        "主問題値・上界の推移"
        "<br><sup>実線: 最良実行可能解（下界） / 破線: 最良上界</sup>"
    ),
)
figure.update_traces(line_shape="hv")

# 凡例は色を表すモデル名だけ表示し、線種はタイトルで説明する。
for trace in figure.data:
    model, problem = trace.name.split(", ")
    trace.legendgroup = model
    trace.legendrank = 0 if model == "PuLP / HiGHS" else 1
    trace.zorder = 2 if model == "PuLP / HiGHS" else 1
    trace.name = model
    trace.showlegend = problem == "主問題"

# ほぼ同じ値になる上界同士が重なっても、PuLP系列を前面に描画する。
figure.data = tuple(
    sorted(
        figure.data,
        key=lambda trace: trace.legendgroup == "PuLP / HiGHS",
    )
)

figure.update_layout(
    height=600,
    legend_title_text="モデル",
    legend_font_size=14,
    legend_title_font_size=14,
    hovermode="x unified",
)
figure.update_xaxes(range=time_range)
figure.show()

# %%
optimality_gap = (
    raw_progress.pivot(
        on="problem",
        index=["model", "time"],
        values="objective",
        aggregate_function="last",
    )
    .rename({"主問題": "primal", "双対問題": "upper"})
    .sort(["model", "time"])
    .with_columns(
        pl.col("primal").forward_fill().over("model"),
        pl.col("upper").forward_fill().over("model"),
    )
    .drop_nulls(["primal", "upper"])
    .with_columns(
        (
            (pl.col("upper") - pl.col("primal"))
            / pl.max_horizontal(pl.lit(1.0), pl.col("primal").abs())
        ).alias("optimality_gap")
    )
    .filter(pl.col("optimality_gap") > 0)
    .with_columns(
        (
            pl.col("optimality_gap")
            != pl.col("optimality_gap").shift().over("model")
        ).alias("changed"),
        (pl.col("time") == pl.col("time").max().over("model")).alias("last"),
    )
    .filter(pl.col("changed").fill_null(True) | pl.col("last"))
    .drop("changed", "last")
    .sort("time")
)

gap_figure = px.line(
    optimality_gap,
    x="time",
    y="optimality_gap",
    color="model",
    log_y=True,
    category_orders={"model": ["PuLP / HiGHS", "Lagrangian"]},
    hover_data={
        "primal": ":,.3f",
        "upper": ":,.3f",
        "optimality_gap": ":.4%",
    },
    labels={
        "time": "経過時間（秒）",
        "optimality_gap": "相対最適性ギャップ",
        "primal": "最良実行可能値",
        "upper": "最良上界",
        "model": "モデル",
    },
    title=(
        "相対最適性ギャップの推移"
        "<br><sup>(最良上界 − 最良実行可能値) / |最良実行可能値|、対数軸</sup>"
    ),
)
gap_figure.update_traces(line_shape="hv")
for trace in gap_figure.data:
    trace.zorder = 2 if trace.name == "PuLP / HiGHS" else 1
gap_figure.update_layout(
    height=500,
    legend_title_text="モデル",
    legend_font_size=14,
    legend_title_font_size=14,
    hovermode="x unified",
)
gap_figure.update_xaxes(range=time_range)
gap_figure.update_yaxes(tickformat=".2%")
gap_figure.show()

# %%
