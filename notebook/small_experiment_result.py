# %%
from pathlib import Path

import plotly.express as px
import polars as pl

# 小規模実験の結果可視化

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "outputs" / "small_experiment"

RUNS = {
    "Greedy": "_greedy",
    "PuLP / HiGHS (rho=0)": "_pulp_highs_rho_0",
    "PuLP / HiGHS (rho=10)": "_pulp_highs_rho_10",
    "PuLP / HiGHS (rho=100)": "_pulp_highs_rho_100",
}


def load_latest_result(run_suffix: str) -> tuple[Path, pl.DataFrame]:
    """指定したsuffixに一致する最新runのコンテンツ別結果を読み込む。"""
    candidates = sorted(
        path
        for path in EXPERIMENT_DIR.iterdir()
        if path.is_dir()
        and path.name.endswith(run_suffix)
        and (path / "df_content_result.parquet").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed run ending with {run_suffix!r} in {EXPERIMENT_DIR}"
        )

    run_dir = candidates[-1]
    result = (
        pl.read_parquet(run_dir / "df_content_result.parquet")
        .sort(
            ["delivery_count", "content_id"],
            descending=[True, False],
        )
        .with_row_index("rank", offset=1)
    )
    return run_dir, result


def common_y_range(results: list[pl.DataFrame]) -> tuple[float, float]:
    """すべてのプロットで共有するY軸範囲を返す。"""
    maximum = max(
        float(
            result.select(
                pl.max_horizontal(
                    "delivery_count",
                    "minimum_delivery_count",
                ).max()
            ).item()
        )
        for result in results
    )
    return 0.0, float(maximum) * 1.05


def make_figure(
    title: str,
    run_dir: Path,
    result: pl.DataFrame,
    y_range: tuple[float, float],
):
    """配信回数を降順に並べ、目標配信回数を点線で重ねる。"""
    targets = result.get_column("minimum_delivery_count").unique()
    if len(targets) != 1:
        raise ValueError(
            "minimum_delivery_count must be identical for all contents "
            "to draw a horizontal target line"
        )
    target = int(targets.item())
    figure = px.bar(
        result,
        x="rank",
        y="delivery_count",
        hover_data={"content_id": True, "rank": False},
        labels={
            "rank": "順位",
            "content_id": "コンテンツID",
            "delivery_count": "配信回数",
        },
        title=f"{title}<br><sup>{run_dir.name}</sup>",
    )
    figure.add_hline(
        y=target,
        line_dash="dash",
        line_color="red",
        annotation_text=f"目標配信回数: {target}",
        annotation_position="top right",
    )
    figure.update_layout(
        height=500,
        showlegend=False,
        bargap=0.08,
    )
    figure.update_xaxes(dtick=10)
    figure.update_yaxes(range=y_range)
    return figure


# %%
loaded_results = [
    (title, *load_latest_result(run_suffix))
    for title, run_suffix in RUNS.items()
]
y_range = common_y_range([result for _, _, result in loaded_results])

for title, run_dir, result in loaded_results:
    make_figure(title, run_dir, result, y_range).show()


# %%
