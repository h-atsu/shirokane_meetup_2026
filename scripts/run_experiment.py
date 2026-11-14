"""コンテンツ配信最適化を実行する。"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
from omegaconf.errors import OmegaConfBaseException

from shirokane_meetup_2026.consts import ROOT
from shirokane_meetup_2026.domain.config import Config
from shirokane_meetup_2026.domain.metric import Metrics
from shirokane_meetup_2026.evaluator import SolutionEvaluator
from shirokane_meetup_2026.loader import InstanceLoader
from shirokane_meetup_2026.model_factory import create_model
from shirokane_meetup_2026.writer import ResultWriter

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _create_output_dir(experiment_name: str, run_name: str) -> Path:
    output_dir = ROOT / "outputs" / experiment_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@app.command()
def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="設定YAMLのファイルパス",
        ),
    ] = Path("configs/default.yaml"),
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="設定をkey=value形式で上書き（複数指定可）",
        ),
    ] = None,
    register_mlflow: Annotated[
        bool,
        typer.Option(
            "--register-mlflow/--no-register-mlflow",
            help="実行結果をMLflowへ登録するか",
        ),
    ] = True,
) -> None:
    """YAML設定とCLI上書きを反映してコンテンツ配信最適化を実行する。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("最適化設定を読み込みます config=%s", config_path)

    try:
        config = Config.from_yaml(config_path, overrides or ())
    except OmegaConfBaseException as error:
        raise typer.BadParameter(
            f"設定を読み込めません: {error}",
            param_hint="--set",
        ) from error
    config.run_name = config.run_name or datetime.now(UTC).astimezone().strftime(
        "%Y%m%d_%H%M%S"
    )
    output_dir = _create_output_dir(config.experiment_name, config.run_name)

    instance_dir = ROOT / "data" / config.instance_name
    loader = InstanceLoader(instance_dir)
    writer = ResultWriter(output_dir)
    writer.write_config(config)

    logger.info("入力データを読み込みます instance=%s", config.instance_name)
    input_data = loader.load_input_data()
    logger.info(
        "入力データを読み込みました users=%d contents=%d",
        input_data.score.shape[0],
        input_data.score.shape[1],
    )

    logger.info("最適化モデルを構築します model=%s", config.model_name)
    model = create_model(input_data, config)
    build_started_at = perf_counter()
    model.build_model()
    build_time = perf_counter() - build_started_at
    logger.info(
        "最適化モデルを構築しました model=%s time=%.3f",
        type(model).__name__,
        build_time,
    )

    logger.info("最適化モデルを求解します")
    solve_started_at = perf_counter()
    solve_stats = model.solve()
    solve_time = perf_counter() - solve_started_at
    solve_stats.build_time = build_time
    solve_stats.solve_time = solve_time
    logger.info(
        "最適化モデルを求解しました status=%s time=%.3f",
        solve_stats.status,
        solve_stats.solve_time,
    )
    writer.write_solve_stats(solve_stats)

    metrics: Metrics | None = None
    if solve_stats.has_solution:
        solution = model.get_solution()
        metrics = SolutionEvaluator(input_data, config, solution).evaluate()
        writer.write_solution(
            loader=loader,
            input_data=input_data,
            solution=solution,
            metrics=metrics,
        )
        logger.info("実行結果を出力しました output_dir=%s", output_dir)
    else:
        logger.warning("実行可能解が得られなかったため結果評価をスキップします")

    if register_mlflow:
        from shirokane_meetup_2026.register import MLflowRegister

        run_id = MLflowRegister().register(
            config=config,
            solve_stats=solve_stats,
            metrics=metrics,
            artifact_dir=output_dir,
        )
        logger.info("実行結果をMLflowへ登録しました run_id=%s", run_id)


if __name__ == "__main__":
    app()
