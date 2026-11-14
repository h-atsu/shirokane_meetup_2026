# shirokane-meetup-2026

コンテンツ配信最適化を題材に、数理最適化モデルの実装とソルバーの比較を行うためのプロジェクトです。

問題設定と数理モデルの定式化は [docs/index.md](docs/index.md) を参照してください。

## 必要な環境

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

依存パッケージをインストールします。

```bash
uv sync
```

## ディレクトリ構成

```text
.
├── configs/                    # 実験設定
│   └── default.yaml
├── data/                       # 生成した入力データ（Git管理外）
├── docs/                       # 定式化などのドキュメント
├── notebook/                   # ベンチマーク・検証用スクリプト
├── outputs/                    # 実験結果（Git管理外）
├── scripts/
│   ├── generate_data.py        # 入力データ生成
│   ├── run_experiment.py       # 実験の実行
│   └── compare_pulp_solvers.sh # PuLPソルバーの一括比較
├── src/shirokane_meetup_2026/
│   ├── domain/                 # 設定、入出力、評価指標の定義
│   ├── models/                 # PuLP・ラグランジュ分解モデル
│   ├── evaluator.py            # 解の評価
│   ├── loader.py               # 入力データの読み込み
│   ├── register.py             # MLflowへの登録
│   └── writer.py               # 実験結果の保存
└── tests/                      # テスト
```

## 実行方法

### 1. 入力データを生成する

デフォルト設定 `configs/default.yaml` が参照する小規模データを生成します。

```bash
uv run python scripts/generate_data.py users_100_contents_30_seed_0 \
  --users 100 \
  --contents 30 \
  --seed 0
```

生成物は `data/users_100_contents_30_seed_0/` に保存されます。

```text
data/users_100_contents_30_seed_0/
├── users.parquet
├── contents.parquet
├── user_contents.parquet
└── metadata.json
```

引数を省略すると、50,000ユーザー・100コンテンツ・乱数シード0のデータを生成します。

```bash
uv run python scripts/generate_data.py
```

利用可能なオプションは次のコマンドで確認できます。

```bash
uv run python scripts/generate_data.py --help
```

### 2. 実験を実行する

`configs/default.yaml` の設定で実験を実行します。

```bash
uv run python scripts/run_experiment.py
```

別の設定ファイルを使う場合は `--config`（短縮形は `-c`）で指定します。

```bash
uv run python scripts/run_experiment.py --config path/to/config.yaml
```

設定を一時的に変更する場合は、`--set key=value` を複数指定できます。ネストした設定キーはドット区切りで指定します。

```bash
uv run python scripts/run_experiment.py \
  --set pulp.solver=cbc \
  --set time_limit=120 \
  --set experiment_name=pulp_solver_comparison \
  --set run_name=cbc
```

未知の設定キーや、設定された型に変換できない値はエラーになります。

MLflowへ登録せず、ファイル出力だけを行う場合は `--no-register-mlflow` を指定します。

```bash
uv run python scripts/run_experiment.py --no-register-mlflow
```

### 3. PuLPソルバーを比較する

CBC、HiGHS、SCIPを同じ条件で順番に実行します。比較スクリプトは `users_10000_contents_100_seed_0` を使用するため、先に対象データを生成してください。

```bash
uv run python scripts/generate_data.py users_10000_contents_100_seed_0 \
  --users 10000 \
  --contents 100 \
  --seed 0

scripts/compare_pulp_solvers.sh
```

各実行は共通のMLflow experiment `pulp_solver_comparison` に登録され、run名には実行時刻とソルバー名が付与されます。

## 主な設定項目

設定は [configs/default.yaml](configs/default.yaml) で管理します。

| キー | 内容 |
|---|---|
| `instance_name` | `data/` 以下にある入力データのディレクトリ名 |
| `experiment_name` | 出力先とMLflowで使用する実験名 |
| `run_name` | 実行名。`null` の場合は実行日時から自動生成 |
| `model_name` | 使用するモデル。`pulp` または `lagrangian` |
| `penalty` | 最低配信回数の不足に対するペナルティ |
| `time_limit` | 求解時間の上限（秒） |
| `relative_gap` | 許容する相対ギャップ |
| `threads` | ソルバーのスレッド数 |
| `pulp.solver` | PuLPで使用するソルバー |
| `lagrangian.max_iterations` | ラグランジュ分解の最大反復回数 |
| `lagrangian.initial_step_size` | ラグランジュ分解の初期ステップ幅 |

## 実行結果

結果は `outputs/<experiment_name>/<run_name>/` に保存されます。

```text
outputs/<experiment_name>/<run_name>/
├── config.yaml                # 実行時の設定
├── solve_stats.json           # 求解ステータスと実行時間
├── metrics.json               # 解の評価指標
├── df_result.parquet          # ユーザーごとの配信結果
└── df_content_result.parquet  # コンテンツごとの配信回数と不足量
```

解が得られなかった場合、`metrics.json` と2つのParquetファイルは作成されません。

デフォルトでは実行結果をローカルのMLflowにも登録します。MLflow UIは次のコマンドで起動できます。

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## テストと静的解析

```bash
uv run pytest
uv run ruff check .
uv run ty check
```

## ドキュメント

ローカルでドキュメントを確認するには、Zensicalの開発サーバーを起動します。

```bash
uv run zensical serve
```

静的サイトを生成する場合は次を実行します。生成先の `site/` はGit管理外です。

```bash
uv run zensical build --clean
```
