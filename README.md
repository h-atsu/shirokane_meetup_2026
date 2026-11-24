# shirokane-meetup-2026

このリポジトリは、[白金鉱業 Meetup Vol.26@六本木（数理最適化編）](https://brainpad-meetup.connpass.com/event/405702/) の発表「困難は分割せよ！ ～大規模問題のためのラグランジュ分解入門～」で使用した実験コードです。

コンテンツ配信最適化問題に対して、汎用MIPソルバー、NumPy実装のラグランジュ分解、自動ラグランジュ分解、貪欲法を比較します。問題設定と定式化は [docs/index.md](docs/index.md) を参照してください。

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
├── notebook/
│   ├── small_experiment_result.py # 小規模実験の可視化
│   └── large_experiment_result.py # 大規模実験の可視化
├── outputs/                    # 実験結果（Git管理外）
├── scripts/
│   ├── generate_data.py        # 入力データ生成
│   ├── prepare_data.sh         # 比較用入力データの一括生成
│   ├── run_experiment.py       # 実験の実行
│   ├── run_small_experiment.sh # 小規模実験の一括実行
│   └── run_large_experiment.sh # 大規模実験の一括実行
├── src/shirokane_meetup_2026/
│   ├── domain/                 # 設定、入出力、評価指標の定義
│   ├── models/                 # 各最適化モデル
│   ├── evaluator.py            # 解の評価
│   ├── loader.py               # 入力データの読み込み
│   ├── register.py             # MLflowへの登録
│   └── writer.py               # 実験結果の保存
└── tests/                      # テスト
```

## クイックスタート

依存関係と再現用データを準備し、小規模実験を実行します。

```bash
uv sync
scripts/prepare_data.sh
scripts/run_small_experiment.sh
```

結果は `outputs/small_experiment/` とローカルのMLflowに保存されます。

## 入力データ

### 再現用データを生成する

比較に使用する500ユーザー・100コンテンツと30,000ユーザー・100コンテンツのデータをまとめて生成します。

```bash
scripts/prepare_data.sh
```

乱数シードは両方とも `0` に固定されています。

| 用途 | ユーザー数 | コンテンツ数 | 保存先 |
|---|---:|---:|---|
| デフォルト・小規模実験 | 500 | 100 | `data/users_500_contents_100_seed_0/` |
| 大規模実験 | 30,000 | 100 | `data/users_30000_contents_100_seed_0/` |

各インスタンスには次のファイルが作成されます。

```text
data/<instance_name>/
├── users.parquet
├── contents.parquet
├── user_contents.parquet
└── metadata.json
```

### 任意の条件で生成する

個別の条件で生成する場合は `generate_data.py` を実行します。

```bash
uv run python scripts/generate_data.py users_100_contents_30_seed_0 \
  --users 100 \
  --contents 30 \
  --seed 0
```

引数を省略すると、50,000ユーザー・100コンテンツ・乱数シード0のデータを生成します。

```bash
uv run python scripts/generate_data.py
```

利用可能なオプションは次のコマンドで確認できます。

```bash
uv run python scripts/generate_data.py --help
```

## 実験を実行する

### デフォルト設定

`configs/default.yaml` は500ユーザー・100コンテンツ、ラグランジュ分解、ペナルティ係数10、TimeLimit 60秒、相対Gap 0.01%を指定しています。

```bash
uv run python scripts/run_experiment.py
```

### 小規模実験

500ユーザー・100コンテンツに対して、次の9 runを順番に実行します。

- 貪欲法（評価用ペナルティ係数10）
- PuLP/HiGHS（ペナルティ係数0、10、100、1,000）
- ラグランジュ分解（ペナルティ係数0、10、100、1,000）

PuLP/HiGHSとラグランジュ分解のTimeLimitは60秒、相対Gap許容値は0.01%です。ラグランジュ分解の最大反復数は十分大きな値に設定し、通常はGapまたはTimeLimitで終了させます。

```bash
scripts/run_small_experiment.sh
```

結果はMLflow experiment `small_experiment` と `outputs/small_experiment/` に保存されます。

### 大規模実験

30,000ユーザー・100コンテンツ、ペナルティ係数100の同一条件で、PuLP/HiGHSとラグランジュ分解を順番に実行します。各モデルのTimeLimitは3,600秒、相対Gap許容値は0.01%です。

```bash
scripts/run_large_experiment.sh
```

結果はMLflow experiment `large_experiment` と `outputs/large_experiment/` に保存されます。

### 任意の設定

利用可能なモデルは `pulp`、`lagrangian`、`descartes_mip`、`greedy` です。選択肢の正は [`ModelName`](src/shirokane_meetup_2026/domain/config.py) で管理しています。

```bash
uv run python scripts/run_experiment.py \
  --set model_name=pulp \
  --set pulp.solver=highs

uv run python scripts/run_experiment.py \
  --set model_name=lagrangian

uv run python scripts/run_experiment.py \
  --set model_name=descartes_mip

uv run python scripts/run_experiment.py \
  --set model_name=greedy
```

`greedy` は各ユーザーについて `score / weight` の降順に、残り容量へ収まる正のスコアのコンテンツを選択します。最低配信回数と不足ペナルティは選択には使用せず、選択後の評価指標に反映します。

別の設定ファイルを使う場合は `--config`（短縮形は `-c`）で指定します。

```bash
uv run python scripts/run_experiment.py --config path/to/config.yaml
```

設定を一時的に変更する場合は、`--set key=value` を複数指定できます。ネストした設定キーはドット区切りで指定します。

```bash
uv run python scripts/run_experiment.py \
  --set model_name=pulp \
  --set pulp.solver=cbc \
  --set time_limit=120 \
  --set experiment_name=custom_experiment \
  --set run_name=cbc
```

MLflowへ登録せず、ファイル出力だけを行う場合は `--no-register-mlflow` を指定します。

```bash
uv run python scripts/run_experiment.py --no-register-mlflow
```

## 主な設定項目

設定値は [configs/default.yaml](configs/default.yaml)、型と利用可能な値は [config.py](src/shirokane_meetup_2026/domain/config.py) で管理しています。`--set` に未知のキーや不正な型を指定するとエラーになります。

| キー | 内容 |
|---|---|
| `instance_name` | `data/` 以下にある入力データのディレクトリ名 |
| `experiment_name` | 出力先とMLflowで使用する実験名 |
| `run_name` | 実行名。`null` の場合は実行日時から自動生成 |
| `model_name` | 使用するモデル。`pulp`、`lagrangian`、`descartes_mip`、`greedy` |
| `penalty` | 最低配信回数の不足に対するペナルティ |
| `time_limit` | 求解時間の上限（秒） |
| `relative_gap_tol` | 求解を終了する相対Gap許容値。`0.0001` は0.01% |
| `threads` | ソルバーのスレッド数 |


## 実行結果

結果は `outputs/<experiment_name>/<run_name>/` に保存されます。

```text
outputs/<experiment_name>/<run_name>/
├── run.log                    # アプリケーションとPythonモデルのログ
├── solver.log                 # PuLPソルバーのネイティブログ（PuLPのみ）
├── config.yaml                # 実行時の設定
├── solve_stats.json           # 求解ステータスと実行時間
├── metrics.json               # 解の評価指標
├── df_result.parquet          # ユーザーごとの配信結果
└── df_content_result.parquet  # コンテンツごとの配信回数と不足量
```

解が得られなかった場合、`metrics.json` と2つのParquetファイルは作成されません。

ラグランジュ分解の各反復における主問題値・上界・相対Gapは `run.log` に記録されます。PuLP実行時はHiGHSなどの進捗表が `solver.log` に記録されます。

## 結果を可視化する

小規模実験では、各設定の最新runを読み込み、コンテンツ別配信回数を比較します。

```bash
uv run python notebook/small_experiment_result.py
```

大規模実験では、指定したPuLP/HiGHSとラグランジュ分解のログから、主問題値・上界・相対最適性Gapの推移を比較します。対象runは `notebook/large_experiment_result.py` 冒頭の `PULP_RUN_DIR` と `LAGRANGIAN_RUN_DIR` で指定します。

```bash
uv run python notebook/large_experiment_result.py
```

## MLflow

デフォルトでは実行結果をローカルのMLflowにも登録します。MLflow UIは次のコマンドで起動できます。

```bash
uv run mlflow ui 
```

## テストと静的解析

```bash
uv run pytest
uv run ruff check .
uv run ty check \
  src scripts tests \
  notebook/small_experiment_result.py \
  notebook/large_experiment_result.py
```

`notebook/old/` は過去の検証コードであり、現在の型チェック対象から除外しています。

## ドキュメント

ローカルでドキュメントを確認するには、Zensicalの開発サーバーを起動します。

```bash
uv run zensical serve
```

静的サイトを生成する場合は次を実行します。生成先の `site/` はGit管理外です。

```bash
uv run zensical build --clean
```
