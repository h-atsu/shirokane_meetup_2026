# クーポン配布最適化：MIP とラグランジュ分解の比較

予算・配布枚数の上限を**超過ペナルティ付きのソフト制約**として扱います。
ユーザー間を結合する制約を緩和すると、ユーザーごとのクーポン選択に分解できます。
分解後の選択をそのまま元のソフト制約付き問題で評価するため、実行可能解への修復は行いません。

最新の [CBC・SCIP・descartes-mip 横比較](notebook/results/solver_comparison_n100000.md) も測定済みです。
CBC・SCIPでは実行可能解が得られ、HiGHSだけの結果を全ソルバーには一般化できません。

## 1. 問題設定

ユーザー集合を $I$、クーポン集合を $K$、共有リソース集合を $M$ とします。

| 記号 | 意味 |
|---|---|
| $v_{ik}$ | ユーザー $i$ にクーポン $k$ を配る場合の期待増分利益 |
| $a_{ikm}\geq 0$ | その配布で消費するリソース $m$ の量 |
| $B_m>0$ | リソース $m$ の目標上限 |
| $\rho_m>0$ | リソース $m$ の超過1単位に対するペナルティ |
| $x_{ik}\in\{0,1\}$ | クーポンを配布するかどうか |
| $s_m\geq 0$ | 上限からの超過を許す連続スラック変数 |

共有リソースは、全体の値引き原資、券種別の配布枚数、重複する属性セグメント別の原資です。
ユーザー1人につき最大1種類とする制約はハード制約として残します。配布しない選択も可能です。

## 2. 元の MIP（ソフト制約付き）

$$
\begin{aligned}
\max_{x,s}\quad
&\sum_{i\in I}\sum_{k\in K} v_{ik}x_{ik}
-\sum_{m\in M}\rho_m s_m \\
\text{s.t.}\quad
&\sum_{k\in K}x_{ik}\leq 1 &&(i\in I),\\
&\sum_{i\in I}\sum_{k\in K}a_{ikm}x_{ik}\leq B_m+s_m &&(m\in M),\\
&x_{ik}\in\{0,1\},\quad s_m\geq 0.
\end{aligned}
$$

以下では $A_m(x)=\sum_{i,k}a_{ikm}x_{ik}$、$[z]_+=\max(0,z)$ と書きます。
任意の配布 $x$ に対して、最適なスラックは

$$s_m(x)=[A_m(x)-B_m]_+$$

です。したがって、元問題の目的関数は次のようにも書けます。

$$F(x)=\sum_{i,k}v_{ik}x_{ik}-\sum_m\rho_m[A_m(x)-B_m]_+.$$

ユーザーごとの選択制約を満たす $x$ は、上記の $s(x)$ と組み合わせると必ず実行可能です。
**予算超過を無視してよいのではなく、超過分を目的関数で支払う**モデルです。
これはハード制約版とは別の最適化問題なので、両者の実験結果は混ぜません。

## 3. ラグランジュ緩和と分解

共有リソース制約 $A_m(x)-B_m-s_m\leq 0$ を緩和します。
最大化問題なので $\lambda_m\geq 0$ に対して、ラグランジアンは

$$
\begin{aligned}
\mathcal L(x,s;\lambda)
&=\sum_{i,k}v_{ik}x_{ik}-\sum_m\rho_m s_m
-\sum_m\lambda_m(A_m(x)-B_m-s_m)\\
&=\sum_m\lambda_m B_m
+\sum_{i,k}\left(v_{ik}-\sum_m\lambda_ma_{ikm}\right)x_{ik}
+\sum_m(\lambda_m-\rho_m)s_m.
\end{aligned}
$$

$s_m$ は上に非有界なので、$\lambda_m>\rho_m$ だと $\max_{s_m\geq 0}\mathcal L=+\infty$ です。
**有限の双対関数を得るため、乗数を $0\leq\lambda_m\leq\rho_m$ に制限します。**
この範囲ではスラックに関する最大値は0です（$\lambda_m=\rho_m$ の場合はどのスラックでも同じ値）。

双対関数は次の形に分解できます。

$$
U(\lambda)=\sum_m\lambda_m B_m
+\sum_{i\in I}\max\left\{0,\ \max_{k\in K}\left(v_{ik}-\sum_m\lambda_ma_{ikm}\right)\right\}.
$$

ユーザーごとに「価格調整後の利益」が最大となるクーポンを選ぶだけです。
その最大値が正でない場合は配布しません。部分問題に MIP ソルバーを使う必要はありません。
双対問題は

$$\min_{0\leq\lambda\leq\rho} U(\lambda)$$

です。

## 4. 修復なしで上界・下界を計算する

各反復の部分問題の解を $x(\lambda)$ とします。
これに $s_m=[A_m(x(\lambda))-B_m]_+$ を設定すれば、元のソフト制約付き問題でも実行可能です。
配布先やクーポンを変更する修復処理はありません。

ただし、**緩和問題の最適解が元問題でも最適になるとは限りません。**
最適性は、ペナルティを含む下界 $F(x(\lambda))$ と双対上界 $U(\lambda)$ で評価します。

$$\max_{\tau\leq t}F(x(\lambda^\tau))\ \leq\ F^*\ \leq\ \min_{\tau\leq t}U(\lambda^\tau).$$

同じ反復の解については、差は次のように書けます。

$$
U(\lambda)-F(x(\lambda))
=\sum_m\left\{\lambda_m[B_m-A_m(x(\lambda))]_+
+(\rho_m-\lambda_m)[A_m(x(\lambda))-B_m]_+\right\}\geq 0.
$$

これが0なら、その配布は元問題でも最適です。
実験では $\mathrm{gap}=(\mathrm{UB}-\mathrm{LB})/\max(1,|\mathrm{LB}|)$ を用い、
MIP と分解法に同じ目標ギャップを指定します。

このモデルのラグランジュ双対最適値は、元 MIP の LP 緩和最適値と一致します。
分解法の狙いは境界を強めることより、大きな MIP/LP を扱わずに上界・下界を更新することです。

## 5. 乗数の更新

$g_m=B_m-A_m(x(\lambda))$ は双対関数の劣勾配です。
射影付き劣勾配法で

$$\lambda_m^{t+1}=\operatorname{clip}(\lambda_m^t-\eta_tg_m^t,\ 0,\ \rho_m)$$

と更新します。超過していれば価格を上げ、余裕があれば下げます。
実装では桁を揃えるため $b_{ikm}=a_{ikm}/B_m$、$p_m=\lambda_mB_m$ と正規化し、
$0\leq p_m\leq\rho_mB_m$ に射影します。
既知の実行可能目的値を使う Polyak 型のステップを採用していますが、有限反復での最適到達は保証しません。

## 6. 実行方法

```bash
# 小規模の MIP・LP 緩和・ラグランジュ分解の比較
uv run python notebook/coupon_lagrangian.py --users 3000 --gap 0.001

# 同じインスタンスを各手法で解くスケーリング実験
uv run python notebook/coupon_scaling.py \
  --sizes 1000 10000 50000 100000 --seeds 0 1 2 \
  --coupons 5 --segments 8 --tightness 0.45 --penalty-scale 1 \
  --seconds 20 --wall-limit 60 --gap 0.001 \
  --methods LR NATIVE --lp-max-users 0

# クーポン種類数・属性数・予算の厳しさを変える条件探索
uv run python notebook/coupon_parameter_sweep.py

# 結果の図を作成（PNG / SVG）
uv run --with matplotlib python notebook/plot_coupon_scaling.py
```

`notebook/coupon_lagrangian.py` は `# %%` のセル形式で、関数をセルから呼び出せます。
`NATIVE` は同じ MIP を疎行列で直接 HiGHS に渡す対照実験、`MIP` は PuLP 経由です。
`--gap 0.001` は0.1%です。ソルバーの `Optimal` 表示も、指定ギャップ内での終了を含みます。

ペナルティの既定値は、全体原資の超過1円あたり0.8、券種別の超過1枚あたり150、
属性別原資の超過1円あたり0.15です。`--penalty-scale` はこれらを一律に乗じます。
`--tightness` は、全員にランダムな券種を配布した場合の期待消費量に対する目標上限の比率です。
データは同じ seed から再現可能な合成データであり、実際の顧客データではありません。

## 7. 時間の測り方と結果の読み方

| CSV の列 | 計測区間 |
|---|---|
| `data_s` | 全手法共通の問題データ生成 |
| `model_s` | PuLP モデルの作成。NATIVE では疎行列の作成・HiGHS への登録まで |
| `transfer_s` | PuLP モデルを HiGHS に変換・登録する時間。NATIVE は `model_s` に含む |
| `build_s` | `model_s + transfer_s` |
| `solve_s` | HiGHS の `run()`、または分解法の反復。モデル構築を含めない |
| `extract_s` | 解の取り出し・元の目的関数による評価 |
| `total_s` | 構築から解の評価まで。データ生成とプロセス起動は含めない |
| `highs_run_s` | HiGHS 自身が報告した内部求解時間 |
| `peak_rss_mb` | 問題データを含む独立プロセス全体のピーク RSS（MiB） |

分解法の `build_s` は正規化・初期化です。`solve_s` に全反復と目的値評価を含め、修復時間はありません。
各測定を独立プロセスで順に実行し、HiGHS は1スレッドに固定します。
求解時間と構築時間は別々に比較してください。
HiGHS の時間制限が実際の経過時間を超過するケースがあるため、実測時間も保存します。
外部打切りの行は到達時間として解釈せず、目標未達の打切り観測として扱います。

`notebook/results/soft_*.csv` と `notebook/results/coupon_soft_*.csv` がソフト制約版です。
それ以外の従来の CSV は修復を含むハード制約版の予備実験で、今回のモデルとの速度比較には使用しません。
ユーザー数を増やしても、結合制約数を固定すれば双対変数の数は増えません。
各反復の計算量は $O(|I||K||M|)$、現在の配列実装のメモリも $O(|I||K||M|)$ です。
反復回数や MIP との優劣はインスタンスと精度によります。

## 8. ソフト制約版の実測結果

Apple M2 / 8 GiB、HiGHS 1.15.1、1スレッドで測定。
クーポン5種類・属性8個・予算比率0.45・ペナルティ倍率1を固定し、人数だけを増やしました。
両手法の目標ギャップは0.1%、各数値は seed=0,1,2 の中央値です。
MIP は PuLP 経由の構築オーバーヘッドを除くため、同じモデルを疎行列で直接 HiGHS に渡しています。

| 人数 | バイナリ変数 | LR準備(秒) | LR求解(秒) | MIP構築(秒) | HiGHS求解(秒) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 5,000 | 0.00007 | 0.0055 | 0.0020 | 0.119 |
| 10,000 | 50,000 | 0.00087 | 0.0417 | 0.0164 | 4.168 |
| 50,000 | 250,000 | 0.00387 | 0.2377 | 0.0735 | 約60秒で外部打切り |
| 100,000 | 500,000 | 0.00844 | 0.4917 | 0.1673 | 約60秒で外部打切り |

分解法は全12ケースで目標精度に到達しました。
1万人では MIP も全 seed で解けますが、5万・10万人では全 seed が外部上限までに結果を返していません。
60秒はプロセス全体の外部上限で、実際の HiGHS 求解区間は約59.7〜59.8秒です。
ここでの打切り時間は**最適解に到達した時間ではありません**。
分解法の目的値はソフト制約の超過ペナルティ控除後であり、元のハード予算を守ることは保証しません。

![構築時間・求解時間・ギャップ・メモリの実測](notebook/results/coupon_scaling.png)

[全 seed と条件探索のレポート](notebook/results/coupon_soft_report.md)、
[生データ CSV](notebook/results/coupon_soft_scaling.csv)、
[各反復の履歴 CSV](notebook/results/coupon_soft_scaling_history.csv) を保存しています。
条件探索では券種数・属性数・予算比率・ペナルティ倍率を変えた結果も掲載します。

既定のペナルティでは、券種等の上限を約70%超過する解も出ています。
これを許すことで得られる利益とペナルティを比較する問題設定です。
ペナルティ倍率2の1万人ケースも測定し、最大超過率は約0.35%、
LR求解0.67秒・HiGHS求解4.53秒でした。
許容したい超過量に応じてペナルティを設定し、速度差と合わせて解釈してください。

## 9. 5分経っても MIP の incumbent が得られなかった実測例

10万人・クーポン5種類・属性8個・予算比率0.45・ペナルティ倍率2・seed=0を使用。
50万バイナリ変数、14個の共有リソース制約です。HiGHS 1.15.1を1スレッド、初期解なしで実行しました。
求解時間の設定は300秒、目標ギャップは両手法とも0.1%です。

| 手法 | 構築・準備(秒) | 実測求解(秒) | 結果 |
|---|---:|---:|---|
| ラグランジュ分解 | 0.0082 | 0.8684 | 目的値26,054,678.40、上界26,066,347.03、ギャップ0.0448% |
| HiGHS MIP | 0.1823 | 808.9477 | `Time limit reached`、最終 incumbent なし、incumbent 通知0件 |

分解法の解の最大超過率は約0.69%です。目的値は超過ペナルティ控除後です。
MIPは300秒以内にincumbentを返さず、実際には約809秒まで設定時間を超過してから終了しました。
最終ログも `Primal bound -inf`、`Solution status -` で、外部打切りによる欠測ではありません。
ログ上の `Nodes` と `LP iterations` はともに0です。この例は探索木が膨張したという主張ではなく、
汎用MIPの大規模な初期処理を含む、コールドスタート時の実測です。

ソフト制約版は、空配布などの実行可能解を直ちに作れます。
したがって「問題自体の実行可能性が難しい」のではなく、
**初期解を渡さない汎用ソルバーがincumbentを登録できない間に、分解法は良い解を返した**例です。

```bash
# 小規模・中規模・大規模の比較。同条件の完了済みCSVは再利用する
uv run python notebook/coupon_five_minute_demo.py

# 大規模ケースを再測定（約5分で外部打切り）
uv run python notebook/coupon_five_minute_demo.py --users 100000 --rerun

# 人数・種類数を変更
uv run python notebook/coupon_five_minute_demo.py --users 10000 100000 --coupons 10
```

再現用スクリプトは、現在はソルバーの設定時間に5秒を加えたプロセス上限で外部打切りします。
この上限はプロセス起動・データ生成・構築も含みます。
従って再測定では通常809秒まで待たず、外部打切りとして記録されます。
上記809秒の記録は、この外部上限を厳しくする前に取得した、ソルバー自身の終了結果です。

[実測CSV](notebook/results/five_minute_n100000_k5_p2.csv)、
[HiGHSログ](notebook/results/five_minute_n100000_k5_p2.highs.log)、
[incumbentイベント記録](notebook/results/five_minute_n100000_k5_p2.events.jsonl) を保存しています。

## 10. CBC・SCIP・descartes-mip との横比較

第9節と同じ10万人×5種類、ペナルティ倍率2、seed=0、各1スレッドで測定しました。
HiGHSは同条件の前回結果を再利用し、残りは今回の新規測定です。
モデル作成と求解は別計測です。

| 手法 | 構築・準備(秒) | 求解API(秒) | 得られた解・候補 |
|---|---:|---:|---|
| HiGHS 1.15.1 | 0.182 | 808.948 | 最終 incumbent なし（前回測定） |
| CBC / cbcbox 2.929（Devel、出力待ち再測定） | 7.101 | 308.778 | 目的値26,065,697.5、ギャップ約0.000058%、最終割当を検証済み |
| SCIP 10.0.2 | 3.536 | 300.514 | 初回実行可能解1.141秒、最終目的値0（空配布）。割当を検証済み |
| descartes-mip 0.2.1 | 3.738 | 303.205で外部打切り | 自動分解中、完了反復0、解未取得 |
| 専用NumPy分解 | 0.007 | 0.866 | 目的値26,054,678.4、ギャップ0.0448% |

**「どのMIPソルバーでも5分間実行可能解すら得られない」という結果ではありません。**
CBCは初回の303秒打切りでは割当を回収できませんでしたが、外部から切らずに再測定すると約309秒で最終解が返りました。
その割当から整数性・選択制約・元の目的値を独立検証し、許容ギャップ内の解が得られたことを確認しました。
SCIPは早い段階で実行可能解を持つ一方、このケースでは300秒を使っても空配布から改善しませんでした。
分解法の利点は、良い解と近い双対境界を短時間で得られる点として評価できます。

descartes-mipには明示的スラックを作らず、共有制約を `Model.relax(..., penalty=...)` でソフト化しました。
専用実装と同じく `a/B <= 1` に正規化し、ペナルティを `rho*B` とした同値モデルです。
ライブラリ内部は変更せず、`analyze()` に時間測定だけのラッパーを付けています。
v0.2.1の `_public_subproblem()` は部分問題ごとに全体の目的項・緩和制約項を再走査するため、
この「ユーザーごとに部分問題を作る」ケースでは前処理がスケールしにくい構造です。
この実測では反復前の自動分解で期限となったため、双対最適化の収束速度を測れたわけではありません。

各ソルバーの求解時間設定は300秒です。初回の横比較には別途303秒の外部期限を設けましたが、
CBCの追加再測定では外部期限を無効にし、終了と解の出力まで待っています。
SCIPは300.51秒で求解を終え、検証済み結果JSONを保存した後の終了処理中に外部期限となりました。
保存済みの結果を採用し、元のコントローラ出力と解釈用CSVは両方残しています。
CBCの求解API時間はMPS読込みを、descartes-mipの求解API時間は自動分解・部分問題準備を含みます。
詳細は [横比較レポート](notebook/results/solver_comparison_n100000.md) を参照してください。

```bash
uv run python notebook/coupon_solver_comparison.py --users 100000 --seconds 300
uv run python notebook/report_coupon_solvers.py

# CBCは出力まで待つ。前回データを残すため別ファイルへ保存
uv run python notebook/coupon_solver_comparison.py --users 100000 --methods CBC --seconds 300 --wait-for-result --output notebook/results/solver_comparison_n100000_cbc_wait.csv

# 手法を絞る／小規模で確認する
uv run python notebook/coupon_solver_comparison.py --users 1000 --seconds 30 --methods CBC SCIP DESCARTES NUMPY
```

## 11. MIPを難しくする strongly-correlated profile

従来の合成データは共有制約が14本だけで、属性別の係数も「費用×属性所属」と
券種間で強く相関していました。MIPの探索そのものを難しくする候補として、
`hard-correlated` profileを追加しました。定式化は第2節と同じで、全共有制約を
ソフト化しているため、ラグランジュ解の修復は不要です。

このprofileでは、全体原資に加えて多数の疎な重複campaign予算を作ります。
各ユーザー・券種の選択肢は平均8個のcampaign予算を消費し、利益を資源消費量と
強く相関させます。これにより、明らかに支配される選択肢を減らし、似た価値を持つ
多数の組合せから共有予算間のトレードオフを選ばせます。一方、ラグランジュ緩和後は
従来どおり各ユーザーが価格調整後利益の最大券種を選ぶだけです。

1万人・10券種・64 campaign予算・seed=0・ペナルティ倍率1を1スレッドで測定した
初期結果は次のとおりです。目標相対ギャップは0.1%、各汎用ソルバーの設定時間は
30秒で、終了後の解出力までは外部から打ち切っていません。

| 手法 | 構築・準備(秒) | 求解API(秒) | 実行可能目的値 | 上界またはギャップ |
|---|---:|---:|---:|---:|
| 専用NumPy分解 | 0.005 | 1.657 | 4,130,570.10 | 上界4,134,327.57、ギャップ0.0910% |
| CBC / cbcbox 2.929 | 2.198 | 38.887 | 4,132,852.78 | ギャップ0.0355% |
| SCIP 10.0.2 | 1.056 | 30.106 | 3,606,077.02 | root LP未完了、ギャップ2,373.87% |

CBCはこの規模では最も良い整数解を得ていますが、分解法も目標ギャップを約1.7秒で
満たし、目的値差はCBC比約0.055%です。SCIPはroot LPが時間内に完了せず、分解法より
約52万低い解でした。またHiGHSの10秒設定による別測定では、求解APIが約37.8秒まで
かかり、目的値792に対し、分解法は1.61秒で4,130,570.10を得ました。

これは「すべてのMIPソルバーが実行可能解を得られない」例ではなく、汎用MIPより
専用分解が短時間で高品質な解と境界を得る例です。より大規模なケースでCBCにも
限界が来るかは、実測してから判断します。

```bash
# HiGHS MIP・LP・専用分解の短時間スクリーニング
uv run python notebook/coupon_lagrangian.py \
  --users 10000 --coupons 10 --segments 64 --profile hard-correlated \
  --seconds 10 --iterations 5000 --gap 0.001

# CBC・SCIP・専用分解。内部制限後の解出力まで待つ
uv run python notebook/coupon_solver_comparison.py \
  --users 10000 --coupons 10 --segments 64 --profile hard-correlated \
  --penalty-scale 1 --seconds 30 --gap 0.001 \
  --methods CBC SCIP NUMPY --wait-for-result \
  --output notebook/results/hard_correlated_solvers_n10000.csv
```

[横比較CSV](notebook/results/hard_correlated_solvers_n10000.csv)、
[2,000人スクリーニング](notebook/results/hard_correlated_screen_n2000.csv)、
[10,000人スクリーニング](notebook/results/hard_correlated_screen_n10000.csv)を保存しています。
