# Lengthscale Lower Bound Comparison

目的: LinBandit-BO の現行アルゴリズム（勾配ベース報酬, L下限付きスケーリング）に対して、RBFカーネルの長さスケールに下限 `l_min` を課す条件（入力正規化前提）を比較します。形式・実行フロー・出力は `exp/cosine_similarity_comparison/` と同一です（アルゴリズム部分のみ差し替え）。

- 条件A（Baseline）: GP入力を正規化せず、長さスケール下限なし。
- 条件B（Floor）: 入力Xを `[0,1]^d` に正規化し、`RBFKernel(..., lengthscale_constraint=GreaterThan(l_min))` を適用（単一モデル）。
- 既定の `l_min=0.05`（正規化空間での比率）を使用可能。

再現手順:
- ノートブック: `lengthscale_lower_bound_comparison_experiment.ipynb` を上から実行。
- 出力先: `output_results_lengthscale_lower_bound_comparison/`
  - `*_comparison.png`: 収束履歴の比較（Baseline vs Floor）
  - `*_reward_analysis.png`: 報酬・指標（長さスケール等）の分析可視化
  - `*_reward_history.csv`: 報酬履歴CSV（全ラン、各次元）
  - `*_dimension_summary.csv`: 次元別要約CSV（平均/標準偏差/累積）
  - `*_results.npy`: 反復ごとの最良事後平均の履歴（複数ラン）
  - `*_diagnostics.png`: 追加診断（L_hat推移・最終値、|∇μ|分布、r≈1到達率）
  - `*_lhat_history.csv`: L_hat の時系列（条件・ラン別）
  - `*_grad_norms.csv`: 正規化前の勾配ノルム ∥∇μ∥ の時系列（条件・ラン別）
  - `*_r_upper_hit_rate.csv`: 報酬の上限到達率（r≥0.95）の時系列（条件・ラン別）

注意:
- 本比較は「LinUCB 方向選択 + EI による1次元最適化」を共通とし、GPの入力正規化とカーネル制約のみを切り替えます。
- 再現性のため `torch.manual_seed` / `np.random.seed` を各ランで固定しています。

## 実験結果（要約）

- 設定: `dim=20`, `n_initial=5`, `n_max=300`, `coordinate_ratio=0.8`, `n_arms=dim/2`, ラン数 `n_runs=20`。

### 画像とファイルの場所
- 収束比較: `output_results_lengthscale_lower_bound_comparison/[Function]_comparison.png`
- 報酬・長さスケール分析: `output_results_lengthscale_lower_bound_comparison/[Function]_reward_analysis.png`
- 追加診断（本タスクで追加）: `output_results_lengthscale_lower_bound_comparison/[Function]_diagnostics.png`
  - L_hat推移（平均±STD）/最終L_hat箱ひげ
  - 正規化前の勾配ノルム分布（|∇μ|）
  - 報酬の上限到達率（r≥0.95）の推移
- CSV: 報酬履歴 `*_reward_history.csv` / 次元別要約 `*_dimension_summary.csv` / L_hat `*_lhat_history.csv` / 勾配ノルム `*_grad_norms.csv` / 上限到達率 `*_r_upper_hit_rate.csv`

### テスト関数別の結果まとめ

1) Styblinski‑Tang（滑らか）
- 収束: Floor（長さスケール下限あり）が一貫して良好（`*_comparison.png`）。最終値の箱ひげでも優位。
- L_hat: Floorの方が小さく安定（`*_diagnostics.png` 上段左・右）。
- |∇μ|分布: Baselineは右尾が重くスパイク多め、Floorはスパイク抑制（下段左）。
- r≈1到達率: Floorの方が高め（下段右）。これは分母のL_hatが過大化しづらい帰結で、見かけの差である点に注意。

2) Rastrigin（高周波・周期）
- 収束: Baselineが優位（`*_comparison.png`）。Floorは平滑化により細かい構造を捉えにくく、収束が鈍化。
- L_hat: Floorは小さく安定。Baselineはスパイクで履歴最大が大きくなりやすい。
- |∇μ|分布: Baselineはスパイク頻度・最大値が大きい。Floorは分布が集中。
- r≈1到達率: Floorが高めだが、これはL_hatの履歴効果によるもの。性能優位とは直結しない。

3) Ackley（高周波・多峰）
- 収束: Baselineが優位（`*_comparison.png`）。挙動はRastriginと同傾向。
- L_hat/|∇μ|/r≈1: Rastriginと同様の傾向。Floorは安定・平滑、Baselineはスパイク・高周波検出力が相対的に高い。

参考ファイル
- 収束比較: `*_comparison.png`
- 結果配列: `*_results.npy`（各ランの `best_value`・履歴を格納）
- 報酬・長さスケール分析: `*_reward_analysis.png`
- 報酬履歴/次元別要約: `*_reward_history.csv`, `*_dimension_summary.csv`

## なぜ「Floorの方が報酬平均が大きく」見えるのか

- 報酬設計は r = |∇μ(x)| / max(L_hat, L_min)（各次元）で、L_hat は「そのランで観測した勾配ノルムの最大値」を増加のみで更新します。
- Floor（入力正規化＋長さスケール下限）を入れると、極端なスパイクが起きにくくなり、結果として L_hat 自体が小さくなりがちです。
- 分母 max(L_hat, L_min) が小さくなるため、同程度の“生の勾配”でも r は相対的に大きくなります。
- よって「スパイク抑制」という目的と、「r の平均が増える」という観測結果は両立します（見かけの差であり、実勾配の暴れを抑えつつ、正規化の分母が小さくなった副作用）。

直観例
- Baseline: L_hat ≈ 10、平均|∇μ| ≈ 1 → 平均 r ≈ 0.1
- Floor:    L_hat ≈  2、平均|∇μ| ≈ 0.8 → 平均 r ≈ 0.4（分母が小さい分だけ大きく見える）

## 関数別の性能差の考察

- Styblinski‑Tang: 比較的滑らか。過小な長さスケール（過度適合）を下限で抑える効果が支配的で、EI と方向選択が安定化 → Floor が有利。
- Rastrigin/Ackley: 周期性・高周波成分が強い。小さな長さスケールを禁じると局所構造を捉えにくく、事後平均が平坦化 → EI/方向選択が鈍化し、Baseline が有利になりやすい。

## 追加診断（推奨）

- 条件A/Bの L_hat の時系列・最終値比較（Floorの方が小さいはず）→ `*_lhat_history.csv`, `*_diagnostics.png`
- 正規化前の |∇μ| 分布（スパイク頻度と最大値の比較）→ `*_grad_norms.csv`, `*_diagnostics.png`
- 報酬の上限到達率（r≈1 の頻度）→ `*_r_upper_hit_rate.csv`, `*_diagnostics.png`
- Floor の ARD 長さスケール分布（`lengthscale_history`）の可視化は `*_reward_analysis.png` に含む

## 公平性・安定性の改善案

- 共通基準での正規化: 条件間で同一の L_ref を用意して r = |∇μ| / max(L_ref, L_min) に統一（分母差による見かけの差を排除）
- l_min のチューニング: Rastrigin/Ackley向けに `l_min=0.01` 等を検討（微細構造を許容）
- 条件の切り分け: 両条件とも入力正規化を実施し、差分を「l_min の有無」だけに限定
- カーネル選択: 周期性には Periodic/Spectral Mixture 等のカーネルを検討（本比較はRBF固定）

## 実務的示唆（まとめ）
- 滑らかな関数（例: Styblinski‑Tang）では、Floor（正規化＋l下限）により過小長さスケールを抑え、収束が安定・高速化。
- 高周波・周期的関数（Rastrigin/Ackley）では、Baselineが有利になりやすい。Floorを使う場合は l_min を小さめに再調整、あるいは周期カーネルの検討が有効。
- r≈1の比較は分母の定義（L_hat履歴）に依存するため、性能の直接指標としては注意が必要。共通L_refでの再評価が公平。
