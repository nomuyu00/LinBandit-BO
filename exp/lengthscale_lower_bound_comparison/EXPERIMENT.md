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

注意:
- 本比較は「LinUCB 方向選択 + EI による1次元最適化」を共通とし、GPの入力正規化とカーネル制約のみを切り替えます。
- 再現性のため `torch.manual_seed` / `np.random.seed` を各ランで固定しています。

## 実験結果（要約）

- 設定: `dim=20`, `n_initial=5`, `n_max=300`, `coordinate_ratio=0.8`, `n_arms=dim/2`, ラン数 `n_runs=20`。
- テスト関数ごとの傾向（出力: `output_results_lengthscale_lower_bound_comparison/*` を参照）
  - Styblinski‑Tang: Floor（長さスケール下限あり）の方が良好に収束。箱ひげでもFloorの最終値が優位。
  - Rastrigin: Baselineの方が良好。Floorは平坦化の影響で探索が鈍く、収束が遅い。
  - Ackley: Rastriginと同様にBaseline優位。

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

- 条件A/Bの L_hat の時系列・最終値比較（Floorの方が小さいはず）
- 正規化前の |∇μ| 分布（スパイク頻度と最大値の比較）
- Floor の ARD 長さスケール分布（`lengthscale_history`）の可視化

## 公平性・安定性の改善案

- 共通基準での正規化: 条件間で同一の L_ref を用意して r = |∇μ| / max(L_ref, L_min) に統一（分母差による見かけの差を排除）
- l_min のチューニング: Rastrigin/Ackley向けに `l_min=0.01` 等を検討（微細構造を許容）
- 条件の切り分け: 両条件とも入力正規化を実施し、差分を「l_min の有無」だけに限定
- カーネル選択: 周期性には Periodic/Spectral Mixture 等のカーネルを検討（本比較はRBF固定）
