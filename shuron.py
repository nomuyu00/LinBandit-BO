# -*- coding: utf-8 -*-  # 文字コード指定（日本語コメントのため）
import math  # 数学関数（sqrt, log など）
import numpy as np  # 乱数や配列の補助（主に初期化用）
import torch  # テンソル演算の基盤
from botorch.models import SingleTaskGP  # 単一タスクのガウス過程モデル
from botorch import fit_gpytorch_model  # GPyTorch モデルの学習ユーティリティ
from botorch.acquisition import ExpectedImprovement  # 取得関数 EI
from gpytorch.mlls import ExactMarginalLogLikelihood  # 厳密周辺尤度（GP 学習用）
from gpytorch.kernels import RBFKernel, ScaleKernel  # RBF(ARD) + スケールカーネル
from gpytorch.constraints import GreaterThan  # （必要なら）ハイパーパラメータ下限
import warnings  # 警告抑制
warnings.filterwarnings("ignore")  # 学習時の警告を見やすさのため抑制

torch.set_default_dtype(torch.double)  # BoTorch/GPyTorch は double が安定

# ===== ユーティリティ（指数移動平均：NLPD のスケール正規化に使用） =====
class EMA:  # EMA = Exponential Moving Average の簡単な実装
    def __init__(self, alpha: float = 0.1, eps: float = 1e-8):  # 平滑係数とε
        self.alpha = float(alpha)  # 新しい値にかける重み
        self.eps = float(eps)  # ゼロ割防止用の小さな値
        self._m = None  # 内部状態（平均）の初期値

    def update(self, x: float) -> float:  # 新しい観測 x で平均を更新
        if self._m is None:  # 初回のみそのまま代入
            self._m = float(x)  # 初期化
        else:  # 2 回目以降は指数移動平均
            self._m = self.alpha * float(x) + (1.0 - self.alpha) * self._m  # EMA 更新
        return float(self._m)  # 現在の平均値を返す

    @property
    def value(self) -> float:  # 現在の平均値（未初期化なら eps）
        return float(self._m if self._m is not None else self.eps)  # 安定化して返す


# ===== 本体：超シンプル LinBandit-BO（Continuous-Fixed + EI + NLPD*dir） =====
class SimpleLinBanditBONLPDDir:  # 読みやすさ重視の最小クラス
    def __init__(self, objective_function, bounds, n_initial=5, n_max=100, l_min=0.0):  # 主要パラメータ
        self.objective_function = objective_function  # 最小化したい目的関数 f(x)
        self.bounds = bounds.detach().clone().double()  # 探索範囲 [2, d]（下限・上限）
        self.dim = self.bounds.shape[1]  # 次元 d
        self.n_initial = int(n_initial)  # 初期サンプル数
        self.n_max = int(n_max)  # 総評価回数（初期点を含む）
        self.coord_arms = torch.eye(self.dim, dtype=self.bounds.dtype)  # アーム＝各次元の単位ベクトル
        self.A = torch.eye(self.dim, dtype=self.bounds.dtype)  # LinUCB の A 行列（初期は単位行列）
        self.b = torch.zeros(self.dim, dtype=self.bounds.dtype)  # LinUCB の b ベクトル（初期ゼロ）
        self._ema_nlpd = EMA(alpha=0.1)  # NLPD の EMA 正規化器
        self.model = None  # GP モデル（あとで初期化）
        self.X = None  # 既存の入力履歴（N×d）
        self.Y = None  # 既存の出力履歴（N×1）
        self.best_x = None  # これまでの最良点（観測上あるいは事後平均上の指標）
        self.best_y = None  # その値
        self.total_iterations = 0  # 反復カウンタ
        self.l_min = float(l_min)  # RBF の lengthscale 下限（0で無効）

    # ---- LinUCB の β_t（簡易式）：連続方向の信頼半径に使う ----
    def _beta_t(self) -> float:  # 解析的厳密式ではなく実務的に安定な簡易式
        sigma = 1.0  # ノイズの上限（簡略化）
        lam = 1.0  # リッジ正則化（A の初期 I に対応）
        delta = 0.1  # 信頼度（1-δ）
        S = 1.0  # パラメータノルムの上限（簡略化）
        t = max(1, self.total_iterations)  # 反復番号（ゼロ回避）
        val = sigma * math.sqrt(self.dim * math.log(max(1e-9, 1 + (t - 1) / lam) / delta)) + math.sqrt(lam) * S  # β_t
        return float(val)  # スカラーで返す

    # ---- Continuous-Fixed：固定点反復で連続最適方向を近似（単位ベクトルを返す） ----
    def _select_direction_continuous_fixed(self) -> torch.Tensor:  # 方向ベクトル a（||a||=1）を返す
        A = 0.5 * (self.A + self.A.t())  # 数値安定のため対称化
        A_inv = torch.inverse(A)  # A の逆行列（小規模 d を想定）
        theta_hat = A_inv @ self.b  # θ̂ = A^{-1} b（LinUCB の重み推定）
        beta = self._beta_t()  # β_t（信頼幅）
        x = theta_hat.clone()  # 初期ベクトルを θ̂ から開始
        if float(x.norm()) < 1e-12:  # ほぼゼロなら乱数で初期化
            x = torch.randn_like(theta_hat)  # ランダム初期化
        x = x / (x.norm() + 1e-12)  # 単位ベクトル化
        for _ in range(50):  # 固定点反復（上限50回）
            y = A_inv @ x  # y = A^{-1} x
            denom = torch.sqrt(torch.clamp(x @ y, min=1e-18))  # sqrt(x^T A^{-1} x)
            z = theta_hat + beta * (y / denom)  # z = θ̂ + β * A^{-1}x / sqrt(x^T A^{-1} x)
            x_new = z / (z.norm() + 1e-12)  # 正規化
            if float((x_new - x).norm()) < 1e-8:  # 収束チェック
                x = x_new  # 収束
                break  # 反復終了
            x = x_new  # 継続
        return x  # 連続最適方向（近似）

    # ---- GP のフィット（SingleTaskGP + RBF(ARD) + MLL 最適化） ----
    def _fit_gp(self):  # 既存データ X, Y で GP を再学習
        # RBF の lengthscale に下限を課したい場合のみ制約を付与（0 なら無効）
        base = RBFKernel(ard_num_dims=self.dim, lengthscale_constraint=GreaterThan(self.l_min) if self.l_min > 0 else None)  # RBF(ARD)
        kernel = ScaleKernel(base).to(self.X)  # Scale を噛ませる（出力スケール）
        self.model = SingleTaskGP(self.X, self.Y, covar_module=kernel)  # 単一タスク GP
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)  # MLL
        fit_gpytorch_model(mll)  # MLL 最大化でハイパーパラメータ学習

    # ---- EI の 1 次元ライン最大化（グリッドなし：t に対する勾配上昇） ----
    def _maximize_ei_along_direction(self, x_best: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:  # 次の x を返す
        ei = ExpectedImprovement(self.model, best_f=float(self.Y.min().item()), maximize=False)  # 最小化設定の EI
        dir_unit = direction / (direction.norm() + 1e-12)  # 念のため単位化
        # 箱型制約のもとで t の許容区間 [t_low, t_high] を求める
        lb, ub = self.bounds[0], self.bounds[1]  # 下限・上限（d次元）
        t_low, t_high = -float("inf"), float("inf")  # 初期は広く
        for i in range(self.dim):  # 次元ごとに交差区間を絞る
            a = float(dir_unit[i].item())  # 方向の i 成分
            if abs(a) < 1e-12:  # 成分が 0 に近い場合は無視
                continue  # 制約無し
            # (lb[i] - x_best[i]) / a ≤ t ≤ (ub[i] - x_best[i]) / a を反映
            lo = (float(lb[i].item()) - float(x_best[i].item())) / a  # 下側比
            hi = (float(ub[i].item()) - float(x_best[i].item())) / a  # 上側比
            lo, hi = (min(lo, hi), max(lo, hi))  # 順序を正す
            t_low = max(t_low, lo)  # 全次元の下限の最大
            t_high = min(t_high, hi)  # 全次元の上限の最小
        if not (math.isfinite(t_low) and math.isfinite(t_high)) or t_low >= t_high:  # 区間が壊れた時の保険
            width = float((ub - lb).mean().item())  # 平均幅
            t_low, t_high = -0.1 * width, 0.1 * width  # 小区間に退避

        # t に対する勾配上昇（複数リスタートで堅牢化）
        def ei_of_t(t_scalar: torch.Tensor) -> torch.Tensor:  # t→EI(t) を返す関数
            x = x_best + t_scalar * dir_unit  # ライン上の点 x(t)
            x = x.clamp(lb, ub)  # 箱内に投影（誤差があっても安全）
            return ei(x.unsqueeze(0))  # EI は [N, d] 入力なので [1, d] で評価

        best_val, best_t = -float("inf"), None  # ベスト EI 値と t
        for seed in [0.1, 0.5, 0.9, 0.25, 0.75]:  # 5 リスタート（中点付近＋端寄り）
            t = torch.tensor([(1 - seed) * t_low + seed * t_high], dtype=self.bounds.dtype, requires_grad=True)  # 初期 t
            lr = 0.2 * (t_high - t_low)  # 学習率（区間幅に比例）
            for _ in range(25):  # ステップ回数（軽量）
                val = ei_of_t(t)  # EI(t) を計算
                (-val).backward()  # 最大化したいので負号を付けて“最小化”の勾配に
                with torch.no_grad():  # 勾配更新部分は勾配追跡しない
                    grad = -t.grad  # d(EI)/dt（正方向に上昇）
                    t += lr * grad  # 勾配上昇ステップ
                    t.clamp_(min=t_low, max=t_high)  # t を許容区間に射影
                    t.grad.zero_()  # 勾配をクリア
                    lr *= 0.9  # 少しずつ減衰（安定化）
            with torch.no_grad():  # ベスト更新チェック
                v = float(ei_of_t(t).item())  # 最終 EI
                if v > best_val:  # 改善なら採用
                    best_val, best_t = v, float(t.item())  # ベストを更新
        x_new = x_best + best_t * dir_unit  # ベスト t の点を採用
        return x_new.clamp(lb, ub)  # 箱内に収めて返す

    # ---- NLPD＊dir のスカラー報酬 r′ を計算し、方向でベクトル化して返す ----
    def _nlpd_reward_vector(self, x: torch.Tensor, y_actual: float, direction: torch.Tensor) -> torch.Tensor:  # r′·a を返す
        with torch.no_grad():  # 予測時は勾配不要
            post = self.model.posterior(x.unsqueeze(0))  # 事後分布 p(f(x)|D)
            mu = float(post.mean.squeeze().item())  # 予測平均 μ(x)
            var = float(post.variance.squeeze().item())  # 予測分散 s^2(x)
        try:
            noise_var = float(self.model.likelihood.noise.mean().item())  # 観測ノイズ分散（推定値）
        except Exception:
            noise_var = 1e-6  # 取得できない場合の小さな既定値
        sigma2 = max(var + noise_var, 1e-12)  # 合成分散 σ_y^2（下限で安定化）
        resid2 = (y_actual - mu) ** 2  # 残差の二乗 (y-μ)^2
        nlpd = 0.5 * math.log(2.0 * math.pi * sigma2) + 0.5 * (resid2 / sigma2)  # NLPD の定義
        ema = self._ema_nlpd.value  # 現在の EMA
        r_prime = max(0.0, min(1.0, nlpd / max(ema, 1e-12)))  # EMA で正規化→[0,1] へクリップ
        self._ema_nlpd.update(nlpd)  # EMA を更新
        direction_unit = direction / (direction.norm() + 1e-12)  # 方向を単位化（符号は保持）
        return torch.as_tensor(r_prime, dtype=self.bounds.dtype) * direction_unit  # r′·a（ベクトル）を返す

    # ---- 初期化：初期点評価→GP 構築→最良点の初期化 ----
    def initialize(self):  # 実験開始時に 1 度だけ呼ぶ
        # 一様乱数で初期点を生成（[lb, ub] の箱内）
        lb, ub = self.bounds[0], self.bounds[1]  # 下限・上限
        self.X = torch.rand(self.n_initial, self.dim, dtype=self.bounds.dtype) * (ub - lb) + lb  # 初期 X
        with torch.no_grad():  # 目的関数評価は勾配不要
            y = self.objective_function(self.X)  # ベクトル化評価に対応していると速い
        self.Y = y.reshape(-1, 1).double()  # 列ベクトルに整形
        self._fit_gp()  # GP 学習
        # 観測最小値で最良を初期化（最小化設定）
        idx = int(torch.argmin(self.Y).item())  # 最良インデックス
        self.best_x = self.X[idx].detach().clone()  # 最良点
        self.best_y = float(self.Y[idx].item())  # その値

    # ---- メインループ：Continuous-Fixed で方向→EI でライン最適化→NLPD＊dir で更新 ----
    def optimize(self):  # 最適化本体（最良点と値を返す）
        self.initialize()  # 初期化
        n_eval = self.X.shape[0]  # 既に評価済みの回数（初期点数）
        while n_eval < self.n_max:  # 目標回数まで繰り返す
            self.total_iterations += 1  # 反復カウンタを進める
            a = self._select_direction_continuous_fixed()  # Continuous‑Fixed で方向を選択
            x_new = self._maximize_ei_along_direction(self.best_x, a)  # 方向に沿って EI を最大化（グリッドなし）
            with torch.no_grad():  # 目的関数評価（最小化）
                y_new = float(self.objective_function(x_new.unsqueeze(0)).item())  # 新規点の y
            r_vec = self._nlpd_reward_vector(x_new, y_new, a)  # NLPD＊dir のベクトル報酬 r′·a
            x_arm = a.view(-1, 1)  # 方向を列ベクトルに
            self.A += x_arm @ x_arm.t()  # LinUCB の A ← A + a a^T
            self.b += r_vec  # LinUCB の b ← b + r′·a
            # データを追加して GP を再学習
            self.X = torch.cat([self.X, x_new.unsqueeze(0)], dim=0)  # X に追加
            self.Y = torch.cat([self.Y, torch.tensor([[y_new]], dtype=self.bounds.dtype)], dim=0)  # Y に追加
            self._fit_gp()  # GP をアップデート
            # 最良の更新（観測最小値ベースで簡素化）
            if y_new < self.best_y:  # 改善していれば更新
                self.best_y = y_new  # 最良値更新
                self.best_x = x_new.detach().clone()  # 最良点更新
            n_eval += 1  # 評価回数をインクリメント
        return self.best_x, self.best_y  # 探索終了：最良点と値を返す


# ===== 使い方デモ（必要なら実行） =====
if __name__ == "__main__":  # スクリプトとして実行された場合のみ
    # 例：Styblinski–Tang（最小値 = -39.16599 * d）を 10 次元で最小化
    def styblinski_tang(x: torch.Tensor) -> torch.Tensor:  # ベンチ関数（ベクトル化対応）
        z = x  # そのまま使う（前処理なし）
        return 0.5 * torch.sum(z**4 - 16.0 * z**2 + 5.0 * z, dim=-1)  # 定義式

    d = 10  # 次元
    bounds = torch.tensor([[-5.0] * d, [5.0] * d], dtype=torch.double)  # 箱型制約
    opt = SimpleLinBanditBONLPDDir(objective_function=styblinski_tang, bounds=bounds, n_initial=5, n_max=60)  # インスタンス
    best_x, best_y = opt.optimize()  # 最適化を実行
    print("best y:", best_y)  # 最終最良値を表示
    print("best x (first 5 dims):", best_x[:5])  # 最良点の先頭5成分を表示
