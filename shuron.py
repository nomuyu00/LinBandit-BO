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
from botorch.models.transforms.outcome import Standardize  # 出力標準化
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


# ===== 本体：超シンプル LinBandit-BO（Continuous-Fixed + EI + 改善量×方向） =====
class LinBanditBO:  # 読みやすさ重視の最小クラス
    def __init__(self, objective_function, bounds, n_initial=5, n_max=100, l_min=0.0, initial_X=None):
        self.objective_function = objective_function  # 最小化したい目的関数 f(x)
        self.bounds = bounds.detach().clone().double()  # 探索範囲 [2, d]（下限・上限）
        self.dim = self.bounds.shape[1]  # 次元 d
        self.n_initial = int(n_initial)  # 初期サンプル数
        self.n_max = int(n_max)  # 総評価回数（初期点を含む）
        self.coord_arms = torch.eye(self.dim, dtype=self.bounds.dtype)  # アーム＝各次元の単位ベクトル
        self.A = torch.eye(self.dim, dtype=self.bounds.dtype)  # LinUCB の A 行列（初期は単位行列）
        self.b = torch.zeros(self.dim, dtype=self.bounds.dtype)  # LinUCB の b ベクトル（初期ゼロ）
        self._ema_nlpd = EMA(alpha=0.1)  # NLPD の EMA 正規化器（今は未使用）
        self.model = None  # GP モデル（あとで初期化）
        self.X = None  # 既存の入力履歴（N×d）
        self.Y = None  # 既存の出力履歴（N×1）
        self.best_x = None  # これまでの最良点
        self.best_y = None  # その値
        self.total_iterations = 0  # 反復カウンタ
        self.l_min = float(l_min)  # RBF の lengthscale 下限（正規化空間での下限値）
        self.initial_X = initial_X  # 事前に与えられた初期点（任意）

        # 入力正規化（[0,1]^d）をGP学習/予測に適用
        self._range = (self.bounds[1] - self.bounds[0]).clamp_min(1e-12)
        self._use_input_normalization = True
        self.X_norm = None  # 正規化済み入力（GP用）

        # 出力標準化の利用（EIのスケール安定化）
        self._use_output_standardize = True

        # ライン探索に用いる信頼領域(TR)の状態（TuRBO 風の簡易バージョン）
        self.tr_length = 0.8
        self.tr_min_length = 0.5 ** 7
        self.tr_max_length = 1.6
        self.tr_success = 0
        self.tr_failure = 0

        # 可視化用の履歴（実験スクリプトで参照）
        # - continuous_fixed で選択された方向ベクトル（単位化後）
        # - スカラー報酬 r と方向 a_unit の積 r * a_unit
        self.selected_direction_history = []  # List[torch.Tensor] 形状 [d]
        self.reward_history = []              # List[torch.Tensor] 形状 [d]

    # ---- LinUCB の β_t（簡易式）：連続方向の信頼半径に使う ----
    def _beta_t(self) -> float:
        sigma = 1.0  # ノイズの上限（簡略化）
        lam = 1.0  # リッジ正則化（A の初期 I に対応）
        delta = 0.1  # 信頼度（1-δ）
        S = 1.0  # パラメータノルムの上限（簡略化）
        t = max(1, self.total_iterations)  # 反復番号（ゼロ回避）
        val = sigma * math.sqrt(self.dim * math.log(max(1e-9, 1 + (t - 1) / lam) / delta)) + math.sqrt(lam) * S
        return float(val)

    # ---- Continuous-Fixed：固定点反復で連続最適方向を近似（単位ベクトルを返す） ----
    def _select_direction_continuous_fixed(self) -> torch.Tensor:
        A = 0.5 * (self.A + self.A.t())  # 数値安定のため対称化
        A_inv = torch.inverse(A)
        theta_hat = A_inv @ self.b  # θ̂ = A^{-1} b
        beta = self._beta_t()
        x = theta_hat.clone()
        if float(x.norm()) < 1e-12:
            x = torch.randn_like(theta_hat)
        x = x / (x.norm() + 1e-12)
        for _ in range(50):
            y = A_inv @ x
            denom = torch.sqrt(torch.clamp(x @ y, min=1e-18))  # sqrt(x^T A^{-1} x)
            z = theta_hat + beta * (y / denom)
            x_new = z / (z.norm() + 1e-12)
            if float((x_new - x).norm()) < 1e-8:
                x = x_new
                break
            x = x_new
        return x  # 連続最適方向（近似, ほぼ単位ベクトル）

    # ---- GP のフィット（SingleTaskGP + RBF(ARD) + MLL 最適化） ----
    def _to_normalized(self, X: torch.Tensor) -> torch.Tensor:
        if not self._use_input_normalization:
            return X
        return torch.clamp((X - self.bounds[0]) / self._range, 0.0, 1.0)

    def _fit_gp(self):
        X_gp = self._to_normalized(self.X)
        base = RBFKernel(
            ard_num_dims=self.dim,
            lengthscale_constraint=GreaterThan(self.l_min) if self.l_min > 0 else None,
        )
        kernel = ScaleKernel(base).to(X_gp)
        if self._use_output_standardize:
            self.model = SingleTaskGP(X_gp, self.Y, covar_module=kernel, outcome_transform=Standardize(m=1))
        else:
            self.model = SingleTaskGP(X_gp, self.Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    # ---- EI の 1 次元ライン最大化（TR内・粗グリッド→局所33点） ----
    def _maximize_ei_along_direction(self, x_best: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        # EIのbest_fは出力標準化を考慮
        if getattr(self.model, "outcome_transform", None) is not None:
            try:
                y_t, _ = self.model.outcome_transform(self.Y)
                best_f_val = float(y_t.min().item())
            except Exception:
                best_f_val = float(self.Y.min().item())
        else:
            best_f_val = float(self.Y.min().item())
        ei = ExpectedImprovement(self.model, best_f=best_f_val, maximize=False)

        dir_unit = direction / (direction.norm() + 1e-12)
        lb, ub = self.bounds[0], self.bounds[1]

        # 箱由来/ TR由来の t 範囲
        def interval_from_box(box_lb: torch.Tensor, box_ub: torch.Tensor) -> tuple[float, float]:
            t_lo, t_hi = -float("inf"), float("inf")
            for i in range(self.dim):
                a = float(dir_unit[i].item())
                if abs(a) < 1e-12:
                    continue
                lo = (float(box_lb[i].item()) - float(x_best[i].item())) / a
                hi = (float(box_ub[i].item()) - float(x_best[i].item())) / a
                lo, hi = (min(lo, hi), max(lo, hi))
                t_lo = max(t_lo, lo)
                t_hi = min(t_hi, hi)
            return t_lo, t_hi

        span = (ub - lb)
        half = 0.5 * self.tr_length * span
        tr_lb = torch.max(x_best - half, lb)
        tr_ub = torch.min(x_best + half, ub)

        g_lo, g_hi = interval_from_box(lb, ub)
        tr_lo, tr_hi = interval_from_box(tr_lb, tr_ub)
        t_low = max(g_lo, tr_lo)
        t_high = min(g_hi, tr_hi)
        if not (math.isfinite(t_low) and math.isfinite(t_high)) or t_low >= t_high:
            width = float(span.mean().item())
            t_low, t_high = -0.1 * width, 0.1 * width

        n_grid = max(128, min(512, 64 + 4 * self.dim))
        t_grid = torch.linspace(t_low, t_high, steps=n_grid, dtype=self.bounds.dtype)
        pts = x_best.unsqueeze(0) + t_grid.reshape(-1, 1) * dir_unit.unsqueeze(0)
        pts = torch.clamp(pts, lb.unsqueeze(0), ub.unsqueeze(0))
        pts_n = self._to_normalized(pts)
        with torch.no_grad():
            vals = ei(pts_n.unsqueeze(1)).view(-1)
        mask = torch.isfinite(vals)
        if not mask.any():
            alpha_star = 0.0
        else:
            vals[~mask] = -float("inf")
            i = int(torch.argmax(vals).item())
            alpha_star = float(t_grid[i].item())
            step = float((t_high - t_low) / (n_grid - 1)) if n_grid > 1 else 0.0
            if step > 0:
                loc_lb = max(t_low, alpha_star - 5 * step)
                loc_ub = min(t_high, alpha_star + 5 * step)
                t_local = torch.linspace(loc_lb, loc_ub, steps=33, dtype=self.bounds.dtype)
                pts_l = x_best.unsqueeze(0) + t_local.reshape(-1, 1) * dir_unit.unsqueeze(0)
                pts_l = torch.clamp(pts_l, lb.unsqueeze(0), ub.unsqueeze(0))
                pts_l_n = self._to_normalized(pts_l)
                with torch.no_grad():
                    vals_l = ei(pts_l_n.unsqueeze(1)).view(-1)
                mask_l = torch.isfinite(vals_l)
                if mask_l.any():
                    vals_l[~mask_l] = -float("inf")
                    j = int(torch.argmax(vals_l).item())
                    alpha_star = float(t_local[j].item())
        x_new = x_best + alpha_star * dir_unit
        return torch.clamp(x_new, lb, ub)

    # ---- スカラー報酬 r：best_y からの改善量（悪化は 0） ----
    def _scalar_improvement_reward(self, old_best: float, new_val: float) -> float:
        """観測が改善したときだけ、その改善量を返す（悪化・同値は 0）。"""
        delta = float(old_best - new_val)
        if delta > 0.0:
            return delta
        return 0.0

    # ---- 初期化：初期点評価→GP 構築→最良点の初期化 ----
    def initialize(self):
        lb, ub = self.bounds[0], self.bounds[1]
        if self.initial_X is not None:
            X0 = torch.as_tensor(self.initial_X, dtype=self.bounds.dtype)
            assert X0.shape == (self.n_initial, self.dim)
            self.X = torch.clamp(X0, lb, ub)
        else:
            self.X = torch.rand(self.n_initial, self.dim, dtype=self.bounds.dtype) * (ub - lb) + lb
        with torch.no_grad():
            y = self.objective_function(self.X)
        self.Y = y.reshape(-1, 1).double()
        self.X_norm = self._to_normalized(self.X)
        self._fit_gp()
        idx = int(torch.argmin(self.Y).item())
        self.best_x = self.X[idx].detach().clone()
        self.best_y = float(self.Y[idx].item())
        self.selected_direction_history = []
        self.reward_history = []

    # ---- メインループ：Continuous-Fixed で方向→EI でライン最適化→改善量×方向で LinUCB 更新 ----
    def optimize(self):
        self.initialize()
        n_eval = self.X.shape[0]
        while n_eval < self.n_max:
            self.total_iterations += 1
            a = self._select_direction_continuous_fixed()
            a_unit = a / (a.norm() + 1e-12)

            old_best = self.best_y
            x_new = self._maximize_ei_along_direction(self.best_x, a)
            with torch.no_grad():
                y_new = float(self.objective_function(x_new.unsqueeze(0)).item())

            # スカラー報酬（改善量。悪化・据え置きは 0）
            r_scalar = self._scalar_improvement_reward(old_best, y_new)
            reward_vec = a_unit * r_scalar  # r * a_unit（各次元への割り振り）

            # 可視化用履歴
            try:
                self.selected_direction_history.append(a_unit.detach().clone())
                self.reward_history.append(reward_vec.detach().clone())
            except Exception:
                pass

            # LinUCB の更新
            x_arm = a_unit.view(-1, 1)
            self.A += x_arm @ x_arm.t()
            self.b += reward_vec

            # データを追加して GP を再学習
            self.X = torch.cat([self.X, x_new.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, torch.tensor([[y_new]], dtype=self.bounds.dtype)], dim=0)
            self.X_norm = self._to_normalized(self.X)
            self._fit_gp()

            # 最良点の更新（観測最小値ベース）
            improved = False
            if y_new < self.best_y:
                self.best_y = y_new
                self.best_x = x_new.detach().clone()
                improved = True

            # TR の更新
            if improved:
                self.tr_success += 1
                self.tr_failure = 0
            else:
                self.tr_success = 0
                self.tr_failure += 1
            fail_tol = max(4, int(math.ceil(self.dim / 1)))
            succ_tol = 3
            if self.tr_success >= succ_tol:
                self.tr_length = min(self.tr_length * 2.0, self.tr_max_length)
                self.tr_success = 0
            elif self.tr_failure >= fail_tol:
                self.tr_length = self.tr_length / 2.0
                self.tr_failure = 0
            if self.tr_length < self.tr_min_length:
                self.tr_length = 0.8
                self.tr_success = 0
                self.tr_failure = 0

            n_eval += 1
        return self.best_x, self.best_y


# ===== 使い方デモ（必要なら実行） =====
if __name__ == "__main__":

    def styblinski_tang(x: torch.Tensor) -> torch.Tensor:
        z = x
        return 0.5 * torch.sum(z**4 - 16.0 * z**2 + 5.0 * z, dim=-1)

    d = 10
    bounds = torch.tensor([[-5.0] * d, [5.0] * d], dtype=torch.double)
    opt = LinBanditBO(objective_function=styblinski_tang, bounds=bounds, n_initial=5, n_max=60)
    best_x, best_y = opt.optimize()
    print("best y:", best_y)
    print("best x (first 5 dims):", best_x[:5])
