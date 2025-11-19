#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LinBandit-BO: Linear Bandit-based Bayesian Optimization (NLPD*dir 版)

高次元最適化問題に対して、Linear Bandit (LinUCB) と Bayesian Optimization (BO) を
組み合わせたアルゴリズムです。本実装は hpo_benchmark_reward_function_comparison で
比較している「NLPD*dir」を基本アルゴリズムとして採用しています。

要点:
- 方向選択: Continuous-Fixed（固定点反復）で信頼楕円体に基づく方向を選択
- 取得関数: Expected Improvement (EI)
- ラインサーチ: 方向に沿った1次元の粗グリッド + 局所33点で安定最適化
- 報酬: NLPD（負の対数予測密度）を平滑化正規化したスカラー r を用い、b ← b + r·direction
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional

# BoTorch / GPyTorch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.constraints import GreaterThan
from botorch.acquisition import ExpectedImprovement


# 実験側で適切な dtype を設定するため、ここでは強制しない（実験スクリプトで上書き）

# プロット設定
# NOTE: フォントは実験スクリプト側（例: japanize_matplotlib など）で適切に設定する。
# ここで固定すると日本語フォント指定を上書きしてしまい、文字化けの原因になるため設定しない。
plt.rcParams["figure.dpi"] = 100

import warnings
warnings.filterwarnings("ignore")


class EMA:
    """指数移動平均（NLPD 正規化用途）。"""
    def __init__(self, alpha: float = 0.1, eps: float = 1e-8) -> None:
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._m = None

    def update(self, x: float) -> float:
        if self._m is None:
            self._m = float(x)
        else:
            self._m = self.alpha * float(x) + (1.0 - self.alpha) * self._m
        return float(self._m)

    @property
    def value(self) -> float:
        return float(self._m if self._m is not None else self.eps)


class LinBanditBO:
    """
    LinBandit-BO: Linear Bandit-based Bayesian Optimization（NLPD*dir）

    各イテレーションで、固定点反復（Continuous-Fixed）により探索方向を決定し、
    その方向に沿って EI を 1 次元探索（粗グリッド＋局所 33 点）で最大化して次点を提案。
    予測の負の対数予測密度（NLPD）をスカラー報酬（EMA 正規化）として用い、
    b ← b + r·direction で LinUCB の十分統計を更新します。
    """
    
    def __init__(self, objective_function, bounds, n_initial=5, n_max=100,
                 use_lengthscale_lower_bound: bool = False,
                 l_min: float = 0.05,
                 normalize_inputs_for_gp: bool = False,
                 track_history: bool = False,
                 device: Optional[torch.device] = None,
                 initial_X: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        objective_function : callable
            最小化したい目的関数。torch.Tensorを入力として受け取り、スカラー値を返す。
        bounds : torch.Tensor
            各次元の探索範囲。shape=[2, dim]で、bounds[0]が下限、bounds[1]が上限。
        n_initial : int
            初期サンプル数（ランダムサンプリング）
        n_max : int
            最大評価回数
        use_lengthscale_lower_bound : bool
            RBFカーネルの長さスケールに下限を課す（入力は正規化空間を前提）。
        l_min : float
            正規化空間 [0,1]^d における長さスケールの下限（例: 0.05）。
        normalize_inputs_for_gp : bool
            True の場合、GPへ渡す入力Xを [0,1]^d に正規化して学習・予測。
            use_lengthscale_lower_bound が True の場合は自動的に正規化を有効化。
        track_history : bool
            実験用に、選択方向および報酬ベクトルの履歴を保存。
        """
        self.objective_function = objective_function
        # device 選択
        self.device = bounds.device if device is None else device
        # bounds の dtype を尊重（実験側で統一）
        self.bounds = bounds.to(device=self.device)
        self.dim = bounds.shape[1]
        self.n_initial = n_initial
        self.n_max = n_max
        
        # Linear Banditのパラメータ
        self.A = torch.eye(self.dim, dtype=self.bounds.dtype, device=self.device)
        self.b = torch.zeros(self.dim, dtype=self.bounds.dtype, device=self.device)
        
        # 初期点の生成（外部から与えられた場合はそれを使用）
        if initial_X is not None:
            X0 = initial_X.to(dtype=self.bounds.dtype, device=self.device)
            assert X0.shape == (n_initial, self.dim), "initial_X shape must be (n_initial, dim)"
            # 念のため境界にクランプ
            self.X = torch.clamp(X0, self.bounds[0], self.bounds[1])
        else:
            self.X = torch.rand(n_initial, self.dim, dtype=self.bounds.dtype, device=self.device) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        
        # 状態変数
        self.Y = None
        self.best_value = None
        self.best_point = None
        self.model = None
        self.eval_history = []
        self.theta_history = []
        self.total_iterations = 0

        # カーネル長さスケール下限/正規化/履歴
        self.use_lengthscale_lower_bound = bool(use_lengthscale_lower_bound)
        self.l_min = float(l_min)
        self.normalize_inputs_for_gp = bool(normalize_inputs_for_gp) or self.use_lengthscale_lower_bound
        self._range = (self.bounds[1] - self.bounds[0]).to(dtype=self.bounds.dtype, device=self.device)
        self.track_history = bool(track_history)
        if self.track_history:
            self.reward_history = []
            self.selected_direction_history = []

        # NLPD の指数移動平均（スケール正規化用）
        self._ema_nlpd = EMA(alpha=0.1)

    def _compute_beta_t(self):
        """LinUCBのβ_t（現行式）を計算。"""
        sigma = 1.0
        L = 1.0
        lambda_reg = 1.0
        delta = 0.1
        S = 1.0
        current_round_t = max(1, self.total_iterations)
        log_term = max(1e-9, 1 + (current_round_t - 1) * (L**2) / lambda_reg)
        beta_t = sigma * math.sqrt(self.dim * math.log(log_term / delta)) + math.sqrt(lambda_reg) * S
        return beta_t
        
    def _to_normalized(self, X):
        return torch.clamp((X - self.bounds[0]) / self._range, 0.0, 1.0)

    def update_model(self):
        """ガウス過程モデルの更新（必要に応じて入力正規化・l下限）"""
        X_gp = self._to_normalized(self.X) if self.normalize_inputs_for_gp else self.X
        # RBFカーネル（必要に応じて長さスケールに下限を課す）。dtype/deviceは to(X_gp) で揃える。
        if self.use_lengthscale_lower_bound:
            base = RBFKernel(ard_num_dims=self.X.shape[-1],
                             lengthscale_constraint=GreaterThan(self.l_min))
        else:
            base = RBFKernel(ard_num_dims=self.X.shape[-1])
        kernel = ScaleKernel(base).to(X_gp)
        self.model = SingleTaskGP(X_gp, self.Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)
        
    def initialize(self):
        """初期化：初期点での評価とモデル構築"""
        y_val = self.objective_function(self.X)
        self.Y = y_val.unsqueeze(-1).to(dtype=self.bounds.dtype, device=self.device)
        
        # モデルの初期化
        self.update_model()
        
        # 最良点の初期化
        X_query = self._to_normalized(self.X) if self.normalize_inputs_for_gp else self.X
        post_mean = self.model.posterior(X_query).mean.squeeze(-1)
        bi = post_mean.argmin()
        self.best_value = post_mean[bi].item()
        self.best_point = self.X[bi]
        self.eval_history = [self.best_value] * self.n_initial
        
    # generate_arms / select_arm / select_direction_continuous_evd は
    # 固定点反復で方向を選ぶ設計では不要のため削除。

    def _select_direction_continuous_fixed(self):
        """固定点反復による連続最適方向の近似。単位ベクトルを返す。"""
        A = 0.5 * (self.A + self.A.t())
        A_inv = torch.inverse(A)
        theta_hat = A_inv @ self.b
        beta_t = self._compute_beta_t()

        # 初期ベクトル
        if float(theta_hat.norm()) < 1e-12:
            x = torch.randn_like(theta_hat)
        else:
            x = theta_hat.clone()
        x = x / (x.norm() + 1e-12)

        for _ in range(50):
            y = A_inv @ x
            denom = torch.sqrt(torch.clamp(x @ y, min=1e-18))
            z = theta_hat + beta_t * (y / denom)
            x_new = z / (z.norm() + 1e-12)
            if float((x_new - x).norm()) < 1e-8:
                x = x_new
                break
            x = x_new
        return x
    
    def propose_new_x(self, direction):
        """方向に沿った1D EI最大化（粗グリッド + 局所33点）。"""
        assert self.model is not None and self.best_point is not None
        ei = ExpectedImprovement(self.model, best_f=self.best_value, maximize=False)

        # 可動範囲 [lb, ub] を direction に沿って算出
        active = direction.abs() > 1e-9
        if not active.any():
            lb, ub = -1.0, 1.0
        else:
            ratios_lower = (self.bounds[0] - self.best_point) / (direction + 1e-12 * (~active))
            ratios_upper = (self.bounds[1] - self.best_point) / (direction + 1e-12 * (~active))
            t_bounds = torch.stack([
                torch.minimum(ratios_lower, ratios_upper),
                torch.maximum(ratios_lower, ratios_upper)
            ], dim=-1)
            lb = -float("inf"); ub = float("inf")
            for idx in range(self.dim):
                if active[idx]:
                    lb = max(lb, float(t_bounds[idx, 0]))
                    ub = min(ub, float(t_bounds[idx, 1]))
        if not (math.isfinite(lb) and math.isfinite(ub)) or lb >= ub:
            width = float(self.bounds[1, 0] - self.bounds[0, 0])
            lb, ub = -0.1 * width, 0.1 * width

        device = self.bounds.device; dtype = self.bounds.dtype
        n_grid = max(128, min(512, 64 + 4 * self.dim))
        t_grid = torch.linspace(lb, ub, steps=n_grid, device=device, dtype=dtype)
        pts = self.best_point.unsqueeze(0) + t_grid.reshape(-1, 1) * direction.unsqueeze(0)
        pts = torch.clamp(pts, self.bounds[0].unsqueeze(0), self.bounds[1].unsqueeze(0))
        pts_n = self._to_normalized(pts) if self.normalize_inputs_for_gp else pts
        with torch.no_grad():
            vals = ei(pts_n.unsqueeze(1)).view(-1)
        mask = torch.isfinite(vals)
        if not mask.any():
            alpha_star = 0.0
        else:
            vals[~mask] = -float("inf")
            best_idx = int(torch.argmax(vals).item())
            alpha_star = float(t_grid[best_idx].item())
            step = float((ub - lb) / (n_grid - 1)) if n_grid > 1 else 0.0
            if step > 0:
                local_lb = max(lb, alpha_star - 5 * step)
                local_ub = min(ub, alpha_star + 5 * step)
                t_local = torch.linspace(local_lb, local_ub, steps=33, device=device, dtype=dtype)
                pts_l = self.best_point.unsqueeze(0) + t_local.reshape(-1, 1) * direction.unsqueeze(0)
                pts_l = torch.clamp(pts_l, self.bounds[0].unsqueeze(0), self.bounds[1].unsqueeze(0))
                pts_l_n = self._to_normalized(pts_l) if self.normalize_inputs_for_gp else pts_l
                with torch.no_grad():
                    vals_l = ei(pts_l_n.unsqueeze(1)).view(-1)
                mask_l = torch.isfinite(vals_l)
                if mask_l.any():
                    vals_l[~mask_l] = -float("inf")
                    j = int(torch.argmax(vals_l).item())
                    alpha_star = float(t_local[j].item())
        x_new = self.best_point + alpha_star * direction
        return torch.clamp(x_new, self.bounds[0], self.bounds[1])

    def _compute_nlpd_reward(self, y_actual: float, mu_pred: float, var_pred: float, noise_var: float) -> float:
        """NLPD を EMA でスケール正規化し [0,1] にクリップして返す。"""
        eps = 1e-9
        sigma2 = max(float(var_pred + noise_var), eps)
        resid2 = float((y_actual - mu_pred) ** 2)
        nlpd = 0.5 * math.log(2.0 * math.pi * sigma2) + 0.5 * (resid2 / sigma2)
        ema = self._ema_nlpd.value
        r_scaled = nlpd / max(ema, eps)
        self._ema_nlpd.update(nlpd)
        # 数値安定化のためクリップ
        return max(0.0, min(1.0, float(r_scaled)))
    
    def optimize(self):
        """メインの最適化ループ（方向は固定点反復、報酬はNLPD*dir）。"""
        print(f"LinBandit-BO開始: {self.dim}次元, 最大{self.n_max}回評価")
        print(f"最適化設定: 方向=Continuous-Fixed, 報酬=NLPD*dir, 1D-EI(粗+局所33)")
        
        # 初期化
        self.initialize()
        n_iter = self.n_initial
        
        while n_iter < self.n_max:
            self.total_iterations += 1
            
            # 探索方向の選択（固定点反復）
            direction = self._select_direction_continuous_fixed()
            direction = direction / (direction.norm() + 1e-12)
            
            # 選択された方向に沿った最適化
            new_x = self.propose_new_x(direction)
            
            # 予測（平均・分散）と実際の評価
            with torch.no_grad():
                x_q = self._to_normalized(new_x.unsqueeze(0)) if self.normalize_inputs_for_gp else new_x.unsqueeze(0)
                post = self.model.posterior(x_q)
                mu_pred = float(post.mean.squeeze().item())
                var_pred = float(post.variance.squeeze().item())
            actual_y = float(self.objective_function(new_x.unsqueeze(0)).squeeze().item())

            # 観測ノイズ分散の推定
            try:
                noise_var = float(self.model.likelihood.noise.mean().item())
            except Exception:
                noise_var = 1e-6

            # NLPD スカラー報酬 r（[0,1]にクリップ）
            r_scalar = self._compute_nlpd_reward(actual_y, mu_pred, var_pred, noise_var)
            reward_vector = r_scalar * (direction / (direction.norm() + 1e-12))

            # Linear Bandit パラメータの更新
            x_arm = direction.view(-1, 1)
            self.A += x_arm @ x_arm.t()
            self.b += reward_vector
            # θ の履歴も更新
            try:
                self.theta_history.append(torch.inverse(self.A) @ self.b)
            except Exception:
                pass
            if self.track_history:
                self.selected_direction_history.append(direction.detach().clone())
                self.reward_history.append(torch.as_tensor(reward_vector).detach().clone())
            
            # データとモデルの更新
            self.X = torch.cat([self.X, new_x.unsqueeze(0)], 0)
            self.Y = torch.cat([self.Y, torch.tensor([[actual_y]], dtype=torch.float32, device=self.X.device)], 0)
            self.update_model()
            
            # 最良点の更新
            with torch.no_grad():
                X_query2 = self._to_normalized(self.X) if self.normalize_inputs_for_gp else self.X
                posterior_mean = self.model.posterior(X_query2).mean.squeeze(-1)
            current_best_idx = posterior_mean.argmin()
            self.best_value = posterior_mean[current_best_idx].item()
            self.best_point = self.X[current_best_idx]
            
            self.eval_history.append(self.best_value)
            n_iter += 1
            
            # 進捗表示
            if n_iter % 10 == 0:
                print(f"  評価回数: {n_iter}/{self.n_max}, 現在の最良値: {self.best_value:.6f}")
                
        print(f"\n最適化完了!")
        print(f"最良値: {self.best_value:.6f}")
        print(f"最良点: {self.best_point[:5]}... (最初の5次元)")
        
        
        return self.best_point, self.best_value


# デモ用のテスト関数
def styblinski_tang(x):
    """Styblinski-Tang関数（最小値: -39.16599 * dim）"""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return 0.5 * torch.sum(x**4 - 16.0*x**2 + 5.0*x, dim=-1)

def rastrigin(x):
    """Rastrigin関数（最小値: 0）"""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sum(x**2 - 10.0*torch.cos(2*math.pi*x) + 10.0, dim=-1)


def run_demo():
    """LinBandit-BOのデモ実行（最適化された設定）"""
    print("=" * 60)
    print("LinBandit-BO デモ（最適化設定版）")
    print("=" * 60)
    
    # 問題設定
    dim = 10  # 10次元の最適化問題
    bounds = torch.tensor([[-5.0]*dim, [5.0]*dim], dtype=torch.float32)
    
    # テスト関数の選択
    test_func = styblinski_tang
    global_optimum = -39.16599 * dim
    
    # LinBandit-BOの実行
    optimizer = LinBanditBO(
        objective_function=test_func,
        bounds=bounds,
        n_initial=5,
        n_max=100,
    )
    
    best_x, best_y = optimizer.optimize()
    
    # 結果の可視化
    plt.figure(figsize=(10, 6))
    
    # 収束履歴
    plt.subplot(1, 2, 1)
    iterations = np.arange(1, len(optimizer.eval_history) + 1)
    plt.plot(iterations, optimizer.eval_history, 'b-', linewidth=2)
    plt.axhline(y=global_optimum, color='r', linestyle='--', label=f'Global optimum: {global_optimum:.2f}')
    plt.xlabel('Iterations')
    plt.ylabel('Best Value Found')
    plt.title('Convergence History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 各次元の使用頻度（theta_historyの最終値）
    plt.subplot(1, 2, 2)
    if optimizer.theta_history:
        final_theta = optimizer.theta_history[-1].abs().cpu().numpy()
        plt.bar(range(dim), final_theta)
        plt.xlabel('Dimension')
        plt.ylabel('Absolute Theta Value')
        plt.title('Dimension Importance (Final Theta)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


if __name__ == "__main__":
    # デモの実行
    run_demo()
