#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LinBandit-BO: Linear Bandit-based Bayesian Optimization

高次元最適化問題に対して、Linear Bandit (LinUCB) とBayesian Optimization (BO)を
組み合わせたアルゴリズムです。

主な特徴:
- Linear UCBで探索方向を適応的に選択
- ベイズ最適化で選択された方向に沿って最適点を探索
- 高次元でも効率的に動作（100次元以上でも実用的）
- 最適化された設定：0.5x arms + 勾配ベース報酬
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch

# BoTorch / GPyTorch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.constraints import GreaterThan
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# デフォルトのdtypeをfloat32に設定
torch.set_default_dtype(torch.float32)

# プロット設定
plt.rcParams["figure.dpi"] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'

import warnings
warnings.filterwarnings("ignore")


class LinBanditBO:
    """
    LinBandit-BO: Linear Bandit-based Bayesian Optimization
    
    高次元最適化のためのアルゴリズムで、Linear UCBとBayesian Optimizationを統合。
    各イテレーションで、LinUCBが探索方向を選択し、その方向に沿ってBOが最適化を行います。
    """
    
    def __init__(self, objective_function, bounds, n_initial=5, n_max=100, 
                 coordinate_ratio=0.8, n_arms=None, L_min: float = 0.1,
                 use_lengthscale_lower_bound: bool = False,
                 l_min: float = 0.05,
                 normalize_inputs_for_gp: bool = False,
                 track_history: bool = False):
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
        coordinate_ratio : float
            座標方向の割合（0.0-1.0）。1.0なら全て座標方向、0.0なら全てランダム方向。
        n_arms : int or None
            アーム数。Noneの場合は次元数の半分（最適化された設定）を使用。
        L_min : float
            勾配ノルムの推定値 L_hat に対する下限値。報酬スケーリングの安定化のために使用。
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
        self.bounds = bounds.float()
        self.dim = bounds.shape[1]
        self.n_initial = n_initial
        self.n_max = n_max
        self.coordinate_ratio = coordinate_ratio
        
        # 最適なアーム数設定（実験結果に基づく）
        self.n_arms = n_arms if n_arms is not None else max(1, self.dim // 2)
        
        # Linear Banditのパラメータ
        self.A = torch.eye(self.dim)
        self.b = torch.zeros(self.dim)
        
        # 初期点の生成
        self.X = torch.rand(n_initial, self.dim) * (bounds[1] - bounds[0]) + bounds[0]
        self.X = self.X.float()
        
        # 状態変数
        self.Y = None
        self.best_value = None
        self.best_point = None
        self.model = None
        self.eval_history = []
        self.theta_history = []
        self.scale_init = 1.0
        self.total_iterations = 0
        
        # 推定リプシッツ定数と下限
        self.L_min = float(L_min)
        self.L_hat = max(1.0, self.L_min)

        # カーネル長さスケール下限/正規化/履歴
        self.use_lengthscale_lower_bound = bool(use_lengthscale_lower_bound)
        self.l_min = float(l_min)
        self.normalize_inputs_for_gp = bool(normalize_inputs_for_gp) or self.use_lengthscale_lower_bound
        self._range = (self.bounds[1] - self.bounds[0]).float()
        self.track_history = bool(track_history)
        if self.track_history:
            self.reward_history = []
            self.selected_direction_history = []
        
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
        self.Y = y_val.unsqueeze(-1).float()
        
        # スケーリング係数の計算
        y_max, y_min = self.Y.max().item(), self.Y.min().item()
        self.scale_init = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
        
        # モデルの初期化
        self.update_model()
        
        # 最良点の初期化
        X_query = self._to_normalized(self.X) if self.normalize_inputs_for_gp else self.X
        post_mean = self.model.posterior(X_query).mean.squeeze(-1)
        bi = post_mean.argmin()
        self.best_value = post_mean[bi].item()
        self.best_point = self.X[bi]
        self.eval_history = [self.best_value] * self.n_initial
        
    def generate_arms(self):
        """
        LinBandit-BOの特徴的な部分：ランダムに方向を選択
        座標方向とランダム方向を組み合わせた候補集合を生成
        実験結果に基づき、最適なアーム数（0.5x arms）を使用
        """
        num_coord = int(self.coordinate_ratio * self.n_arms)
        num_coord = min(num_coord, self.dim)
        
        # ランダムに座標を選択（LinBandit-BOの特徴）
        idxs = np.random.choice(self.dim, num_coord, replace=False)
        
        # 座標方向の生成
        coords = []
        for i in idxs:
            e = torch.zeros(self.dim, device=self.X.device)
            e[i] = 1.0
            coords.append(e)
            
        coord_arms = torch.stack(coords, 0) if coords else torch.zeros(0, self.dim, device=self.X.device)
        
        # ランダム方向の生成
        num_rand = self.n_arms - num_coord
        rand_arms = torch.randn(num_rand, self.dim, device=self.X.device) if num_rand > 0 else torch.zeros(0, self.dim, device=self.X.device)
        
        if num_rand > 0:
            norms = rand_arms.norm(dim=1, keepdim=True)
            rand_arms = torch.where(norms > 1e-9, rand_arms / norms, 
                                   torch.randn_like(rand_arms) / (torch.randn_like(rand_arms).norm(dim=1,keepdim=True)+1e-9))
            
        return torch.cat([coord_arms, rand_arms], 0)
    
    def select_arm(self, arms_features):
        """Linear UCBによる方向選択"""
        # LinUCBパラメータ
        sigma = 1.0
        L = 1.0
        lambda_reg = 1.0
        delta = 0.1
        S = 1.0
        
        # 現在のパラメータ推定
        A_inv = torch.inverse(self.A)
        theta = A_inv @ self.b
        self.theta_history.append(theta.clone())
        
        # 信頼幅の計算
        current_round_t = max(1, self.total_iterations)
        log_term_numerator = max(1e-9, 1 + (current_round_t - 1) * L**2 / lambda_reg)
        beta_t = (sigma * math.sqrt(self.dim * math.log(log_term_numerator / delta)) + 
                  math.sqrt(lambda_reg) * S)
        
        # UCBスコアの計算
        ucb_scores = []
        for i in range(arms_features.shape[0]):
            x = arms_features[i].view(-1, 1)
            mean = (theta.view(1, -1) @ x).item()
            try:
                var = (x.t() @ A_inv @ x).item()
            except torch.linalg.LinAlgError:
                var = (x.t() @ torch.linalg.pinv(self.A) @ x).item()
                
            ucb_scores.append(mean + beta_t * math.sqrt(max(var, 0)))
            
        return int(np.argmax(ucb_scores))
    
    def propose_new_x(self, direction):
        """選択された方向に沿った最適化"""
        ei = ExpectedImprovement(self.model, best_f=self.best_value, maximize=False)
        
        # 方向に沿った探索範囲の計算
        active_dims_mask = direction.abs() > 1e-9
        if not active_dims_mask.any():
            lb, ub = -1.0, 1.0
        else:
            ratios_lower = (self.bounds[0] - self.best_point) / (direction + 1e-12 * (~active_dims_mask))
            ratios_upper = (self.bounds[1] - self.best_point) / (direction + 1e-12 * (~active_dims_mask))
            
            t_bounds = torch.zeros(self.dim, 2, device=self.X.device)
            t_bounds[:, 0] = torch.minimum(ratios_lower, ratios_upper)
            t_bounds[:, 1] = torch.maximum(ratios_lower, ratios_upper)
            
            lb = -float('inf')
            ub = float('inf')
            for i in range(self.dim):
                if active_dims_mask[i]:
                    lb = max(lb, t_bounds[i, 0].item())
                    ub = min(ub, t_bounds[i, 1].item())
                    
        if lb > ub:
            domain_width = (self.bounds[1, 0] - self.bounds[0, 0]).item()
            lb = -0.1 * domain_width
            ub = 0.1 * domain_width
            
        one_d_bounds = torch.tensor([[lb], [ub]], dtype=torch.float32, device=self.X.device)
        
        def ei_on_line(t_scalar_tensor):
            t_values = t_scalar_tensor.squeeze(-1)
            points_on_line = self.best_point.unsqueeze(0) + t_values.reshape(-1, 1) * direction.unsqueeze(0)
            points_on_line_clamped = torch.clamp(points_on_line, self.bounds[0].unsqueeze(0), self.bounds[1].unsqueeze(0))
            if self.normalize_inputs_for_gp:
                pts = self._to_normalized(points_on_line_clamped)
            else:
                pts = points_on_line_clamped
            return ei(pts.unsqueeze(1))
        
        # 獲得関数の最適化
        cand_t, _ = optimize_acqf(
            ei_on_line,
            bounds=one_d_bounds,
            q=1,
            num_restarts=10,
            raw_samples=100
        )
        
        alpha_star = cand_t.item()
        new_x = self.best_point + alpha_star * direction
        new_x_clamped = torch.clamp(new_x, self.bounds[0], self.bounds[1])
        
        return new_x_clamped
    
    def optimize(self):
        """メインの最適化ループ"""
        print(f"LinBandit-BO開始: {self.dim}次元, アーム数: {self.n_arms}本, 最大{self.n_max}回評価")
        print(f"最適化設定: 勾配ベース報酬 + 0.5x arms (実験結果に基づく)")
        
        # 初期化
        self.initialize()
        n_iter = self.n_initial
        
        while n_iter < self.n_max:
            self.total_iterations += 1
            
            # 探索方向の候補生成
            arms_features = self.generate_arms()
            
            # Linear UCBによる方向選択
            sel_idx = self.select_arm(arms_features)
            direction = arms_features[sel_idx]
            
            # 選択された方向に沿った最適化
            new_x = self.propose_new_x(direction)
            
            # 予測と実際の評価
            with torch.no_grad():
                x_q = self._to_normalized(new_x.unsqueeze(0)) if self.normalize_inputs_for_gp else new_x.unsqueeze(0)
                predicted_mean = self.model.posterior(x_q).mean.squeeze().item()
            actual_y = self.objective_function(new_x.unsqueeze(0)).squeeze().item()
            
            # 報酬の計算（勾配ベース - 実験結果に基づく最適設計）
            if self.normalize_inputs_for_gp:
                new_x_for_grad = self._to_normalized(new_x.clone()).unsqueeze(0)
                new_x_for_grad.requires_grad_(True)
                posterior = self.model.posterior(new_x_for_grad)
                mean_at_new_x = posterior.mean
                mean_at_new_x.sum().backward()
                grad_vector_normed = new_x_for_grad.grad.squeeze(0)
                grad_vector = grad_vector_normed / (self._range + 1e-12)
            else:
                new_x_for_grad = new_x.clone().unsqueeze(0)
                new_x_for_grad.requires_grad_(True)
                posterior = self.model.posterior(new_x_for_grad)
                mean_at_new_x = posterior.mean
                mean_at_new_x.sum().backward()
                grad_vector = new_x_for_grad.grad.squeeze(0)
            
            # 報酬ベクトルを定義（絶対値を取ることで影響の大きさを評価）
            reward_vector = grad_vector.abs()
            
            # 推定リプシッツ定数の更新（下限付き）
            grad_norm = reward_vector.norm().item()
            if grad_norm > self.L_hat:
                self.L_hat = grad_norm
            # 下限を適用
            L_effective = max(self.L_hat, self.L_min)

            # リプシッツ定数でスケーリング（下限によりスパイク抑制）
            scaled_reward_vector = reward_vector / L_effective
            
            # Linear Banditパラメータの更新
            x_arm = direction.view(-1, 1)
            self.A += x_arm @ x_arm.t()
            self.b += scaled_reward_vector  # 勾配ベース報酬（リプシッツスケーリング）
            if self.track_history:
                self.selected_direction_history.append(direction.detach().clone())
                self.reward_history.append(scaled_reward_vector.detach().clone())
            
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
                print(f"  評価回数: {n_iter}/{self.n_max}, 現在の最良値: {self.best_value:.6f}, L_hat: {self.L_hat:.4f}")
                
        print(f"\n最適化完了!")
        print(f"最良値: {self.best_value:.6f}")
        print(f"最良点: {self.best_point[:5]}... (最初の5次元)")
        print(f"最終L_hat: {self.L_hat:.6f}")
        
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
        coordinate_ratio=0.8  # 80%を座標方向、20%をランダム方向
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
