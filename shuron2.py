# -*- coding: utf-8 -*-
import math
import warnings
import torch
from typing import Optional
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)


class SimpleAxisUCB_NLPD_EI:
    """
    各次元=アーム(ワンホット)とする単純UCB + GP(EI) 最小化器（簡潔版）

    - アーム: 標準基底 e_i（軸 i のみ動かす 1D 探索）
    - アーム選択: 単純UCB (平均報酬 + c * sqrt(2 log t / n_i))
    - 提案点: 選択軸に沿って EI を 1D 勾配上昇で最大化（グリッド不使用）
    - 報酬: NLPD を「選択アーム（次元）」の平均にのみ反映
    - 目的: f(x) を最小化
    """

    def __init__(
        self,
        objective_function,
        bounds: torch.Tensor,   # 形状 [2, d] （[下限, 上限]）
        n_initial: int = 5,
        n_max: int = 100,
        ucb_c: float = 1.0,     # UCB の探索係数
        ei_steps: int = 30,     # 1D 勾配上昇のステップ数
        ei_restarts: int = 3,   # 1D 勾配上昇のリスタート回数
        initial_X: Optional[torch.Tensor] = None,  # 任意の初期点（[n_initial, d]）
    ):
        self.objective_function = objective_function
        self.bounds = bounds.detach().clone().double()
        self.dim = int(self.bounds.shape[1])

        self.n_initial = int(n_initial)
        self.n_max = int(n_max)
        self.ucb_c = float(ucb_c)
        self.ei_steps = int(ei_steps)
        self.ei_restarts = int(ei_restarts)

        # アーム統計（UCB 用）
        self.arm_counts = torch.zeros(self.dim, dtype=torch.long)
        self.arm_means = torch.zeros(self.dim, dtype=torch.double)  # 平均報酬（=NLPD の平均）

        # データ/モデル
        self.X = None  # [N, d]
        self.Y = None  # [N, 1]
        self.model = None
        # 入力正規化（[0,1]^d）と出力標準化を適用するための状態
        self._range = (self.bounds[1] - self.bounds[0]).clamp_min(1e-12)
        self.X_norm = None  # 正規化済み入力（GP用）

        # ベスト
        self.best_x = None  # [d]
        self.best_y = None  # float
        self.initial_X = initial_X

    # ===== GP 学習 =====
    def _to_normalized(self, X: torch.Tensor) -> torch.Tensor:
        return torch.clamp((X - self.bounds[0]) / self._range, 0.0, 1.0)

    def _fit_gp(self):
        # 入力は正規化、出力は標準化（Standardize）
        X_gp = self.X_norm if self.X_norm is not None else self._to_normalized(self.X)
        self.model = SingleTaskGP(X_gp, self.Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    # ===== UCB（UCB1 風）でアーム選択 =====
    def _choose_arm_ucb(self) -> int:
        # 未試行アームを優先
        untried = torch.nonzero(self.arm_counts == 0, as_tuple=False)
        if untried.numel() > 0:
            return int(untried[0].item())

        t = int(self.arm_counts.sum().item()) + 1  # 総試行回数（1始まり）
        counts = self.arm_counts.double()
        bonus = self.ucb_c * torch.sqrt(2.0 * torch.log(torch.tensor(float(t))) / counts)
        scores = self.arm_means + bonus
        return int(torch.argmax(scores).item())

    # ===== 軸 i に沿って EI を 1D 勾配上昇で最大化（グリッド不使用） =====
    def _maximize_ei_along_axis(self, axis: int) -> torch.Tensor:
        # best_f はモデルの出力スケールに合わせる（Standardize を考慮）
        if getattr(self.model, "outcome_transform", None) is not None:
            try:
                y_t, _ = self.model.outcome_transform(self.Y)
                best_f_val = float(y_t.min().item())
            except Exception:
                best_f_val = float(self.Y.min().item())
        else:
            best_f_val = float(self.Y.min().item())
        ei = ExpectedImprovement(self.model, best_f=best_f_val, maximize=False)
        lb, ub = self.bounds[0], self.bounds[1]
        lb_i = float(lb[axis].item())
        ub_i = float(ub[axis].item())
        width = max(ub_i - lb_i, 1e-9)

        # 単位ベクトル e_i（double）
        e = torch.zeros(self.dim, dtype=self.bounds.dtype)
        e[axis] = 1.0

        def ei_of_t(t_scalar: torch.Tensor) -> torch.Tensor:
            # x(t) = best_x + (t - best_x_i) * e_i （軸 i だけ変える）
            x = self.best_x + (t_scalar - self.best_x[axis]) * e
            x = torch.max(torch.min(x, ub), lb)  # 箱内にクランプ
            x_n = self._to_normalized(x)
            return ei(x_n.unsqueeze(0)).squeeze()

        # いくつかの初期値から勾配上昇（簡潔な実装）
        seeds = [
            float(self.best_x[axis].item()),             # 現在の座標
            0.25 * lb_i + 0.75 * ub_i,                   # 上寄り
            0.75 * lb_i + 0.25 * ub_i,                   # 下寄り
        ][: self.ei_restarts]

        best_val = -float("inf")
        best_t = None

        for s in seeds:
            t = torch.tensor(s, dtype=self.bounds.dtype, requires_grad=True)
            lr = 0.2 * width  # 区間幅に比例した素朴な学習率
            for _ in range(self.ei_steps):
                val = ei_of_t(t)
                (-val).backward()  # 最大化したいので負号をかけて最小化の勾配に
                with torch.no_grad():
                    grad = -t.grad  # d(EI)/dt
                    t += lr * grad
                    # 区間に射影
                    if t.item() < lb_i:
                        t.copy_(torch.tensor(lb_i, dtype=self.bounds.dtype))
                    elif t.item() > ub_i:
                        t.copy_(torch.tensor(ub_i, dtype=self.bounds.dtype))
                    t.grad.zero_()
                    lr *= 0.9  # 緩やかに減衰
            with torch.no_grad():
                v = float(ei_of_t(t).item())
                if v > best_val:
                    best_val = v
                    best_t = float(t.item())

        # 最良 t での x を返す
        x_new = self.best_x + (torch.tensor(best_t, dtype=self.bounds.dtype) - self.best_x[axis]) * e
        return torch.max(torch.min(x_new, ub), lb)

    # ===== NLPD（Negative Log Predictive Density） =====
    def _nlpd(self, x: torch.Tensor, y_actual: float) -> float:
        with torch.no_grad():
            x_n = self._to_normalized(x)
            post = self.model.posterior(x_n.unsqueeze(0))  # p(f(x)|D)（正規化入力）
            mu = float(post.mean.squeeze().item())
            var = float(post.variance.squeeze().item())
            try:
                noise_var = float(self.model.likelihood.noise.mean().item())
            except Exception:
                noise_var = 1e-6
        sigma2 = max(var + noise_var, 1e-12)
        resid2 = (y_actual - mu) ** 2
        return 0.5 * math.log(2.0 * math.pi * sigma2) + 0.5 * (resid2 / sigma2)

    # ===== 初期化 =====
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

    # ===== メインループ =====
    def optimize(self):
        self.initialize()
        n_eval = int(self.X.shape[0])

        while n_eval < self.n_max:
            # 1) UCB でアーム（軸）選択
            arm = self._choose_arm_ucb()

            # 2) 選択軸に沿って EI を勾配上昇で最大化 → 提案点
            x_new = self._maximize_ei_along_axis(arm)

            # 3) 観測（最小化）
            with torch.no_grad():
                y_new = float(self.objective_function(x_new.unsqueeze(0)).item())

            # 4) 報酬（NLPD）を「選んだアーム」にのみ付与（モデルは観測前のもの）
            reward = float(self._nlpd(x_new, y_new))
            n_i = int(self.arm_counts[arm].item())
            mu_i = float(self.arm_means[arm].item())
            self.arm_means[arm] = (mu_i * n_i + reward) / (n_i + 1)
            self.arm_counts[arm] = n_i + 1

            # 5) データ/GP 更新
            self.X = torch.cat([self.X, x_new.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, torch.tensor([[y_new]], dtype=self.bounds.dtype)], dim=0)
            self.X_norm = self._to_normalized(self.X)
            self._fit_gp()

            # 6) ベスト更新
            if y_new < self.best_y:
                self.best_y = y_new
                self.best_x = x_new.detach().clone()

            n_eval += 1

        return self.best_x, self.best_y


# ===== デモ（必要に応じて実行） =====
if __name__ == "__main__":
    # Styblinski–Tang（最小値 = -39.16599 * d）
    def styblinski_tang(x: torch.Tensor) -> torch.Tensor:
        z = x
        return 0.5 * torch.sum(z**4 - 16.0 * z**2 + 5.0 * z, dim=-1)

    d = 10
    bounds = torch.tensor([[-5.0] * d, [5.0] * d], dtype=torch.double)

    opt = SimpleAxisUCB_NLPD_EI(
        objective_function=styblinski_tang,
        bounds=bounds,
        n_initial=5,
        n_max=60,
        ucb_c=1.0,      # 探索を強めるなら 1.5~2.0 など
        ei_steps=30,    # 勾配上昇のステップ数
        ei_restarts=3,  # リスタート数
    )
    best_x, best_y = opt.optimize()
    print("best y:", best_y)
    print("best x (first 5 dims):", best_x[:5])
