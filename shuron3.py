# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import warnings
import torch
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)


class AxisILCB_DeltaF:
    """
    座標=腕 として、Influential-LCB で軸を選び、固定ステップ α だけその軸に沿って
    1ステップ移動する最小化器。

    - 腕（アーム）: 標準基底 e_i（i 番目の座標軸）
    - 腕選択: Influential-LCB（最後に観測した一歩コストの推定値 - B * 経過ラウンド）
    - 提案点: x_{t+1} = x_t + s_t * α * e_i（符号 s_t は GP 平均で決める。理論部では s_t=+1 版を基準化）
    - 観測損失: L^{(t)} = f(x_{t+1}) - f(x_t)（=「軸 i に一歩進んだときの一歩コスト」）
    - 目的: f(x) を最小化

    重要:
      * 理論移植を明快にするため、損失は EI/NLPD ではなく「実際の一歩コスト Δf」で定義。
      * B は |一手あたり損失が下がり得る最大量| の上界。未知ならオンライン推定可。
      * 実装では s_t（進む符号）を GP 平均で決めるため 1評価/ラウンドを維持。
        （理論部では s_t=+1 の理想化を用い、差分を雑音項に吸収。）
    """

    def __init__(
        self,
        objective_function,
        bounds: torch.Tensor,   # [2, d]（[下限, 上限]）
        n_initial: int = 5,
        n_max: int = 100,
        alpha: float = 0.25,    # 固定ステップ幅（各次元で同一スカラー）
        B: float | None = None, # ILCB の影響上界（未知なら None で推定）
        warmup_per_axis: bool = True,  # 最初に各軸を1回ずつ観測して lhat を埋める
        sign_rule: str = "gp_mean",    # "gp_mean" | "none_plus"
    ):
        self.objective_function = objective_function
        self.bounds = bounds.detach().clone().double()
        self.dim = int(self.bounds.shape[1])

        self.n_initial = int(n_initial)
        self.n_max = int(n_max)
        self.alpha = float(alpha)
        self.B = None if B is None else float(B)
        self.warmup_per_axis = bool(warmup_per_axis)
        self.sign_rule = str(sign_rule)

        # ILCB 状態
        self.c = torch.zeros(self.dim, dtype=torch.long)               # 各腕の「最後の観測からの経過ラウンド数」
        self.lhat = torch.full((self.dim,), float("-inf"), dtype=torch.double)  # 各腕の「最後に観測した一歩コスト（推定値）」
        self.last_loss_per_axis = torch.full((self.dim,), float("nan"), dtype=torch.double)

        # データ/GP
        self.X = None  # [N, d]
        self.Y = None  # [N, 1]
        self.model = None

        # 現在点（常に移動する）
        self.cur_x = None  # [d]
        self.cur_y = None  # float

        # 最良
        self.best_x = None
        self.best_y = None

    # ===== GP 学習 =====
    def _fit_gp(self):
        self.model = SingleTaskGP(self.X, self.Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    # ===== 初期化 =====
    def initialize(self):
        lb, ub = self.bounds[0], self.bounds[1]
        # ランダム初期点を n_initial 個評価 → 最良を現在点に
        X0 = torch.rand(self.n_initial, self.dim, dtype=self.bounds.dtype) * (ub - lb) + lb
        with torch.no_grad():
            y0 = self.objective_function(X0)
        Y0 = y0.reshape(-1, 1).double()

        self.X, self.Y = X0, Y0
        self._fit_gp()

        idx = int(torch.argmin(self.Y).item())
        self.cur_x = self.X[idx].detach().clone()
        self.cur_y = float(self.Y[idx].item())

        self.best_x = self.cur_x.detach().clone()
        self.best_y = float(self.cur_y)

        # ILCB の統計は未観測から開始（lhat=-inf, c=0 のまま）
        # B が None の場合はオンライン推定を有効化
        # 以後のループで warmup_per_axis が True なら各軸一回ずつ観測して lhat を埋める

    # ===== ILCB の腕選択 =====
    def _choose_arm_ilcb(self) -> int:
        # 未観測腕を優先
        unobs = torch.nonzero(torch.isinf(self.lhat), as_tuple=False)
        if unobs.numel() > 0:
            return int(unobs[0].item())
        # 既観測のみなら LCB = lhat - B * c の最小を選ぶ
        B = float(self.B) if self.B is not None else 1.0
        scores = self.lhat - B * self.c.double()
        return int(torch.argmin(scores).item())

    # ===== 進む符号の決め方 =====
    def _decide_sign(self, axis: int) -> int:
        if self.sign_rule == "none_plus":
            return +1  # 理論の理想化：常に+方向へ（実用性は劣る）
        # 既定: GP 平均で x±α e_i を比較して小さい方へ
        with torch.no_grad():
            e = torch.zeros(self.dim, dtype=self.bounds.dtype)
            e[axis] = 1.0
            lb, ub = self.bounds[0], self.bounds[1]
            xp = torch.max(torch.min(self.cur_x + self.alpha * e, ub), lb)
            xm = torch.max(torch.min(self.cur_x - self.alpha * e, ub), lb)
            mp = float(self.model.posterior(xp.unsqueeze(0)).mean.squeeze().item())
            mm = float(self.model.posterior(xm.unsqueeze(0)).mean.squeeze().item())
        return +1 if mp <= mm else -1

    # ===== ILCB 統計の更新 =====
    def _update_ilcb_stats(self, arm: int, loss: float):
        self.lhat[arm] = float(loss)
        self.c += 1
        self.c[arm] = 1

        # B のオンライン推定（同一腕の連続観測などで差分を上方推定）
        if self.B is None and not math.isnan(self.last_loss_per_axis[arm].item()):
            delta = abs(loss - float(self.last_loss_per_axis[arm].item()))
            self.B = delta if (self.B is None) else max(self.B, delta)
        self.last_loss_per_axis[arm] = float(loss)

    # ===== 1 ステップ（座標 i に α だけ進む） =====
    def _take_step(self, axis: int):
        s = self._decide_sign(axis)
        e = torch.zeros(self.dim, dtype=self.bounds.dtype); e[axis] = 1.0
        lb, ub = self.bounds[0], self.bounds[1]
        x_new = torch.max(torch.min(self.cur_x + s * self.alpha * e, ub), lb)

        # 観測（1評価）
        with torch.no_grad():
            y_new = float(self.objective_function(x_new.unsqueeze(0)).item())

        # 観測損失（一歩コスト）= f(x_new) - f(x_cur)
        loss = y_new - float(self.cur_y)

        # ILCB 統計更新
        self._update_ilcb_stats(axis, loss)

        # GP/データ更新
        self.X = torch.cat([self.X, x_new.unsqueeze(0)], dim=0)
        self.Y = torch.cat([self.Y, torch.tensor([[y_new]], dtype=self.bounds.dtype)], dim=0)
        self._fit_gp()

        # 現在点を更新（常に移動）
        self.cur_x = x_new.detach().clone()
        self.cur_y = float(y_new)

        # ベスト更新
        if y_new < self.best_y:
            self.best_y = y_new
            self.best_x = x_new.detach().clone()

    # ===== メインループ =====
    def optimize(self):
        self.initialize()
        n_eval = int(self.X.shape[0])
        rounds_left = self.n_max - n_eval

        # （任意）各軸を1回ずつ試して lhat を初期化（総評価回数の制約内で行う）
        if self.warmup_per_axis:
            for i in range(self.dim):
                if n_eval >= self.n_max: break
                if torch.isinf(self.lhat[i]):
                    self._take_step(i)
                    n_eval += 1

        # 本ループ
        while n_eval < self.n_max:
            arm = self._choose_arm_ilcb()
            self._take_step(arm)
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

    opt = AxisILCB_DeltaF(
        objective_function=styblinski_tang,
        bounds=bounds,
        n_initial=5,
        n_max=60,
        alpha=0.25,          # ステップ幅（箱 [-5,5] なら 0.25~0.5 あたりから）
        B=None,              # 既知なら固定。未知なら None でオンライン推定
        warmup_per_axis=True,
        sign_rule="gp_mean", # GP平均で ± を決定（1評価/ラウンド）
    )
    best_x, best_y = opt.optimize()
    print("best y:", best_y)
    print("best x (first 5 dims):", best_x[:5])
