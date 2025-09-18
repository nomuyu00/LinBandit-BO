#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lengthscale Lower Bound Sweep Experiment.

lengthscale_lower_bound_comparison を拡張し、RBF カーネルの長さスケール下限
l_min を {0.001, 0.01, 0.03, 0.05, 0.1} でグリッドサーチする。
3 つのテスト関数 (Styblinski-Tang / Rastrigin / Ackley) について
各条件 20 ラン × 300 評価で LinBandit-BO を実行し、主要統計と可視化を生成する。
"""

from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib import cm

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

# グローバル設定
OUTPUT_ROOT = os.path.join(
    os.path.dirname(__file__), "output_results_lengthscale_lower_bound_sweep"
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.unicode_minus"] = False

# 和文フォント設定（利用可能なものにフォールバック）
try:
    import japanize_matplotlib  # type: ignore
except ImportError:  # pragma: no cover - フォールバック処理
    if os.name == "nt":
        plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
    else:
        plt.rcParams["font.family"] = [
            "DejaVu Sans",
            "IPAexGothic",
            "IPAGothic",
            "Noto Sans CJK JP",
            "TakaoGothic",
            "sans-serif",
        ]


# PyTorch の float デフォルトを揃える
torch.set_default_dtype(torch.float32)


@dataclass
class ExperimentConfig:
    dim: int = 20
    effective_dims: int = 5
    bounds_low: float = -5.0
    bounds_high: float = 5.0
    n_initial: int = 5
    n_max: int = 300
    coordinate_ratio: float = 0.8
    n_runs: int = 20
    lengthscale_grid: tuple = (0.001, 0.01, 0.03, 0.05, 0.1)
    L_min: float = 0.1
    reward_upper_threshold: float = 0.95

    def __post_init__(self) -> None:
        runs = os.getenv("LS_SWEEP_RUNS")
        if runs is not None:
            self.n_runs = int(runs)
        n_iters = os.getenv("LS_SWEEP_ITERS")
        if n_iters is not None:
            self.n_max = int(n_iters)

    @property
    def bounds_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [[self.bounds_low] * self.dim, [self.bounds_high] * self.dim],
            dtype=torch.float32,
        )


class LengthscaleSweepLinBanditBO:
    """LinBandit-BO 実装 (正規化 + 長さスケール下限 + 計測拡張)。"""

    def __init__(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        bounds: torch.Tensor,
        l_min: float,
        n_initial: int = 5,
        n_max: int = 300,
        coordinate_ratio: float = 0.8,
        n_arms: int | None = None,
        L_min: float = 0.1,
        initial_X: torch.Tensor | None = None,
    ) -> None:
        self.objective_function = objective_function
        self.bounds = bounds.float()
        self.dim = bounds.shape[1]
        self.n_initial = int(n_initial)
        self.n_max = int(n_max)
        self.coordinate_ratio = float(coordinate_ratio)
        self.n_arms = n_arms if n_arms is not None else max(1, self.dim // 2)
        self.L_min = float(L_min)
        self.l_min = float(l_min)

        # 線形バンディットパラメータ
        self.A = torch.eye(self.dim)
        self.b = torch.zeros(self.dim)

        # 初期点
        if initial_X is None:
            self.X = torch.rand(self.n_initial, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        else:
            self.X = initial_X.clone().float()
        self.X = self.X.float()

        # 状態
        self.Y: torch.Tensor | None = None
        self.best_value: float | None = None
        self.best_point: torch.Tensor | None = None
        self.model: SingleTaskGP | None = None
        self.eval_history: List[float] = []
        self.theta_history: List[torch.Tensor] = []
        self.total_iterations = 0

        # 計測用
        self.reward_history: List[np.ndarray] = []
        self.lengthscale_history: List[np.ndarray] = []
        self.direction_history: List[np.ndarray] = []
        self.grad_norm_history: List[float] = []
        self.L_hat_history: List[float] = []
        self.reward_hit_history: List[float] = []

        self._range = (self.bounds[1] - self.bounds[0]).float()
        self.L_hat = max(1.0, self.L_min)

    # --- 内部ユーティリティ -------------------------------------------------
    def _to_normalized(self, X: torch.Tensor) -> torch.Tensor:
        return torch.clamp((X - self.bounds[0]) / self._range, 0.0, 1.0)

    def update_model(self) -> None:
        X_gp = self._to_normalized(self.X)
        base = RBFKernel(
            ard_num_dims=self.X.shape[-1],
            lengthscale_constraint=GreaterThan(self.l_min),
        )
        kernel = ScaleKernel(base).to(X_gp)
        self.model = SingleTaskGP(X_gp, self.Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def initialize(self) -> None:
        with torch.no_grad():
            y_val = self.objective_function(self.X)
        self.Y = y_val.unsqueeze(-1).float()
        self.update_model()

        Xn = self._to_normalized(self.X)
        with torch.no_grad():
            post_mean = self.model.posterior(Xn).mean.squeeze(-1)
        best_idx = post_mean.argmin()
        self.best_value = float(post_mean[best_idx].item())
        self.best_point = self.X[best_idx].clone()
        self.eval_history = [self.best_value] * self.n_initial

    def generate_arms(self) -> torch.Tensor:
        num_coord = min(int(self.coordinate_ratio * self.n_arms), self.dim)
        idxs = np.random.choice(self.dim, num_coord, replace=False) if num_coord > 0 else []

        coord_arms = []
        for idx in idxs:
            e = torch.zeros(self.dim)
            e[idx] = 1.0
            coord_arms.append(e)
        coord_arms = torch.stack(coord_arms) if coord_arms else torch.zeros(0, self.dim)

        num_rand = self.n_arms - coord_arms.shape[0]
        if num_rand > 0:
            rand_arms = torch.randn(num_rand, self.dim)
            norms = rand_arms.norm(dim=1, keepdim=True)
            rand_arms = torch.where(norms > 1e-9, rand_arms / norms, rand_arms)
        else:
            rand_arms = torch.zeros(0, self.dim)

        return torch.cat([coord_arms, rand_arms], dim=0)

    def select_arm(self, arms_features: torch.Tensor) -> int:
        sigma = 1.0
        L = 1.0
        lambda_reg = 1.0
        delta = 0.1
        S = 1.0

        A_inv = torch.inverse(self.A)
        theta = A_inv @ self.b
        self.theta_history.append(theta.clone())

        current_round_t = max(1, self.total_iterations)
        log_term = max(1e-9, 1 + (current_round_t - 1) * L ** 2 / lambda_reg)
        beta_t = sigma * math.sqrt(self.dim * math.log(log_term / delta)) + math.sqrt(lambda_reg) * S

        scores: List[float] = []
        for i in range(arms_features.shape[0]):
            x = arms_features[i].view(-1, 1)
            mean = (theta.view(1, -1) @ x).item()
            try:
                var = (x.t() @ A_inv @ x).item()
            except torch.linalg.LinAlgError:
                var = (x.t() @ torch.linalg.pinv(self.A) @ x).item()
            scores.append(mean + beta_t * math.sqrt(max(var, 0)))
        return int(np.argmax(scores))

    def propose_new_x(self, direction: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        ei = ExpectedImprovement(self.model, best_f=self.best_value, maximize=False)
        active = direction.abs() > 1e-9
        if not active.any():
            lb, ub = -1.0, 1.0
        else:
            ratios_lower = (self.bounds[0] - self.best_point) / (direction + 1e-12 * (~active))
            ratios_upper = (self.bounds[1] - self.best_point) / (direction + 1e-12 * (~active))
            t_bounds = torch.stack([torch.minimum(ratios_lower, ratios_upper), torch.maximum(ratios_lower, ratios_upper)], dim=-1)
            lb = -float("inf")
            ub = float("inf")
            for idx in range(self.dim):
                if active[idx]:
                    lb = max(lb, float(t_bounds[idx, 0]))
                    ub = min(ub, float(t_bounds[idx, 1]))
        if lb > ub:
            width = float(self.bounds[1, 0] - self.bounds[0, 0])
            lb, ub = -0.1 * width, 0.1 * width
        one_d_bounds = torch.tensor([[lb], [ub]], dtype=torch.float32)

        def ei_on_line(t_tensor: torch.Tensor) -> torch.Tensor:
            t_vals = t_tensor.squeeze(-1)
            points = self.best_point.unsqueeze(0) + t_vals.reshape(-1, 1) * direction.unsqueeze(0)
            points = torch.clamp(points, self.bounds[0], self.bounds[1])
            pts_n = self._to_normalized(points)
            return ei(pts_n.unsqueeze(1))

        cand_t, _ = optimize_acqf(
            ei_on_line,
            bounds=one_d_bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
        alpha = cand_t.item()
        new_x = self.best_point + alpha * direction
        return torch.clamp(new_x, self.bounds[0], self.bounds[1])

    def optimize(self) -> None:
        self.initialize()
        n_iter = self.n_initial
        while n_iter < self.n_max:
            self.total_iterations += 1
            arms = self.generate_arms()
            chosen = arms[self.select_arm(arms)]
            self.direction_history.append(chosen.detach().cpu().numpy())

            new_x = self.propose_new_x(chosen)
            new_x_norm = self._to_normalized(new_x.unsqueeze(0))
            with torch.no_grad():
                pred_mean = self.model.posterior(new_x_norm).mean.squeeze().item()
            actual_y = self.objective_function(new_x.unsqueeze(0)).squeeze().item()

            # 勾配計算
            new_x_var = new_x.clone().unsqueeze(0).requires_grad_(True)
            x_norm = self._to_normalized(new_x_var)
            x_norm.retain_grad()
            posterior = self.model.posterior(x_norm)
            posterior.mean.sum().backward()
            grad_normed = x_norm.grad.squeeze(0)
            grad_vector = grad_normed / (self._range + 1e-12)
            reward_vector = grad_vector.abs()
            grad_norm = float(reward_vector.norm().item())
            self.grad_norm_history.append(grad_norm)

            if grad_norm > self.L_hat:
                self.L_hat = grad_norm
            L_effective = max(self.L_hat, self.L_min)
            scaled_reward = reward_vector / L_effective
            hit_rate = float((scaled_reward >= 0.95).float().mean().item())
            self.reward_hit_history.append(hit_rate)
            self.L_hat_history.append(self.L_hat)

            self.reward_history.append(scaled_reward.detach().cpu().numpy())

            self.A += chosen.view(-1, 1) @ chosen.view(1, -1)
            self.b += scaled_reward.detach()

            self.X = torch.cat([self.X, new_x.unsqueeze(0)], dim=0)
            self.Y = torch.cat([
                self.Y,
                torch.tensor([[actual_y]], dtype=torch.float32),
            ], dim=0)
            self.update_model()

            with torch.no_grad():
                Xn_all = self._to_normalized(self.X)
                posterior_mean = self.model.posterior(Xn_all).mean.squeeze(-1)
            best_idx = posterior_mean.argmin()
            self.best_value = float(posterior_mean[best_idx].item())
            self.best_point = self.X[best_idx].clone()
            self.eval_history.append(self.best_value)

            try:
                ls = (
                    self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
                )
                self.lengthscale_history.append(ls.copy())
            except Exception:
                pass

            n_iter += 1


# --- テスト関数 --------------------------------------------------------------

def styblinski_tang_effective(x: torch.Tensor, effective_dims: int = 5) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x_eff = x[..., :effective_dims]
    return 0.5 * torch.sum(x_eff ** 4 - 16.0 * x_eff ** 2 + 5.0 * x_eff, dim=-1)


def rastrigin_effective(x: torch.Tensor, effective_dims: int = 5) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x_eff = x[..., :effective_dims]
    return torch.sum(x_eff ** 2 - 10.0 * torch.cos(2 * math.pi * x_eff) + 10.0, dim=-1)


def ackley_effective(x: torch.Tensor, effective_dims: int = 5) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x_eff = x[..., :effective_dims]
    d = x_eff.shape[-1]
    sum1 = torch.sum(x_eff ** 2, dim=-1)
    sum2 = torch.sum(torch.cos(2 * math.pi * x_eff), dim=-1)
    return -20.0 * torch.exp(-0.2 * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + 20.0 + math.e


TEST_FUNCTIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "Styblinski-Tang": styblinski_tang_effective,
    "Rastrigin": rastrigin_effective,
    "Ackley": ackley_effective,
}

GLOBAL_OPTIMA = {
    "Styblinski-Tang": -39.16599 * 5,
    "Rastrigin": 0.0,
    "Ackley": 0.0,
}


# --- 実験実行ロジック --------------------------------------------------------

def run_lengthscale_sweep_for_function(
    func_name: str,
    objective: Callable[[torch.Tensor], torch.Tensor],
    cfg: ExperimentConfig,
) -> Dict[str, List[Dict[str, object]]]:
    bounds = cfg.bounds_tensor
    algo_labels = [f"l_min={value:.3f}" for value in cfg.lengthscale_grid]
    results: Dict[str, List[Dict[str, object]]] = {label: [] for label in algo_labels}

    for run_idx in range(cfg.n_runs):
        seed = run_idx * 100
        torch.manual_seed(seed)
        np.random.seed(seed)
        initial_X = torch.rand(cfg.n_initial, cfg.dim) * (bounds[1] - bounds[0]) + bounds[0]

        for value, label in zip(cfg.lengthscale_grid, algo_labels):
            torch.manual_seed(seed)
            np.random.seed(seed)
            optimizer = LengthscaleSweepLinBanditBO(
                objective_function=lambda x, f=objective: f(x),
                bounds=bounds,
                l_min=value,
                n_initial=cfg.n_initial,
                n_max=cfg.n_max,
                coordinate_ratio=cfg.coordinate_ratio,
                n_arms=cfg.dim // 2,
                L_min=cfg.L_min,
                initial_X=initial_X,
            )
            optimizer.optimize()
            run_data = {
                "best_value": optimizer.best_value,
                "eval_history": optimizer.eval_history,
                "theta_history": optimizer.theta_history,
                "reward_history": optimizer.reward_history,
                "lengthscale_history": optimizer.lengthscale_history,
                "direction_history": optimizer.direction_history,
                "L_hat_history": optimizer.L_hat_history,
                "grad_norm_history": optimizer.grad_norm_history,
                "reward_hit_history": optimizer.reward_hit_history,
            }
            results[label].append(run_data)
        print(f"\r{func_name}: run {run_idx + 1}/{cfg.n_runs} 完了", end="")
    print()
    return results


# --- 解析ユーティリティ ----------------------------------------------------

def _collect_histories(results: Dict[str, List[Dict[str, object]]], key: str) -> Dict[str, np.ndarray]:
    collected: Dict[str, np.ndarray] = {}
    for alg, runs in results.items():
        sequences = [np.asarray(run[key], dtype=float) for run in runs]
        collected[alg] = np.array(sequences)
    return collected


def save_results_binary(func_name: str, results: Dict[str, List[Dict[str, object]]]) -> None:
    eval_histories = {
        alg: np.array([run["eval_history"] for run in runs], dtype=float)
        for alg, runs in results.items()
    }
    np.save(os.path.join(OUTPUT_ROOT, f"{func_name}_results.npy"), eval_histories, allow_pickle=True)


def export_reward_history(func_name: str, results: Dict[str, List[Dict[str, object]]], cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for alg, runs in results.items():
        for run_idx, run in enumerate(runs):
            for iter_idx, reward_vec in enumerate(run["reward_history"], start=cfg.n_initial):
                for dim_idx, reward in enumerate(reward_vec):
                    rows.append(
                        {
                            "Algorithm": alg,
                            "Run": run_idx,
                            "Iteration": iter_idx,
                            "Dimension": dim_idx,
                            "Reward": float(reward),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUTPUT_ROOT, f"{func_name}_reward_history.csv"),
        index=False,
    )
    summary = (
        df.groupby(["Algorithm", "Dimension"])['Reward']
        .agg(mean="mean", std="std", cumulative="sum")
        .reset_index()
    )
    summary.to_csv(
        os.path.join(OUTPUT_ROOT, f"{func_name}_dimension_summary.csv"),
        index=False,
    )
    return df


def export_metric_history(func_name: str, results: Dict[str, List[Dict[str, object]]], cfg: ExperimentConfig) -> None:
    def _flatten(metric_key: str, filename: str) -> None:
        rows = []
        for alg, runs in results.items():
            for run_idx, run in enumerate(runs):
                history = run.get(metric_key, [])
                for iter_idx, value in enumerate(history, start=cfg.n_initial):
                    rows.append(
                        {
                            "Algorithm": alg,
                            "Run": run_idx,
                            "Iteration": iter_idx,
                            metric_key: float(value),
                        }
                    )
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTPUT_ROOT, f"{func_name}_{filename}.csv"),
            index=False,
        )

    _flatten("L_hat_history", "lhat_history")
    _flatten("grad_norm_history", "grad_norms")
    _flatten("reward_hit_history", "r_upper_hit_rate")


# --- 可視化 -----------------------------------------------------------------

def plot_convergence(func_name: str, results: Dict[str, List[Dict[str, object]]], cfg: ExperimentConfig) -> None:
    colors = cm.get_cmap("viridis", len(results))
    iterations = np.arange(1, cfg.n_max + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 収束履歴
    ax = axes[0, 0]
    for idx, (alg, runs) in enumerate(results.items()):
        histories = np.array([run["eval_history"] for run in runs], dtype=float)
        mean = histories.mean(axis=0)
        std = histories.std(axis=0)
        ax.plot(iterations, mean, color=colors(idx), label=alg, linewidth=2)
        ax.fill_between(iterations, mean - std, mean + std, color=colors(idx), alpha=0.15)
    ax.axhline(GLOBAL_OPTIMA[func_name], color="black", linestyle="--", linewidth=1.2)
    ax.set_title(f"{func_name}: 収束履歴 (平均±標準偏差)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 最終値箱ひげ
    ax = axes[0, 1]
    final_box_data = [
        [run["best_value"] for run in runs]
        for runs in results.values()
    ]
    ax.boxplot(final_box_data, labels=list(results.keys()), patch_artist=True)
    ax.axhline(GLOBAL_OPTIMA[func_name], color="black", linestyle="--", linewidth=1.2)
    ax.set_title(f"{func_name}: 最終最良値の分布")
    ax.set_ylabel("Best Value")
    ax.grid(True, alpha=0.3)

    # |theta| ヒートマップ
    ax = axes[1, 0]
    theta_means = []
    for runs in results.values():
        final_thetas = []
        for run in runs:
            if run["theta_history"]:
                final_thetas.append(run["theta_history"][-1].abs().cpu().numpy())
        theta_means.append(np.mean(final_thetas, axis=0))
    im = ax.imshow(theta_means, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(results)))
    ax.set_yticklabels(list(results.keys()))
    ax.set_xticks(np.arange(cfg.dim))
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Algorithm")
    ax.set_title(f"{func_name}: |theta| 平均 (最終イテレーション)")
    fig.colorbar(im, ax=ax, orientation="vertical")

    # 平均長さスケール
    ax = axes[1, 1]
    means = []
    stds = []
    for runs in results.values():
        ls_values = []
        for run in runs:
            for arr in run["lengthscale_history"]:
                ls_values.extend(arr.tolist())
        ls_arr = np.array(ls_values, dtype=float) if ls_values else np.array([np.nan])
        means.append(np.nanmean(ls_arr))
        stds.append(np.nanstd(ls_arr))
    ax.errorbar(list(results.keys()), means, yerr=stds, fmt="o", capsize=5)
    ax.set_title(f"{func_name}: 長さスケール平均±STD")
    ax.set_ylabel("Lengthscale")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, f"{func_name}_comparison.png"), dpi=300)
    plt.close(fig)


def plot_reward_analysis(func_name: str, df_rewards: pd.DataFrame, results: Dict[str, List[Dict[str, object]]], cfg: ExperimentConfig) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 累積報酬ヒートマップ
    ax = axes[0, 0]
    pivot = df_rewards.pivot_table(
        index="Algorithm", columns="Dimension", values="Reward", aggfunc="sum"
    ).reindex(results.keys())
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title(f"{func_name}: 累積報酬ヒートマップ")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Algorithm")
    fig.colorbar(im, ax=ax)

    # 有効次元の平均報酬推移
    ax = axes[0, 1]
    for alg in results.keys():
        subset = df_rewards[(df_rewards["Algorithm"] == alg) & (df_rewards["Dimension"] < cfg.effective_dims)]
        mean_series = subset.groupby("Iteration")["Reward"].mean()
        ax.plot(mean_series.index, mean_series.values, label=alg)
    ax.set_title(f"{func_name}: 有効次元平均報酬推移")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward (mean)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # r>=0.95 到達率
    ax = axes[1, 0]
    for alg, runs in results.items():
        rates = np.array([run["reward_hit_history"] for run in runs])
        if rates.size == 0:
            continue
        iterations = np.arange(cfg.n_initial, cfg.n_initial + rates.shape[1])
        ax.plot(iterations, rates.mean(axis=0), label=alg)
    ax.set_title(f"{func_name}: r≥{cfg.reward_upper_threshold:.2f} 到達率")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 生勾配ノルム分布
    ax = axes[1, 1]
    sns_rows = []
    for alg, runs in results.items():
        for run in runs:
            for value in run["grad_norm_history"]:
                sns_rows.append({"Algorithm": alg, "GradNorm": value})
    if sns_rows:
        sns_df = pd.DataFrame(sns_rows)
        sns_df.boxplot(column="GradNorm", by="Algorithm", ax=ax)
        ax.set_title(f"{func_name}: 勾配ノルム分布")
        ax.set_ylabel("‖∇μ‖")
        ax.set_xlabel("Algorithm")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, f"{func_name}_reward_analysis.png"), dpi=300)
    plt.close(fig)


def plot_diagnostics(func_name: str, results: Dict[str, List[Dict[str, object]]], cfg: ExperimentConfig) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # L_hat 推移
    ax = axes[0]
    for alg, runs in results.items():
        arr = np.array([run["L_hat_history"] for run in runs], dtype=float)
        if arr.size == 0:
            continue
        iterations = np.arange(cfg.n_initial, cfg.n_initial + arr.shape[1])
        ax.plot(iterations, arr.mean(axis=0), label=alg)
    ax.set_title("L_hat 平均推移")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L_hat")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 最終 L_hat 箱ひげ
    ax = axes[1]
    final_vals = []
    labels = []
    for alg, runs in results.items():
        values = [run["L_hat_history"][-1] for run in runs if run["L_hat_history"]]
        if values:
            final_vals.append(values)
            labels.append(alg)
    if final_vals:
        ax.boxplot(final_vals, labels=labels, patch_artist=True)
        ax.set_title("最終 L_hat 分布")
        ax.set_ylabel("L_hat")
        ax.grid(True, alpha=0.3)

    # 勾配ノルム KDE
    ax = axes[2]
    for alg, runs in results.items():
        all_norms = np.concatenate([run["grad_norm_history"] for run in runs])
        if all_norms.size:
            ax.hist(all_norms, bins=40, alpha=0.35, label=alg, density=True)
    ax.set_title("‖∇μ‖ ヒストグラム")
    ax.set_xlabel("Norm")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, f"{func_name}_diagnostics.png"), dpi=300)
    plt.close(fig)


# --- 実験メイン --------------------------------------------------------------

def run_all_experiments() -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    cfg = ExperimentConfig()
    all_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    whitelist = os.getenv("LS_SWEEP_FUNCS")
    target_funcs = TEST_FUNCTIONS
    if whitelist:
        allowed = {name.strip() for name in whitelist.split(",") if name.strip()}
        target_funcs = {k: v for k, v in TEST_FUNCTIONS.items() if k in allowed}
    for func_name, func in target_funcs.items():
        results = run_lengthscale_sweep_for_function(func_name, func, cfg)
        save_results_binary(func_name, results)
        df_rewards = export_reward_history(func_name, results, cfg)
        export_metric_history(func_name, results, cfg)
        plot_convergence(func_name, results, cfg)
        plot_reward_analysis(func_name, df_rewards, results, cfg)
        plot_diagnostics(func_name, results, cfg)
        all_results[func_name] = results
    with open(os.path.join(OUTPUT_ROOT, "experiment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg.__dict__}, f, ensure_ascii=False, indent=2)
    return all_results


if __name__ == "__main__":
    run_all_experiments()
