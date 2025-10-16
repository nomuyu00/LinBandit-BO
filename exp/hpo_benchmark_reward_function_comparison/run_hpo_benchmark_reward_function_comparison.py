#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HPO Benchmark: Reward Function Comparison with Continuous-Fixed Direction.

目的:
- 方向選択は Continuous-Fixed（固定点反復で (★) 停留条件を満たす方向）に固定。
- 報酬設計は reward_function_comparison で比較した4種を比較:
  - Gradient(Lmin=0.10)
  - NLPD*dir
  - NLPD*w_dir
  - NLPD*w_grad

対象:
- exp/hpo_benchmark_directional_reward と同じ HPO ターゲット（RF×digits / SVM×breast_cancer / NN×wine /（任意で XGB×california_housing））。

実験フォーマット（出力・図・CSV）は hpo_benchmark_directional_reward に準拠。
"""

from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib import cm

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from sklearn.datasets import load_digits, load_wine, load_breast_cancer, fetch_california_housing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# 出力ディレクトリ
OUTPUT_ROOT = os.path.join(
    os.path.dirname(__file__), "output_results_hpo_benchmark_reward_function_comparison"
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.unicode_minus"] = False

try:
    import japanize_matplotlib  # type: ignore
except ImportError:  # pragma: no cover
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


torch.set_default_dtype(torch.float64)


# ---------------------------- Utils ----------------------------

class EMA:
    def __init__(self, alpha: float = 0.1, eps: float = 1e-8) -> None:
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._m: float | None = None

    def update(self, x: float) -> float:
        if self._m is None:
            self._m = float(x)
        else:
            self._m = self.alpha * float(x) + (1.0 - self.alpha) * self._m
        return self._m

    @property
    def value(self) -> float:
        return float(self._m if self._m is not None else self.eps)


# ---------------------------- BO with Continuous-Fixed Direction ----------------------------

class RewardFunctionFixedDirectionBO:
    def __init__(
        self,
        objective_function: Callable[[torch.Tensor], torch.Tensor],
        bounds: torch.Tensor,
        reward_mode: str,
        n_initial: int = 5,
        n_max: int = 200,
        L_min: float = 0.1,
        l_min_floor: float = 0.10,
        initial_X: torch.Tensor | None = None,
    ) -> None:
        self.objective_function = objective_function
        self.bounds = bounds.to(dtype=torch.get_default_dtype())
        self.dim = bounds.shape[1]
        self.n_initial = int(n_initial)
        self.n_max = int(n_max)
        self.reward_mode = str(reward_mode)
        self.L_min = float(L_min)
        self.l_min = float(l_min_floor)

        # 線形バンディット
        self.A = torch.eye(self.dim, dtype=torch.get_default_dtype())
        self.b = torch.zeros(self.dim, dtype=torch.get_default_dtype())

        # 初期点
        if initial_X is None:
            self.X = torch.rand(self.n_initial, self.dim, dtype=torch.get_default_dtype()) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        else:
            self.X = initial_X.clone().to(dtype=torch.get_default_dtype())

        # 状態
        self.Y: torch.Tensor | None = None
        self.best_value: float | None = None
        self.best_point: torch.Tensor | None = None
        self.model: SingleTaskGP | None = None
        self.eval_history: List[float] = []
        self.theta_history: List[torch.Tensor] = []
        self.total_iterations = 0

        # ログ
        self.reward_history: List[np.ndarray] = []
        self.lengthscale_history: List[np.ndarray] = []
        self.direction_history: List[np.ndarray] = []
        self.grad_norm_history: List[float] = []
        self.L_hat_history: List[float] = []

        # 勾配スケール/正規化
        self._range = (self.bounds[1] - self.bounds[0]).to(dtype=torch.get_default_dtype())
        self.L_hat = max(1.0, self.L_min)
        self.ema_nlpd = EMA(alpha=0.1)

    # ---- GP helpers ----
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
        # 長さ尺度の記録（平均）
        try:
            ls = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
            self.lengthscale_history.append(ls.copy())
        except Exception:
            pass

    def initialize(self) -> None:
        with torch.no_grad():
            y_val = self.objective_function(self.X)
        self.Y = y_val.unsqueeze(-1).to(dtype=torch.get_default_dtype())
        self.update_model()
        Xn = self._to_normalized(self.X)
        with torch.no_grad():
            post_mean = self.model.posterior(Xn).mean.squeeze(-1)
        best_idx = post_mean.argmin()
        self.best_value = float(post_mean[best_idx].item())
        self.best_point = self.X[best_idx].clone()
        self.eval_history = [self.best_value] * self.n_initial

    # ---- UCB radius ----
    def _compute_beta_t(self) -> float:
        sigma = 1.0
        L = 1.0
        lambda_reg = 1.0
        delta = 0.1
        S = 1.0
        current_round_t = max(1, self.total_iterations)
        log_term = max(1e-9, 1 + (current_round_t - 1) * L ** 2 / lambda_reg)
        beta_t = sigma * math.sqrt(self.dim * math.log(log_term / delta)) + math.sqrt(lambda_reg) * S
        return float(beta_t)

    # ---- Direction selection: Continuous-Fixed ----
    def _select_direction_continuous_fixed(self) -> torch.Tensor:
        A = 0.5 * (self.A + self.A.t())
        A_inv = torch.inverse(A)
        theta_hat = A_inv @ self.b
        beta_t = self._compute_beta_t()
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

    # ---- 1D EI line-search on direction (grid + local refine) ----
    def propose_new_x(self, direction: torch.Tensor) -> torch.Tensor:
        assert self.model is not None and self.best_point is not None
        ei = ExpectedImprovement(self.model, best_f=self.best_value, maximize=False)
        active = direction.abs() > 1e-9
        if not active.any():
            lb, ub = -1.0, 1.0
        else:
            ratios_lower = (self.bounds[0] - self.best_point) / (direction + 1e-12 * (~active))
            ratios_upper = (self.bounds[1] - self.best_point) / (direction + 1e-12 * (~active))
            t_bounds = torch.stack([torch.minimum(ratios_lower, ratios_upper), torch.maximum(ratios_lower, ratios_upper)], dim=-1)
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
        pts_n = self._to_normalized(pts)
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
                pts_l_n = self._to_normalized(pts_l)
                with torch.no_grad():
                    vals_l = ei(pts_l_n.unsqueeze(1)).view(-1)
                mask_l = torch.isfinite(vals_l)
                if mask_l.any():
                    vals_l[~mask_l] = -float("inf")
                    j = int(torch.argmax(vals_l).item())
                    alpha_star = float(t_local[j].item())
        x_new = self.best_point + alpha_star * direction
        return torch.clamp(x_new, self.bounds[0], self.bounds[1])

    # ---- Grad / NLPD helpers ----
    def _compute_gradient_vector(self, x_new: torch.Tensor) -> torch.Tensor:
        x_var = x_new.clone().unsqueeze(0).requires_grad_(True)
        x_norm = self._to_normalized(x_var)
        x_norm.retain_grad()
        posterior = self.model.posterior(x_norm)  # type: ignore[union-attr]
        posterior.mean.sum().backward()
        grad_normed = x_norm.grad.squeeze(0)
        grad_vec = grad_normed / (self._range + 1e-12)
        return grad_vec

    def _compute_nlpd_reward(self, y_actual: float, mu_pred: float, var_pred: float, noise_var: float) -> float:
        eps = 1e-9
        sigma2 = max(float(var_pred + noise_var), eps)
        resid2 = float((y_actual - mu_pred) ** 2)
        nlpd = 0.5 * math.log(2.0 * math.pi * sigma2) + 0.5 * (resid2 / sigma2)
        ema = self.ema_nlpd.value
        r_scaled = nlpd / max(ema, eps)
        self.ema_nlpd.update(nlpd)
        return max(0.0, min(1.0, r_scaled))

    # ---- Main loop ----
    def optimize(self) -> None:
        self.initialize()
        n_iter = self.n_initial
        while n_iter < self.n_max:
            self.total_iterations += 1
            direction = self._select_direction_continuous_fixed()
            direction = direction / (direction.norm() + 1e-12)
            self.direction_history.append(direction.detach().cpu().numpy())

            new_x = self.propose_new_x(direction)

            # 予測と観測
            x_norm = self._to_normalized(new_x.unsqueeze(0))
            with torch.no_grad():
                post = self.model.posterior(x_norm)  # type: ignore[union-attr]
                mu_pred = float(post.mean.squeeze().item())
                var_pred = float(post.variance.squeeze().item())
            y_actual = float(self.objective_function(new_x.unsqueeze(0)).squeeze().item())

            # 勾配と L_hat
            grad_vec = self._compute_gradient_vector(new_x)
            grad_norm = float(grad_vec.abs().norm().item())
            if grad_norm > self.L_hat:
                self.L_hat = grad_norm
            self.L_hat_history.append(self.L_hat)
            self.grad_norm_history.append(grad_norm)

            # 観測ノイズ推定
            try:
                noise_var = float(self.model.likelihood.noise.mean().item())  # type: ignore[union-attr]
            except Exception:
                noise_var = 1e-6

            # 報酬ベクトルの構成
            eps = 1e-12
            if self.reward_mode == "Gradient(Lmin=0.10)":
                L_eff = max(self.L_hat, self.L_min)
                reward_vec = torch.clamp(grad_vec.abs() / L_eff, 0.0, 1.0)
            else:
                r_scalar = self._compute_nlpd_reward(y_actual, mu_pred, var_pred, noise_var)
                if self.reward_mode == "NLPD*dir":
                    reward_vec = r_scalar * direction
                elif self.reward_mode == "NLPD*w_dir":
                    w = direction.abs() / (direction.abs().sum() + eps)
                    reward_vec = r_scalar * w
                elif self.reward_mode == "NLPD*w_grad":
                    gabs = grad_vec.abs()
                    w = gabs / (gabs.sum() + eps)
                    reward_vec = r_scalar * w
                else:
                    raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

            # LinUCB の A, b を更新
            x_arm = direction.view(-1, 1)
            self.A += x_arm @ x_arm.t()
            self.b += reward_vec.detach()

            # ログ
            reward_log_vec = reward_vec.abs().detach().cpu().numpy()
            self.reward_history.append(reward_log_vec)
            self.theta_history.append(torch.inverse(self.A) @ self.b)

            # GP に追加学習
            self.X = torch.cat([self.X, new_x.unsqueeze(0)], dim=0)
            self.Y = torch.cat([self.Y, torch.tensor([[y_actual]], dtype=torch.get_default_dtype())], dim=0)
            self.update_model()

            # 最良値更新（GP 予測値の最小）
            with torch.no_grad():
                Xn_all = self._to_normalized(self.X)
                posterior_mean = self.model.posterior(Xn_all).mean.squeeze(-1)  # type: ignore[union-attr]
            best_idx = posterior_mean.argmin()
            self.best_value = float(posterior_mean[best_idx].item())
            self.best_point = self.X[best_idx].clone()
            self.eval_history.append(self.best_value)
            n_iter += 1


# ---------------------------- HPO Benchmark Task ----------------------------

@dataclass
class HPOTask:
    model_type: str
    dataset_name: str
    task_type: str  # 'classification' or 'regression'


def build_hpo_task(model_type: str, dataset_name: str) -> Tuple[HPOTask, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    # dataset
    if dataset_name == 'digits':
        data = load_digits(); task_type = 'classification'
    elif dataset_name == 'wine':
        data = load_wine(); task_type = 'classification'
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer(); task_type = 'classification'
    elif dataset_name == 'california_housing':
        data = fetch_california_housing(); task_type = 'regression'
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    X, y = data.data, data.target

    # parameter space (match notebook practical ranges; no extra 10** transform)
    if model_type == 'xgboost':
        if not HAS_XGB:
            raise RuntimeError('xgboost not available')
        param_names = ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','gamma']
        bounds = torch.tensor([[10,300],[1,10],[0.01,1.0],[0.5,1.0],[0.5,1.0],[0.0,1.0]], dtype=torch.get_default_dtype()).T
    elif model_type == 'random_forest':
        param_names = ['n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features']
        bounds = torch.tensor([[10,300],[1,30],[2,20],[1,20],[0.1,1.0]], dtype=torch.get_default_dtype()).T
    elif model_type == 'svm':
        param_names = ['C','gamma']
        bounds = torch.tensor([[0.001,1000],[0.001,10]], dtype=torch.get_default_dtype()).T
    elif model_type == 'neural_network':
        param_names = ['hidden_layer_1','hidden_layer_2','learning_rate','alpha','batch_size']
        bounds = torch.tensor([[10,200],[10,200],[0.0001,0.1],[0.0001,0.1],[16,256]], dtype=torch.get_default_dtype()).T
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    dim = len(param_names)
    bounds = bounds.to(dtype=torch.get_default_dtype())

    # objective wrapper
    def objective(x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        def to_params(vec: np.ndarray) -> Dict[str, object]:
            v = vec.copy()
            params: Dict[str, object] = {}
            for i, name in enumerate(param_names):
                val = v[i]
                if name in ['n_estimators','max_depth','min_samples_split','min_samples_leaf','hidden_layer_1','hidden_layer_2','batch_size']:
                    val = int(round(val))
                params[name] = val
            return params

        def eval_single(vec: np.ndarray) -> float:
            params = to_params(vec)
            if model_type == 'xgboost':
                if not HAS_XGB:
                    return 1.0
                if task_type == 'classification':
                    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', random_state=42)
                    pipe = model
                else:
                    model = xgb.XGBRegressor(**params, random_state=42)
                    pipe = model
            elif model_type == 'random_forest':
                if task_type == 'classification':
                    model = RandomForestClassifier(**params, random_state=42)
                else:
                    model = RandomForestRegressor(**params, random_state=42)
                pipe = model
            elif model_type == 'svm':
                if task_type == 'classification':
                    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(**params, random_state=42))])
                else:
                    pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(**params))])
            elif model_type == 'neural_network':
                if task_type == 'classification':
                    pipe = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(params['hidden_layer_1'], params['hidden_layer_2']), learning_rate_init=params['learning_rate'], alpha=params['alpha'], batch_size=params['batch_size'], max_iter=200, random_state=42))])
                else:
                    pipe = Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(params['hidden_layer_1'], params['hidden_layer_2']), learning_rate_init=params['learning_rate'], alpha=params['alpha'], batch_size=params['batch_size'], max_iter=200, random_state=42))])
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            try:
                if task_type == 'classification':
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', error_score=np.nan)
                else:
                    scores = cross_val_score(pipe, X, y, cv=cv, scoring='r2', error_score=np.nan)
                m = float(np.nanmean(scores))
                if not np.isfinite(m):
                    return 1.0
                return -m
            except Exception as e:
                print('Eval error:', e)
                return 1.0

        if x.ndim == 2:
            xs = x.detach().cpu().numpy()
            vals = [eval_single(xs[i]) for i in range(xs.shape[0])]
            return torch.tensor(vals, dtype=torch.get_default_dtype())
        else:
            val = eval_single(x.detach().cpu().numpy())
            return torch.tensor(val, dtype=torch.get_default_dtype())

    task = HPOTask(model_type=model_type, dataset_name=dataset_name, task_type=task_type)
    return task, bounds, objective


# ---------------------------- Experiment Harness ----------------------------

@dataclass
class ExperimentConfig:
    n_initial: int = 5
    n_max: int = 200
    n_runs: int = 10
    L_min: float = 0.1
    l_min_floor: float = 0.10
    reward_algs: tuple = (
        "Gradient(Lmin=0.10)",
        "NLPD*dir",
        "NLPD*w_dir",
        "NLPD*w_grad",
    )

    def __post_init__(self) -> None:
        runs = os.getenv("AS_RUNS") or os.getenv("RF_RUNS")
        if runs is not None:
            self.n_runs = int(runs)
        iters = os.getenv("AS_ITERS") or os.getenv("RF_ITERS")
        if iters is not None:
            self.n_max = int(iters)
        whitelist = os.getenv("RF_REWARD_ALGS")
        if whitelist:
            allowed = [s.strip() for s in whitelist.split(",") if s.strip()]
            self.reward_algs = tuple([a for a in self.reward_algs if a in allowed])


TARGETS_DEFAULT: List[Tuple[str, str]] = [
    ("random_forest", "digits"),
    ("svm", "breast_cancer"),
    ("neural_network", "wine"),
]
if HAS_XGB:
    TARGETS_DEFAULT.append(("xgboost", "california_housing"))


def run_for_target(model_type: str, dataset_name: str, cfg: ExperimentConfig) -> Dict[str, List[Dict[str, object]]]:
    task, bounds, objective = build_hpo_task(model_type, dataset_name)
    dim = bounds.shape[1]
    algo_labels = list(cfg.reward_algs)
    results: Dict[str, List[Dict[str, object]]] = {label: [] for label in algo_labels}
    for run_idx in range(cfg.n_runs):
        seed = run_idx * 100
        torch.manual_seed(seed); np.random.seed(seed)
        initial_X = torch.rand(cfg.n_initial, dim, dtype=torch.get_default_dtype()) * (bounds[1] - bounds[0]) + bounds[0]
        for label in algo_labels:
            torch.manual_seed(seed); np.random.seed(seed)
            optimizer = RewardFunctionFixedDirectionBO(
                objective_function=objective,
                bounds=bounds,
                reward_mode=label,
                n_initial=cfg.n_initial,
                n_max=cfg.n_max,
                L_min=cfg.L_min,
                l_min_floor=cfg.l_min_floor,
                initial_X=initial_X,
            )
            print(f"Run {run_idx+1}/{cfg.n_runs} - {model_type}:{dataset_name} - {label}")
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
            }
            results[label].append(run_data)
    return results


def save_results_binary(tag: str, results: Dict[str, List[Dict[str, object]]]) -> None:
    eval_histories = {alg: np.array([run["eval_history"] for run in runs], dtype=float) for alg, runs in results.items()}
    np.save(os.path.join(OUTPUT_ROOT, f"{tag}_results.npy"), eval_histories, allow_pickle=True)


def export_reward_history(tag: str, results: Dict[str, List[Dict[str, object]]], n_initial: int) -> pd.DataFrame:
    rows = []
    for alg, runs in results.items():
        for run_idx, run in enumerate(runs):
            for iter_idx, reward_vec in enumerate(run["reward_history"], start=n_initial):
                for dim_idx, reward in enumerate(reward_vec):
                    rows.append({"Algorithm": alg, "Run": run_idx, "Iteration": iter_idx, "Dimension": dim_idx, "Reward": float(reward)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_ROOT, f"{tag}_reward_history.csv"), index=False)
    summary = df.groupby(["Algorithm","Dimension"])['Reward'].agg(mean="mean", std="std", cumulative="sum").reset_index()
    summary.to_csv(os.path.join(OUTPUT_ROOT, f"{tag}_dimension_summary.csv"), index=False)
    return df


def export_metric_history(tag: str, results: Dict[str, List[Dict[str, object]]], n_initial: int) -> None:
    def _flatten(metric_key: str, filename: str) -> None:
        rows = []
        for alg, runs in results.items():
            for run_idx, run in enumerate(runs):
                history = run.get(metric_key, [])
                for iter_idx, value in enumerate(history, start=n_initial):
                    rows.append({"Algorithm": alg, "Run": run_idx, "Iteration": iter_idx, metric_key: float(value)})
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_ROOT, f"{tag}_{filename}.csv"), index=False)
    _flatten("L_hat_history", "lhat_history")
    _flatten("grad_norm_history", "grad_norms")


def plot_convergence(tag: str, results: Dict[str, List[Dict[str, object]]], n_max: int) -> None:
    colors = cm.get_cmap("viridis", len(results))
    iterations = np.arange(1, n_max + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    for idx, (alg, runs) in enumerate(results.items()):
        histories = np.array([run["eval_history"] for run in runs], dtype=float)
        mean = histories.mean(axis=0); std = histories.std(axis=0)
        ax.plot(iterations, mean, color=colors(idx), label=alg, linewidth=2)
        ax.fill_between(iterations, mean - std, mean + std, color=colors(idx), alpha=0.15)
    ax.set_title(f"{tag}: 収束履歴 (平均±標準偏差)"); ax.set_xlabel("Iteration"); ax.set_ylabel("Best Value"); ax.grid(True, alpha=0.3); ax.legend()
    ax = axes[0, 1]
    final_box_data = [[run["best_value"] for run in runs] for runs in results.values()]
    ax.boxplot(final_box_data, labels=list(results.keys()), patch_artist=True)
    ax.set_title(f"{tag}: 最終最良値の分布"); ax.set_ylabel("Best Value"); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    theta_means = []
    dim_guess = None
    for runs in results.values():
        final_thetas = []
        for run in runs:
            if run["theta_history"]:
                arr = run["theta_history"][-1].abs().cpu().numpy(); final_thetas.append(arr); dim_guess = len(arr)
        theta_means.append(np.mean(final_thetas, axis=0) if final_thetas else np.zeros(dim_guess or 1))
    im = ax.imshow(theta_means, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(results))); ax.set_yticklabels(list(results.keys()))
    ax.set_xlabel("Dimension"); ax.set_ylabel("Algorithm"); ax.set_title(f"{tag}: |theta| 平均 (最終)")
    fig.colorbar(im, ax=ax, orientation="vertical")
    ax = axes[1, 1]
    means = []; stds = []
    for runs in results.values():
        ls_values = []
        for run in runs:
            for ls in run.get("lengthscale_history", []):
                ls_values.append(float(np.mean(ls)))
        means.append(np.mean(ls_values) if ls_values else 0.0)
        stds.append(np.std(ls_values) if ls_values else 0.0)
    ax.errorbar(list(results.keys()), means, yerr=stds, fmt="o", capsize=5)
    ax.set_title(f"{tag}: 平均 lengthscale"); ax.set_ylabel("lengthscale"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_ROOT, f"{tag}_comparison.png"), dpi=300); plt.close(fig)


def plot_reward_analysis(tag: str, df_rewards: pd.DataFrame, results: Dict[str, List[Dict[str, object]]], n_initial: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    for alg in results.keys():
        subset = df_rewards[df_rewards["Algorithm"] == alg]
        mean_series = subset.groupby("Iteration")["Reward"].mean()
        ax.plot(mean_series.index, mean_series.values, label=alg)
    ax.set_title(f"{tag}: 平均報酬推移"); ax.set_xlabel("Iteration"); ax.set_ylabel("Reward (mean)"); ax.grid(True, alpha=0.3); ax.legend()
    ax = axes[0, 1]
    # HPO では有効次元の概念が薄いので平均を再掲
    for alg in results.keys():
        subset = df_rewards[df_rewards["Algorithm"] == alg]
        mean_series = subset.groupby("Iteration")["Reward"].mean()
        ax.plot(mean_series.index, mean_series.values, label=alg)
    ax.set_title(f"{tag}: 平均報酬推移(再掲)"); ax.set_xlabel("Iteration"); ax.set_ylabel("Reward (mean)"); ax.grid(True, alpha=0.3); ax.legend()
    ax = axes[1, 0]
    for alg, runs in results.items():
        arr = np.array([run["L_hat_history"] for run in runs], dtype=float)
        if arr.size == 0: continue
        iterations = np.arange(n_initial, n_initial + arr.shape[1])
        ax.plot(iterations, arr.mean(axis=0), label=alg)
    ax.set_title(f"{tag}: L_hat 平均推移"); ax.set_xlabel("Iteration"); ax.set_ylabel("L_hat"); ax.grid(True, alpha=0.3); ax.legend()
    ax = axes[1, 1]
    rows = []
    for alg, runs in results.items():
        for run in runs:
            for v in run.get("grad_norm_history", []):
                rows.append({"Algorithm": alg, "GradNorm": v})
    if rows:
        df = pd.DataFrame(rows)
        df.boxplot(column="GradNorm", by="Algorithm", ax=ax)
        ax.set_title(f"{tag}: 勾配ノルム分布"); ax.set_ylabel("‖∇μ‖"); ax.set_xlabel("Algorithm"); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)
    plt.suptitle(""); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_ROOT, f"{tag}_reward_analysis.png"), dpi=300); plt.close(fig)


def plot_diagnostics(tag: str, results: Dict[str, List[Dict[str, object]]], n_initial: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    theta_norms = []; labels = []
    for alg, runs in results.items():
        vals = [float(run["theta_history"][-1].norm().item()) for run in runs if run["theta_history"]]
        if vals:
            theta_norms.append(vals); labels.append(alg)
    if theta_norms:
        ax.boxplot(theta_norms, labels=labels, patch_artist=True)
    ax.set_title(f"{tag}: 最終 |theta| 分布"); ax.grid(True, alpha=0.3)
    ax = axes[1]
    final_vals = []; labels2 = []
    for alg, runs in results.items():
        values = [run["L_hat_history"][-1] for run in runs if run["L_hat_history"]]
        if values:
            final_vals.append(values); labels2.append(alg)
    if final_vals:
        ax.boxplot(final_vals, labels=labels2, patch_artist=True)
    ax.set_title(f"{tag}: 最終 L_hat 分布"); ax.grid(True, alpha=0.3)
    ax = axes[2]
    dir_norm_rows = []
    for alg, runs in results.items():
        for run in runs:
            for dvec in run.get("direction_history", []):
                dir_norm_rows.append({"Algorithm": alg, "DirNorm": float(np.linalg.norm(dvec))})
    if dir_norm_rows:
        df = pd.DataFrame(dir_norm_rows)
        df.boxplot(column="DirNorm", by="Algorithm", ax=ax)
        ax.set_title(f"{tag}: 方向ベクトルノルム分布"); ax.grid(True, alpha=0.3)
    plt.suptitle(""); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_ROOT, f"{tag}_diagnostics.png"), dpi=300); plt.close(fig)


def run_all() -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    cfg = ExperimentConfig()
    targets_env = os.getenv("HPO_TARGETS")
    if targets_env:
        targets = []
        for tok in targets_env.split(','):
            tok = tok.strip();
            if not tok: continue
            model, dataset = tok.split(':')
            targets.append((model.strip(), dataset.strip()))
    else:
        targets = TARGETS_DEFAULT
    all_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for model_type, dataset_name in targets:
        tag = f"{model_type}_{dataset_name}"
        try:
            results = run_for_target(model_type, dataset_name, cfg)
        except RuntimeError as e:
            print(f"Skip {tag}: {e}")
            continue
        save_results_binary(tag, results)
        df_rewards = export_reward_history(tag, results, cfg.n_initial)
        export_metric_history(tag, results, cfg.n_initial)
        plot_convergence(tag, results, cfg.n_max)
        plot_reward_analysis(tag, df_rewards, results, cfg.n_initial)
        plot_diagnostics(tag, results, cfg.n_initial)
        # メタデータ
        meta = {
            "targets": targets,
            "config": {
                "n_initial": cfg.n_initial,
                "n_max": cfg.n_max,
                "n_runs": cfg.n_runs,
                "L_min": cfg.L_min,
                "l_min_floor": cfg.l_min_floor,
                "reward_algs": list(cfg.reward_algs),
            },
        }
        with open(os.path.join(OUTPUT_ROOT, f"{tag}_experiment_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        all_results[tag] = results
    return all_results


if __name__ == "__main__":
    run_all()
