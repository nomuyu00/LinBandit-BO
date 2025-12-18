#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # moved under exp/shuron/
"""LinBanditBO vs. 高次元BOベースライン実験スクリプト

実験仕様:
    * ベンチマーク: Styblinski‑Tang / Rastrigin / Ackley
    * 次元: 20次元（有効5次元） と 2次元（有効1次元）
    * LinBanditBO をベースラインとし、TuRBO-1 / VanillaBO / DropoutBO と比較
    * 20次元では 300 評価まで実行（n_initial を含む）
    * 出力形式は exp/linbandit_nlpd_dir/ を踏襲し、PNG図・CSV・NPYを保存

使い方:
    $ python exp/simple_linbandit_comparison/run_simple_linbandit_comparison.py

環境変数:
    SL_RUNS: ラン数（既定 20）
    SL_INIT: 初期点数（既定 5）
    SL_ITERS_20D: 20次元での総評価回数（既定 300）
    SL_ITERS_2D: 2次元での総評価回数（既定 120）
    SL_DEVICE: 強制デバイス指定（'cpu' / 'cuda' / 'mps'）
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from gpytorch.kernels import ScaleKernel, RBFKernel
from matplotlib import font_manager

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    japanize_matplotlib = None
import numpy as np
import torch
from sklearn.decomposition import PCA
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
import csv

# リポジトリルートから LinBanditBO を読み込む
from importlib.machinery import SourceFileLoader


# 日本語フォント設定（使用可能なものを動的に選択）
def _configure_fonts() -> None:
    candidates = [
        "Yu Gothic",
        "Yu Mincho",
        "MS Gothic",
        "MS Mincho",
        "Hiragino Sans",
        "Noto Sans CJK JP",
        "TakaoGothic",
        "IPAGothic",
        "IPAexGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if any(name == fam or name in fam for fam in available):
            plt.rcParams["font.family"] = name
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


_configure_fonts()

# アルゴリズムごとの配色
ALGO_COLORS = {
    "LinBandit": "#1f77b4",
    "TuRBO-1": "#ff7f0e",
    "VanillaBO": "#9467bd",
    "DropoutBO": "#8c564b",
    "SAASBO": "#2ca02c",
    "ALEBO": "#d62728",
}
PRIMARY_ALGO = "LinBandit"
STAGE_BOUNDS = [100, 200]  # 20d用に方向/報酬バーを序盤・中盤・終盤で区切る


def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        if (p / "LinBandit-BO.py").exists():
            return p
    # フォールバック（まず起きない想定）
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _find_repo_root()
SHURON_PATH = REPO_ROOT / "shuron.py"
LINBANDIT_PATH = REPO_ROOT / "LinBandit-BO.py"
shuron_mod = SourceFileLoader("shuron", str(SHURON_PATH)).load_module()
linbandit_mod = SourceFileLoader("linbandit_bo", str(LINBANDIT_PATH)).load_module()

LinBanditBO = shuron_mod.LinBanditBO
styblinski_tang = linbandit_mod.styblinski_tang
rastrigin = linbandit_mod.rastrigin

# 一部モジュール（例: コア実装）が rcParams のフォントを変更する可能性があるため、
# ここで再度日本語対応フォント設定を適用しておく。
_configure_fonts()


def ackley(x: torch.Tensor, a: float = 20.0, b: float = 0.2, c: float = 2.0 * math.pi) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.get_default_dtype())
    x = x.to(dtype=torch.get_default_dtype())
    d = x.shape[-1]
    sum_sq = torch.sum(x * x, dim=-1)
    term1 = -a * torch.exp(-b * torch.sqrt(sum_sq / d))
    term2 = -torch.exp(torch.sum(torch.cos(c * x), dim=-1) / d)
    return term1 + term2 + a + math.e


# 有効次元を制御するためのラッパ
def make_effective_function(
    base_func: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    active_indices: List[int],
) -> Callable[[torch.Tensor], torch.Tensor]:
    mask = torch.zeros(dim, dtype=torch.bool)
    mask[active_indices] = True

    def wrapped(x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.get_default_dtype())
        m = mask.to(device=x.device)
        if x.ndim == 1:
            z = x.clone()
            z[~m] = 0.0
        else:
            z = x.clone()
            z[..., ~m] = 0.0
        return base_func(z)

    return wrapped


def select_device() -> torch.device:
    forced = os.getenv("SL_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    ...
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Scenario:
    name: str
    func: Callable[[torch.Tensor], torch.Tensor]
    dim: int
    active: List[int]
    budget: int


def cumulative_best(values: torch.Tensor) -> np.ndarray:
    arr = values.detach().cpu().view(-1).numpy()
    return np.minimum.accumulate(arr)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# 方向ノルムと累積報酬（改善量に比例配分）を集計
def accumulate_dimension_stats(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_initial: int,
    active: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """各次元ごとの方向ノルム・報酬の累積とステージ別集計を返す。"""
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().reshape(-1)
    dim = X_np.shape[1]
    dir_sum = np.zeros(dim, dtype=np.float64)
    reward_sum = np.zeros(dim, dtype=np.float64)
    weights = np.full(dim, 1.0, dtype=np.float64)
    weights[active] = 1.0

    stage_limits = STAGE_BOUNDS + [len(X_np) - n_initial]
    stage_dir = [np.zeros(dim, dtype=np.float64) for _ in stage_limits]
    stage_reward = [np.zeros(dim, dtype=np.float64) for _ in stage_limits]
    stage_idx = 0
    step_count = 0

    best_idx = int(np.argmin(Y_np[:n_initial]))
    best_point = X_np[best_idx].copy()
    best_value = float(Y_np[best_idx])

    for idx in range(n_initial, len(X_np)):
        step = X_np[idx] - best_point
        abs_step = np.abs(step)
        weighted_step = abs_step * weights
        dir_sum += weighted_step
        while stage_idx < len(STAGE_BOUNDS) and step_count >= STAGE_BOUNDS[stage_idx]:
            stage_idx += 1

        total = float(weighted_step.sum())
        improvement = max(best_value - float(Y_np[idx]), 0.0)
        if improvement > 0.0 and total > 1e-12:
            reward_sum += improvement * (weighted_step / total)
            reward_sum += np.full(dim, 0.01 * improvement / dim, dtype=np.float64)
            stage_reward[stage_idx] += improvement * (weighted_step / total)
            stage_reward[stage_idx] += np.full(dim, 0.01 * improvement / dim, dtype=np.float64)

        stage_dir[stage_idx] += weighted_step

        if float(Y_np[idx]) < best_value:
            best_value = float(Y_np[idx])
            best_point = X_np[idx].copy()
        step_count += 1

    return dir_sum, reward_sum, stage_dir, stage_reward


def best_point_sequence(X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy().reshape(-1)
    seq: List[np.ndarray] = []
    best_val = float("inf")
    for x_row, y_val in zip(X_np, Y_np):
        if y_val < best_val - 1e-12:
            best_val = float(y_val)
            seq.append(np.copy(x_row))
    return np.stack(seq, axis=0) if seq else np.zeros((0, X_np.shape[1]), dtype=X_np.dtype)


def theoretical_minimum(scenario: Scenario) -> float:
    name = scenario.name.lower()
    active_dim = max(len(scenario.active), 1)
    if "styblinski" in name:
        return -39.16599 * active_dim
    if "rastrigin" in name:
        return 0.0
    if "ackley" in name:
        return 0.0
    return float("nan")


def normalize_inputs(X: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return torch.clamp((X - lb) / (ub - lb), 0.0, 1.0)


def stage_sums_from_history(
    history: List[torch.Tensor],
    stage_bounds: List[int],
) -> List[np.ndarray]:
    if len(history) == 0:
        return []
    mat = torch.stack(history, dim=0)  # [T, d]
    mat2 = mat.pow(2)
    sums = []
    start = 0
    total_steps = mat2.shape[0]
    for b in stage_bounds + [total_steps]:
        end = min(b, total_steps)
        if end <= start:
            sums.append(np.zeros(mat2.shape[1], dtype=np.float64))
        else:
            sums.append(mat2[start:end].sum(dim=0).detach().cpu().numpy())
        start = end
    return sums


# === LinBanditBO ランナー ===
def run_simple_linbandit(
    scenario: Scenario,
    bounds: torch.Tensor,
    n_initial: int,
    n_eval: int,
    seed: int,
    device: torch.device,
    initial_X: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    obj = make_effective_function(scenario.func, scenario.dim, scenario.active)

    runner = LinBanditBO(
        objective_function=obj,
        bounds=bounds.to(dtype=torch.double),
        n_initial=n_initial,
        n_max=n_eval,
        l_min=0.10,
        initial_X=initial_X,
    )
    start = time.time()
    _, best_y = runner.optimize()
    elapsed = time.time() - start

    history = cumulative_best(runner.Y.squeeze(-1))

    dir_sum: np.ndarray
    reward_sum: np.ndarray
    warmup_skip = int(os.getenv("SL_WARMUP_SKIP", 0))
    dir_stages: List[np.ndarray] = []
    reward_stages: List[np.ndarray] = []

    if hasattr(runner, "selected_direction_history") and len(getattr(runner, "selected_direction_history", [])) > 0:
        try:
            dirs = torch.stack(runner.selected_direction_history, dim=0)  # [T, d]
            if warmup_skip > 0 and dirs.shape[0] > warmup_skip:
                dirs = dirs[warmup_skip:]
            elif warmup_skip >= dirs.shape[0]:
                dirs = dirs[:0]
            dir_sum = (dirs.pow(2)).sum(dim=0).detach().cpu().numpy()
            dir_stages = stage_sums_from_history(list(dirs), STAGE_BOUNDS)
        except Exception:
            dir_sum, _, dir_stages, _ = accumulate_dimension_stats(runner.X, runner.Y, n_initial, scenario.active)
    else:
        dir_sum, _, dir_stages, _ = accumulate_dimension_stats(runner.X, runner.Y, n_initial, scenario.active)

    if hasattr(runner, "reward_history") and len(getattr(runner, "reward_history", [])) > 0:
        try:
            rmat = torch.stack(runner.reward_history, dim=0)  # [T, d] （r * a_unit）
            if warmup_skip > 0 and rmat.shape[0] > warmup_skip:
                rmat = rmat[warmup_skip:]
            elif warmup_skip >= rmat.shape[0]:
                rmat = rmat[:0]
            reward_sum = (rmat.pow(2)).sum(dim=0).detach().cpu().numpy()
            reward_stages = stage_sums_from_history(list(rmat), STAGE_BOUNDS)
        except Exception:
            _, reward_sum, _, reward_stages = accumulate_dimension_stats(runner.X, runner.Y, n_initial, scenario.active)
    else:
        _, reward_sum, _, reward_stages = accumulate_dimension_stats(runner.X, runner.Y, n_initial, scenario.active)
    best_path = best_point_sequence(runner.X, runner.Y)
    return {
        "best_history": history,
        "best_value": float(best_y),
        "elapsed": elapsed,
        "direction": dir_sum,
        "reward": reward_sum,
        "direction_stages": dir_stages,
        "reward_stages": reward_stages,
        "best_path": best_path,
    }


# === TuRBO-1 実装 ===


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    min_length: float = 0.5 ** 7
    max_length: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    best_value: float = math.inf

    @property
    def failure_tolerance(self) -> int:
        return max(4, int(math.ceil(self.dim / self.batch_size)))

    @property
    def success_tolerance(self) -> int:
        return 3


def update_state(state: TurboState, y_next: torch.Tensor) -> None:
    y_min = float(y_next.min().item())
    if y_min + 1e-8 < state.best_value:
        state.best_value = y_min
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter >= state.success_tolerance:
        state.length = min(state.length * 2.0, state.max_length)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.length = state.length / 2.0
        state.failure_counter = 0


def trust_region_bounds(
    center: torch.Tensor,
    state: TurboState,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    span = (ub - lb)
    half_length = 0.5 * state.length * span
    lower = torch.max(center - half_length, lb)
    upper = torch.min(center + half_length, ub)
    return lower, upper


def run_turbo(
    scenario: Scenario,
    bounds: torch.Tensor,
    n_initial: int,
    n_eval: int,
    seed: int,
    device: torch.device,
    initial_X: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    obj = make_effective_function(scenario.func, scenario.dim, scenario.active)

    dtype = torch.get_default_dtype()
    lb = bounds[0].to(device=device, dtype=dtype)
    ub = bounds[1].to(device=device, dtype=dtype)

    dim = scenario.dim
    if initial_X is not None:
        X_init = torch.as_tensor(initial_X, dtype=dtype, device=device)
    else:
        X_init = draw_sobol_samples(bounds=torch.stack([lb, ub]), n=1, q=n_initial).squeeze(0)
    Y_init = obj(X_init).unsqueeze(-1)

    train_x = X_init.clone()
    train_y = Y_init.clone()
    state = TurboState(dim=dim, batch_size=1)

    total_budget = n_eval - n_initial
    start = time.time()

    for _ in range(total_budget):
        norm_x = (train_x - lb) / (ub - lb)
        model = SingleTaskGP(norm_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        center = train_x[torch.argmin(train_y)].clone()
        lower, upper = trust_region_bounds(center, state, lb, ub)

        num_candidates = 256
        cand = lower + (upper - lower) * torch.rand(num_candidates, dim, device=device, dtype=dtype)
        cand_norm = (cand - lb) / (ub - lb)

        with torch.no_grad():
            posterior = model.posterior(cand_norm)
            mean = posterior.mean.view(-1)
            var = posterior.variance.view(-1)
            std = torch.sqrt(torch.clamp(var, min=1e-12))
        scores = mean - 1.96 * std
        idx = torch.argmin(scores)
        x_next = cand[idx]
        y_next = obj(x_next.unsqueeze(0)).unsqueeze(-1)

        train_x = torch.cat([train_x, x_next.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, y_next], dim=0)

        update_state(state, y_next)

        if state.length < state.min_length:
            state.length = 0.8
            state.failure_counter = 0
            state.success_counter = 0

    elapsed = time.time() - start
    history = np.minimum.accumulate(train_y.view(-1).detach().cpu().numpy())
    dir_sum, reward_sum, dir_stages, reward_stages = accumulate_dimension_stats(train_x, train_y, n_initial, scenario.active)
    best_path = best_point_sequence(train_x, train_y)
    return {
        "best_history": history,
        "best_value": float(train_y.min().item()),
        "elapsed": elapsed,
        "direction": dir_sum,
        "reward": reward_sum,
        "direction_stages": dir_stages,
        "reward_stages": reward_stages,
        "best_path": best_path,
    }


# === SAASBO（簡易ARD版） ===


def run_saasbo(
    scenario: Scenario,
    bounds: torch.Tensor,
    n_initial: int,
    n_eval: int,
    seed: int,
    device: torch.device,
    initial_X: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """SAASBOの軽量近似版（ARD+GammaPriorでスパース性を誘導）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    obj = make_effective_function(scenario.func, scenario.dim, scenario.active)
    dtype = torch.get_default_dtype()
    lb = bounds[0].to(device=device, dtype=dtype)
    ub = bounds[1].to(device=device, dtype=dtype)
    dim = scenario.dim

    if initial_X is not None:
        X_init = torch.as_tensor(initial_X, dtype=dtype, device=device)
    else:
        X_init = draw_sobol_samples(bounds=torch.stack([lb, ub]), n=1, q=n_initial).squeeze(0)
    Y_init = obj(X_init).unsqueeze(-1)
    train_x = X_init.clone()
    train_y = Y_init.clone()

    total_budget = n_eval - n_initial
    start = time.time()
    for _ in range(total_budget):
        Xn = normalize_inputs(train_x, lb, ub)
        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=dim,
                lengthscale_prior=GammaPrior(0.5, 1.0),  # 小さいlengthscaleを優遇しスパース性を促す
            )
        )
        model = SingleTaskGP(Xn, train_y, covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        acq = qExpectedImprovement(model=model, best_f=float(train_y.min().item()), maximize=False)
        unit = torch.stack(
            [torch.zeros(dim, device=device, dtype=dtype), torch.ones(dim, device=device, dtype=dtype)]
        )
        try:
            cand_norm, _ = optimize_acqf(
                acq,
                bounds=unit,
                q=1,
                num_restarts=8,
                raw_samples=128,
                options={"batch_limit": 4, "maxiter": 200},
            )
            x_new = lb + cand_norm.squeeze(0) * (ub - lb)
        except Exception:
            # 数値不安定時はランダム候補でフォールバック
            x_new = lb + (ub - lb) * torch.rand(dim, device=device, dtype=dtype)
        y_new = obj(x_new.unsqueeze(0)).unsqueeze(-1)
        train_x = torch.cat([train_x, x_new.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)

    elapsed = time.time() - start
    history = np.minimum.accumulate(train_y.view(-1).detach().cpu().numpy())
    dir_sum, reward_sum, dir_stages, reward_stages = accumulate_dimension_stats(train_x, train_y, n_initial, scenario.active)
    best_path = best_point_sequence(train_x, train_y)
    return {
        "best_history": history,
        "best_value": float(train_y.min().item()),
        "elapsed": elapsed,
        "direction": dir_sum,
        "reward": reward_sum,
        "direction_stages": dir_stages,
        "reward_stages": reward_stages,
        "best_path": best_path,
    }


# === ALEBO（PCAベース簡易版） ===


def run_alebo(
    scenario: Scenario,
    bounds: torch.Tensor,
    n_initial: int,
    n_eval: int,
    seed: int,
    device: torch.device,
    initial_X: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Adaptive Linear Embedding BO（ALEBO）の簡易版：PCAベースで埋め込みを更新。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    obj = make_effective_function(scenario.func, scenario.dim, scenario.active)
    dtype = torch.get_default_dtype()
    lb = bounds[0].to(device=device, dtype=dtype)
    ub = bounds[1].to(device=device, dtype=dtype)
    dim = scenario.dim
    low_dim = min(len(scenario.active), max(2, dim // 2))
    update_freq = int(os.getenv("SL_ALEBO_UPDATE", 20))

    if initial_X is not None:
        X_init = torch.as_tensor(initial_X, dtype=dtype, device=device)
    else:
        X_init = draw_sobol_samples(bounds=torch.stack([lb, ub]), n=1, q=n_initial).squeeze(0)
    Y_init = obj(X_init).unsqueeze(-1)
    train_x = X_init.clone()
    train_y = Y_init.clone()

    # 初期埋め込み（直交ランダム）
    A = random_orth_matrix(dim, low_dim, device=device)

    total_budget = n_eval - n_initial
    start = time.time()
    for it in range(total_budget):
        if it > 0 and it % update_freq == 0 and train_x.shape[0] > low_dim:
            # PCAで主要成分を学習し直す
            with torch.no_grad():
                Xcpu = train_x.detach().cpu().numpy()
                pca = PCA(n_components=low_dim)
                pca.fit(Xcpu)
                comp = torch.tensor(pca.components_.T, device=device, dtype=dtype)
                # 直交化
                q, _ = torch.linalg.qr(comp)
                A = q[:, :low_dim]

        # 埋め込み空間でGP
        Z = train_x @ A  # [N,k]
        Z_bounds = torch.tensor([[-1.0] * low_dim, [1.0] * low_dim], device=device, dtype=dtype)
        Zn = normalize_inputs(Z, Z_bounds[0], Z_bounds[1])
        model = SingleTaskGP(Zn, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        acq = qExpectedImprovement(model=model, best_f=float(train_y.min().item()), maximize=False)
        cand_norm, _ = optimize_acqf(
            acq,
            bounds=torch.stack([torch.zeros(low_dim, device=device, dtype=dtype), torch.ones(low_dim, device=device, dtype=dtype)]),
            q=1,
            num_restarts=8,
            raw_samples=128,
            options={"batch_limit": 4, "maxiter": 200},
        )
        z_new = Z_bounds[0] + cand_norm.squeeze(0) * (Z_bounds[1] - Z_bounds[0])
        x_new = torch.matmul(A, z_new)
        x_new = torch.clamp(x_new, lb, ub)
        y_new = obj(x_new.unsqueeze(0)).unsqueeze(-1)
        train_x = torch.cat([train_x, x_new.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)

    elapsed = time.time() - start
    history = np.minimum.accumulate(train_y.view(-1).detach().cpu().numpy())
    dir_sum, reward_sum, dir_stages, reward_stages = accumulate_dimension_stats(train_x, train_y, n_initial, scenario.active)
    best_path = best_point_sequence(train_x, train_y)
    return {
        "best_history": history,
        "best_value": float(train_y.min().item()),
        "elapsed": elapsed,
        "direction": dir_sum,
        "reward": reward_sum,
        "direction_stages": dir_stages,
        "reward_stages": reward_stages,
        "best_path": best_path,
    }


# === REMBO ===


def random_orth_matrix(dim: int, k: int, device: torch.device) -> torch.Tensor:
    mat = torch.randn(dim, k, device=device)
    q, _ = torch.linalg.qr(mat)
    return q[:, :k]


def run_rembo(
    scenario: Scenario,
    bounds: torch.Tensor,
    n_initial: int,
    n_eval: int,
    seed: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    obj = make_effective_function(scenario.func, scenario.dim, scenario.active)
    dim = scenario.dim
    low_dim = min(len(scenario.active), 10)

    lb = bounds[0].to(device=device, dtype=torch.get_default_dtype())
    ub = bounds[1].to(device=device, dtype=torch.get_default_dtype())

    A = random_orth_matrix(dim, low_dim, device=device)

    def project(z: torch.Tensor) -> torch.Tensor:
        x = A @ z
        return torch.clamp(x, lb, ub)

    z_bounds = torch.tensor([[-1.0] * low_dim, [1.0] * low_dim], device=device, dtype=torch.get_default_dtype())
    Z_init = draw_sobol_samples(bounds=z_bounds, n=1, q=n_initial).squeeze(0)
    X_init = torch.einsum("ij,nj->ni", A, Z_init)
    X_init = torch.clamp(X_init, lb, ub)

    Y_init = obj(X_init).unsqueeze(-1)

    train_z = Z_init.clone()
    train_y = Y_init.clone()
    train_x = X_init.clone()

    total_budget = n_eval - n_initial
    start = time.time()

    for _ in range(total_budget):
        model = SingleTaskGP(train_z, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq = qExpectedImprovement(model=model, best_f=float(train_y.min().item()), maximize=False)
        cand, _ = optimize_acqf(
            acq,
            bounds=z_bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
            options={"batch_limit": 4, "maxiter": 200},
        )
        z_new = cand.squeeze(0)
        x_new = project(z_new)
        y_new = obj(x_new.unsqueeze(0)).unsqueeze(-1)

        train_z = torch.cat([train_z, z_new.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)
        train_x = torch.cat([train_x, x_new.unsqueeze(0)], dim=0)

    elapsed = time.time() - start
    history = np.minimum.accumulate(train_y.view(-1).detach().cpu().numpy())
    dir_sum, reward_sum, dir_stages, reward_stages = accumulate_dimension_stats(train_x, train_y, n_initial, scenario.active)
    best_path = best_point_sequence(train_x, train_y)
    return {
        "best_history": history,
        "best_value": float(train_y.min().item()),
        "elapsed": elapsed,
        "direction": dir_sum,
        "reward": reward_sum,
        "direction_stages": dir_stages,
        "reward_stages": reward_stages,
        "best_path": best_path,
    }


def run_vanilla_bo(
    scenario: Scenario,
    bounds: torch.Tensor,
    n_initial: int,
    n_eval: int,
    seed: int,
    device: torch.device,
    initial_X: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    obj = make_effective_function(scenario.func, scenario.dim, scenario.active)
    dtype = torch.get_default_dtype()
    lb = bounds[0].to(device=device, dtype=dtype)
    ub = bounds[1].to(device=device, dtype=dtype)
    dim = scenario.dim

    if initial_X is not None:
        X_init = torch.as_tensor(initial_X, dtype=dtype, device=device)
    else:
        X_init = draw_sobol_samples(bounds=torch.stack([lb, ub]), n=1, q=n_initial).squeeze(0)
    Y_init = obj(X_init).unsqueeze(-1)

    train_x = X_init.clone()
    train_y = Y_init.clone()
    total_budget = n_eval - n_initial
    start = time.time()

    unit_bounds = torch.stack(
        [torch.zeros(dim, device=device, dtype=dtype), torch.ones(dim, device=device, dtype=dtype)]
    )

    for _ in range(total_budget):
        norm_x = (train_x - lb) / (ub - lb)
        model = SingleTaskGP(norm_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq = qExpectedImprovement(model=model, best_f=float(train_y.min().item()), maximize=False)
        cand_norm, _ = optimize_acqf(
            acq,
            bounds=unit_bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
            options={"batch_limit": 4, "maxiter": 200},
        )
        x_new = lb + cand_norm.squeeze(0) * (ub - lb)
        y_new = obj(x_new.unsqueeze(0)).unsqueeze(-1)

        train_x = torch.cat([train_x, x_new.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)

    elapsed = time.time() - start
    history = np.minimum.accumulate(train_y.view(-1).detach().cpu().numpy())
    dir_sum, reward_sum, dir_stages, reward_stages = accumulate_dimension_stats(train_x, train_y, n_initial, scenario.active)
    best_path = best_point_sequence(train_x, train_y)
    return {
        "best_history": history,
        "best_value": float(train_y.min().item()),
        "elapsed": elapsed,
        "direction": dir_sum,
        "reward": reward_sum,
        "direction_stages": dir_stages,
        "reward_stages": reward_stages,
        "best_path": best_path,
    }


def run_dropout_bo(
    scenario: Scenario,
    bounds: torch.Tensor,
    n_initial: int,
    n_eval: int,
    seed: int,
    device: torch.device,
    initial_X: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """論文 'High-Dimensional BO using Dropout' (arXiv:1802.05400) に基づく Dropout-BO 実装."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    obj = make_effective_function(scenario.func, scenario.dim, scenario.active)
    dtype = torch.get_default_dtype()
    lb = bounds[0].to(device=device, dtype=dtype)
    ub = bounds[1].to(device=device, dtype=dtype)
    dim = scenario.dim

    # 部分次元 d を指定（既定: min(10, dim)）
    k = int(os.getenv("SL_DROPOUT_K", min(10, dim)))
    k = max(1, min(k, dim))
    p_mix = float(os.getenv("SL_DROPOUT_PMIX", 0.5))  # Dropout-Mix: pでcopy, 1-pでrandom

    if initial_X is not None:
        X_init = torch.as_tensor(initial_X, dtype=dtype, device=device)
    else:
        X_init = draw_sobol_samples(bounds=torch.stack([lb, ub]), n=1, q=n_initial).squeeze(0)
    Y_init = obj(X_init).unsqueeze(-1)

    train_x = X_init.clone()
    train_y = Y_init.clone()

    total_budget = n_eval - n_initial
    start = time.time()

    for _ in range(total_budget):
        # 1) サブセット選択
        subset = np.random.choice(dim, size=k, replace=False)
        subset = np.sort(subset)
        mask = torch.tensor(subset, device=device, dtype=torch.long)
        # 2) サブ空間GPを学習
        X_sub = train_x[:, mask]
        Xn = normalize_inputs(X_sub, lb[mask], ub[mask])
        model = SingleTaskGP(Xn, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        # 3) サブ空間でEI最適化
        bounds_sub = torch.stack(
            [torch.zeros(k, device=device, dtype=dtype), torch.ones(k, device=device, dtype=dtype)]
        )
        acq = qExpectedImprovement(model=model, best_f=float(train_y.min().item()), maximize=False)
        cand_norm, _ = optimize_acqf(
            acq,
            bounds=bounds_sub,
            q=1,
            num_restarts=8,
            raw_samples=128,
            options={"batch_limit": 4, "maxiter": 200},
        )
        x_sub = lb[mask] + cand_norm.squeeze(0) * (ub[mask] - lb[mask])
        # 4) ドロップされた次元を埋める（Mix戦略）
        x_full = torch.empty(dim, device=device, dtype=dtype)
        x_full[mask] = x_sub
        drop_idx = [i for i in range(dim) if i not in subset]
        if len(drop_idx) > 0:
            drop_tensor = torch.tensor(drop_idx, device=device, dtype=torch.long)
            best_idx = int(torch.argmin(train_y).item())
            if np.random.rand() < p_mix:
                x_full[drop_tensor] = train_x[best_idx, drop_tensor]  # Copy-best
            else:
                x_full[drop_tensor] = lb[drop_tensor] + (ub[drop_tensor] - lb[drop_tensor]) * torch.rand(
                    len(drop_idx), device=device, dtype=dtype
                )
        # 5) 評価
        y_new = obj(x_full.unsqueeze(0)).unsqueeze(-1)
        train_x = torch.cat([train_x, x_full.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)

    elapsed = time.time() - start
    history = np.minimum.accumulate(train_y.view(-1).detach().cpu().numpy())
    dir_sum, reward_sum, dir_stages, reward_stages = accumulate_dimension_stats(train_x, train_y, n_initial, scenario.active)
    best_path = best_point_sequence(train_x, train_y)
    return {
        "best_history": history,
        "best_value": float(train_y.min().item()),
        "elapsed": elapsed,
        "direction": dir_sum,
        "reward": reward_sum,
        "direction_stages": dir_stages,
        "reward_stages": reward_stages,
        "best_path": best_path,
    }


ALGORITHMS = {
    "LinBandit": run_simple_linbandit,
    "TuRBO-1": run_turbo,
    "VanillaBO": run_vanilla_bo,
    "DropoutBO": run_dropout_bo,
    "SAASBO": run_saasbo,
    "ALEBO": run_alebo,
}


def run_scenario(
    scenario: Scenario,
    n_initial: int,
    runs: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    bounds = torch.tensor([[-5.0] * scenario.dim, [5.0] * scenario.dim], dtype=torch.double)
    all_histories: Dict[str, List[np.ndarray]] = {name: [] for name in ALGORITHMS}
    elapsed: Dict[str, List[float]] = {name: [] for name in ALGORITHMS}
    dir_records: Dict[str, List[np.ndarray]] = {name: [] for name in ALGORITHMS}
    reward_records: Dict[str, List[np.ndarray]] = {name: [] for name in ALGORITHMS}
    best_paths: Dict[str, List[np.ndarray]] = {name: [] for name in ALGORITHMS}
    dir_stage_records: Dict[str, List[List[np.ndarray]]] = {name: [] for name in ALGORITHMS}
    reward_stage_records: Dict[str, List[List[np.ndarray]]] = {name: [] for name in ALGORITHMS}

    # 既存結果を読み込み（LinBandit以外）。無ければ後で実行。
    scenario_dir = (Path(__file__).resolve().parent / "output_results_simple_linbandit_comparison" / scenario.name)
    cached_stats: Dict[str, Dict[str, np.ndarray]] = {}
    for name in ALGORITHMS:
        if name == "LinBandit":
            continue
        cached = load_cached_algorithm_stats(scenario, scenario_dir, name)
        if cached is not None:
            cached_stats[name] = cached

    for run_id in range(runs):
        seed = 1000 * run_id + 7
        torch.manual_seed(seed)
        np.random.seed(seed)
        # 共通初期点（Sobol）を生成して初期区間の収束値を揃える
        dtype = torch.get_default_dtype()
        lb = bounds[0].to(dtype=dtype)
        ub = bounds[1].to(dtype=dtype)
        obj = make_effective_function(scenario.func, scenario.dim, scenario.active)
        init_x = draw_sobol_samples(bounds=torch.stack([lb, ub]), n=1, q=n_initial).squeeze(0)
        with torch.no_grad():
            init_y = obj(init_x).unsqueeze(-1)
        baseline = float(init_y.min().item())

        for name, fn in ALGORITHMS.items():
            # LinBanditは必ず再実行、他はキャッシュが無い場合のみ実行
            if name != "LinBandit" and name in cached_stats:
                continue
            kwargs = dict(
                scenario=scenario,
                bounds=bounds,
                n_initial=n_initial,
                n_eval=scenario.budget,
                seed=seed,
                device=device,
            )
            kwargs["initial_X"] = init_x
            result = fn(**kwargs)

            hist = result["best_history"].copy()
            if hist.shape[0] >= n_initial:
                hist[:n_initial] = baseline
            all_histories[name].append(hist)
            elapsed[name].append(result["elapsed"])
            dir_records[name].append(result["direction"])
            reward_records[name].append(result["reward"])
            best_paths[name].append(result["best_path"])
            dir_stage_records[name].append(result.get("direction_stages", []))
            reward_stage_records[name].append(result.get("reward_stages", []))

    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for name, histories in all_histories.items():
        # キャッシュがあればそれを採用
        if name != "LinBandit" and name in cached_stats and not histories:
            stats[name] = cached_stats[name]
            continue
        max_len = max(len(h) for h in histories)
        padded = []
        for hist in histories:
            if len(hist) < max_len:
                pad = np.full(max_len - len(hist), hist[-1])
                hist = np.concatenate([hist, pad])
            padded.append(hist)
        arr = np.stack(padded, axis=0)
        stats[name] = {
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0),
            "histories": arr,
            "elapsed": np.array(elapsed[name], dtype=np.float64),
            "direction": np.stack(dir_records[name], axis=0),
            "reward": np.stack(reward_records[name], axis=0),
            "best_paths": best_paths[name],
        }
        # ステージ別の平均（二乗和）を格納（空ならスキップ）
        if dir_stage_records[name] and len(dir_stage_records[name][0]) > 0:
            # dir_stage_records: List[run][stage][dim]
            stage_tensor = np.stack(
                [np.stack(stages, axis=0) for stages in dir_stage_records[name]], axis=0
            )  # [runs, num_stage, dim]
            reward_stage_tensor = np.stack(
                [np.stack(stages, axis=0) for stages in reward_stage_records[name]], axis=0
            )
            stats[name]["direction_stages"] = stage_tensor
            stats[name]["reward_stages"] = reward_stage_tensor
    return stats


def plot_convergence(stats: Dict[str, Dict[str, np.ndarray]], scenario: Scenario, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(stats[next(iter(stats))]["mean"].shape[0])
    for name, data in stats.items():
        mean = data["mean"]
        std = data["std"]
        color = ALGO_COLORS.get(name)
        linewidth = 3.0 if name == PRIMARY_ALGO else 1.6
        plt.plot(x, mean, label=name, color=color, linewidth=linewidth)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    min_val = theoretical_minimum(scenario)
    if math.isfinite(min_val):
        plt.axhline(min_val, color="red", linestyle="--", linewidth=1.5, label="理論下限")
    plt.xlabel("Iteration")
    plt.ylabel("Best value (mean ± std)")
    plt.title(f"{scenario.name}: convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_dimension_bars(
    stats: Dict[str, Dict[str, np.ndarray]],
    scenario: Scenario,
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if PRIMARY_ALGO not in stats:
        return
    data = stats[PRIMARY_ALGO][key]
    dim = data.shape[1]
    x = np.arange(dim)
    labels = [f"{idx}{'*' if idx in scenario.active else ''}" for idx in range(dim)]
    active_mask = np.zeros(dim, dtype=bool)
    active_mask[scenario.active] = True

    plt.figure(figsize=(10, 6))
    colors = np.array(["#9ecae1"] * dim)
    colors[active_mask] = ALGO_COLORS[PRIMARY_ALGO]
    mean = data.mean(axis=0)
    plt.bar(x, mean, color=colors)
    for idx, val in enumerate(mean):
        if active_mask[idx]:
            plt.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, labels)
    plt.xlabel("次元 (* は有効次元)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.2)
    legend_handles = [plt.Line2D([0], [0], color=ALGO_COLORS[PRIMARY_ALGO], lw=8, label="有効次元"),
                      plt.Line2D([0], [0], color="#9ecae1", lw=8, label="非有効次元")]
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_dimension_bars_stage(
    stats: Dict[str, Dict[str, np.ndarray]],
    scenario: Scenario,
    key: str,
    ylabel: str,
    title_prefix: str,
    out_dir: Path,
) -> None:
    """20次元シナリオ向けにステージ毎（序盤/中盤/終盤）のバー図を描画。"""
    if PRIMARY_ALGO not in stats:
        return
    stage_key = f"{key}_stages"
    if stage_key not in stats[PRIMARY_ALGO]:
        return
    stage_tensor = stats[PRIMARY_ALGO][stage_key]  # shape [runs, stages, dim]
    stage_mean = stage_tensor.mean(axis=0)  # [stages, dim]
    stage_labels = ["early(0-99)", "mid(100-199)", "late(200-)"]
    for idx, arr in enumerate(stage_mean):
        plt.figure(figsize=(10, 6))
        dim = arr.shape[0]
        x = np.arange(dim)
        labels = [f"{i}{'*' if i in scenario.active else ''}" for i in range(dim)]
        active_mask = np.zeros(dim, dtype=bool)
        active_mask[scenario.active] = True
        colors = np.array(["#9ecae1"] * dim)
        colors[active_mask] = ALGO_COLORS.get(PRIMARY_ALGO, "#1f77b4")
        plt.bar(x, arr, color=colors)
        for i, val in enumerate(arr):
            if active_mask[i]:
                plt.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        plt.xticks(x, labels)
        plt.xlabel("次元 (* は有効次元)")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} ({stage_labels[idx]})")
        plt.grid(True, axis="y", alpha=0.2)
        legend_handles = [
            plt.Line2D([0], [0], color=ALGO_COLORS.get(PRIMARY_ALGO, "#1f77b4"), lw=8, label="有効次元"),
            plt.Line2D([0], [0], color="#9ecae1", lw=8, label="非有効次元"),
        ]
        plt.legend(handles=legend_handles, loc="best")
        plt.tight_layout()
        filename = out_dir / f"{scenario.name}_{key}_stage{idx+1}.png"
        plt.savefig(filename)
        plt.close()


def load_cached_algorithm_stats(
    scenario: Scenario,
    scenario_dir: Path,
    algo: str,
) -> Optional[Dict[str, np.ndarray]]:
    """既存の出力から指定アルゴリズムの統計を読み込む。無ければ None。"""
    try:
        histories = np.load(scenario_dir / f"{scenario.name}_{algo}_histories.npy")
        direction = np.load(scenario_dir / f"{scenario.name}_{algo}_direction.npy")
        reward = np.load(scenario_dir / f"{scenario.name}_{algo}_reward.npy")
    except FileNotFoundError:
        return None

    # summaryから平均時間を取得（無ければ0で埋める）
    summary_path = scenario_dir / f"{scenario.name}_summary.csv"
    avg_elapsed = 0.0
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Algorithm") == algo:
                    try:
                        avg_elapsed = float(row.get("AvgElapsedSec", 0.0))
                    except Exception:
                        avg_elapsed = 0.0
                    break
    runs = histories.shape[0]
    elapsed = np.full(runs, avg_elapsed, dtype=np.float64)
    mean = np.mean(histories, axis=0)
    std = np.std(histories, axis=0)
    stats = {
        "mean": mean,
        "std": std,
        "histories": histories,
        "elapsed": elapsed,
        "direction": direction,
        "reward": reward,
        "best_paths": [],  # 軌跡は未保存なので空
    }
    return stats


def plot_trajectories(
    stats: Dict[str, Dict[str, np.ndarray]],
    scenario: Scenario,
    out_dir: Path,
) -> None:
    if scenario.dim != 2:
        return

    lb, ub = -5.0, 5.0
    grid_points = 200
    xs = np.linspace(lb, ub, grid_points)
    ys = np.linspace(lb, ub, grid_points)
    Xg, Yg = np.meshgrid(xs, ys)
    grid = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)

    obj = make_effective_function(scenario.func, scenario.dim, scenario.active)
    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.double)
        Z = obj(grid_tensor).reshape(Xg.shape).detach().cpu().numpy()

    for algo_name, data in stats.items():
        color = ALGO_COLORS.get(algo_name, "#333333")
        linewidth = 3.0 if algo_name == PRIMARY_ALGO else 1.5
        for run_idx, path in enumerate(data["best_paths"], start=1):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.contourf(Xg, Yg, Z, levels=50, cmap="viridis")
            cs = ax.contour(Xg, Yg, Z, levels=25, colors="k", linewidths=0.3, alpha=0.4)
            path_np = np.asarray(path)
            if path_np.size == 0:
                plt.close(fig)
                continue
            ax.plot(path_np[:, 0], path_np[:, 1], "-o", color=color, linewidth=linewidth, markersize=4)
            ax.scatter(path_np[0, 0], path_np[0, 1], color="white", edgecolor=color, s=60, zorder=5, label="初期点")
            ax.scatter(path_np[-1, 0], path_np[-1, 1], color="red", s=60, zorder=6, label="最終点")
            ax.set_xlabel("x0")
            ax.set_ylabel("x1")
            ax.set_title(f"{scenario.name} - {algo_name} Run {run_idx:02d}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_dir / f"{scenario.name}_{algo_name}_run{run_idx:02d}_trajectory.png")
            plt.close(fig)


def save_summary(stats: Dict[str, Dict[str, np.ndarray]], scenario: Scenario, out_dir: Path) -> None:
    rows = []
    dir_rows = []
    for name, data in stats.items():
        hist = data["histories"]
        final_vals = hist[:, -1]
        rows.append({
            "Algorithm": name,
            "FinalMean": float(final_vals.mean()),
            "FinalStd": float(final_vals.std()),
            "BestAcrossRuns": float(final_vals.min()),
            "Median": float(np.median(final_vals)),
            "AvgElapsedSec": float(data["elapsed"].mean()),
        })
        np.save(out_dir / f"{scenario.name}_{name}_histories.npy", hist)
        np.save(out_dir / f"{scenario.name}_{name}_direction.npy", data["direction"])
        np.save(out_dir / f"{scenario.name}_{name}_reward.npy", data["reward"])

        dir_mean = data["direction"].mean(axis=0)
        dir_std = data["direction"].std(axis=0)
        reward_mean = data["reward"].mean(axis=0)
        reward_std = data["reward"].std(axis=0)
        for dim_idx in range(dir_mean.shape[0]):
            dir_rows.append({
                "Algorithm": name,
                "Dimension": dim_idx,
                "Active": int(dim_idx in scenario.active),
                "DirectionMean": float(dir_mean[dim_idx]),
                "DirectionStd": float(dir_std[dim_idx]),
                "RewardMean": float(reward_mean[dim_idx]),
                "RewardStd": float(reward_std[dim_idx]),
            })
    import csv

    with open(out_dir / f"{scenario.name}_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if dir_rows:
        with open(out_dir / f"{scenario.name}_dimension_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Algorithm",
                    "Dimension",
                    "Active",
                    "DirectionMean",
                    "DirectionStd",
                    "RewardMean",
                    "RewardStd",
                ],
            )
            writer.writeheader()
            writer.writerows(dir_rows)

    meta = {
        "scenario": {
            "name": scenario.name,
            "dim": scenario.dim,
            "active": scenario.active,
            "budget": scenario.budget,
        },
        "algorithms": list(ALGORITHMS.keys()),
        "runs": len(stats[next(iter(stats))]["histories"]),
    }
    with open(out_dir / f"{scenario.name}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    runs = int(os.getenv("SL_RUNS", 20))
    n_initial = int(os.getenv("SL_INIT", 30))
    budget_20d = int(os.getenv("SL_ITERS_20D", 300))
    budget_2d = int(os.getenv("SL_ITERS_2D", 120))

    scenarios = [
        Scenario(name="20d_5active_StyblinskiTang", func=styblinski_tang, dim=20, active=list(range(5)), budget=budget_20d),
        Scenario(name="20d_5active_Ackley", func=ackley, dim=20, active=list(range(5)), budget=budget_20d),
        Scenario(name="20d_5active_Rastrigin", func=rastrigin, dim=20, active=list(range(5)), budget=budget_20d),
        Scenario(name="2d_1active_StyblinskiTang", func=styblinski_tang, dim=2, active=[0], budget=budget_2d),
        Scenario(name="2d_1active_Ackley", func=ackley, dim=2, active=[0], budget=budget_2d),
        Scenario(name="2d_1active_Rastrigin", func=rastrigin, dim=2, active=[0], budget=budget_2d),
    ]

    device = select_device()
    print(f"[INFO] device = {device}")

    out_root = Path(__file__).resolve().parent / "output_results_simple_linbandit_comparison"
    ensure_dir(out_root)

    for scenario in scenarios:
        print(f"[INFO] Scenario {scenario.name} ...")
        stats = run_scenario(scenario, n_initial=n_initial, runs=runs, device=device)
        scenario_dir = out_root / scenario.name
        ensure_dir(scenario_dir)
        plot_convergence(stats, scenario, scenario_dir / f"{scenario.name}_convergence.png")
        plot_dimension_bars(
            stats,
            scenario,
            key="direction",
            ylabel="累積方向ノルム",
            title=f"{scenario.name}: 方向ノルム集計",
            out_path=scenario_dir / f"{scenario.name}_direction_usage.png",
        )
        plot_dimension_bars(
            stats,
            scenario,
            key="reward",
            ylabel="累積報酬 (改善量配分)",
            title=f"{scenario.name}: 累積報酬配分",
            out_path=scenario_dir / f"{scenario.name}_reward_allocation.png",
        )
        if scenario.dim == 20:
            plot_dimension_bars_stage(
                stats,
                scenario,
                key="direction",
                ylabel="累積方向ノルム（ステージ別）",
                title_prefix=f"{scenario.name}: 方向ノルム集計",
                out_dir=scenario_dir,
            )
            plot_dimension_bars_stage(
                stats,
                scenario,
                key="reward",
                ylabel="累積報酬（ステージ別）",
                title_prefix=f"{scenario.name}: 累積報酬配分",
                out_dir=scenario_dir,
            )
        save_summary(stats, scenario, scenario_dir)
        plot_trajectories(stats, scenario, scenario_dir)
        print(f"  -> 完了 (結果: {scenario_dir})")


if __name__ == "__main__":
    main()
