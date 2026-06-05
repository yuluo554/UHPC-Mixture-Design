# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign 贝叶斯优化模块
基于 BoTorch/Ax 的多目标贝叶斯优化，含 EHVI/qNEHVI 采集函数

核心类：
- ParetoCollector: Pareto 前沿收集/去重/超体积计算
- BayesianOptimizer: 多目标贝叶斯优化器
"""

import os
import sys
import warnings
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── 可选依赖检查 ─────────────────────────────────────────────────
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import gpytorch

    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False

try:
    import botorch

    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False

# 本地模块
from core.config import (
    get_all_feature_names,
    get_all_target_names,
    get_feature_ranges,
    get_categorical_features,
    get_target_direction,
    get_feature_label,
)


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def _normalize_for_minimization(y: np.ndarray, directions: List[str]) -> np.ndarray:
    """
    将所有目标统一转换为最小化问题。
    - maximize → 取负
    - minimize  → 保持
    - target    → 到目标区间中点的距离（最小化）

    Args:
        y: (n_samples, n_objectives) 原始目标值
        directions: 各目标的方向列表

    Returns:
        y_norm: 统一最小化后的值
    """
    y_norm = y.astype(np.float64).copy()
    for i, d in enumerate(directions):
        if d == "maximize":
            y_norm[:, i] = -y_norm[:, i]
        elif d == "target":
            # 默认目标区间中点，实际应动态设定
            median = np.median(y[:, i])
            y_norm[:, i] = np.abs(y[:, i] - median)
        # minimize 保持原样
    return y_norm


def _denormalize_objectives(y_norm: np.ndarray, directions: List[str]) -> np.ndarray:
    """将最小化值还原为原始方向"""
    y_orig = y_norm.astype(np.float64).copy()
    for i, d in enumerate(directions):
        if d == "maximize":
            y_orig[:, i] = -y_orig[:, i]
    return y_orig


def compute_reference_point(
    Y: np.ndarray, directions: List[str], inflation: float = 0.1
) -> np.ndarray:
    """
    计算超体积参考点（所有目标统一为最小化后的 nadir 点 + 膨胀）。

    Args:
        Y: (n_samples, n_objectives) 原始目标值
        directions: 各目标方向
        inflation: 膨胀系数（默认 10%）

    Returns:
        ref_point: 参考点
    """
    Y_norm = _normalize_for_minimization(Y, directions)
    nadir = np.max(Y_norm, axis=0)  # 最差点
    range_ = np.ptp(Y_norm, axis=0)  # 极差
    # 膨胀: 如果没有变化则至少膨胀 0.1
    inflation_amount = np.maximum(range_ * inflation, 0.1)
    return nadir + inflation_amount


def _validate_directions(targets: List[str]) -> List[str]:
    """从 config 获取目标方向，未找到则默认 minimize"""
    dirs = []
    for t in targets:
        d = get_target_direction(t)
        dirs.append(d if d else "minimize")
    return dirs


def _build_bounds_tensor(
    feature_names: List[str], device: torch.device
) -> torch.Tensor:
    """
    根据 config 构建特征边界张量 (2, d)，行0=下界，行1=上界。
    只处理数值型连续特征，分类型特征将被排除。
    """
    ranges = get_feature_ranges()
    cat_features = get_categorical_features()

    lower, upper = [], []
    for name in feature_names:
        if name in cat_features:
            continue  # 跳过分类特征
        if name in ranges:
            lo, hi = ranges[name]
            lower.append(float(lo))
            upper.append(float(hi))
        else:
            # 默认宽边界
            lower.append(0.0)
            upper.append(1000.0)

    if not lower:
        return torch.zeros(2, 1, device=device)
    return torch.tensor([lower, upper], device=device, dtype=torch.float64)


def _get_continuous_mask(feature_names: List[str]) -> List[int]:
    """返回连续特征在特征列表中的索引"""
    cat_features = get_categorical_features()
    return [i for i, name in enumerate(feature_names) if name not in cat_features]


# ═══════════════════════════════════════════════════════════════════
# ParetoCollector — Pareto 前沿管理
# ═══════════════════════════════════════════════════════════════════

class ParetoCollector:
    """
    收集并管理 Pareto 前沿点，支持去重与超体积追踪。

    Attributes:
        n_obj: 目标数
        directions: 各目标方向
        X_all: 所有已评估的输入
        Y_all: 所有已评估的原始目标值
        Y_norm: 统一最小化后的目标值
        ref_point: 超体积参考点
        history: 每步超体积记录
    """

    def __init__(
        self,
        directions: Optional[List[str]] = None,
        ref_point: Optional[np.ndarray] = None,
    ):
        self.directions = directions or []
        self.ref_point = ref_point
        self.X_all: Optional[np.ndarray] = None
        self.Y_all: Optional[np.ndarray] = None
        self.Y_norm: Optional[np.ndarray] = None
        self.history: List[Dict] = []  # [{iteration, n_frontier, hypervolume}]
        self._init_ref_point: Optional[np.ndarray] = None  # 初始参考点（固定）

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def n_obj(self) -> int:
        return len(self.directions) if self.directions else 0

    @property
    def n_points(self) -> int:
        return len(self.Y_all) if self.Y_all is not None else 0

    # ── 核心方法 ──────────────────────────────────────────────

    def update(
        self,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        iteration: int = 0,
    ) -> Dict:
        """
        添加新点并更新 Pareto 前沿。

        Args:
            X_new: (n, d) 输入
            Y_new: (n, m) 原始目标值
            iteration: 当前迭代编号

        Returns:
            dict: {n_frontier, hypervolume, is_new_pareto}
        """
        Y_norm_new = _normalize_for_minimization(Y_new, self.directions)

        if self.X_all is None:
            self.X_all = X_new
            self.Y_all = Y_new
            self.Y_norm = Y_norm_new
        else:
            self.X_all = np.vstack([self.X_all, X_new])
            self.Y_all = np.vstack([self.Y_all, Y_new])
            self.Y_norm = np.vstack([self.Y_norm, Y_norm_new])

        # 去重：移除完全重复的行
        self._deduplicate()

        # 设置/扩展参考点（确保始终覆盖所有数据）
        if self._init_ref_point is None and self.ref_point is None:
            # 无显式参考点：从数据计算
            self._init_ref_point = compute_reference_point(
                self.Y_all, self.directions, inflation=0.3
            )
        elif self._init_ref_point is None and self.ref_point is not None:
            # 使用显式传入的参考点
            self._init_ref_point = np.asarray(self.ref_point, dtype=np.float64)
        else:
            # 扩展已有参考点以覆盖新数据
            current_ref = compute_reference_point(
                self.Y_all, self.directions, inflation=0.3
            )
            self._init_ref_point = np.maximum(self._init_ref_point, current_ref)
        self.ref_point = self._init_ref_point

        # 计算前沿与超体积
        frontier_mask = self._pareto_mask(self.Y_norm)
        n_frontier = int(frontier_mask.sum())
        hv = self._compute_hypervolume(self.Y_norm[frontier_mask])

        record = {
            "iteration": iteration,
            "n_points": self.n_points,
            "n_frontier": n_frontier,
            "hypervolume": hv,
        }
        self.history.append(record)
        return record

    def get_frontier(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            (X_frontier, Y_frontier) — Pareto 最优点的输入和原始目标值
        """
        if self.Y_norm is None or len(self.Y_norm) == 0:
            return np.array([]), np.array([])
        mask = self._pareto_mask(self.Y_norm)
        return self.X_all[mask], self.Y_all[mask]

    def get_history_df(self) -> pd.DataFrame:
        """返回超体积收敛历史的 DataFrame"""
        return pd.DataFrame(self.history)

    def get_hypervolume(self) -> float:
        """返回当前超体积"""
        if self.Y_norm is None or len(self.Y_norm) == 0:
            return 0.0
        mask = self._pareto_mask(self.Y_norm)
        return self._compute_hypervolume(self.Y_norm[mask])

    # ── 内部方法 ──────────────────────────────────────────────

    def _pareto_mask(self, Y: np.ndarray) -> np.ndarray:
        """返回非支配点的布尔掩码"""
        n = len(Y)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                # j 支配 i?
                if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                    mask[i] = False
                    break
        return mask

    def _compute_hypervolume(self, Y_frontier: np.ndarray) -> float:
        """计算超体积（2D 精确，3D+ 采用逐维近似）"""
        if len(Y_frontier) == 0:
            return 0.0
        if self.ref_point is None:
            return 0.0

        m = Y_frontier.shape[1]
        ref = self.ref_point

        if m == 1:
            return float(ref[0] - np.min(Y_frontier[:, 0]))

        if m == 2:
            return self._hv_2d(Y_frontier, ref)

        # 3D+: 使用 Lebesgue 测度近似
        return self._hv_approx(Y_frontier, ref)

    def _hv_2d(self, Y: np.ndarray, ref: np.ndarray) -> float:
        """二维精确超体积（标准算法）"""
        # 按第一目标升序排列
        idx = np.argsort(Y[:, 0])
        Y_sorted = Y[idx]
        hv = 0.0
        k = len(Y_sorted)
        for i in range(k - 1):
            hv += (Y_sorted[i + 1, 0] - Y_sorted[i, 0]) * (ref[1] - Y_sorted[i, 1])
        hv += (ref[0] - Y_sorted[k - 1, 0]) * (ref[1] - Y_sorted[k - 1, 1])
        return hv

    def _hv_approx(self, Y: np.ndarray, ref: np.ndarray, n_samples: int = 20000) -> float:
        """Monte Carlo 超体积近似"""
        m = Y.shape[1]
        # 采样边界盒
        Y_min = np.min(Y, axis=0)
        samples = np.random.uniform(Y_min, ref, size=(n_samples, m))
        # 计数被支配的样本
        dominated = np.any(np.all(samples[:, None, :] >= Y[None, :, :], axis=2), axis=1)
        volume = np.prod(ref - Y_min)
        return float(volume * dominated.mean())

    def _deduplicate(self, tolerance: float = 1e-8):
        """移除 Y 中的重复行（同时更新 X）"""
        if self.Y_norm is None or len(self.Y_norm) <= 1:
            return
        _, unique_idx = np.unique(
            np.round(self.Y_norm, decimals=8), axis=0, return_index=True
        )
        if len(unique_idx) < len(self.Y_norm):
            self.X_all = self.X_all[unique_idx]
            self.Y_all = self.Y_all[unique_idx]
            self.Y_norm = self.Y_norm[unique_idx]


# ═══════════════════════════════════════════════════════════════════
# BayesianOptimizer — 多目标贝叶斯优化器
# ═══════════════════════════════════════════════════════════════════

class BayesianOptimizer:
    """
    多目标贝叶斯优化器，基于 BoTorch GP 模型和 EHVI/qNEHVI 采集函数。

    使用示例:
        opt = BayesianOptimizer(
            feature_names=['cement', 'silica_fume', ...],
            target_names=['compressive_strength_28d', 'cost'],
            directions=['maximize', 'minimize'],
            surrogate_type='gp',
            acq_type='qnehvi',
        )
        results = opt.optimize(X_init, Y_init, n_iter=50)
        pareto_X, pareto_Y = results['pareto_frontier']

    Attributes:
        feature_names: 特征名列表
        target_names: 目标名列表
        directions: 目标方向
        surrogate_type: 'gp' / 'rf' / 'ensemble'
        acq_type: 'ehvi' / 'qnehvi'
        n_iter: 优化迭代次数
        bounds: 特征边界 (2, d) torch 张量
        device: torch 设备
    """

    SURROGATE_TYPES = ["gp", "rf", "ensemble"]
    ACQ_TYPES = ["ehvi", "qnehvi"]

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        directions: Optional[List[str]] = None,
        surrogate_type: str = "gp",
        acq_type: str = "qnehvi",
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        # ── 验证参数 ──────────────────────────────────────────
        if surrogate_type not in self.SURROGATE_TYPES:
            raise ValueError(
                f"surrogate_type 必须是 {self.SURROGATE_TYPES}，收到: {surrogate_type}"
            )
        if acq_type not in self.ACQ_TYPES:
            raise ValueError(
                f"acq_type 必须是 {self.ACQ_TYPES}，收到: {acq_type}"
            )
        if not HAS_TORCH and surrogate_type in ("gp", "ensemble"):
            raise ImportError("GP/Ensemble 代理模型需要 PyTorch，请: pip install torch")
        if not HAS_BOTORCH and surrogate_type in ("gp", "ensemble"):
            raise ImportError("GP/Ensemble 代理模型需要 BoTorch，请: pip install botorch")

        self.feature_names = feature_names or get_all_feature_names()
        self.target_names = target_names or get_all_target_names()
        self.directions = directions or _validate_directions(self.target_names)
        self.surrogate_type = surrogate_type
        self.acq_type = acq_type
        self.random_state = random_state

        # ── 设备 ──────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── 状态 ──────────────────────────────────────────────
        self._trained = False
        self._models: List = []  # GP 模型列表（每个目标一个）
        self._rf_model = None  # RF fallback: MultiTargetTrainer
        self._rf_predictor = None  # RF predict function
        self.pareto_collector: Optional[ParetoCollector] = None
        self.bounds: Optional[torch.Tensor] = None
        self._cont_mask: List[int] = []  # 连续特征索引
        self._cont_names: List[str] = []  # 连续特征名

        # ── 结果 ──────────────────────────────────────────────
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None
        self.optimization_log: List[Dict] = []

        # ── 预处理 ────────────────────────────────────────────
        self._cont_mask = _get_continuous_mask(self.feature_names)
        self._cont_names = [self.feature_names[i] for i in self._cont_mask]
        self._cat_features = [
            f for f in self.feature_names if f in get_categorical_features()
        ]

        np.random.seed(random_state)
        if HAS_TORCH:
            torch.manual_seed(random_state)

    # ── 代理模型 ─────────────────────────────────────────────────

    def _init_bounds(self):
        """初始化特征边界张量"""
        self.bounds = _build_bounds_tensor(self._cont_names, self.device)

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """numpy → torch（自动检测并裁剪为连续特征）"""
        return torch.tensor(
            self._to_cont(X), device=self.device, dtype=torch.float64
        )

    def _to_tensor_y(self, Y: np.ndarray) -> torch.Tensor:
        """目标值 numpy → torch，并转为最小化"""
        Y_norm = _normalize_for_minimization(Y, self.directions)
        return torch.tensor(Y_norm, device=self.device, dtype=torch.float64)

    def _create_gp_surrogate(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        """
        创建 GP 代理模型（每个目标一个独立 SingleTaskGP）。
        """
        if not HAS_BOTORCH:
            raise ImportError("需要安装 BoTorch")

        from botorch.models import SingleTaskGP, ModelListGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize

        d = train_X.shape[-1]
        models = []
        for j in range(train_Y.shape[-1]):
            # 每个目标使用独立 GP，含输入归一化和输出标准化
            gp = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y[:, j : j + 1],
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
            )
            models.append(gp)

        self._models = models
        return ModelListGP(*models)

    def _fit_gp(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        """拟合 GP 模型"""
        if not HAS_BOTORCH:
            return

        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        for j, model in enumerate(self._models):
            model.train()
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                # 拟合失败不中断，保留上次模型
                pass
            model.eval()

    def _create_rf_surrogate(self, X: np.ndarray, Y: np.ndarray):
        """创建 RF 代理模型"""
        from core.model_trainer import MultiTargetTrainer

        X_cont = self._to_cont(X)
        Y_norm = _normalize_for_minimization(Y, self.directions)
        target_df = pd.DataFrame(Y_norm, columns=self.target_names)

        trainer = MultiTargetTrainer(model_type="random_forest")
        trainer.train_all(
            pd.DataFrame(X_cont, columns=self._cont_names), target_df
        )
        self._rf_model = trainer

    def _to_cont(self, X: np.ndarray) -> np.ndarray:
        """将完整特征矩阵裁剪为连续特征矩阵（自动检测输入维度）。"""
        n_cols = X.shape[1]
        n_full = len(self.feature_names)
        n_cont = len(self._cont_mask)
        if n_cols == n_cont and n_cont < n_full:
            return X  # 已经是连续特征
        elif self._cont_mask:
            return X[:, self._cont_mask]
        return X

    def _rf_predict(self, X: np.ndarray) -> np.ndarray:
        """RF 预测（返回统一最小化后的值）"""
        if self._rf_model is None:
            raise RuntimeError("RF 模型未训练")
        X_cont = self._to_cont(X)
        pred_df = self._rf_model.predict_all(
            pd.DataFrame(X_cont, columns=self._cont_names)
        )
        return pred_df.values

    def _surrogate_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        代理模型预测。
        Returns:
            (mean, std) — 预测均值和标准差（统一最小化空间）
        """
        if self.surrogate_type in ("gp", "ensemble"):
            X_t = self._to_tensor(X)
            means, stds = [], []
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for model in self._models:
                    post = model.posterior(X_t)
                    m = post.mean.cpu().numpy().flatten()
                    s = post.variance.sqrt().cpu().numpy().flatten()
                    means.append(m)
                    stds.append(s)
            gp_mean = np.column_stack(means)
            gp_std = np.column_stack(stds)

            if self.surrogate_type == "gp":
                return gp_mean, gp_std

        if self.surrogate_type in ("rf", "ensemble"):
            rf_mean = self._rf_predict(X)
            # RF 无原生不确定性，用预测值范围估计
            rf_std = np.full_like(rf_mean, 0.05 * np.abs(rf_mean) + 0.01)

            if self.surrogate_type == "rf":
                return rf_mean, rf_std

        # ensemble: 平均 GP 和 RF 预测
        mean = 0.5 * (gp_mean + rf_mean)
        std = np.sqrt(0.5 * (gp_std**2 + rf_std**2))
        return mean, std

    # ── 采集函数优化 ─────────────────────────────────────────────

    def _create_acq_func(self, ref_point: torch.Tensor):
        """创建多目标采集函数"""
        if not HAS_BOTORCH:
            raise ImportError("需要安装 BoTorch")

        from botorch.acquisition.multi_objective import (
            qExpectedHypervolumeImprovement,
            qNoisyExpectedHypervolumeImprovement,
        )
        from botorch.models import ModelListGP

        model_list = ModelListGP(*self._models)

        if self.acq_type == "ehvi":
            return qExpectedHypervolumeImprovement(
                model=model_list,
                ref_point=ref_point,
                partitioning=model_list,
            )
        else:  # qnehvi
            return qNoisyExpectedHypervolumeImprovement(
                model=model_list,
                ref_point=ref_point,
                X_baseline=self._to_tensor(self.X_train),
                prune_baseline=True,
            )

    def _generate_initial_candidates(
        self, n_candidates: int = 5000
    ) -> torch.Tensor:
        """使用 Sobol 序列 + 随机采样生成候选点"""
        if not HAS_TORCH:
            return torch.rand(n_candidates, len(self._cont_names), dtype=torch.float64)

        try:
            from botorch.utils.sampling import draw_sobol_samples

            sobol = draw_sobol_samples(
                bounds=self.bounds, n=1, q=n_candidates, seed=self.random_state
            ).squeeze(0)
            return sobol
        except Exception:
            # Sobol 失败时回退到均匀随机
            lb = self.bounds[0]
            ub = self.bounds[1]
            return lb + (ub - lb) * torch.rand(
                n_candidates, len(self._cont_names),
                device=self.device, dtype=torch.float64
            )

    def _optimize_acqf_and_get_candidate(
        self, ref_point: torch.Tensor
    ) -> np.ndarray:
        """
        优化采集函数并返回下一个候选点（连续特征值）。
        """
        if not HAS_BOTORCH or self.surrogate_type == "rf":
            # 无 BoTorch 或 RF 模式：随机采样 + 预测 → 近似最优
            return self._rf_best_candidate()

        from botorch.optim import optimize_acqf

        acq_func = self._create_acq_func(ref_point)

        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            options={"maxiter": 200},
        )
        return candidate.detach().cpu().numpy().reshape(1, -1)

    def _rf_best_candidate(self) -> np.ndarray:
        """RF 模式：随机采样后用预测值挑选最佳候选"""
        n_random = 5000
        lb = self.bounds[0].cpu().numpy() if HAS_TORCH else np.zeros(len(self._cont_names))
        ub = self.bounds[1].cpu().numpy() if HAS_TORCH else np.ones(len(self._cont_names))

        X_candidates = np.random.uniform(lb, ub, size=(n_random, len(self._cont_names)))

        # 通过约束过滤
        X_checked = self._filter_by_constraints(X_candidates)
        if len(X_checked) == 0:
            # 无可行点，返回边界中点
            return (lb + ub).reshape(1, -1) / 2

        means, stds = self._surrogate_predict(X_checked)
        # 简单 EI 近似: mean + std (最大化不确定性+最小化目标)
        combined = means + 0.1 * stds
        # 选综合最好的
        best_idx = int(np.argmin(np.sum(combined, axis=1)))
        return X_checked[best_idx : best_idx + 1]

    def _filter_by_constraints(self, X_cont: np.ndarray) -> np.ndarray:
        """根据 UHPC 物理约束过滤候选点"""
        if len(X_cont) == 0:
            return X_cont

        # 构建完整 X（含分类特征默认值）
        X_full = self._reconstruct_full_X(X_cont)
        mask = self._check_uhpc_constraints(X_full)
        return X_cont[mask]

    def _reconstruct_full_X(self, X_cont: np.ndarray) -> np.ndarray:
        """
        将连续特征矩阵重建为完整特征矩阵。
        分类特征使用默认值填充。
        """
        n = len(X_cont)
        d = len(self.feature_names)
        X_full = np.zeros((n, d))

        # 填充连续特征
        for j, idx in enumerate(self._cont_mask):
            X_full[:, idx] = X_cont[:, j]

        # 填充分类特征（默认值）
        for name in self._cat_features:
            idx = self.feature_names.index(name)
            cat_vals = self._get_cat_values(name)
            X_full[:, idx] = cat_vals[0] if cat_vals else 90  # 默认 90°C

        return X_full

    def _get_cat_values(self, feature_name: str) -> List:
        """获取分类特征的允许值"""
        from core.config import _load_config

        cfg = _load_config()
        for group in ["composition", "fiber", "curing"]:
            for item in cfg[group]:
                if item["name"] == feature_name and "values" in item:
                    return item["values"]
        return []

    def _check_uhpc_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        检查 UHPC 物理约束。
        Returns:
            bool 数组，True = 可行
        """
        n = len(X)
        feasible = np.ones(n, dtype=bool)

        # 约束 1: 水胶比 ∈ [0.14, 0.24]
        # w/b = water / (cement + silica_fume + quartz_powder)
        try:
            w_idx = self.feature_names.index("water")
            c_idx = self.feature_names.index("cement")
            sf_idx = self.feature_names.index("silica_fume")
            qp_idx = self.feature_names.index("quartz_powder")

            water = X[:, w_idx]
            binder = X[:, c_idx] + X[:, sf_idx] + X[:, qp_idx]
            binder = np.where(binder == 0, 1.0, binder)  # 避免除零
            wb_ratio = water / binder

            feasible &= (wb_ratio >= 0.14) & (wb_ratio <= 0.24)
        except ValueError:
            pass  # 特征名不匹配，跳过约束

        return feasible

    # ── 优化循环 ─────────────────────────────────────────────────

    def _initialize_data(self, X: np.ndarray, Y: np.ndarray):
        """初始化训练数据"""
        self._init_bounds()

        # 过滤掉违反约束的点
        mask = self._check_uhpc_constraints(X)
        if not mask.all():
            X = X[mask]
            Y = Y[mask]

        self.X_train = X.astype(np.float64)
        self.Y_train = Y.astype(np.float64)

        # 初始化 Pareto 收集器
        self.pareto_collector = ParetoCollector(
            directions=self.directions,
        )
        self.pareto_collector.update(self.X_train, self.Y_train, iteration=0)

        self.optimization_log = []

    def _fit_surrogates(self):
        """训练所有代理模型"""
        # 准备标准化输入
        X_cont = self._to_cont(self.X_train)

        if self.surrogate_type in ("gp", "ensemble"):
            train_X = torch.tensor(
                X_cont, device=self.device, dtype=torch.float64
            )
            train_Y = self._to_tensor_y(self.Y_train)
            self._create_gp_surrogate(train_X, train_Y)
            self._fit_gp(train_X, train_Y)

        if self.surrogate_type in ("rf", "ensemble"):
            self._create_rf_surrogate(self.X_train, self.Y_train)

        self._trained = True

    def step(self, iteration: int = 0) -> Dict:
        """
        执行一步贝叶斯优化。

        Returns:
            dict: {candidate_X, predicted_Y, hypervolume, n_frontier, ...}
        """
        if not self._trained:
            self._fit_surrogates()

        # 参考点
        Y_norm = _normalize_for_minimization(self.Y_train, self.directions)
        ref_point = compute_reference_point(self.Y_train, self.directions)
        ref_tensor = torch.tensor(ref_point, device=self.device, dtype=torch.float64)

        # 优化采集函数 → 候选点（连续特征空间）
        try:
            X_cand_cont = self._optimize_acqf_and_get_candidate(ref_tensor)
        except Exception as e:
            # 采集优化失败，使用随机搜索
            X_cand_cont = self._rf_best_candidate()

        # 预测候选点目标值
        Y_pred_norm, Y_pred_std = self._surrogate_predict(X_cand_cont)
        Y_pred = _denormalize_objectives(Y_pred_norm, self.directions)

        # 重建完整 X（含分类特征）
        X_cand_full = self._reconstruct_full_X(X_cand_cont)

        # 更新训练集
        self.X_train = np.vstack([self.X_train, X_cand_full])
        self.Y_train = np.vstack([self.Y_train, Y_pred])

        # 更新 Pareto 收集器
        pareto_info = self.pareto_collector.update(
            X_cand_full, Y_pred, iteration=iteration
        )

        # 记录日志
        log_entry = {
            "iteration": iteration,
            "candidate": X_cand_full[0].tolist(),
            "predicted_y": Y_pred[0].tolist(),
            "predicted_std": Y_pred_std[0].tolist(),
            **pareto_info,
        }
        self.optimization_log.append(log_entry)

        # 重新拟合（每步更新）
        self._trained = False

        return log_entry

    def optimize(
        self,
        X_init: np.ndarray,
        Y_init: np.ndarray,
        n_iter: int = 50,
        verbose: bool = True,
        callback: Optional[callable] = None,
    ) -> Dict:
        """
        运行完整多目标贝叶斯优化。

        Args:
            X_init: (n, d) 初始输入
            Y_init: (n, m) 初始目标值
            n_iter: 迭代次数
            verbose: 是否打印进度
            callback: 每步回调 fn(log_entry)

        Returns:
            dict: {
                pareto_X, pareto_Y, hypervolume_history,
                optimization_log, n_iter_completed
            }
        """
        self._initialize_data(X_init, Y_init)
        self._fit_surrogates()

        for i in range(1, n_iter + 1):
            log_entry = self.step(iteration=i)
            if verbose and i % max(1, n_iter // 10) == 0:
                print(
                    f"  Iter {i:4d}/{n_iter} | "
                    f"HV={log_entry['hypervolume']:.4f} | "
                    f"Frontier={log_entry['n_frontier']}"
                )
            if callback:
                callback(log_entry)

        pX, pY = self.pareto_collector.get_frontier()

        return {
            "pareto_X": pX,
            "pareto_Y": pY,
            "hypervolume_history": self.pareto_collector.get_history_df(),
            "optimization_log": self.optimization_log,
            "n_iter_completed": n_iter,
            "X_all": self.X_train,
            "Y_all": self.Y_train,
        }

    # ── 结果查询 ─────────────────────────────────────────────────

    def get_results(self) -> Dict:
        """获取当前优化结果摘要"""
        if self.pareto_collector is None:
            return {"error": "尚未运行优化"}
        pX, pY = self.pareto_collector.get_frontier()
        return {
            "n_total": self.pareto_collector.n_points,
            "n_frontier": len(pX),
            "hypervolume": self.pareto_collector.get_hypervolume(),
            "pareto_X": pX,
            "pareto_Y": pY,
            "hypervolume_history": self.pareto_collector.get_history_df(),
            "optimization_log": self.optimization_log,
        }

    def get_pareto_df(self) -> pd.DataFrame:
        """返回 Pareto 前沿点 DataFrame（包含特征+目标）"""
        pX, pY = self.pareto_collector.get_frontier()
        if len(pX) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(pX, columns=self.feature_names)
        for j, name in enumerate(self.target_names):
            df[f"{name}_pred"] = pY[:, j]
        df["rank"] = range(1, len(df) + 1)
        return df

    def get_convergence_df(self) -> pd.DataFrame:
        """返回收敛数据 DataFrame"""
        if self.pareto_collector is None:
            return pd.DataFrame()
        return self.pareto_collector.get_history_df()

    def get_best_point(self) -> Dict:
        """
        返回 trade-off 最优的单个点（TOPSIS 简化版：到理想点距离最小）。
        """
        pX, pY = self.pareto_collector.get_frontier()
        if len(pX) == 0:
            return {}
        Y_norm = _normalize_for_minimization(pY, self.directions)
        # 归一化到 [0,1]
        y_min = Y_norm.min(axis=0)
        y_max = Y_norm.max(axis=0)
        y_range = y_max - y_min
        y_range[y_range == 0] = 1.0
        y_scaled = (Y_norm - y_min) / y_range
        # 理想点 = 原点
        dists = np.sqrt(np.sum(y_scaled**2, axis=1))
        best_idx = int(np.argmin(dists))
        return {
            "X": pX[best_idx],
            "Y": pY[best_idx],
            "distance": float(dists[best_idx]),
            "index": best_idx,
        }

    # ── 持久化 ───────────────────────────────────────────────────

    def save(self, path: str):
        """保存优化器状态"""
        import joblib

        state = {
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "directions": self.directions,
            "surrogate_type": self.surrogate_type,
            "acq_type": self.acq_type,
            "X_train": self.X_train,
            "Y_train": self.Y_train,
            "optimization_log": self.optimization_log,
        }
        if self.pareto_collector is not None:
            state["history"] = self.pareto_collector.history
            state["X_all"] = self.pareto_collector.X_all
            state["Y_all"] = self.pareto_collector.Y_all
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(state, path)

    def load(self, path: str):
        """加载优化器状态"""
        import joblib

        state = joblib.load(path)
        self.feature_names = state["feature_names"]
        self.target_names = state["target_names"]
        self.directions = state["directions"]
        self.surrogate_type = state.get("surrogate_type", "gp")
        self.acq_type = state.get("acq_type", "qnehvi")
        self.X_train = state["X_train"]
        self.Y_train = state["Y_train"]
        self.optimization_log = state.get("optimization_log", [])

        # 重建 Pareto 收集器
        self.pareto_collector = ParetoCollector(directions=self.directions)
        if "X_all" in state and state["X_all"] is not None:
            self.pareto_collector.X_all = state["X_all"]
            self.pareto_collector.Y_all = state["Y_all"]
            self.pareto_collector.Y_norm = _normalize_for_minimization(
                state["Y_all"], self.directions
            )
            self.pareto_collector.history = state.get("history", [])
        self._trained = False
        return self


# ═══════════════════════════════════════════════════════════════════
# 快捷函数
# ═══════════════════════════════════════════════════════════════════

def create_bayesian_optimizer(
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
    surrogate_type: str = "gp",
    acq_type: str = "qnehvi",
) -> BayesianOptimizer:
    """
    快速创建贝叶斯优化器的便捷函数。
    自动从 config 读取特征/目标/方向配置。
    """
    features = feature_names or get_all_feature_names()
    targets = target_names or get_all_target_names()
    directions = _validate_directions(targets)
    return BayesianOptimizer(
        feature_names=features,
        target_names=targets,
        directions=directions,
        surrogate_type=surrogate_type,
        acq_type=acq_type,
    )
