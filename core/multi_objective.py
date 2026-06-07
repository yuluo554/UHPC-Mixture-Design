# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign 多目标进化优化模块
基于 pymoo 的 NSGA-II / UNSGA-III / AGE-MOEA 多目标优化

核心类：
- UHPCProblem: pymoo.Problem 子类，嵌入 UHPC 物理约束（水胶比 + 体积和）
- MultiObjectiveOptimizer: 多目标进化优化器，支持算法切换 + 代理模型评估

使用方式：
    from core.multi_objective import MultiObjectiveOptimizer, create_multi_objective_optimizer

    # 便捷创建
    opt = create_multi_objective_optimizer(
        algorithm="nsga2",
        surrogate_model=trained_multi_target_model,
        pop_size=100,
    )
    results = opt.optimize(n_generations=200)
    df = opt.get_pareto_df()
"""

import os
import sys
import warnings
import time
from typing import Optional, Dict, List, Tuple, Union, Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── 可选依赖检查 ─────────────────────────────────────────────────
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.age import AGEMOEA
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions

    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False

# 本地模块
from core.config import (
    get_all_feature_names,
    get_all_target_names,
    get_feature_ranges,
    get_categorical_features,
    get_target_direction,
    get_feature_label,
    get_constraints,
)

# ═══════════════════════════════════════════════════════════════════
# UHPC 物理常量
# ═══════════════════════════════════════════════════════════════════

# 材料密度 (kg/m³) — 用于体积和约束计算
MATERIAL_DENSITY = {
    "cement": 3150.0,
    "silica_fume": 2200.0,
    "quartz_powder": 2650.0,
    "quartz_sand": 2650.0,
    "superplasticizer": 1100.0,
    "water": 1000.0,
    "steel_fiber": 7850.0,
}

# 水胶比约束范围
WB_MIN, WB_MAX = 0.14, 0.24

# 体积和容差
VOLUME_TOLERANCE = 0.05

# curing_temp 离散值映射
CURING_TEMP_VALUES = np.array([20, 60, 90, 250], dtype=np.float64)
CURING_TEMP_ENCODED_MIN, CURING_TEMP_ENCODED_MAX = 0, 3


# ═══════════════════════════════════════════════════════════════════
# UHPCProblem — pymoo Problem 子类
# ═══════════════════════════════════════════════════════════════════

class UHPCProblem(Problem):
    """
    UHPC 多目标优化问题定义。

    嵌入物理约束：
    1. 水胶比 ∈ [0.14, 0.24]（2 个不等式约束）
    2. 各组分体积和 ≈ 1 m³（2 个不等式约束）

    目标函数值由外部代理模型（RF/GP）评估。

    Parameters
    ----------
    feature_names : list[str]
        12 个输入特征名。
    target_names : list[str]
        5 个优化目标名。
    directions : list[str]
        各目标方向: 'maximize' / 'minimize' / 'target'。
    xl : np.ndarray
        各变量下界。
    xu : np.ndarray
        各变量上界。
    surrogate_predict : callable
        代理模型预测函数，签名 f(X) -> (Y_mean, Y_std)。
    feature_ranges : dict, optional
        特征范围 {name: (min, max)}。
    """

    def __init__(
        self,
        feature_names: List[str],
        target_names: List[str],
        directions: List[str],
        xl: np.ndarray,
        xu: np.ndarray,
        surrogate_predict: Callable,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.directions = list(directions)
        self._surrogate_predict = surrogate_predict
        self.feature_ranges = feature_ranges or {}

        n_var = len(feature_names)
        n_obj = len(target_names)
        n_constr = 4  # w/b 下限 + w/b 上限 + 体积下限 + 体积上限

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        评估种群个体的目标值和约束违反量。

        pymoo 约束语义：G_i(x) ≤ 0 表示可行。

        Parameters
        ----------
        X : np.ndarray, shape (n, n_var)
            设计变量矩阵。
        out : dict
            输出字典，需设置 'F'（目标值）和 'G'（约束值）。
        """
        n = X.shape[0]

        # ── 目标函数：通过代理模型预测 ──
        try:
            Y_mean, _ = self._surrogate_predict(X)
            # 确保形状正确
            if Y_mean.shape[0] != n:
                Y_mean = np.zeros((n, len(self.target_names)))
        except Exception:
            Y_mean = np.zeros((n, len(self.target_names)))

        # 统一转换为最小化问题
        F = np.zeros((n, len(self.target_names)))
        for i, d in enumerate(self.directions):
            if d == "maximize":
                F[:, i] = -Y_mean[:, i]
            elif d == "target":
                # 目标型：到中位数的绝对距离
                median_val = float(np.median(Y_mean[:, i])) if n > 0 else 200.0
                F[:, i] = np.abs(Y_mean[:, i] - median_val)
            else:  # minimize
                F[:, i] = Y_mean[:, i]

        out["F"] = F

        # ── 物理约束 ──
        G = np.zeros((n, self.n_constr))

        # 定位关键特征索引
        try:
            idx_cement = self.feature_names.index("cement")
            idx_silica_fume = self.feature_names.index("silica_fume")
            idx_water = self.feature_names.index("water")
        except ValueError:
            out["G"] = G
            return

        cement = X[:, idx_cement]
        silica_fume = X[:, idx_silica_fume]
        water = X[:, idx_water]
        binder = np.maximum(cement + silica_fume, 1e-6)

        wb_ratio = water / binder

        # 约束 0: w/b >= WB_MIN  →  G_0 = WB_MIN - w/b ≤ 0
        G[:, 0] = WB_MIN - wb_ratio

        # 约束 1: w/b <= WB_MAX  →  G_1 = w/b - WB_MAX ≤ 0
        G[:, 1] = wb_ratio - WB_MAX

        # 约束 2-3: 体积和 ≈ 1 m³
        vol_sum = self._compute_volume_sum(X)
        G[:, 2] = (1.0 - VOLUME_TOLERANCE) - vol_sum
        G[:, 3] = vol_sum - (1.0 + VOLUME_TOLERANCE)

        out["G"] = G

    def _compute_volume_sum(self, X: np.ndarray) -> np.ndarray:
        """
        计算各组分体积和（不含纤维时的近似 1 m³）。

        Parameters
        ----------
        X : np.ndarray, shape (n, n_var)

        Returns:
            np.ndarray, shape (n,): 各组分的近似体积和
        """
        n = X.shape[0]
        vol = np.zeros(n)

        # 组成材料体积
        for comp, density in [
            ("cement", MATERIAL_DENSITY["cement"]),
            ("silica_fume", MATERIAL_DENSITY["silica_fume"]),
            ("quartz_powder", MATERIAL_DENSITY["quartz_powder"]),
            ("quartz_sand", MATERIAL_DENSITY["quartz_sand"]),
            ("superplasticizer", MATERIAL_DENSITY["superplasticizer"]),
            ("water", MATERIAL_DENSITY["water"]),
        ]:
            if comp in self.feature_names:
                idx = self.feature_names.index(comp)
                vol += X[:, idx] / density

        # 纤维体积（fiber_volume 是百分比，需转换为体积分数）
        if "fiber_volume" in self.feature_names:
            idx = self.feature_names.index("fiber_volume")
            vol += X[:, idx] / 100.0

        return vol


# ═══════════════════════════════════════════════════════════════════
# MultiObjectiveOptimizer
# ═══════════════════════════════════════════════════════════════════

class MultiObjectiveOptimizer:
    """
    多目标进化优化器。

    集成 pymoo 的 NSGA-II / UNSGA-III / AGE-MOEA 算法，
    使用预训练的代理模型（RF/GP）进行适应度评估。

    Parameters
    ----------
    feature_names : list[str], optional
        输入特征名。默认从 config 读取。
    target_names : list[str], optional
        优化目标名。默认从 config 读取。
    directions : list[str], optional
        优化方向。默认从 config 读取。
    algorithm : str
        算法标识: 'nsga2' / 'nsga3' / 'agemoea'。
    surrogate_model : object, optional
        预训练的代理模型，需实现 predict_all(X_df) -> DataFrame。
    pop_size : int
        种群大小。
    verbose : bool
        是否打印进度信息。
    """

    # 支持的算法映射
    ALGORITHMS = {
        "nsga2": "NSGA-II",
        "nsga3": "NSGA-III",
        "agemoea": "AGE-MOEA",
    }

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        directions: Optional[List[str]] = None,
        algorithm: str = "nsga2",
        surrogate_model=None,
        pop_size: int = 100,
        verbose: bool = True,
    ):
        if not HAS_PYMOO:
            raise ImportError(
                "pymoo 未安装，请执行: python -m pip install pymoo"
            )

        self.feature_names = feature_names or get_all_feature_names()
        self.target_names = target_names or get_all_target_names()
        self.directions = directions or [
            get_target_direction(t) for t in self.target_names
        ]
        self.algorithm = algorithm.lower()
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"不支持的算法: {algorithm}。可选: {list(self.ALGORITHMS.keys())}"
            )
        self.surrogate_model = surrogate_model
        self.pop_size = pop_size
        self.verbose = verbose

        # 获取特征范围
        all_ranges = get_feature_ranges()
        self.feature_ranges = {
            name: all_ranges[name]
            for name in self.feature_names
            if name in all_ranges
        }

        # 构建变量边界
        self._build_bounds()

        # 结果存储
        self.problem: Optional[UHPCProblem] = None
        self.result = None
        self.pareto_X: Optional[np.ndarray] = None
        self.pareto_Y: Optional[np.ndarray] = None
        self.history: List[Dict] = []
        self.optimization_time: float = 0.0

    def _build_bounds(self):
        """构建 pymoo 变量边界数组。"""
        xl_list, xu_list = [], []
        for name in self.feature_names:
            if name in self.feature_ranges:
                lo, hi = self.feature_ranges[name]
            elif name == "curing_temp":
                lo, hi = CURING_TEMP_ENCODED_MIN, CURING_TEMP_ENCODED_MAX
            elif name == "curing_time":
                lo, hi = 24, 168
            else:
                lo, hi = 0, 1
            xl_list.append(float(lo))
            xu_list.append(float(hi))

        self.xl = np.array(xl_list, dtype=np.float64)
        self.xu = np.array(xu_list, dtype=np.float64)

    def _decode_curing_temp(self, X: np.ndarray) -> np.ndarray:
        """
        将 curing_temp 从编码值 [0,3] 映射回实际温度 {20, 60, 90, 250}°C。

        Parameters
        ----------
        X : np.ndarray, shape (n, n_var)

        Returns:
            np.ndarray: 处理后的 X（复制）
        """
        if "curing_temp" not in self.feature_names:
            return X

        idx = self.feature_names.index("curing_temp")
        X_out = X.copy()
        # 将 [0, 3] 连续值取整并映射到离散温度
        encoded = np.clip(np.round(X[:, idx]).astype(int), 0, 3)
        X_out[:, idx] = CURING_TEMP_VALUES[encoded]
        return X_out

    def _surrogate_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        代理模型预测。

        自动处理 curing_temp 的反编码：将 [0,3] 映射到 {20, 60, 90, 250}。

        Parameters
        ----------
        X : np.ndarray, shape (n, n_var)
            设计变量（含编码的 curing_temp）。

        Returns:
            Y_mean : np.ndarray, shape (n, n_obj)
                目标预测均值。
            Y_std : np.ndarray, shape (n, n_obj)
                目标预测标准差（RF 模式下为经验估计）。
        """
        X_decoded = self._decode_curing_temp(X)
        n = X_decoded.shape[0]
        n_obj = len(self.target_names)

        if self.surrogate_model is not None:
            try:
                X_df = pd.DataFrame(X_decoded, columns=self.feature_names)
                Y_pred = self.surrogate_model.predict_all(X_df)
                if isinstance(Y_pred, pd.DataFrame):
                    Y_mean = Y_pred.values.astype(np.float64)
                else:
                    Y_mean = np.asarray(Y_pred, dtype=np.float64)

                # RF 不提供标准差 → 使用预测均值的 5% 作为经验估计
                Y_std = np.full((n, n_obj), 0.05 * np.abs(Y_mean).mean())
                Y_std[Y_std < 1e-6] = 1.0
            except Exception as e:
                if self.verbose:
                    print(f"  [警告] 代理模型预测失败: {e}，回退到零值")
                Y_mean = np.zeros((n, n_obj))
                Y_std = np.ones((n, n_obj))
        else:
            # 无代理模型 — 使用虚拟值（仅用于功能测试）
            Y_mean = np.zeros((n, n_obj))
            Y_std = np.ones((n, n_obj))

        return Y_mean, Y_std

    def _create_algorithm(self):
        """根据配置创建 pymoo 算法实例。"""
        algo = self.algorithm

        if algo == "nsga2":
            return NSGA2(
                pop_size=self.pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=1.0 / len(self.feature_names), eta=20),
                eliminate_duplicates=True,
            )
        elif algo == "nsga3":
            n_obj = len(self.target_names)
            # 生成参考方向 — 优先 das-dennis，失败回退 energy
            n_partitions = max(2, min(5, 12 // n_obj))  # 自适应分区数
            try:
                ref_dirs = get_reference_directions(
                    "das-dennis", n_obj, n_partitions=n_partitions
                )
            except Exception:
                ref_dirs = get_reference_directions(
                    "energy", n_obj, n_points=self.pop_size
                )

            return NSGA3(
                pop_size=self.pop_size,
                ref_dirs=ref_dirs,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=1.0 / len(self.feature_names), eta=20),
                eliminate_duplicates=True,
            )
        elif algo == "agemoea":
            return AGEMOEA(
                pop_size=self.pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=1.0 / len(self.feature_names), eta=20),
                eliminate_duplicates=True,
            )

    def _create_problem(self) -> UHPCProblem:
        """创建 UHPCProblem 实例。"""
        return UHPCProblem(
            feature_names=self.feature_names,
            target_names=self.target_names,
            directions=self.directions,
            xl=self.xl,
            xu=self.xu,
            surrogate_predict=self._surrogate_predict,
            feature_ranges=self.feature_ranges,
        )

    # ── 主优化方法 ────────────────────────────────────────────────

    def optimize(
        self,
        n_generations: int = 200,
        seed: int = 42,
        termination: Optional[str] = None,
        save_history: bool = True,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        执行多目标进化优化。

        Parameters
        ----------
        n_generations : int
            最大进化代数。
        seed : int
            随机种子。
        termination : str, optional
            pymoo 终止条件字符串（如 'n_gen 200'），
            未指定时等价于 ``get_termination("n_gen", n_generations)``。
        save_history : bool
            是否每代记录收敛历史。
        callback : callable, optional
            每代回调函数 ``callback(algorithm)``。

        Returns:
            dict: 包含以下键的字典 ——
            - pareto_X: np.ndarray, Pareto 最优设计变量
            - pareto_Y: np.ndarray, Pareto 最优目标值
            - n_pareto: int, Pareto 解数量
            - history: list[dict], 收敛历史
            - time: float, 优化耗时（秒）
            - algorithm: str, 使用的算法名
            - pop_size: int, 种群大小
        """
        algo_label = self.ALGORITHMS.get(self.algorithm, self.algorithm)
        if self.verbose:
            print(f"启动 {algo_label} 优化: pop={self.pop_size}, gen={n_generations}")

        self.problem = self._create_problem()
        algorithm = self._create_algorithm()

        if termination is None:
            termination_obj = get_termination("n_gen", n_generations)
        else:
            termination_obj = get_termination(termination)

        t0 = time.time()

        # ── 历史回调 ──
        if save_history:
            self.history = []

            class _HistoryCallback:
                def __init__(self, parent):
                    self.parent = parent

                def __call__(self, alg):
                    n_gen = alg.n_gen
                    n_nds = len(alg.opt) if alg.opt is not None else 0
                    self.parent.history.append({
                        "generation": n_gen,
                        "n_nondominated": n_nds,
                    })
                    if callback:
                        callback(alg)

            self.result = minimize(
                self.problem,
                algorithm,
                termination_obj,
                seed=seed,
                save_history=True,
                verbose=self.verbose,
                callback=_HistoryCallback(self),
            )
        else:
            self.result = minimize(
                self.problem,
                algorithm,
                termination_obj,
                seed=seed,
                save_history=True,
                verbose=self.verbose,
            )

        self.optimization_time = time.time() - t0

        # ── 提取 Pareto 前沿 ──
        if self.result.X is not None and self.result.F is not None:
            self.pareto_X = self._decode_curing_temp(self.result.X)
            # 还原原始目标值（取消最小化转换）
            self.pareto_Y = self._reverse_objectives(self.result.F)
        else:
            self.pareto_X = np.array([])
            self.pareto_Y = np.array([])

        if self.verbose:
            print(
                f"优化完成: {len(self.pareto_X)} 个 Pareto 最优解, "
                f"耗时 {self.optimization_time:.1f}s"
            )

        return self.get_results()

    def _reverse_objectives(self, F: np.ndarray) -> np.ndarray:
        """
        将最小化形式的目标值还原为原始方向。

        Parameters
        ----------
        F : np.ndarray, shape (n, n_obj)
            最小化形式的目标值。

        Returns:
            np.ndarray: 原始方向的目标值。
        """
        Y = F.astype(np.float64).copy()
        for i, d in enumerate(self.directions):
            if d == "maximize":
                Y[:, i] = -Y[:, i]
            # target 和 minimize 保持原样
        return Y

    # ── 结果查询 ──────────────────────────────────────────────────

    def get_results(self) -> Dict:
        """返回优化结果摘要字典。"""
        return {
            "pareto_X": self.pareto_X,
            "pareto_Y": self.pareto_Y,
            "n_pareto": len(self.pareto_X) if self.pareto_X is not None else 0,
            "history": self.history,
            "time": self.optimization_time,
            "algorithm": self.ALGORITHMS.get(self.algorithm, self.algorithm),
            "pop_size": self.pop_size,
        }

    def get_pareto_df(self) -> pd.DataFrame:
        """
        返回 Pareto 前沿的 DataFrame（特征 + 目标合一）。

        Returns:
            pd.DataFrame: 列包含 feature_names + target_names。
        """
        if self.pareto_X is None or len(self.pareto_X) == 0:
            return pd.DataFrame()

        df_X = pd.DataFrame(self.pareto_X, columns=self.feature_names)
        df_Y = pd.DataFrame(self.pareto_Y, columns=self.target_names)

        # curing_temp 取整显示
        if "curing_temp" in df_X.columns:
            df_X["curing_temp"] = df_X["curing_temp"].round().astype(int)

        df = pd.concat([df_X, df_Y], axis=1)
        return df

    def get_convergence_df(self) -> pd.DataFrame:
        """
        返回收敛历史 DataFrame。

        Returns:
            pd.DataFrame: 列 generation, n_nondominated。
        """
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    # ── 持久化 ────────────────────────────────────────────────────

    def save(self, path: str):
        """
        将优化结果保存为 joblib 文件。

        Parameters
        ----------
        path : str
            保存路径（建议 .pkl 或 .joblib 后缀）。
        """
        import joblib

        data = {
            "pareto_X": self.pareto_X,
            "pareto_Y": self.pareto_Y,
            "history": self.history,
            "time": self.optimization_time,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "directions": self.directions,
            "algorithm": self.algorithm,
            "pop_size": self.pop_size,
        }
        joblib.dump(data, path)

    def load(self, path: str) -> "MultiObjectiveOptimizer":
        """
        从 joblib 文件加载优化结果。

        Parameters
        ----------
        path : str
            文件路径。

        Returns:
            MultiObjectiveOptimizer: self（链式调用）。
        """
        import joblib

        # BUG-20260607-022 修复：使用白名单只恢复 save() 写入的合法字段，
        # 避免 setattr 全量覆盖导致类属性 ALGORITHMS（dict）被旧 pkl 中同名字段
        # 覆盖（虽然 type(self).ALGORITHMS 可恢复，但中间状态混乱）。
        # 同时兼容 "time"（历史 save()）和 "optimization_time"（prompt 中公开名）。
        _ALLOWED_LOAD_KEYS = {
            "pareto_X", "pareto_Y", "history",
            "time", "optimization_time",
            "feature_names", "target_names",
            "directions", "algorithm", "pop_size",
        }
        data = joblib.load(path)
        for k in _ALLOWED_LOAD_KEYS:
            if k in data:
                setattr(self, k, data[k])
        return self


# ═══════════════════════════════════════════════════════════════════
# 便捷创建函数
# ═══════════════════════════════════════════════════════════════════

def create_multi_objective_optimizer(
    algorithm: str = "nsga2",
    surrogate_model=None,
    pop_size: int = 100,
    verbose: bool = False,
) -> MultiObjectiveOptimizer:
    """
    便捷函数：从 config 模板自动创建 MultiObjectiveOptimizer。

    Parameters
    ----------
    algorithm : str
        算法标识: 'nsga2' / 'nsga3' / 'agemoea'。
    surrogate_model : object, optional
        预训练的 MultiTargetTrainer 实例。
    pop_size : int
        种群大小。
    verbose : bool
        是否打印进度。

    Returns:
        MultiObjectiveOptimizer
    """
    return MultiObjectiveOptimizer(
        feature_names=get_all_feature_names(),
        target_names=get_all_target_names(),
        directions=[get_target_direction(t) for t in get_all_target_names()],
        algorithm=algorithm,
        surrogate_model=surrogate_model,
        pop_size=pop_size,
        verbose=verbose,
    )
