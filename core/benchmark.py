# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign Benchmark 评估框架
====================================

参考 Bgolearn BGO_eval.py（MIT 许可证），适配 UHPC 多目标代理模型场景。

核心差异：
  - Bgolearn 内嵌 Kriging 模型 → UHPCBenchmark 通过 BayesianOptimizer 接口解耦
  - Bgolearn 使用解析函数（如 Rosenbrock）→ UHPCBenchmark 使用代理模型 + 实测数据
  - Bgolearn 返回 (mean_iter, var_iter) → UHPCBenchmark 返回 DataFrame 全记录

Trail/Count 算法来源：
  - Trail: Bgolearn BGO_eval.py:188-301
  - Count: Bgolearn BGO_eval.py:585-667

Copyright (c) 2023 caobin — MIT License
Ref: MultiBgolearn (arXiv:2601.06820)
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Optional


class UHPCBenchmark:
    """采集函数对比评估框架。

    在给定代理模型 + 数据上，对多种采集函数运行多次独立试验，
    记录收敛速度和成功率。

    使用示例::

        from core.benchmark import UHPCBenchmark

        bench = UHPCBenchmark(
            X_train, Y_train,
            feature_names, target_names,
            directions=['minimize'] * 5,
            seed=42,
        )
        result = bench.trail(
            acq_types=['ei', 'ucb', 'poi'],
            n_trials=10,
            max_iter=30,
            n_initial=10,
        )
        print(UHPCBenchmark.summary(result))
    """

    def __init__(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        feature_names: list[str],
        target_names: list[str],
        directions: Optional[list[str]] = None,
        surrogate_type: str = "rf",
        seed: int = 42,
    ) -> None:
        """初始化 Benchmark。

        Args:
            X_train: 训练特征 (n, d)
            Y_train: 训练目标 (n, m)
            feature_names: 特征名列表
            target_names: 目标名列表
            directions: 优化方向列表，默认全 'minimize'
            surrogate_type: 代理模型类型（'rf' / 'gp' / 'ensemble'）
            seed: 基础随机种子（每次试验递增）
        """
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.Y_train = np.asarray(Y_train, dtype=np.float64)
        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        if directions is None:
            directions = ["minimize"] * self.Y_train.shape[1]
        self.directions = list(directions)
        self.surrogate_type = surrogate_type
        self.seed = seed

        # 预计算全局最优（用于收敛判定）
        self._validate()

    def _validate(self) -> None:
        """验证输入形状一致性"""
        n, d = self.X_train.shape
        n2, m = self.Y_train.shape
        if n != n2:
            raise ValueError(
                f"X_train 和 Y_train 样本数不一致: {n} vs {n2}"
            )
        if d != len(self.feature_names):
            raise ValueError(
                f"feature_names 长度 {len(self.feature_names)} "
                f"与 X_train 列数 {d} 不匹配"
            )
        if m != len(self.target_names):
            raise ValueError(
                f"target_names 长度 {len(self.target_names)} "
                f"与 Y_train 列数 {m} 不匹配"
            )

    # ── Trail: 收敛速度统计 ──────────────────────────────────────

    def trail(
        self,
        acq_types: Optional[list[str]] = None,
        n_trials: int = 10,
        max_iter: int = 30,
        n_initial: int = 10,
        target_idx: int = 0,
        tol: float = 0.01,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """多轮收敛速度统计（对应 Bgolearn Trail）。

        对每种采集函数运行 n_trials 次独立试验，
        每次试验从随机 n_initial 个初始点出发，
        记录每轮迭代发现的当前最优值。

        Args:
            acq_types: 采集函数列表，默认 ['ei', 'ucb', 'poi', 'augmented_ei', 'thompson']
            n_trials: 每种采集函数的试验次数
            max_iter: 每次试验的 BO 迭代次数
            n_initial: 每次试验的初始训练点数
            target_idx: 评估目标索引
            tol: 收敛容差（达到 global_best * (1+tol) 视为收敛）
            verbose: 是否打印进度

        Returns:
            DataFrame，列为：
              - acq_type: 采集函数名
              - trial: 试验编号
              - iteration: 迭代编号（0..max_iter）
              - best_value: 该迭代的最优值
              - converged_at: 收敛的迭代编号（-1 = 未收敛）
        """
        if acq_types is None:
            acq_types = ["ei", "ucb", "poi", "augmented_ei", "thompson"]

        # 收敛判定必须在统一空间内进行：原实现 line 145 在原始空间算 global_best
        # （maximize 目标时是最小值=最差），line 188 在归一化空间算 best_so_far
        # （maximize 目标时是 -max=最优），两者跨空间比较导致 maximize 永远"立即收敛"。
        # 修复：global_best 也改到归一化空间计算（_normalize_for_minimization 把
        # maximize 取负，最小值 = 最优）。
        from core.bayesian_optimizer import _normalize_for_minimization

        Y_train_norm_for_global = _normalize_for_minimization(
            self.Y_train, self.directions
        )
        global_best = float(np.min(Y_train_norm_for_global[:, target_idx]))
        n_total = len(self.X_train)
        n_init = min(n_initial, n_total // 2)

        rows = []
        n_configs = len(acq_types) * n_trials

        for acq_idx, acq_type in enumerate(acq_types):
            for trial in range(n_trials):
                trial_seed = self.seed + trial * 100 + acq_idx
                np.random.seed(trial_seed)

                try:
                    # 随机抽样初始点
                    init_indices = np.random.choice(
                        n_total, size=n_init, replace=False
                    )
                    X_init = self.X_train[init_indices].copy()
                    Y_init = self.Y_train[init_indices].copy()

                    # 创建优化器
                    from core.bayesian_optimizer import BayesianOptimizer

                    opt = BayesianOptimizer(
                        feature_names=self.feature_names,
                        target_names=self.target_names,
                        directions=self.directions,
                        surrogate_type=self.surrogate_type,
                        acq_type=acq_type,
                        random_state=trial_seed,
                    )

                    # 初始化并拟合
                    opt._initialize_data(X_init, Y_init)
                    opt._fit_surrogates()

                    # 初始最优值（归一化空间 — _normalize_for_minimization 已在函数顶部导入）
                    Y_norm_init = _normalize_for_minimization(
                        Y_init, self.directions
                    )
                    best_so_far = float(np.min(Y_norm_init[:, target_idx]))
                    converged_at = -1

                    rows.append({
                        "acq_type": acq_type,
                        "trial": trial,
                        "iteration": 0,
                        "best_value": best_so_far,
                        "converged_at": -1,  # iteration 0 不判定收敛
                    })

                    # BO 迭代
                    for it in range(1, max_iter + 1):
                        log_entry = opt.step(iteration=it)

                        # 当前最优值（最小化空间）
                        Y_all_norm = _normalize_for_minimization(
                            opt.Y_train, self.directions
                        )
                        best_so_far = float(
                            np.min(Y_all_norm[:, target_idx])
                        )

                        # 收敛判定：global_best 和 best_so_far 都在归一化空间（统一为 minimize），
                        # 比较语义正确 —— 1% 容差应用于归一化空间的最优值。
                        if converged_at < 0 and best_so_far <= global_best * (
                            1.0 + tol
                        ):
                            converged_at = it

                        rows.append({
                            "acq_type": acq_type,
                            "trial": trial,
                            "iteration": it,
                            "best_value": best_so_far,
                            "converged_at": (
                                converged_at
                                if (converged_at >= 0 and it >= converged_at)
                                else -1
                            ),
                        })

                except Exception as e:
                    if verbose:
                        print(
                            f"  [WARN] {acq_type} trial {trial} 失败: {e}"
                        )
                    # 记录失败试验（仅标记行）
                    rows.append({
                        "acq_type": acq_type,
                        "trial": trial,
                        "iteration": 0,
                        "best_value": np.nan,
                        "converged_at": -2,  # -2 = 试验失败
                    })

                # 进度报告
                current = acq_idx * n_trials + trial + 1
                if verbose and current % max(1, n_configs // 5) == 0:
                    print(
                        f"  Trail 进度: {current}/{n_configs} "
                        f"({current / n_configs * 100:.0f}%)"
                    )

        result = pd.DataFrame(rows)
        # 填充 converged_at：将试验中最先出现的 converged_at 传播到后续行
        for (acq, tr), grp in result.groupby(["acq_type", "trial"]):
            conv_vals = grp["converged_at"].values
            if np.any(conv_vals > 0):
                conv_iter = int(np.min(conv_vals[conv_vals > 0]))
                mask = (
                    (result["acq_type"] == acq)
                    & (result["trial"] == tr)
                    & (result["iteration"] >= conv_iter)
                )
                result.loc[mask, "converged_at"] = conv_iter

        return result.reset_index(drop=True)

    # ── Count: 有限预算成功率 ─────────────────────────────────────

    def count(
        self,
        acq_types: Optional[list[str]] = None,
        n_trials: int = 30,
        max_iter: int = 10,
        n_initial: int = 10,
        target_idx: int = 0,
        tol: float = 0.01,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """有限预算成功率（对应 Bgolearn Count）。

        在 max_iter 有限预算内，统计每种采集函数的成功率。

        Args:
            acq_types: 采集函数列表，默认同上
            n_trials: 每种采集函数的试验次数
            max_iter: 每次试验的 BO 迭代次数（预算）
            n_initial: 每次试验的初始训练点数
            target_idx: 评估目标索引
            tol: 收敛容差
            verbose: 是否打印进度

        Returns:
            DataFrame，列为：
              - acq_type: 采集函数名
              - success_count: 成功收敛次数
              - total_trials: 总试验次数
              - success_rate: 成功率（0-1）
              - mean_converge_iter: 平均收敛迭代数（仅成功试验）
              - mean_best_value: 平均最终最优值
        """
        if acq_types is None:
            acq_types = ["ei", "ucb", "poi", "augmented_ei", "thompson"]

        trail_result = self.trail(
            acq_types=acq_types,
            n_trials=n_trials,
            max_iter=max_iter,
            n_initial=n_initial,
            target_idx=target_idx,
            tol=tol,
            verbose=verbose,
        )

        summary_rows = []
        for acq_type in acq_types:
            subset = trail_result[trail_result["acq_type"] == acq_type]
            # 排除失败试验（converged_at == -2）
            valid_trials = subset[subset["converged_at"] != -2]
            n_valid = valid_trials["trial"].nunique()

            if n_valid == 0:
                summary_rows.append({
                    "acq_type": acq_type,
                    "success_count": 0,
                    "total_trials": n_trials,
                    "success_rate": 0.0,
                    "mean_converge_iter": np.nan,
                    "mean_best_value": np.nan,
                })
                continue

            # 成功：最终 converged_at > 0
            final_rows = valid_trials.loc[
                valid_trials.groupby("trial")["iteration"].idxmax()
            ]
            success_mask = final_rows["converged_at"] > 0
            n_success = int(success_mask.sum())

            # 平均收敛迭代数（仅成功试验）
            if n_success > 0:
                mean_conv = float(
                    final_rows.loc[success_mask, "converged_at"].mean()
                )
            else:
                mean_conv = np.nan

            # 平均最终最优值
            mean_best = float(final_rows["best_value"].mean())

            summary_rows.append({
                "acq_type": acq_type,
                "success_count": n_success,
                "total_trials": n_valid,
                "success_rate": n_success / n_valid,
                "mean_converge_iter": mean_conv,
                "mean_best_value": mean_best,
            })

        return pd.DataFrame(summary_rows)

    # ── 一键对比 ──────────────────────────────────────────────────

    def compare(
        self,
        acq_types: Optional[list[str]] = None,
        n_trials: int = 10,
        max_iter: int = 30,
        n_initial: int = 10,
        target_idx: int = 0,
    ) -> dict[str, pd.DataFrame]:
        """一键对比：同时运行 trail + count，返回两个 DataFrame。

        Returns:
            dict: {'trail': DataFrame, 'count': DataFrame}
        """
        if acq_types is None:
            acq_types = ["ei", "ucb", "poi", "augmented_ei", "thompson"]

        trail_result = self.trail(
            acq_types=acq_types,
            n_trials=n_trials,
            max_iter=max_iter,
            n_initial=n_initial,
            target_idx=target_idx,
            verbose=True,
        )
        count_result = self.count(
            acq_types=acq_types,
            n_trials=n_trials,
            max_iter=max_iter,
            n_initial=n_initial,
            target_idx=target_idx,
            verbose=False,  # trail 已打印进度
        )
        return {"trail": trail_result, "count": count_result}

    # ── 摘要 ──────────────────────────────────────────────────────

    @staticmethod
    def summary(trail_result: pd.DataFrame) -> str:
        """生成 trail 结果的人类可读摘要。

        Args:
            trail_result: trail() 返回的 DataFrame

        Returns:
            多行文本摘要
        """
        lines = ["=== UHPCBenchmark Trail Summary ===", ""]
        for acq_type in sorted(trail_result["acq_type"].unique()):
            subset = trail_result[trail_result["acq_type"] == acq_type]
            n_trials = subset["trial"].nunique()
            final_rows = subset.loc[
                subset.groupby("trial")["iteration"].idxmax()
            ]
            final_vals = final_rows["best_value"].dropna()
            if len(final_vals) == 0:
                lines.append(f"  {acq_type}: 无有效数据")
                continue
            mean_final = final_vals.mean()
            std_final = final_vals.std()
            n_converged = int(
                (final_rows["converged_at"] > 0).sum()
            )
            lines.append(
                f"  {acq_type}: "
                f"best={mean_final:.3f}±{std_final:.3f}, "
                f"converged={n_converged}/{n_trials}"
            )
        return "\n".join(lines)

