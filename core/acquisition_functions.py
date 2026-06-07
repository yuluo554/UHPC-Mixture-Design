# -*- coding: utf-8 -*-
"""单目标采集函数（纯 NumPy + scipy）。

核心公式源自 Bgolearn `BGOmin.py` / `BGOmax.py`，经 MIT 许可证授权使用。
Copyright (c) 2023 caobin — MIT License
Ref: MultiBgolearn (arXiv:2601.06820)

UHPC-AutoDesign 内部将所有目标统一为最小化空间
（_normalize_for_minimization），因此本模块默认 minimize=True。
所有方法接收 (mean, std) 数组，返回 acquisition 值数组。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


class SingleObjAcquisition:
    """单目标采集函数（纯 NumPy + scipy）。

    UHPC-AutoDesign 内部将所有目标统一为最小化空间
    （_normalize_for_minimization），因此本模块默认 minimize=True。
    所有方法接收 (mean, std) 数组，返回 acquisition 值数组。
    """

    # ── 核心采集函数 ──────────────────────────────────────────

    @staticmethod
    def EI(mean: np.ndarray, std: np.ndarray,
           best_f: float | None = None) -> np.ndarray:
        """期望改进（Expected Improvement）。

        EI = (f_min - μ) · Φ(Z) + σ · φ(Z)
        Z = (f_min - μ) / σ

        Args:
            mean: 预测均值 (n_candidates,) 或可广播
            std: 预测标准差 (n_candidates,) 或可广播
            best_f: 当前最优值（最小化空间）。若为 None，则取 mean 的最小值。

        Returns:
            EI 值数组 (n_candidates,)。值越大越值得采样。
        """
        mean = np.asarray(mean).ravel()
        # BUG-20260607-027 注释：当 std=0 时，EI/UCB/PoI 会退化为基于 mean 的简单函数，
        # 不会出现除零异常（1e-10 floor 保证N86 IEEE 下），
        # 但采集函数退化为纯利用，无探索能力——用户需保证 std 非 0。
        std = np.maximum(np.asarray(std).ravel(), 1e-10)
        if best_f is None:
            best_f = float(np.min(mean))
        improvement = best_f - mean
        Z = improvement / std
        return improvement * norm.cdf(Z) + std * norm.pdf(Z)

    @staticmethod
    def UCB(mean: np.ndarray, std: np.ndarray,
            alpha: float = 1.0) -> np.ndarray:
        """上置信界（Upper Confidence Bound —— 最小化方向）。

        UCB = μ - α · σ

        注意：这是 LCB（Lower Confidence Bound）在最小化问题中的等效形式。
        取 argmin 作为推荐采样点。

        Args:
            mean: 预测均值
            std: 预测标准差
            alpha: 探索-利用权衡系数。α > 1 倾向探索，α < 1 倾向利用。

        Returns:
            UCB 值数组（取 argmin 为推荐）。
        """
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        return mean - alpha * std

    @staticmethod
    def PoI(mean: np.ndarray, std: np.ndarray,
            best_f: float | None = None,
            tau: float = 0.0) -> np.ndarray:
        """改进概率（Probability of Improvement）。

        PoI = Φ((f_min · (1 - τ) - μ) / σ)

        Args:
            mean: 预测均值
            std: 预测标准差
            best_f: 当前最优值。若为 None，则取 mean 的最小值。
            tau: 改进阈值比率。τ = 0 表示任何改进都算；τ = 0.01 要求 1% 改进。

        Returns:
            PoI 值数组（0-1 之间）。值越大越值得采样。
        """
        mean = np.asarray(mean).ravel()
        # BUG-20260607-027 注释：当 std=0 时，EI/UCB/PoI 会退化为基于 mean 的简单函数，
        # 不会出现除零异常（1e-10 floor 保证N86 IEEE 下），
        # 但采集函数退化为纯利用，无探索能力——用户需保证 std 非 0。
        std = np.maximum(np.asarray(std).ravel(), 1e-10)
        if best_f is None:
            best_f = float(np.min(mean))
        threshold = (1.0 - tau) * best_f
        return norm.cdf((threshold - mean) / std)

    @staticmethod
    def Augmented_EI(mean: np.ndarray, std: np.ndarray,
                     best_f: float | None = None,
                     tau: float = 0.01) -> np.ndarray:
        """增强期望改进（Augmented Expected Improvement）。

        AEI = EI · (1 - τ / √(σ² + τ²))

        通过噪声惩罚项 τ / √(σ² + τ²) 降低高不确定性区域的权重，
        避免在已探索充分的区域过度采样。

        Args:
            mean: 预测均值
            std: 预测标准差
            best_f: 当前最优值。若为 None，则取 mean 的最小值。
            tau: 噪声标准差（控制惩罚强度）。τ = 0 时退化为标准 EI。

        Returns:
            AEI 值数组。值越大越值得采样。
        """
        mean = np.asarray(mean).ravel()
        # BUG-20260607-027 注释：当 std=0 时，EI/UCB/PoI 会退化为基于 mean 的简单函数，
        # 不会出现除零异常（1e-10 floor 保证N86 IEEE 下），
        # 但采集函数退化为纯利用，无探索能力——用户需保证 std 非 0。
        std = np.maximum(np.asarray(std).ravel(), 1e-10)
        if best_f is None:
            best_f = float(np.min(mean))
        improvement = best_f - mean
        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        decay = 1.0 - tau / np.sqrt(std ** 2 + tau ** 2)
        return ei * decay

    @staticmethod
    def Thompson(mean: np.ndarray, std: np.ndarray,
                 seed: int | None = None) -> np.ndarray:
        """汤普森采样（Thompson Sampling）。

        从后验分布 N(μ, σ²) 中抽样，返回抽样值。
        取 argmin 为推荐采样点（最小化方向）。

        Args:
            mean: 预测均值
            std: 预测标准差
            seed: 随机种子（可复现性）

        Returns:
            抽样值数组（取 argmin 为推荐）。
        """
        mean = np.asarray(mean).ravel()
        std = np.asarray(std).ravel()
        rng = np.random.RandomState(seed)
        return rng.normal(loc=mean, scale=std)

    # ── EI 分解（Day 5 可视化） ─────────────────────────────────

    @staticmethod
    def EI_decompose(mean: np.ndarray, std: np.ndarray,
                     best_f: float | None = None) -> dict:
        """将 EI 分解为探索项（σ · φ(Z)）和利用项（(f_min - μ) · Φ(Z)）。

        EI = (f_min - μ) · Φ(Z) + σ · φ(Z)
             └── 利用项 ──┘   └── 探索项 ──┘

        Args:
            mean: 预测均值
            std: 预测标准差
            best_f: 当前最优值。若为 None，则取 mean 的最小值。

        Returns:
            dict with keys:
                'exploration': (n_candidates,) 探索项数组
                'exploitation': (n_candidates,) 利用项数组
                'ei_total': (n_candidates,) 总 EI 数组
                'Z': (n_candidates,) Z 值数组
        """
        mean = np.asarray(mean).ravel()
        # BUG-20260607-027 注释：当 std=0 时，EI/UCB/PoI 会退化为基于 mean 的简单函数，
        # 不会出现除零异常（1e-10 floor 保证N86 IEEE 下），
        # 但采集函数退化为纯利用，无探索能力——用户需保证 std 非 0。
        std = np.maximum(np.asarray(std).ravel(), 1e-10)
        if best_f is None:
            best_f = float(np.min(mean))
        improvement = best_f - mean
        Z = improvement / std
        exploitation = improvement * norm.cdf(Z)
        exploration = std * norm.pdf(Z)
        return {
            'exploration': exploration,
            'exploitation': exploitation,
            'ei_total': exploitation + exploration,
            'Z': Z,
        }
