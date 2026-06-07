# -*- coding: utf-8 -*-
"""
虚拟空间优化器 — CTGAN + 贝叶斯采集函数融合。

VirtualSpaceOptimizer 接收 CTGAN 生成的虚拟候选池，使用代理模型 +
采集函数进行智能评分和 Top-N 推荐，替代人工肉眼挑选。

工作流:
  1. 接收 CTGAN 生成的虚拟候选池（DataFrame）
  2. 物理约束过滤（可选，4 条 UHPC 约束）
  3. 代理模型预测（mean + std）
  4. 采集函数评分（EI / UCB / PoI / Thompson）
  5. Top-N 推荐（采集函数最优的前 N 个配比）

参考 Bgolearn BGO_Efficient 设计模式（BGO_eval.py:63-186，MIT 许可证），
但本模块不内置 Kriging 模型——通过 MultiTargetTrainer 接口解耦代理模型。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.model_trainer import MultiTargetTrainer


class VirtualSpaceOptimizer:
    """虚拟空间优化器 — CTGAN + 贝叶斯采集函数融合。

    使用示例::

        from core.virtual_optimizer import VirtualSpaceOptimizer
        from core.model_trainer import MultiTargetTrainer

        trainer = MultiTargetTrainer('random_forest')
        trainer.train_all(X_real, Y_real)

        vo = VirtualSpaceOptimizer(trainer, feature_names, target_idx=0)
        recs = vo.recommend(synthetic_df, acq_type='ei', top_n=10)
        print(recs[['acq_score', 'predicted_mean', 'predicted_std']])
    """

    def __init__(
        self,
        trainer: "MultiTargetTrainer",
        feature_names: list[str],
        target_idx: int = 0,
        seed: int = 42,
    ) -> None:
        """初始化虚拟空间优化器。

        Args:
            trainer: 已训练的 MultiTargetTrainer（RF/GP/Ensemble 均可）
            feature_names: 特征名列表（必须与候选池列对齐）
            target_idx: 目标索引（默认 0 = 第一个目标，通常为 compressive_strength_28d）
            seed: 随机种子（Thompson 采样用）
        """
        self.trainer = trainer
        self.feature_names = list(feature_names)
        self.target_idx = target_idx
        self.seed = seed
        # BUG-20260607-026 修复：对 trainer.models 非空情况添加 target_idx 边界检查，
        # 直接 list(trainer.models.keys())[target_idx] 在
        # 键数 < target_idx 时会 IndexError。空 trainer 走 fallback 路径。
        if trainer.models:
            keys = list(trainer.models.keys())
            if target_idx >= len(keys):
                raise IndexError(
                    f"target_idx={target_idx} 超出 trainer.models 范围（{len(keys)} 个目标）"
                )
            self._target_name = keys[target_idx]
        else:
            self._target_name = f"target_{target_idx}"

    # ── 约束过滤 ──────────────────────────────────────────────────

    def filter_constraints(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """使用 4 条 UHPC 物理约束过滤候选池。

        约束（同 BayesianOptimizer._check_uhpc_constraints）：
          1. 水胶比 ∈ [0.14, 0.24]
          2. 砂胶比 ∈ [0.5, 1.5]
          3. 减水剂/胶凝比 ∈ [0.005, 0.05]
          4. 纤维体积率 ∈ [0.5, 3.0]

        缺失约束所需列时优雅跳过该条约束。

        Args:
            candidates: 候选配比 DataFrame

        Returns:
            通过所有有效约束的子集 DataFrame
        """
        from core.domain_features import check_uhpc_constraints_df

        mask = check_uhpc_constraints_df(candidates, self.feature_names)
        return candidates.loc[mask].copy()

    # ── 采集函数评分排序 ──────────────────────────────────────────

    def score_and_rank(
        self,
        candidates: pd.DataFrame,
        acq_type: str = "ei",
        best_f: Optional[float] = None,
        **acq_params,
    ) -> pd.DataFrame:
        """对候选池进行采集函数评分并排序。

        调用 trainer.predict_all_with_uncertainty() 获取后验分布，
        使用 SingleObjAcquisition 计算采集函数评分，按优劣排列。

        Args:
            candidates: 候选配比 DataFrame
            acq_type: 采集函数类型（'ei' / 'ucb' / 'poi' / 'thompson'）
            best_f: 当前最优目标值。若为 None，取代理模型预测均值的最小值。
            **acq_params: 采集函数参数（如 alpha / tau）

        Returns:
            原 DataFrame + 新增列 ['predicted_mean', 'predicted_std', 'acq_score']，
            按采集函数优劣排序：
              - EI / PoI → 降序（acq_score 越大越好）
              - UCB / Thompson → 升序（acq_score 越小越好）
        """
        from core.acquisition_functions import SingleObjAcquisition as SA

        # 1. 对齐特征
        avail_X = [c for c in self.feature_names if c in candidates.columns]
        X_sub = candidates[avail_X]

        # 2. 代理模型预测
        mean_df, std_df = self.trainer.predict_all_with_uncertainty(X_sub)
        target_name = self._target_name
        if target_name not in mean_df.columns:
            # fallback: 使用第一个目标列
            target_name = mean_df.columns[0]
        m = mean_df[target_name].values
        s = std_df[target_name].values

        # 3. best_f 默认值
        if best_f is None:
            best_f = float(np.min(m))

        # 4. 计算采集函数
        if acq_type == "ei":
            scores = SA.EI(m, s, best_f=best_f)
            ascending = False  # EI 越大越好
        elif acq_type == "ucb":
            alpha = acq_params.get("alpha", 1.0)
            scores = SA.UCB(m, s, alpha=alpha)
            ascending = True  # UCB 越小越好
        elif acq_type == "poi":
            tau = acq_params.get("tau", 0.0)
            scores = SA.PoI(m, s, best_f=best_f, tau=tau)
            ascending = False  # PoI 越大越好
        elif acq_type == "thompson":
            scores = SA.Thompson(m, s, seed=self.seed)
            ascending = True  # Thompson 抽样值越小越好
        else:
            valid = ["ei", "ucb", "poi", "thompson"]
            raise ValueError(
                f"不支持的采集函数类型: {acq_type}，支持: {valid}"
            )

        # 5. 构建结果 DataFrame
        result = candidates.copy()
        result["predicted_mean"] = m
        result["predicted_std"] = s
        result["acq_score"] = scores

        # 6. 排序
        result = result.sort_values("acq_score", ascending=ascending).reset_index(
            drop=True
        )
        return result

    # ── 一键推荐 ──────────────────────────────────────────────────

    def recommend(
        self,
        candidates: pd.DataFrame,
        acq_type: str = "ei",
        top_n: int = 10,
        apply_constraints: bool = True,
        **acq_params,
    ) -> pd.DataFrame:
        """一键推荐：约束过滤 → 评分 → Top-N。

        Args:
            candidates: CTGAN 虚拟候选池 DataFrame
            acq_type: 采集函数类型（'ei' / 'ucb' / 'poi' / 'thompson'）
            top_n: 返回前 N 个最佳配比
            apply_constraints: 是否先应用物理约束过滤
            **acq_params: 采集函数参数（如 alpha / tau）

        Returns:
            top_n 个最佳配比，含 predicted_mean / predicted_std / acq_score
        """
        # 1. 约束过滤
        if apply_constraints:
            pool = self.filter_constraints(candidates)
            if len(pool) == 0:
                raise ValueError(
                    "约束过滤后无可行候选。请放宽约束条件或关闭 apply_constraints。"
                )
        else:
            pool = candidates.copy()

        # 2. 评分排序
        scored = self.score_and_rank(
            pool, acq_type=acq_type, **acq_params
        )

        # 3. Top-N
        return scored.head(top_n).reset_index(drop=True)
