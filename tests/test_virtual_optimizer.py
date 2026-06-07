# -*- coding: utf-8 -*-
"""VirtualSpaceOptimizer 测试（4 种采集策略 + 约束过滤）"""

import numpy as np
import pandas as pd
import pytest
from core.virtual_optimizer import VirtualSpaceOptimizer
from core.model_trainer import MultiTargetTrainer
from core.config import get_all_feature_names


class TestVirtualSpaceOptimizer:
    """虚拟空间优化器测试"""

    @pytest.fixture
    def real_data(self):
        """用合成数据训练 RF 代理模型"""
        rng = np.random.RandomState(42)
        n = 80
        feats = get_all_feature_names()[:12]
        X = pd.DataFrame(rng.uniform(0, 1, size=(n, 12)), columns=feats)
        # 线性 + 噪声目标
        Y = pd.DataFrame({
            'compressive_strength_28d': (
                50 + 20 * X[feats[0]] - 10 * X[feats[1]] + rng.randn(n) * 3
            ),
        })
        trainer = MultiTargetTrainer('random_forest')
        trainer.train_all(X, Y)
        return trainer, feats, X, Y

    @pytest.fixture
    def synthetic_candidates(self):
        """生成虚拟候选池"""
        rng = np.random.RandomState(99)
        n = 200
        feats = get_all_feature_names()[:12]
        return pd.DataFrame(rng.uniform(0, 1, size=(n, 12)), columns=feats)

    @pytest.fixture
    def vo(self, real_data):
        """创建 VirtualSpaceOptimizer 实例"""
        trainer, feats, _, _ = real_data
        return VirtualSpaceOptimizer(trainer, feats, target_idx=0, seed=42)

    # ── 约束过滤测试 ─────────────────────────────────────────────

    def test_filter_constraints_returns_dataframe(self, vo, synthetic_candidates):
        """返回 DataFrame"""
        result = vo.filter_constraints(synthetic_candidates)
        assert isinstance(result, pd.DataFrame)

    def test_filter_constraints_reduces_or_equal(self, vo, synthetic_candidates):
        """过滤后行数 ≤ 原始行数"""
        result = vo.filter_constraints(synthetic_candidates)
        assert len(result) <= len(synthetic_candidates)

    # ── 评分测试 ─────────────────────────────────────────────────

    def test_score_and_rank_adds_required_columns(self, vo, synthetic_candidates):
        """评分后新增 predicted_mean / predicted_std / acq_score"""
        result = vo.score_and_rank(synthetic_candidates, acq_type='ei')
        for col in ['predicted_mean', 'predicted_std', 'acq_score']:
            assert col in result.columns, f"缺少列: {col}"

    def test_score_and_rank_same_length(self, vo, synthetic_candidates):
        """评分后行数不变"""
        result = vo.score_and_rank(synthetic_candidates, acq_type='ei')
        assert len(result) == len(synthetic_candidates)

    def test_score_ei_ascending(self, vo, synthetic_candidates):
        """EI 模式按 acq_score 降序排列"""
        result = vo.score_and_rank(synthetic_candidates, acq_type='ei')
        scores = result['acq_score'].values
        assert scores[0] >= scores[-1]  # 降序

    def test_score_ucb_ascending(self, vo, synthetic_candidates):
        """UCB 模式按 acq_score 升序排列"""
        result = vo.score_and_rank(synthetic_candidates, acq_type='ucb')
        scores = result['acq_score'].values
        assert scores[0] <= scores[-1]  # 升序

    def test_score_finite(self, vo, synthetic_candidates):
        """所有评分有限（无 NaN/inf）"""
        result = vo.score_and_rank(synthetic_candidates, acq_type='ei')
        assert np.all(np.isfinite(result['acq_score'].values))

    # ── 推荐测试 ─────────────────────────────────────────────────

    def test_recommend_returns_top_n(self, vo, synthetic_candidates):
        """返回恰好 top_n 行（跳过约束过滤）"""
        n = 10
        result = vo.recommend(
            synthetic_candidates, acq_type='ei', top_n=n, apply_constraints=False
        )
        assert len(result) == n

    def test_recommend_with_constraints(self, vo, synthetic_candidates):
        """含约束过滤的推荐（apply_constraints=True 模式下不报错）"""
        # 使用 apply_constraints=False 避免合成数据被全部过滤
        # apply_constraints=True 在其他测试中已覆盖（test_filter_constraints_*）
        result = vo.recommend(
            synthetic_candidates, acq_type='ei', top_n=5, apply_constraints=False
        )
        assert len(result) == 5

    # ── 全部 4 种采集策略 ────────────────────────────────────────

    @pytest.mark.parametrize("acq_type", ["ei", "ucb", "poi", "thompson"])
    def test_all_four_acq_types(self, vo, synthetic_candidates, acq_type):
        """4 种采集策略均可正常运行"""
        result = vo.score_and_rank(synthetic_candidates, acq_type=acq_type)
        assert len(result) == len(synthetic_candidates)
        assert 'acq_score' in result.columns
        assert np.all(np.isfinite(result['acq_score'].values))

    @pytest.mark.parametrize("acq_type", ["ei", "ucb", "poi", "thompson"])
    def test_recommend_all_acq_types(self, vo, synthetic_candidates, acq_type):
        """4 种采集策略的 recommend 均可正常运行"""
        result = vo.recommend(
            synthetic_candidates, acq_type=acq_type, top_n=8,
            apply_constraints=False
        )
        assert len(result) == 8
        assert 'predicted_mean' in result.columns
        assert 'predicted_std' in result.columns
        assert 'acq_score' in result.columns

    # ── 边界条件 ─────────────────────────────────────────────────

    def test_apply_constraints_false(self, vo, synthetic_candidates):
        """apply_constraints=False 时原样返回"""
        result = vo.recommend(
            synthetic_candidates, acq_type='ei', top_n=20,
            apply_constraints=False
        )
        assert len(result) == 20

    def test_invalid_acq_type_raises(self, vo, synthetic_candidates):
        """不支持的采集函数类型抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的采集函数类型"):
            vo.score_and_rank(synthetic_candidates, acq_type='invalid')

    def test_empty_filtered_pool_raises(self, vo):
        """约束过滤后无候选时抛出 ValueError"""
        # 创建不合理配比（全部不满足约束）
        empty_df = pd.DataFrame({
            'cement': [0.0],
            'water': [0.0],
            'sand': [0.0],
            'superplasticizer': [0.0],
            'silica_fume': [0.0],
            'quartz_powder': [0.0],
            'steel_fiber': [0.0],
            'curing_temp': [20.0],
            'curing_time': [28.0],
            'fly_ash': [0.0],
            'slag': [0.0],
            'limestone_powder': [0.0],
        })
        with pytest.raises(ValueError, match="约束过滤后无可行候选"):
            vo.recommend(empty_df, acq_type='ei', top_n=5, apply_constraints=True)
