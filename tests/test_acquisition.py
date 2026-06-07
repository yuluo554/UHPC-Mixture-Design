# -*- coding: utf-8 -*-
"""采集函数单元测试（EI/UCB/PoI/AugmentedEI/Thompson）"""

import numpy as np
import pytest
from core.acquisition_functions import SingleObjAcquisition as SA


class TestAcquisitionFunctions:
    """5 种采集函数的输入输出形状 + 边界条件"""

    @pytest.fixture
    def sample_data(self):
        """生成模拟预测数据"""
        rng = np.random.RandomState(42)
        mean = rng.uniform(30, 80, size=100)
        std = rng.uniform(1, 10, size=100)
        return mean, std

    # ── EI 测试 ──────────────────────────────────────────────────

    def test_ei_shape(self, sample_data):
        """EI 输出形状与输入一致"""
        mean, std = sample_data
        scores = SA.EI(mean, std)
        assert scores.shape == mean.shape
        assert scores.shape == (100,)

    def test_ei_best_f_default(self, sample_data):
        """best_f=None 时自动取 mean 最小值"""
        mean, std = sample_data
        scores_default = SA.EI(mean, std)
        scores_explicit = SA.EI(mean, std, best_f=float(np.min(mean)))
        np.testing.assert_array_almost_equal(scores_default, scores_explicit)

    def test_ei_non_negative(self, sample_data):
        """EI 值 >= 0"""
        mean, std = sample_data
        scores = SA.EI(mean, std)
        assert np.all(scores >= 0)

    def test_ei_improvement_signal(self):
        """更低的 mean 值应产生更高的 EI"""
        std = np.array([2.0, 2.0])
        mean_better = np.array([25.0, 45.0])   # 第一个更好
        mean_worse = np.array([45.0, 25.0])    # 第二个更好
        best_f = 30.0
        ei_better = SA.EI(mean_better, std, best_f=best_f)
        ei_worse = SA.EI(mean_worse, std, best_f=best_f)
        assert ei_better[0] > ei_better[1]   # mean=25 比 45 的 EI 更高
        assert ei_worse[1] > ei_worse[0]

    # ── UCB 测试 ─────────────────────────────────────────────────

    def test_ucb_alpha_effect(self):
        """alpha 越大，UCB 值越小（探索越激进）"""
        mean = np.array([50.0, 50.0])
        std = np.array([5.0, 5.0])
        ucb_small_alpha = SA.UCB(mean, std, alpha=0.5)
        ucb_large_alpha = SA.UCB(mean, std, alpha=2.0)
        assert np.all(ucb_large_alpha < ucb_small_alpha)

    def test_ucb_shape(self, sample_data):
        """UCB 输出形状与输入一致"""
        mean, std = sample_data
        scores = SA.UCB(mean, std)
        assert scores.shape == mean.shape

    # ── PoI 测试 ─────────────────────────────────────────────────

    def test_poi_bounds(self, sample_data):
        """PoI 值在 [0, 1] 之间"""
        mean, std = sample_data
        scores = SA.PoI(mean, std)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_poi_tau_effect(self):
        """tau 越大，PoI 值越小（更严格的改进要求）"""
        mean = np.array([35.0])
        std = np.array([5.0])
        best_f = 40.0
        poi_no_tau = SA.PoI(mean, std, best_f=best_f, tau=0.0)
        poi_with_tau = SA.PoI(mean, std, best_f=best_f, tau=0.1)
        assert poi_with_tau[0] <= poi_no_tau[0]

    # ── Thompson 测试 ─────────────────────────────────────────────

    def test_thompson_reproducibility(self):
        """相同 seed 产生相同抽样"""
        mean = np.array([50.0, 60.0, 70.0])
        std = np.array([5.0, 5.0, 5.0])
        t1 = SA.Thompson(mean, std, seed=42)
        t2 = SA.Thompson(mean, std, seed=42)
        np.testing.assert_array_equal(t1, t2)

    def test_thompson_shape(self, sample_data):
        """Thompson 输出形状与输入一致"""
        mean, std = sample_data
        samples = SA.Thompson(mean, std, seed=42)
        assert samples.shape == mean.shape

    # ── AugmentedEI 测试 ─────────────────────────────────────────

    def test_augmented_ei_decay(self):
        """高方差区域的 AEI < 标准 EI（惩罚效应）"""
        mean = np.array([40.0, 40.0])
        std = np.array([1.0, 20.0])
        best_f = 45.0
        ei = SA.EI(mean, std, best_f=best_f)
        aei = SA.Augmented_EI(mean, std, best_f=best_f, tau=0.5)
        # 高方差 (idx=1) 的 AEI < EI
        assert aei[1] < ei[1]

    # ── EI 分解测试 ──────────────────────────────────────────────

    def test_ei_decompose_sum(self, sample_data):
        """EI 分解后探索+利用 = 总 EI"""
        mean, std = sample_data
        result = SA.EI_decompose(mean, std)
        np.testing.assert_array_almost_equal(
            result['exploration'] + result['exploitation'],
            result['ei_total']
        )

    def test_ei_decompose_keys(self, sample_data):
        """返回 dict 含所有必需 key"""
        mean, std = sample_data
        result = SA.EI_decompose(mean, std)
        assert 'exploration' in result
        assert 'exploitation' in result
        assert 'ei_total' in result
        assert 'Z' in result

    # ── 边界条件测试 ─────────────────────────────────────────────

    def test_zero_std(self):
        """std=0 时不抛出除零异常"""
        mean = np.array([50.0, 60.0])
        std = np.array([0.0, 0.0])
        # 所有函数不应抛出异常
        SA.EI(mean, std)
        SA.UCB(mean, std)
        SA.PoI(mean, std)
        SA.Augmented_EI(mean, std)
        SA.Thompson(mean, std, seed=42)
        SA.EI_decompose(mean, std)

    def test_all_methods_return_finite(self, sample_data):
        """所有方法返回有限值（无 NaN/inf）"""
        mean, std = sample_data
        methods = {
            'EI': SA.EI(mean, std),
            'UCB': SA.UCB(mean, std),
            'PoI': SA.PoI(mean, std),
            'AugmentedEI': SA.Augmented_EI(mean, std),
            'Thompson': SA.Thompson(mean, std, seed=42),
        }
        for name, scores in methods.items():
            assert np.all(np.isfinite(scores)), f"{name} 含 NaN 或 inf"

    def test_single_input(self):
        """单样本输入正常工作"""
        mean = np.array([50.0])
        std = np.array([5.0])
        assert SA.EI(mean, std).shape == (1,)
        assert SA.UCB(mean, std).shape == (1,)
        assert SA.PoI(mean, std).shape == (1,)
        assert SA.Augmented_EI(mean, std).shape == (1,)
        assert SA.Thompson(mean, std, seed=1).shape == (1,)
