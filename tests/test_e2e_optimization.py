# -*- coding: utf-8 -*-
"""端到端集成测试：数据加载 → 模型训练 → BO 优化 → 虚拟筛选 → Benchmark"""

import numpy as np
import pandas as pd
import pytest
from core.config import get_all_feature_names, get_all_target_names


class TestE2EOptimization:
    """Day 9：全流程串联验证"""

    @pytest.fixture
    def e2e_data(self):
        """生成合成数据集"""
        rng = np.random.RandomState(42)
        n = 80
        feats = get_all_feature_names()
        X = pd.DataFrame(
            rng.uniform(0.1, 1.0, size=(n, len(feats))),
            columns=feats,
        )
        # 线性 + 噪声目标
        Y = pd.DataFrame({
            'compressive_strength_28d': (
                50 + 15 * X[feats[0]] - 8 * X[feats[1]] + rng.randn(n) * 3
            ),
        })
        return X, Y, feats

    def test_data_to_benchmark_full_pipeline(self, e2e_data):
        """数据加载 → RF 训练 → EI/UCB 优化 → VirtualOptimizer → Benchmark"""
        X, Y, feats = e2e_data

        # 1. 训练 RF 代理模型
        from core.model_trainer import MultiTargetTrainer
        trainer = MultiTargetTrainer('random_forest')
        trainer.train_all(X, Y)

        # 2. 生成虚拟候选池
        rng = np.random.RandomState(99)
        n_candidates = 100
        candidates = pd.DataFrame(
            rng.uniform(0.1, 1.0, size=(n_candidates, len(feats))),
            columns=feats,
        )

        # 3. VirtualSpaceOptimizer 虚拟筛选
        from core.virtual_optimizer import VirtualSpaceOptimizer
        vo = VirtualSpaceOptimizer(trainer, feats, target_idx=0, seed=42)
        recs_ei = vo.recommend(
            candidates, acq_type='ei', top_n=10, apply_constraints=False
        )
        assert len(recs_ei) == 10
        assert 'predicted_mean' in recs_ei.columns
        assert 'predicted_std' in recs_ei.columns
        assert 'acq_score' in recs_ei.columns

        # 4. Benchmark 评估
        from core.benchmark import UHPCBenchmark
        bench = UHPCBenchmark(
            X.values, Y.values,
            feature_names=feats,
            target_names=list(Y.columns),
            directions=['minimize'],
            surrogate_type='rf',
        )
        trail_df = bench.trail(
            acq_types=['ei', 'ucb'],
            n_trials=2,
            max_iter=3,
        )
        assert len(trail_df) > 0
        assert 'acq_type' in trail_df.columns
        assert 'trial' in trail_df.columns
        assert 'best_value' in trail_df.columns

        # 5. Benchmark count
        count_df = bench.count(acq_types=['ei', 'ucb'], n_trials=2, max_iter=3)
        assert len(count_df) > 0
        assert 'success_rate' in count_df.columns

    def test_acq_switch_ei_to_ucb(self, e2e_data):
        """EI 切换到 UCB 不报错"""
        X, Y, feats = e2e_data

        from core.model_trainer import MultiTargetTrainer
        trainer = MultiTargetTrainer('random_forest')
        trainer.train_all(X, Y)

        rng = np.random.RandomState(99)
        candidates = pd.DataFrame(
            rng.uniform(0.1, 1.0, size=(50, len(feats))),
            columns=feats,
        )

        from core.virtual_optimizer import VirtualSpaceOptimizer
        vo = VirtualSpaceOptimizer(trainer, feats, target_idx=0, seed=42)

        # EI
        r1 = vo.recommend(candidates, acq_type='ei', top_n=5, apply_constraints=False)
        # UCB
        r2 = vo.recommend(candidates, acq_type='ucb', top_n=5, apply_constraints=False)

        assert len(r1) == 5
        assert len(r2) == 5
        # 两种采集函数的推荐应有所不同（至少 score 排序不同）
        assert not r1['acq_score'].equals(r2['acq_score'])

    def test_virtual_optimizer_with_benchmark_data(self, e2e_data):
        """Benchmark 产出的最优值可用于 VirtualSpaceOptimizer 的 best_f"""
        X, Y, feats = e2e_data

        from core.model_trainer import MultiTargetTrainer
        trainer = MultiTargetTrainer('random_forest')
        trainer.train_all(X, Y)

        # 得到真实最优值
        best_f = float(Y.iloc[:, 0].min())

        rng = np.random.RandomState(99)
        candidates = pd.DataFrame(
            rng.uniform(0.1, 1.0, size=(50, len(feats))),
            columns=feats,
        )

        from core.virtual_optimizer import VirtualSpaceOptimizer
        vo = VirtualSpaceOptimizer(trainer, feats, target_idx=0, seed=42)

        # 使用真实 best_f
        recs = vo.score_and_rank(candidates, acq_type='ei', best_f=best_f)
        assert len(recs) == len(candidates)
        assert np.all(np.isfinite(recs['acq_score'].values))

    def test_e2e_with_four_acq_types(self, e2e_data):
        """4 种采集策略端到端正常运行"""
        X, Y, feats = e2e_data

        from core.model_trainer import MultiTargetTrainer
        trainer = MultiTargetTrainer('random_forest')
        trainer.train_all(X, Y)

        rng = np.random.RandomState(99)
        candidates = pd.DataFrame(
            rng.uniform(0.1, 1.0, size=(50, len(feats))),
            columns=feats,
        )

        from core.virtual_optimizer import VirtualSpaceOptimizer
        vo = VirtualSpaceOptimizer(trainer, feats, target_idx=0, seed=42)

        for acq in ['ei', 'ucb', 'poi', 'thompson']:
            recs = vo.recommend(
                candidates, acq_type=acq, top_n=8, apply_constraints=False
            )
            assert len(recs) == 8, f"{acq} 推荐数量错误"
            assert np.all(np.isfinite(recs['acq_score'].values)), \
                f"{acq} 评分含 NaN/inf"
