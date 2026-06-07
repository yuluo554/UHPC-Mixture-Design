# -*- coding: utf-8 -*-
"""BayesianOptimizer + ParetoCollector 测试"""

import os
import pytest
import numpy as np
from core.config import get_all_feature_names, get_all_target_names
from core.bayesian_optimizer import (
    BayesianOptimizer,
    ParetoCollector,
    _normalize_for_minimization,
    compute_reference_point,
    create_bayesian_optimizer,
)


class TestBayesianOptimizer:
    """贝叶斯优化器测试"""

    @pytest.fixture
    def bob_data(self, dm):
        """准备贝叶斯优化数据"""
        features = [f for f in get_all_feature_names() if f in dm.cleaned_data.columns]
        targets = [t for t in get_all_target_names() if t in dm.cleaned_data.columns]
        X = dm.cleaned_data[features].values.astype(np.float64)
        Y = dm.cleaned_data[targets].values.astype(np.float64)
        # 移除含 NaN 的行
        valid = np.isfinite(Y).all(axis=1)
        return features, targets, X[valid], Y[valid]

    # ── ParetoCollector ──────────────────────────────────────

    def test_pareto_collector_basic(self):
        """测试 ParetoCollector 基本功能"""
        pc = ParetoCollector(directions=["minimize", "minimize"])
        Y = np.array([
            [0.5, 0.5],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.4, 0.4],
            [0.6, 0.6],
        ])
        X = np.random.randn(5, 3)
        pc.update(X, Y)
        pX, pY = pc.get_frontier()
        # 应该有 3 个非支配点: [0.3,0.7], [0.4,0.4], [0.7,0.3]
        assert len(pX) == 3

    def test_pareto_collector_maximize(self):
        """测试最大化目标的 ParetoCollector"""
        pc = ParetoCollector(directions=["maximize", "maximize"])
        Y = np.array([
            [10, 10],
            [12, 8],
            [8, 12],
            [11, 11],
        ])
        X = np.random.randn(4, 2)
        pc.update(X, Y)
        _, pY = pc.get_frontier()
        assert len(pY) > 0
        hv = pc.get_hypervolume()
        assert hv >= 0

    def test_pareto_collector_hypervolume_increasing(self):
        """测试超体积随迭代单调不减（固定参考点）"""
        # 使用远大于数据范围的固定参考点，确保 HV 单调性
        ref_point = np.array([10.0, 10.0])
        pc = ParetoCollector(
            directions=["minimize", "minimize"],
            ref_point=ref_point,
        )
        hv_values = []
        for i in range(5):
            Y_new = np.random.rand(3, 2)  # 值在 [0,1] 远小于 ref_point
            X_new = np.random.randn(3, 2)
            record = pc.update(X_new, Y_new, iteration=i)
            hv_values.append(record["hypervolume"])
        # 固定参考点下，添加新点不应减少超体积
        for j in range(1, len(hv_values)):
            assert hv_values[j] >= hv_values[j - 1] - 1e-9, \
                f"HV decreased from {hv_values[j-1]} to {hv_values[j]}"

    def test_pareto_collector_history(self):
        """测试历史记录"""
        pc = ParetoCollector(directions=["minimize", "minimize", "minimize"])
        for i in range(3):
            Y_new = np.random.rand(2, 3)
            X_new = np.random.randn(2, 3)
            pc.update(X_new, Y_new, iteration=i)
        df = pc.get_history_df()
        assert len(df) == 3
        assert "hypervolume" in df.columns
        assert "n_frontier" in df.columns

    # ── BayesianOptimizer — GP ───────────────────────────────

    def test_gp_optimizer_creation(self, bob_data):
        """测试 GP 优化器创建"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            directions=["maximize", "maximize", "minimize", "minimize", "minimize"],
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        assert opt.surrogate_type == "gp"
        assert opt.acq_type == "qnehvi"
        assert len(opt.feature_names) == 12
        assert len(opt.target_names) == 5

    def test_gp_initialize_and_fit(self, bob_data):
        """测试 GP 代理模型初始化和训练"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        opt._initialize_data(X, Y)
        opt._fit_surrogates()
        assert len(opt._models) == len(targets)
        assert opt._trained is True

    def test_gp_single_step(self, bob_data):
        """测试 GP 单步优化"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        opt._initialize_data(X, Y)
        opt._fit_surrogates()
        result = opt.step(iteration=1)
        assert "candidate" in result
        assert "predicted_y" in result
        assert "hypervolume" in result
        assert result["hypervolume"] >= 0

    def test_gp_optimize_loop(self, bob_data):
        """测试 GP 完整优化循环"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        results = opt.optimize(X, Y, n_iter=5, verbose=False)
        assert results["n_iter_completed"] == 5
        assert len(results["pareto_X"]) > 0
        assert len(results["pareto_Y"]) > 0
        assert len(results["hypervolume_history"]) == 6  # 初始 + 5 步
        assert len(results["optimization_log"]) == 5

    # ── BayesianOptimizer — RF ───────────────────────────────

    def test_rf_optimizer_creation(self, bob_data):
        """测试 RF 优化器创建"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="rf",
            acq_type="qnehvi",
        )
        assert opt.surrogate_type == "rf"

    def test_rf_initialize_and_fit(self, bob_data):
        """测试 RF 代理模型初始化和训练"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="rf",
            acq_type="qnehvi",
        )
        opt._initialize_data(X, Y)
        opt._fit_surrogates()
        assert opt._rf_model is not None
        assert len(opt._rf_model.models) == len(targets)

    def test_rf_single_step(self, bob_data):
        """测试 RF 单步优化"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="rf",
            acq_type="qnehvi",
        )
        opt._initialize_data(X, Y)
        opt._fit_surrogates()
        result = opt.step(iteration=1)
        assert result["hypervolume"] >= 0
        assert len(result["predicted_y"]) == len(targets)

    def test_rf_optimize_loop(self, bob_data):
        """测试 RF 完整优化循环"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="rf",
            acq_type="qnehvi",
        )
        results = opt.optimize(X, Y, n_iter=8, verbose=False)
        assert results["n_iter_completed"] == 8
        assert len(results["pareto_X"]) > 0

    # ── BayesianOptimizer — Ensemble ─────────────────────────

    def test_ensemble_optimize(self, bob_data):
        """测试 Ensemble 模式优化"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="ensemble",
            acq_type="qnehvi",
        )
        results = opt.optimize(X, Y, n_iter=5, verbose=False)
        assert results["n_iter_completed"] == 5
        assert len(results["pareto_X"]) > 0

    # ── 结果方法 ────────────────────────────────────────────

    def test_get_pareto_df(self, bob_data):
        """测试 get_pareto_df 结果格式化"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        opt.optimize(X, Y, n_iter=5, verbose=False)
        df = opt.get_pareto_df()
        assert len(df) > 0
        # 应包含 rank 列
        assert "rank" in df.columns

    def test_get_best_point(self, bob_data):
        """测试获取最佳点"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        opt.optimize(X, Y, n_iter=5, verbose=False)
        best = opt.get_best_point()
        assert "X" in best
        assert "Y" in best
        assert best["distance"] >= 0

    def test_get_convergence_df(self, bob_data):
        """测试收敛数据"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        opt.optimize(X, Y, n_iter=3, verbose=False)
        conv = opt.get_convergence_df()
        assert len(conv) > 0
        assert "hypervolume" in conv.columns

    # ── 持久化 ──────────────────────────────────────────────

    def test_save_and_load(self, bob_data, tmp_path):
        """测试优化器保存和加载"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        opt.optimize(X, Y, n_iter=5, verbose=False)

        save_path = os.path.join(tmp_path, "test_bayes.pkl")
        opt.save(save_path)
        assert os.path.exists(save_path)

        opt2 = BayesianOptimizer().load(save_path)
        assert opt2.feature_names == features
        assert opt2.target_names == targets
        assert len(opt2.optimization_log) == 5

        # 加载后仍可查询结果
        res = opt2.get_results()
        assert res["n_frontier"] > 0

    # ── 工具函数 ────────────────────────────────────────────

    def test_normalize_minimization(self):
        """测试目标归一化"""
        y = np.array([
            [200, 30, 220, 3000, 800],
            [150, 25, 200, 2500, 700],
        ])
        dirs = ["maximize", "maximize", "target", "minimize", "minimize"]
        y_norm = _normalize_for_minimization(y, dirs)
        assert y_norm.shape == (2, 5)
        # maximize 被取负
        assert y_norm[0, 0] < y_norm[1, 0]  # 200→-200 < 150→-150
        # minimize 保持
        assert y_norm[0, 3] == 3000  # 未变
        assert y_norm[1, 4] == 700

    def test_reference_point_computation(self):
        """测试参考点计算"""
        y = np.array([
            [10, 20],
            [15, 15],
            [20, 10],
        ])
        dirs = ["minimize", "minimize"]
        ref = compute_reference_point(y, dirs)
        assert ref.shape == (2,)
        # 参考点应比所有点差
        assert ref[0] >= y[:, 0].max()
        assert ref[1] >= y[:, 1].max()

    def test_create_bayesian_optimizer(self):
        """测试便捷创建函数"""
        opt = create_bayesian_optimizer()
        assert len(opt.feature_names) == 12
        assert len(opt.target_names) == 5
        assert opt.surrogate_type == "gp"
        assert opt.acq_type == "qnehvi"

    def test_constraint_checking(self, bob_data):
        """测试 UHPC 约束检查"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        # 所有文献数据应该都通过水胶比约束
        mask = opt._check_uhpc_constraints(X)
        assert mask.sum() >= len(X) * 0.8  # 至少 80% 通过

    def test_surrogate_predict_shapes(self, bob_data):
        """测试代理模型预测输出形状"""
        features, targets, X, Y = bob_data
        opt = BayesianOptimizer(
            feature_names=features,
            target_names=targets,
            surrogate_type="gp",
            acq_type="qnehvi",
        )
        opt._initialize_data(X, Y)
        opt._fit_surrogates()

        # 用连续特征预测
        X_cont = opt._to_cont(X[:5])
        mean, std = opt._surrogate_predict(X_cont)
        assert mean.shape == (5, len(targets))
        assert std.shape == (5, len(targets))
        # 标准差应为正
        assert (std > 0).all()

