# -*- coding: utf-8 -*-
"""
端到端集成测试 — 完整 UHPC-AutoDesign 管线
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.config import get_all_feature_names, get_all_target_names, get_categorical_features
from core.data_manager import DataManager
from core.model_trainer import ModelTrainer, MultiTargetTrainer
from core.shap_analyzer import SHAPAnalyzer
from core.bayesian_optimizer import (
    BayesianOptimizer,
    ParetoCollector,
    _normalize_for_minimization,
    compute_reference_point,
    create_bayesian_optimizer,
)
from core.multi_objective import (
    MultiObjectiveOptimizer,
    UHPCProblem,
    create_multi_objective_optimizer,
    HAS_PYMOO,
)
from core.topsis_selector import (
    TOPSISSelector,
    create_topsis_selector,
    HAS_PLOTLY as HAS_PLOTLY_TOPSIS,
)
from core.feedback_manager import FeedbackManager, TARGET_LABELS


@pytest.fixture
def dm():
    """加载示例数据并清洗"""
    mgr = DataManager()
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "literature.csv"
    )
    mgr.load_csv(sample_path)
    mgr.clean(remove_outliers=True, impute=True, contamination=0.05)
    return mgr


@pytest.fixture
def features_targets(dm):
    """获取特征和目标"""
    features = [f for f in get_all_feature_names() if f in dm.cleaned_data.columns]
    targets = [t for t in get_all_target_names() if t in dm.cleaned_data.columns]
    X = dm.cleaned_data[features].select_dtypes(include=[np.number])
    Y = dm.cleaned_data[targets]
    return features, targets, X, Y


class TestDataManager:
    """数据管理器测试"""

    def test_load_and_clean(self, dm):
        assert dm.raw_data is not None
        assert dm.cleaned_data is not None
        assert len(dm.cleaned_data) <= len(dm.raw_data)
        assert dm.cleaned_data.isnull().sum().sum() == 0

    def test_statistics(self, dm):
        stats = dm.get_statistics()
        assert stats is not None
        assert len(stats) > 0

    def test_correlation(self, dm):
        corr = dm.get_correlation_matrix()
        assert corr is not None
        assert corr.shape[0] > 0

    def test_split(self, dm):
        X_train, X_test, y_train, y_test = dm.split(test_size=0.2)
        assert len(X_train) >= len(X_test)
        assert len(y_train) >= len(y_test)


class TestModelTrainer:
    """模型训练器测试"""

    def test_rf_train(self, features_targets):
        _, _, X, Y = features_targets
        target = Y.columns[0]
        trainer = ModelTrainer("random_forest")
        trainer.train(X, Y[target])
        preds = trainer.predict(X.head(3))
        assert len(preds) == 3

    def test_cv(self, features_targets):
        _, _, X, Y = features_targets
        target = Y.columns[0]
        trainer = ModelTrainer("random_forest")
        cv = trainer.cross_validate(X, Y[target], cv=5)
        assert "r2_mean" in cv
        assert cv["r2_mean"] > 0  # 应该有正向预测能力

    def test_multi_target(self, features_targets):
        _, _, X, Y = features_targets
        mt = MultiTargetTrainer("random_forest")
        mt.train_all(X, Y)
        preds = mt.predict_all(X.head(3))
        assert preds.shape == (3, len(Y.columns))
        assert len(mt.models) == len(Y.columns)


class TestSHAP:
    """SHAP 分析器测试"""

    def test_compute(self, features_targets):
        _, _, X, Y = features_targets
        target = Y.columns[0]
        trainer = ModelTrainer("random_forest")
        trainer.train(X, Y[target])

        analyzer = SHAPAnalyzer(trainer.model, X)
        analyzer.compute(max_samples=30)
        assert analyzer.shap_values is not None

    def test_importance_df(self, features_targets):
        _, _, X, Y = features_targets
        target = Y.columns[0]
        trainer = ModelTrainer("random_forest")
        trainer.train(X, Y[target])

        analyzer = SHAPAnalyzer(trainer.model, X)
        analyzer.compute(max_samples=30)
        imp = analyzer.get_importance_df()
        assert len(imp) == X.shape[1]
        assert imp["SHAP重要性"].iloc[0] >= imp["SHAP重要性"].iloc[-1]  # 降序


class TestGANGenerator:
    """CTGAN 虚拟生成器测试"""

    @pytest.fixture
    def gan_data(self, dm):
        """准备 GAN 训练数据"""
        features = [f for f in get_all_feature_names() if f in dm.cleaned_data.columns]
        targets = [t for t in get_all_target_names() if t in dm.cleaned_data.columns]
        all_cols = features + targets
        cat_cols = [c for c in get_categorical_features() if c in dm.cleaned_data.columns]
        train_df = dm.cleaned_data[all_cols].copy()
        return train_df, features, targets, cat_cols

    def test_ctgan_fit_and_generate(self, gan_data):
        """测试 CTGAN 训练 + 生成"""
        from core.gan_generator import CTGANGenerator
        train_df, features, targets, cat_cols = gan_data

        gen = CTGANGenerator(epochs=20, batch_size=30, verbose=False)
        gen.fit(train_df, categorical_columns=cat_cols)

        synth = gen.generate(50)
        assert synth is not None
        assert len(synth) == 50
        # 应包含所有列
        for col in train_df.columns:
            assert col in synth.columns

        # 生成的数据不应有 NaN
        assert synth.isnull().sum().sum() == 0

    def test_ctgan_conditional_generate(self, gan_data):
        """测试条件定向生成"""
        from core.gan_generator import CTGANGenerator
        train_df, features, targets, cat_cols = gan_data

        gen = CTGANGenerator(epochs=20, batch_size=30, verbose=False)
        gen.fit(train_df, categorical_columns=cat_cols)

        # 条件生成：只测试不抛异常
        cond_synth = gen.conditional_generate(
            conditions={"curing_temp": 90},
            num_rows=20,
        )
        assert cond_synth is None or len(cond_synth) <= 20

    def test_ctgan_save_load(self, gan_data, tmp_path):
        """测试模型持久化"""
        from core.gan_generator import CTGANGenerator
        train_df, features, targets, cat_cols = gan_data

        gen = CTGANGenerator(epochs=20, batch_size=30, verbose=False)
        gen.fit(train_df, categorical_columns=cat_cols)

        # 保存
        save_path = os.path.join(tmp_path, "test_ctgan.pkl")
        gen.save(save_path)
        assert os.path.exists(save_path)

        # 加载
        gen2 = CTGANGenerator()
        gen2.load(save_path)
        assert gen2.synthesizer is not None

        # 加载后仍可生成
        synth = gen2.generate(30)
        assert len(synth) == 30

    def test_tvae_fit_and_generate(self, gan_data):
        """测试 TVAE 备选生成器"""
        from core.gan_generator import TVAEGenerator
        train_df, features, targets, cat_cols = gan_data

        gen = TVAEGenerator(epochs=20, batch_size=30, verbose=False)
        gen.fit(train_df, categorical_columns=cat_cols)

        synth = gen.generate(50)
        assert synth is not None
        assert len(synth) == 50

    def test_tstr_validator(self, gan_data):
        """测试 TSTR 验证器"""
        from core.gan_generator import CTGANGenerator, TSTRValidator
        train_df, features, targets, cat_cols = gan_data

        gen = CTGANGenerator(epochs=20, batch_size=30, verbose=False)
        gen.fit(train_df, categorical_columns=cat_cols)
        synth = gen.generate(100)

        validator = TSTRValidator(
            real_data=train_df,
            synthetic_data=synth,
            feature_columns=features,
            target_columns=targets,
        )
        results = validator.evaluate()

        # 应包含所有评估模块
        assert "statistics" in results
        assert "distributions" in results
        assert "correlation" in results
        assert "tstr" in results
        assert "coverage" in results

        # TSTR 应有汇总
        assert "_summary" in results["tstr"]
        assert "avg_r2_ratio" in results["tstr"]["_summary"]

        # 报告可生成
        report = validator.get_report()
        assert "TSTR" in report

        # 分布图可生成
        if features:
            fig = validator.plot_distribution_comparison(features[0])
            assert fig is not None

        # 摘要 DataFrame
        summary_df = validator.get_summary_dataframe()
        assert len(summary_df) > 0

    def test_auto_generate_with_validation(self, gan_data):
        """测试一键生成 + 验证"""
        from core.gan_generator import auto_generate_with_validation
        train_df, features, targets, cat_cols = gan_data

        result = auto_generate_with_validation(
            real_data=train_df,
            feature_columns=features,
            target_columns=targets,
            categorical_columns=cat_cols,
            epochs=20,
            batch_size=30,
            num_rows=50,
            run_validation=True,
        )

        assert "generator" in result
        assert "synthetic_data" in result
        assert len(result["synthetic_data"]) == 50
        assert "validation" in result


class TestConfig:
    """配置模块测试"""

    def test_feature_names(self):
        features = get_all_feature_names()
        assert len(features) == 12

    def test_target_names(self):
        targets = get_all_target_names()
        assert len(targets) == 5

    def test_ranges(self):
        from core.config import get_feature_ranges
        ranges = get_feature_ranges()
        assert "cement" in ranges
        assert ranges["cement"] == (600, 1000)
        assert "water" in ranges
        assert ranges["water"] == (140, 220)


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


# ═══════════════════════════════════════════════════════════════════
# 多目标进化优化测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_PYMOO, reason="pymoo 未安装")
class TestMultiObjectiveOptimizer:
    """多目标进化优化器测试"""

    @pytest.fixture
    def moo_data(self, dm):
        """准备优化数据"""
        features = [f for f in get_all_feature_names() if f in dm.cleaned_data.columns]
        targets = [t for t in get_all_target_names() if t in dm.cleaned_data.columns]
        X = dm.cleaned_data[features].select_dtypes(include=[np.number])
        Y = dm.cleaned_data[targets]
        return features, targets, X, Y

    def test_create_optimizer(self, moo_data):
        """测试优化器创建"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=50,
            verbose=False,
        )
        assert opt.algorithm == "nsga2"
        assert len(opt.feature_names) == 12
        assert opt.pop_size == 50

    def test_create_via_factory(self):
        """测试便捷创建函数"""
        opt = create_multi_objective_optimizer(
            algorithm="nsga2",
            pop_size=50,
            verbose=False,
        )
        assert len(opt.feature_names) == 12
        assert len(opt.target_names) == 5
        assert opt.algorithm == "nsga2"
        assert opt.pop_size == 50

    def test_build_bounds(self, moo_data):
        """测试变量边界构建"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=50,
            verbose=False,
        )
        assert opt.xl.shape == (len(features),)
        assert opt.xu.shape == (len(features),)
        # cement 范围 [600, 1000]
        idx_cement = features.index("cement")
        assert opt.xl[idx_cement] == 600
        assert opt.xu[idx_cement] == 1000

    def test_decode_curing_temp(self, moo_data):
        """测试 curing_temp 解码"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=50,
            verbose=False,
        )
        # 创建模拟 X
        X_test = np.zeros((4, len(features)))
        if "curing_temp" in features:
            idx = features.index("curing_temp")
            X_test[:, idx] = [0, 1, 2, 3]  # 编码值
            X_decoded = opt._decode_curing_temp(X_test)
            # 应映射到 [20, 60, 90, 250]
            assert X_decoded[0, idx] == 20
            assert X_decoded[1, idx] == 60
            assert X_decoded[2, idx] == 90
            assert X_decoded[3, idx] == 250

    def test_uhpc_problem_creation(self, moo_data):
        """测试 UHPCProblem 创建"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=50,
            verbose=False,
        )
        problem = opt._create_problem()
        assert problem.n_var == len(features)
        assert problem.n_obj == 3
        assert problem.n_constr == 4  # w/b 下限 + w/b 上限 + 体积下限 + 体积上限

    def test_uhpc_problem_evaluate(self, moo_data):
        """测试 UHPCProblem._evaluate"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=50,
            verbose=False,
        )
        problem = opt._create_problem()

        # 生成几个随机个体
        X_test = np.random.uniform(opt.xl, opt.xu, size=(5, len(features)))
        out = {}
        problem._evaluate(X_test, out)

        assert "F" in out
        assert "G" in out
        assert out["F"].shape == (5, 3)
        assert out["G"].shape == (5, 4)

    def test_surrogate_predict_no_model(self, moo_data):
        """测试无代理模型时的预测回退"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=50,
            verbose=False,
        )
        X_test = np.random.uniform(opt.xl, opt.xu, size=(5, len(features)))
        Y_mean, Y_std = opt._surrogate_predict(X_test)
        assert Y_mean.shape == (5, 3)
        assert Y_std.shape == (5, 3)

    def test_surrogate_predict_with_rf(self, moo_data):
        """测试带 RF 代理模型的预测"""
        features, targets, X, Y = moo_data
        # 训练 RF 模型
        from core.model_trainer import MultiTargetTrainer
        mt = MultiTargetTrainer("random_forest")
        X_numeric = X.select_dtypes(include=[np.number])
        mt.train_all(X_numeric, Y[targets[:3]])

        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            surrogate_model=mt,
            pop_size=50,
            verbose=False,
        )
        X_test = np.random.uniform(opt.xl, opt.xu, size=(5, len(features)))
        Y_mean, Y_std = opt._surrogate_predict(X_test)
        assert Y_mean.shape == (5, 3)
        assert Y_std.shape == (5, 3)
        # RF 预测值应在合理范围内
        assert not np.any(np.isnan(Y_mean))
        assert not np.any(np.isinf(Y_mean))

    def test_nsga2_optimize_no_model(self, moo_data):
        """测试 NSGA-II 优化（无代理模型，快速冒烟测试）"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=40,
            verbose=False,
        )
        results = opt.optimize(n_generations=5, seed=42)
        # 无代理模型时使用虚拟预测值，Pareto 前沿应为空或至少不报错
        assert "n_pareto" in results
        assert results["time"] >= 0
        assert len(results["history"]) > 0

    def test_nsga2_optimize_with_rf(self, moo_data):
        """测试 NSGA-II 优化（带 RF 代理模型）"""
        features, targets, X, Y = moo_data
        from core.model_trainer import MultiTargetTrainer
        mt = MultiTargetTrainer("random_forest")
        X_numeric = X.select_dtypes(include=[np.number])
        mt.train_all(X_numeric, Y[targets[:3]])

        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            surrogate_model=mt,
            pop_size=50,
            verbose=False,
        )
        results = opt.optimize(n_generations=10, seed=42)
        assert results["n_pareto"] >= 0
        assert len(results["history"]) == 10

        # Pareto 解的特征值应在合理范围内
        if results["n_pareto"] > 0:
            pX = results["pareto_X"]
            idx_cement = features.index("cement")
            assert pX[:, idx_cement].min() >= 500  # 不应偏离太远
            assert pX[:, idx_cement].max() <= 1100

    def test_agemoea_optimize(self, moo_data):
        """测试 AGE-MOEA 优化（冒烟测试）"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="agemoea",
            pop_size=40,
            verbose=False,
        )
        results = opt.optimize(n_generations=5, seed=42)
        assert results["algorithm"] == "AGE-MOEA"
        assert results["time"] >= 0

    def test_nsga3_optimize(self, moo_data):
        """测试 NSGA-III 优化（冒烟测试）"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga3",
            pop_size=40,
            verbose=False,
        )
        results = opt.optimize(n_generations=5, seed=42)
        assert results["algorithm"] == "NSGA-III"
        assert results["time"] >= 0

    def test_get_pareto_df(self, moo_data):
        """测试 get_pareto_df"""
        features, targets, X, Y = moo_data
        from core.model_trainer import MultiTargetTrainer
        mt = MultiTargetTrainer("random_forest")
        X_numeric = X.select_dtypes(include=[np.number])
        mt.train_all(X_numeric, Y[targets[:3]])

        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            surrogate_model=mt,
            pop_size=50,
            verbose=False,
        )
        opt.optimize(n_generations=10, seed=42)
        df = opt.get_pareto_df()
        assert len(df) >= 0
        # 应包含特征列和目标列
        for f in features:
            assert f in df.columns
        for t in targets[:3]:
            assert t in df.columns

    def test_get_convergence_df(self, moo_data):
        """测试收敛历史 DataFrame"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=40,
            verbose=False,
        )
        opt.optimize(n_generations=5, seed=42)
        conv = opt.get_convergence_df()
        assert len(conv) == 5
        assert "generation" in conv.columns
        assert "n_nondominated" in conv.columns

    def test_save_and_load(self, moo_data, tmp_path):
        """测试优化结果持久化"""
        features, targets, X, Y = moo_data
        from core.model_trainer import MultiTargetTrainer
        mt = MultiTargetTrainer("random_forest")
        X_numeric = X.select_dtypes(include=[np.number])
        mt.train_all(X_numeric, Y[targets[:3]])

        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            surrogate_model=mt,
            pop_size=40,
            verbose=False,
        )
        opt.optimize(n_generations=5, seed=42)

        save_path = os.path.join(tmp_path, "test_moo.pkl")
        opt.save(save_path)
        assert os.path.exists(save_path)

        opt2 = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=40,
            verbose=False,
        )
        opt2.load(save_path)
        assert opt2.feature_names == features
        assert opt2.target_names == targets[:3]
        assert opt2.pop_size == 40

    def test_invalid_algorithm(self, moo_data):
        """测试无效算法名应报错"""
        features, targets, X, Y = moo_data
        with pytest.raises(ValueError):
            MultiObjectiveOptimizer(
                feature_names=features,
                target_names=targets[:3],
                algorithm="invalid_algo",
            )

    def test_reverse_objectives(self, moo_data):
        """测试目标值还原"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=40,
            verbose=False,
        )
        # maximize 被取负后应还原
        F = np.array([
            [-200.0, -30.0, 3000.0],
            [-180.0, -35.0, 2500.0],
        ])
        Y_rev = opt._reverse_objectives(F)
        assert Y_rev[0, 0] == 200.0  # -(-200)
        assert Y_rev[0, 1] == 30.0   # -(-30)
        assert Y_rev[0, 2] == 3000.0  # minimize 不变

    def test_constraints_in_evaluation(self, moo_data):
        """测试约束在评估中的计算"""
        features, targets, X, Y = moo_data
        opt = MultiObjectiveOptimizer(
            feature_names=features,
            target_names=targets[:3],
            directions=["maximize", "maximize", "minimize"],
            algorithm="nsga2",
            pop_size=40,
            verbose=False,
        )
        problem = opt._create_problem()

        # 测试水胶比约束
        X_test = np.ones((3, len(features))) * opt.xl  # 全取下界
        out = {}
        problem._evaluate(X_test, out)
        G = out["G"]
        assert G.shape == (3, 4)

        # 水胶比低于下限，G[:,0] > 0（违反约束）
        if "water" in features and "cement" in features:
            idx_w = features.index("water")
            idx_c = features.index("cement")
            # 构造 w/b < 0.14 的数据
            X_bad = X_test.copy()
            X_bad[:, idx_c] = 1000
            X_bad[:, idx_w] = 100  # w/b = 100/1000 = 0.10 < 0.14
            out_bad = {}
            problem._evaluate(X_bad, out_bad)
            # G[:,0] = 0.14 - w/b = 0.14 - 0.10 > 0 → 违反约束
            assert out_bad["G"][0, 0] > 0


# ═══════════════════════════════════════════════════════════════════
# TOPSIS 多准则决策测试
# ═══════════════════════════════════════════════════════════════════

class TestTOPSISSelector:
    """TOPSIS 选择器测试"""

    @pytest.fixture
    def sample_pareto(self):
        """生成示例 Pareto 前沿数据"""
        np.random.seed(42)
        n, m = 20, 3
        # 模拟 3 目标的 Pareto 前沿
        Y = np.column_stack([
            np.random.uniform(150, 220, n),   # 抗压强度 (maximize)
            np.random.uniform(25, 55, n),     # 抗弯强度 (maximize)
            np.random.uniform(2000, 4000, n),  # 成本 (minimize)
        ])
        return Y

    def test_create_selector(self, sample_pareto):
        """测试 TOPSIS 选择器创建"""
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        assert selector.pareto_Y.shape == (20, 3)
        assert len(selector.weights) == 3
        # 默认等权
        assert np.allclose(selector.weights.sum(), 1.0)

    def test_custom_weights(self, sample_pareto):
        """测试自定义权重"""
        weights = np.array([0.5, 0.3, 0.2])
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
            weights=weights,
        )
        assert np.allclose(selector.weights, weights)

    def test_compute_scores(self, sample_pareto):
        """测试 TOPSIS 得分计算"""
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        scores = selector.compute_scores()
        assert len(scores) == 20
        # 得分应在 [0, 1] 范围内
        assert scores.min() >= 0
        assert scores.max() <= 1.0
        # 得分应全为非 NaN
        assert not np.any(np.isnan(scores))

    def test_rank(self, sample_pareto):
        """测试 TOPSIS 排序"""
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        df = selector.rank()
        assert len(df) == 20
        assert "Rank" in df.columns
        assert "TOPSIS得分" in df.columns
        # 应按得分降序排列
        assert df["TOPSIS得分"].iloc[0] >= df["TOPSIS得分"].iloc[-1]

    def test_get_top_n(self, sample_pareto):
        """测试获取 Top-N 方案"""
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        top5 = selector.get_top_n(5)
        assert len(top5) == 5
        assert top5["Rank"].iloc[0] == 1

        top10 = selector.get_top_n(10)
        assert len(top10) == 10

    def test_get_top_n_with_X(self, sample_pareto):
        """测试带特征值的 Top-N 查询"""
        n = len(sample_pareto)
        X = np.random.randn(n, 5)
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
            pareto_X=X,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )
        top3 = selector.get_top_n(3)
        assert len(top3) == 3
        # 应包含特征列
        for f in ["f1", "f2", "f3", "f4", "f5"]:
            assert f in top3.columns

    def test_to_benefit(self, sample_pareto):
        """测试目标方向统一转换"""
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        Yb = selector._to_benefit(sample_pareto)
        assert Yb.shape == sample_pareto.shape
        # 转换后所有值应非负
        assert (Yb >= 0).all()

    def test_get_scores_detail(self, sample_pareto):
        """测试 TOPSIS 详细结果"""
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        detail = selector.get_scores_detail()
        assert "scores" in detail
        assert "dist_to_pis" in detail
        assert "dist_to_nis" in detail
        assert "pis" in detail
        assert "nis" in detail
        assert "weights" in detail

        # PIS 各维应 ≥ NIS
        assert (detail["pis"] >= detail["nis"]).all()

    def test_plot_radar(self, sample_pareto):
        """测试雷达图生成"""
        if not HAS_PLOTLY_TOPSIS:
            pytest.skip("plotly 未安装")
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        selector.compute_scores()
        fig = selector.plot_radar(n_top=3)
        assert fig is not None

    def test_plot_parallel_coordinates(self, sample_pareto):
        """测试平行坐标图生成"""
        if not HAS_PLOTLY_TOPSIS:
            pytest.skip("plotly 未安装")
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        selector.compute_scores()
        fig = selector.plot_parallel_coordinates(highlight_top=3)
        assert fig is not None

    def test_plot_score_bar(self, sample_pareto):
        """测试得分柱状图生成"""
        if not HAS_PLOTLY_TOPSIS:
            pytest.skip("plotly 未安装")
        selector = TOPSISSelector(
            pareto_Y=sample_pareto,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        selector.compute_scores()
        fig = selector.plot_score_bar(n_top=10)
        assert fig is not None

    def test_create_via_factory(self, sample_pareto):
        """测试便捷创建函数"""
        selector = create_topsis_selector(
            pareto_Y=sample_pareto,
            target_names=["compressive_strength_28d", "flexural_strength_28d", "cost"],
        )
        assert len(selector.target_names) == 3
        assert selector.pareto_Y.shape == (20, 3)

    def test_zero_variance_column(self):
        """测试零方差目标列（边界情况）"""
        Y = np.array([
            [200, 30, 3000],
            [200, 35, 2800],
            [200, 28, 3200],
        ])  # 第 1 列全 200
        selector = TOPSISSelector(
            pareto_Y=Y,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        scores = selector.compute_scores()
        assert not np.any(np.isnan(scores))
        assert len(scores) == 3

    def test_single_solution(self):
        """测试单个 Pareto 解的边界情况"""
        Y = np.array([[180, 40, 2500]])
        selector = TOPSISSelector(
            pareto_Y=Y,
            target_names=["strength", "flexural", "cost"],
            directions=["maximize", "maximize", "minimize"],
        )
        scores = selector.compute_scores()
        assert len(scores) == 1
        # 单个解时，得分为 1（到 NIS 距离 = 到 PIS 距离）
        assert 0 <= scores[0] <= 1

    def test_empty_pareto_raises(self):
        """测试空 Pareto 前沿应报错"""
        with pytest.raises(ValueError):
            TOPSISSelector(
                pareto_Y=np.array([]).reshape(0, 3),
                target_names=["a", "b", "c"],
                directions=["maximize", "maximize", "minimize"],
            )


# ═══════════════════════════════════════════════════════════════════
# 闭环反馈测试
# ═══════════════════════════════════════════════════════════════════

class TestFeedbackManager:
    """闭环反馈管理器测试"""

    @pytest.fixture
    def fb_data(self, dm):
        """准备反馈测试数据"""
        features = [f for f in get_all_feature_names() if f in dm.cleaned_data.columns]
        targets = [t for t in get_all_target_names() if t in dm.cleaned_data.columns]
        X = dm.cleaned_data[features].select_dtypes(include=[np.number])
        Y = dm.cleaned_data[targets]
        return features, targets, X, Y

    @pytest.fixture
    def fb_with_model(self, fb_data):
        """带已训练模型的 FeedbackManager"""
        features, targets, X, Y = fb_data
        fm = FeedbackManager()
        # 前 25 条作为原始训练数据
        fm.original_X = X.iloc[:25].copy()
        fm.original_Y = Y.iloc[:25].copy()
        fm.feature_names = features
        fm.target_names = targets
        # 训练模型
        from core.model_trainer import MultiTargetTrainer
        mt = MultiTargetTrainer("random_forest")
        mt.train_all(fm.original_X, fm.original_Y)
        fm.original_trainer = mt
        # 后 10 条作为实验数据
        exp_df = pd.concat([X.iloc[25:35], Y.iloc[25:35]], axis=1)
        fm.import_experiment_data(exp_df, features, targets)
        return fm, features, targets

    # ── 数据导入 ────────────────────────────────────────────────

    def test_import_experiment_data(self, fb_data):
        """测试导入有效实验数据"""
        features, targets, X, Y = fb_data
        fm = FeedbackManager()
        exp_df = pd.concat([X.head(5), Y.head(5)], axis=1)
        result = fm.import_experiment_data(exp_df, features, targets)
        assert len(result) == 5
        assert fm.feature_names == features
        assert fm.target_names == targets
        assert fm.experiment_data is not None

    def test_import_missing_columns(self, fb_data):
        """测试缺列时正确报错"""
        features, targets, X, Y = fb_data
        fm = FeedbackManager()
        # 只给特征不给目标
        with pytest.raises(ValueError, match="缺少以下列"):
            fm.import_experiment_data(X.head(5), features, targets)

    def test_import_drops_nan_rows(self, fb_data):
        """测试导入时自动去除含 NaN 的行"""
        features, targets, X, Y = fb_data
        fm = FeedbackManager()
        exp_df = pd.concat([X.head(10), Y.head(10)], axis=1)
        # 人为制造 NaN
        exp_df.iloc[2, 0] = np.nan  # 特征 NaN
        exp_df.iloc[5, 12] = np.nan  # 目标 NaN (col 12 = first target col)
        result = fm.import_experiment_data(exp_df, features, targets)
        # 应移除 2 行
        assert len(result) == 8

    # ── 预测 ────────────────────────────────────────────────────

    def test_predict_on_experiments(self, fb_with_model):
        """测试预测输出形状正确"""
        fm, features, targets = fb_with_model
        preds = fm.predict_on_experiments()
        assert preds.shape == (10, len(targets))
        assert list(preds.columns) == targets

    def test_predict_no_model_raises(self, fb_data):
        """测试无模型时预测报错"""
        features, targets, X, Y = fb_data
        fm = FeedbackManager()
        exp_df = pd.concat([X.head(5), Y.head(5)], axis=1)
        fm.import_experiment_data(exp_df, features, targets)
        with pytest.raises(RuntimeError, match="没有可用的模型"):
            fm.predict_on_experiments()

    def test_predict_no_experiment_raises(self):
        """测试无实验数据时预测报错"""
        fm = FeedbackManager()
        from core.model_trainer import MultiTargetTrainer
        mt = MultiTargetTrainer("random_forest")
        # 手动设一个空模型
        fm.original_trainer = mt
        with pytest.raises(RuntimeError, match="尚未导入实验数据"):
            fm.predict_on_experiments()

    # ── 对比指标 ────────────────────────────────────────────────

    def test_compare_metrics(self, fb_with_model):
        """测试对比指标完整且合理"""
        fm, features, targets = fb_with_model
        preds = fm.predict_on_experiments()
        measured = fm.get_experiment_Y().values
        comp = fm.compare(measured, preds.values, targets)

        assert "per_target" in comp
        assert "overall" in comp
        assert len(comp["per_target"]) == len(targets)

        # 每个目标应有完整指标
        for t in targets:
            m = comp["per_target"][t]
            assert "r2" in m
            assert "mae" in m
            assert "rmse" in m
            assert "mape" in m
            assert "max_error" in m
            # R² 应在合理范围内
            if not np.isnan(m["r2"]):
                assert m["r2"] <= 1.0
            # MAE 和 RMSE 应为非负
            if not np.isnan(m["mae"]):
                assert m["mae"] >= 0
            if not np.isnan(m["rmse"]):
                assert m["rmse"] >= 0

        # 汇总指标
        ov = comp["overall"]
        assert "r2_mean" in ov
        assert "mae_mean" in ov

    def test_comparison_df(self, fb_with_model):
        """测试对比指标 DataFrame 格式"""
        fm, features, targets = fb_with_model
        preds = fm.predict_on_experiments()
        measured = fm.get_experiment_Y().values
        comp = fm.compare(measured, preds.values, targets)
        df = fm.get_comparison_df(comp)
        assert len(df) == len(targets) + 1  # 每个目标 + 汇总行
        assert "目标" in df.columns
        assert "R²" in df.columns
        assert "MAE" in df.columns
        # 最后一行应为汇总
        assert "平均" in df["目标"].iloc[-1]

    def test_compare_dimension_mismatch(self, fb_with_model):
        """测试维度不匹配时报错"""
        fm, features, targets = fb_with_model
        preds = fm.predict_on_experiments()
        measured = fm.get_experiment_Y().values
        with pytest.raises(ValueError, match="维度不匹配"):
            fm.compare(measured[:3, :], preds.values, targets)

    # ── 模型更新 ────────────────────────────────────────────────

    def test_incremental_update(self, fb_with_model):
        """测试增量更新后模型可预测"""
        fm, features, targets = fb_with_model
        new_trainer = fm.incremental_update("random_forest")
        assert new_trainer is not None
        assert fm.updated_trainer is not None
        assert len(new_trainer.models) == len(targets)
        # 更新后应能预测
        X_exp = fm.get_experiment_X()
        preds = new_trainer.predict_all(X_exp)
        assert preds.shape == (10, len(targets))

    def test_full_retrain(self, fb_with_model):
        """测试全量重训练后模型可预测"""
        fm, features, targets = fb_with_model
        new_trainer = fm.full_retrain("random_forest")
        assert new_trainer is not None
        assert fm.updated_trainer is not None
        assert len(new_trainer.models) == len(targets)

    def test_update_improves_or_stable(self, fb_with_model):
        """测试更新后 R² 不显著下降（在测试集上）"""
        fm, features, targets = fb_with_model
        from sklearn.model_selection import train_test_split

        # 划分测试集
        X_base = fm.original_X
        Y_base = fm.original_Y
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_base, Y_base, test_size=0.3, random_state=42
        )

        # 临时调整训练数据
        orig_X = fm.original_X
        orig_Y = fm.original_Y
        fm.original_X = X_train
        fm.original_Y = Y_train
        # 重新训练原始模型
        from core.model_trainer import MultiTargetTrainer
        mt = MultiTargetTrainer("random_forest")
        mt.train_all(X_train, Y_train)
        fm.original_trainer = mt

        # 执行增量更新
        fm.incremental_update("random_forest")

        # 评估
        eval_result = fm.evaluate_update(X_test, Y_test)
        for t in eval_result["delta"]:
            # R² 下降不应超过 0.3（由于数据量小，允许一定波动）
            if not np.isnan(eval_result["delta"][t]["r2_delta"]):
                assert eval_result["delta"][t]["r2_delta"] > -0.3, \
                    f"{t}: R² dropped by {eval_result['delta'][t]['r2_delta']}"

        # 恢复
        fm.original_X = orig_X
        fm.original_Y = orig_Y

    def test_update_without_original_data(self, fb_data):
        """测试无原始数据时更新报错"""
        features, targets, X, Y = fb_data
        fm = FeedbackManager()
        exp_df = pd.concat([X.head(5), Y.head(5)], axis=1)
        fm.import_experiment_data(exp_df, features, targets)
        with pytest.raises(RuntimeError, match="original_X"):
            fm.incremental_update()

    def test_evaluate_without_updated_model(self, fb_with_model):
        """测试尚未更新时评估报错"""
        fm, features, targets = fb_with_model
        X_test = fm.original_X.head(5)
        Y_test = fm.original_Y.head(5)
        with pytest.raises(RuntimeError, match="没有更新后的模型"):
            fm.evaluate_update(X_test, Y_test)

    # ── 闭环日志 ────────────────────────────────────────────────

    def test_log_feedback(self, fb_with_model):
        """测试日志追加正确"""
        fm, features, targets = fb_with_model
        entry = fm.log_feedback(
            source="测试数据",
            n_samples=10,
            update_type="增量更新",
            metrics_before={"r2_mean": 0.85, "mae_mean": 12.5},
            metrics_after={"r2_mean": 0.88, "mae_mean": 10.2},
        )
        assert entry["cycle"] == 1
        assert entry["source"] == "测试数据"
        assert entry["n_samples"] == 10
        assert entry["r2_before"] == 0.85
        assert entry["r2_after"] == 0.88

        # 第二次日志
        fm.log_feedback(
            source="第二轮实验",
            n_samples=5,
            update_type="全量重训练",
        )
        assert len(fm.feedback_log) == 2
        assert fm.feedback_log[1]["cycle"] == 2

    def test_get_log_df(self, fb_with_model):
        """测试日志转 DataFrame 格式正确"""
        fm, features, targets = fb_with_model
        fm.log_feedback("测试", 10, "增量更新",
                        {"r2_mean": 0.8}, {"r2_mean": 0.85})
        fm.log_feedback("测试2", 5, "全量重训练",
                        {"r2_mean": 0.85}, {"r2_mean": 0.88})

        df = fm.get_log_df()
        assert len(df) == 2
        assert "cycle" in df.columns
        assert "source" in df.columns
        assert "n_samples" in df.columns
        assert "update_type" in df.columns

    def test_empty_log_df(self):
        """测试空日志返回空 DataFrame"""
        fm = FeedbackManager()
        df = fm.get_log_df()
        assert df.empty

    # ── 持久化 ──────────────────────────────────────────────────

    def test_save_load_state(self, fb_with_model, tmp_path):
        """测试持久化往返正确"""
        fm, features, targets = fb_with_model
        # 先训练更新模型
        fm.incremental_update("random_forest")
        fm.log_feedback("测试", 10, "增量更新",
                        {"r2_mean": 0.8}, {"r2_mean": 0.85})

        save_path = os.path.join(tmp_path, "test_feedback.pkl")
        fm.save_state(save_path)
        assert os.path.exists(save_path)

        # 加载到新实例
        fm2 = FeedbackManager()
        fm2.load_state(save_path)
        assert fm2.feature_names == features
        assert fm2.target_names == targets
        assert len(fm2.feedback_log) == 1
        assert fm2.experiment_data is not None
        assert len(fm2.experiment_data) == 10
        assert fm2.original_X is not None
        assert fm2.original_Y is not None

    # ── 边界情况 ────────────────────────────────────────────────

    def test_empty_experiment_after_nan_cleanup(self, fb_data):
        """测试全NaN数据导入后实验数据为空"""
        features, targets, X, Y = fb_data
        fm = FeedbackManager()
        exp_df = pd.concat([X.head(5), Y.head(5)], axis=1)
        # 所有目标列设为 NaN
        for t in targets:
            exp_df[t] = np.nan
        result = fm.import_experiment_data(exp_df, features, targets)
        # 应移除所有行
        assert len(result) == 0

    def test_single_sample_experiment(self, fb_data):
        """测试单条实验数据"""
        features, targets, X, Y = fb_data
        fm = FeedbackManager()
        from core.model_trainer import MultiTargetTrainer

        # 用 30 条训练
        fm.original_X = X.iloc[:30].copy()
        fm.original_Y = Y.iloc[:30].copy()
        mt = MultiTargetTrainer("random_forest")
        mt.train_all(fm.original_X, fm.original_Y)
        fm.original_trainer = mt

        # 单条实验数据
        exp_df = pd.concat([X.iloc[30:31], Y.iloc[30:31]], axis=1)
        fm.import_experiment_data(exp_df, features, targets)
        assert len(fm.experiment_data) == 1

        # 应能预测
        preds = fm.predict_on_experiments()
        assert preds.shape == (1, len(targets))

        # 应能对比（R² 对单样本为 NaN）
        measured = fm.get_experiment_Y().values
        comp = fm.compare(measured, preds.values, targets)
        # R² 需要至少 2 个样本
        for t in targets:
            assert comp["per_target"][t]["n_valid"] == 1
            assert np.isnan(comp["per_target"][t]["r2"])

    def test_plot_functions(self, fb_with_model):
        """测试可视化函数正常生成"""
        fm, features, targets = fb_with_model
        preds = fm.predict_on_experiments()
        measured = fm.get_experiment_Y().values

        # 散点图
        fig = fm.plot_pred_vs_measured(
            measured[:, 0], preds.values[:, 0], targets[0],
        )
        assert fig is not None
        plt.close(fig)

        # 残差图
        fig = fm.plot_residuals(
            measured[:, 0], preds.values[:, 0], targets[0],
        )
        assert fig is not None
        plt.close(fig)

        # 误差柱状图
        comp = fm.compare(measured, preds.values, targets)
        fig = fm.plot_error_bar(comp)
        assert fig is not None
        plt.close(fig)

    def test_plot_without_comparison(self):
        """测试无对比结果时 plot_error_bar 报错"""
        fm = FeedbackManager()
        with pytest.raises(RuntimeError, match="没有对比结果"):
            fm.plot_error_bar()
