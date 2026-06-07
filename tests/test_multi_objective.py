# -*- coding: utf-8 -*-
"""多目标优化器 + TOPSIS 决策测试"""

import os
import pytest
import numpy as np
from core.config import get_all_feature_names, get_all_target_names
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

