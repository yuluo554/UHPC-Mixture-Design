# -*- coding: utf-8 -*-
"""FeedbackManager 测试"""

import os
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.config import get_all_feature_names, get_all_target_names
from core.model_trainer import ModelTrainer, MultiTargetTrainer
from core.feedback_manager import FeedbackManager, TARGET_LABELS


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

