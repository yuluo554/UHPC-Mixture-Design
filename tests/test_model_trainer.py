# -*- coding: utf-8 -*-
"""ModelTrainer + MultiTargetTrainer 测试"""

import pytest
from core.model_trainer import ModelTrainer, MultiTargetTrainer


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
