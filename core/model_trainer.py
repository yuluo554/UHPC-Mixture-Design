# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign 模型训练模块
支持 GP / RandomForest / XGBoost / Ensemble
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# 可选依赖
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class ModelTrainer:
    """UHPC 性能预测模型训练器"""

    MODEL_TYPES = ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]

    def __init__(self, model_type: str = "random_forest"):
        """
        Args:
            model_type: 模型类型
        """
        self.model_type = model_type
        self.model = None
        self.cv_results = {}

    def _create_model(self, **kwargs):
        """创建底层模型实例"""
        default_rf = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
        default_gb = {"n_estimators": 200, "max_depth": 6, "random_state": 42}
        default_xgb = {"n_estimators": 200, "max_depth": 6, "random_state": 42, "verbosity": 0}
        default_lgb = {"n_estimators": 200, "max_depth": 6, "random_state": 42, "verbose": -1}

        if self.model_type == "random_forest":
            params = {**default_rf, **kwargs}
            return RandomForestRegressor(**params)
        elif self.model_type == "gradient_boosting":
            params = {**default_gb, **kwargs}
            return GradientBoostingRegressor(**params)
        elif self.model_type == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError("请安装 xgboost: pip install xgboost")
            params = {**default_xgb, **kwargs}
            return xgb.XGBRegressor(**params)
        elif self.model_type == "lightgbm":
            if not HAS_LIGHTGBM:
                raise ImportError("请安装 lightgbm: pip install lightgbm")
            params = {**default_lgb, **kwargs}
            return lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """训练单个模型"""
        self.model = self._create_model(**kwargs)
        self.model.fit(X, y)
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")
        return self.model.predict(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 10) -> dict:
        """
        K 折交叉验证
        Returns:
            {r2_mean, r2_std, mae_mean, mae_std, rmse_mean, rmse_std, scores}
        """
        model = self._create_model()
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        r2_scores = []
        mae_scores = []
        rmse_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            r2_scores.append(r2_score(y_val, y_pred))
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        self.cv_results = {
            "r2_mean": np.mean(r2_scores),
            "r2_std": np.std(r2_scores),
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
            "rmse_mean": np.mean(rmse_scores),
            "rmse_std": np.std(rmse_scores),
            "r2_scores": r2_scores,
        }
        return self.cv_results

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """在测试集上评估"""
        y_pred = self.predict(X_test)
        return {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        """加载模型"""
        self.model = joblib.load(path)
        return self


class MultiTargetTrainer:
    """多目标模型训练器：为每个目标训练独立的模型"""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.models = {}  # {target_name: ModelTrainer}
        self.feature_names = None

    def train_all(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        """为每个目标列训练一个模型"""
        self.feature_names = list(X.columns)
        for col in Y.columns:
            trainer = ModelTrainer(model_type=self.model_type)
            trainer.train(X, Y[col], **kwargs)
            self.models[col] = trainer
        return self

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """返回所有目标的预测值 DataFrame"""
        results = {}
        for name, trainer in self.models.items():
            results[name] = trainer.predict(X)
        return pd.DataFrame(results)

    def cross_validate_all(self, X: pd.DataFrame, Y: pd.DataFrame, cv: int = 10) -> dict:
        """对所有目标做交叉验证"""
        all_results = {}
        for col in Y.columns:
            trainer = ModelTrainer(model_type=self.model_type)
            all_results[col] = trainer.cross_validate(X, Y[col], cv=cv)
        return all_results

    def save_all(self, directory: str):
        """保存所有模型到指定目录"""
        os.makedirs(directory, exist_ok=True)
        for name, trainer in self.models.items():
            path = os.path.join(directory, f"{self.model_type}_{name}_model.pkl")
            trainer.save(path)

    def load_all(self, directory: str):
        """从目录加载所有模型"""
        self.models = {}
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                # 解析文件名: "{model_type}_{target}_model.pkl"
                parts = filename.replace("_model.pkl", "").split("_", 1)
                target_name = parts[1] if len(parts) > 1 else parts[0]
                trainer = ModelTrainer(model_type=self.model_type)
                trainer.load(os.path.join(directory, filename))
                self.models[target_name] = trainer
        return self
