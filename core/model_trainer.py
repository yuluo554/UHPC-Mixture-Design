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

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple:
        """预测均值与逐样本标准差。

        不同模型类型的方差估计策略：
          - random_forest:     所有树的预测值标准差（真实 epistemic uncertainty）
          - gradient_boosting: staged_predict 阶段间标准差
          - xgboost:          按迭代间隔采样的预测值标准差
          - lightgbm:         按迭代间隔采样的预测值标准差

        Args:
            X: 特征 DataFrame

        Returns:
            (mean, std) — 预测均值 (n,) 和标准差 (n,)

        Raises:
            RuntimeError: 模型未训练
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        X_vals = X.values if isinstance(X, pd.DataFrame) else X

        if self.model_type == "random_forest":
            # 所有树的预测 → 树间标准差
            tree_preds = np.array([
                tree.predict(X_vals) for tree in self.model.estimators_
            ])  # (n_trees, n_samples)
            mean = tree_preds.mean(axis=0)
            std = tree_preds.std(axis=0, ddof=1)
            # 防止单样本/纯节点导致 std=0
            std = np.maximum(std, np.abs(mean) * 0.001 + 1e-6)

        elif self.model_type == "gradient_boosting":
            # staged_predict 逐阶段预测
            staged = list(self.model.staged_predict(X_vals))
            if len(staged) <= 1:
                mean = staged[0] if staged else self.predict(X)
                std = np.full_like(mean, 0.05 * np.abs(mean) + 0.01)
            else:
                staged_arr = np.column_stack(staged)  # (n, n_stages)
                mean = staged_arr[:, -1]  # 最终阶段
                std = staged_arr.std(axis=1, ddof=1)
                std = np.maximum(std, np.abs(mean) * 0.001 + 1e-6)

        elif self.model_type == "xgboost":
            # 按迭代间隔采样（避免 200 棵树 → 200 次预测）
            n_trees = self.model.n_estimators
            sample_interval = max(1, n_trees // 50)
            sampled = []
            for i in range(sample_interval, n_trees + 1, sample_interval):
                try:
                    pred = self.model.predict(X_vals, iteration_range=(0, i))
                except TypeError:
                    pred = self.model.predict(X_vals, ntree_limit=i)
                sampled.append(pred)
            final_mean = sampled[-1] if sampled else self.predict(X)
            if len(sampled) <= 1:
                std = np.full_like(final_mean, 0.05 * np.abs(final_mean) + 0.01)
            else:
                all_preds = np.column_stack(sampled)
                std = all_preds.std(axis=1, ddof=1)
                std = np.maximum(std, np.abs(final_mean) * 0.001 + 1e-6)
            mean = final_mean

        elif self.model_type == "lightgbm":
            # 按迭代间隔采样
            n_trees = self.model.n_estimators
            sample_interval = max(1, n_trees // 50)
            final_mean = self.model.predict(X_vals)
            sampled = []
            for i in range(sample_interval, n_trees + 1, sample_interval):
                pred = self.model.predict(X_vals, num_iteration=i)
                sampled.append(pred)
            if len(sampled) <= 1:
                std = np.full_like(final_mean, 0.05 * np.abs(final_mean) + 0.01)
            else:
                all_preds = np.column_stack(sampled)
                std = all_preds.std(axis=1, ddof=1)
                std = np.maximum(std, np.abs(final_mean) * 0.001 + 1e-6)
            mean = final_mean

        else:
            raise ValueError(
                f"不支持的模型类型: {self.model_type}，"
                f"支持: {self.MODEL_TYPES}"
            )

        return mean.ravel(), std.ravel()

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

    def predict_all_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple:
        """返回所有目标的预测均值和标准差。

        Args:
            X: 特征 DataFrame

        Returns:
            (mean_df, std_df) — 列名均为 self.models 的键
        """
        means = {}
        stds = {}
        for name, trainer in self.models.items():
            m, s = trainer.predict_with_uncertainty(X)
            means[name] = m
            stds[name] = s
        return pd.DataFrame(means), pd.DataFrame(stds)

    def cross_validate_all(self, X: pd.DataFrame, Y: pd.DataFrame, cv: int = 10) -> dict:
        """对所有目标做交叉验证"""
        all_results = {}
        for col in Y.columns:
            trainer = ModelTrainer(model_type=self.model_type)
            all_results[col] = trainer.cross_validate(X, Y[col], cv=cv)
        return all_results

    def evaluate_on_test(
        self, X_test: pd.DataFrame, Y_test: pd.DataFrame
    ) -> pd.DataFrame:
        """在独立测试集上评估所有目标。

        与 cross_validate_all 不同：此方法使用已训练的模型
        （train_all 之后），在留出的测试集上一次性评估。
        返回每个目标的 R² / MAE / RMSE。

        Args:
            X_test: 测试集特征
            Y_test: 测试集目标

        Returns:
            DataFrame，列为 target, r2, mae, rmse
        """
        rows = []
        for target_name, trainer in self.models.items():
            if target_name not in Y_test.columns:
                continue
            y_true = Y_test[target_name].values
            y_pred = trainer.predict(X_test)
            rows.append({
                "target": target_name,
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            })
        return pd.DataFrame(rows)

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
