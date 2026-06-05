# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign 数据管理模块
数据加载、清洗、预处理、统计
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class DataManager:
    """UHPC 配比数据管理器"""

    def __init__(self, feature_columns=None, target_columns=None):
        """
        Args:
            feature_columns: 输入特征列名列表
            target_columns: 目标列名列表
        """
        self.feature_columns = feature_columns or []
        self.target_columns = target_columns or []
        self.raw_data = None
        self.cleaned_data = None
        self.outlier_mask = None
        self.scaler_X = None
        self.scaler_y = None

    # ─── 数据加载 ────────────────────────────────────────────────

    def load_csv(self, path: str) -> pd.DataFrame:
        """加载 CSV 文件并自动检测列"""
        df = pd.read_csv(path)
        self.raw_data = df

        # 如果没有手动指定列，尝试自动推断
        if not self.feature_columns:
            self._auto_detect_columns(df)

        return df

    def load_dataframe(self, df: pd.DataFrame):
        """直接加载 DataFrame"""
        self.raw_data = df.copy()
        if not self.feature_columns:
            self._auto_detect_columns(df)
        return self

    def _auto_detect_columns(self, df: pd.DataFrame):
        """从 DataFrame 列中自动区分特征和目标"""
        from core.config import get_all_feature_names, get_all_target_names

        known_features = get_all_feature_names()
        known_targets = get_all_target_names()

        self.feature_columns = [c for c in known_features if c in df.columns]
        self.target_columns = [c for c in known_targets if c in df.columns]

        # 如果配置中没有，尝试从实际列中推断
        if not self.feature_columns:
            self.feature_columns = [c for c in df.columns if c not in known_targets]
        if not self.target_columns:
            self.target_columns = [c for c in df.columns if c in known_targets]

    # ─── 数据清洗 ────────────────────────────────────────────────

    def detect_outliers(self, contamination: float = 0.05) -> pd.Series:
        """
        使用 Isolation Forest 检测异常值
        Args:
            contamination: 预期异常比例
        Returns:
            bool Series: True=正常, False=异常
        """
        df = self.raw_data[self.feature_columns].copy()
        df = df.fillna(df.median())

        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        preds = iso.fit_predict(df)
        self.outlier_mask = preds == 1  # 1=正常, -1=异常
        return self.outlier_mask

    def impute_missing(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        使用 MICE (IterativeImputer) 多重插补缺失值
        """
        if df is None:
            df = self.cleaned_data if self.cleaned_data is not None else self.raw_data

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = IterativeImputer(max_iter=10, random_state=42)
        df_imputed = df.copy()
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df_imputed

    def clean(self, remove_outliers: bool = True, impute: bool = True,
              contamination: float = 0.05) -> pd.DataFrame:
        """
        一键清洗：异常值检测 + 缺失值插补
        Returns:
            清洗后的 DataFrame
        """
        df = self.raw_data.copy()

        if remove_outliers and self.feature_columns:
            self.detect_outliers(contamination=contamination)
            df = df[self.outlier_mask].copy()

        if impute:
            df = self.impute_missing(df)

        self.cleaned_data = df.reset_index(drop=True)
        return self.cleaned_data

    # ─── 标准化 ──────────────────────────────────────────────────

    def normalize(self, method: str = "minmax", fit: bool = True):
        """
        标准化特征和目标
        Args:
            method: "minmax" 或 "standard"
            fit: True 则 fit_transform, False 则 transform
        Returns:
            (X_scaled, y_scaled) 或 None
        """
        if method == "minmax":
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()

        if fit:
            X = self.cleaned_data[self.feature_columns]
            y = self.cleaned_data[self.target_columns] if self.target_columns else None
            X_scaled = self.scaler_X.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)

            if y is not None:
                y_scaled = self.scaler_y.fit_transform(y)
                y_scaled = pd.DataFrame(y_scaled, columns=self.target_columns)
                return X_scaled, y_scaled
            return X_scaled, None
        return None

    # ─── 数据划分 ────────────────────────────────────────────────

    def split(self, test_size: float = 0.2, random_state: int = 42):
        """划分训练/测试集"""
        X = self.cleaned_data[self.feature_columns]
        y = self.cleaned_data[self.target_columns] if self.target_columns else None
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    # ─── 统计信息 ────────────────────────────────────────────────

    def get_statistics(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """返回描述性统计 DataFrame"""
        if df is None:
            df = self.cleaned_data if self.cleaned_data is not None else self.raw_data
        if df is None:
            raise ValueError("没有可用数据，请先加载数据")
        return df.describe().T

    def get_missing_report(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """返回缺失值报告"""
        if df is None:
            df = self.raw_data
        if df is None:
            raise ValueError("没有可用数据")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        report = pd.DataFrame({
            "缺失数量": missing,
            "缺失比例(%)": missing_pct.round(2),
        })
        report = report[report["缺失数量"] > 0].sort_values("缺失数量", ascending=False)
        return report

    def get_correlation_matrix(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """返回特征相关性矩阵"""
        if df is None:
            df = self.cleaned_data if self.cleaned_data is not None else self.raw_data
        if df is None:
            raise ValueError("没有可用数据")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].corr()

    # ─── 导出 ────────────────────────────────────────────────────

    def export_csv(self, df: pd.DataFrame, path: str):
        """导出 DataFrame 为 CSV"""
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
