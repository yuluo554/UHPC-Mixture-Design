# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign SHAP 可解释性分析模块
特征重要性分析 + 可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class SHAPAnalyzer:
    """SHAP 特征可解释性分析器"""

    def __init__(self, model, X: pd.DataFrame, feature_names: list = None):
        """
        Args:
            model: 训练好的 sklearn 兼容模型
            X: 特征 DataFrame
            feature_names: 特征中文名映射列表
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names or list(X.columns)
        self.explainer = None
        self.shap_values = None

    def compute(self, max_samples: int = 200):
        """
        计算 SHAP 值
        Args:
            max_samples: 用于计算 SHAP 的最大样本数（加速）
        """
        X_sample = self.X
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(max_samples, random_state=42)

        # 尝试使用 TreeExplainer（适用于树模型）
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)
        except Exception:
            # 回退到 KernelExplainer
            background = shap.kmeans(self.X, min(50, len(self.X)))
            self.explainer = shap.KernelExplainer(
                self.model.predict, background
            )
            self.shap_values = self.explainer.shap_values(X_sample, nsamples=100)

        return self

    # ─── 可视化方法 ──────────────────────────────────────────────

    def plot_summary(self, show: bool = False) -> plt.Figure:
        """SHAP 特征重要性摘要图（条形图）"""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, self.X,
            feature_names=self.feature_names,
            plot_type="bar", show=False
        )
        fig = plt.gcf()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_beeswarm(self, show: bool = False) -> plt.Figure:
        """SHAP 蜜蜂群图"""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, self.X,
            feature_names=self.feature_names,
            show=False
        )
        fig = plt.gcf()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_dependence(self, feature: str, show: bool = False) -> plt.Figure:
        """单个特征的 SHAP 依赖图"""
        plt.figure(figsize=(8, 5))
        idx = list(self.X.columns).index(feature) if feature in self.X.columns else 0
        shap.dependence_plot(
            idx, self.shap_values, self.X,
            feature_names=self.feature_names,
            show=False
        )
        fig = plt.gcf()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_waterfall(self, index: int = 0, show: bool = False) -> plt.Figure:
        """单个样本的 Waterfall 瀑布图"""
        plt.figure(figsize=(8, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[index],
                base_values=self.explainer.expected_value if hasattr(self.explainer, "expected_value") else 0,
                data=self.X.iloc[index].values,
                feature_names=self.feature_names,
            ),
            show=False
        )
        fig = plt.gcf()
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def get_importance_df(self) -> pd.DataFrame:
        """返回特征重要性 DataFrame（按 |SHAP| 均值排序）"""
        if isinstance(self.shap_values, list):
            vals = self.shap_values[0] if len(self.shap_values) > 0 else self.shap_values
        else:
            vals = self.shap_values

        importance = np.abs(vals).mean(axis=0)
        df = pd.DataFrame({
            "特征": self.feature_names,
            "SHAP重要性": importance,
        }).sort_values("SHAP重要性", ascending=False).reset_index(drop=True)
        return df
