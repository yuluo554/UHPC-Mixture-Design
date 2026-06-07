# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign 闭环反馈管理器
支持实验数据导入、预测对比、模型增量/全量更新、闭环日志
"""

import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from core.model_trainer import MultiTargetTrainer

# ─── 中文字体设置 ────────────────────────────────────────────────
_ZH_FONT = None
_known_zh_fonts = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
_avail_fonts = {f.name for f in fm.fontManager.ttflist}
for name in _known_zh_fonts:
    if name in _avail_fonts:
        _ZH_FONT = name
        break

if _ZH_FONT:
    plt.rcParams["font.sans-serif"] = [_ZH_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ─── 目标中文标签 ────────────────────────────────────────────────
TARGET_LABELS = {
    "compressive_strength_28d": "抗压强度 (MPa)",
    "flexural_strength_28d": "抗弯强度 (MPa)",
    "slump_flow": "坍落度 (mm)",
    "cost": "成本 (¥/m³)",
    "carbon_emission": "碳排放 (kg CO₂/m³)",
}


def _target_label(name: str) -> str:
    return TARGET_LABELS.get(name, name)


class FeedbackManager:
    """闭环反馈管理器

    职责：导入实验数据 → 模型预测 → 预测vs实测对比 → 模型更新 → 日志记录
    """

    def __init__(self):
        self.original_trainer: Optional[MultiTargetTrainer] = None
        self.updated_trainer: Optional[MultiTargetTrainer] = None
        self.experiment_data: Optional[pd.DataFrame] = None
        self.original_X: Optional[pd.DataFrame] = None
        self.original_Y: Optional[pd.DataFrame] = None
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self.feedback_log: list[dict] = []
        self._last_comparison: Optional[dict] = None

    # ═══════════════════════════════════════════════════════════════
    # 数据导入
    # ═══════════════════════════════════════════════════════════════

    def import_experiment_data(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        target_columns: list[str],
    ) -> pd.DataFrame:
        """导入实验数据并校验列完整性。

        Args:
            data: 实验数据 DataFrame
            feature_columns: 特征列名列表
            target_columns: 目标列名列表

        Returns:
            校验通过的实验数据（仅含特征+目标列）

        Raises:
            ValueError: 缺少必要的特征或目标列
        """
        all_needed = feature_columns + target_columns
        missing = [c for c in all_needed if c not in data.columns]
        if missing:
            raise ValueError(
                f"实验数据缺少以下列 ({len(missing)}/{len(all_needed)}): {missing}"
            )

        self.feature_names = list(feature_columns)
        self.target_names = list(target_columns)
        self.experiment_data = data[all_needed].copy()

        # 类型转换：确保数值列是 float
        for col in self.feature_names:
            if col in self.experiment_data.columns:
                self.experiment_data[col] = pd.to_numeric(
                    self.experiment_data[col], errors="coerce"
                )
        for col in self.target_names:
            if col in self.experiment_data.columns:
                self.experiment_data[col] = pd.to_numeric(
                    self.experiment_data[col], errors="coerce"
                )

        # 移除含 NaN 的行
        before = len(self.experiment_data)
        self.experiment_data = self.experiment_data.dropna()
        after = len(self.experiment_data)
        if after < before:
            import warnings
            warnings.warn(f"移除了 {before - after} 行含缺失值的实验数据")

        return self.experiment_data

    def get_experiment_X(self) -> pd.DataFrame:
        """获取实验数据的特征部分"""
        if self.experiment_data is None:
            raise RuntimeError("尚未导入实验数据")
        return self.experiment_data[self.feature_names]

    def get_experiment_Y(self) -> pd.DataFrame:
        """获取实验数据的目标部分"""
        if self.experiment_data is None:
            raise RuntimeError("尚未导入实验数据")
        return self.experiment_data[self.target_names]

    # ═══════════════════════════════════════════════════════════════
    # 预测
    # ═══════════════════════════════════════════════════════════════

    def predict_on_experiments(
        self, trainer: Optional[MultiTargetTrainer] = None
    ) -> pd.DataFrame:
        """使用指定模型对实验数据做预测。

        Args:
            trainer: 模型训练器，默认使用 original_trainer

        Returns:
            预测值 DataFrame，列名为各目标
        """
        if trainer is None:
            trainer = self.original_trainer
        if trainer is None:
            raise RuntimeError("没有可用的模型，请先训练或加载模型")
        if self.experiment_data is None:
            raise RuntimeError("尚未导入实验数据")

        X_exp = self.get_experiment_X()
        # 确保特征列顺序与模型训练时一致
        if trainer.feature_names is not None:
            X_exp = X_exp[trainer.feature_names]
        return trainer.predict_all(X_exp)

    # ═══════════════════════════════════════════════════════════════
    # 预测 vs 实测对比
    # ═══════════════════════════════════════════════════════════════

    def compare(
        self,
        measured: np.ndarray,
        predicted: np.ndarray,
        target_names: Optional[list[str]] = None,
    ) -> dict:
        """计算预测 vs 实测的对比指标。

        Args:
            measured: 实测值 (n_samples × n_targets)
            predicted: 预测值 (n_samples × n_targets)
            target_names: 目标名列表

        Returns:
            {
                "per_target": {target: {r2, mae, rmse, mape, max_error}},
                "overall": {r2_mean, mae_mean, rmse_mean, mape_mean},
                "target_names": [...],
            }
        """
        if target_names is None:
            target_names = self.target_names

        measured = np.asarray(measured, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)

        if measured.shape != predicted.shape:
            raise ValueError(
                f"实测和预测维度不匹配: {measured.shape} vs {predicted.shape}"
            )

        n_samples, n_targets = measured.shape
        per_target = {}

        for i, name in enumerate(target_names):
            y_true = measured[:, i]
            y_pred = predicted[:, i]

            # 过滤 NaN
            valid = np.isfinite(y_true) & np.isfinite(y_pred)
            if valid.sum() < 2:
                per_target[name] = {
                    "r2": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "max_error": np.nan,
                    "n_valid": int(valid.sum()),
                }
                continue

            y_true_v = y_true[valid]
            y_pred_v = y_pred[valid]

            r2 = r2_score(y_true_v, y_pred_v)
            mae = mean_absolute_error(y_true_v, y_pred_v)
            rmse = np.sqrt(mean_squared_error(y_true_v, y_pred_v))
            # MAPE: 避免除零
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.abs((y_true_v - y_pred_v) / np.where(y_true_v != 0, y_true_v, np.nan))
                mape = np.nanmean(ape) * 100 if np.any(np.isfinite(ape)) else np.nan
            max_error = np.max(np.abs(y_true_v - y_pred_v))

            per_target[name] = {
                "r2": round(float(r2), 4),
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "mape": round(float(mape), 2),
                "max_error": round(float(max_error), 4),
                "n_valid": int(valid.sum()),
            }

        # 汇总
        r2s = [v["r2"] for v in per_target.values() if not np.isnan(v["r2"])]
        maes = [v["mae"] for v in per_target.values() if not np.isnan(v["mae"])]
        rmses = [v["rmse"] for v in per_target.values() if not np.isnan(v["rmse"])]
        mapes = [v["mape"] for v in per_target.values() if not np.isnan(v["mape"])]

        overall = {
            "r2_mean": round(float(np.mean(r2s)), 4) if r2s else np.nan,
            "mae_mean": round(float(np.mean(maes)), 4) if maes else np.nan,
            "rmse_mean": round(float(np.mean(rmses)), 4) if rmses else np.nan,
            "mape_mean": round(float(np.mean(mapes)), 2) if mapes else np.nan,
        }

        self._last_comparison = {
            "per_target": per_target,
            "overall": overall,
            "target_names": target_names,
        }
        return self._last_comparison

    def get_comparison_df(self, comparison_results: Optional[dict] = None) -> pd.DataFrame:
        """将对比结果转为格式化 DataFrame"""
        if comparison_results is None:
            comparison_results = self._last_comparison
        if comparison_results is None:
            raise RuntimeError("没有对比结果，请先调用 compare()")

        rows = []
        for name in comparison_results["target_names"]:
            metrics = comparison_results["per_target"][name]
            rows.append({
                "目标": _target_label(name),
                "R²": metrics["r2"],
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "MAPE(%)": metrics["mape"],
                "Max Error": metrics["max_error"],
                "有效样本": metrics["n_valid"],
            })

        df = pd.DataFrame(rows)
        # 添加汇总行
        ov = comparison_results["overall"]
        summary = {
            "目标": "📊 平均",
            "R²": ov["r2_mean"],
            "MAE": ov["mae_mean"],
            "RMSE": ov["rmse_mean"],
            "MAPE(%)": ov["mape_mean"],
            "Max Error": "-",
            "有效样本": "-",
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        return df

    # ═══════════════════════════════════════════════════════════════
    # 模型更新
    # ═══════════════════════════════════════════════════════════════

    def incremental_update(self, model_type: str = "random_forest") -> MultiTargetTrainer:
        """增量更新：合并原始训练数据 + 实验数据后重新训练。

        Args:
            model_type: 模型类型

        Returns:
            更新后的 MultiTargetTrainer
        """
        if self.original_X is None or self.original_Y is None:
            raise RuntimeError("请先设置 original_X 和 original_Y（原始训练数据）")
        if self.experiment_data is None:
            raise RuntimeError("尚未导入实验数据")

        X_exp = self.get_experiment_X()
        Y_exp = self.get_experiment_Y()

        # 对齐列
        common_targets = [t for t in self.target_names if t in self.original_Y.columns and t in Y_exp.columns]
        if not common_targets:
            raise ValueError("原始数据和实验数据没有共同的目标列")

        # 合并
        X_all = pd.concat([self.original_X, X_exp], ignore_index=True)
        Y_all = pd.concat([self.original_Y[common_targets], Y_exp[common_targets]], ignore_index=True)

        # 重训练
        trainer = MultiTargetTrainer(model_type=model_type)
        trainer.train_all(X_all, Y_all)
        self.updated_trainer = trainer
        return trainer

    def full_retrain(self, model_type: str = "random_forest") -> MultiTargetTrainer:
        """全量重训练：使用全部可用数据从头训练。

        当前实现与增量更新相同（合并全部数据重训练），预留加权采样等扩展点。

        Args:
            model_type: 模型类型

        Returns:
            重训练后的 MultiTargetTrainer
        """
        return self.incremental_update(model_type=model_type)

    def evaluate_update(
        self,
        X_test: pd.DataFrame,
        Y_test: pd.DataFrame,
    ) -> dict:
        """评估模型更新前后的性能变化。

        Args:
            X_test: 测试集特征
            Y_test: 测试集目标

        Returns:
            {
                "before": {target: {r2, mae, rmse}},
                "after":  {target: {r2, mae, rmse}},
                "delta":  {target: {r2_delta, mae_delta, rmse_delta}},
            }
        """
        before_trainer = self.original_trainer
        after_trainer = self.updated_trainer

        if before_trainer is None:
            raise RuntimeError("没有原始模型")
        if after_trainer is None:
            raise RuntimeError("没有更新后的模型，请先执行更新")

        targets = [t for t in self.target_names if t in Y_test.columns]
        X_test_aligned = X_test
        if before_trainer.feature_names is not None:
            X_test_aligned = X_test[before_trainer.feature_names]

        pred_before = before_trainer.predict_all(X_test_aligned)
        pred_after = after_trainer.predict_all(X_test_aligned)

        def _eval(y_true, y_pred):
            return {
                "r2": round(float(r2_score(y_true, y_pred)), 4),
                "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            }

        result = {"before": {}, "after": {}, "delta": {}}
        for t in targets:
            b = _eval(Y_test[t].values, pred_before[t].values)
            a = _eval(Y_test[t].values, pred_after[t].values)
            result["before"][t] = b
            result["after"][t] = a
            result["delta"][t] = {
                "r2_delta": round(a["r2"] - b["r2"], 4),
                "mae_delta": round(a["mae"] - b["mae"], 4),
                "rmse_delta": round(a["rmse"] - b["rmse"], 4),
            }
        return result

    # ═══════════════════════════════════════════════════════════════
    # 闭环日志
    # ═══════════════════════════════════════════════════════════════

    def log_feedback(
        self,
        source: str,
        n_samples: int,
        update_type: str,
        metrics_before: Optional[dict] = None,
        metrics_after: Optional[dict] = None,
    ) -> dict:
        """记录一次闭环反馈。

        Args:
            source: 数据来源描述
            n_samples: 实验数据样本数
            update_type: 更新类型（增量更新 / 全量重训练）
            metrics_before: 更新前指标
            metrics_after: 更新后指标

        Returns:
            日志条目 dict
        """
        entry = {
            "cycle": len(self.feedback_log) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "n_samples": n_samples,
            "update_type": update_type,
            "r2_before": metrics_before.get("r2_mean") if metrics_before else None,
            "mae_before": metrics_before.get("mae_mean") if metrics_before else None,
            "r2_after": metrics_after.get("r2_mean") if metrics_after else None,
            "mae_after": metrics_after.get("mae_mean") if metrics_after else None,
        }
        self.feedback_log.append(entry)
        return entry

    def get_log_df(self) -> pd.DataFrame:
        """返回闭环日志 DataFrame"""
        if not self.feedback_log:
            return pd.DataFrame()
        df = pd.DataFrame(self.feedback_log)
        # 格式化
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ═══════════════════════════════════════════════════════════════
    # 可视化
    # ═══════════════════════════════════════════════════════════════

    def plot_pred_vs_measured(
        self,
        measured: np.ndarray,
        predicted: np.ndarray,
        target_name: str,
        figsize: tuple = (5, 4),
    ) -> plt.Figure:
        """绘制预测 vs 实测散点图（带 y=x 参考线）。

        Args:
            measured: 实测值
            predicted: 预测值
            target_name: 目标名称
            figsize: 图像尺寸

        Returns:
            matplotlib Figure
        """
        y_true = np.asarray(measured).ravel()
        y_pred = np.asarray(predicted).ravel()

        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidth=0.5, s=60)

        # y=x 参考线
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims[0] -= margin
        lims[1] += margin
        ax.plot(lims, lims, "r--", alpha=0.6, label="y=x (理想)")

        ax.set_xlabel(f"实测 {_target_label(target_name)}")
        ax.set_ylabel(f"预测 {_target_label(target_name)}")
        ax.set_title(f"{_target_label(target_name)} — 预测 vs 实测")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # 标注 R²
        if len(y_true) >= 2:
            r2 = r2_score(y_true, y_pred)
            ax.text(
                0.05, 0.95, f"R² = {r2:.3f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        fig.tight_layout()
        return fig

    def plot_residuals(
        self,
        measured: np.ndarray,
        predicted: np.ndarray,
        target_name: str,
        figsize: tuple = (5, 4),
    ) -> plt.Figure:
        """绘制残差分布直方图。

        Args:
            measured: 实测值
            predicted: 预测值
            target_name: 目标名称
            figsize: 图像尺寸

        Returns:
            matplotlib Figure
        """
        y_true = np.asarray(measured).ravel()
        y_pred = np.asarray(predicted).ravel()

        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        residuals = y_pred[valid] - y_true[valid]

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(residuals, bins=max(8, len(residuals) // 2), edgecolor="k",
                alpha=0.7, color="steelblue")
        ax.axvline(0, color="r", linestyle="--", alpha=0.7, label="零误差线")
        ax.axvline(residuals.mean(), color="orange", linestyle="-", alpha=0.8,
                   label=f"均值 = {residuals.mean():.3f}")

        ax.set_xlabel("残差 (预测 - 实测)")
        ax.set_ylabel("频数")
        ax.set_title(f"{_target_label(target_name)} — 残差分布")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig

    def plot_error_bar(
        self,
        comparison_results: Optional[dict] = None,
        figsize: tuple = (8, 4),
    ) -> plt.Figure:
        """绘制各目标误差对比柱状图。

        Args:
            comparison_results: compare() 返回的结果
            figsize: 图像尺寸

        Returns:
            matplotlib Figure
        """
        if comparison_results is None:
            comparison_results = self._last_comparison
        if comparison_results is None:
            raise RuntimeError("没有对比结果")

        targets = comparison_results["target_names"]
        labels = [_target_label(t) for t in targets]
        maes = [comparison_results["per_target"][t]["mae"] for t in targets]
        rmses = [comparison_results["per_target"][t]["rmse"] for t in targets]

        x = np.arange(len(targets))
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width / 2, maes, width, label="MAE", color="steelblue")
        bars2 = ax.bar(x + width / 2, rmses, width, label="RMSE", color="coral")

        ax.set_xlabel("优化目标")
        ax.set_ylabel("误差值")
        ax.set_title("各目标预测误差对比")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 标注数值
        for bar in bars1:
            h = bar.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════════

    def save_state(self, path: str):
        """保存反馈管理器状态"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "feedback_log": self.feedback_log,
            "experiment_data": self.experiment_data,
            "original_X": self.original_X,
            "original_Y": self.original_Y,
        }
        joblib.dump(state, path)

    def load_state(self, path: str) -> "FeedbackManager":
        """加载反馈管理器状态"""
        state = joblib.load(path)
        self.feature_names = state.get("feature_names", [])
        self.target_names = state.get("target_names", [])
        self.feedback_log = state.get("feedback_log", [])
        self.experiment_data = state.get("experiment_data")
        self.original_X = state.get("original_X")
        self.original_Y = state.get("original_Y")
        return self
