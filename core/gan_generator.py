# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign CTGAN 虚拟数据生成模块
基于 SDV/CTGAN 从有限配比数据生成高质量合成数据
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


# ─── CTGAN 生成器 ─────────────────────────────────────────────────

class CTGANGenerator:
    """
    CTGAN 合成数据生成器

    使用 SDV 的 CTGANSynthesizer 从 UHPC 配比数据学习联合分布，
    生成统计特性高度相似的合成配比数据。CTGAN 对混合类型数据
    （数值 + 分类）特别有效 — 这正是 UHPC 参数体系的特点。

    Examples:
        >>> gen = CTGANGenerator(epochs=500, batch_size=100)
        >>> gen.fit(real_data, categorical_columns=['curing_temp'])
        >>> synthetic = gen.generate(1000)
    """

    def __init__(self, epochs: int = 500, batch_size: int = 100,
                 cuda: bool = False, verbose: bool = False, **kwargs):
        """
        Args:
            epochs: 训练轮数（默认 500, 数据越多可设越大）
            batch_size: 批次大小（默认 100，受限于 GPU 显存）
            cuda: 是否使用 CUDA 加速
            verbose: 是否打印训练日志
            **kwargs: 传递给 CTGANSynthesizer 的额外参数
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.cuda = cuda
        self.verbose = verbose
        self.extra_kwargs = kwargs
        self.synthesizer = None
        self.metadata = None
        self._fitted_data = None
        self._loss_history = []

    def fit(self, data: pd.DataFrame, categorical_columns: list = None) -> "CTGANGenerator":
        """
        训练 CTGAN 模型

        Args:
            data: 真实数据 DataFrame
            categorical_columns: 分类型列名列表（会自动推断）

        Returns:
            self
        """
        if data is None or len(data) == 0:
            raise ValueError("训练数据为空")

        self._fitted_data = data.copy()

        # 创建元数据
        self.metadata = Metadata.detect_from_dataframe(data)

        # 手动标记分类型列
        if categorical_columns:
            for col in categorical_columns:
                if col in data.columns:
                    self.metadata.update_column(
                        column_name=col,
                        sdtype="categorical"
                    )

        # 创建并训练合成器
        synthesizer_kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "enable_gpu": self.cuda,  # SDV 1.x renamed cuda → enable_gpu
            "verbose": self.verbose,
            **self.extra_kwargs,
        }

        self.synthesizer = CTGANSynthesizer(
            self.metadata,
            **synthesizer_kwargs,
        )

        self.synthesizer.fit(data)

        # 提取损失历史（如果可用）
        if hasattr(self.synthesizer, '_model'):
            model = self.synthesizer._model
            if hasattr(model, 'loss_values'):
                self._loss_history = model.loss_values

        return self

    def generate(self, num_rows: int = 1000) -> pd.DataFrame:
        """
        生成合成数据

        Args:
            num_rows: 生成的行数

        Returns:
            合成数据 DataFrame
        """
        if self.synthesizer is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        synthetic = self.synthesizer.sample(num_rows=num_rows)
        return synthetic

    def conditional_generate(self, conditions: dict,
                             num_rows: int = 100) -> pd.DataFrame | None:
        """
        条件定向生成 — 在指定约束条件下生成数据

        Args:
            conditions: 条件字典，如 {"curing_temp": 90, "cement": (700, 900)}
            num_rows: 每个条件下生成的行数

        Returns:
            满足条件的合成数据
        """
        if self.synthesizer is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        try:
            # 将条件转换为 SDV 格式
            sdv_conditions = []
            for col, val in conditions.items():
                if isinstance(val, tuple):
                    # 范围条件
                    sdv_conditions.append({col: val})
                else:
                    sdv_conditions.append({col: val})

            # 逐条件生成后合并
            results = []
            remaining = num_rows
            while remaining > 0:
                batch = min(remaining, 500)
                try:
                    synth = self.synthesizer.sample_from_conditions(
                        conditions=sdv_conditions,
                        num_rows=batch,
                    )
                    results.append(synth)
                except Exception:
                    # 回退到无条件生成 + 后过滤
                    synth = self.generate(batch * 3)
                    for col, val in conditions.items():
                        if isinstance(val, tuple):
                            synth = synth[
                                (synth[col] >= val[0]) & (synth[col] <= val[1])
                            ]
                        else:
                            synth = synth[synth[col] == val]
                    results.append(synth.head(batch))
                remaining -= batch

            if results:
                result = pd.concat(results, ignore_index=True)
                return result.head(num_rows)
            return None

        except Exception as e:
            # 回退：无条件生成 + 过滤
            import sys
            print(f"条件生成失败 ({e})，回退到无条件生成 + 过滤", file=sys.stderr)
            synth = self.generate(num_rows * 5)
            for col, val in conditions.items():
                if isinstance(val, tuple):
                    synth = synth[
                        (synth[col] >= val[0]) & (synth[col] <= val[1])
                    ]
                else:
                    synth = synth[synth[col] == val]
            return synth.head(num_rows)

    def get_loss_history(self) -> list:
        """返回训练损失历史"""
        return self._loss_history

    def get_metadata_summary(self) -> dict:
        """返回训练数据的元数据摘要"""
        if self._fitted_data is None:
            return {}
        return {
            "n_rows": len(self._fitted_data),
            "n_columns": len(self._fitted_data.columns),
            "columns": list(self._fitted_data.columns),
            "dtypes": self._fitted_data.dtypes.astype(str).to_dict(),
        }

    def save(self, path: str):
        """保存模型到文件"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "synthesizer": self.synthesizer,
            "metadata": self.metadata,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "_fitted_data": self._fitted_data,
            "_loss_history": self._loss_history,
        }, path)

    def load(self, path: str):
        """从文件加载模型"""
        data = joblib.load(path)
        self.synthesizer = data["synthesizer"]
        self.metadata = data.get("metadata")
        self.epochs = data.get("epochs", self.epochs)
        self.batch_size = data.get("batch_size", self.batch_size)
        self._fitted_data = data.get("_fitted_data")
        self._loss_history = data.get("_loss_history", [])
        return self


# ─── TVAE 生成器（备选） ──────────────────────────────────────────

class TVAEGenerator:
    """
    TVAE 合成数据生成器（备选方案）

    TVAE 使用变分自编码器生成数据，在某些场景下
    比 CTGAN 更适合连续型占主导的数据。
    """

    def __init__(self, epochs: int = 500, batch_size: int = 100,
                 cuda: bool = False, verbose: bool = False, **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.cuda = cuda
        self.verbose = verbose
        self.extra_kwargs = kwargs
        self.synthesizer = None
        self.metadata = None
        self._fitted_data = None

    def fit(self, data: pd.DataFrame,
            categorical_columns: list = None) -> "TVAEGenerator":
        if data is None or len(data) == 0:
            raise ValueError("训练数据为空")

        self._fitted_data = data.copy()
        self.metadata = Metadata.detect_from_dataframe(data)

        if categorical_columns:
            for col in categorical_columns:
                if col in data.columns:
                    self.metadata.update_column(
                        column_name=col,
                        sdtype="categorical"
                    )

        synthesizer_kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "enable_gpu": self.cuda,
            "verbose": self.verbose,
            **self.extra_kwargs,
        }

        self.synthesizer = TVAESynthesizer(
            self.metadata,
            **synthesizer_kwargs,
        )
        self.synthesizer.fit(data)
        return self

    def generate(self, num_rows: int = 1000) -> pd.DataFrame:
        if self.synthesizer is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self.synthesizer.sample(num_rows=num_rows)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        joblib.dump({
            "synthesizer": self.synthesizer,
            "metadata": self.metadata,
            "_fitted_data": self._fitted_data,
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.synthesizer = data["synthesizer"]
        self.metadata = data.get("metadata")
        self._fitted_data = data.get("_fitted_data")
        return self


# ─── TSTR 验证器 ──────────────────────────────────────────────────

class TSTRValidator:
    """
    Train-on-Synthetic, Test-on-Real 验证器

    核心思想：如果合成数据质量足够好，用合成数据训练的模型
    在真实测试集上的表现应该接近用真实数据训练的模型。

    指标包括：
    - TSTR/TRTR R² 比值（越接近 1.0 越好）
    - 每列 Wasserstein 距离（越小越好）
    - 每列 Jensen-Shannon 散度
    - 列间相关性差异
    - 基本统计量对比
    """

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                 feature_columns: list, target_columns: list = None):
        """
        Args:
            real_data: 真实数据
            synthetic_data: 合成数据
            feature_columns: 特征列名列表
            target_columns: 目标列名列表（用于 TSTR 评估，可选）
        """
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.feature_columns = feature_columns
        self.target_columns = target_columns or []
        self._results = {}

    def evaluate(self, model_type: str = "random_forest",
                 test_size: float = 0.3) -> dict:
        """
        全面评估合成数据质量

        Args:
            model_type: "random_forest" 或 "gradient_boosting"
            test_size: TSTR 中真实数据测试集比例

        Returns:
            包含所有评估指标的字典
        """
        results = {}

        # 1. 统计量对比
        results["statistics"] = self._compare_statistics()

        # 2. 分布距离
        results["distributions"] = self._compute_distribution_distances()

        # 3. 相关性对比
        results["correlation"] = self._compare_correlation()

        # 4. TSTR — Train on Synthetic, Test on Real
        if self.target_columns:
            results["tstr"] = self._tstr_evaluate(
                model_type=model_type,
                test_size=test_size,
            )

        # 5. 列覆盖率检查
        results["coverage"] = self._check_coverage()

        self._results = results
        return results

    def _compare_statistics(self) -> dict:
        """对比真实与合成数据的统计量"""
        stats = {}
        numeric_cols = (
            self.real_data[self.feature_columns]
            .select_dtypes(include=[np.number]).columns
        )

        for col in numeric_cols:
            if col in self.synthetic_data.columns:
                real_vals = self.real_data[col].dropna()
                synth_vals = self.synthetic_data[col].dropna()
                if len(real_vals) == 0 or len(synth_vals) == 0:
                    continue
                stats[col] = {
                    "real_mean": float(real_vals.mean()),
                    "synth_mean": float(synth_vals.mean()),
                    "real_std": float(real_vals.std()),
                    "synth_std": float(synth_vals.std()),
                    "mean_diff_pct": float(
                        abs(real_vals.mean() - synth_vals.mean())
                        / max(abs(real_vals.mean()), 1e-8) * 100
                    ),
                }

        return stats

    def _compute_distribution_distances(self) -> dict:
        """计算每列的 Wasserstein 距离和 JS 散度"""
        distances = {}
        numeric_cols = (
            self.real_data[self.feature_columns]
            .select_dtypes(include=[np.number]).columns
        )

        for col in numeric_cols:
            if col in self.synthetic_data.columns:
                real_vals = self.real_data[col].dropna()
                synth_vals = self.synthetic_data[col].dropna()
                if len(real_vals) == 0 or len(synth_vals) == 0:
                    continue

                # Wasserstein 距离（标准化到 [0,1] 后计算）
                real_min, real_max = real_vals.min(), real_vals.max()
                real_range = real_max - real_min
                if real_range > 0:
                    real_norm = (real_vals - real_min) / real_range
                    synth_norm = (synth_vals - real_min) / real_range
                    synth_norm = synth_norm.clip(0, 1)
                else:
                    real_norm = real_vals
                    synth_norm = synth_vals

                w_dist = wasserstein_distance(real_norm, synth_norm)

                # Jensen-Shannon 散度（需要直方图近似）
                try:
                    bins = min(50, max(10, len(real_vals) // 3))
                    real_hist, bin_edges = np.histogram(
                        real_norm, bins=bins, density=True
                    )
                    synth_hist, _ = np.histogram(
                        synth_norm.clip(0, 1), bins=bin_edges, density=True
                    )
                    # 平滑处理，避免零值
                    real_hist = real_hist + 1e-10
                    synth_hist = synth_hist + 1e-10
                    real_hist = real_hist / real_hist.sum()
                    synth_hist = synth_hist / synth_hist.sum()
                    js_div = float(jensenshannon(real_hist, synth_hist) ** 2)
                except Exception:
                    js_div = None

                distances[col] = {
                    "wasserstein": round(float(w_dist), 6),
                    "js_divergence": round(js_div, 6) if js_div is not None else None,
                }

        return distances

    def _compare_correlation(self) -> dict:
        """对比真实与合成数据的相关性矩阵"""
        numeric_cols = (
            self.real_data[self.feature_columns]
            .select_dtypes(include=[np.number]).columns
        )
        common = [c for c in numeric_cols if c in self.synthetic_data.columns]

        if len(common) < 2:
            return {"error": "不足2列数值列，无法计算相关性"}

        real_corr = self.real_data[common].corr()
        synth_corr = self.synthetic_data[common].corr()

        # 相关性矩阵差异 Frobenius 范数
        diff_matrix = real_corr - synth_corr
        frob_norm = float(np.linalg.norm(diff_matrix.values, ord="fro"))
        max_diff = float(diff_matrix.abs().max().max())
        mean_abs_diff = float(diff_matrix.abs().mean().mean())

        return {
            "frobenius_norm": round(frob_norm, 4),
            "max_abs_diff": round(max_diff, 4),
            "mean_abs_diff": round(mean_abs_diff, 4),
            "real_corr": real_corr,
            "synth_corr": synth_corr,
        }

    def _tstr_evaluate(self, model_type: str = "random_forest",
                       test_size: float = 0.3) -> dict:
        """
        TSTR (Train on Synthetic, Test on Real) 核心评估

        1. 用真实数据训练 → 在真实测试集上评估（TRTR baseline）
        2. 用合成数据训练 → 在真实测试集上评估（TSTR）
        3. 比较两者 R² / MAE 比值
        """
        from sklearn.model_selection import train_test_split

        common_targets = [
            t for t in self.target_columns
            if t in self.real_data.columns and t in self.synthetic_data.columns
        ]
        common_features = [
            f for f in self.feature_columns
            if f in self.real_data.columns and f in self.synthetic_data.columns
        ]

        if not common_targets or not common_features:
            return {"error": "特征或目标列不匹配"}

        # 真实数据划分
        X_real = self.real_data[common_features].select_dtypes(include=[np.number])
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_real,
            self.real_data[common_targets],
            test_size=test_size,
            random_state=42,
        )

        # 合成数据作为训练集
        X_synth = self.synthetic_data[common_features].select_dtypes(
            include=[np.number]
        )
        y_synth = self.synthetic_data[common_targets]

        tstr_results = {}

        for target in common_targets:
            yt_train = y_train_r[target]
            yt_test = y_test_r[target]
            ys_train = y_synth[target]

            # TRTR: Train on Real, Test on Real
            model_real = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            model_real.fit(X_train_r, yt_train)
            y_pred_real = model_real.predict(X_test_r)
            r2_real = r2_score(yt_test, y_pred_real)
            mae_real = mean_absolute_error(yt_test, y_pred_real)

            # TSTR: Train on Synthetic, Test on Real
            model_synth = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            model_synth.fit(X_synth, ys_train)
            y_pred_synth = model_synth.predict(X_test_r)
            r2_synth = r2_score(yt_test, y_pred_synth)
            mae_synth = mean_absolute_error(yt_test, y_pred_synth)

            # 对 R² 比值做鲁棒性处理
            if r2_real > 0:
                raw_ratio = r2_synth / r2_real
                # 当合成数据训练的模型表现很差时，ratio 会很大或很负
                r2_ratio_clipped = max(-2.0, min(raw_ratio, 2.0))
            else:
                raw_ratio = None
                r2_ratio_clipped = None

            mae_ratio = mae_synth / mae_real if mae_real > 0 else None
            mae_ratio_clipped = max(0.1, min(mae_ratio, 5.0)) if mae_ratio else None

            tstr_results[target] = {
                "trtr_r2": round(float(r2_real), 4),
                "tstr_r2": round(float(r2_synth), 4),
                "r2_ratio": round(float(r2_ratio_clipped), 4) if r2_ratio_clipped is not None else None,
                "r2_ratio_raw": round(float(raw_ratio), 4) if raw_ratio is not None else None,
                "trtr_mae": round(float(mae_real), 4),
                "tstr_mae": round(float(mae_synth), 4),
                "mae_ratio": round(float(mae_ratio_clipped), 4) if mae_ratio_clipped is not None else None,
                "mae_ratio_raw": round(float(mae_ratio), 4) if mae_ratio else None,
            }

        # 汇总
        avg_r2_ratio = np.mean([
            v["r2_ratio"] for v in tstr_results.values()
            if v["r2_ratio"] is not None
        ])
        avg_mae_ratio = np.mean([
            v["mae_ratio"] for v in tstr_results.values()
            if v["mae_ratio"] is not None
        ])

        tstr_results["_summary"] = {
            "avg_r2_ratio": round(float(avg_r2_ratio), 4),
            "avg_mae_ratio": round(float(avg_mae_ratio), 4),
            "quality_rating": self._rate_tstr(avg_r2_ratio),
        }

        return tstr_results

    @staticmethod
    def _rate_tstr(r2_ratio: float) -> str:
        """根据 TSTR R² 比值给质量评级"""
        if r2_ratio is None:
            return "无法评估（测试集 R²=0）"
        if r2_ratio >= 0.9:
            return "优秀 — 合成数据几乎可以替代真实数据"
        elif r2_ratio >= 0.7:
            return "良好 — 合成数据有较好的代表性"
        elif r2_ratio >= 0.5:
            return "一般 — 合成数据可用但信息有损失"
        elif r2_ratio >= 0.0:
            return "需改进 — 合成数据质量较低，建议增加 epochs 或真实数据量"
        else:
            return "较差 — 建议增加真实数据量或调整训练参数"

    def _check_coverage(self) -> dict:
        """检查合成数据的列覆盖和范围覆盖"""
        coverage = {}
        for col in self.feature_columns:
            if col in self.real_data.columns and col in self.synthetic_data.columns:
                real_min = float(self.real_data[col].min())
                real_max = float(self.real_data[col].max())
                synth_min = float(self.synthetic_data[col].min())
                synth_max = float(self.synthetic_data[col].max())

                coverage[col] = {
                    "real_range": [real_min, real_max],
                    "synth_range": [synth_min, synth_max],
                    "min_ok": synth_min >= real_min * 0.9,
                    "max_ok": synth_max <= real_max * 1.1,
                    "range_ratio": round(
                        (synth_max - synth_min) / max(real_max - real_min, 1e-8), 4
                    ),
                }

        return coverage

    def get_report(self) -> str:
        """生成人类可读的评估报告"""
        if not self._results:
            return "尚未执行评估，请先调用 evaluate()"

        r = self._results
        lines = [
            "=" * 60,
            "  TSTR 合成数据质量评估报告",
            "=" * 60,
            "",
            f"  真实数据: {len(self.real_data)} 条",
            f"  合成数据: {len(self.synthetic_data)} 条",
            f"  特征数: {len(self.feature_columns)}",
        ]

        if "tstr" in r and "_summary" in r["tstr"]:
            s = r["tstr"]["_summary"]
            lines += [
                "",
                "── TSTR 性能 ──",
                f"  平均 R² 比值 (TSTR/TRTR): {s['avg_r2_ratio']:.4f}",
                f"  平均 MAE 比值 (TSTR/TRTR): {s['avg_mae_ratio']:.4f}",
                f"  质量评级: {s['quality_rating']}",
            ]

        if "distributions" in r:
            avg_w = np.mean([
                v["wasserstein"] for v in r["distributions"].values()
            ])
            lines += [
                "",
                "── 分布相似度 ──",
                f"  平均 Wasserstein 距离: {avg_w:.4f}",
            ]

        if "correlation" in r and "error" not in r["correlation"]:
            c = r["correlation"]
            lines += [
                "",
                "── 相关性保持 ──",
                f"  Frobenius 范数: {c['frobenius_norm']:.4f}",
                f"  最大偏差: {c['max_abs_diff']:.4f}",
                f"  平均偏差: {c['mean_abs_diff']:.4f}",
            ]

        lines += ["", "=" * 60]
        return "\n".join(lines)

    def plot_distribution_comparison(self, column: str):
        """
        绘制单列的真实 vs 合成分布对比图

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if column not in self.real_data.columns:
            raise ValueError(f"列 '{column}' 不存在")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        real_vals = self.real_data[column].dropna()
        synth_vals = self.synthetic_data[column].dropna()

        # 直方图重叠
        bins = min(40, max(10, len(real_vals) // 2))
        axes[0].hist(real_vals, bins=bins, alpha=0.6, label="真实",
                     color="#3498db", edgecolor="white", density=True)
        axes[0].hist(synth_vals, bins=bins, alpha=0.6, label="合成",
                     color="#e74c3c", edgecolor="white", density=True)
        axes[0].set_title(f"{column} 分布对比")
        axes[0].legend()
        axes[0].set_ylabel("密度")

        # KDE 曲线
        try:
            from scipy.stats import gaussian_kde
            kde_real = gaussian_kde(real_vals)
            kde_synth = gaussian_kde(synth_vals)
            x_range = np.linspace(
                min(real_vals.min(), synth_vals.min()),
                max(real_vals.max(), synth_vals.max()),
                200
            )
            axes[1].plot(x_range, kde_real(x_range), label="真实",
                        color="#3498db", linewidth=2)
            axes[1].plot(x_range, kde_synth(x_range), label="合成",
                        color="#e74c3c", linewidth=2)
            axes[1].fill_between(x_range, kde_real(x_range), alpha=0.1,
                                 color="#3498db")
            axes[1].fill_between(x_range, kde_synth(x_range), alpha=0.1,
                                 color="#e74c3c")
            axes[1].set_title(f"{column} KDE 曲线")
            axes[1].legend()
        except Exception:
            axes[1].text(0.5, 0.5, "KDE 不可用",
                        ha="center", va="center", transform=axes[1].transAxes)

        # 箱线图
        box_data = [real_vals, synth_vals]
        bp = axes[2].boxplot(box_data, tick_labels=["真实", "合成"],
                             patch_artist=True)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][1].set_facecolor("#e74c3c")
        axes[2].set_title(f"{column} 箱线图")

        plt.tight_layout()
        return fig

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        返回评估摘要 DataFrame（供 UI 表格展示）
        """
        if not self._results:
            raise RuntimeError("尚未执行评估，请先调用 evaluate()")

        rows = []
        r = self._results

        # 列级指标
        for col in self.feature_columns:
            row = {"列名": col}

            if "distributions" in r and col in r["distributions"]:
                d = r["distributions"][col]
                row["Wasserstein 距离"] = d.get("wasserstein", "—")
                row["JS 散度"] = d.get("js_divergence", "—")

            if "statistics" in r and col in r["statistics"]:
                s = r["statistics"][col]
                row["均值差异%"] = round(s.get("mean_diff_pct", 0), 2)

            rows.append(row)

        return pd.DataFrame(rows)


# ─── 辅助函数 ─────────────────────────────────────────────────────

def auto_generate_with_validation(
    real_data: pd.DataFrame,
    feature_columns: list,
    target_columns: list = None,
    categorical_columns: list = None,
    gen_type: str = "ctgan",
    epochs: int = 500,
    batch_size: int = 100,
    num_rows: int = 1000,
    run_validation: bool = True,
) -> dict:
    """
    一键生成 + 验证：自动训练生成器、生成数据、运行 TSTR 验证

    Args:
        real_data: 真实数据
        feature_columns: 特征列
        target_columns: 目标列（可选，用于 TSTR）
        categorical_columns: 分类型列
        gen_type: "ctgan" 或 "tvae"
        epochs: 训练轮数
        batch_size: 批次大小
        num_rows: 生成行数
        run_validation: 是否运行 TSTR 验证

    Returns:
        {"generator": ..., "synthetic_data": DataFrame, "validation": dict}
    """
    # 创建生成器
    if gen_type == "tvae":
        generator = TVAEGenerator(epochs=epochs, batch_size=batch_size)
    else:
        generator = CTGANGenerator(epochs=epochs, batch_size=batch_size)

    # 训练
    generator.fit(real_data, categorical_columns=categorical_columns)

    # 生成
    synthetic = generator.generate(num_rows=num_rows)

    result = {
        "generator": generator,
        "synthetic_data": synthetic,
    }

    # 验证
    if run_validation and target_columns:
        validator = TSTRValidator(
            real_data=real_data,
            synthetic_data=synthetic,
            feature_columns=feature_columns,
            target_columns=target_columns,
        )
        result["validation"] = validator.evaluate()
        result["validator"] = validator

    return result
