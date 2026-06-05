# -*- coding: utf-8 -*-
"""
01_数据管理 — UHPC 数据上传、清洗、可视化与 SHAP 分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_all_feature_names, get_all_target_names, get_feature_label
from core.data_manager import DataManager
from core.shap_analyzer import SHAPAnalyzer
from core.model_trainer import ModelTrainer

# ─── 页面配置 ────────────────────────────────────────────────────
st.set_page_config(page_title="数据管理 - UHPC-AutoDesign", page_icon="📊", layout="wide")

st.title("📊 数据管理")
st.caption("数据上传、清洗、可视化与 SHAP 特征分析")

# ─── 初始化 Session State ────────────────────────────────────────
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = get_all_feature_names()
if "target_columns" not in st.session_state:
    st.session_state.target_columns = get_all_target_names()

dm = st.session_state.data_manager

# ─── 标签页 ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📥 数据加载", "🧹 数据清洗", "📈 可视化", "🔍 SHAP分析"])

# ═══════════════════════════════════════════════════════════════════
# Tab 1: 数据加载
# ═══════════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_sample = st.columns([1, 1])

    with col_upload:
        st.subheader("上传 CSV 数据")
        uploaded_file = st.file_uploader(
            "选择 UHPC 配比数据 CSV 文件",
            type=["csv"],
            help="CSV 应包含组分、纤维、养护参数及性能目标列"
        )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                dm.load_dataframe(raw_df)
                st.session_state.feature_columns = dm.feature_columns
                st.session_state.target_columns = dm.target_columns
                st.success(f"✅ 成功加载 {len(raw_df)} 条数据")
                st.info(f"识别到 {len(dm.feature_columns)} 个特征, {len(dm.target_columns)} 个目标")
            except Exception as e:
                st.error(f"❌ 加载失败: {e}")

    with col_sample:
        st.subheader("或使用示例数据")
        if st.button("📂 加载 UHPC 文献基准数据", type="primary"):
            sample_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "literature.csv"
            )
            if os.path.exists(sample_path):
                dm.load_csv(sample_path)
                st.session_state.feature_columns = dm.feature_columns
                st.session_state.target_columns = dm.target_columns
                st.success(f"✅ 已加载 {len(dm.raw_data)} 条 UHPC 文献数据")
            else:
                st.error(f"示例数据文件不存在: {sample_path}")

    # 数据预览
    if dm.raw_data is not None:
        st.divider()
        st.subheader(f"📋 数据预览（共 {len(dm.raw_data)} 条）")
        st.dataframe(dm.raw_data.head(50), use_container_width=True)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("数据行数", len(dm.raw_data))
        with col_s2:
            st.metric("特征数", len(dm.feature_columns))
        with col_s3:
            st.metric("目标数", len(dm.target_columns))
        with col_s4:
            missing_cnt = dm.raw_data.isnull().sum().sum()
            st.metric("缺失值总数", missing_cnt)

# ═══════════════════════════════════════════════════════════════════
# Tab 2: 数据清洗
# ═══════════════════════════════════════════════════════════════════
with tab2:
    if dm.raw_data is None:
        st.warning("⚠️ 请先在「数据加载」中上传数据或加载示例数据")
    else:
        st.subheader("🧹 数据清洗")

        col_settings, col_report = st.columns([1, 2])

        with col_settings:
            contamination = st.slider(
                "Isolation Forest 异常值比例",
                min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                help="预期异常样本占比"
            )
            remove_outliers = st.checkbox("移除异常值", value=True)
            impute_missing = st.checkbox("MICE 多重插补缺失值", value=True)

            if st.button("🔧 执行清洗", type="primary", use_container_width=True):
                with st.spinner("清洗中..."):
                    dm.feature_columns = st.session_state.feature_columns
                    dm.target_columns = st.session_state.target_columns
                    cleaned = dm.clean(
                        remove_outliers=remove_outliers,
                        impute=impute_missing,
                        contamination=contamination
                    )
                    st.session_state.cleaned_data = cleaned
                    st.success(f"✅ 清洗完成！保留 {len(cleaned)} 条数据")

                    if remove_outliers:
                        outlier_count = (dm.outlier_mask == False).sum() if dm.outlier_mask is not None else 0
                        st.info(f"检测到 {outlier_count} 条异常值（已移除）")

        with col_report:
            if st.session_state.cleaned_data is not None:
                st.subheader("📊 清洗后数据统计")
                st.dataframe(dm.get_statistics(st.session_state.cleaned_data), use_container_width=True)

                # 缺失值报告
                missing_report = dm.get_missing_report(st.session_state.cleaned_data)
                if len(missing_report) > 0:
                    st.subheader("⚠️ 残留缺失值")
                    st.dataframe(missing_report, use_container_width=True)
                else:
                    st.success("✅ 无缺失值")

                # 导出按钮
                csv = st.session_state.cleaned_data.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 导出清洗后数据 CSV",
                    csv,
                    "uhpc_cleaned_data.csv",
                    "text/csv",
                )

# ═══════════════════════════════════════════════════════════════════
# Tab 3: 可视化
# ═══════════════════════════════════════════════════════════════════
with tab3:
    data_to_viz = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else dm.raw_data

    if data_to_viz is None:
        st.warning("⚠️ 请先加载并清洗数据")
    else:
        st.subheader("📈 数据可视化")

        viz_type = st.radio(
            "图表类型",
            ["箱线图", "相关性热力图", "散点图矩阵", "分布直方图"],
            horizontal=True
        )

        numeric_cols = data_to_viz.select_dtypes(include=[np.number]).columns.tolist()

        if viz_type == "箱线图":
            st.subheader("特征分布箱线图")
            selected_cols = st.multiselect(
                "选择列", numeric_cols,
                default=numeric_cols[:min(6, len(numeric_cols))]
            )
            if selected_cols:
                fig, ax = plt.subplots(figsize=(12, 5))
                data_to_viz[selected_cols].boxplot(ax=ax, rot=45)
                ax.set_title("特征箱线图")
                st.pyplot(fig)

        elif viz_type == "相关性热力图":
            st.subheader("特征相关性矩阵")
            corr = dm.get_correlation_matrix(data_to_viz)
            fig, ax = plt.subplots(figsize=(12, 9))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                        center=0, square=True, ax=ax, linewidths=0.5)
            ax.set_title("特征相关性热力图")
            st.pyplot(fig)

        elif viz_type == "散点图矩阵":
            st.subheader("多变量散点图矩阵")
            cols = st.multiselect(
                "选择 2-5 列用于散点图矩阵",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            if len(cols) >= 2:
                fig = sns.pairplot(data_to_viz[cols].sample(min(100, len(data_to_viz))))
                st.pyplot(fig.fig)

        elif viz_type == "分布直方图":
            st.subheader("单变量分布")
            col = st.selectbox("选择列", numeric_cols)
            if col:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].hist(data_to_viz[col].dropna(), bins=30, edgecolor="white", alpha=0.8)
                axes[0].set_title(f"{col} 分布直方图")
                axes[0].set_xlabel(col)
                axes[0].set_ylabel("频率")

                # Q-Q 参考线
                from scipy import stats
                stats.probplot(data_to_viz[col].dropna(), dist="norm", plot=axes[1])
                axes[1].set_title(f"{col} Q-Q 图")

                st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════
# Tab 4: SHAP 分析
# ═══════════════════════════════════════════════════════════════════
with tab4:
    data_to_analyze = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else dm.raw_data

    if data_to_analyze is None:
        st.warning("⚠️ 请先加载并清洗数据")
    else:
        st.subheader("🔍 SHAP 特征重要性分析")

        st.info("""
        **SHAP (SHapley Additive exPlanations)** 基于博弈论解释模型预测。
        选择目标变量后，将训练模型并计算各特征对预测的贡献度。
        """)

        col_shap1, col_shap2 = st.columns([1, 2])

        with col_shap1:
            target_col = st.selectbox(
                "选择预测目标",
                st.session_state.target_columns,
                help="选择要解释的目标变量"
            )

            model_type = st.selectbox(
                "代理模型",
                ["random_forest", "gradient_boosting", "xgboost"],
            )

            max_samples_shap = st.slider("SHAP 计算样本数", 50, 500, 200, 50,
                                         help="样本太多会减慢计算速度")

            if st.button("🚀 运行 SHAP 分析", type="primary", use_container_width=True):
                with st.spinner(f"训练 {model_type} 模型 + 计算 SHAP 值..."):
                    try:
                        # 准备数据
                        X = data_to_analyze[st.session_state.feature_columns].select_dtypes(
                            include=[np.number]
                        )
                        y = data_to_analyze[target_col]

                        # 训练模型
                        trainer = ModelTrainer(model_type=model_type)
                        trainer.train(X, y)
                        cv_results = trainer.cross_validate(X, y, cv=5)

                        st.session_state.shap_trainer = trainer
                        st.session_state.shap_cv = cv_results

                        # SHAP 分析
                        analyzer = SHAPAnalyzer(trainer.model, X)
                        analyzer.compute(max_samples=max_samples_shap)
                        st.session_state.shap_analyzer = analyzer
                        st.session_state.shap_importance = analyzer.get_importance_df()

                        st.success(f"✅ SHAP 分析完成！R²(CV) = {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")

                    except Exception as e:
                        st.error(f"❌ SHAP 分析失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        with col_shap2:
            if "shap_analyzer" in st.session_state and st.session_state.shap_analyzer is not None:
                analyzer = st.session_state.shap_analyzer

                st.subheader("📊 SHAP 特征重要性")
                st.dataframe(
                    st.session_state.shap_importance,
                    use_container_width=True,
                    hide_index=True,
                )

                # SHAP 摘要图
                try:
                    fig = analyzer.plot_summary()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"摘要图渲染失败: {e}")

                # 依赖图
                st.subheader("🔬 单特征依赖图")
                dep_feature = st.selectbox(
                    "选择特征查看 SHAP 依赖",
                    st.session_state.feature_columns,
                )
                try:
                    fig = analyzer.plot_dependence(dep_feature)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"依赖图渲染失败: {e}")

            # CV 结果
            if "shap_cv" in st.session_state and st.session_state.shap_cv is not None:
                st.divider()
                st.subheader("📋 模型交叉验证结果")
                cv = st.session_state.shap_cv
                col_cv1, col_cv2, col_cv3 = st.columns(3)
                col_cv1.metric("R² (均值)", f"{cv['r2_mean']:.4f}")
                col_cv2.metric("MAE (均值)", f"{cv['mae_mean']:.4f}")
                col_cv3.metric("RMSE (均值)", f"{cv['rmse_mean']:.4f}")
