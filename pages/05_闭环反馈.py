# -*- coding: utf-8 -*-
"""
05_闭环反馈 — 实验数据导入、预测 vs 实测对比、模型增量/全量更新、闭环日志
"""

import sys
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

# 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    get_all_feature_names,
    get_all_target_names,
    get_feature_ranges,
    get_feature_label,
    get_target_direction,
    get_targets,
)
from core.model_trainer import ModelTrainer, MultiTargetTrainer
from core.feedback_manager import FeedbackManager, TARGET_LABELS, _target_label

st.set_page_config(
    page_title="闭环反馈 - UHPC-AutoDesign",
    page_icon="🔄",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
.metric-card.green {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}
.metric-card.orange {
    background: linear-gradient(135deg, #f12711 0%, #f5af19 100%);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
}
.metric-card .label {
    font-size: 0.85rem;
    opacity: 0.85;
    margin-top: 0.3rem;
}
.feedback-log-table {
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════════════════════════════════

if "feedback_manager" not in st.session_state:
    st.session_state.feedback_manager = FeedbackManager()

if "feedback_has_experiment" not in st.session_state:
    st.session_state.feedback_has_experiment = False

if "feedback_has_model" not in st.session_state:
    st.session_state.feedback_has_model = False

if "feedback_comparison" not in st.session_state:
    st.session_state.feedback_comparison = None

if "feedback_eval_result" not in st.session_state:
    st.session_state.feedback_eval_result = None

fm = st.session_state.feedback_manager

# ═══════════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════════

st.sidebar.header("📦 数据与模型")

# ── 数据源 ──
st.sidebar.subheader("训练数据来源")
data_source = st.sidebar.radio(
    "选择基础训练数据",
    ["📄 文献数据 (literature.csv)", "🧬 合成数据 (CTGAN)", "📊 当前清洗数据"],
    key="fb_data_source",
)

# 加载数据
FEATURES_ALL = get_all_feature_names()
TARGETS_ALL = get_all_target_names()

_avail_features = FEATURES_ALL
_avail_targets = TARGETS_ALL

if data_source == "📄 文献数据 (literature.csv)":
    lit_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "literature.csv")
    if os.path.exists(lit_path):
        _df = pd.read_csv(lit_path)
        _avail_features = [f for f in FEATURES_ALL if f in _df.columns]
        _avail_targets = [t for t in TARGETS_ALL if t in _df.columns]
        st.sidebar.caption(f"✅ literature.csv ({len(_df)} 条)")
    else:
        st.sidebar.warning("literature.csv 未找到")
elif data_source == "🧬 合成数据 (CTGAN)":
    if "synthetic_data" in st.session_state and st.session_state.synthetic_data is not None:
        _df = st.session_state.synthetic_data
        _avail_features = [f for f in FEATURES_ALL if f in _df.columns]
        _avail_targets = [t for t in TARGETS_ALL if t in _df.columns]
        st.sidebar.caption(f"✅ 合成数据 ({len(_df)} 条)")
    else:
        st.sidebar.warning("请先在「虚拟生成」页面生成合成数据")
elif data_source == "📊 当前清洗数据":
    if "cleaned_data" in st.session_state and st.session_state.cleaned_data is not None:
        _df = st.session_state.cleaned_data
        _avail_features = [f for f in FEATURES_ALL if f in _df.columns]
        _avail_targets = [t for t in TARGETS_ALL if t in _df.columns]
        st.sidebar.caption(f"✅ 清洗数据 ({len(_df)} 条)")
    else:
        st.sidebar.warning("请先在「数据管理」页面加载并清洗数据")

# ── 模型选择 ──
st.sidebar.subheader("🤖 基础模型")
model_type = st.sidebar.selectbox(
    "模型类型",
    ["random_forest", "gradient_boosting"],
    key="fb_model_type",
    format_func=lambda x: {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}[x],
)

if st.sidebar.button("🛠️ 训练基础模型", use_container_width=True):
    if "cleaned_data" not in st.session_state or st.session_state.cleaned_data is None:
        # 尝试直接加载文献数据
        lit_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "literature.csv")
        if os.path.exists(lit_path):
            _df = pd.read_csv(lit_path)
            _avail_features = [f for f in FEATURES_ALL if f in _df.columns]
            _avail_targets = [t for t in TARGETS_ALL if t in _df.columns]
        else:
            st.sidebar.error("无可用训练数据")
            _df = None

        if _df is not None and len(_df) > 0:
            X_train = _df[_avail_features].select_dtypes(include=[np.number])
            Y_train = _df[_avail_targets]
            mt = MultiTargetTrainer(model_type=model_type)
            with st.spinner("训练基础模型中..."):
                mt.train_all(X_train, Y_train)
            fm.original_trainer = mt
            fm.original_X = X_train.copy()
            fm.original_Y = Y_train.copy()
            fm.feature_names = _avail_features
            fm.target_names = _avail_targets
            st.session_state.feedback_has_model = True
            st.sidebar.success(f"✅ 模型已训练 ({len(_avail_targets)} 目标)")
    else:
        _df = st.session_state.cleaned_data
        X_train = _df[_avail_features].select_dtypes(include=[np.number])
        Y_train = _df[_avail_targets]
        mt = MultiTargetTrainer(model_type=model_type)
        with st.spinner("训练基础模型中..."):
            mt.train_all(X_train, Y_train)
        fm.original_trainer = mt
        fm.original_X = X_train.copy()
        fm.original_Y = Y_train.copy()
        fm.feature_names = _avail_features
        fm.target_names = _avail_targets
        st.session_state.feedback_has_model = True
        st.sidebar.success(f"✅ 模型已训练 ({len(_avail_targets)} 目标)")

# 显示模型状态
if st.session_state.feedback_has_model:
    st.sidebar.info(f"📌 基础模型: {model_type}\n📌 原始模型: {len(fm.original_X) if fm.original_X is not None else 0} 条训练数据")
    if fm.updated_trainer is not None:
        st.sidebar.success("🔄 已有更新后的模型")

st.sidebar.divider()
st.sidebar.caption("🔄 闭环反馈 v0.1.0")

# ═══════════════════════════════════════════════════════════════════
# 主页
# ═══════════════════════════════════════════════════════════════════

st.title("🔄 闭环实验反馈")
st.caption("导入实验结果 → 预测对比 → 模型增量更新 → 闭环日志追踪")

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 实验数据导入",
    "📊 预测对比",
    "🔄 模型更新",
    "📋 闭环日志",
])

# ═══════════════════════════════════════════════════════════════════
# Tab 1: 实验数据导入
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("📥 导入实验数据")

    col_csv, col_manual = st.columns(2)

    with col_csv:
        st.markdown("#### 📂 CSV 上传")
        uploaded_file = st.file_uploader(
            "上传实验数据 CSV",
            type=["csv"],
            key="fb_upload_csv",
            help="CSV 需包含与训练数据相同的特征列和目标列",
        )

        if uploaded_file is not None:
            try:
                exp_df = pd.read_csv(uploaded_file)
                st.success(f"✅ 已读取 {len(exp_df)} 行 × {len(exp_df.columns)} 列")

                # 自动检测列匹配
                matched_features = [c for c in FEATURES_ALL if c in exp_df.columns]
                matched_targets = [c for c in TARGETS_ALL if c in exp_df.columns]

                st.markdown(f"**匹配特征:** {len(matched_features)}/{len(FEATURES_ALL)} 列")
                st.markdown(f"**匹配目标:** {len(matched_targets)}/{len(TARGETS_ALL)} 列")

                if len(matched_features) < 2 or len(matched_targets) < 1:
                    st.warning("⚠️ 匹配列太少，请确认 CSV 列名与参数模板一致")
                else:
                    if st.button("📥 确认导入实验数据", use_container_width=True, key="fb_confirm_csv"):
                        try:
                            fm.import_experiment_data(exp_df, matched_features, matched_targets)
                            st.session_state.feedback_has_experiment = True
                            st.session_state.feedback_comparison = None  # 重置对比结果
                            st.success(f"🎉 成功导入 {len(fm.experiment_data)} 条实验数据")
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))

            except Exception as e:
                st.error(f"读取 CSV 失败: {e}")

    with col_manual:
        st.markdown("#### ✏️ 手动录入")
        st.caption("逐条输入实验配比与实测结果")

        with st.form("fb_manual_entry"):
            st.markdown("**组成参数**")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                cement = st.number_input("Cement (kg/m³)", 600, 1000, 800, step=10)
                silica_fume = st.number_input("Silica Fume (kg/m³)", 50, 250, 150, step=5)
            with mc2:
                quartz_powder = st.number_input("Quartz Powder (kg/m³)", 0, 400, 200, step=10)
                quartz_sand = st.number_input("Quartz Sand (kg/m³)", 800, 1200, 1000, step=10)
            with mc3:
                superplasticizer = st.number_input("Superplasticizer (kg/m³)", 5, 40, 20, step=1)
                water = st.number_input("Water (kg/m³)", 140, 220, 180, step=5)

            st.markdown("**纤维参数**")
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                fiber_volume = st.number_input("Fiber Volume (%)", 0.5, 3.0, 1.5, step=0.1)
            with fc2:
                fiber_diameter = st.number_input("Fiber Diameter (mm)", 0.12, 0.6, 0.2, step=0.01)
            with fc3:
                fiber_length = st.number_input("Fiber Length (mm)", 6, 20, 13, step=1)
            with fc4:
                fiber_aspect_ratio = st.number_input("Aspect Ratio", 40, 100, 65, step=5)

            st.markdown("**养护参数**")
            cc1, cc2 = st.columns(2)
            with cc1:
                curing_temp = st.selectbox("Curing Temp (°C)", [20, 60, 90, 250], index=1)
            with cc2:
                curing_time = st.number_input("Curing Time (h)", 24, 168, 48, step=12)

            st.markdown("**实测结果**")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                comp_strength = st.number_input("抗压强度 (MPa)", 100.0, 300.0, 180.0, step=5.0)
                flex_strength = st.number_input("抗弯强度 (MPa)", 10.0, 80.0, 35.0, step=1.0)
            with tc2:
                slump = st.number_input("坍落度 (mm)", 150, 300, 220, step=5)
            with tc3:
                cost = st.number_input("成本 (¥/m³)", 1000.0, 8000.0, 3000.0, step=100.0)
                carbon = st.number_input("碳排放 (kg/m³)", 200.0, 1500.0, 700.0, step=50.0)

            submitted = st.form_submit_button("✅ 添加此条实验数据", use_container_width=True)

            if submitted:
                new_row = {
                    "cement": cement, "silica_fume": silica_fume,
                    "quartz_powder": quartz_powder, "quartz_sand": quartz_sand,
                    "superplasticizer": superplasticizer, "water": water,
                    "fiber_volume": fiber_volume, "fiber_diameter": fiber_diameter,
                    "fiber_length": fiber_length, "fiber_aspect_ratio": fiber_aspect_ratio,
                    "curing_temp": curing_temp, "curing_time": curing_time,
                    "compressive_strength_28d": comp_strength,
                    "flexural_strength_28d": flex_strength,
                    "slump_flow": slump,
                    "cost": cost,
                    "carbon_emission": carbon,
                }
                new_df = pd.DataFrame([new_row])

                # 合并到已有实验数据
                if fm.experiment_data is not None:
                    merged = pd.concat([fm.experiment_data, new_df], ignore_index=True)
                else:
                    merged = new_df

                try:
                    fm.import_experiment_data(merged, FEATURES_ALL, TARGETS_ALL)
                    st.session_state.feedback_has_experiment = True
                    st.session_state.feedback_comparison = None
                    st.success(f"🎉 已添加，当前共 {len(fm.experiment_data)} 条实验数据")
                except ValueError as e:
                    st.error(str(e))

    # ── 实验数据预览 ──
    if st.session_state.feedback_has_experiment and fm.experiment_data is not None:
        st.divider()
        st.subheader("📋 实验数据概览")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("实验样本数", len(fm.experiment_data))
        with col_b:
            st.metric("特征列", len(fm.feature_names))
        with col_c:
            st.metric("目标列", len(fm.target_names))
        with col_d:
            missing_pct = fm.experiment_data.isnull().sum().sum() / (len(fm.experiment_data) * len(fm.experiment_data.columns)) * 100
            st.metric("缺失率", f"{missing_pct:.1f}%")

        st.dataframe(fm.experiment_data.head(10), use_container_width=True)

        # 导出
        csv_bytes = fm.experiment_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 导出实验数据 CSV",
            csv_bytes,
            f"experiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
        )

# ═══════════════════════════════════════════════════════════════════
# Tab 2: 预测对比
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("📊 预测 vs 实测对比")

    if not st.session_state.feedback_has_experiment:
        st.info("👈 请先在「实验数据导入」标签页导入实验数据")
    elif not st.session_state.feedback_has_model:
        st.info("👈 请先在侧边栏训练基础模型")
    else:
        # 模型选择
        model_choice = st.radio(
            "选择用于预测的模型",
            ["🔵 原始模型 (original)", "🟢 更新后模型 (updated)"],
            horizontal=True,
            key="fb_pred_model_choice",
        )
        use_trainer = fm.original_trainer if "原始" in model_choice else fm.updated_trainer
        if "更新" in model_choice and fm.updated_trainer is None:
            st.warning("尚未执行模型更新，请使用原始模型或先去「模型更新」标签页")
            use_trainer = fm.original_trainer

        if st.button("🔮 执行预测", use_container_width=True, key="fb_run_pred"):
            with st.spinner("预测中..."):
                try:
                    preds = fm.predict_on_experiments(use_trainer)
                    measured = fm.get_experiment_Y().values
                    comp = fm.compare(measured, preds.values, fm.target_names)
                    st.session_state.feedback_comparison = comp
                    st.session_state.feedback_last_pred = preds
                except Exception as e:
                    st.error(f"预测失败: {e}")

        # 展示对比结果
        if st.session_state.feedback_comparison is not None:
            comp = st.session_state.feedback_comparison

            st.divider()
            st.markdown("### 📈 误差指标")

            # 指标卡片
            ov = comp["overall"]
            card_cols = st.columns(4)
            with card_cols[0]:
                r2_val = ov.get("r2_mean", np.nan)
                color_class = "green" if not np.isnan(r2_val) and r2_val > 0.7 else ("orange" if not np.isnan(r2_val) and r2_val > 0.4 else "")
                st.markdown(
                    f"""<div class="metric-card {color_class}">
                    <div class="value">{r2_val:.3f}</div>
                    <div class="label">平均 R²</div></div>""",
                    unsafe_allow_html=True,
                )
            with card_cols[1]:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="value">{ov.get('mae_mean', 0):.1f}</div>
                    <div class="label">平均 MAE</div></div>""",
                    unsafe_allow_html=True,
                )
            with card_cols[2]:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="value">{ov.get('rmse_mean', 0):.1f}</div>
                    <div class="label">平均 RMSE</div></div>""",
                    unsafe_allow_html=True,
                )
            with card_cols[3]:
                mape_val = ov.get("mape_mean", np.nan)
                color_class = "green" if not np.isnan(mape_val) and mape_val < 15 else ("orange" if not np.isnan(mape_val) and mape_val < 30 else "")
                st.markdown(
                    f"""<div class="metric-card {color_class}">
                    <div class="value">{mape_val:.1f}%</div>
                    <div class="label">平均 MAPE</div></div>""",
                    unsafe_allow_html=True,
                )

            # 对比详情表
            st.markdown("#### 📋 逐目标指标")
            comp_df = fm.get_comparison_df(comp)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # 散点图
            st.markdown("#### 🎯 预测 vs 实测散点图")
            preds = st.session_state.get("feedback_last_pred")
            if preds is not None:
                measured_arr = fm.get_experiment_Y().values
                preds_arr = preds.values
                n_targets = len(fm.target_names)
                n_cols = min(3, n_targets)
                n_rows = (n_targets + n_cols - 1) // n_cols

                for row in range(n_rows):
                    cols = st.columns(n_cols)
                    for ci in range(n_cols):
                        idx = row * n_cols + ci
                        if idx < n_targets:
                            with cols[ci]:
                                fig = fm.plot_pred_vs_measured(
                                    measured_arr[:, idx],
                                    preds_arr[:, idx],
                                    fm.target_names[idx],
                                    figsize=(4.5, 3.8),
                                )
                                st.pyplot(fig)
                                plt.close(fig)

            # 残差图
            st.markdown("#### 📉 残差分布")
            show_residuals = st.checkbox("显示残差分布直方图", value=False, key="fb_show_residuals")
            if show_residuals and preds is not None:
                measured_arr = fm.get_experiment_Y().values
                preds_arr = preds.values
                for row in range(n_rows):
                    cols = st.columns(n_cols)
                    for ci in range(n_cols):
                        idx = row * n_cols + ci
                        if idx < n_targets:
                            with cols[ci]:
                                fig = fm.plot_residuals(
                                    measured_arr[:, idx],
                                    preds_arr[:, idx],
                                    fm.target_names[idx],
                                    figsize=(4.5, 3.8),
                                )
                                st.pyplot(fig)
                                plt.close(fig)

            # 误差柱状图
            st.markdown("#### 📊 误差对比柱状图")
            fig_bar = fm.plot_error_bar(comp, figsize=(9, 4.5))
            st.pyplot(fig_bar)
            plt.close(fig_bar)

            # 导出对比结果
            csv_comp = comp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 导出对比指标 CSV",
                csv_comp,
                f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
            )

# ═══════════════════════════════════════════════════════════════════
# Tab 3: 模型更新
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🔄 模型更新")

    if not st.session_state.feedback_has_model:
        st.info("👈 请先在侧边栏训练基础模型")
    elif not st.session_state.feedback_has_experiment:
        st.info("👈 请先在「实验数据导入」标签页导入实验数据")
    else:
        # 更新策略
        st.markdown("### ⚙️ 更新设置")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            update_strategy = st.radio(
                "更新策略",
                ["📈 增量更新 (追加实验数据后重训练)", "🔄 全量重训练 (合并所有数据从头训练)"],
                key="fb_update_strategy",
                help="增量更新：原始数据 + 实验数据 → 重训练\n全量重训练：同增量，预留加权采样扩展",
            )
        with col_s2:
            update_model_type = st.selectbox(
                "更新用模型类型",
                ["random_forest", "gradient_boosting"],
                key="fb_update_model_type",
                format_func=lambda x: {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}[x],
            )

        # 数据概览
        st.markdown("#### 📊 数据概览")
        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            st.metric("原始训练样本", len(fm.original_X) if fm.original_X is not None else 0)
        with cd2:
            st.metric("实验样本", len(fm.experiment_data) if fm.experiment_data is not None else 0)
        with cd3:
            total = (len(fm.original_X) if fm.original_X is not None else 0) + \
                    (len(fm.experiment_data) if fm.experiment_data is not None else 0)
            st.metric("合并后总样本", total)

        # 测试集比例
        test_size = st.slider("Holdout 测试集比例", 0.1, 0.4, 0.2, 0.05, key="fb_test_size",
                              help="从原始数据中留出用于评估更新前后性能的测试集")

        # 执行更新
        st.divider()
        if st.button("🚀 执行模型更新", use_container_width=True, type="primary", key="fb_execute_update"):
            # 准备测试集（从原始数据中分割）
            if fm.original_X is not None and fm.original_Y is not None and len(fm.original_X) > 5:
                common_targets = [t for t in fm.target_names if t in fm.original_Y.columns]
                X_base = fm.original_X
                Y_base = fm.original_Y[common_targets]
                X_train_base, X_test, Y_train_base, Y_test = train_test_split(
                    X_base, Y_base, test_size=test_size, random_state=42
                )
            else:
                st.error("原始数据不足，无法划分测试集")
                st.stop()

            # 临时设置 original 为分割后的训练集以执行更新
            orig_X_backup = fm.original_X
            orig_Y_backup = fm.original_Y
            fm.original_X = X_train_base
            fm.original_Y = Y_train_base

            # 记录更新前性能
            metrics_before = None
            try:
                if fm.original_trainer is not None:
                    pred_before = fm.original_trainer.predict_all(X_test)
                    from sklearn.metrics import r2_score, mean_absolute_error
                    r2s, maes = [], []
                    for t in common_targets:
                        if t in Y_test.columns and t in pred_before.columns:
                            r2s.append(r2_score(Y_test[t], pred_before[t]))
                            maes.append(mean_absolute_error(Y_test[t], pred_before[t]))
                    metrics_before = {"r2_mean": np.mean(r2s) if r2s else None,
                                      "mae_mean": np.mean(maes) if maes else None}
            except Exception:
                pass

            # 执行更新
            with st.spinner(f"执行{update_strategy.split('(')[0].strip()}中..."):
                try:
                    if "增量" in update_strategy:
                        fm.incremental_update(model_type=update_model_type)
                        _update_type = "增量更新"
                    else:
                        fm.full_retrain(model_type=update_model_type)
                        _update_type = "全量重训练"

                    # 记录更新后性能
                    pred_after = fm.updated_trainer.predict_all(X_test)
                    r2s_a, maes_a = [], []
                    for t in common_targets:
                        if t in Y_test.columns and t in pred_after.columns:
                            r2s_a.append(r2_score(Y_test[t], pred_after[t]))
                            maes_a.append(mean_absolute_error(Y_test[t], pred_after[t]))
                    metrics_after = {"r2_mean": np.mean(r2s_a) if r2s_a else None,
                                     "mae_mean": np.mean(maes_a) if maes_a else None}

                    # 记录日志
                    fm.log_feedback(
                        source=data_source,
                        n_samples=len(fm.experiment_data),
                        update_type=_update_type,
                        metrics_before=metrics_before,
                        metrics_after=metrics_after,
                    )

                    # 恢复原始数据引用
                    fm.original_X = orig_X_backup
                    fm.original_Y = orig_Y_backup

                    st.success(f"✅ {_update_type}完成！")
                    st.balloons()

                    # 展示前后对比
                    st.markdown("### 📊 更新前后对比（测试集）")
                    cb1, cb2, cb3, cb4 = st.columns(4)
                    with cb1:
                        delta_r2 = (metrics_after["r2_mean"] - metrics_before["r2_mean"]) if metrics_before and metrics_after else 0
                        st.metric("R² (更新前)", f"{metrics_before['r2_mean']:.4f}" if metrics_before else "N/A",
                                  delta=f"{delta_r2:+.4f}" if metrics_before else None)
                    with cb2:
                        delta_mae = (metrics_after["mae_mean"] - metrics_before["mae_mean"]) if metrics_before and metrics_after else 0
                        st.metric("MAE (更新前)", f"{metrics_before['mae_mean']:.2f}" if metrics_before else "N/A",
                                  delta=f"{delta_mae:+.2f}" if metrics_before else None, delta_color="inverse")
                    with cb3:
                        st.metric("R² (更新后)", f"{metrics_after['r2_mean']:.4f}" if metrics_after else "N/A")
                    with cb4:
                        st.metric("MAE (更新后)", f"{metrics_after['mae_mean']:.2f}" if metrics_after else "N/A")

                    # 详细 before/after 对比表
                    if fm.updated_trainer is not None and fm.original_trainer is not None:
                        eval_result = fm.evaluate_update(X_test, Y_test)
                        st.session_state.feedback_eval_result = eval_result

                        st.markdown("#### 逐目标性能变化")
                        eval_rows = []
                        for t in common_targets:
                            if t in eval_result["before"] and t in eval_result["after"]:
                                b = eval_result["before"][t]
                                a = eval_result["after"][t]
                                d = eval_result["delta"][t]
                                eval_rows.append({
                                    "目标": _target_label(t),
                                    "R² 前": b["r2"], "R² 后": a["r2"], "ΔR²": d["r2_delta"],
                                    "MAE 前": b["mae"], "MAE 后": a["mae"], "ΔMAE": d["mae_delta"],
                                    "RMSE 前": b["rmse"], "RMSE 后": a["rmse"], "ΔRMSE": d["rmse_delta"],
                                })
                        if eval_rows:
                            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

                except Exception as e:
                    # 恢复
                    fm.original_X = orig_X_backup
                    fm.original_Y = orig_Y_backup
                    st.error(f"更新失败: {e}")

        # 显示上次更新结果
        if st.session_state.feedback_eval_result is not None:
            st.divider()
            st.markdown("### 📌 最近一次更新结果")
            eval_result = st.session_state.feedback_eval_result
            eval_rows = []
            for t in eval_result["before"]:
                b = eval_result["before"][t]
                a = eval_result["after"][t]
                d = eval_result["delta"][t]
                eval_rows.append({
                    "目标": _target_label(t),
                    "R² 前": b["r2"], "R² 后": a["r2"], "ΔR²": d["r2_delta"],
                    "MAE 前": b["mae"], "MAE 后": a["mae"], "ΔMAE": d["mae_delta"],
                })
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# Tab 4: 闭环日志
# ═══════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("📋 闭环反馈日志")

    log_df = fm.get_log_df()

    if log_df.empty:
        st.info("还没有反馈记录。完成一次模型更新后，日志会自动生成。")
    else:
        # 统计摘要
        st.markdown("### 📊 累计统计")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("反馈周期数", len(log_df))
        with sc2:
            st.metric("累计实验样本", int(log_df["n_samples"].sum()) if "n_samples" in log_df.columns else 0)
        with sc3:
            if "r2_before" in log_df.columns and log_df["r2_before"].notna().any():
                latest = log_df.iloc[-1]
                delta_r2 = (latest["r2_after"] - latest["r2_before"]) if pd.notna(latest.get("r2_before")) and pd.notna(latest.get("r2_after")) else 0
                st.metric("最近 ΔR²", f"{delta_r2:+.4f}" if isinstance(delta_r2, (int, float)) else "N/A")
            else:
                st.metric("最近 ΔR²", "N/A")
        with sc4:
            if "update_type" in log_df.columns:
                type_counts = log_df["update_type"].value_counts()
                st.metric("更新类型分布", ", ".join([f"{k}:{v}" for k, v in type_counts.items()]))
            else:
                st.metric("更新类型分布", "N/A")

        # 日志表
        st.markdown("### 📜 详细日志")
        display_df = log_df.copy()
        if "timestamp" in display_df.columns:
            display_df["时间"] = display_df["timestamp"].dt.strftime("%m-%d %H:%M")
            display_df = display_df.drop(columns=["timestamp"])
        # 重命名列
        col_rename = {
            "cycle": "周期", "source": "数据来源", "n_samples": "实验样本数",
            "update_type": "更新类型", "r2_before": "R² 更新前", "mae_before": "MAE 更新前",
            "r2_after": "R² 更新后", "mae_after": "MAE 更新后",
        }
        display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # 导出日志
        csv_log = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 导出闭环日志 CSV",
            csv_log,
            f"feedback_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
        )

        # 清空日志
        if st.button("🗑️ 清空日志", key="fb_clear_log", type="secondary"):
            fm.feedback_log = []
            st.session_state.feedback_eval_result = None
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════════════════════

st.divider()
st.caption("💡 提示：闭环反馈通过实验数据回流持续改进模型预测精度。建议每次实验后及时导入数据。")
