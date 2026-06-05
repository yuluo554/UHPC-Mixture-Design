# -*- coding: utf-8 -*-
"""
03_高通量筛选 — 贝叶斯优化高通量筛选
基于 BoTorch + GP 的多目标贝叶斯优化，含 EHVI/qNEHVI 采集函数
"""

import sys
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    get_all_feature_names,
    get_all_target_names,
    get_feature_ranges,
    get_categorical_features,
    get_target_direction,
    get_feature_label,
    get_targets,
    get_constraints,
)
from core.bayesian_optimizer import (
    BayesianOptimizer,
    ParetoCollector,
    create_bayesian_optimizer,
    _normalize_for_minimization,
    compute_reference_point,
)

st.set_page_config(
    page_title="高通量筛选 - UHPC-AutoDesign",
    page_icon="🔬",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════
# CSS 调整
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
.metric-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.metric-card.orange { background: linear-gradient(135deg, #f12711 0%, #f5af19 100%); }
.metric-card .value { font-size: 2rem; font-weight: 700; }
.metric-card .label { font-size: 0.85rem; opacity: 0.85; margin-top: 0.3rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
# 侧边栏：数据源配置
# ═══════════════════════════════════════════════════════════════════

st.sidebar.header("📦 数据源")

data_source = st.sidebar.radio(
    "选择数据来源",
    ["文献基准数据 (literature.csv)", "CTGAN 合成数据", "两者合并"],
    index=0,
    key="bayes_data_source",
)

# ── 加载数据 ────────────────────────────────────────────────────

DATA_LOADED = False
_X_data: Optional[pd.DataFrame] = None
_Y_data: Optional[pd.DataFrame] = None
_available_features: list = []
_available_targets: list = []


def _load_literature_data():
    """加载文献数据"""
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "literature.csv",
    )
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def _get_dataset():
    """根据侧边栏选择返回 (X_df, Y_df, feature_cols, target_cols)"""
    features = [f for f in get_all_feature_names()]
    targets = [t for t in get_all_target_names()]

    df = None
    if "两者合并" in data_source:
        dfs = []
        lit = _load_literature_data()
        if lit is not None:
            dfs.append(lit)
        if "synthetic_data" in st.session_state and st.session_state.synthetic_data is not None:
            dfs.append(st.session_state.synthetic_data)
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
    elif "CTGAN" in data_source:
        if "synthetic_data" in st.session_state and st.session_state.synthetic_data is not None:
            df = st.session_state.synthetic_data.copy()
        else:
            st.sidebar.warning("⚠️ 没有合成数据，请先在「虚拟生成」页面生成")
            return None, None, [], []
    else:
        df = _load_literature_data()

    if df is None:
        return None, None, [], []

    # 匹配特征和目标
    avail_f = [f for f in features if f in df.columns]
    avail_t = [t for t in targets if t in df.columns]
    all_cols = avail_f + avail_t
    df_clean = df[all_cols].dropna().copy()

    if len(df_clean) < 5:
        st.sidebar.warning("⚠️ 可用数据太少")
        return None, None, [], []

    X_df = df_clean[avail_f]
    Y_df = df_clean[avail_t]
    return X_df, Y_df, avail_f, avail_t


X_df, Y_df, _available_features, _available_targets = _get_dataset()

if X_df is not None and len(X_df) > 0:
    DATA_LOADED = True
    st.sidebar.success(
        f"✅ 已加载 {len(X_df)} 条数据 | "
        f"{len(_available_features)} 特征 × {len(_available_targets)} 目标"
    )
else:
    st.sidebar.error("❌ 没有可用数据")

# ═══════════════════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════════════════

st.title("🔬 贝叶斯优化高通量筛选")
st.caption(
    "基于 BoTorch + 高斯过程 (GP) 的多目标贝叶斯优化，"
    "使用 EHVI / qNEHVI 采集函数快速探索 Pareto 前沿"
)

if not DATA_LOADED:
    st.warning("👈 请先在侧边栏选择数据来源，或前往「数据管理」页面上传数据。")
    st.stop()

# ── 标签页 ──────────────────────────────────────────────────────

tab_setup, tab_monitor, tab_pareto, tab_export = st.tabs(
    ["🎯 优化设置", "📈 优化监控", "📊 Pareto 前沿", "📋 结果导出"]
)

# ═══════════════════════════════════════════════════════════════════
# Tab 1: 优化设置
# ═══════════════════════════════════════════════════════════════════

with tab_setup:
    st.subheader("优化目标与算法配置")

    # 已有优化结果时显示成功横幅
    if "bayes_results" in st.session_state:
        results_cache = st.session_state.bayes_results
        n_pts = len(results_cache.get("pareto_X", []))
        hv_val = results_cache.get("hypervolume_history", pd.DataFrame())
        hv_final = hv_val["hypervolume"].iloc[-1] if len(hv_val) > 0 else 0
        st.success(
            f"✅ 已有优化结果：**{n_pts}** 个 Pareto 最优解 | "
            f"最终超体积 **{hv_final:,.0f}** | "
            f"共 **{results_cache.get('n_iter_completed', 0)}** 次迭代\n\n"
            f"👉 切换到「📈 优化监控」和「📊 Pareto 前沿」查看详细结果"
        )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 🎯 选择优化目标")

        # 目标选择，显示中文标签和方向
        target_info = get_targets()
        target_options = []
        for t in target_info:
            if t["name"] in _available_targets:
                direction_emoji = {"maximize": "↑", "minimize": "↓", "target": "⊙"}
                emoji = direction_emoji.get(t["direction"], "→")
                target_options.append(f"{emoji} {t['label']} ({t['name']})")

        # 默认选择前 4 个目标
        default_indices = list(range(min(4, len(target_options))))

        selected_target_labels = st.multiselect(
            "勾选要优化的目标（至少 2 个）",
            options=target_options,
            default=[target_options[i] for i in default_indices],
            key="bayes_targets",
            help="↑ = 最大化 | ↓ = 最小化 | ⊙ = 趋近目标值",
        )

        # 解析选中的目标名
        selected_targets = []
        selected_dirs = []
        for label in selected_target_labels:
            name = label.split("(")[-1].rstrip(")")
            selected_targets.append(name)
            d = get_target_direction(name)
            selected_dirs.append(d if d else "minimize")

        st.caption(f"已选 **{len(selected_targets)}** 个目标: {', '.join(selected_targets)}")

    with col_right:
        st.markdown("#### ⚙️ 物理约束")

        constraints_list = get_constraints()
        apply_constraints = st.checkbox(
            "启用物理约束过滤",
            value=True,
            key="bayes_apply_constraints",
            help="水胶比 ∈ [0.14, 0.24]",
        )

        for c in constraints_list:
            if c.get("range"):
                lo, hi = c["range"]
                st.caption(f"• {c['description']}: [{lo}, {hi}]")
            else:
                st.caption(f"• {c['description']}: ±{c.get('tolerance', 0.001)}")

    st.divider()

    # ── 算法参数 ──────────────────────────────────────────────

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        surrogate_type = st.selectbox(
            "代理模型",
            options=["gp", "rf", "ensemble"],
            index=0,
            format_func=lambda x: {
                "gp": "🔵 高斯过程 (GP)",
                "rf": "🌲 随机森林 (RF)",
                "ensemble": "🧬 GP+RF 集成",
            }[x],
            key="bayes_surrogate",
            help="GP = 精确+不确定性 | RF = 鲁棒+快速 | Ensemble = 综合两者",
        )

    with col2:
        acq_type = st.selectbox(
            "采集函数",
            options=["qnehvi", "ehvi"],
            index=0,
            format_func=lambda x: {
                "qnehvi": "📊 qNEHVI (噪声版)",
                "ehvi": "📈 EHVI (无噪声版)",
            }[x],
            key="bayes_acq",
            help="qNEHVI = 考虑预测噪声，更鲁棒 | EHVI = 标准期望超体积改进",
        )

    with col3:
        n_iter = st.slider(
            "优化迭代次数",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="bayes_n_iter",
            help="迭代越多，Pareto 前沿越精细，但耗时越长",
        )

    with col4:
        random_seed = st.number_input(
            "随机种子",
            value=42,
            min_value=0,
            max_value=9999,
            key="bayes_seed",
        )

    # ── 数据预览 ──────────────────────────────────────────────

    with st.expander("📋 数据预览（前 5 行）"):
        preview_df = X_df[_available_features].head(5).copy()
        for t in selected_targets:
            if t in Y_df.columns:
                preview_df[t] = Y_df[t].head(5).values
        st.dataframe(preview_df, use_container_width=True)

    # ── 启动优化按钮 ──────────────────────────────────────────

    st.divider()

    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        start_btn = st.button(
            "🚀 开始贝叶斯优化",
            type="primary",
            use_container_width=True,
            disabled=len(selected_targets) < 2,
        )

    if len(selected_targets) < 2:
        st.warning("⚠️ 请至少选择 2 个目标才能进行多目标优化")

    if start_btn and len(selected_targets) >= 2:
        with st.status("🔬 贝叶斯优化进行中...", expanded=True) as status:
            # ── 准备数据 ────────────────────────────────────
            st.write("📦 准备训练数据...")
            X = X_df[_available_features].values.astype(np.float64)
            Y = Y_df[selected_targets].values.astype(np.float64)

            # 过滤无效值
            valid_mask = np.isfinite(Y).all(axis=1)
            X = X[valid_mask]
            Y = Y[valid_mask]

            st.write(f"   训练数据: {X.shape[0]} 样本 × {X.shape[1]} 特征 × {Y.shape[1]} 目标")

            # ── 创建优化器 ──────────────────────────────────
            st.write(f"🔧 创建优化器: surrogate={surrogate_type}, acq={acq_type}")

            opt = BayesianOptimizer(
                feature_names=_available_features,
                target_names=selected_targets,
                directions=selected_dirs,
                surrogate_type=surrogate_type,
                acq_type=acq_type,
                random_state=random_seed,
            )

            # ── 运行优化 ────────────────────────────────────
            progress_bar = st.progress(0, text="优化中...")
            status_text = st.empty()

            # 回调函数更新 UI
            def progress_callback(entry):
                pct = entry["iteration"] / n_iter
                progress_bar.progress(
                    min(pct, 1.0),
                    text=f"迭代 {entry['iteration']}/{n_iter} | "
                    f"HV={entry['hypervolume']:.4f} | "
                    f"前沿点={entry['n_frontier']}",
                )
                status_text.caption(
                    f"当前候选预测: {[round(y, 2) for y in entry['predicted_y']]}"
                )

            t0 = time.time()
            results = opt.optimize(
                X_init=X,
                Y_init=Y,
                n_iter=n_iter,
                verbose=False,
                callback=progress_callback,
            )
            elapsed = time.time() - t0

            progress_bar.progress(1.0, text=f"✅ 优化完成 ({elapsed:.1f}s)")
            status.update(label=f"✅ 优化完成（{elapsed:.1f}秒）", state="complete")

            # ── 存储到 session_state ────────────────────────
            st.session_state.bayes_optimizer = opt
            st.session_state.bayes_results = results
            st.session_state.bayes_selected_targets = selected_targets
            st.session_state.bayes_selected_dirs = selected_dirs

            # 摘要指标（精简版，完整版在 rerun 后显示）
            pX, pY = results["pareto_X"], results["pareto_Y"]
            n_frontier = len(pX)
            hv_final = results["hypervolume_history"]["hypervolume"].iloc[-1]
            hv_init = results["hypervolume_history"]["hypervolume"].iloc[0]
            hv_improvement = (hv_final - hv_init) / abs(hv_init) * 100 if abs(hv_init) > 1e-6 else 0

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(
                f'<div class="metric-card"><div class="value">{n_frontier}</div>'
                f'<div class="label">Pareto 最优解</div></div>',
                unsafe_allow_html=True,
            )
            mc2.markdown(
                f'<div class="metric-card green"><div class="value">{hv_final:,.0f}</div>'
                f'<div class="label">最终超体积</div></div>',
                unsafe_allow_html=True,
            )
            mc3.markdown(
                f'<div class="metric-card orange"><div class="value">{hv_improvement:+.1f}%</div>'
                f'<div class="label">超体积改进率</div></div>',
                unsafe_allow_html=True,
            )
            mc4.markdown(
                f'<div class="metric-card"><div class="value">{elapsed:.1f}s</div>'
                f'<div class="label">总耗时 ({n_iter} 次迭代)</div></div>',
                unsafe_allow_html=True,
            )

            # 触发页面刷新，使其他标签页也能看到结果
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# Tab 2: 优化监控
# ═══════════════════════════════════════════════════════════════════

with tab_monitor:
    st.subheader("📈 超体积收敛曲线")

    if "bayes_results" not in st.session_state:
        st.info("👈 请先在「优化设置」标签页点击「🚀 开始贝叶斯优化」按钮启动优化，完成后自动显示结果")
    else:
        results = st.session_state.bayes_results
        targets = st.session_state.bayes_selected_targets
        hist_df = results["hypervolume_history"]

        # ── 收敛曲线 ──────────────────────────────────────
        fig_hv = go.Figure()

        fig_hv.add_trace(
            go.Scatter(
                x=hist_df["iteration"],
                y=hist_df["hypervolume"],
                mode="lines+markers",
                name="超体积 (HV)",
                line=dict(color="#667eea", width=2.5),
                marker=dict(size=6, color="#667eea"),
                fill="tozeroy",
                fillcolor="rgba(102, 126, 234, 0.15)",
            )
        )

        fig_hv.update_layout(
            title="Hypervolume Convergence",
            xaxis_title="迭代次数",
            yaxis_title="Hypervolume",
            height=400,
            hovermode="x unified",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_hv, use_container_width=True)

        # ── Pareto 前沿规模变化 ────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            fig_size = go.Figure()
            fig_size.add_trace(
                go.Scatter(
                    x=hist_df["iteration"],
                    y=hist_df["n_frontier"],
                    mode="lines+markers",
                    name="前沿点数",
                    line=dict(color="#11998e", width=2.5),
                    marker=dict(size=6),
                    fill="tozeroy",
                    fillcolor="rgba(17, 153, 142, 0.15)",
                )
            )
            fig_size.update_layout(
                title="Pareto 前沿规模变化",
                xaxis_title="迭代次数",
                yaxis_title="非支配解数量",
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_size, use_container_width=True)

        with col_b:
            fig_pts = go.Figure()
            fig_pts.add_trace(
                go.Scatter(
                    x=hist_df["iteration"],
                    y=hist_df["n_points"],
                    mode="lines+markers",
                    name="总数据点",
                    line=dict(color="#f5af19", width=2.5),
                    marker=dict(size=6),
                    fill="tozeroy",
                    fillcolor="rgba(245, 175, 25, 0.15)",
                )
            )
            fig_pts.update_layout(
                title="累计评估点数",
                xaxis_title="迭代次数",
                yaxis_title="总点数",
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_pts, use_container_width=True)

        # ── 优化日志表 ─────────────────────────────────────
        with st.expander("📜 优化步骤日志"):
            log_df = pd.DataFrame(results["optimization_log"])
            col_order = ["iteration", "n_frontier", "hypervolume"]
            col_order += [c for c in log_df.columns if c not in col_order]
            st.dataframe(
                log_df[[c for c in col_order if c in log_df.columns]],
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════
# Tab 3: Pareto 前沿
# ═══════════════════════════════════════════════════════════════════

with tab_pareto:
    st.subheader("📊 Pareto 前沿可视化")

    if "bayes_results" not in st.session_state:
        st.info("👈 请先在「优化设置」标签页点击「🚀 开始贝叶斯优化」按钮启动优化，完成后自动显示结果")
    else:
        results = st.session_state.bayes_results
        targets = st.session_state.bayes_selected_targets
        dirs = st.session_state.bayes_selected_dirs
        pX, pY = results["pareto_X"], results["pareto_Y"]

        if len(pX) == 0:
            st.warning("没有 Pareto 最优解")
        else:
            n_obj = len(targets)
            target_labels = [get_feature_label(t) for t in targets]

            st.caption(
                f"**{len(pX)}** 个 Pareto 最优解分布在 **{n_obj}** 维目标空间中 | "
                + " | ".join([f"{get_feature_label(t)} ({d})" for t, d in zip(targets, dirs)])
            )

            # ── 2D 散点图 ──────────────────────────────────
            if n_obj >= 2:
                st.markdown("#### 二维目标空间")

                col_x, col_y = st.columns(2)
                with col_x:
                    x_idx = st.selectbox(
                        "X 轴目标",
                        options=list(range(n_obj)),
                        format_func=lambda i: f"{target_labels[i]} ({dirs[i]})",
                        index=0,
                        key="pareto_x",
                    )
                with col_y:
                    default_y = min(1, n_obj - 1)
                    y_idx = st.selectbox(
                        "Y 轴目标",
                        options=list(range(n_obj)),
                        format_func=lambda i: f"{target_labels[i]} ({dirs[i]})",
                        index=default_y,
                        key="pareto_y",
                    )

                if x_idx == y_idx:
                    st.warning("请选择不同的 X 和 Y 目标")
                else:
                    # 区分 Pareto 点和被支配点
                    fig_2d = go.Figure()

                    # 所有点
                    if results.get("Y_all") is not None:
                        Y_all = results["Y_all"]
                        dominated_indices = list(
                            set(range(len(Y_all))) - set(range(len(pY)))
                        )
                        if dominated_indices:
                            fig_2d.add_trace(
                                go.Scatter(
                                    x=Y_all[dominated_indices, x_idx],
                                    y=Y_all[dominated_indices, y_idx],
                                    mode="markers",
                                    name="被支配点",
                                    marker=dict(
                                        size=6, color="lightgrey", opacity=0.5,
                                    ),
                                    hovertemplate=(
                                        f"{target_labels[x_idx]}: %{{x:.2f}}<br>"
                                        f"{target_labels[y_idx]}: %{{y:.2f}}<extra></extra>"
                                    ),
                                )
                            )

                    # Pareto 前沿点
                    fig_2d.add_trace(
                        go.Scatter(
                            x=pY[:, x_idx],
                            y=pY[:, y_idx],
                            mode="markers+lines",
                            name="Pareto 前沿",
                            marker=dict(
                                size=10,
                                color=np.arange(len(pY)),
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(title="排序"),
                                line=dict(color="white", width=1),
                            ),
                            line=dict(color="rgba(102,126,234,0.3)", width=1),
                            hovertemplate=(
                                f"{target_labels[x_idx]}: %{{x:.2f}}<br>"
                                f"{target_labels[y_idx]}: %{{y:.2f}}<extra></extra>"
                            ),
                        )
                    )

                    fig_2d.update_layout(
                        title=f"Pareto 前沿: {target_labels[x_idx]} vs {target_labels[y_idx]}",
                        xaxis_title=f"{target_labels[x_idx]} ({dirs[x_idx]})",
                        yaxis_title=f"{target_labels[y_idx]} ({dirs[y_idx]})",
                        height=450,
                        hovermode="closest",
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)

            # ── 3D 散点图 ──────────────────────────────────
            if n_obj >= 3:
                st.markdown("#### 三维目标空间")

                col_z1, col_z2, col_z3 = st.columns(3)
                with col_z1:
                    x3d_idx = st.selectbox(
                        "X 轴",
                        options=list(range(n_obj)),
                        format_func=lambda i: target_labels[i],
                        index=0,
                        key="pareto_3d_x",
                    )
                with col_z2:
                    y3d_idx = st.selectbox(
                        "Y 轴",
                        options=list(range(n_obj)),
                        format_func=lambda i: target_labels[i],
                        index=min(1, n_obj - 1),
                        key="pareto_3d_y",
                    )
                with col_z3:
                    z3d_idx = st.selectbox(
                        "Z 轴",
                        options=list(range(n_obj)),
                        format_func=lambda i: target_labels[i],
                        index=min(2, n_obj - 1),
                        key="pareto_3d_z",
                    )

                if len({x3d_idx, y3d_idx, z3d_idx}) < 3:
                    st.warning("请选择 3 个不同的目标")
                else:
                    fig_3d = go.Figure()
                    fig_3d.add_trace(
                        go.Scatter3d(
                            x=pY[:, x3d_idx],
                            y=pY[:, y3d_idx],
                            z=pY[:, z3d_idx],
                            mode="markers",
                            marker=dict(
                                size=6,
                                color=np.arange(len(pY)),
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(title="排序"),
                                line=dict(color="white", width=0.5),
                            ),
                            hovertemplate=(
                                f"{target_labels[x3d_idx]}: %{{x:.2f}}<br>"
                                f"{target_labels[y3d_idx]}: %{{y:.2f}}<br>"
                                f"{target_labels[z3d_idx]}: %{{z:.2f}}<extra></extra>"
                            ),
                        )
                    )
                    fig_3d.update_layout(
                        title="3D Pareto 前沿",
                        scene=dict(
                            xaxis_title=target_labels[x3d_idx],
                            yaxis_title=target_labels[y3d_idx],
                            zaxis_title=target_labels[z3d_idx],
                        ),
                        height=500,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

            # ── 平行坐标图 ──────────────────────────────────
            if n_obj >= 2:
                st.markdown("#### 平行坐标图")
                pareto_df = pd.DataFrame(pY, columns=target_labels)
                pareto_df["_idx"] = range(len(pareto_df))

                fig_par = go.Figure(
                    go.Parcoords(
                        line=dict(
                            color=pareto_df[target_labels[0]],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=target_labels[0]),
                        ),
                        dimensions=[
                            dict(
                                label=lbl,
                                values=pareto_df[lbl],
                            )
                            for lbl in target_labels
                        ],
                    )
                )
                fig_par.update_layout(
                    title="Pareto 前沿平行坐标图",
                    height=350,
                )
                st.plotly_chart(fig_par, use_container_width=True)

            # ── 前沿点详情表 ────────────────────────────────
            with st.expander("📋 Pareto 最优解详情"):
                opt = st.session_state.get("bayes_optimizer")
                if opt is not None:
                    pareto_table = opt.get_pareto_df()
                    st.dataframe(
                        pareto_table.style.background_gradient(
                            subset=[c for c in pareto_table.columns if "_pred" in c],
                            cmap="viridis",
                        ),
                        use_container_width=True,
                    )

# ═══════════════════════════════════════════════════════════════════
# Tab 4: 结果导出
# ═══════════════════════════════════════════════════════════════════

with tab_export:
    st.subheader("📋 结果导出")

    if "bayes_results" not in st.session_state:
        st.info("👈 请先在「优化设置」标签页点击「🚀 开始贝叶斯优化」按钮启动优化，完成后自动显示结果")
    else:
        results = st.session_state.bayes_results
        opt = st.session_state.bayes_optimizer
        targets = st.session_state.bayes_selected_targets

        # ── 摘要 ──────────────────────────────────────────
        st.markdown("#### 优化摘要")

        hist_df = results["hypervolume_history"]
        pX, pY = results["pareto_X"], results["pareto_Y"]

        summary_data = {
            "指标": [
                "优化迭代次数",
                "初始数据量",
                "最终数据量",
                "Pareto 最优解数",
                "初始超体积",
                "最终超体积",
                "超体积改进率 (%)",
                "代理模型类型",
                "采集函数类型",
                "优化目标",
            ],
            "值": [
                str(len(results["optimization_log"])),
                str(hist_df["n_points"].iloc[0]),
                str(hist_df["n_points"].iloc[-1]),
                str(len(pX)),
                f"{hist_df['hypervolume'].iloc[0]:,.0f}",
                f"{hist_df['hypervolume'].iloc[-1]:,.0f}",
                f"{(hist_df['hypervolume'].iloc[-1] - hist_df['hypervolume'].iloc[0]) / abs(hist_df['hypervolume'].iloc[0]) * 100:+.1f}%",
                opt.surrogate_type if opt else "N/A",
                opt.acq_type if opt else "N/A",
                ", ".join([get_feature_label(t) for t in targets]),
            ],
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True,
        )

        # ── 最佳推荐点 ──────────────────────────────────────
        st.markdown("#### 🏆 推荐配比（到理想点距离最小）")
        if opt is not None:
            best = opt.get_best_point()
            if best:
                best_dict = {}
                for j, name in enumerate(opt.feature_names):
                    best_dict[get_feature_label(name)] = f"{best['X'][j]:.3f}"
                for j, name in enumerate(targets):
                    best_dict[f"预测 {get_feature_label(name)}"] = f"{best['Y'][j]:.2f}"
                best_dict["综合距离"] = f"{best['distance']:.4f}"

                best_df = pd.DataFrame([best_dict])
                st.dataframe(best_df.T.rename(columns={0: "值"}), use_container_width=True)

        # ── CSV 导出 ───────────────────────────────────────
        st.markdown("#### 💾 下载")

        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            if opt is not None:
                pareto_df = opt.get_pareto_df()
                csv_data = pareto_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 下载 Pareto 最优解 (CSV)",
                    data=csv_data,
                    file_name="uhpc_pareto_frontier.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col_dl2:
            hist_csv = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 下载收敛历史 (CSV)",
                data=hist_csv,
                file_name="uhpc_convergence_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_dl3:
            if opt is not None and opt.X_train is not None:
                full_df = pd.DataFrame(
                    opt.X_train, columns=opt.feature_names
                )
                for j, name in enumerate(opt.target_names):
                    full_df[f"{name}_pred"] = opt.Y_train[:, j]
                full_csv = full_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 下载全部评估点 (CSV)",
                    data=full_csv,
                    file_name="uhpc_all_evaluated_points.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # ── 模型持久化 ──────────────────────────────────────
        st.markdown("#### 🔧 优化器持久化")
        st.caption("保存当前优化器状态，下次可直接加载继续优化")

        col_save1, col_save2 = st.columns(2)
        with col_save1:
            save_name = st.text_input(
                "文件名",
                value="bayesian_optimizer.pkl",
                key="bayes_save_name",
            )
        with col_save2:
            st.caption("")
            if st.button("💾 保存优化器状态", use_container_width=True):
                import tempfile

                models_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models",
                )
                os.makedirs(models_dir, exist_ok=True)
                save_path = os.path.join(models_dir, save_name)
                opt.save(save_path)
                st.success(f"✅ 已保存到 `{save_path}`")

        # ── 加载 ──────────────────────────────────────────
        with st.expander("📂 加载已保存的优化器"):
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
            )
            if os.path.isdir(models_dir):
                pkl_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
                if pkl_files:
                    selected_pkl = st.selectbox("选择文件", pkl_files, key="bayes_load")
                    if st.button("📂 加载", key="bayes_load_btn"):
                        load_path = os.path.join(models_dir, selected_pkl)
                        try:
                            opt2 = BayesianOptimizer().load(load_path)
                            st.session_state.bayes_optimizer = opt2
                            st.session_state.bayes_results = opt2.get_results()
                            st.success(f"✅ 已加载 `{selected_pkl}`")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载失败: {e}")
                else:
                    st.caption("暂无已保存的优化器文件")
