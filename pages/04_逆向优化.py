# -*- coding: utf-8 -*-
"""
04_逆向优化 — 多目标进化优化 + TOPSIS 方案优选
基于 pymoo NSGA-II/UNSGA-III/AGE-MOEA + TOPSIS 多准则决策
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

# 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    get_all_feature_names,
    get_all_target_names,
    get_feature_ranges,
    get_target_direction,
    get_feature_label,
    get_targets,
    get_constraints,
)
from core.model_trainer import MultiTargetTrainer

st.set_page_config(
    page_title="逆向优化 - UHPC-AutoDesign",
    page_icon="🎯",
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
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════
# 侧边栏：数据源 + 模型
# ═══════════════════════════════════════════════════════════════════

st.sidebar.header("📦 数据源与模型")

DATA_LOADED = False
_available_features: list = []
_available_targets: list = []
_X_data: Optional[pd.DataFrame] = None
_Y_data: Optional[pd.DataFrame] = None


def _load_data():
    """加载数据：优先 session_state，其次 literature.csv"""
    # 检查 session state
    for key in ["cleaned_data", "synthetic_data"]:
        if key in st.session_state and st.session_state[key] is not None:
            return st.session_state[key]

    # 回退到 literature.csv
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "literature.csv",
    )
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


df_all = _load_data()
if df_all is not None and len(df_all) > 3:
    features = [f for f in get_all_feature_names() if f in df_all.columns]
    targets = [t for t in get_all_target_names() if t in df_all.columns]
    all_cols = features + targets
    df_clean = df_all[all_cols].dropna()
    if len(df_clean) >= 5:
        DATA_LOADED = True
        _available_features = features
        _available_targets = targets
        _X_data = df_clean[features]
        _Y_data = df_clean[targets]
        st.sidebar.success(
            f"✅ 已加载 {len(df_clean)} 条数据 | "
            f"{len(features)} 特征 × {len(targets)} 目标"
        )
    else:
        st.sidebar.warning("⚠️ 清洗后数据不足（<5 条）")
else:
    st.sidebar.error("❌ 没有可用数据")

# ═══════════════════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════════════════

st.title("🎯 多目标逆向优化 + TOPSIS 方案优选")
st.caption(
    "基于 pymoo 进化算法（NSGA-II / UNSGA-III / AGE-MOEA）进行多目标优化，"
    "通过 TOPSIS 从 Pareto 前沿推荐最优配方方案"
)

if not DATA_LOADED:
    st.warning("👈 请先确保有可用数据。可前往「数据管理」上传数据或「虚拟生成」生成合成数据。")
    st.stop()

# ── 标签页 ──────────────────────────────────────────────────────

tab_setup, tab_pareto, tab_topsis, tab_details = st.tabs(
    ["🎯 优化设置", "📊 Pareto 前沿", "🏆 TOPSIS 优选", "📋 详细结果"]
)

# ═══════════════════════════════════════════════════════════════════
# Tab 1: 优化设置
# ═══════════════════════════════════════════════════════════════════

with tab_setup:
    st.subheader("优化目标与算法配置")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 🎯 选择优化目标")

        target_info = get_targets()
        target_options = []
        for t in target_info:
            if t["name"] in _available_targets:
                emoji = {"maximize": "↑", "minimize": "↓", "target": "⊙"}
                target_options.append(
                    f"{emoji.get(t['direction'], '→')} {t['label']} ({t['name']})"
                )

        default_idx = list(range(min(5, len(target_options))))
        selected_labels = st.multiselect(
            "勾选要优化的目标（至少 2 个）",
            options=target_options,
            default=[target_options[i] for i in default_idx],
            key="moo_targets",
            help="↑ 最大化 | ↓ 最小化 | ⊙ 趋近目标值",
        )

        selected_targets = []
        selected_dirs = []
        for label in selected_labels:
            name = label.split("(")[-1].rstrip(")")
            selected_targets.append(name)
            d = get_target_direction(name)
            selected_dirs.append(d if d else "minimize")

        st.caption(
            f"已选 **{len(selected_targets)}** 个目标: {', '.join(selected_targets)}"
        )

    with col_right:
        st.markdown("#### ⚙️ 物理约束")
        constraints_list = get_constraints()
        for c in constraints_list:
            if c.get("range"):
                lo, hi = c["range"]
                st.caption(f"• {c['description']}: [{lo}, {hi}]")
            else:
                st.caption(f"• {c['description']}: ±{c.get('tolerance', 0.001)}")
        st.caption("_约束在 pymoo Problem 中自动执行_")

    st.divider()

    # ── 算法参数 ──────────────────────────────────────────────

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        algorithm = st.selectbox(
            "进化算法",
            options=["nsga2", "nsga3", "agemoea"],
            index=0,
            format_func=lambda x: {
                "nsga2": "🧬 NSGA-II",
                "nsga3": "🧬 NSGA-III",
                "agemoea": "🧬 AGE-MOEA",
            }[x],
            key="moo_algo",
            help="NSGA-II = 经典通用 | NSGA-III = 高维目标 | AGE-MOEA = 自适应网格",
        )

    with col2:
        pop_size = st.slider(
            "种群大小",
            min_value=40,
            max_value=400,
            value=100,
            step=20,
            key="moo_pop",
            help="种群越大探索越充分，但耗时更长",
        )

    with col3:
        n_gen = st.slider(
            "进化代数",
            min_value=20,
            max_value=500,
            value=200,
            step=20,
            key="moo_gen",
            help="代数越多收敛越好",
        )

    with col4:
        moo_seed = st.number_input(
            "随机种子",
            value=42,
            min_value=0,
            max_value=9999,
            key="moo_seed",
        )

    st.divider()

    # ── TOPSIS 权重 ───────────────────────────────────────────

    st.markdown("#### ⚖️ TOPSIS 目标权重")

    if len(selected_targets) < 2:
        st.warning("⚠️ 请至少选择 2 个优化目标")
        st.stop()

    weight_cols = st.columns(min(len(selected_targets), 5))
    weights_dict = {}
    for i, t_name in enumerate(selected_targets):
        t_label = get_feature_label(t_name) if t_name in get_all_feature_names() else t_name
        d = selected_dirs[i]
        arrow = {"maximize": "↑", "minimize": "↓", "target": "⊙"}.get(d, "")
        with weight_cols[i % len(weight_cols)]:
            weights_dict[t_name] = st.slider(
                f"{arrow} {t_label}",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.1,
                key=f"weight_{t_name}",
                help=f"方向: {d}",
            )

    st.caption(
        "权重将自动归一化。权重越大，TOPSIS 排序时该目标越重要。"
    )

    st.divider()

    # ── 启动优化 ──────────────────────────────────────────────

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        start_opt = st.button(
            "🚀 启动多目标优化",
            type="primary",
            use_container_width=True,
            key="moo_start",
        )

    # ── 执行优化 ──────────────────────────────────────────────

    if start_opt:
        if len(selected_targets) < 2:
            st.error("❌ 请至少选择 2 个优化目标")
        else:
            # 1) 训练代理模型
            with st.status("🔄 训练代理模型 (RF)...", expanded=True) as status:
                st.write("使用 Random Forest 作为代理模型进行目标预测...")
                X_train = _X_data[_available_features]
                trainer = MultiTargetTrainer("random_forest")
                trainer.train_all(
                    X_train.select_dtypes(include=[np.number]),
                    _Y_data[selected_targets],
                )
                st.write("✅ 代理模型训练完成")
                status.update(label="✅ 代理模型就绪", state="complete")

            # 2) 运行 pymoo 优化
            with st.status("🧬 运行多目标进化优化...", expanded=True) as status:
                from core.multi_objective import MultiObjectiveOptimizer

                w_raw = np.array([weights_dict[t] for t in selected_targets])
                opt = MultiObjectiveOptimizer(
                    feature_names=_available_features,
                    target_names=selected_targets,
                    directions=selected_dirs,
                    algorithm=algorithm,
                    surrogate_model=trainer,
                    pop_size=pop_size,
                    verbose=False,
                )

                st.write(
                    f"算法: {opt.ALGORITHMS[algorithm]} | "
                    f"种群: {pop_size} | 代数: {n_gen} | "
                    f"目标数: {len(selected_targets)}"
                )

                progress_bar = st.progress(0, text="优化进行中...")
                results = opt.optimize(n_generations=n_gen, seed=moo_seed)
                progress_bar.progress(1.0, text="优化完成 ✅")
                status.update(
                    label=f"✅ 优化完成 — {results['n_pareto']} 个 Pareto 最优解, "
                          f"耗时 {results['time']:.1f}s",
                    state="complete",
                )

            # 3) TOPSIS 排序
            from core.topsis_selector import TOPSISSelector

            w_norm = w_raw / w_raw.sum()
            selector = TOPSISSelector(
                pareto_Y=opt.pareto_Y,
                target_names=selected_targets,
                directions=selected_dirs,
                weights=w_norm,
                pareto_X=opt.pareto_X,
                feature_names=_available_features,
            )
            selector.compute_scores()

            # 存入 session_state
            st.session_state.moo_optimizer = opt
            st.session_state.moo_selector = selector
            st.session_state.moo_targets = selected_targets
            st.session_state.moo_dirs = selected_dirs
            st.session_state.moo_weights = w_norm
            st.session_state.moo_ran = True

            st.success(
                f"🎉 优化完成！{results['n_pareto']} 个 Pareto 最优解，"
                f"耗时 {results['time']:.1f}s。"
                f"切换到「Pareto 前沿」和「TOPSIS 优选」标签页查看结果。"
            )

# ═══════════════════════════════════════════════════════════════════
# Tab 2: Pareto 前沿
# ═══════════════════════════════════════════════════════════════════

with tab_pareto:
    if "moo_ran" not in st.session_state or not st.session_state.moo_ran:
        st.info("👈 请先在「优化设置」标签页启动优化")
    else:
        opt = st.session_state.moo_optimizer
        selector = st.session_state.moo_selector
        targets = st.session_state.moo_targets
        dirs = st.session_state.moo_dirs

        st.subheader("📊 Pareto 前沿可视化")

        # ── 汇总卡片 ──
        n_pareto = len(opt.pareto_X) if opt.pareto_X is not None else 0
        card_cols = st.columns(4)
        card_cols[0].markdown(
            f"<div class='metric-card'><div class='value'>{n_pareto}</div>"
            f"<div class='label'>Pareto 最优解</div></div>",
            unsafe_allow_html=True,
        )
        card_cols[1].markdown(
            f"<div class='metric-card green'><div class='value'>{opt.optimization_time:.0f}s</div>"
            f"<div class='label'>优化耗时</div></div>",
            unsafe_allow_html=True,
        )
        card_cols[2].markdown(
            f"<div class='metric-card orange'><div class='value'>{opt.ALGORITHMS.get(opt.algorithm, opt.algorithm)}</div>"
            f"<div class='label'>算法</div></div>",
            unsafe_allow_html=True,
        )
        top_score = selector._scores.max() if selector._scores is not None else 0
        card_cols[3].markdown(
            f"<div class='metric-card'><div class='value'>{top_score:.3f}</div>"
            f"<div class='label'>最优 TOPSIS 得分</div></div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── 3D / 2D Pareto 前沿 ──
        if n_pareto > 0 and len(targets) >= 2:
            pareto_df = opt.get_pareto_df()

            col3d, col_ctrl = st.columns([3, 1])

            with col_ctrl:
                st.markdown("**坐标轴选择**")
                if len(targets) >= 3:
                    x_axis = st.selectbox("X 轴", targets, index=0, key="pareto_x")
                    y_axis = st.selectbox("Y 轴", targets, index=min(1, len(targets)-1), key="pareto_y")
                    z_axis = st.selectbox("Z 轴", targets, index=min(2, len(targets)-1), key="pareto_z")
                    use_3d = True
                else:
                    x_axis = st.selectbox("X 轴", targets, index=0, key="pareto_x2d")
                    y_axis = st.selectbox("Y 轴", targets, index=min(1, len(targets)-1), key="pareto_y2d")
                    use_3d = False

                # 按 TOPSIS 得分着色
                scores = selector._scores
                color_by = st.selectbox(
                    "着色依据",
                    options=["TOPSIS得分"] + targets,
                    index=0,
                    key="pareto_color",
                )

            with col3d:
                if use_3d:
                    color_vals = scores if color_by == "TOPSIS得分" else pareto_df[color_by].values
                    fig = go.Figure(
                        go.Scatter3d(
                            x=pareto_df[x_axis],
                            y=pareto_df[y_axis],
                            z=pareto_df[z_axis],
                            mode="markers",
                            marker=dict(
                                size=5,
                                color=color_vals,
                                colorscale="Viridis",
                                colorbar=dict(title=color_by),
                                opacity=0.85,
                            ),
                            hovertemplate=(
                                f"{x_axis}: %{{x:.1f}}<br>"
                                f"{y_axis}: %{{y:.1f}}<br>"
                                f"{z_axis}: %{{z:.1f}}<br>"
                                f"TOPSIS: %{{customdata:.3f}}<extra></extra>"
                            ),
                            customdata=scores,
                        )
                    )
                    fig.update_layout(
                        scene=dict(
                            xaxis_title=x_axis,
                            yaxis_title=y_axis,
                            zaxis_title=z_axis,
                        ),
                        title="3D Pareto 前沿",
                        height=550,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                else:
                    color_vals = scores if color_by == "TOPSIS得分" else pareto_df[color_by].values
                    fig = go.Figure(
                        go.Scatter(
                            x=pareto_df[x_axis],
                            y=pareto_df[y_axis],
                            mode="markers",
                            marker=dict(
                                size=8,
                                color=color_vals,
                                colorscale="Viridis",
                                colorbar=dict(title=color_by),
                                opacity=0.85,
                            ),
                            hovertemplate=(
                                f"{x_axis}: %{{x:.1f}}<br>"
                                f"{y_axis}: %{{y:.1f}}<br>"
                                f"TOPSIS: %{{customdata:.3f}}<extra></extra>"
                            ),
                            customdata=scores,
                        )
                    )
                    fig.update_layout(
                        xaxis_title=x_axis,
                        yaxis_title=y_axis,
                        title="2D Pareto 前沿",
                        height=500,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                st.plotly_chart(fig, use_container_width=True)

            # ── 收敛曲线 ──
            st.divider()
            st.subheader("📈 收敛曲线")
            conv_df = opt.get_convergence_df()
            if len(conv_df) > 0:
                fig_conv = go.Figure(
                    go.Scatter(
                        x=conv_df["generation"],
                        y=conv_df["n_nondominated"],
                        mode="lines+markers",
                        line=dict(color="#667eea", width=2),
                        marker=dict(size=4),
                        fill="tozeroy",
                        fillcolor="rgba(102,126,234,0.15)",
                    )
                )
                fig_conv.update_layout(
                    xaxis_title="进化代数",
                    yaxis_title="非支配解数量",
                    height=300,
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig_conv, use_container_width=True)

        else:
            st.warning("Pareto 前沿为空，请检查优化设置或数据质量。")

# ═══════════════════════════════════════════════════════════════════
# Tab 3: TOPSIS 优选
# ═══════════════════════════════════════════════════════════════════

with tab_topsis:
    if "moo_ran" not in st.session_state or not st.session_state.moo_ran:
        st.info("👈 请先在「优化设置」标签页启动优化")
    else:
        selector = st.session_state.moo_selector
        targets = st.session_state.moo_targets
        features = _available_features

        st.subheader("🏆 TOPSIS 方案优选结果")

        top_n = st.slider(
            "推荐方案数量",
            min_value=3,
            max_value=min(20, len(selector.pareto_Y)),
            value=5,
            key="topsis_top_n",
        )

        # ── Top-N 推荐表 ──
        top_df = selector.get_top_n(n=top_n)

        st.markdown(f"#### 📋 Top-{top_n} 推荐方案")
        st.caption(
            "基于加权 TOPSIS 排序的最优配比方案。得分越接近 1 越优。"
        )

        # 权重显示
        w_display = ", ".join([
            f"{t}: {st.session_state.moo_weights[i]:.2f}"
            for i, t in enumerate(targets)
        ])
        st.caption(f"**当前权重:** {w_display}")

        st.dataframe(
            top_df.style
            .background_gradient(subset=["TOPSIS得分"], cmap="YlOrRd")
            .format({"TOPSIS得分": "{:.4f}"}),
            use_container_width=True,
        )

        # ── 导出 Top-N ──
        csv_top = top_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=f"📥 导出 Top-{top_n} 方案 (CSV)",
            data=csv_top,
            file_name=f"uhpc_topsis_top{top_n}.csv",
            mime="text/csv",
        )

        st.divider()

        # ── 可视化 ──
        col_radar, col_parallel = st.columns(2)

        with col_radar:
            st.markdown("#### 🕸️ 雷达图")
            top_indices = np.argsort(selector._scores)[::-1][:min(top_n, 6)]
            fig_radar = selector.plot_radar(
                top_indices=list(top_indices),
                title=f"Top-{min(top_n, 6)} 方案雷达图",
            )
            if fig_radar is not None:
                st.plotly_chart(fig_radar, use_container_width=True)

        with col_parallel:
            st.markdown("#### 📐 平行坐标图")
            fig_parallel = selector.plot_parallel_coordinates(
                highlight_top=min(top_n, 8),
                title="Pareto 前沿平行坐标图",
            )
            if fig_parallel is not None:
                st.plotly_chart(fig_parallel, use_container_width=True)

        # ── TOPSIS 得分柱状图 ──
        st.divider()
        st.markdown("#### 📊 TOPSIS 得分排名")
        fig_bar = selector.plot_score_bar(n_top=min(top_n + 5, len(selector._scores)))
        if fig_bar is not None:
            st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# Tab 4: 详细结果
# ═══════════════════════════════════════════════════════════════════

with tab_details:
    if "moo_ran" not in st.session_state or not st.session_state.moo_ran:
        st.info("👈 请先在「优化设置」标签页启动优化")
    else:
        opt = st.session_state.moo_optimizer
        selector = st.session_state.moo_selector

        st.subheader("📋 全部 Pareto 解详情")

        # 完整 Pareto 表
        full_df = selector.rank()
        st.dataframe(
            full_df.style.background_gradient(subset=["TOPSIS得分"], cmap="YlOrRd"),
            use_container_width=True,
        )

        st.caption(f"共 {len(full_df)} 个 Pareto 非支配解")

        # ── 导出 ──
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            csv_pareto = opt.get_pareto_df().to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 导出 Pareto 前沿 (CSV)",
                data=csv_pareto,
                file_name="uhpc_pareto_frontier.csv",
                mime="text/csv",
                key="export_pareto",
            )

        with col_exp2:
            csv_ranked = full_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 导出 TOPSIS 排序表 (CSV)",
                data=csv_ranked,
                file_name="uhpc_topsis_ranked.csv",
                mime="text/csv",
                key="export_topsis",
            )

        # ── TOPSIS 详细中间结果 ──
        with st.expander("🔍 TOPSIS 计算详情"):
            detail = selector.get_scores_detail()
            st.markdown(f"**权重:** {np.round(detail['weights'], 3)}")
            st.markdown(f"**理想解 (PIS):** {np.round(detail['pis'], 4)}")
            st.markdown(f"**负理想解 (NIS):** {np.round(detail['nis'], 4)}")
            st.markdown(f"**得分范围:** [{detail['scores'].min():.4f}, {detail['scores'].max():.4f}]")
            st.markdown(f"**平均得分:** {detail['scores'].mean():.4f}")

            dist_df = pd.DataFrame({
                "方案": [f"#{i+1}" for i in range(len(detail["scores"]))],
                "到PIS距离": np.round(detail["dist_to_pis"], 4),
                "到NIS距离": np.round(detail["dist_to_nis"], 4),
                "相对贴近度": np.round(detail["scores"], 4),
            }).sort_values("相对贴近度", ascending=False)
            st.dataframe(dist_df.head(10), use_container_width=True)

        # ── 收敛历史 ──
        with st.expander("📈 收敛历史数据"):
            conv_df = opt.get_convergence_df()
            if len(conv_df) > 0:
                st.dataframe(conv_df, use_container_width=True)
