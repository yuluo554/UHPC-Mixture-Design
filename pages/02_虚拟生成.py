# -*- coding: utf-8 -*-
"""
02_虚拟生成 — CTGAN/TVAE 虚拟 UHPC 配比数据生成
基于 SDV 从有限配比数据生成高质量合成数据，含 TSTR 验证
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import time
from io import BytesIO

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    get_all_feature_names, get_all_target_names,
    get_feature_label, get_categorical_features,
)
from core.data_manager import DataManager
from core.gan_generator import (
    CTGANGenerator, TVAEGenerator, TSTRValidator,
    auto_generate_with_validation,
)

# ─── 页面配置 ────────────────────────────────────────────────────
st.set_page_config(page_title="虚拟生成 - UHPC-AutoDesign", page_icon="🧬", layout="wide")

st.title("🧬 CTGAN 虚拟配比生成")
st.caption("使用 SDV/CTGAN 从有限真实数据中学习联合分布，生成统计等效的虚拟 UHPC 配比数据")

# ─── 初始化 Session State ────────────────────────────────────────
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager()
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
if "feature_columns" not in st.session_state:
    st.session_state.feature_columns = get_all_feature_names()
if "target_columns" not in st.session_state:
    st.session_state.target_columns = get_all_target_names()

# GAN 专用 state
if "gan_generator" not in st.session_state:
    st.session_state.gan_generator = None
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None
if "gan_validator" not in st.session_state:
    st.session_state.gan_validator = None
if "gan_validation_results" not in st.session_state:
    st.session_state.gan_validation_results = None
if "gan_is_fitting" not in st.session_state:
    st.session_state.gan_is_fitting = False

dm = st.session_state.data_manager

# ─── 标签页 ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 训练生成器", "📊 质量验证", "📋 数据浏览", "🎯 条件定向生成"
])

# ═══════════════════════════════════════════════════════════════════
# 获取可用数据（优先用 session_state 中的清洗数据）
# ═══════════════════════════════════════════════════════════════════
def _get_working_data():
    """获取当前可用的工作数据"""
    if st.session_state.cleaned_data is not None:
        return st.session_state.cleaned_data
    if dm.raw_data is not None:
        return dm.raw_data
    return None


def _load_sample_if_needed():
    """如果没有数据，自动加载示例数据"""
    data = _get_working_data()
    if data is None:
        sample_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "literature.csv"
        )
        if os.path.exists(sample_path):
            dm.load_csv(sample_path)
            dm.clean(remove_outliers=True, impute=True, contamination=0.05)
            st.session_state.cleaned_data = dm.cleaned_data
            st.session_state.feature_columns = dm.feature_columns
            st.session_state.target_columns = dm.target_columns
            return dm.cleaned_data
    return data


# ═══════════════════════════════════════════════════════════════════
# Tab 1: 训练生成器
# ═══════════════════════════════════════════════════════════════════
with tab1:
    data = _load_sample_if_needed()

    if data is None:
        st.warning("⚠️ 请先在「📊 数据管理」中加载数据")
    else:
        st.subheader("⚙️ 训练参数设置")

        col_params, col_info = st.columns([1, 1])

        with col_params:
            generator_type = st.selectbox(
                "生成器类型",
                ["ctgan", "tvae"],
                format_func=lambda x: "CTGAN (条件表格 GAN)" if x == "ctgan" else "TVAE (表格变分自编码器)",
                help="CTGAN 对混合类型数据效果好；TVAE 更适合连续型占主导的数据"
            )

            epochs = st.slider(
                "训练轮数 (Epochs)",
                min_value=50, max_value=2000, value=300, step=50,
                help="数据越少，需要的 epochs 越多。37条数据建议 300-500 epochs"
            )

            batch_size = st.slider(
                "批次大小 (Batch Size)",
                min_value=10, max_value=500, value=100, step=10,
                help="每批处理的样本数。数据少时建议用较小值"
            )

            gen_count = st.number_input(
                "生成数据量",
                min_value=100, max_value=50000, value=1000, step=100,
                help="要生成的合成数据行数"
            )

            # 显示数据概要
            st.divider()
            st.caption(f"📋 训练数据：{len(data)} 条 × {len(data.columns)} 列")
            st.caption(f"📋 计划生成：{gen_count} 条合成数据")

            # 训练按钮
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                train_clicked = st.button(
                    "🚀 训练生成器 + 生成数据",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.gan_is_fitting,
                )

        with col_info:
            st.info("""
            **CTGAN 原理简介**

            CTGAN (Conditional Tabular GAN) 是专门为表格数据设计的
            生成对抗网络，核心能力：

            - **混合类型处理**：同时处理数值型和分类型特征
            - **多模态分布**：捕获非正态、多峰分布
            - **条件生成**：可按预设条件定向生成数据
            - **隐私保护**：生成的数据不直接复制原始记录

            **数据量建议**
            - ≥ 200 条：较好效果
            - 50-200 条：可用，需增加 epochs
            - < 50 条：生成质量有限，建议先用少量数据探索
            """)

        # ─── 执行训练 ──────────────────────────────────────────────
        if train_clicked:
            st.session_state.gan_is_fitting = True

            try:
                # 准备数据
                all_cols = (
                    st.session_state.feature_columns + st.session_state.target_columns
                )
                available_cols = [c for c in all_cols if c in data.columns]
                train_data = data[available_cols].copy()
                cat_cols = [
                    c for c in get_categorical_features() if c in available_cols
                ]

                # 进度区域
                progress_bar = st.progress(0, text="初始化生成器...")
                status_text = st.empty()

                # 创建生成器
                if generator_type == "tvae":
                    generator = TVAEGenerator(
                        epochs=epochs, batch_size=batch_size, verbose=False
                    )
                else:
                    generator = CTGANGenerator(
                        epochs=epochs, batch_size=batch_size, verbose=False
                    )

                # 模拟进度更新（SDV 的 fit 不提供回调，用分阶段更新）
                progress_bar.progress(10, text="正在训练 CTGAN 生成器...")
                status_text.info(f"🔧 训练 {generator_type.upper()} — 这可能需要 1-5 分钟...")
                start_time = time.time()

                generator.fit(train_data, categorical_columns=cat_cols)

                elapsed = time.time() - start_time
                progress_bar.progress(60, text=f"训练完成（{elapsed:.0f}秒），生成数据中...")

                # 生成数据
                synthetic = generator.generate(num_rows=gen_count)
                progress_bar.progress(90, text="生成完成，保存结果...")

                # 保存到 session state
                st.session_state.gan_generator = generator
                st.session_state.synthetic_data = synthetic
                st.session_state.gan_validator = None
                st.session_state.gan_validation_results = None

                progress_bar.progress(100, text="全部完成！✅")
                status_text.success(
                    f"✅ 训练完成！耗时 {elapsed:.1f} 秒，生成 {len(synthetic)} 条合成数据"
                )

                # 显示快速对比
                st.divider()
                st.subheader("📊 真实 vs 合成数据快速对比")

                quick_cols = st.multiselect(
                    "选择要对比的列",
                    available_cols,
                    default=available_cols[:4],
                    key="quick_compare_cols"
                )

                if quick_cols:
                    compare_data = []
                    for col in quick_cols:
                        compare_data.append({
                            "列": col,
                            "真实均值": f"{data[col].mean():.2f}",
                            "合成均值": f"{synthetic[col].mean():.2f}",
                            "真实标准差": f"{data[col].std():.2f}",
                            "合成标准差": f"{synthetic[col].std():.2f}",
                        })
                    st.dataframe(
                        pd.DataFrame(compare_data),
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as e:
                st.error(f"❌ 训练失败: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                st.session_state.gan_is_fitting = False

        # 显示已有的合成数据摘要（如果之前已生成）
        elif st.session_state.synthetic_data is not None:
            synth = st.session_state.synthetic_data
            st.divider()
            st.success(f"✅ 已有合成数据：{len(synth)} 条（无需重复训练）")
            st.caption("切换到其他标签页查看质量验证和详细数据")

# ═══════════════════════════════════════════════════════════════════
# Tab 2: 质量验证
# ═══════════════════════════════════════════════════════════════════
with tab2:
    data = _get_working_data()
    synth = st.session_state.synthetic_data

    if synth is None:
        st.info("👈 请先在「🧬 训练生成器」中训练并生成数据")
    else:
        st.subheader("📊 合成数据质量验证")

        # 运行 TSTR 验证按钮
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            if st.button("🔬 运行 TSTR 全面验证", type="primary",
                         disabled=st.session_state.gan_is_fitting):
                with st.spinner("TSTR 验证中..."):
                    feature_cols = [
                        c for c in st.session_state.feature_columns
                        if c in data.columns and c in synth.columns
                    ]
                    target_cols = [
                        c for c in st.session_state.target_columns
                        if c in data.columns and c in synth.columns
                    ]
                    validator = TSTRValidator(
                        real_data=data,
                        synthetic_data=synth,
                        feature_columns=feature_cols,
                        target_columns=target_cols,
                    )
                    results = validator.evaluate()
                    st.session_state.gan_validator = validator
                    st.session_state.gan_validation_results = results
                    st.success("✅ TSTR 验证完成！")

        # ─── 展示验证结果 ──────────────────────────────────────────
        results = st.session_state.gan_validation_results
        if results is not None:
            # TSTR 性能卡片
            if "tstr" in results and "_summary" in results["tstr"]:
                s = results["tstr"]["_summary"]
                st.divider()
                st.subheader("🎯 TSTR (Train-on-Synthetic, Test-on-Real) 性能")

                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    r2_ratio = s.get("avg_r2_ratio", 0) or 0
                    st.metric(
                        "TSTR/TRTR R² 比值",
                        f"{r2_ratio:.4f}",
                        help="越接近 1.0 越好。表示合成数据训练模型 Vs 真实数据训练模型",
                    )
                with col_t2:
                    mae_ratio = s.get("avg_mae_ratio", 0) or 0
                    st.metric(
                        "TSTR/TRTR MAE 比值",
                        f"{mae_ratio:.4f}",
                        help="越接近 1.0 越好",
                    )
                with col_t3:
                    rating = s.get("quality_rating", "—")
                    st.info(f"**综合评级**\n\n{rating}")

                # 逐目标 TSTR 详情
                with st.expander("📋 逐目标 TSTR 详情", expanded=False):
                    tstr_rows = []
                    for target, v in results["tstr"].items():
                        if not target.startswith("_"):
                            tstr_rows.append({
                                "目标": target,
                                "TRTR R²": v.get("trtr_r2"),
                                "TSTR R²": v.get("tstr_r2"),
                                "R² 比值": v.get("r2_ratio"),
                                "TRTR MAE": v.get("trtr_mae"),
                                "TSTR MAE": v.get("tstr_mae"),
                            })
                    st.dataframe(
                        pd.DataFrame(tstr_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

            # 分布距离
            if "distributions" in results:
                st.divider()
                st.subheader("📏 分布距离（越小越好）")

                dist_data = []
                for col, d in results["distributions"].items():
                    dist_data.append({
                        "列": col,
                        "Wasserstein": d.get("wasserstein"),
                        "JS 散度": d.get("js_divergence"),
                    })
                dist_df = pd.DataFrame(dist_data)

                col_d1, col_d2 = st.columns([3, 2])
                with col_d1:
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
                with col_d2:
                    if len(dist_df) > 0:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        wass_vals = [
                            d.get("wasserstein", 0)
                            for d in results["distributions"].values()
                        ]
                        cols = list(results["distributions"].keys())
                        colors = ["#e74c3c" if v > 0.1 else "#27ae60" for v in wass_vals]
                        ax.barh(range(len(cols)), wass_vals, color=colors)
                        ax.set_yticks(range(len(cols)))
                        ax.set_yticklabels(cols, fontsize=8)
                        ax.set_xlabel("Wasserstein 距离")
                        ax.axvline(x=0.1, color="orange", linestyle="--",
                                  label="良好阈值 (0.1)")
                        ax.legend(fontsize=8)
                        ax.set_title("列级 Wasserstein 距离")
                        plt.tight_layout()
                        st.pyplot(fig)

            # 相关性保持
            if "correlation" in results and "error" not in results["correlation"]:
                st.divider()
                st.subheader("🔗 相关性保持")
                corr_info = results["correlation"]

                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.metric("Frobenius 范数", f"{corr_info['frobenius_norm']:.4f}")
                col_c2.metric("最大偏差", f"{corr_info['max_abs_diff']:.4f}")
                col_c3.metric("平均偏差", f"{corr_info['mean_abs_diff']:.4f}")

            # 单列分布对比图
            st.divider()
            st.subheader("🔬 单列分布对比")

            validator = st.session_state.gan_validator
            if validator is not None:
                all_numeric = data.select_dtypes(include=[np.number]).columns.tolist()
                compare_col = st.selectbox(
                    "选择要对比的列",
                    all_numeric,
                    key="dist_compare_col"
                )
                if compare_col:
                    try:
                        fig = validator.plot_distribution_comparison(compare_col)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"分布图绘制失败: {e}")
            else:
                st.caption("点击上方「运行 TSTR 全面验证」后可查看分布对比图")

# ═══════════════════════════════════════════════════════════════════
# Tab 3: 数据浏览
# ═══════════════════════════════════════════════════════════════════
with tab3:
    synth = st.session_state.synthetic_data
    data = _get_working_data()

    if synth is None:
        st.info("👈 请先在「🧬 训练生成器」中训练并生成数据")
    else:
        st.subheader("📋 合成数据浏览")

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            st.metric("合成数据量", len(synth))
        with col_f2:
            st.metric("列数", len(synth.columns))

        # 数据表
        st.dataframe(synth.head(100), use_container_width=True)

        # 统计对比：真实 vs 合成
        st.divider()
        st.subheader("📊 真实 vs 合成 统计对比")

        compare_col = st.selectbox(
            "选择对比列",
            synth.select_dtypes(include=[np.number]).columns.tolist(),
            key="browse_compare_col"
        )

        if compare_col and data is not None and compare_col in data.columns:
            col_box1, col_box2 = st.columns(2)
            with col_box1:
                st.caption(f"**真实数据** ({compare_col})")
                st.dataframe(
                    data[compare_col].describe().to_frame("值"),
                    use_container_width=True,
                )
            with col_box2:
                st.caption(f"**合成数据** ({compare_col})")
                st.dataframe(
                    synth[compare_col].describe().to_frame("值"),
                    use_container_width=True,
                )

        # 导出按钮
        st.divider()
        st.subheader("💾 导出合成数据")

        col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])
        with col_exp1:
            csv_bytes = synth.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 导出 CSV",
                csv_bytes,
                f"uhpc_synthetic_{len(synth)}.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_exp2:
            # Excel 导出
            excel_buffer = BytesIO()
            synth.to_excel(excel_buffer, index=False, engine="openpyxl")
            st.download_button(
                "📥 导出 Excel",
                excel_buffer.getvalue(),
                f"uhpc_synthetic_{len(synth)}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # 保存/加载模型
        with st.expander("💿 模型持久化 (保存/加载生成器)", expanded=False):
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models"
            )

            col_save, col_load = st.columns(2)
            with col_save:
                model_name = st.text_input("模型文件名", "ctgan_uhpc.pkl")
                if st.button("💾 保存生成器模型"):
                    gen = st.session_state.gan_generator
                    if gen is not None:
                        save_path = os.path.join(model_dir, model_name)
                        gen.save(save_path)
                        st.success(f"✅ 模型已保存到 {save_path}")
                    else:
                        st.warning("没有可保存的生成器")
            with col_load:
                if st.button("📂 加载已有模型"):
                    load_path = os.path.join(model_dir, "ctgan_uhpc.pkl")
                    if os.path.exists(load_path):
                        try:
                            gen = CTGANGenerator()
                            gen.load(load_path)
                            st.session_state.gan_generator = gen
                            # 用加载的模型生成数据
                            st.session_state.synthetic_data = gen.generate(
                                gen_count if 'gen_count' in locals() else 1000
                            )
                            st.success(f"✅ 模型已从 {load_path} 加载")
                        except Exception as e:
                            st.error(f"❌ 加载失败: {e}")
                    else:
                        st.warning(f"模型文件不存在: {load_path}")

# ═══════════════════════════════════════════════════════════════════
# Tab 4: 条件定向生成
# ═══════════════════════════════════════════════════════════════════
with tab4:
    gen = st.session_state.gan_generator
    data = _get_working_data()

    if gen is None:
        st.info("👈 请先在「🧬 训练生成器」中训练模型")
    else:
        st.subheader("🎯 条件定向生成")

        st.info("""
        **条件定向生成**允许您指定某些参数的值或范围，
        CTGAN 将在满足这些条件的约束下生成数据。
        例如：只生成「养护温度=90°C」的配比，或「水泥用量 700-900」的配比。
        """)

        # 构建条件界面
        st.subheader("设定条件约束")

        all_features = st.session_state.feature_columns
        available_features = [
            f for f in all_features
            if f in st.session_state.synthetic_data.columns
        ] if st.session_state.synthetic_data is not None else all_features

        # 每种类型建立不同条件
        conditions = {}

        # 数值型特征用范围滑块
        st.caption("**数值型条件**（留空 = 不限制）")
        num_cols = st.columns(3)

        numeric_features = [
            f for f in available_features
            if f != "curing_temp"  # 排除分类型
        ]

        for i, feat in enumerate(numeric_features):
            col_idx = i % 3
            with num_cols[col_idx]:
                enable = st.checkbox(
                    f"限制 {get_feature_label(feat)}",
                    key=f"cond_enable_{feat}"
                )
                if enable and data is not None and feat in data.columns:
                    feat_min = float(data[feat].min())
                    feat_max = float(data[feat].max())
                    val_range = st.slider(
                        get_feature_label(feat),
                        feat_min, feat_max,
                        (feat_min, feat_max),
                        key=f"cond_range_{feat}"
                    )
                    conditions[feat] = val_range

        # 分类型特征
        st.caption("**分类型条件**（留空 = 不限制）")
        if "curing_temp" in available_features:
            enable_cat = st.checkbox("限制 养护温度", key="cond_enable_curing_temp")
            if enable_cat:
                curing_val = st.selectbox(
                    "养护温度",
                    [20, 60, 90, 250],
                    format_func=lambda x: {
                        20: "常温(20°C)", 60: "蒸汽(60°C)",
                        90: "热水(90°C)", 250: "干热(250°C)"
                    }.get(x, str(x)),
                )
                conditions["curing_temp"] = curing_val

        # 生成按钮
        col_gen1, col_gen2 = st.columns([1, 2])
        with col_gen1:
            cond_count = st.number_input(
                "每个条件生成数量",
                min_value=10, max_value=1000, value=100, step=10,
            )

        with col_gen2:
            if conditions:
                st.caption(f"当前条件：{len(conditions)} 个")
                st.json(conditions)
            else:
                st.caption("未设定条件时将无条件生成")

        if st.button("🎯 执行条件生成", type="primary",
                     disabled=st.session_state.gan_is_fitting):
            with st.spinner("条件生成中..."):
                try:
                    if conditions:
                        cond_synth = gen.conditional_generate(
                            conditions=conditions,
                            num_rows=cond_count,
                        )
                    else:
                        cond_synth = gen.generate(num_rows=cond_count)

                    if cond_synth is not None and len(cond_synth) > 0:
                        st.success(
                            f"✅ 生成 {len(cond_synth)} 条满足条件的数据"
                        )
                        st.dataframe(cond_synth, use_container_width=True)

                        # 导出
                        csv_cond = cond_synth.to_csv(index=False).encode("utf-8-sig")
                        cond_desc = "_".join([
                            f"{k}={v}" for k, v in conditions.items()
                        ])[:80]
                        st.download_button(
                            "📥 导出条件生成数据 CSV",
                            csv_cond,
                            f"uhpc_conditional_{cond_desc}.csv",
                            "text/csv",
                        )
                    else:
                        st.warning("⚠️ 未生成任何满足条件的数据，请放宽条件")

                except Exception as e:
                    st.error(f"❌ 条件生成失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ─── 侧边栏 ──────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("📋 数据状态")
    d = _get_working_data()
    if d is not None:
        st.success(f"真实数据：{len(d)} 条")
    else:
        st.warning("无真实数据")

    synth = st.session_state.synthetic_data
    if synth is not None:
        st.success(f"合成数据：{len(synth)} 条")
        if d is not None:
            ratio = len(synth) / len(d)
            st.caption(f"合成/真实比例：{ratio:.1f}x")
    else:
        st.info("尚无合成数据")

    st.divider()

    st.subheader("📖 使用说明")
    st.markdown("""
    1. **加载数据**：先在「📊 数据管理」中上传或加载示例数据
    2. **训练 CTGAN**：设置 epochs 和生成数量，点击训练
    3. **质量验证**：运行 TSTR 验证，查看分布对比
    4. **导出数据**：下载 CSV/Excel 用于后续贝叶斯优化
    """)

    st.divider()

    st.subheader("⚡ 操作历史")
    if "gan_generator" in st.session_state and st.session_state.gan_generator is not None:
        meta = st.session_state.gan_generator.get_metadata_summary()
        if meta:
            st.caption(f"上次训练数据：{meta.get('n_rows', '?')} 条 × {meta.get('n_columns', '?')} 列")
    else:
        st.caption("尚无操作记录")
