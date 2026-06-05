# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign — UHPC 智能配比设计全流程自动化平台

基于 AI 与 MGI 的钢纤维 UHPC 配比智能设计与多目标优化研究
"""

import streamlit as st

# ─── 页面配置 ────────────────────────────────────────────────────
st.set_page_config(
    page_title="UHPC-AutoDesign",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 侧边栏导航 ──────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/concrete-mixer.png",
        width=64,
    )

    st.title("🧱 UHPC-AutoDesign")
    st.caption("钢纤维 UHPC 智能配比设计平台")
    st.divider()

    st.markdown("""
    ### 📋 功能模块
    """)

    st.markdown("""
    | 页面 | 功能 |
    |------|------|
    | 📊 数据管理 | 上传/清洗/可视化/SHAP |
    | 🧬 虚拟生成 | CTGAN 合成配比 |
    | 🔬 高通量筛选 | 贝叶斯优化 |
    | 🎯 逆向优化 | 多目标 + TOPSIS |
    | 🔄 闭环反馈 | 实验数据回流 |

    > 使用 Streamlit 原生多页机制：
    > 将页面文件放入 `pages/` 目录
    """)

    st.divider()
    st.caption("v0.1.0 | UHPC-AutoDesign")

# ─── 主页内容 ────────────────────────────────────────────────────
st.title("🧱 UHPC 智能配比设计全流程自动化平台")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 数据管理", "就绪", "CSV 上传/清洗/SHAP")
with col2:
    st.metric("🧬 虚拟生成", "CTGAN", "合成数据扩增")
with col3:
    st.metric("🎯 多目标优化", "NSGA-II", "pymoo 引擎")

st.divider()

# ─── 流程图 ──────────────────────────────────────────────────────
st.markdown("""
### 🔄 工作流程

```
② 数据库构建  →  ③ 虚拟生成  →  ④ 高通量筛选  →  ⑤ 逆向优化  →  ⑥ 闭环反馈
   CSV上传        CTGAN合成      贝叶斯优化      NSGA-II        实验导入
   MICE填充       TSTR验证       GP代理模型     TOPSIS优选      增量更新
   SHAP分析       条件生成       EI采集函数     Pareto前沿      日志追踪
```
""")

st.divider()

# ─── UHPC 参数体系概览 ───────────────────────────────────────────
st.markdown("### 📐 UHPC 参数体系")

tab1, tab2, tab3, tab4 = st.tabs(["组成参数", "钢纤维参数", "养护参数", "优化目标"])

with tab1:
    comp_data = [
        {"参数": "水泥 (Cement)", "范围": "600-1000 kg/m³", "步长": "10"},
        {"参数": "硅灰 (Silica Fume)", "范围": "50-250 kg/m³", "步长": "5"},
        {"参数": "石英粉 (Quartz Powder)", "范围": "0-400 kg/m³", "步长": "10"},
        {"参数": "石英砂 (Quartz Sand)", "范围": "800-1200 kg/m³", "步长": "10"},
        {"参数": "减水剂 (SP)", "范围": "5-40 kg/m³", "步长": "0.5"},
        {"参数": "水 (Water)", "范围": "140-220 kg/m³", "步长": "1"},
    ]
    st.dataframe(comp_data, use_container_width=True, hide_index=True)

with tab2:
    fiber_data = [
        {"参数": "纤维体积率", "范围": "0.5-3.0%", "步长": "0.1"},
        {"参数": "纤维直径", "范围": "0.12-0.6 mm", "步长": "0.01"},
        {"参数": "纤维长度", "范围": "6-20 mm", "步长": "1"},
        {"参数": "长径比", "范围": "40-100", "步长": "5"},
    ]
    st.dataframe(fiber_data, use_container_width=True, hide_index=True)

with tab3:
    curing_data = [
        {"参数": "养护温度", "选项": "20°C / 60°C / 90°C / 250°C"},
        {"参数": "养护时间", "范围": "24-168 h"},
    ]
    st.dataframe(curing_data, use_container_width=True, hide_index=True)

with tab4:
    target_data = [
        {"目标": "28d抗压强度", "方向": "最大化", "范围": "120-250 MPa"},
        {"目标": "28d抗弯强度", "方向": "最大化", "范围": "20-60 MPa"},
        {"目标": "坍落扩展度", "方向": "目标区间", "范围": "180-250 mm"},
        {"目标": "材料成本", "方向": "最小化", "单位": "¥/m³"},
        {"目标": "碳排放", "方向": "最小化", "单位": "kg CO₂/m³"},
    ]
    st.dataframe(target_data, use_container_width=True, hide_index=True)

st.divider()

# ─── 快速开始 ────────────────────────────────────────────────────
st.markdown("""
### 🚀 快速开始

1. **数据管理** → 上传您的 UHPC 配比数据（CSV），进行清洗和可视化
2. **虚拟生成** → 使用 CTGAN 生成虚拟配比，扩增数据集
3. **高通量筛选** → 贝叶斯优化快速发现高潜力配比区域
4. **逆向优化** → 多目标 NSGA-II 优化 + TOPSIS 方案推荐
5. **闭环反馈** → 导入实验结果，自动更新模型

> 💡 提示：每个模块通过左侧导航栏访问，数据通过 `st.session_state` 在模块间共享。
""")
