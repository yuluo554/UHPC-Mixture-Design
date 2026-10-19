# 🧱 UHPC-AutoDesign

**钢纤维 UHPC 智能配比设计全流程自动化平台**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-237%2F237-brightgreen)](tests/)

基于 **AI 与材料基因组工程（MGI）** 的钢纤维 UHPC 配比智能设计与多目标优化平台。将传统"试错法"升级为 **数据驱动 + AI 驱动** 的 6 阶段全流程自动化。

> **开箱即用：仓库自带 37 行文献数据，克隆后即可跑通全部 6 个阶段。**
> 要复现论文级规模（约 2200 组配比 / 3700 条强度记录），按 [`docs/DATA.md`](docs/DATA.md)
> 运行 `python scripts/fetch_data.py` 下载数据到 `data/raw/`（该目录不入库）。
> 两者的区别只在统计意义——37 行够验证流程，不够训出可信模型。

---

> ### ⚠️ 衍生项目说明
>
> 本项目基于 [**Khuong Le Nguyen**](mailto:lekhuong.jmi@gmail.com)（University of Transport Technology, Vietnam / University of Canberra, Australia）的原作 **Concrete-Mixture-Design-Framework** 进行深度改造与扩展。
>
> - **原作**：ECC（工程水泥基复合材料）配比优化 Web 应用 — [GitHub](https://github.com/lekhuong/Concrete-Mixture-Design-Framework)
> - **本衍生版**：在原作基础上将材料体系转向钢纤维 UHPC，全面重写核心模块（数据处理 → 虚拟生成 → 贝叶斯优化 → 进化优化 → TOPSIS 决策 → 闭环反馈），扩展为 6 阶段全流程自动化平台。
>
> **致谢**：感谢原作者的开源贡献，使本项目得以在此基础上进一步发展。

---

## 📖 目录

- [概述](#-概述)
- [功能特性](#-功能特性)
- [技术栈](#-技术栈)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [使用指南](#-使用指南)
- [API 参考](#-api-参考)
- [常见问题](#-常见问题)
- [性能基准](#-性能基准)
- [引用参考](#-引用参考)
- [版本](#-版本)
- [许可证](#-许可证)

---

## 🏷 版本

| 版本 | 状态 | 关键变化 |
|---|---|---|
| **v2.0.0** | **当前** | 贝叶斯优化内核（BoTorch GP/EHVI）、虚拟空间优化、UHPCBenchmark、237 个测试；依赖清单补全（`python-calamine`/`numba`/`torch`，锁 `sdv<2`）；数据走「轻仓库 + `docs/DATA.md` + `scripts/fetch_data.py`」；文档与实测对齐 |
| v0.1.0 | 历史 | 初始平台（`main` 上最早的提交） |

> 版本号单一真相源：`core/__init__.py` 的 `__version__`。

---

## 🎯 概述

### 6 阶段全流程自动化

```
① 材料基因识别  →  ② 数据库构建  →  ③ 虚拟生成  →  ④ 高通量筛选  →  ⑤ 逆向优化  →  ⑥ 闭环反馈
   参数体系构建       CSV 上传/MICE     CTGAN 合成      贝叶斯优化       NSGA-II          实验数据导入
   特征工程          SHAP 分析          TSTR 验证      GP 代理模型      TOPSIS 优选       增量/全量更新
                     异常检测                            EHVI 采集        Pareto 前沿       日志追踪
```

### UHPC 参数体系

| 类别 | 参数 | 范围 |
|------|------|------|
| **组成参数 (6)** | 水泥、硅灰、石英粉、石英砂、减水剂、水 | 见 YAML 模板 |
| **纤维参数 (4)** | 体积率、直径、长度、长径比 | 0.5-3.0% / 0.12-0.6mm 等 |
| **养护参数 (2)** | 温度、时间 | 20/60/90/250°C, 24-168h |
| **优化目标 (5)** | 抗压强度 ↑ / 抗弯强度 ↑ / 坍落度 ⊙ / 成本 ↓ / 碳排放 ↓ | — |

---

## ✨ 功能特性

### 📊 数据管理 (`pages/01_数据管理.py`)

- CSV 数据上传 + 一键加载文献基准数据（37 条 UHPC 配比）
- 自动异常检测（Isolation Forest，可调 contamination）
- MICE 多重缺失插补（IterativeImputer）
- 交互式可视化：箱线图 / 相关性热力图 / 散点矩阵 / 分布直方图 / Q-Q 图
- SHAP 特征重要性分析：summary / beeswarm / dependence / waterfall 四图

### 🧬 虚拟生成 (`pages/02_虚拟生成.py`)

- **CTGAN**（主）+ **TVAE**（备选）— 基于 SDV 的合成数据生成
- TSTR（Train-on-Synthetic-Test-on-Real）完整验证：
  - 统计量对比（均值差异%）
  - 分布距离（Wasserstein + Jensen-Shannon）
  - 相关性保持（Frobenius 范数）
  - TSTR/TRTR R² 比值
- 分布对比图（直方图 + KDE + 箱线图）
- 条件定向生成（按养护温度等分类特征筛选）
- 模型持久化（保存 / 加载生成器）
- CSV / Excel 导出

### 🔬 高通量筛选 (`pages/03_高通量筛选.py`)

- **多目标贝叶斯优化**（BoTorch 引擎）
- 代理模型：**GP (Gaussian Process)** / **RF (Random Forest)** / **Ensemble (GP+RF)**
- 采集函数：**qNEHVI**（考虑噪声的超体积改进）/ **EHVI**
- UHPC 物理约束自动过滤（水胶比 ∈ [0.14, 0.24]）
- 实时监控：Hypervolume 收敛曲线 + Pareto 前沿规模 + 累计评估点数
- 2D/3D Pareto 前沿 Plotly 交互图 + 平行坐标图
- 结果导出：Pareto 集 CSV + 收敛历史 + 优化器状态持久化

### 🎯 逆向优化 (`pages/04_逆向优化.py`)

- **pymoo 多目标进化优化**：NSGA-II / NSGA-III / AGE-MOEA
- UHPC 嵌入约束：水胶比 + 体积和约束（自动编码/解码 curing_temp）
- 可选 RF 代理模型加速适应度评估
- **TOPSIS 多准则决策**（纯 NumPy 实现）：
  - 交互式权重滑块（5 目标）
  - Top-N 推荐表（含特征值）
  - 雷达图 + 平行坐标图 + 得分柱状图
- 支持随机权重自动搜索（RAN 模式）
- 详细计算过程展示（向量归一化 → PIS/NIS → 欧氏距离 → 贴近度）

### 🔄 闭环反馈 (`pages/05_闭环反馈.py`)

- **实验数据导入**：CSV 上传 + 手动录入表单（12 特征 + 5 目标）
- **预测对比**：R² / MAE / RMSE / MAPE(%) / MaxError 五指标
- 可视化：预测 vs 实测散点图 + 残差分布直方图 + 误差柱状图
- **模型更新**：增量更新 + 全量重训练，before/after 性能对比
- **闭环日志**：自动追踪每轮反馈的来源、样本数、指标变化
- 完整状态持久化（`save_state` / `load_state`）

---

## 🛠 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **Web 框架** | Streamlit ≥1.28 | 多页交互界面 |
| **数据处理** | Pandas, NumPy, Scikit-learn | 清洗/插补/特征工程 |
| **合成数据** | SDV (CTGAN/TVAE) | 虚拟配比生成 |
| **贝叶斯优化** | BoTorch + GPyTorch | 多目标代理模型优化 |
| **进化优化** | pymoo (NSGA-II/III, AGE-MOEA) | 多目标 Pareto 搜索 |
| **多准则决策** | TOPSIS（纯 NumPy） | 方案优选排序 |
| **可解释性** | SHAP | 特征重要性分析 |
| **可视化** | Plotly + Matplotlib + Seaborn | 交互图表 + 静态图 |
| **容器化** | Docker + docker-compose | 一键部署 |

---

## 🚀 快速开始

### 方式一：Docker（推荐）

```bash
# 1. 克隆项目
git clone <repo-url> && cd UHPC-AutoDesign

# 2. 构建并启动
docker compose up -d

# 3. 访问
# 浏览器打开 http://localhost:8501

# 4. 停止
docker compose down
```

### 方式二：本地 Python

**前置要求：** Python 3.12（与 `runtime.txt`、`Dockerfile` 保持一致）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python -m streamlit run app.py

# 3. 访问
# 浏览器打开 http://localhost:8501
```

> ⚠️ **注意：** 如果使用 Python 3.14，部分包（shap/numba）可能存在兼容性问题。 推荐 Python 3.12（Docker 镜像已配置）。

### 数据：默认 37 行，要完整规模需要额外下载

克隆下来就能跑，因为 `data/literature.csv`（37 行 × 17 列）随仓库提供。
要做论文级建模，再执行：

```bash
python scripts/fetch_data.py            # 自动下载 8 个数据集并校验 SHA256
python scripts/fetch_data.py --list     # 只看清单
```

`data/raw/` 不入库（体积 + 独立许可），来源 URL、SHA256、落盘路径见
[`docs/DATA.md`](docs/DATA.md)。Kaggle 与 Figshare 两个数据集需要手动下载，脚本会打印指令。

### 运行测试

```bash
# 全量测试（237 个用例，约 13 分钟）
python -m pytest tests/ -q

# 快速冒烟测试（跳过耗时用例）
python -m pytest tests/ -q -k "test_load or test_rf or test_config"

# 只看单个文件
python -m pytest tests/test_pipeline.py -v
```

### 可选依赖（不装也能跑，功能降级）

`requirements.txt` 已包含全部必需依赖；下面这些是**可选增强**，缺失时代码走降级分支：

| 包 | 缺失时的能力差异 | 代码位置 |
|---|---|---|
| `xgboost` | `ModelTrainer` 不能选 XGBoost 模型（RF/GB 仍可用） | `core/model_trainer.py`（`HAS_XGBOOST`） |
| `lightgbm` | 同上，LightGBM 不可选 | `core/model_trainer.py`（`HAS_LIGHTGBM`） |
| `numba` | `algorithm="agemoea"` 报 ImportError；NSGA-II/III 不受影响（已列入 requirements） | `core/multi_objective.py` |
| `torch` / `botorch` | 贝叶斯优化的 GP/Ensemble 路径不可用，自动回退 RF；`AppendUHPCFeatures` 会明确报错 | `core/bayesian_optimizer.py`、`core/domain_features.py` |

---

## 📁 项目结构

```
UHPC-AutoDesign/
├── app.py                          # Streamlit 入口 + 首页仪表盘
├── pages/
│   ├── 01_数据管理.py               # 数据上传/清洗/可视化/SHAP
│   ├── 02_虚拟生成.py               # CTGAN 合成数据生成
│   ├── 03_高通量筛选.py             # 贝叶斯优化筛选
│   ├── 04_逆向优化.py               # 多目标进化优化 + TOPSIS
│   └── 05_闭环反馈.py               # 实验数据导入 + 模型更新
├── core/
│   ├── config.py                   # YAML 参数模板加载（单一真相源）
│   ├── data_manager.py             # 数据加载/清洗/插补/归一化
│   ├── dataset.py                  # UHPCDataset dataclass（数据契约）
│   ├── data_bridge.py              # 统一数据入口（Streamlit 页面唯一来源）
│   ├── model_trainer.py            # RF/GB/XGB/LGBM + 多目标训练器
│   ├── shap_analyzer.py            # SHAP 特征重要性 + 可视化
│   ├── gan_generator.py            # CTGAN/TVAE 生成器 + TSTR 验证
│   ├── bayesian_optimizer.py       # 贝叶斯多目标优化（BoTorch GP/EHVI）
│   ├── acquisition_functions.py    # 采集函数（EI/UCB/PoI/Thompson，纯 NumPy）
│   ├── domain_features.py          # BoTorch 输入变换（水胶比等导出特征）
│   ├── multi_objective.py          # pymoo 进化优化（NSGA-II/III/AGE-MOEA）
│   ├── topsis_selector.py          # TOPSIS 多准则决策（纯 NumPy）
│   ├── virtual_optimizer.py        # 虚拟空间优化（CTGAN + BO 融合）
│   ├── feedback_manager.py         # 闭环反馈（导入/对比/更新/日志）
│   └── benchmark.py                # UHPCBenchmark：Trail/Count/Compare 对比
├── data/
│   ├── literature.csv              # 37 条 UHPC 文献基准数据（随仓库提供）
│   ├── column_mapping.py           # 10 个外部数据集 → 17 标准列 映射（876 行）
│   ├── raw/                        # 外部数据集（.gitignore 忽略，见 docs/DATA.md）
│   ├── generated/                  # CTGAN 生成数据输出（首次运行时自动创建）
│   └── experiments/                # 闭环实验数据导入（首次运行时自动创建）
├── docs/
│   └── DATA.md                     # 数据集来源 URL / SHA256 / 落盘路径
├── scripts/
│   └── fetch_data.py               # 按 docs/DATA.md 下载并校验 SHA256
├── models/                         # 训练好的模型持久化（首次运行时自动创建）
├── templates/
│   └── uhpc_params.yaml            # UHPC 参数模板（12特征+5目标+约束）
├── tests/                          # 237 个用例，12 个文件（含 95 个端到端）
│   ├── conftest.py
│   ├── test_pipeline.py            # 95 个端到端集成测试
│   ├── test_column_mapping.py      # 列映射 / 空白串 → NaN（4 个）
│   └── test_{acquisition,bayesian_optimizer,config,data_bridge,data_manager,...
│       e2e_optimization,feedback,model_trainer,multi_objective,virtual_optimizer}.py
├── Dockerfile                      # Docker 镜像定义（python:3.12-slim）
├── docker-compose.yml              # Docker 编排
├── runtime.txt                     # 目标 Python 版本（3.12）
├── requirements.txt                # Python 依赖清单
└── README.md                       # 本文件
```

> `models/`、`data/generated/`、`data/experiments/` 三个目录**首次运行相关页面时自动创建**，
> 因此仓库里不预置（内容均为运行产物，已在 `.gitignore` 中忽略）。

---

## 📘 使用指南

### Step 1：数据管理

1. 点击侧边栏 **📊 数据管理**
2. 上传您的 UHPC 配比 CSV 文件，或点击「加载示例数据」
3. 切换到「🧹 数据清洗」，调整 contamination 滑块，点击「开始清洗」
4. 切换到「📈 可视化」，探索数据分布和相关性
5. 切换到「🔍 SHAP 分析」，选择目标变量 → 训练模型 → 查看特征重要性

### Step 2：虚拟生成

1. 点击侧边栏 **🧬 虚拟生成**
2. 设置生成器类型（CTGAN 推荐）、epochs、batch_size
3. 点击「开始训练」，观察进度条
4. 切换到「📊 质量验证」，查看 TSTR 指标（R² 比值、Wasserstein 距离）
5. 切换到「📋 数据浏览」，查看合成数据统计，导出为 CSV/Excel

### Step 3：高通量筛选

1. 点击侧边栏 **🔬 高通量筛选**
2. 选择优化目标（多选，注意方向图标 ↑↓⊙）
3. 选择代理模型（GP 推荐用于精确优化，RF 用于快速探索）
4. 设置迭代次数，点击「开始优化」
5. 切换到「📊 Pareto 前沿」，查看 2D/3D 前沿图
6. 切换到「📋 结果导出」，下载 Pareto 集

### Step 4：逆向优化

1. 点击侧边栏 **🎯 逆向优化**
2. 选择优化目标和进化算法（NSGA-II 推荐）
3. 设置种群大小和代数
4. 调整 TOPSIS 权重滑块（和为 1.0）
5. 点击「开始优化」
6. 切换到「🏆 TOPSIS 优选」，查看 Top-5 推荐配比
7. 查看雷达图和平行坐标图辅助决策

### Step 5：闭环反馈

1. 点击侧边栏 **🔄 闭环反馈**
2. 上传实验数据 CSV 或手动录入新的实验结果
3. 切换到「📊 预测对比」，查看模型预测 vs 实验实测
4. 切换到「🔄 模型更新」，选择增量/全量更新，对比 before/after
5. 切换到「📋 闭环日志」，查看所有反馈记录

---

## 📚 API 参考

所有核心模块通过 `core/` 目录的 Python 类暴露，可在脚本中直接调用：

```python
from core.config import get_all_feature_names, get_all_target_names
from core.data_manager import DataManager
from core.model_trainer import ModelTrainer, MultiTargetTrainer
from core.gan_generator import CTGANGenerator, TVAEGenerator, TSTRValidator
from core.bayesian_optimizer import BayesianOptimizer, create_bayesian_optimizer
from core.multi_objective import MultiObjectiveOptimizer, create_multi_objective_optimizer
from core.topsis_selector import TOPSISSelector, create_topsis_selector
from core.feedback_manager import FeedbackManager
```

### 关键 API 示例

```python
# 贝叶斯优化
opt = create_bayesian_optimizer()
results = opt.optimize(X, Y, n_iter=50)
pareto_df = opt.get_pareto_df()   # Pareto 前沿 DataFrame

# 进化优化 + TOPSIS 推荐
moo = create_multi_objective_optimizer(algorithm="nsga2", pop_size=100)
results = moo.optimize(n_generations=50)
selector = create_topsis_selector(results["pareto_Y"])
top5 = selector.get_top_n(5)      # Top-5 推荐配比

# 闭环反馈
fm = FeedbackManager()
fm.import_experiment_data(exp_df, features, targets)
preds = fm.predict_on_experiments()
comp = fm.compare(measured, preds.values, targets)
fm.incremental_update("random_forest")
```

---

## ❓ 常见问题

### Q1：启动后页面空白？

检查控制台错误：`python -m streamlit run app.py 2>&1 | head -20`。确保 `pages/` 目录下所有页面文件存在且可导入。

### Q2：`pip install` 后包找不到？

使用 `python -m pip install <package>` 代替直接 `pip install`，避免多 Python 环境混用。验证：`python -c "import sys; print(sys.executable)"`

### Q3：SHAP 分析卡住不动？

调低「SHAP 计算样本数」滑块（默认 200），或先切换到 RandomForest + TreeExplainer（速度快 100 倍）。

### Q4：CTGAN 训练很慢？

37 条数据 × 50 epochs ≈ 3s 是正常的。如需 500 epochs，预计 30s~2min。可以在训练设置中调整参数。

### Q5：Docker 构建失败？

确保 Docker Desktop 已安装并运行。如果在 Windows 上遇到路径问题，确保项目在 WSL2 文件系统中或使用 Git Bash。

### Q6：贝叶斯优化报 nan 或 inf？

可能是数据中存在异常值。先经过数据清洗步骤（contamination=0.05），或减少目标数量（≤3 个目标更稳定）。

---

## ⏱ 性能基准

> 下表是**开发环境参考值**（CPU、默认 37 行文献数据），不是 CI 实测；用于判断数量级。

| 操作 | 数据量 | 耗时（参考） |
|------|--------|------------|
| DataManager.clean() | 37 行 | <0.1s |
| RF train + CV (5 fold) | 35 行 | ~1s |
| SHAP TreeExplainer | 35 行 | <0.5s |
| CTGAN train (50 epochs) | 37 行 | ~3s |
| TSTR validate (5 targets) | 137 行 | ~2s |
| BoTorch GP 1 step | 37 行 | ~2-5s |
| GP 50 iterations | 37 行 | ~100-200s |
| RF 50 iterations | 37 行 | ~25s |
| NSGA-II (50 pop × 10 gen) | — | ~15s |
| TOPSIS (20×3 排序) | 20 行 | <0.01s |

---

## 📖 引用参考

### 参考论文

- **BOxCrete / SustainableConcrete**: [arxiv.org/abs/2603.21525](https://arxiv.org/abs/2603.21525) — 高斯过程 + 贝叶斯优化在混凝土配比中的应用
- **MultiBgolearn**: [arxiv.org/abs/2601.06820](https://arxiv.org/abs/2601.06820) — 多目标贝叶斯全局优化方法
- **MGI for UHPC**: 基于 AI 与材料基因组工程（MGI）的钢纤维 UHPC 配比智能设计

### 原作信息

本项目衍生于 [**Concrete-Mixture-Design-Framework**](https://github.com/lekhuong/Concrete-Mixture-Design-Framework)，原作详情如下：

| 项目 | 信息 |
|------|------|
| **原作作者** | Khuong Le Nguyen |
| **所属机构** | University of Transport Technology, Vietnam / University of Canberra, Australia |
| **原作仓库** | [github.com/lekhuong/Concrete-Mixture-Design-Framework](https://github.com/lekhuong/Concrete-Mixture-Design-Framework) |
| **原作许可** | CC0 1.0 Universal |

### 本衍生版与原作的主要差异

| 维度 | 原作 (ECC) | 本衍生版 (UHPC) |
|------|-----------|----------------|
| **材料体系** | ECC 工程水泥基复合材料 | 钢纤维 UHPC 超高性能混凝土 |
| **特征数** | 10 个输入参数 | 12 个 UHPC 专用参数 |
| **目标数** | 3 个（强度/CO₂/成本） | 5 个（抗压/抗弯/坍落度/成本/碳排放） |
| **优化引擎** | Platypus (单一算法) | pymoo (NSGA-II/III/AGE-MOEA) |
| **虚拟生成** | ❌ 无 | ✅ CTGAN + TVAE + TSTR 验证 |
| **贝叶斯优化** | ❌ 无 | ✅ BoTorch GP + qNEHVI/EHVI |
| **多准则决策** | ❌ 无 | ✅ TOPSIS 交互式权重优选 |
| **闭环反馈** | ❌ 无 | ✅ 实验导入 → 预测对比 → 模型更新 → 日志追踪 |
| **部署** | 单一 Streamlit | Docker + docker-compose 一键部署 |

### 致谢

由衷感谢原作者 **[Khuong Le Nguyen](mailto:lekhuong.jmi@gmail.com)** 的开源贡献。本衍生版站在巨人的肩膀上，在原作框架基础上进行了材料体系转换与功能扩展。原作采用 CC0 1.0 协议开放源码，本衍生版遵循 MIT 协议开源。

---

## 📄 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

> 原作 [Concrete-Mixture-Design-Framework](https://github.com/lekhuong/Concrete-Mixture-Design-Framework) 以 CC0 1.0 Universal 协议发布（公有领域）。本衍生版在原作基础上进行深度改造与扩展，新增代码以 MIT 协议发布。CC0 协议允许在遵守其条款的前提下进行再许可和重新发布。

---

<p align="center">
  <b>UHPC-AutoDesign</b> — 让 AI 驱动 UHPC 配比设计
</p>
