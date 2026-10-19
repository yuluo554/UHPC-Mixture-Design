# CHANGELOG

本项目采用 `vMAJOR.MINOR.PATCH` 版本号，`core/__init__.py` 的 `__version__` 是唯一真相源。

---

## [2.0.0] — 2026-10-19

「装完能跑」收口版本：把 v2 内核、依赖、数据链、文档、CI 一次性对齐。

### 修复（四批 bug 修复的合并落点）

- **critical（4 个）**：`core/config.py` 参数读取、`core/data_manager.py` 清洗边界、
  逆向优化页约束、`core/benchmark.py` 引入。
- **high（10 个）**：贝叶斯优化方向归一化、闭环反馈指标、GAN 训练与保存、
  进化优化约束编码、高通量筛选页。
- **medium（12 个）**：首页与各页面状态、SHAP、TOPSIS、反馈管理器。
- **low（9 个）**：文案与边界处理，并恢复 v2.0 缺失文件
  （`core/data_bridge.py`、`core/dataset.py`、`core/domain_features.py`、
  `core/acquisition_functions.py`、`core/virtual_optimizer.py`、`data/column_mapping.py`、
  `tests/` 11 个文件）。

### 本次迭代（v2.0.0 收口）

**新增能力**

- 虚拟空间优化（`core/virtual_optimizer.py`）、UHPCBenchmark 对比工具（`core/benchmark.py`）
- 采集函数扩展至 EI/UCB/PoI/AugmentedEI/Thompson（`core/acquisition_functions.py`）
- 领域特征变换 `AppendUHPCFeatures`（水胶比 / 砂胶比 / 减水剂胶凝比）

**依赖与可复现性（批次 1）**

- `requirements.txt` 锁 `sdv>=1.10,<2.0`（sdv 2.0 移除了 `sdv.evaluation.single_table`，
  会让 `core/gan_generator.py` 导入即崩、6 个 GAN 测试失败）
- 补齐此前缺失的 4 个包：`python-calamine`（读唯一在仓的 xlsx）、`numba`（pymoo AGEMOEA 必需）、
  显式 `torch`、显式 `scipy`；`xgboost` / `lightgbm` 标注为可选增强
- 数据策略：`data/raw/` **不入库**，改为 `docs/DATA.md`（来源 URL + SHA256 + 落盘路径）
  \+ `scripts/fetch_data.py`（下载并校验 SHA256，失败报错）+ `.gitignore`
- 修 `global_uhpc` 空白字符串导致的数据缺陷：melt 后的强度列曾是 `object`，
  `notna()` 把 9641 个空白串算作非空；现在统一转 `NaN` 并数值化
  （实测 3915 个真实强度值，4.1–298.0 MPa）

**文档与代码一致性（批次 2）**

- README 徽章与文字改为实测数字：**237 个用例**（原写 95/95；v2 实际 233 个，
  本次新增 4 个列映射回归测试）
- README 项目结构补齐真实文件树；性能基准表明确标注为开发环境参考值
- 修「假降级」：
  - `core/domain_features.py` 顶层 `import torch` 加保护（缺依赖时模块仍可 import）
  - `core/multi_objective.py` 的 AGEMOEA 单独 try/except `Exception`
    （pymoo 缺 numba 时抛的是 `Exception` 而非 `ImportError`）
  - `core/bayesian_optimizer.py` 在缺 torch 时给出可操作的报错
- Python 版本三处统一 3.12（`runtime.txt` / `Dockerfile` / README）

**清洁度与工程化（批次 3）**

- 删除无代码引用的原作用遗留文件：`Readme`、`Data_for_Opt.csv`（4937 行 ECC 数据）
- 新增 GitHub Actions CI（导入面 / 无数据不崩 / DATA.md 覆盖 / 清单路径一致 /
  全量测试 / README 数字一致 / Python 版本一致 / Docker 构建）
- 版本号统一为 `2.0.0`

### 是否需要重跑历史结果

**需要。** 优化器（贝叶斯 + 进化）、数据桥、虚拟生成与反馈链在本轮都有改动，
旧版本导出的 Pareto 集 / 推荐配比 / 已训模型建议重新生成。

### 依赖版本（实测可跑组合，Python 3.12）

| 包 | 版本 | 包 | 版本 |
|---|---|---|---|
| pandas | 2.3.3 | botorch | 0.18.1 |
| numpy | 2.5.4 | torch | 2.14.1 |
| scipy | 1.18.1 | shap | 0.54.1 |
| scikit-learn | 1.9.1 | sdv | 1.38.5（`<2.0`） |
| pymoo | 0.6.2 | python-calamine | 0.9.0 |
| numba | 0.68.0 | streamlit | 1.66.0 |

### 已知限制

1. **默认数据只有 37 行**（`data/literature.csv`）。仓库可克隆即跑，但要论文级建模需按
   `docs/DATA.md` 下载 `data/raw/`；Kaggle 与 Figshare 两个数据集需手动下载。
   37 行能验证流程，不能支撑可信模型。
2. **决策支持 ≠ 施工配比**：本平台输出为候选方案与趋势，不能替代试配与规范校核。
3. `scc_zenodo` / `steelfiber_mendeley` 无强度数据，只贡献工作性，会抬高 `n_total`
   但不增加可用强度样本。
4. `kaggle_steelfiber` / `figshare_recycled` 含粗骨料或非 UHPC 配比，属宽口径数据。
5. 纤维量单位在数据集间不统一（vol% 与 kg/m³ 混用），代码记录原始单位但**未统一换算**。

---

## [0.1.0] — 初始提交

Streamlit 单页版 UHPC 配比设计平台原型（`main` 上最早的提交），
仅含基础上传、清洗、可视化与简易优化。