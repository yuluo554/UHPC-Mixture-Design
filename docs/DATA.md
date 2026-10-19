# 数据说明 · 默认 37 行，可按需补齐到约 2200 行

> 结论先说：**仓库自带 `data/literature.csv`（37 行）就能跑通全部 6 个阶段。**
> `data/raw/` 有意不入库（体积、许可、可复现性），需要论文级规模时按本文档下载。

---

## 一、两种数据规模

| 规模 | 来源 | 需做什么 | 可用行数 | 有抗压强度 |
|---|---|---|---|---|
| **默认（克隆即用）** | `data/literature.csv`（仓库自带） | 什么都不用做 | 37 | 37 |
| **完整（论文级）** | `data/raw/`（本文档 + `scripts/fetch_data.py`） | 下载 8 个外部数据集 | 7010 | 3685 |

两种规模都能跑完 `01_数据管理 → 02_虚拟生成 → 03_高通量筛选 → 04_逆向优化 → 05_闭环反馈`。
区别只在**统计意义**：37 行够验证流程，不够训出可信模型。

```bash
# 查看清单（不下载）
python scripts/fetch_data.py --list

# 下载全部可直连的数据集，并逐个校验 SHA256
python scripts/fetch_data.py

# 只校验已有文件
python scripts/fetch_data.py --verify-only
```

脚本退出码：`0` 全部就绪 ｜ `1` 有文件校验失败 ｜ `2` 仍有需要手动下载的数据集。
**校验失败会报错退出，不会静默跳过。**

---

## 二、数据集清单

路径必须与 `data/column_mapping.py` 的 `DATASET_CONFIGS` 完全一致（包括文件名里的空格）。
下表 `key` 就是 `DATASET_CONFIGS` 的键名。

### 1. `literature` — 项目原生文献数据

| 项 | 值 |
|---|---|
| 文件 | `data/literature.csv` |
| 行数（列数） | 37 × 17 |
| 获取 | **仓库自带，无需下载** |
| 许可 | 项目自带（MIT 项目的一部分） |
| 状态 | 17 个标准列全覆盖，含成本与碳排放 |

### 2. `bolbolvand_comp` — Bolbolvand UHPFRC 抗压强度

| 项 | 值 |
|---|---|
| 文件 | `data/raw/bolbolvand_uhpc/uhpc_compressive_strength.csv` |
| 行数（列数） | 863 × 18 |
| 来源 | https://github.com/MiladBolbolvand/UHPC-Machine-Learning |
| 原始文件名 | `uhpc_compressive_strength_1.csv`（落盘时改名） |
| SHA256 | `b22c2daa8b83921ec886daadc6e3ca132590eb3da706f3456987a669b2569c40` |
| 获取 | `scripts/fetch_data.py --only bolbolvand_comp`（自动） |
| 备注 | 28 天 335 行，含钢纤维 108 行；强度 40.4–228.7 MPa |

### 3. `bolbolvand_flex` — Bolbolvand UHPFRC 抗弯强度

| 项 | 值 |
|---|---|
| 文件 | `data/raw/bolbolvand_uhpc/uhpc_flexural_strength.csv` |
| 行数（列数） | 321 × 18 |
| 来源 | https://github.com/MiladBolbolvand/UHPC-Machine-Learning |
| 原始文件名 | `uhpc_flexural_strength_1.csv` |
| SHA256 | `ccab44a420a2dadbf862593f12400decb8490214337e6e92f090442fefc8a7cf` |
| 获取 | `scripts/fetch_data.py --only bolbolvand_flex`（自动） |
| 备注 | 28 天 162 行，含钢纤维 153 行 |

### 4. `sean_jiao` — Sean Jiao UHPC 综合库

| 项 | 值 |
|---|---|
| 文件 | `data/raw/sean_jiao_uhpc/Bayesian Neural Networks/抗压强度数据.xlsx` |
| 行数（列数） | 799 × 17 |
| 来源 | https://github.com/sean-jiao/UHPC-prediction （`UHPC-pred.zip` 内的成员） |
| SHA256（成员） | `b52b06e48d5fbe0a93060edd09476004d3b1c20995b8e32b3472559739f9e510` |
| SHA256（整个 zip） | `9531a1773d64055ee5b8d946f7c454007f0a5410b923a9be8c8e67cfb458b943` |
| 获取 | `scripts/fetch_data.py --only sean_jiao`（自动下载 zip 并解出该成员） |
| 备注 | `Fi` 单位是 kg/m³，不是 vol%；28 天 302 行 |

### 5. `global_uhpc` — Mendeley 全球 UHPC 数据库

| 项 | 值 |
|---|---|
| 文件 | `data/raw/global_uhpc/Global Dataset of UHPC Mix Designs with Supplement/UHPC Dataset  (Version-2).xlsx` |
| 行数（列数） | 2188 × 92（3 层表头） |
| 来源 | https://data.mendeley.com/datasets/czb7ww5pkz/2 ｜ DOI `10.17632/czb7ww5pkz.2` ｜ CC BY 4.0 |
| SHA256 | `4274da215fd1884a8c9be46f1459c2e1dff44b52ad34b9a529fda75a5c077d3a` |
| 获取 | `scripts/fetch_data.py --only global_uhpc`（自动） |
| 备注 | **文件名里有两个空格**（`UHPC Dataset  (Version-2).xlsx`）且 sheet 名 `UHPC Dataset ` 末尾有空格；读取需要 `python-calamine` |

### 6. `scc_zenodo` — Zenodo 自密实混凝土数据库

| 项 | 值 |
|---|---|
| 文件 | `data/raw/scc_zenodo/SCC Dataset on Mixture Design & Key Properties - v1.csv` |
| 行数（列数） | 2504 × 115 |
| 来源 | https://zenodo.org/records/10569517 ｜ DOI `10.5281/zenodo.10569517` ｜ CC BY 4.0 |
| SHA256 | `71629f3ba531cc2d236abaf9921bc943a0e187fdc74710d2e57b8e401efb981a` |
| 获取 | `scripts/fetch_data.py --only scc_zenodo`（自动） |
| 备注 | 读取需 `encoding="latin-1"`、`sep=";"`、`skiprows=8`；**无强度、无钢纤维**，仅坍落扩展度可用 |

### 7. `steelfiber_mendeley` — Mendeley 钢纤维混凝土材料表征

| 项 | 值 |
|---|---|
| 文件 | `data/raw/steelfiber_mendeley/characterizacion and slump test.xlsx` |
| 内容 | 2 个 sheet：Slump 6 行 + 骨料 5 行 |
| 来源 | https://data.mendeley.com/datasets/22y4cxbsd8/3 ｜ DOI `10.17632/22y4cxbsd8.3` ｜ CC BY 4.0 |
| SHA256 | `06a89175f36119b1490977cff161c42dae4a803e59d33d056808e3c438685e5a` |
| 获取 | `scripts/fetch_data.py --only steelfiber_mendeley`（自动） |
| 备注 | 只有坍落度可映射，对建模价值很低 |

### 8. `umn_uhpc` — UMN UHPC 抗压 + UPV + 回弹

| 项 | 值 |
|---|---|
| 文件 | `data/raw/umn_uhpc/UHPC NDE Dataset.xlsx` |
| 行数（列数） | 主表 108 × 9（另有 4 个 UPV sheet） |
| 来源 | https://conservancy.umn.edu/handle/11299/272536 ｜ DOI `10.13020/h3m2-1h76` |
| SHA256 | `5ceb566051d2845a9169a6f873eb7be86475951892eb9113b64046ec1fd94f48` |
| 获取 | `scripts/fetch_data.py --only umn_uhpc`（自动） |
| 备注 | 龄期列含 `"12 hrs"` / `"28 day"` 文本，需 `parse_umn_age` 解析 |

### 9. `kaggle_steelfiber` — Kaggle 钢纤维混凝土 ANN

| 项 | 值 |
|---|---|
| 文件 | `data/raw/kaggle_steelfiber/database_en.xlsx` |
| 内容 | 3 个 sheet：抗压 521 / 抗拉 252 / 抗弯 278 |
| 来源 | https://www.kaggle.com/datasets/arthurmurta/steel-fiber-reinforced-concreteann |
| SHA256 | `209b41ad3f486c9f2084b2691e5d3480bcab340231b02759cff62d34c52c89fc` |
| 获取 | **手动**：Kaggle 需登录，`fetch_data.py` 只打印指令 |
| 备注 | 含粗骨料，非纯 UHPC；列名读取后需 `.str.strip()` |

### 10. `figshare_recycled` — Figshare 再生骨料钢纤维混凝土

| 项 | 值 |
|---|---|
| 文件 | `data/raw/figshare_recycled/Dataset.csv` |
| 行数（列数） | 405 × 10 |
| 来源 | https://tandf.figshare.com/articles/dataset/31376593 ｜ CC BY |
| SHA256 | `4deeb02f656752f1f0dcbb3c4b94cd72510408323c4648f40010d0f5486d9325` |
| 获取 | **手动**：Figshare 有 WAF 挑战，脚本无法直连 |
| 备注 | 龄期完全缺失；RHA 代用 silica_fume |

---

## 三、手动数据集的处理

两个手动数据集不会阻塞默认流程：缺它们时 `get_unified_dataset()` 会打印
`[警告] 加载 xxx 失败` 并继续，最终 `n_total` 仍来自其余数据集。

补齐后跑一次校验：

```bash
python scripts/fetch_data.py --verify-only
```

若 SHA256 不匹配，脚本以退出码 1 结束，并同时打印期望值与实际值——**这是有意为之**：
数据集是建模前提，允许「缺」，但不允许「悄悄用错版本」。

---

## 四、数据量与已知质量说明

| 数据集 | 声明行数 | 合入后行数 | 有抗压强度 |
|---|---|---|---|
| literature | 37 | 37 | 37 |
| bolbolvand_comp | 863 | 335 | 335 |
| bolbolvand_flex | 321 | 162 | 0（抗弯目标） |
| sean_jiao | 799 | 302 | 302 |
| global_uhpc | 2188 | 2188 | 2073 |
| scc_zenodo | 2504 | 2504 | 0 |
| steelfiber_mendeley | 11 | 14（2 个 sheet） | 0 |
| umn_uhpc | 108 | 12（仅 28 天） | 12 |
| kaggle_steelfiber | 1051 | 1051（3 个 sheet） | 521 |
| figshare_recycled | 405 | 405 | 405 |
| **合计** | — | **7010** | **3685** |

> 「合入后行数」= 经过 28 天龄期筛选、按 `DATASET_CONFIGS` 规则映射后进入统一数据集的行数，
> 由 `get_unified_dataset()` 实测得出（2026-10-19）。声明行数是原始文件行数。

已知限制（**不要误读为代码 bug**）：

1. `global_uhpc` 的强度是宽格式 8 个龄期列，melt 后未测龄期单元格曾是空白字符串；
   已统一转为 `NaN`（`data/column_mapping.py` 的 `_blank_to_nan` + `pd.to_numeric(errors="coerce")`）。
2. `scc_zenodo` / `steelfiber_mendeley` 没有强度数据，只贡献工作性；
   它们会让 `n_total` 变大但 `n_strength` 不变。
3. `kaggle_steelfiber` / `figshare_recycled` 含粗骨料或非 UHPC 配比，属于「宽口径」数据。
4. 纤维量单位不统一（vol% 与 kg/m³ 混用），`DATASET_CONFIGS` 的 `fiber_unit` 字段记录原始单位，
   代码目前**未统一换算**——跨数据集做纤维相关分析前需自行处理。

---

## 五、为什么不入库

- 单是 `global_uhpc` 就有 6.7 MB，10 个数据集合计约 12 MB（解压后更多）
- 各数据集有独立许可（CC BY 4.0 / 公开但非 CC0），转发需保留出处
- 数据可复现：URL + SHA256 + 脚本足以重建，Clone 体积保持轻量

`.gitignore` 已忽略 `data/raw/`，因此 `git add -A` 不会把数据吞进仓库。
