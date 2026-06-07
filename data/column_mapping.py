# -*- coding: utf-8 -*-
"""
数据集列名 → 项目标准列名 统一映射模块

每个数据集一个 dict，key=原始列名，value=标准列名。
未覆盖的列用 None 值占位表示缺失。

模块位置：data/column_mapping.py
路径约定：所有路径相对于 data/ 目录（本模块所在目录）。
"""

import os
import pandas as pd
import numpy as np

# ─── 17 个标准列名 ─────────────────────────────────────────────────

STD_COLS = [
    "cement",
    "silica_fume",
    "quartz_powder",
    "quartz_sand",
    "superplasticizer",
    "water",
    "fiber_volume",
    "fiber_diameter",
    "fiber_length",
    "fiber_aspect_ratio",
    "curing_temp",
    "curing_time",
    "compressive_strength_28d",
    "flexural_strength_28d",
    "slump_flow",
    "cost",
    "carbon_emission",
]

# ─── 本模块所在目录（data/）─────────────────────────────────────────

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _data_path(relpath):
    """将相对于 data/ 的路径转为绝对路径"""
    return os.path.join(_MODULE_DIR, relpath)


# ─── Global UHPC 自定义加载器（处理 3 层表头）──────────────────────

# ─── Global UHPC 位置→列名映射（处理 header=2 后的 Unnamed 列）───

# 当 header=2 后列名为 "Unnamed: N" 时，按列位置索引回退到 row1 子组名
_GLOBAL_UHPC_UNNAMED_MAP = {
    39: "Curing",              # Curing Regime → Curing
    40: "Temperature (o C)",   # Curing Regime → Temperature
    41: "Humidity (%)",        # Curing Regime → Humidity
    42: "Pressure (MPa)",      # Curing Regime → Pressure
    43: "Testing Standard",    # Curing Regime → Testing Standard
    44: "Slump (mm)",          # Workability → Slump
    45: "Slump Flow (mm)",     # Workability → Slump Flow
}

# Water 和 Superplasticizer 的 Amount 列容易混淆，header=2 结果中：
# col 21: Sand → "Amount  (kg/m³)"  → quartz_sand
# col 36: Water → "Amount  (kg/m³)"  → pandas 去重为 "Amount  (kg/m³).1"
# col 38: Superplasticizer → "Amount  (kg/m³)" → pandas 去重为 "Amount  (kg/m³).2"
# col 25: Fiber → "Amount / Quantity of Fiber " → fiber_volume
# col 26: Fiber → "Length (mm)" → fiber_length
# col 27: Fiber → "Diameter (mm)" → fiber_diameter
# col 68: Mechanical Properties → " MOR (MPa)" → flexural (前导空格)

_GLOBAL_UHPC_AMOUNT_DEDUP = {
    # 当 Amount  (kg/m³) 出现多次时，pandas 自动加 .N 后缀
    # 第一个出现在 col21 (Sand)，后续在 col36 (Water), col38 (SP)
    21: "Amount  (kg/m³)",       # Sand Amount → quartz_sand
    36: "Amount  (kg/m³).1",     # Water Amount
    38: "Amount  (kg/m³).2",     # Superplasticizer Amount
}


def _load_global_uhpc(filepath: str) -> pd.DataFrame:
    """
    加载 Global UHPC Excel 文件，智能处理 3 层分层表头。

    使用 python-calamine 直接读取（避免 openpyxl 对该文件的崩溃），
    然后对 Unnamed/空列名按位置回退到 row1 子组名。
    """
    import python_calamine

    workbook = python_calamine.CalamineWorkbook.from_path(filepath)
    sheet = workbook.get_sheet_by_name("UHPC Dataset ")
    rows = sheet.to_python()

    # 前 3 行是表头
    if len(rows) < 4:
        raise ValueError(f"Global UHPC 文件行数不足: {len(rows)}")

    header_r0 = rows[0]  # 主组
    header_r1 = rows[1]  # 子组
    header_r2 = rows[2]  # 具体列名

    # 构建列名：row2 优先，nan 则取 row1，再 nan 则 row0
    # 重复列名自动加 [N] 后缀去重
    col_names = []
    seen = {}
    for i in range(len(header_r2)):
        r2 = header_r2[i] if i < len(header_r2) else None
        r1 = header_r1[i] if i < len(header_r1) else None
        r0 = header_r0[i] if i < len(header_r0) else None

        if r2 is not None and str(r2).strip():
            name = str(r2).strip()
        elif r1 is not None and str(r1).strip():
            name = str(r1).strip()
        elif r0 is not None and str(r0).strip():
            name = str(r0).strip()
        else:
            name = f"Unnamed:{i}"

        # 去重
        if name in seen:
            seen[name] += 1
            name = f"{name} [{seen[name]}]"
        else:
            seen[name] = 0

        col_names.append(name)

    # 数据行
    data_rows = rows[3:]
    df = pd.DataFrame(data_rows, columns=col_names)

    # calamine 可能将所有列保留为 object，尝试转换为数值
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # 文本列保持 object

    return df


# ═════════════════════════════════════════════════════════════════════
# 数据集配置
# ═════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    # ── 1. literature.csv ──────────────────────────────────────────
    "literature": {
        "path": _data_path("literature.csv"),
        "loader": "csv",
        "loader_kwargs": {},
        "description": "项目原生数据，37 行 × 17 列，已使用标准列名",
        "mapping": {c: c for c in STD_COLS},  # 自身即标准列名
        "fiber_unit": "vol%",
        "skip_age_filter": True,  # curing_time=48 是养护时长(小时)，非测试龄期
        "notes": "列最全（5 目标覆盖），但行数太少，仅文献来源",
    },
    # ── 2. Bolbolvand 抗压强度 ──────────────────────────────────────
    "bolbolvand_comp": {
        "path": _data_path("raw/bolbolvand_uhpc/uhpc_compressive_strength.csv"),
        "loader": "csv",
        "loader_kwargs": {},
        "description": "Bolbolvand UHPC 抗压强度，863 行 × 18 列",
        "mapping": {
            "C": "cement",
            "SF": "silica_fume",
            "QP": "quartz_powder",
            "QS": "quartz_sand",
            "SP": "superplasticizer",
            "W": "water",
            "SSV": "fiber_volume",         # vol% ✅
            "D": "fiber_diameter",
            "L": "fiber_length",
            "Age": "curing_time",          # 天
            "CS": "compressive_strength_28d",  # 多龄期，需筛选 Age==28
            # 缺失列（映射为 None 表示该数据集不提供）
            "fiber_aspect_ratio": None,
            "curing_temp": None,
            "flexural_strength_28d": None,
            "slump_flow": None,
            "cost": None,
            "carbon_emission": None,
            # 额外列（不映射到标准列）：
            # FA(粉煤灰), SL(矿渣), MK(偏高岭土), S(未知),
            # BV(玄武岩纤维), PPV(聚丙烯纤维), GV(玻璃纤维)
        },
        "fiber_unit": "vol%",
        "notes": "28 天 335 行（含钢纤维 108 行），强度 40.4-228.7 MPa",
    },
    # ── 3. Bolbolvand 抗弯强度 ──────────────────────────────────────
    "bolbolvand_flex": {
        "path": _data_path("raw/bolbolvand_uhpc/uhpc_flexural_strength.csv"),
        "loader": "csv",
        "loader_kwargs": {},
        "description": "Bolbolvand UHPC 抗弯强度，321 行 × 18 列",
        "mapping": {
            "C": "cement",
            "SF": "silica_fume",
            "QP": "quartz_powder",
            "QS": "quartz_sand",
            "SP": "superplasticizer",
            "W": "water",
            "SSV": "fiber_volume",
            "D": "fiber_diameter",
            "L": "fiber_length",
            "Age": "curing_time",
            "FS": "flexural_strength_28d",  # 多龄期，需筛选 Age==28
            "fiber_aspect_ratio": None,
            "curing_temp": None,
            "compressive_strength_28d": None,
            "slump_flow": None,
            "cost": None,
            "carbon_emission": None,
        },
        "fiber_unit": "vol%",
        "notes": "28 天 162 行（含钢纤维 153 行）",
    },
    # ── 4. Sean Jiao UHPC 综合库 ────────────────────────────────────
    "sean_jiao": {
        "path": _data_path(
            "raw/sean_jiao_uhpc/Bayesian Neural Networks/抗压强度数据.xlsx"
        ),
        "loader": "excel",
        "loader_kwargs": {},
        "description": "Sean Jiao UHPC 综合库，799 行 × 17 列",
        "mapping": {
            "C": "cement",
            "SF": "silica_fume",
            "QP": "quartz_powder",
            "Sand": "quartz_sand",
            "SP": "superplasticizer",
            "W": "water",
            "Fi": "fiber_volume",          # kg/m³ ⚠️ 需转换
            "T": "curing_temp",            # °C, 20-210
            "Age": "curing_time",          # 天
            "compressive strength": "compressive_strength_28d",
            "fiber_diameter": None,
            "fiber_length": None,
            "fiber_aspect_ratio": None,
            "flexural_strength_28d": None,
            "slump_flow": None,
            "cost": None,
            "carbon_emission": None,
            # 额外列：S(未知), FA(粉煤灰), LP(石灰石粉), NS(纳米硅),
            #         Gravel(粗骨料), RH(相对湿度)
        },
        "fiber_unit": "kg/m³",
        "notes": "28 天 302 行，Fi 中位数=0（仅 50% 行有纤维），含粗骨料",
    },
    # ── 5. Mendeley 全球 UHPC 库 ────────────────────────────────────
    "global_uhpc": {
        "path": _data_path(
            "raw/global_uhpc/"
            "Global Dataset of UHPC Mix Designs with Supplement/"
            "UHPC Dataset  (Version-2).xlsx"
        ),
        "loader": "global_uhpc",           # 自定义加载器，处理 3 层表头
        "loader_kwargs": {},
        "description": "Mendeley 全球 UHPC 数据库，2188 行 × 92 列，13/17 特征覆盖",
        "mapping": {
            # 组成（列名由 3 层表头智能构建，重复列名自动加 [N] 去重）
            "Cement Amount  (kg/m³)": "cement",
            "Silica Fume (kg/m³)": "silica_fume",
            "Quartzpowder (kg/m3)": "quartz_powder",
            "Amount  (kg/m³)": "quartz_sand",          # col21 Sand（第一次出现）
            "Amount  (kg/m³) [1]": "water",            # col36 Water（第二次出现）
            "Amount  (kg/m³) [2]": "superplasticizer", # col38 SP（第三次出现）
            # 纤维（单位 kg/m³ ⚠️）
            "Amount / Quantity of Fiber": "fiber_volume",  # col25
            "Length (mm)": "fiber_length",               # col26
            "Diameter (mm)": "fiber_diameter",           # col27
            # 养护（Row2 为 nan，回退到 Row1 名称，strip 后无前后空格）
            "Curing": "curing_time",                 # 文本列 col39，60+ 变体（养护方法）
            "curing_time": "curing_time",            # melt 后的龄期标签（覆盖上述原始列）
            "Temperature (o C)": "curing_temp",      # col40
            # 工作性（Row2 为 nan，回退到 Row1 名称）
            "Slump Flow (mm)": "slump_flow",          # col45
            # 强度（宽格式 8 列需 melt，见 melt_global_uhpc_ages）
            # melt 后 compressive_strength_28d / flexural_strength_28d 已是标准列名，
            # 需穿透映射以保留 melt 结果（否则 map_columns 初始化为 NaN）
            "compressive_strength_28d": "compressive_strength_28d",
            "flexural_strength_28d": "flexural_strength_28d",
            # 抗弯
            "MOR (MPa)": "flexural_strength_28d",    # col68
            # 缺失
            "fiber_aspect_ratio": None,
            "cost": None,
            "carbon_emission": None,
        },
        "fiber_unit": "kg/m³",
        "wide_age_cols": [
            "1-day",
            "3-Day",
            "7-Day",
            "14-Day",
            "21 days",
            "28-Day",
            "56-Day",
            "90-Day",
        ],
        "wide_mor_cols": None,  # MOR 仅单列 28-Day 附近，非宽格式
        "notes": (
            "强度为宽格式 8 个龄期列需 melt；"
            "纤维量 kg/m³ 需转 vol%；"
            "curing 文本 60+ 变体需标准化；"
            "28-Day 非空 2073 行，范围 80-298 MPa"
        ),
    },
    # ── 6. Zenodo SCC 数据库 ────────────────────────────────────────
    "scc_zenodo": {
        "path": _data_path(
            "raw/scc_zenodo/SCC Dataset on Mixture Design & Key Properties - v1.csv"
        ),
        "loader": "csv",
        "loader_kwargs": {
            "encoding": "latin-1",
            "sep": ";",
            "skiprows": 8,  # 4 行统计 + 1 空行 + 3 行表头
        },
        "description": "Zenodo 自密实混凝土数据库，2504 行 × 115 列",
        "mapping": {
            # SCC 无纤维、无强度，仅流变性可映射
            # 大部分标准列缺失
            "cement": None,
            "silica_fume": None,
            "quartz_powder": None,
            "quartz_sand": None,
            "superplasticizer": None,
            "water": None,
            "fiber_volume": None,
            "fiber_diameter": None,
            "fiber_length": None,
            "fiber_aspect_ratio": None,
            "curing_temp": None,
            "curing_time": None,
            "compressive_strength_28d": None,
            "flexural_strength_28d": None,
            "slump_flow": "Slump flow (mm)",
            "cost": None,
            "carbon_emission": None,
        },
        "fiber_unit": None,
        "notes": "完全无钢纤维、无强度数据，对 UHPC 建模价值评级较低",
    },
    # ── 7. Mendeley 钢纤维混凝土力学 ─────────────────────────────────
    "steelfiber_mendeley": {
        "path": _data_path(
            "raw/steelfiber_mendeley/characterizacion and slump test.xlsx"
        ),
        "loader": "excel",
        "loader_kwargs": {"sheet_name": None},  # 加载所有 sheet
        "description": "Mendeley 钢纤维混凝土力学，Slump 6 行 + Agg 5 行",
        "mapping": {
            # 仅坍落度可映射
            "Slump (cm)": "slump_flow",
            "cement": None,
            "silica_fume": None,
            "quartz_powder": None,
            "quartz_sand": None,
            "superplasticizer": None,
            "water": None,
            "fiber_volume": None,
            "fiber_diameter": None,
            "fiber_length": None,
            "fiber_aspect_ratio": None,
            "curing_temp": None,
            "curing_time": None,
            "compressive_strength_28d": None,
            "flexural_strength_28d": None,
            "cost": None,
            "carbon_emission": None,
        },
        "fiber_unit": None,
        "notes": "仅坍落度 + 骨料表征，几乎无建模价值",
    },
    # ── 8. UMN UHPC 实验数据 ─────────────────────────────────────────
    "umn_uhpc": {
        "path": _data_path("raw/umn_uhpc/UHPC NDE Dataset.xlsx"),
        "loader": "excel",
        "loader_kwargs": {"sheet_name": "Strength and NDE values"},
        "description": "UMN UHPC 实验数据，主表 108 行 × 9 列",
        "mapping": {
            # 列名含换行符和特殊间距，需精确匹配
            "Fiber Content\n(% Volume)": "fiber_volume",
            "Sample  Age": "curing_time",             # ⚠️ 双空格，文本+数值
            "Compressive Strength\n(MPa)": "compressive_strength_28d",
            "cement": None,
            "silica_fume": None,
            "quartz_powder": None,
            "quartz_sand": None,
            "superplasticizer": None,
            "water": None,
            "fiber_diameter": None,
            "fiber_length": None,
            "fiber_aspect_ratio": None,
            "curing_temp": None,
            "flexural_strength_28d": None,
            "slump_flow": None,
            "cost": None,
            "carbon_emission": None,
            # 额外列：UPV, Rebound, Distance, Time
        },
        "fiber_unit": "vol%",
        "age_parser": "umn",  # Sample Age 列含 "12 hrs"/"28 day" 等文本，需 parse_umn_age
        "notes": "Ductal 预混料，纤维 0/1/2/3%，强度 2.8-81.0 MPa（含极早期）",
    },
    # ── 9. Kaggle 钢纤维混凝土 ANN ───────────────────────────────────
    "kaggle_steelfiber": {
        "path": _data_path("raw/kaggle_steelfiber/database_en.xlsx"),
        "loader": "excel",
        "loader_kwargs": {
            "sheet_name": None,  # 所有 3 个 sheet
        },
        "description": (
            "Kaggle 钢纤维混凝土 ANN，"
            "3 sheets: compressivestrength(521) / tensilestrength(252) / flexuralstrength(278)"
        ),
        "mapping": {
            "cementecontent": "cement",
            "silicafumecontent": "silica_fume",
            "fine aggregate": "quartz_sand",       # ⚠️ 河砂，非石英砂
            "additivecontent": "superplasticizer",
            "water": "water",
            "fibercontent": "fiber_volume",        # kg/m³ ⚠️ 需转换
            "fiberLength": "fiber_length",
            "Aspect ratio": "fiber_aspect_ratio",  # ✅ 唯一直接提供
            # 强度按 sheet 分别映射
            "compressivestrength": "compressive_strength_28d",
            "flexuralstrength": "flexural_strength_28d",
            "quartz_powder": None,
            "fiber_diameter": None,
            "curing_temp": None,
            "curing_time": None,
            "slump_flow": None,
            "cost": None,
            "carbon_emission": None,
            # 额外列：coarse aggregate（粗骨料，非 UHPC）
        },
        "fiber_unit": "kg/m³",
        "notes": (
            "列名读取后需 .str.strip() 去尾部空格；"
            "含 coarse aggregate 非纯 UHPC；"
            "强度 13-121 MPa；"
            "3 个 sheet 是独立样本"
        ),
    },
    # ── 10. Figshare 再生骨料钢纤维 ──────────────────────────────────
    "figshare_recycled": {
        "path": _data_path("raw/figshare_recycled/Dataset.csv"),
        "loader": "csv",
        "loader_kwargs": {},
        "description": "Figshare 再生骨料钢纤维混凝土，405 行 × 10 列",
        "mapping": {
            "Cement": "cement",
            "Rice Husk Ash": "silica_fume",          # ⚠️ 稻壳灰，非硅灰
            "Fine Aggregate": "quartz_sand",         # ⚠️ 河砂，非石英砂
            "Superplasticizers": "superplasticizer",
            "Water": "water",
            "Fiber (%)": "fiber_volume",             # vol% ✅ 注意括号
            "Fiber length (mm)": "fiber_length",
            "Compressive Strength (MPa)": "compressive_strength_28d",  # ⚠️ 龄期不明
            "quartz_powder": None,
            "fiber_diameter": None,
            "fiber_aspect_ratio": None,
            "curing_temp": None,
            "curing_time": None,                     # ❌ 完全缺失
            "flexural_strength_28d": None,
            "slump_flow": None,
            "cost": None,
            "carbon_emission": None,
        },
        "fiber_unit": "vol%",
        "notes": "龄期完全缺失，RHA 代用 silica_fume，再生骨料 176 行，强度 25.2-102.4 MPa",
    },
}


# ═════════════════════════════════════════════════════════════════════
# 辅助函数
# ═════════════════════════════════════════════════════════════════════

def load_dataset(name: str) -> pd.DataFrame:
    """
    按配置加载指定数据集的原始 DataFrame。

    Args:
        name: 数据集键名（DATASET_CONFIGS 的 key）

    Returns:
        原始 DataFrame（未做列名映射）

    Raises:
        KeyError: 未知数据集名称
    """
    if name not in DATASET_CONFIGS:
        raise KeyError(
            f"未知数据集 '{name}'，可用: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[name]
    path = cfg["path"]
    loader = cfg["loader"]
    kwargs = cfg.get("loader_kwargs", {}).copy()

    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集文件不存在: {path}")

    if loader == "csv":
        df = pd.read_csv(path, **kwargs)
    elif loader == "excel":
        df = pd.read_excel(path, **kwargs)
    elif loader == "global_uhpc":
        df = _load_global_uhpc(path)
    else:
        raise ValueError(f"不支持的 loader 类型: {loader}")

    # Kaggle 特殊处理：去除列名尾部空格
    if name == "kaggle_steelfiber":
        if isinstance(df, dict):
            for sname, sdf in df.items():
                sdf.columns = sdf.columns.str.strip()
        else:
            df.columns = df.columns.str.strip()

    return df


def map_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    将 DataFrame 的原始列名映射为标准列名。

    Args:
        df: 原始 DataFrame
        mapping: {原始列名: 标准列名 或 None}

    Returns:
        仅含标准列名（STD_COLS）的 DataFrame，缺失列填 NaN。
        mapping 中值为 None 的列为全 NaN。
    """
    result = pd.DataFrame(index=df.index)

    for std_col in STD_COLS:
        result[std_col] = np.nan

    for raw_col, std_col in mapping.items():
        if std_col is None:
            continue
        if raw_col in df.columns:
            result[std_col] = df[raw_col].values

    return result


def get_28day(df: pd.DataFrame, age_col: str = "curing_time") -> pd.DataFrame:
    """
    筛选 28 天龄期数据。

    支持多种格式：
    - 数值 28（天）
    - "28" / "28d" / "28 day" / "28 days" 等字符串

    Args:
        df: 含 curing_time 列的 DataFrame
        age_col: 龄期列名（默认 curing_time）

    Returns:
        仅含 28 天数据的 DataFrame
    """
    if age_col not in df.columns:
        raise ValueError(f"DataFrame 中不存在列 '{age_col}'")

    ages = df[age_col]

    # NaN 全列保护：无龄期信息的数据集（如 kaggle/figshare/scc_zenodo）
    # curing_time 全 NaN，NaN == 28 会返回全 False → 0 行，直接返回全部数据
    if ages.isna().all():
        return df.copy()

    # 尝试数值匹配
    if pd.api.types.is_numeric_dtype(ages):
        return df[ages == 28].copy()

    # 文本匹配
    s = ages.astype(str).str.lower().str.strip()
    mask = s.isin(["28", "28d", "28 day", "28 days", "28-day", "28days", "28 days "])
    return df[mask].copy()


def melt_global_uhpc_ages(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Global UHPC 宽格式龄期列 melt 为长格式。

    输入：宽格式 DataFrame（每行一个配比，8 个龄期各一列）
    输出：长格式 DataFrame（每行一个配比×龄期），新增两列：
          - curing_time: 龄期标签 (1d/3d/7d/14d/21d/28d/56d/90d)
          - compressive_strength_28d: 该龄期抗压强度
          - flexural_strength_28d: 该龄期抗弯强度 (MOR)

    Args:
        df: Global UHPC 原始 DataFrame（已用 header=2 加载）

    Returns:
        长格式 DataFrame，保留原始行所有非龄期列，每行扩展为最多 8 行
    """
    # 龄期列名候选（实际列名含单位和空格）
    cs_candidates = {
        "1d": ["1-day", "1-day ", "1-Day", "1 day"],
        "3d": ["3-Day", "3-Day ", "3 day"],
        "7d": ["7-Day", "7-Day ", "7 day"],
        "14d": ["14-Day", "14-Day ", "14 day"],
        "21d": ["21 days", "21 days ", "21-Day"],
        "28d": ["28-Day", "28-Day ", "28 day"],
        "56d": ["56-Day", "56-Day ", "56 day"],
        "90d": ["90-Day", "90-Day ", "90 day"],
    }

    mor_candidates = {
        "1d": ["1-day MOR (MPa)", "1-day MOR"],
        "3d": ["3-Day MOR (MPa)", "3-Day MOR"],
        "7d": ["7-Day MOR (MPa)", "7-Day MOR"],
        "14d": ["14-Day MOR (MPa)", "14-Day MOR"],
        "21d": ["21 days MOR (MPa)", "21 days MOR"],
        "28d": ["28-Day MOR (MPa)", "28-Day MOR"],
        "56d": ["56-Day MOR (MPa)", "56-Day MOR"],
        "90d": ["90-Day MOR (MPa)", "90-Day MOR"],
    }

    # 找到实际存在的列名
    existing_cols = set(df.columns)

    def _find_col(candidates_list):
        for c in candidates_list:
            if c in existing_cols:
                return c
        return None

    cs_map = {age: _find_col(cands) for age, cands in cs_candidates.items()}
    mor_map = {age: _find_col(cands) for age, cands in mor_candidates.items()}

    # 收集要 melt 的强度列
    value_vars_cs = [v for v in cs_map.values() if v is not None]
    value_vars_mor = [v for v in mor_map.values() if v is not None]

    if not value_vars_cs and not value_vars_mor:
        raise ValueError(
            "未找到任何龄期列。请检查 Global UHPC 文件列名是否匹配。"
            f"当前列名: {sorted(existing_cols)}"
        )

    # 非龄期列
    all_value_vars = value_vars_cs + value_vars_mor
    id_vars = [c for c in df.columns if c not in all_value_vars]

    # melt 抗压强度
    frames = []
    if value_vars_cs:
        df_cs = df.melt(
            id_vars=id_vars,
            value_vars=value_vars_cs,
            var_name="age_col",
            value_name="compressive_strength_28d",
        )
        frames.append(df_cs)

    # melt 抗弯强度
    if value_vars_mor:
        df_mor = df.melt(
            id_vars=id_vars,
            value_vars=value_vars_mor,
            var_name="age_col_mor",
            value_name="flexural_strength_28d",
        )
        frames.append(df_mor)

    # 合并抗压和抗弯
    if len(frames) == 2:
        df_long = frames[0].merge(
            frames[1],
            on=id_vars + ["age_col"] if "age_col_mor" in frames[1].columns
            else id_vars,
            how="outer",
            suffixes=("", "_mor"),
        )
        # 对齐龄期列
        if "age_col" in df_long.columns and "age_col_mor" in df_long.columns:
            # 使用同一个龄期标签
            pass  # 保留两列供调试
    else:
        df_long = frames[0]

    # 将龄期列值映射为标准龄期标签
    reverse_cs = {v: k for k, v in cs_map.items() if v is not None}
    if "age_col" in df_long.columns:
        df_long["curing_time"] = df_long["age_col"].map(reverse_cs)
        df_long = df_long.drop(columns=["age_col"])

    if "age_col_mor" in df_long.columns:
        reverse_mor = {v: k for k, v in mor_map.items() if v is not None}
        df_long["age_col_mor"] = df_long["age_col_mor"].map(reverse_mor)

    return df_long


def fiber_kg_to_vol(
    df: pd.DataFrame,
    col: str,
    steel_density: float = 7850.0,
) -> pd.Series:
    """
    将纤维量从 kg/m³ 转换为 vol%（体积百分比）。

    公式: vol% = (kg/m³) / steel_density × 100

    Args:
        df: DataFrame
        col: 纤维量列名（单位 kg/m³）
        steel_density: 钢纤维密度 (kg/m³)，默认 7850

    Returns:
        vol% 的 Series
    """
    if col not in df.columns:
        raise ValueError(f"列 '{col}' 不存在于 DataFrame 中")
    return df[col] / steel_density * 100


def parse_umn_age(age_series: pd.Series) -> pd.Series:
    """
    解析 UMN Sample Age 列为统一的小时数。

    支持格式:
    - "12 hrs" → 12
    - "15 hrs" → 15
    - "1 day" → 24
    - "1.5 day" → 36
    - "3 day" → 72
    - "7 day" → 168
    - "14 day" → 336
    - "28 day" → 672

    Args:
        age_series: UMN Sample Age 列

    Returns:
        数值型龄期（小时）
    """
    result = pd.Series(np.nan, index=age_series.index)

    s = age_series.astype(str).str.lower().str.strip()

    # hrs
    hrs_mask = s.str.contains("hrs|hr|hour", na=False)
    result[hrs_mask] = s[hrs_mask].str.extract(r"(\d+\.?\d*)")[0].astype(float)

    # days
    day_mask = s.str.contains("day", na=False)
    day_vals = s[day_mask].str.extract(r"(\d+\.?\d*)")[0].astype(float)
    result[day_mask] = day_vals * 24

    return result


# ─── 批量加载与合并 ──────────────────────────────────────────────────

def load_all_mapped(
    datasets: list = None,
    filter_28day: bool = True,
) -> dict:
    """
    批量加载并映射所有数据集。

    Args:
        datasets: 要加载的数据集键名列表，默认全部
        filter_28day: 是否自动筛选 28 天龄期

    Returns:
        {数据集名: DataFrame} 字典，每个 DataFrame 已做列名映射
    """
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    results = {}
    for name in datasets:
        try:
            cfg = DATASET_CONFIGS[name]
            df = load_dataset(name)

            # global_uhpc 强度数据是宽格式（8 个龄期各一列），
            # 必须先 melt 为长格式再映射列名，否则抗压强度全 NaN
            if name == "global_uhpc":
                df = melt_global_uhpc_ages(df)

            # 多 sheet 数据集特殊处理
            if isinstance(df, dict):
                for sname, sdf in df.items():
                    mapped = map_columns(sdf, cfg["mapping"])
                    # UMN 龄期解析：Sample Age 列含 "12 hrs" / "28 day" 等文本
                    if cfg.get("age_parser") == "umn" and "curing_time" in mapped.columns:
                        mapped["curing_time"] = parse_umn_age(mapped["curing_time"]) / 24
                    if filter_28day and "curing_time" in mapped.columns and not cfg.get("skip_age_filter"):
                        mapped = get_28day(mapped)
                    results[f"{name}/{sname}"] = mapped
            else:
                mapped = map_columns(df, cfg["mapping"])
                # UMN 龄期解析：Sample Age 列含 "12 hrs" / "28 day" 等文本
                if cfg.get("age_parser") == "umn" and "curing_time" in mapped.columns:
                    mapped["curing_time"] = parse_umn_age(mapped["curing_time"]) / 24
                if filter_28day and "curing_time" in mapped.columns and not cfg.get("skip_age_filter"):
                    mapped = get_28day(mapped)
                results[name] = mapped
        except Exception as e:
            print(f"[警告] 加载 {name} 失败: {e}")
            results[name] = None

    return results


def merge_all(datasets: list = None) -> pd.DataFrame:
    """
    加载、映射并合并所有数据集为 unified_dataset。

    Args:
        datasets: 要合并的数据集列表，默认全部

    Returns:
        合并后的 DataFrame，新增 'source' 列标记来源
    """
    mapped = load_all_mapped(datasets, filter_28day=True)
    frames = []
    for name, df in mapped.items():
        if df is not None and len(df) > 0:
            df = df.copy()
            df["source"] = name
            frames.append(df)

    if not frames:
        raise ValueError("没有可用数据可合并")

    unified = pd.concat(frames, ignore_index=True)
    return unified


# ─── CLI 快速测试 ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "literature"

    print(f"加载数据集: {name}")
    cfg = DATASET_CONFIGS.get(name)
    if cfg is None:
        print(f"未知数据集。可用: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    print(f"  路径: {cfg['path']}")
    print(f"  加载器: {cfg['loader']} {cfg.get('loader_kwargs', {})}")
    print(f"  说明: {cfg['description']}")

    df = load_dataset(name)
    if isinstance(df, dict):
        for sname, sdf in df.items():
            print(f"  Sheet [{sname}]: {sdf.shape}")
    else:
        print(f"  形状: {df.shape}")
        print(f"  列名: {list(df.columns)[:20]}...")
        print(f"  前 2 行:\n{df.head(2)}")

    mapped = map_columns(df, cfg["mapping"]) if not isinstance(df, dict) else None
    if mapped is not None:
        non_null = mapped.columns[mapped.notna().any()].tolist()
        print(f"  映射后非空列 ({len(non_null)}/17): {non_null}")
