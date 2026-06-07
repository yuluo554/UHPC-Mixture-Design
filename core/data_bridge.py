# -*- coding: utf-8 -*-
"""
data_bridge.py — 统一数据入口

所有 Streamlit 页面通过本模块获取数据，替代硬编码 `pd.read_csv("data/literature.csv")`。
对外提供两个函数：
  - get_unified_dataset() → {df, n_total, n_strength, source_counts, column_completeness}
  - get_dataset_choices() → [(key, label), ...] 供 UI 下拉框使用
"""

import pandas as pd
from data.column_mapping import merge_all, load_all_mapped, DATASET_CONFIGS

# Streamlit 缓存装饰器：生产环境缓存 1 小时，脱机测试优雅降级
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    _HAS_STREAMLIT = True
except ImportError:
    st = None
    get_script_run_ctx = None
    _HAS_STREAMLIT = False


def _in_streamlit_runtime() -> bool:
    """检测是否在 Streamlit 运行时上下文中"""
    if not _HAS_STREAMLIT:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _with_cache(func):
    """条件缓存装饰器：仅在 Streamlit 运行时中启用 @st.cache_data
    注意：st.cache_data 的包装延迟到首次调用时，避免导入期副作用（如 segfault）"""
    _cached = None

    def wrapper(*args, **kwargs):
        nonlocal _cached
        if _in_streamlit_runtime():
            if _cached is None:
                _cached = st.cache_data(ttl=3600)(func)
            return _cached(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


@_with_cache
def get_unified_dataset(
    datasets: list[str] | None = None,
    filter_28day: bool = True,
) -> dict:
    """
    获取合并后的全量数据集及元信息。

    Args:
        datasets: 要加载的数据集键名列表，默认全部（DATASET_CONFIGS 中的所有键）
        filter_28day: 是否仅保留 28 天龄期数据

    Returns:
        dict:
            - df:                 合并后的 DataFrame（含 'source' 列标记来源）
            - n_total:            总行数
            - n_strength:         有抗压强度数据（compressive_strength_28d 非空）的行数
            - source_counts:      {数据集名: 行数}
            - column_completeness: 各列非空率 DataFrame
    """
    if filter_28day:
        df = merge_all(datasets)
    else:
        mapped = load_all_mapped(datasets, filter_28day=False)
        frames = []
        for name, mdf in mapped.items():
            if mdf is not None and len(mdf) > 0:
                mdf = mdf.copy()
                mdf["source"] = name
                frames.append(mdf)
        if not frames:
            raise ValueError("没有可用数据可合并")
        df = pd.concat(frames, ignore_index=True)

    # 统计来源分布
    source_counts = (
        df["source"].value_counts().to_dict() if "source" in df.columns else {}
    )

    # 各列非空率
    completeness = pd.DataFrame({
        "column": df.columns,
        "non_null_pct": (df.notna().sum() / len(df) * 100).round(1).values,
    }).sort_values("non_null_pct", ascending=False)

    n_strength = (
        df["compressive_strength_28d"].notna().sum()
        if "compressive_strength_28d" in df.columns
        else 0
    )

    return {
        "df": df,
        "n_total": len(df),
        "n_strength": int(n_strength),
        "source_counts": source_counts,
        "column_completeness": completeness,
    }


def get_dataset_choices(
    datasets: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    返回供 UI 下拉框/多选器使用的选项列表。

    Args:
        datasets: 要包含的数据集键名列表，默认全部

    Returns:
        list[tuple[str, str]]: [(key, label), ...]
            第一个元素始终为 ("all", "📦 全部数据集（合并）")
    """
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    # BUG-20260607-030 修复：以前用 _mapped = None 闭包缓存，
    # 在 get_dataset_choices 多次调用之间不会刷新，
    # 文件被外部修改后无法发现。改用 _with_cache
    # 装饰器（只在 Streamlit 运行时缓存 10 分钟）。
    @_with_cache
    def _row_count_map() -> dict[str, int]:
        _mapped = load_all_mapped(filter_28day=False)
        out: dict[str, int] = {}
        for key, df in _mapped.items():
            if df is not None and len(df) > 0:
                out[key] = len(df)
        return out

    def _get_row_count(key):
        row_map = _row_count_map()
        # 精确匹配优先
        if key in row_map:
            return row_map[key]
        # 多 sheet 数据集：统计所有 "{key}/" 前缀的 sheet
        total = 0
        prefix = key + "/"
        for k, n in row_map.items():
            if k.startswith(prefix):
                total += n
        return total

    choices = [("all", "📦 全部数据集（合并）")]

    for key in datasets:
        cfg = DATASET_CONFIGS.get(key, {})
        desc = cfg.get("description", key)
        # 截取描述首句（到第一个逗号或 40 字）
        short = desc.split("，")[0].split(",")[0][:40]
        n = _get_row_count(key)
        label = f"{short}（{n} 条）" if n > 0 else short
        choices.append((key, label))

    return choices
