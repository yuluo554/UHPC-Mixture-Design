# -*- coding: utf-8 -*-
"""列映射测试：空白字符串 → NaN、数值化、文本列保留

这些用例不依赖 data/raw/，因此可以在「只带 37 行文献数据」的默认仓库里跑
（对应缺陷 G：global_uhpc melt 后强度列被空白字符串填满，notna() 判据失真）。
"""

import numpy as np
import pandas as pd
import pytest

from data.column_mapping import STD_COLS, map_columns


def _mapped(raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = map_columns(raw, mapping)
    # map_columns 必须始终返回 17 个标准列
    assert list(out.columns) == STD_COLS
    return out


class TestBlankStringHandling:

    def test_blank_strings_become_nan_in_standard_column(self):
        """空白字符串不能算作「非空」"""
        raw = pd.DataFrame({"CS": [100.0, "", "  ", None, 120.0]})
        out = _mapped(raw, {"CS": "compressive_strength_28d"})
        col = out["compressive_strength_28d"]
        assert col.notna().sum() == 2
        assert np.isnan(col.iloc[1])
        assert np.isnan(col.iloc[2])

    def test_column_with_few_dirty_values_converts_to_numeric(self):
        """非空值里 ≥80% 可转数值时，整列数值化，脏值转 NaN

        真实案例：global_uhpc 的 3915 个强度值里混了 15 行 "EN 196–1"
        （测试标准误入强度列）。整列必须转成 float，而不是停在 object。
        """
        values = [str(v) for v in range(100, 190)] + ["EN 196–1"] * 15
        raw = pd.DataFrame({"CS": values})
        out = _mapped(raw, {"CS": "compressive_strength_28d"})
        col = out["compressive_strength_28d"]
        assert pd.api.types.is_float_dtype(col)
        assert col.notna().sum() == 90

    def test_text_age_column_is_preserved(self):
        """文本龄期（UMN "12 hrs"/"28 day"）不能被数值化抹成 NaN"""
        raw = pd.DataFrame({"Sample  Age": ["12 hrs", "28 day", "3 days"]})
        out = _mapped(raw, {"Sample  Age": "curing_time"})
        col = out["curing_time"]
        assert col.notna().sum() == 3
        assert col.iloc[1] == "28 day"

    def test_unmapped_columns_are_all_nan(self):
        """mapping 里没有的列保持全 NaN"""
        raw = pd.DataFrame({"CS": [1.0, 2.0]})
        out = _mapped(raw, {"CS": "compressive_strength_28d"})
        assert out["flexural_strength_28d"].isna().all()
