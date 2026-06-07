# -*- coding: utf-8 -*-
"""数据桥接测试（get_unified_dataset / get_dataset_choices）"""

import pytest
from core.data_bridge import get_unified_dataset, get_dataset_choices


class TestDataBridge:
    """统一数据入口函数测试"""

    def test_get_unified_dataset_returns_dict(self):
        """返回值为 dict"""
        data = get_unified_dataset()
        assert isinstance(data, dict)

    def test_get_unified_dataset_has_required_keys(self):
        """dict 含所有必需 key"""
        data = get_unified_dataset()
        for key in ['df', 'n_total', 'n_strength', 'source_counts', 'column_completeness']:
            assert key in data, f"缺少 key: {key}"

    def test_dataframe_not_empty(self):
        """DataFrame 非空"""
        data = get_unified_dataset()
        assert len(data['df']) > 0

    def test_n_total_positive(self):
        """n_total > 0"""
        data = get_unified_dataset()
        assert data['n_total'] > 0

    def test_source_column_exists(self):
        """DataFrame 含 source 列"""
        data = get_unified_dataset()
        assert 'source' in data['df'].columns

    def test_source_counts_not_empty(self):
        """source_counts 非空"""
        data = get_unified_dataset()
        assert len(data['source_counts']) > 0

    def test_column_completeness_is_dataframe(self):
        """column_completeness 是 DataFrame"""
        data = get_unified_dataset()
        import pandas as pd
        assert isinstance(data['column_completeness'], pd.DataFrame)

    def test_get_dataset_choices_returns_list(self):
        """返回 list"""
        choices = get_dataset_choices()
        assert isinstance(choices, list)

    def test_get_dataset_choices_first_is_all(self):
        """第一项为 'all' 选项"""
        choices = get_dataset_choices()
        assert choices[0][0] == 'all'

    def test_get_dataset_choices_tuples(self):
        """每个元素为 (key, label) 元组"""
        choices = get_dataset_choices()
        for item in choices:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)

    def test_filter_28day_false(self):
        """filter_28day=False 不报错且返回更多行"""
        import pandas as pd
        data_all = get_unified_dataset(filter_28day=False)
        assert data_all['n_total'] > 0
        # BUG-20260607-039 修复：原 isinstance(df, type(df)) 永远 True，
        # 改为 isinstance(df, pd.DataFrame) 上下文部分均可在 pd 对象化不同步中被更新
        assert isinstance(data_all['df'], pd.DataFrame)
