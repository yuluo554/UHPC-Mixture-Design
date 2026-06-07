# -*- coding: utf-8 -*-
"""DataManager 测试"""

import pytest


class TestDataManager:
    """数据管理器测试"""

    def test_load_and_clean(self, dm):
        assert dm.raw_data is not None
        assert dm.cleaned_data is not None
        assert len(dm.cleaned_data) <= len(dm.raw_data)
        assert dm.cleaned_data.isnull().sum().sum() == 0

    def test_statistics(self, dm):
        stats = dm.get_statistics()
        assert stats is not None
        assert len(stats) > 0

    def test_correlation(self, dm):
        corr = dm.get_correlation_matrix()
        assert corr is not None
        assert corr.shape[0] > 0

    def test_split(self, dm):
        X_train, X_test, y_train, y_test = dm.split(test_size=0.2)
        assert len(X_train) >= len(X_test)
        assert len(y_train) >= len(y_test)

