# -*- coding: utf-8 -*-
"""Config 模块测试"""

import pytest
from core.config import get_all_feature_names, get_all_target_names


class TestConfig:
    """配置模块测试"""

    def test_feature_names(self):
        features = get_all_feature_names()
        assert len(features) == 12

    def test_target_names(self):
        targets = get_all_target_names()
        assert len(targets) == 5

    def test_ranges(self):
        from core.config import get_feature_ranges
        ranges = get_feature_ranges()
        assert "cement" in ranges
        assert ranges["cement"] == (600, 1000)
        assert "water" in ranges
        assert ranges["water"] == (140, 220)

