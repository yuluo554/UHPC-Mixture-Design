# -*- coding: utf-8 -*-
"""共享 fixtures — 所有测试文件通过 pytest 自动发现"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from core.config import get_all_feature_names, get_all_target_names, get_categorical_features
from core.data_manager import DataManager


@pytest.fixture
def dm():
    """加载示例数据并清洗"""
    mgr = DataManager()
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "literature.csv"
    )
    mgr.load_csv(sample_path)
    mgr.clean(remove_outliers=True, impute=True, contamination=0.05)
    return mgr


@pytest.fixture
def features_targets(dm):
    """获取特征和目标"""
    features = [f for f in get_all_feature_names() if f in dm.cleaned_data.columns]
    targets = [t for t in get_all_target_names() if t in dm.cleaned_data.columns]
    X = dm.cleaned_data[features].select_dtypes(include=[np.number])
    Y = dm.cleaned_data[targets]
    return features, targets, X, Y
