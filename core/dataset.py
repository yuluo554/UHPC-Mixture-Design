# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign 统一数据集抽象

提供 X/Y/不确定性/边界/元数据 一体化封装，
替代模块间裸 np.ndarray / pd.DataFrame 传参。

参考: BOxCrete SustainableConcreteDataset (boxcrete/utils.py:91-356)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import get_all_feature_names, get_all_target_names, get_feature_ranges


@dataclass
class UHPCDataset:
    """UHPC 配比-性能数据集统一封装。

    Attributes:
        X: 特征矩阵 (n_samples, n_features)
        Y: 目标矩阵 (n_samples, n_targets)
        X_columns: 特征列名
        Y_columns: 目标列名
        Y_std: 目标预测标准差 (n_samples, n_targets)，可选
        bounds: {feature_name: (min, max)}
        metadata: 附加元信息（来源分布等）
    """

    X: np.ndarray
    Y: np.ndarray
    X_columns: List[str]
    Y_columns: List[str]
    Y_std: Optional[np.ndarray] = None
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """类型转换 + 形状校验。"""
        # 确保 numpy 数组
        if not isinstance(self.X, np.ndarray):
            self.X = np.asarray(self.X, dtype=np.float64)
        if not isinstance(self.Y, np.ndarray):
            self.Y = np.asarray(self.Y, dtype=np.float64)
        if self.Y_std is not None and not isinstance(self.Y_std, np.ndarray):
            self.Y_std = np.asarray(self.Y_std, dtype=np.float64)

        # 形状校验
        if self.X.ndim != 2:
            raise ValueError(f"X 必须为 2D，当前 shape={self.X.shape}")
        if self.Y.ndim != 2:
            raise ValueError(f"Y 必须为 2D，当前 shape={self.Y.shape}")
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                f"X 和 Y 样本数不一致: {self.X.shape[0]} vs {self.Y.shape[0]}"
            )
        if len(self.X_columns) != self.X.shape[1]:
            raise ValueError(
                f"X_columns 长度 ({len(self.X_columns)}) != X.shape[1] ({self.X.shape[1]})"
            )
        if len(self.Y_columns) != self.Y.shape[1]:
            raise ValueError(
                f"Y_columns 长度 ({len(self.Y_columns)}) != Y.shape[1] ({self.Y.shape[1]})"
            )
        if self.Y_std is not None and self.Y_std.shape != self.Y.shape:
            raise ValueError(
                f"Y_std shape {self.Y_std.shape} != Y shape {self.Y.shape}"
            )

    # ── 属性 ─────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        """样本数。"""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """特征维度。"""
        return self.X.shape[1]

    @property
    def n_targets(self) -> int:
        """目标维度。"""
        return self.Y.shape[1]

    @property
    def Y_var(self) -> Optional[np.ndarray]:
        """方差（从 Y_std 计算）。"""
        if self.Y_std is not None:
            return self.Y_std ** 2
        return None

    # ── 核心方法 ─────────────────────────────────────────────────

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[UHPCDataset, UHPCDataset]:
        """划分训练/测试集，返回两个新 UHPCDataset。

        Y_std 和 metadata 按相同索引同步切分，bounds 直接复制。
        """
        from sklearn.model_selection import train_test_split as tts

        indices = np.arange(self.n_samples)
        train_idx, test_idx = tts(
            indices, test_size=test_size, random_state=random_state
        )

        Y_std_train = self.Y_std[train_idx] if self.Y_std is not None else None
        Y_std_test = self.Y_std[test_idx] if self.Y_std is not None else None

        train = UHPCDataset(
            X=self.X[train_idx, :],
            Y=self.Y[train_idx, :],
            X_columns=list(self.X_columns),
            Y_columns=list(self.Y_columns),
            Y_std=Y_std_train,
            bounds=dict(self.bounds),
            metadata={**self.metadata, "split": "train"},
        )
        test = UHPCDataset(
            X=self.X[test_idx, :],
            Y=self.Y[test_idx, :],
            X_columns=list(self.X_columns),
            Y_columns=list(self.Y_columns),
            Y_std=Y_std_test,
            bounds=dict(self.bounds),
            metadata={**self.metadata, "split": "test"},
        )
        return train, test

    def to_dataframe(
        self,
        include_targets: bool = True,
        include_std: bool = False,
    ) -> pd.DataFrame:
        """将 X + Y 合并为 DataFrame。

        Args:
            include_targets: 是否包含目标列
            include_std: 是否包含标准差列（需 Y_std 非空）
        """
        df = pd.DataFrame(self.X, columns=self.X_columns)
        if include_targets:
            for i, col in enumerate(self.Y_columns):
                df[col] = self.Y[:, i]
        if include_std and self.Y_std is not None:
            for i, col in enumerate(self.Y_columns):
                df[f"{col}_std"] = self.Y_std[:, i]
        return df

    # ── 工厂方法 ─────────────────────────────────────────────────

    @classmethod
    def from_unified_dataset(
        cls,
        data: dict,
        X_cols: Optional[List[str]] = None,
        Y_cols: Optional[List[str]] = None,
    ) -> UHPCDataset:
        """从 get_unified_dataset() 的输出构建 UHPCDataset。

        Args:
            data: get_unified_dataset() 返回的字典
            X_cols: 特征列名（默认从 config 获取，仅保留 df 中存在的）
            Y_cols: 目标列名（默认从 config 获取，仅保留 df 中存在的）

        Returns:
            UHPCDataset 实例

        Raises:
            ValueError: dropna 后无有效样本
        """
        df: pd.DataFrame = data["df"].copy()

        if X_cols is None:
            X_cols = get_all_feature_names()
        if Y_cols is None:
            Y_cols = get_all_target_names()

        # 仅保留 DataFrame 中实际存在的列
        X_cols_actual = [c for c in X_cols if c in df.columns]
        Y_cols_actual = [c for c in Y_cols if c in df.columns]

        if not X_cols_actual:
            raise ValueError(f"DataFrame 中未找到任何特征列。预期: {X_cols}")
        if not Y_cols_actual:
            raise ValueError(f"DataFrame 中未找到任何目标列。预期: {Y_cols}")

        # 提取特征和目标矩阵（强制数值转换，非数值→NaN）
        # 下游代码（DataManager.impute_missing, MICE 等）负责处理 NaN
        X = df[X_cols_actual].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(np.float64)
        Y = df[Y_cols_actual].apply(
            pd.to_numeric, errors="coerce"
        ).values.astype(np.float64)

        if len(df) == 0:
            raise ValueError("输入 DataFrame 为空")

        # 边界仅对实际存在的数值特征
        all_bounds = get_feature_ranges()
        bounds = {k: v for k, v in all_bounds.items() if k in X_cols_actual}

        metadata = {
            "source_counts": data.get("source_counts", {}),
            "n_total": data.get("n_total", 0),
            "n_with_strength": data.get("n_strength", 0),
            "n_after_dropna": len(df),
        }

        return cls(
            X=X,
            Y=Y,
            X_columns=X_cols_actual,
            Y_columns=Y_cols_actual,
            bounds=bounds,
            metadata=metadata,
        )
