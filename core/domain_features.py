# -*- coding: utf-8 -*-
"""
UHPC domain-specific feature engineering transform for BoTorch GP models.

Appends physically meaningful derived ratios (water/binder, sand/binder,
superplasticizer/binder) to the input tensor, following the BOxCrete
AppendDerivedFeatures pattern from:

    boxcrete/models.py:48-86

Reference:
    BOxCrete / SustainableConcrete: arxiv.org/abs/2603.21525
"""

# torch / botorch 是可选依赖：BayesianOptimizer 的 GP 路径才需要它们。
# 顶层无保护 import 会让「没装 torch 时也能用 RF/进化优化」的降级失效
# （原始缺陷 A：HAS_TORCH=False 分支永远走不到）。
try:
    import torch
    from torch import Tensor
    from botorch.models.transforms.input import InputTransform

    HAS_TORCH = True
except ImportError:  # noqa: BLE001 — 缺 torch 或缺 botorch 都走降级
    torch = None
    Tensor = "torch.Tensor"
    HAS_TORCH = False

    class InputTransform:  # type: ignore[no-redef]
        """占位基类：仅在缺少 botorch 时用于让本模块可被 import。

        真正使用时（AppendUHPCFeatures.__init__）会抛出 ImportError。
        """

        transform_on_train: bool = True
        transform_on_eval: bool = True
        transform_on_fantasize: bool = True
        is_one_to_many: bool = False

        def __init__(self, *args, **kwargs):
            pass


class AppendUHPCFeatures(InputTransform):
    """Append UHPC domain knowledge derived features to GP input.

    Computes and appends three engineered ratios:
      - water / binder  (binder = cement + silica_fume + quartz_powder)
      - quartz_sand / binder
      - superplasticizer / binder

    These ratios encode physical relationships that stationary GP kernels
    cannot learn from raw composition values alone.  By providing them as
    explicit input dimensions the GP can treat them as first-class features.

    Usage::

        tf = AppendUHPCFeatures(
            cement_idx=0, sf_idx=1, qp_idx=2,
            sand_idx=3, sp_idx=4, water_idx=5,
        )
        model = SingleTaskGP(train_X, train_Y, input_transform=tf)

    When used inside a ChainedInputTransform, this transform should be
    placed BEFORE Normalize so the derived ratios are also normalised::

        Chain = ChainedInputTransform
        input_transform = Chain(
            uhpc=AppendUHPCFeatures(),
            norm=Normalize(d=d_original + 3),
        )
    """

    # --- BoTorch InputTransform contract flags ---
    # BoTorch 0.18 declares these as type annotations without defaults;
    # subclasses MUST set them.
    transform_on_train: bool = True
    transform_on_eval: bool = True
    transform_on_fantasize: bool = True
    is_one_to_many: bool = False

    def __init__(
        self,
        cement_idx: int = 0,
        sf_idx: int = 1,
        qp_idx: int = 2,
        sand_idx: int = 3,
        sp_idx: int = 4,
        water_idx: int = 5,
    ):
        r"""Args:
            cement_idx: Column index of cement in the *continuous* feature tensor.
                UHPC-AutoDesign default is 0.
            sf_idx: Column index of silica_fume.  Default 1.
            qp_idx: Column index of quartz_powder.  Default 2.
            sand_idx: Column index of quartz_sand.  Default 3.
            sp_idx: Column index of superplasticizer.  Default 4.
            water_idx: Column index of water.  Default 5.

        The defaults assume the standard UHPC-AutoDesign feature order
        (composition group first, then fibre, then curing).
        """
        super().__init__()
        if not HAS_TORCH:
            raise ImportError(
                "AppendUHPCFeatures 需要 torch 与 botorch（未安装）。"
                "请执行: python -m pip install torch botorch，"
                "或改用不需要 GP 的贝叶斯优化后端（surrogate_type='rf'）"
            )
        self.cement_idx = cement_idx
        self.sf_idx = sf_idx
        self.qp_idx = qp_idx
        self.sand_idx = sand_idx
        self.sp_idx = sp_idx
        self.water_idx = water_idx

    # -----------------------------------------------------------------
    #  Core transform
    # -----------------------------------------------------------------

    def transform(self, X: Tensor) -> Tensor:
        r"""Append water/binder, sand/binder and sp/binder to the input.

        Args:
            X: ``batch_shape x n x d`` tensor of continuous features.

        Returns:
            ``batch_shape x n x (d + 3)`` tensor.
        """
        # Binder = cement + silica_fume + quartz_powder.
        # Slice indexing X[..., i : i+1] preserves the last dimension so
        # the arithmetic produces (..., 1) tensors that catenate cleanly.
        binder = (
            X[..., self.cement_idx : self.cement_idx + 1]
            + X[..., self.sf_idx : self.sf_idx + 1]
            + X[..., self.qp_idx : self.qp_idx + 1]
        ).clamp(min=1.0)

        wb = X[..., self.water_idx : self.water_idx + 1] / binder
        sb = X[..., self.sand_idx : self.sand_idx + 1] / binder
        spb = X[..., self.sp_idx : self.sp_idx + 1] / binder

        return torch.cat([X, wb, sb, spb], dim=-1)

    # -----------------------------------------------------------------
    #  InputTransform abstract methods
    # -----------------------------------------------------------------

    def untransform(self, X: Tensor) -> Tensor:
        r"""Strip appended derived features, returning composition features only.

        BoTorch rarely calls ``untransform`` during acquisition optimisation
        (the forward path only invokes ``transform``).  This implementation is
        provided for completeness and returns the first six columns that were
        used to derive the ratios.

        Args:
            X: ``batch_shape x n x (d + 3)`` tensor.

        Returns:
            ``batch_shape x n x 6`` tensor.
        """
        return X[..., :6]

    def equals(self, other: InputTransform) -> bool:
        r"""Compare index parameters for equivalence.

        ``AppendUHPCFeatures`` has no learnable parameters — its identity is
        determined solely by the six index attributes.  We override the
        default ``equals`` (which compares ``state_dict``) because two
        instances with different indices would otherwise be considered equal.
        """
        if type(self) is not type(other):
            return False
        return (
            self.cement_idx == other.cement_idx
            and self.sf_idx == other.sf_idx
            and self.qp_idx == other.qp_idx
            and self.sand_idx == other.sand_idx
            and self.sp_idx == other.sp_idx
            and self.water_idx == other.water_idx
        )


# ═══════════════════════════════════════════════════════════════════
# 共享约束函数 — Day 7: VirtualSpaceOptimizer 和 BayesianOptimizer 共用
# ═══════════════════════════════════════════════════════════════════

def check_uhpc_constraints_df(
    df: "pd.DataFrame",
    feature_names: list,
) -> "pd.Series":
    """DataFrame 版本的 UHPC 物理约束检查。

    逻辑与 BayesianOptimizer._check_uhpc_constraints 一致（4 条约束），
    但输入/输出为 pandas，供 VirtualSpaceOptimizer 和 BayesianOptimizer 共用。

    约束清单：
      1. 水胶比 ∈ [0.14, 0.24]
      2. 砂胶比 ∈ [0.5, 1.5]
      3. 减水剂/胶凝比 ∈ [0.005, 0.05]
      4. 纤维体积率 ∈ [0.5, 3.0]（仅纤维参数 >0 时检查）

    每个约束独立 try/except —— 缺失所需列时优雅跳过该条约束。

    Args:
        df: 候选配比 DataFrame
        feature_names: 特征名列表（用于按名称查找列）

    Returns:
        bool Series，True = 该行通过所有有效约束
    """
    import numpy as np
    import pandas as pd

    n = len(df)
    feasible = pd.Series(np.ones(n, dtype=bool), index=df.index)
    col_map = {name: name for name in feature_names if name in df.columns}

    def _has(*names: str) -> bool:
        return all(n in col_map for n in names)

    # ── 约束 1: 水胶比 ∈ [0.14, 0.24] ─────────────────
    # w/b = water / (cement + silica_fume + quartz_powder)
    try:
        if _has("water", "cement", "silica_fume", "quartz_powder"):
            water = df[col_map["water"]].values
            binder = (
                df[col_map["cement"]].values
                + df[col_map["silica_fume"]].values
                + df[col_map["quartz_powder"]].values
            )
            binder = np.where(binder == 0, 1.0, binder)
            wb = water / binder
            feasible &= (wb >= 0.14) & (wb <= 0.24)
    except Exception:
        pass

    # ── 约束 2: 砂胶比 ∈ [0.5, 1.5] ─────────────────
    # sand/binder = quartz_sand / binder
    try:
        if _has("quartz_sand", "cement", "silica_fume", "quartz_powder"):
            binder2 = (
                df[col_map["cement"]].values
                + df[col_map["silica_fume"]].values
                + df[col_map["quartz_powder"]].values
            )
            binder2 = np.where(binder2 == 0, 1.0, binder2)
            sb = df[col_map["quartz_sand"]].values / binder2
            feasible &= (sb >= 0.5) & (sb <= 1.5)
    except Exception:
        pass

    # ── 约束 3: 减水剂/胶凝比 ∈ [0.005, 0.05] ────────
    try:
        if _has("superplasticizer", "cement", "silica_fume", "quartz_powder"):
            binder3 = (
                df[col_map["cement"]].values
                + df[col_map["silica_fume"]].values
                + df[col_map["quartz_powder"]].values
            )
            binder3 = np.where(binder3 == 0, 1.0, binder3)
            spb = df[col_map["superplasticizer"]].values / binder3
            feasible &= (spb >= 0.005) & (spb <= 0.05)
    except Exception:
        pass

    # ── 约束 4: 纤维体积率 ∈ [0.5, 3.0] ──────────────
    # 仅在纤维参数 >0 时检查
    try:
        fv_col = None
        fd_col = None
        fl_col = None
        for name in ["fiber_vol_frac", "fiber_volume"]:
            if name in col_map:
                fv_col = name
                break
        for name in ["fiber_diameter"]:
            if name in col_map:
                fd_col = name
                break
        for name in ["fiber_length"]:
            if name in col_map:
                fl_col = name
                break
        if fv_col and fd_col and fl_col:
            fiber_present = (
                (df[col_map[fd_col]].values > 0)
                & (df[col_map[fl_col]].values > 0)
            )
            feasible[fiber_present] &= (
                (df.loc[fiber_present, col_map[fv_col]].values >= 0.5)
                & (df.loc[fiber_present, col_map[fv_col]].values <= 3.0)
            )
    except Exception:
        pass

    return feasible
