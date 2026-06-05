# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign TOPSIS 多准则决策模块
对 Pareto 前沿进行 TOPSIS 排序，选出最优配比方案

核心类：
- TOPSISSelector: TOPSIS 排序 + 可视化（雷达图 + 平行坐标图）

TOPSIS 算法（纯 NumPy 实现，不依赖 pymcdm）：
1. 构建决策矩阵（Pareto 目标值）
2. 向量归一化
3. 加权归一化决策矩阵
4. 确定理想解 (PIS) 和负理想解 (NIS)
5. 计算各方案到 PIS/NIS 的欧氏距离
6. 计算相对贴近度并排序

使用方式：
    from core.topsis_selector import TOPSISSelector
    selector = TOPSISSelector(pareto_Y, target_names, directions, weights)
    result_df = selector.rank()
    top5 = selector.get_top_n(5)
    fig_radar = selector.plot_radar(top_indices=[0, 1, 2])
    fig_parallel = selector.plot_parallel_coordinates()
"""

import warnings
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 可选：plotly 用于可视化
try:
    import plotly.graph_objects as go
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from core.config import get_target_direction, get_feature_label


# ═══════════════════════════════════════════════════════════════════
# TOPSISSelector
# ═══════════════════════════════════════════════════════════════════

class TOPSISSelector:
    """
    TOPSIS（逼近理想解排序法）多准则决策器。

    对 Pareto 前沿上的非支配解进行加权排序，
    帮助用户从众多 Pareto 最优解中选出最符合偏好的方案。

    Parameters
    ----------
    pareto_Y : np.ndarray, shape (n, m)
        Pareto 前沿的目标值矩阵。
    target_names : list[str]
        目标名称列表（长度 m）。
    directions : list[str]
        各目标方向: 'maximize' / 'minimize' / 'target'。
    weights : np.ndarray or list, optional
        目标权重（长度 m），默认等权。
    pareto_X : np.ndarray, optional
        对应的设计变量矩阵，用于结果展示。
    feature_names : list[str], optional
        特征名列表，用于显示。
    """

    def __init__(
        self,
        pareto_Y: np.ndarray,
        target_names: List[str],
        directions: List[str],
        weights: Optional[np.ndarray] = None,
        pareto_X: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        if len(pareto_Y) == 0:
            raise ValueError("Pareto 前沿为空，无法进行 TOPSIS 排序")

        self.pareto_Y = np.asarray(pareto_Y, dtype=np.float64)
        self.target_names = list(target_names)
        self.directions = list(directions)
        self.pareto_X = (
            np.asarray(pareto_X, dtype=np.float64) if pareto_X is not None else None
        )
        self.feature_names = feature_names or []

        # 权重：默认等权
        if weights is None:
            self.weights = np.ones(len(target_names)) / len(target_names)
        else:
            w = np.asarray(weights, dtype=np.float64)
            self.weights = w / w.sum()  # 归一化

        # 计算结果缓存
        self._scores: Optional[np.ndarray] = None
        self._dist_to_pis: Optional[np.ndarray] = None
        self._dist_to_nis: Optional[np.ndarray] = None
        self._norm_matrix: Optional[np.ndarray] = None
        self._pis: Optional[np.ndarray] = None
        self._nis: Optional[np.ndarray] = None

    # ── TOPSIS 核心算法 ────────────────────────────────────────────

    def compute_scores(self) -> np.ndarray:
        """
        执行 TOPSIS 计算，返回相对贴近度得分。

        步骤：
        1. 将目标统一为最大化（对 minimize/target 取负或距离）
        2. 向量归一化
        3. 加权
        4. 计算 PIS / NIS
        5. 欧氏距离 + 相对贴近度

        Returns:
            np.ndarray, shape (n,): 相对贴近度（0~1，越大越优）。
        """
        n, m = self.pareto_Y.shape

        # ── Step 0: 统一为"越大越好" ──
        Y_benefit = self._to_benefit(self.pareto_Y)

        # ── Step 1: 向量归一化 ──
        # r_ij = y_ij / sqrt(sum(y_kj²))
        norm_sq = (Y_benefit ** 2).sum(axis=0)
        norm_sq[norm_sq == 0] = 1.0  # 防除零
        self._norm_matrix = Y_benefit / np.sqrt(norm_sq)

        # ── Step 2: 加权归一化决策矩阵 ──
        v = self._norm_matrix * self.weights  # (n, m)

        # ── Step 3: 确定理想解 PIS 和负理想解 NIS ──
        # PIS: 各列最大值（越大越好）
        # NIS: 各列最小值
        self._pis = v.max(axis=0)
        self._nis = v.min(axis=0)

        # ── Step 4: 计算欧氏距离 ──
        diff_pis = v - self._pis
        diff_nis = v - self._nis
        self._dist_to_pis = np.sqrt((diff_pis ** 2).sum(axis=1))
        self._dist_to_nis = np.sqrt((diff_nis ** 2).sum(axis=1))

        # ── Step 5: 相对贴近度 ──
        denom = self._dist_to_pis + self._dist_to_nis
        denom[denom == 0] = 1e-12
        self._scores = self._dist_to_nis / denom

        return self._scores

    def _to_benefit(self, Y: np.ndarray) -> np.ndarray:
        """
        将所有目标统一转换为"越大越好"（效益型）。

        - maximize → 保持
        - minimize → 取负（加平移使非负）
        - target → 到目标值距离的负数（加平移）
        """
        Yb = Y.astype(np.float64).copy()
        for i, d in enumerate(self.directions):
            col = Yb[:, i]
            if d == "minimize":
                # 取负后平移使最小值为 0
                neg = -col
                Yb[:, i] = neg - neg.min()
            elif d == "target":
                # 距离越近越好 → 取负距离
                median = float(np.median(col))
                dist = np.abs(col - median)
                neg_dist = -dist
                Yb[:, i] = neg_dist - neg_dist.min()
            # maximize: 保持，确保非负
            Yb[:, i] = Yb[:, i] - Yb[:, i].min()
        return Yb

    # ── 排序与推荐 ────────────────────────────────────────────────

    def rank(self) -> pd.DataFrame:
        """
        返回按 TOPSIS 得分降序排列的完整决策表。

        Returns:
            pd.DataFrame: 包含 Rank, Score, 目标值, （可选）特征值。
        """
        if self._scores is None:
            self.compute_scores()

        results = {
            "Rank": np.arange(1, len(self._scores) + 1),
            "TOPSIS得分": np.round(self._scores, 4),
        }
        for j, name in enumerate(self.target_names):
            results[name] = np.round(self.pareto_Y[:, j], 2)

        df = pd.DataFrame(results).sort_values("TOPSIS得分", ascending=False)
        df["Rank"] = range(1, len(df) + 1)
        df = df.reset_index(drop=True)

        return df

    def get_top_n(self, n: int = 5) -> pd.DataFrame:
        """
        返回 TOPSIS 得分最高的 Top-N 方案。

        Parameters
        ----------
        n : int
            返回的方案数量。

        Returns:
            pd.DataFrame: Top-N 决策表。
        """
        df = self.rank()
        top_df = df.head(n).copy()

        # 如果有设计变量，拼接特征值
        if self.pareto_X is not None and len(self.feature_names) > 0:
            # 得分排序后的索引
            if self._scores is None:
                self.compute_scores()
            top_indices = np.argsort(self._scores)[::-1][:n]
            X_top = self.pareto_X[top_indices]
            for j, fname in enumerate(self.feature_names):
                if j < X_top.shape[1]:
                    vals = X_top[:, j]
                    if fname == "curing_temp":
                        vals = np.round(vals).astype(int)
                    top_df[fname] = np.round(vals, 2)

        return top_df

    def get_scores_detail(self) -> Dict:
        """
        返回 TOPSIS 计算的详细中间结果（用于调试/展示）。

        Returns:
            dict: 包含 scores, dist_to_pis, dist_to_nis, pis, nis 等。
        """
        if self._scores is None:
            self.compute_scores()
        return {
            "scores": self._scores,
            "dist_to_pis": self._dist_to_pis,
            "dist_to_nis": self._dist_to_nis,
            "pis": self._pis,
            "nis": self._nis,
            "weights": self.weights,
        }

    # ── 可视化 ────────────────────────────────────────────────────

    def plot_radar(
        self,
        top_indices: Optional[List[int]] = None,
        n_top: int = 5,
        title: str = "Top 配比方案雷达图",
    ):
        """
        绘制 Pareto 前沿 Top-N 方案的雷达图。

        每个方案的各目标值归一化到 [0,1] 后在雷达图中展示。

        Parameters
        ----------
        top_indices : list[int], optional
            指定要展示的方案在 pareto_Y 中的索引。
            为 None 时自动选取 TOPSIS 得分前 n_top 的方案。
        n_top : int
            选取的 Top 方案数（top_indices 为 None 时生效）。
        title : str
            图表标题。

        Returns:
            plotly.graph_objects.Figure or None
        """
        if not HAS_PLOTLY:
            print("plotly 未安装，无法生成雷达图")
            return None

        if self._scores is None:
            self.compute_scores()

        # 确定要展示的方案索引
        if top_indices is None:
            top_indices = np.argsort(self._scores)[::-1][:n_top]

        # 目标值归一化到 [0, 1]
        Y = self.pareto_Y.astype(np.float64)
        y_min = Y.min(axis=0)
        y_max = Y.max(axis=0)
        y_range = y_max - y_min
        y_range[y_range == 0] = 1.0

        # 标签（含方向指示）
        labels = []
        for j, name in enumerate(self.target_names):
            d = self.directions[j] if j < len(self.directions) else "maximize"
            arrow = {"maximize": "↑", "minimize": "↓", "target": "⊙"}.get(d, "")
            labels.append(f"{name}\n({arrow})")

        fig = go.Figure()
        colors = px.colors.qualitative.Set1

        for rank, idx in enumerate(top_indices):
            y_norm = (Y[idx] - y_min) / y_range
            color = colors[rank % len(colors)]
            score = self._scores[idx]

            fig.add_trace(
                go.Scatterpolar(
                    r=y_norm.tolist() + [y_norm[0]],  # 闭合
                    theta=labels + [labels[0]],
                    name=f"#{rank + 1} (得分: {score:.3f})",
                    fill="toself",
                    fillcolor=color,
                    opacity=0.25,
                    line=dict(color=color, width=2),
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False),
            ),
            title=title,
            showlegend=True,
            height=550,
            margin=dict(l=80, r=80, t=60, b=40),
        )

        return fig

    def plot_parallel_coordinates(
        self,
        highlight_top: int = 5,
        title: str = "Pareto 前沿平行坐标图",
    ):
        """
        绘制 Pareto 前沿的平行坐标图，高亮 Top-N 方案。

        Parameters
        ----------
        highlight_top : int
            高亮 TOPSIS 得分前 N 的方案。
        title : str
            图表标题。

        Returns:
            plotly.graph_objects.Figure or None
        """
        if not HAS_PLOTLY:
            print("plotly 未安装，无法生成平行坐标图")
            return None

        if self._scores is None:
            self.compute_scores()

        n, m = self.pareto_Y.shape

        # 构建平行坐标维度
        dimensions = []
        for j, name in enumerate(self.target_names):
            d = self.directions[j] if j < len(self.directions) else "maximize"
            arrow = {"maximize": "↑", "minimize": "↓", "target": "⊙"}.get(d, "")
            dim_label = f"{name} ({arrow})"
            dimensions.append(
                dict(
                    label=dim_label,
                    values=self.pareto_Y[:, j],
                )
            )

        # 确定高亮行
        highlight_rows = set(np.argsort(self._scores)[::-1][:highlight_top])

        # 构建颜色数组: 高亮 → 各自颜色, 其余 → 浅灰
        line_colors = []
        colors_palette = px.colors.qualitative.Set1
        for i in range(n):
            if i in highlight_rows:
                rank = list(highlight_rows).index(i)
                line_colors.append(colors_palette[rank % len(colors_palette)])
            else:
                line_colors.append("rgba(180,180,180,0.3)")

        # 构建自定义数据标签
        rank_labels = []
        for i in range(n):
            if i in highlight_rows:
                rank_labels.append(f"Rank #{list(highlight_rows).index(i)+1}")
            else:
                rank_labels.append("")

        dimensions.append(
            dict(
                label="TOPSIS得分",
                values=np.round(self._scores, 3),
            )
        )

        # 使用 go.Parcoords
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=self._scores,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="TOPSIS得分"),
                ),
                dimensions=dimensions,
                labelangle=-30,
            )
        )

        fig.update_layout(
            title=title,
            height=500,
            margin=dict(l=120, r=80, t=60, b=40),
        )

        return fig

    def plot_score_bar(
        self,
        n_top: int = 15,
        title: str = "TOPSIS 得分排名",
    ) -> "go.Figure":
        """
        绘制 TOPSIS 得分柱状图（Top-N 方案）。

        Parameters
        ----------
        n_top : int
            展示前 N 个方案。
        title : str
            图表标题。

        Returns:
            plotly.graph_objects.Figure or None
        """
        if not HAS_PLOTLY:
            print("plotly 未安装，无法生成得分柱状图")
            return None

        if self._scores is None:
            self.compute_scores()

        sorted_idx = np.argsort(self._scores)[::-1][:n_top]
        scores = self._scores[sorted_idx]
        ranks = [f"#{i + 1}" for i in range(len(scores))]

        n = len(scores)
        colors = [
            px.colors.sequential.Viridis[int(i * (len(px.colors.sequential.Viridis) - 1) / max(n - 1, 1))]
            for i in range(n)
        ]

        fig = go.Figure(
            go.Bar(
                x=ranks,
                y=scores,
                marker_color=colors,
                text=np.round(scores, 3),
                textposition="outside",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="方案排名",
            yaxis_title="TOPSIS 相对贴近度",
            yaxis=dict(range=[0, max(1.0, scores.max() * 1.15)]),
            height=420,
            margin=dict(l=40, r=20, t=50, b=40),
        )

        return fig


# ═══════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════

def create_topsis_selector(
    pareto_Y: np.ndarray,
    target_names: Optional[List[str]] = None,
    pareto_X: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
) -> TOPSISSelector:
    """
    便捷函数：从 config 读取 direction 并创建 TOPSISSelector。

    Parameters
    ----------
    pareto_Y : np.ndarray
        Pareto 前沿目标值矩阵。
    target_names : list[str], optional
        目标名称，默认从 config 读取全部 5 个。
    pareto_X : np.ndarray, optional
        对应的设计变量矩阵。
    feature_names : list[str], optional
        特征名列表。
    weights : np.ndarray, optional
        目标权重。

    Returns:
        TOPSISSelector
    """
    from core.config import get_all_target_names, get_all_feature_names

    if target_names is None:
        target_names = get_all_target_names()
    if feature_names is None:
        feature_names = get_all_feature_names()

    directions = [get_target_direction(t) for t in target_names]

    return TOPSISSelector(
        pareto_Y=pareto_Y,
        target_names=target_names,
        directions=directions,
        weights=weights,
        pareto_X=pareto_X,
        feature_names=feature_names,
    )
