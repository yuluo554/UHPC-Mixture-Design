# -*- coding: utf-8 -*-
"""
UHPC-AutoDesign 全局配置模块
载入 YAML 参数模板，提供统一参数访问接口
"""

import os
import yaml

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(ROOT_DIR, "templates", "uhpc_params.yaml")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# ─── 懒加载 YAML 配置 ───────────────────────────────────────────
_config_cache = None


def _load_config():
    """加载 UHPC 参数模板 YAML"""
    global _config_cache
    if _config_cache is None:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_composition_params():
    """返回组成参数列表 [{name, label, range, unit, step}, ...]"""
    return _load_config()["composition"]


def get_fiber_params():
    """返回钢纤维参数列表"""
    return _load_config()["fiber"]


def get_curing_params():
    """返回养护参数列表"""
    return _load_config()["curing"]


def get_targets():
    """返回优化目标列表"""
    return _load_config()["targets"]


def get_constraints():
    """返回物理约束列表"""
    return _load_config()["constraints"]


def get_all_feature_names():
    """返回所有输入特征的 name 列表"""
    cfg = _load_config()
    names = []
    for group in ["composition", "fiber", "curing"]:
        for item in cfg[group]:
            names.append(item["name"])
    return names


def get_all_target_names():
    """返回所有优化目标的 name 列表"""
    return [t["name"] for t in _load_config()["targets"]]


def get_feature_ranges():
    """
    返回 {feature_name: (min, max)} 字典
    仅对数值型参数有效，分类型参数不包含
    """
    cfg = _load_config()
    ranges = {}
    for group in ["composition", "fiber", "curing"]:
        for item in cfg[group]:
            if "range" in item:
                ranges[item["name"]] = tuple(item["range"])
    return ranges


def get_categorical_features():
    """返回分类型特征名列表"""
    cfg = _load_config()
    cats = []
    for group in ["composition", "fiber", "curing"]:
        for item in cfg[group]:
            if item.get("type") == "categorical":
                cats.append(item["name"])
    return cats


def get_feature_label(name):
    """根据 name 返回中文 label（输入特征：组成/纤维/养护）"""
    for group in ["composition", "fiber", "curing"]:
        for item in _load_config()[group]:
            if item["name"] == name:
                return item.get("label", name)
    return name


def get_target_label(name):
    """根据 name 返回目标的中文 label（找不到则返回原名）

    与 ``get_feature_label`` 配对使用 — 后者只查 composition/fiber/curing 三组，
    目标（compressive_strength_28d 等）永远查不到，需要单独查询。
    """
    for t in _load_config()["targets"]:
        if t["name"] == name:
            return t.get("label", name)
    return name


def get_target_direction(name):
    """返回优化方向: 'maximize' / 'minimize' / 'target'"""
    for t in _load_config()["targets"]:
        if t["name"] == name:
            return t["direction"]
    return None


def reload_config():
    """强制重新加载配置"""
    global _config_cache
    _config_cache = None
    return _load_config()
