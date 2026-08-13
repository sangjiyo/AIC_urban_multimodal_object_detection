"""配置加载与路径解析工具。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# 项目根目录 (src/ 的上一级)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "aic_trimodal.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """加载 YAML 配置文件，返回 dict。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_data_root(cfg: dict) -> Path:
    """解析数据根目录 (支持相对项目根目录的路径)。"""
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def resolve_test_root(cfg: dict) -> Path:
    """解析测试集根目录 (默认回退到训练集根目录)。"""
    root = Path(cfg["data"].get("test_root", cfg["data"]["root"]))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def class_names(cfg: dict) -> dict:
    """返回 {class_id: name}。"""
    return {int(k): v for k, v in cfg["names"].items()}
