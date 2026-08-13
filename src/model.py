"""多通道 YOLOv8 模型构建与预训练权重加载。

基于 ultralytics 的 DetectionModel, 将输入通道改为 5 通道
(RGB 3 + IR 1 + Depth 1)。加载 COCO 预训练权重时, ultralytics
的 model.load() 会自动把 RGB 前 3 通道的卷积核复制到多通道首层,
其余通道随机初始化 (见 tasks.py BaseModel.load 的 first_conv 处理)。
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


def build_model(cfg: dict, pretrained: str | None = None) -> DetectionModel:
    """构建 5 通道 YOLOv8 检测模型。

    Args:
        cfg: 完整配置 dict (configs/aic_trimodal.yaml)。
        pretrained: 覆盖 cfg 中的预训练权重 (None 表示使用 cfg 值; "" 表示不加载)。

    Returns:
        DetectionModel 实例 (train 模式)。
    """
    mc = cfg["model"]
    nc = int(cfg["nc"])

    model = DetectionModel(mc["yaml"], ch=int(mc["ch"]), nc=nc, verbose=True)

    # 类别名
    model.names = {int(k): v for k, v in cfg["names"].items()}

    # 损失增益 (v8DetectionLoss 需要 model.args.box/cls/dfl)
    t = cfg.get("train", {})
    model.args = SimpleNamespace(
        box=float(t.get("box_gain", 7.5)),
        cls=float(t.get("cls_gain", 0.5)),
        dfl=float(t.get("dfl_gain", 1.5)),
    )

    # 加载 COCO 预训练权重 (自动处理多通道首层)
    pretrained = pretrained if pretrained is not None else mc.get("pretrained")
    if pretrained:
        src = YOLO(pretrained).model  # 下载/加载 COCO 权重
        model.load(src, verbose=True)

    return model


def build_criterion(model: DetectionModel) -> v8DetectionLoss:
    """构建 YOLOv8 检测损失。"""
    return v8DetectionLoss(model)


def load_checkpoint(model: DetectionModel, path: str | None, device: torch.device) -> DetectionModel:
    """加载训练好的权重 (完整 state_dict)。"""
    if path:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        model.load_state_dict(state, strict=True)
        print(f"已加载权重: {path}")
    return model.to(device)
