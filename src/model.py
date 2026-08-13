"""多通道 YOLOv8 模型构建与预训练权重加载。

基于 ultralytics 的 DetectionModel, 将输入通道改为 5 通道
(RGB 3 + IR 1 + Depth 1)。加载 COCO 预训练权重时, ultralytics
的 model.load() 会自动把 RGB 前 3 通道的卷积核复制到多通道首层,
其余通道随机初始化 (见 tasks.py BaseModel.load 的 first_conv 处理)。
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class ModalityFusionStem(nn.Module):
    """双流模态融合 Stem (RGB 主分支 + IR/Depth 残差分支)。

    输入 5 通道 [RGB(3), IR(1), Depth(1)], 输出 out 通道 (与 yolov8 首层一致, 32)。
    RGB 流复用 COCO 预训练权重, IR/Depth 流作为残差从 0 开始学习, 逐步补充
    互补信息。add 融合使 backbone 后续结构与标准 yolov8 完全一致, 可直接
    复用全部预训练权重与 Detect head。
    """

    def __init__(self, in_rgb: int = 3, in_aux: int = 2, out: int = 32):
        super().__init__()
        self.rgb_stem = Conv(in_rgb, out, 3, 2)
        self.aux_stem = Conv(in_aux, out, 3, 2)
        # ultralytics 前向循环依赖 m.f / m.i, 首层固定 from=-1, index=0
        self.f = -1
        self.i = 0

    def forward(self, x):
        return self.rgb_stem(x[:, :3]) + self.aux_stem(x[:, 3:])


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


def build_fusion_model(cfg: dict, pretrained: str | None = None) -> DetectionModel:
    """构建双流融合 YOLOv8 (RGB backbone + IR/Depth 残差分支)。

    在标准 5 通道 YOLOv8 基础上, 将首层 Conv 替换为 ModalityFusionStem:
    - RGB 流复用 COCO 预训练权重 (原首层 Conv 的前 3 通道卷积核)。
    - IR/Depth 流随机初始化, 且其 BN 的 gamma 置 0, 使残差从 0 起步,
      训练初期等价于纯 RGB 模型, 避免破坏预训练特征。
    """
    mc = cfg["model"]
    nc = int(cfg["nc"])

    model = DetectionModel(mc["yaml"], ch=int(mc["ch"]), nc=nc, verbose=False)
    model.names = {int(k): v for k, v in cfg["names"].items()}

    t = cfg.get("train", {})
    model.args = SimpleNamespace(
        box=float(t.get("box_gain", 7.5)),
        cls=float(t.get("cls_gain", 0.5)),
        dfl=float(t.get("dfl_gain", 1.5)),
    )

    # 先加载 COCO 预训练权重到标准首层 (自动处理 5 通道的 RGB 前 3 通道复制)
    pretrained = pretrained if pretrained is not None else mc.get("pretrained")
    if pretrained:
        src = YOLO(pretrained).model
        model.load(src, verbose=True)

    # 替换首层为双流融合 stem, RGB 流复用原首层 RGB 权重
    old_conv = model.model[0].conv  # nn.Conv2d(5, out_ch, 3, 3, stride 2)
    out_ch = old_conv.weight.shape[0]
    stem = ModalityFusionStem(in_rgb=3, in_aux=2, out=out_ch)
    with torch.no_grad():
        stem.rgb_stem.conv.weight.copy_(old_conv.weight[:, :3])
        if old_conv.bias is not None:
            stem.rgb_stem.conv.bias.copy_(old_conv.bias)
        stem.aux_stem.bn.weight.zero_()  # IR/Depth 残差分支从 0 开始
    model.model[0] = stem
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
