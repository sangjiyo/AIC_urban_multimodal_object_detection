"""Smoke test: 验证双流融合模型构建 + 前向 + 权重复用。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config, resolve_data_root
from src.dataset import TriModalDataset
from src.model import ModalityFusionStem, build_fusion_model, build_criterion


def main():
    cfg = load_config()
    device = torch.device("cuda:0")

    fusion = build_fusion_model(cfg).to(device)
    assert isinstance(fusion.model[0], ModalityFusionStem), f"首层未替换: {type(fusion.model[0])}"
    print("[模型] 首层已替换为 ModalityFusionStem")

    # 权重复用检查: rgb_stem 权重 == 标准模型首层 RGB 前 3 通道
    from src.model import build_model
    std = build_model(cfg)
    std_conv = std.model[0].conv
    rgb_w = fusion.model[0].rgb_stem.conv.weight
    assert torch.allclose(rgb_w.cpu(), std_conv.weight[:, :3]), "RGB 流权重未复用"
    assert fusion.model[0].aux_stem.bn.weight.abs().sum().item() == 0.0, "残差分支 gamma 未置 0"
    print("[权重] RGB 流复用 COCO 权重, IR/Depth 残差分支 gamma=0")

    # 前向 + loss
    root = resolve_data_root(cfg)
    ds = TriModalDataset(root, imgsz=640, max_depth=cfg["depth"]["max_val"], train=True, seed=42)
    criterion = build_criterion(fusion)
    batch = TriModalDataset.collate_fn([ds[i] for i in range(4)])
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items() if k in ("img", "batch_idx", "cls", "bboxes")}
    preds = fusion(batch["img"])
    loss, _ = criterion(preds, batch)
    print(f"[模型] 融合模型 forward + loss OK, loss = {loss.sum().item():.3f}")

    # eval 模式输出格式 (与标准模型一致)
    fusion.eval()
    x = torch.randn(1, 5, 640, 640, device=device)
    with torch.no_grad():
        out = fusion(x)
    y = out[0] if isinstance(out, (tuple, list)) else out
    print(f"[模型] eval 输出 shape: {[t.shape for t in y] if isinstance(y, (tuple, list)) else y.shape}")


if __name__ == "__main__":
    main()
