"""推理与提交结果生成。

模型输出经 NMS 后处理, 转换为赛题要求格式:
    [class_id, cx, cy, w, h, confidence]  (原图归一化坐标)
"""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision.ops import nms

from .dataset import TriModalDataset, box_from_letterbox


@torch.no_grad()
def predict_one(model, img: torch.Tensor, meta, conf_thres: float = 0.25,
                iou_thres: float = 0.7, max_det: int = 100) -> list:
    """对单张图推理, 返回 list[(cls, cx, cy, w, h, conf)] (原图归一化)。

    Args:
        model: DetectionModel (eval 模式)。
        img: (5, H, W) tensor (已在 dataset 中预处理)。
        meta: (r, left, top, h0, w0) letterbox 参数。
    """
    model.eval()
    r, left, top, h0, w0 = meta
    imgsz = img.shape[-1]

    out = model(img.unsqueeze(0))  # (y, preds) 或 y
    y = out[0] if isinstance(out, tuple) else out
    y = y[0]  # (4+nc, num_anchors)

    # YOLOv8 推理输出: 前 4 行为 [cx, cy, w, h] 绝对像素 (相对 letterbox 图 imgsz)
    boxes_xywh = y[:4].T  # (N, 4) 像素
    scores = y[4:].T      # (N, nc)
    nc = scores.shape[1]

    # xywh -> xyxy (像素), 供 NMS 使用
    boxes_xyxy = torch.stack([
        boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0,
        boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0,
        boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0,
        boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0,
    ], dim=1)

    results = []
    for cls in range(nc):
        confs = scores[:, cls]
        keep = confs > conf_thres
        if not keep.any():
            continue
        b = boxes_xyxy[keep]
        c = confs[keep]
        nms_idx = nms(b, c, iou_thres)
        b, c = b[nms_idx], c[nms_idx]

        # xyxy(像素) -> cx cy w h(像素)
        cx = (b[:, 0] + b[:, 2]) / 2.0
        cy = (b[:, 1] + b[:, 3]) / 2.0
        w = (b[:, 2] - b[:, 0])
        h = (b[:, 3] - b[:, 1])
        boxes_xywh = torch.stack([cx, cy, w, h], dim=1).cpu().numpy()

        # 像素 -> 归一化(相对 imgsz) -> 反变换回原图归一化
        boxes_norm = box_from_letterbox(boxes_xywh / float(imgsz), r, left, top, h0, w0, imgsz)
        for i in range(len(c)):
            ccx, ccy, ww, hh = boxes_norm[i]
            # 过滤越界框
            if ww <= 0 or hh <= 0:
                continue
            ccx = float(min(max(ccx, 0.0), 1.0))
            ccy = float(min(max(ccy, 0.0), 1.0))
            results.append((cls, ccx, ccy, float(ww), float(hh), float(c[i])))

    results.sort(key=lambda x: -x[5])
    return results[:max_det]


def write_prediction(path: Path, preds: list):
    """将预测结果写为赛题格式 txt。"""
    lines = []
    for cls, cx, cy, w, h, conf in preds:
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_inference(model, dataset: TriModalDataset, out_dir: Path,
                  conf_thres: float = 0.25, iou_thres: float = 0.7,
                  max_det: int = 100, device=None) -> Path:
    """对数据集内所有样本推理, 写出同名 txt 到 out_dir。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        preds = predict_one(model, sample["img"].to(device), sample["meta"],
                            conf_thres, iou_thres, max_det)
        write_prediction(out_dir / (sample["stem"] + ".txt"), preds)

    return out_dir
