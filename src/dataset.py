"""三模态数据集加载与预处理。

将 RGB(可见光) + Infrared(红外) + Depth(深度) 三模态拼接为 5 通道输入:
    channel 0~2 : RGB   (float, ImageNet 标准化)
    channel 3   : IR    (float, [0,1], 由 3 通道红外取灰度)
    channel 4   : Depth (float, [0,1], 16bit 毫米值归一化)

标签为 YOLO 格式 [class_id, cx, cy, w, h] (归一化)。
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet 均值/方差, 用于 RGB 通道标准化 (与 COCO 预训练权重一致)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(h0: int, w0: int, imgsz: int):
    """计算 letterbox 几何参数 (保持宽高比, 中心填充)。

    Returns:
        r, top, left, bottom, right, new_h, new_w
    """
    r = min(imgsz / h0, imgsz / w0)
    new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
    pad_h, pad_w = imgsz - new_h, imgsz - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    return r, top, left, bottom, right, new_h, new_w


def apply_letterbox(img: np.ndarray, top: int, bottom: int, left: int, right: int,
                    new_h: int, new_w: int, pad_value: int) -> np.ndarray:
    """对图像做 resize + 边框填充 (支持 uint8 / uint16)。"""
    interp = cv2.INTER_LINEAR if img.ndim == 2 or img.shape[2] == 3 else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """读取图像 (兼容中文/非 ASCII 路径, cv2.imread 在 Windows 下不支持中文路径)。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def load_depth(path: Path, max_val: int = 19999) -> np.ndarray:
    """加载深度图并归一化到 [0,1] float32 单通道。

    16bit PNG 单位毫米 (有效 [0, max_val]) 为主;
    若意外为彩色图 (如 jpg) 则取灰度并 /255。
    """
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:  # 意外为彩色图, 取灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 255.0:  # 16bit 深度
        img = np.clip(img, 0.0, float(max_val)) / float(max_val)
    else:  # 8bit 灰度
        img = img / 255.0
    return img


def load_infrared_gray(path: Path) -> np.ndarray:
    """加载红外图并归一化到 [0,1] float32 单通道。

    红外为 3 通道灰度堆叠, 取均值作为单通道。
    """
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0


def normalize_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float, ImageNet 标准化。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def box_to_letterbox(boxes: np.ndarray, r: float, left: int, top: int,
                     h0: int, w0: int, imgsz: int) -> np.ndarray:
    """将 [cx, cy, w, h] 归一化(相对原图) 变换到 letterbox 后 (相对 imgsz)。"""
    boxes = boxes.copy().astype(np.float32)
    # 转像素
    boxes[:, 0] *= w0
    boxes[:, 1] *= h0
    boxes[:, 2] *= w0
    boxes[:, 3] *= h0
    # 缩放 + 偏移
    boxes[:, 0] = boxes[:, 0] * r + left
    boxes[:, 1] = boxes[:, 1] * r + top
    boxes[:, 2] *= r
    boxes[:, 3] *= r
    # 归一化到 imgsz
    boxes /= imgsz
    return boxes


def box_from_letterbox(boxes: np.ndarray, r: float, left: int, top: int,
                       h0: int, w0: int, imgsz: int) -> np.ndarray:
    """letterbox 后 (相对 imgsz) 的 [cx, cy, w, h] 反变换回原图归一化。"""
    boxes = boxes.copy().astype(np.float32)
    boxes *= imgsz  # 转像素
    boxes[:, 0] = (boxes[:, 0] - left) / r
    boxes[:, 1] = (boxes[:, 1] - top) / r
    boxes[:, 2] /= r
    boxes[:, 3] /= r
    boxes[:, 0] /= w0
    boxes[:, 1] /= h0
    boxes[:, 2] /= w0
    boxes[:, 3] /= h0
    return boxes


class TriModalDataset(Dataset):
    """三模态目标检测数据集 (训练/验证)。

    Args:
        root: 数据根目录, 含 visible/infrared/depth/labels 子目录。
        imgsz: letterbox 目标尺寸。
        names: 类别字典 {id: name}。
        train: True 用于训练 (启用水平翻转增强), False 用于验证。
    """

    def __init__(self, root: Path, imgsz: int = 640, max_depth: int = 19999,
                 train: bool = True, seed: int = 42):
        self.root = Path(root)
        self.imgsz = imgsz
        self.max_depth = max_depth
        self.train = train

        visible_dir = self.root / "visible"
        infrared_dir = self.root / "infrared"
        depth_dir = self.root / "depth"
        label_dir = self.root / "labels"

        # 以 visible 目录为基准收集样本 stem (三模态+标签按文件名 stem 对齐)
        self.samples = sorted(p.stem for p in visible_dir.glob("*")
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(self.samples) == 0:
            raise FileNotFoundError(f"未在 {visible_dir} 找到任何图像")

        self._visible_dir = visible_dir
        self._infrared_dir = infrared_dir
        self._depth_dir = depth_dir
        self._label_dir = label_dir

        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _find_file(self, directory: Path, stem: str) -> Path:
        """在目录中按 stem 查找文件 (扩展名可能不同)。"""
        for p in directory.glob(stem + ".*"):
            return p
        raise FileNotFoundError(f"未找到 {stem} 在 {directory}")

    def _load_labels(self, stem: str) -> np.ndarray:
        """加载 YOLO 标签, 返回 (N, 5) [cls, cx, cy, w, h] 归一化。

        测试集无标签目录时返回空数组 (无目标)。
        """
        try:
            label_path = self._find_file(self._label_dir, stem)
        except FileNotFoundError:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = []
        if label_path.exists() and label_path.stat().st_size > 0:
            for line in label_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                    boxes.append([cls, cx, cy, w, h])
        return np.array(boxes, dtype=np.float32).reshape(-1, 5) if boxes else np.zeros((0, 5), dtype=np.float32)

    def __getitem__(self, idx: int):
        stem = self.samples[idx]

        # 加载三模态原始图像
        rgb = imread_unicode(self._find_file(self._visible_dir, stem))
        h0, w0 = rgb.shape[:2]

        ir_gray = load_infrared_gray(self._find_file(self._infrared_dir, stem))
        depth = load_depth(self._find_file(self._depth_dir, stem), self.max_depth)

        # letterbox 几何参数 (以 RGB 尺寸为准, 三模态空间对齐)
        r, top, left, bottom, right, new_h, new_w = letterbox(h0, w0, self.imgsz)

        # 各模态分别 letterbox
        rgb_lb = apply_letterbox(rgb, top, bottom, left, right, new_h, new_w, pad_value=114)
        ir_lb = apply_letterbox(ir_gray, top, bottom, left, right, new_h, new_w, pad_value=0)
        depth_lb = apply_letterbox(depth, top, bottom, left, right, new_h, new_w, pad_value=0)

        # 标签: 归一化 -> letterbox 归一化
        labels = self._load_labels(stem)
        if len(labels):
            boxes = box_to_letterbox(labels[:, 1:], r, left, top, h0, w0, self.imgsz)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        # 训练时水平翻转增强
        if self.train and self.rng.random() < 0.5:
            rgb_lb = np.ascontiguousarray(rgb_lb[:, ::-1])
            ir_lb = np.ascontiguousarray(ir_lb[:, ::-1])
            depth_lb = np.ascontiguousarray(depth_lb[:, ::-1])
            if len(boxes):
                boxes[:, 0] = 1.0 - boxes[:, 0]

        # 归一化 + 拼接为 5 通道 (C, H, W)
        rgb_norm = normalize_rgb(rgb_lb)          # (H, W, 3)
        ir_norm = ir_lb[..., None]                 # (H, W, 1)
        depth_norm = depth_lb[..., None]           # (H, W, 1)
        img = np.concatenate([rgb_norm, ir_norm, depth_norm], axis=-1)  # (H, W, 5)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

        cls = torch.from_numpy(labels[:, 0].astype(np.int64)) if len(labels) else torch.empty(0, dtype=torch.int64)
        boxes = torch.from_numpy(boxes)

        return {"img": img, "cls": cls, "boxes": boxes, "stem": stem,
                "meta": (r, left, top, h0, w0)}

    @staticmethod
    def collate_fn(batch):
        """组装一个 batch, 输出与 v8DetectionLoss 期望一致。

        Returns:
            img: (bs, 5, H, W)
            batch_idx: (M, 1)
            cls: (M, 1)
            bboxes: (M, 4)  xywh 归一化
            (以及 stems / metas 用于推理)
        """
        imgs = torch.stack([b["img"] for b in batch], 0)
        batch_idx_list, cls_list, box_list = [], [], []
        for i, b in enumerate(batch):
            n = b["cls"].shape[0]
            if n:
                batch_idx_list.append(torch.full((n, 1), i, dtype=torch.int64))
                cls_list.append(b["cls"].view(-1, 1))
                box_list.append(b["boxes"])
        if batch_idx_list:
            batch_idx = torch.cat(batch_idx_list, 0)
            cls = torch.cat(cls_list, 0)
            bboxes = torch.cat(box_list, 0)
        else:
            batch_idx = torch.zeros((0, 1), dtype=torch.int64)
            cls = torch.zeros((0, 1), dtype=torch.int64)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
        return {"img": imgs, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes,
                "stems": [b["stem"] for b in batch], "metas": [b["meta"] for b in batch]}
