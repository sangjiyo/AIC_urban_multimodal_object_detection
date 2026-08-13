# 面向城市场景的视觉多模态目标检测

全球校园人工智能算法精英大赛 · 算法挑战赛 —— 基于 **RGB(可见光) + Infrared(红外) + Depth(深度)** 三模态数据融合的目标检测。

## 赛题要点

- **任务**：对三模态空间对齐图像中的关键目标做精准定位与分类。
- **类别**（12 类，编号 0~11）：`person, boat, animal, seat, sign, bicycle, car, ball, light, garbage can, uav, tricycle`
- **标签格式**：YOLO `[class_id, cx, cy, w, h]`（归一化）
- **输出格式**：`[class_id, cx, cy, w, h, confidence]`，每张图一个同名 txt，最大 100 个框
- **评测指标**：`mAP@50-95`（IoU 0.50~0.95 步长 0.05，101 点插值 AP）
- **约束**：禁止在线服务/API、禁止手工标注、禁止多模型简单集成投票/平均；允许 ImageNet/COCO/Objects365 预训练权重

## 数据说明

| 模态 | 格式 | 数值范围 |
|------|------|----------|
| RGB 可见光 | 3 通道 uint8 | [0, 255] |
| Infrared 红外 | 3 通道 uint8（灰度堆叠，无彩色语义） | [0, 255]，越大越热 |
| Depth 深度 | 单通道 uint16（mm） | 有效 [0, 19999]，0 为无效 |

数据目录结构（样例）：`visible/` `infrared/` `depth/` `labels/` 四个子目录，按文件名 stem 对齐。

## 方案设计

**多通道拼接融合**：将三模态编码为 5 通道输入，送入单一 YOLOv8 网络：

```
channel 0~2 : RGB  (ImageNet 标准化)
channel 3   : 红外灰度 [0,1]
channel 4   : 深度归一化 [0,1]
```

YOLOv8 首层卷积从 3 通道改为 5 通道，加载 COCO 预训练权重时自动把 RGB 前三通道卷积核复制过去、新增通道随机初始化（ultralytics 内置支持）。

## 项目结构

```
configs/aic_trimodal.yaml    # 类别、数据路径、训练/推理超参
src/
  config.py                  # 配置加载
  dataset.py                 # 三模态加载 / letterbox / 增强 / 5 通道拼接
  model.py                   # 多通道 YOLOv8 构建 + 预训练权重加载
  metrics.py                 # mAP@50-95 评测 (101 点插值)
  infer.py                   # 推理 + NMS 后处理
  train.py                   # 训练
  predict.py                 # 测试集推理
  evaluate.py                # 本地评测
  pack.py                    # 打包提交
```

## 快速开始

### 1. 安装依赖

```bash
# CUDA 版 torch (按显卡 CUDA 版本选择, RTX 30 系可用 cu121)
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
py -m pip install -r requirements.txt
```

### 2. 配置数据

编辑 `configs/aic_trimodal.yaml` 的 `data.root` 指向正式数据目录（含 `visible/infrared/depth/labels`）。

### 3. 训练

```bash
py -m src.train --epochs 100 --batch-size 8 --imgsz 640
```

4GB 显存建议 `--batch-size 4`（或模型改 `yolov8s.yaml` 时用更小 batch）。

### 4. 推理（生成提交结果）

```bash
py -m src.predict --weights runs/train/best.pt --out-dir runs/predict
py -m src.pack --pred-dir runs/predict --out submission.zip
```

### 5. 本地评测（有标签时）

```bash
py -m src.evaluate --pred-dir runs/predict --gt-dir 数据目录/labels
```

## 后续优化方向

- **融合方式**：从简单通道拼接升级为参考论文中的通道切换 + 空间注意力（CVCAM, CVPR 2023）或跨模态 Transformer 融合。
- **数据增强**：加入 Mosaic、随机缩放、多尺度训练，增强泛化。
- **深度图处理**：改用对数映射或反距离（1/z）编码，弱化远距离噪声。
- **红外通道**：保留 3 通道而非取灰度，或与 RGB 做注意力加权。
- **模型**：YOLOv8s/m、更大输入尺寸、或换用更强 backbone。
