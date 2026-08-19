#!/bin/bash
# yolov8x 多尺度渐进训练 (复刻 yolov8l 路径, 模型换 x)。
# 起点 train_x/best.pt (640), 每步 +128 渐进到 1024。
# 每步 6 epoch, lr0=0.001, 不冻结; batch 随尺度递减避免 OOM (x 比 l 更大)。
# 用法: bash run_multiscale_x.sh [device_id]   (默认 device 0)
set -e
cd /data1/mazc/sangjiyo/AIC_urban_multimodal_object_detection
PY=/home/mazc/anaconda3/bin/python
DEVICE=${1:-0}

PREV="runs/train_x/best.pt"
for STAGE in 768 896 1024; do
  case $STAGE in
    768)  BATCH=6;;
    896)  BATCH=4;;
    1024) BATCH=3;;   # x 比 l 更大, 1024 尺度 batch 3; 若 OOM 降 2
  esac
  SAVE="runs/ms_x_$STAGE"
  echo "=============================================================="
  echo "== yolov8x 渐进 imgsz=$STAGE  batch=$BATCH  init=$PREV  device=$DEVICE"
  echo "=============================================================="
  $PY -m src.train --config configs/aic_trimodal_x.yaml --fusion \
    --imgsz "$STAGE" --epochs 6 --lr0 0.001 --init-weights "$PREV" \
    --save-dir "$SAVE" --batch-size "$BATCH" --device "$DEVICE"
  PREV="$SAVE/best.pt"
done

echo "=============================================================="
echo "== yolov8x 多尺度渐进完成: 最终权重 $PREV"
echo "=============================================================="
