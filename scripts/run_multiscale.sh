#!/bin/bash
# 多尺度渐进训练: 从 cw_s/best.pt (640, val 0.2762) 渐进到 832。
# 每步 +64 (+10%), 避免一步跳导致的 DFL 宽高回归崩溃。
set -e
cd "$(dirname "$0")/.."

PREV="runs/cw_s/best.pt"
for STAGE in 704 768 832; do
  SAVE="runs/ms_$STAGE"
  echo "=============================================================="
  echo "== 渐进阶段 imgsz=$STAGE  init=$PREV  save=$SAVE"
  echo "=============================================================="
  py -m src.train --config configs/aic_trimodal_s.yaml --fusion \
    --imgsz "$STAGE" --epochs 6 --lr0 0.001 --init-weights "$PREV" \
    --save-dir "$SAVE" --batch-size 2
  PREV="$SAVE/best.pt"
done

echo "=============================================================="
echo "== 渐进训练完成: 最终权重 runs/ms_832/best.pt"
echo "=============================================================="
