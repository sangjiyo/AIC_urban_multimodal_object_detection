"""打包提交: 将预测 txt 目录打包为 zip。

用法示例:
    py -m src.pack --pred-dir runs/predict --out submission.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="打包预测结果为提交 zip")
    p.add_argument("--pred-dir", type=str, required=True, help="预测 txt 目录")
    p.add_argument("--out", type=str, default="submission.zip", help="输出 zip 路径")
    return p.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = Path.cwd() / pred_dir
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out

    files = sorted(pred_dir.glob("*.txt"))
    if not files:
        print(f"[警告] {pred_dir} 下没有找到任何 txt 文件")
        return
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    print(f"[完成] 已打包 {len(files)} 个 txt 到 {out}")


if __name__ == "__main__":
    main()
