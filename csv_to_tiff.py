# -*- coding: utf-8 -*-
"""CSV 网格数据转 TIFF 图片的小工具。

用法:
    python csv_to_tiff.py input.csv output.tiff [--mode rgb] [--cmap jet]

依赖：matplotlib + tifffile（rgb/float 模式）：pip install matplotlib tifffile
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np
from PIL import Image


def csv_to_tiff(csv_file, tiff_file):
    # 读取 CSV（跳过表头，如果是纯数值可不加）
    data = np.genfromtxt(csv_file, delimiter=',')
    
    # data 是二维数组 (rows, cols)
    data = data.astype(np.float32)

    # 归一化到 0–255（灰度图）
    data_norm = (
        (data - data.min()) 
        / (data.max() - data.min()) 
        * 255
    ).astype(np.uint8)

    # 保存为 TIFF
    Image.fromarray(data_norm, mode='L').save(tiff_file)

    print(f"TIFF 已保存: {tiff_file}")
    print(f"尺寸: {data.shape[1]} x {data.shape[0]}")
    
def csv_to_tiff_float(csv_file, tiff_file):
    import tifffile as tiff

    data = np.genfromtxt(csv_file, delimiter=',')
    data = data.astype(np.float32)
    tiff.imwrite(tiff_file, data)
    
    

def csv_to_rgb_tiff(
    csv_file,
    tiff_file,
    cmap_name='jet',
    vmin=None,
    vmax=None
):
    import matplotlib.pyplot as plt
    import tifffile as tiff

    # 1. 读取 CSV
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    data = data.astype(np.float32)

    # 2. 自动截断异常值（防止一片白）
    if vmin is None:
        vmin = np.percentile(data, 1)
    if vmax is None:
        vmax = np.percentile(data, 99)

    data_clip = np.clip(data, vmin, vmax)

    # 3. 应用 colormap
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(data_clip))          # (ny, nx, 4)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)  # 丢弃 Alpha

    # 4. 保存 RGB TIFF
    tiff.imwrite(tiff_file, rgb)

    print(f"✅ RGB TIFF 已保存: {tiff_file}")
    print(f"   尺寸: {rgb.shape[1]} x {rgb.shape[0]}")
    print(f"   色标: {cmap_name}")
    print(f"   数值范围: [{vmin:.4f}, {vmax:.4f}]")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CSV 网格数据转 TIFF 图片')
    parser.add_argument('input', help='输入 .csv 文件路径')
    parser.add_argument('output', help='输出 .tiff 文件路径')
    parser.add_argument('--mode', choices=['gray', 'float', 'rgb'], default='rgb',
                        help='gray=8bit 灰度, float=float32 原值, rgb=彩色（默认）')
    parser.add_argument('--cmap', default='jet', help='rgb 模式颜色映射名称')
    parser.add_argument('--vmin', type=float, default=None,
                        help='rgb 模式色标下限（默认取 1%% 分位数）')
    parser.add_argument('--vmax', type=float, default=None,
                        help='rgb 模式色标上限（默认取 99%% 分位数）')
    args = parser.parse_args()

    if args.mode == 'gray':
        csv_to_tiff(args.input, args.output)
    elif args.mode == 'float':
        csv_to_tiff_float(args.input, args.output)
    else:
        csv_to_rgb_tiff(args.input, args.output, cmap_name=args.cmap,
                        vmin=args.vmin, vmax=args.vmax)