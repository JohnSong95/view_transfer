# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tiff
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

csv_to_rgb_tiff(r"C:\Users\A\Downloads\传热系数-叶栅-主流湍流度5%\吸力面第三排孔后区域视角\叶珊试验件0-吸力面原始状态\air密度比1.0\吹风比1.5\h3.csv", r"C:\Users\A\Downloads\传热系数-叶栅-主流湍流度5%\吸力面第三排孔后区域视角\叶珊试验件0-吸力面原始状态\air密度比1.0\吹风比1.5\h3_color.tiff",
    cmap_name="jet")