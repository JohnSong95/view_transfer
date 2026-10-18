"""将 Tecplot .plt 文件转换为彩色 TIFF 图片。

用法:
    python plt_to_tiff.py input.plt [output.tiff] [--cmap jet]

依赖 matplotlib（颜色映射）；tifffile 不需要。
"""
import os
import re
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np


def plt_to_tiff(input_path, output_path, cmap='jet'):
    """读取 Tecplot .plt 文件并保存为彩色 TIFF 图片。

    Parameters
    ----------
    input_path : str   .plt 文件路径
    output_path : str  输出的 .tiff 路径
    cmap : str         matplotlib 颜色映射名称（如 'jet', 'turbo', 'viridis'）
    """
    import matplotlib.pyplot as plt

    print(f'Reading: {input_path}')
    # 从 ZONE 行解析 I, J 维度
    with open(input_path, 'r') as f:
        f.readline()
        f.readline()
        zone_line = f.readline()
    m_i = re.search(r'I\s*=\s*(\d+)', zone_line)
    m_j = re.search(r'J\s*=\s*(\d+)', zone_line)
    if m_i and m_j:
        x_count = int(m_i.group(1))
        y_count = int(m_j.group(1))
        print(f'  Dimensions from header: I={x_count}, J={y_count}')
    else:
        # 回退到 max 值
        M = np.loadtxt(input_path, skiprows=3)
        x_count = int(M[:, 0].max()) + 1
        y_count = int(M[:, 1].max()) + 1

    # 重新读取数据
    M = np.loadtxt(input_path, skiprows=3)
    n_expected = x_count * y_count
    n_actual = M.shape[0]
    print(f'  Data points: {n_actual} (expected {n_expected})')

    if n_actual < n_expected:
        # 数据不足，补 0
        h = np.zeros(n_expected)
        h[:n_actual] = M[:, 2]
    else:
        h = M[:n_expected, 2]
    h_2d = h.reshape(y_count, x_count)

    # 越界值置 0（与 read_file.py 一致）
    eta = np.where((h_2d > 1) | (h_2d < 0), 0.0, h_2d)

    # 归一化到 [0, 1] 用于 colormap
    vmin, vmax = eta.min(), eta.max()
    if vmax - vmin > 0:
        eta_norm = (eta - vmin) / (vmax - vmin)
    else:
        eta_norm = np.zeros_like(eta)

    # 应用 colormap 得到 RGBA（shape: H x W x 4）
    colormap = plt.get_cmap(cmap)
    rgba = colormap(eta_norm)
    # 转 0-255 uint8
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # PIL 约定 row 0 在图像顶部，而 .plt 数据 Y=1 通常在数学坐标底部，
    # 为与科学坐标约定（Y 向上）一致，垂直翻转数组
    #rgb = rgb[::-1, :, :]

    # 使用 PIL 保存 TIFF
    try:
        from PIL import Image
        img = Image.fromarray(rgb, 'RGB')
        img.save(output_path, 'TIFF')
        print(f'Saved: {output_path}  (shape={rgb.shape})')
    except ImportError:
        # 无 PIL 则用 matplotlib imsave（origin 默认 upper，也需翻转）
        plt.imsave(output_path, eta[::-1, :], cmap=cmap)
        print(f'Saved (via matplotlib): {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将 Tecplot .plt 转换为彩色 TIFF 图片')
    parser.add_argument('input', help='输入 .plt 文件路径')
    parser.add_argument('output', nargs='?', default=None,
                        help='输出 .tiff 路径（默认与输入同名，后缀 _color.tiff）')
    parser.add_argument('--cmap', default='jet',
                        help='颜色映射名称（如 jet, turbo, viridis）')
    args = parser.parse_args()

    out_file = args.output or os.path.splitext(args.input)[0] + '_color.tiff'
    plt_to_tiff(args.input, out_file, cmap=args.cmap)
