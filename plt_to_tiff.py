"""将 data1/Eta.plt 转换为彩色 TIFF 图片。

用法:
    python plt_to_tiff.py
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def plt_to_tiff(input_path, output_path, cmap='jet'):
    """读取 Tecplot .plt 文件并保存为彩色 TIFF 图片。

    Parameters
    ----------
    input_path : str   .plt 文件路径
    output_path : str  输出的 .tiff 路径
    cmap : str         matplotlib 颜色映射名称（如 'jet', 'turbo', 'viridis'）
    """
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
    folder = r'C:\Users\A\Documents\xwechat_files\wxid_39s4apaf1sug21_ea81\msg\file\2026-08\传热系数csv\传热系数csv\叶栅-主流湍流度5%\压力面前缘与第二排孔区域视角\叶栅试验件0-压力面原始状态\air密度比1.0\吹风比1.5'
    in_file = os.path.join(folder, 'h.plt')
    out_file = os.path.join(folder, 'h_color.tiff')
    plt_to_tiff(in_file, out_file, cmap='jet')
