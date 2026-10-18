import re
import numpy as np


def read_file(A):
    """读取 .plt 文件。A 为不带 .plt 后缀的路径，返回 eta 二维数组。

    对应 MATLAB: read_file.m
    数据按 X 变化最快、Y 其次排列。返回的 eta 形状为 (y_count, x_count)，
    即 eta[y, x] 对应 (X=x, Y=y) 处的值。值大于 1 或小于 0 的格子置 0。
    """
    filename = A + '.plt'
    # 优先从 ZONE 行解析 I/J 维度（更可靠，避免 max+1 与实际点数不匹配）
    x_count = y_count = None
    with open(filename, 'r') as f:
        f.readline()  # TITLE
        f.readline()  # VARIABLES
        zone_line = f.readline()
    m_i = re.search(r'I\s*=\s*(\d+)', zone_line)
    m_j = re.search(r'J\s*=\s*(\d+)', zone_line)
    if m_i and m_j:
        x_count = int(m_i.group(1))
        y_count = int(m_j.group(1))

    # dlmread(filename, ' ', 3, 0) -> 跳过前 3 行
    M = np.loadtxt(filename, skiprows=3)

    if x_count is None:
        # 回退到 max+1
        x_count = int(M[:, 0].max()) + 1
        y_count = int(M[:, 1].max()) + 1

    n_expected = x_count * y_count
    n_actual = M.shape[0]
    if n_actual != n_expected:
        print(f'  Warning: data points {n_actual} != I*J {n_expected}, '
              f'padding/truncating to fit.')

    h = M[:, 2]
    if n_actual < n_expected:
        h_full = np.zeros(n_expected)
        h_full[:n_actual] = h
        h = h_full
    else:
        h = h[:n_expected]

    # 数据按 (X 增长最快, Y 其次) 排列，reshape 成 (y_count, x_count)
    h_2d = h.reshape(y_count, x_count)
    # 数值越界则置 0
    eta = h_2d
    #eta = np.where((h_2d > 1) | (h_2d < 0), 0.0, h_2d)
    print('Read file ' + filename)
    return eta

