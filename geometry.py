# -*- coding: utf-8 -*-
"""随机三维曲面几何生成模块。

面向"未知三维物体几何参数"的训练任务：每个样本随机生成一块带弯度、
扭转、展向弯曲和微小波纹的结构化曲面网格（模拟叶片吸力面/压力面一类
的曲面试验件），输出与 read_mesh.py 完全兼容的 mesh 数组：

    mesh.shape = (3, nx, ny, 1)
    mesh[coord, iu, iv, 0]，coord = 0/1/2 -> X/Y/Z
    u = iu/(nx-1) 弦向(流向)，v = iv/(ny-1) 展向

几何参数(毫米/弧度)全部记录在返回的 params 中，作为训练标签。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np


# ----------------------------------------------------------------------------
# 随机参数范围（可按实际试验件尺寸修改）
# ----------------------------------------------------------------------------
GEOM_RANGES = {
    'L': (80.0, 160.0),          # 弦向长度 mm
    'W': (40.0, 100.0),          # 展向宽度 mm
    'camber_ratio': (-0.18, 0.18),   # 最大弯度 / L
    'span_bend_ratio': (0.0, 0.08),  # 展向弯曲 / L（边缘相对中部下垂）
    'twist0_deg': (-10.0, 10.0),     # 根部扭转角
    'twist1_deg': (-35.0, 35.0),     # 沿弦向扭转变化量
    'waviness_ratio': 0.006,         # 随机波纹幅值 / L
}


def random_geometry_params(rng):
    """从参数分布中抽样一组几何参数。"""
    r = GEOM_RANGES
    p = {
        'name': 'curved_patch',
        'L': float(rng.uniform(*r['L'])),
        'W': float(rng.uniform(*r['W'])),
        'camber': float(rng.uniform(*r['camber_ratio'])
                        * rng.uniform(*r['L'])),
        'span_bend': float(abs(rng.uniform(*r['span_bend_ratio']))
                           * rng.uniform(*r['L'])),
        'twist0': float(np.deg2rad(rng.uniform(*r['twist0_deg']))),
        'twist1': float(np.deg2rad(rng.uniform(*r['twist1_deg']))),
    }
    # 低阶随机波纹（几何不确定性），2 个弦向模态 + 2 个展向模态
    p['waviness'] = [
        (float(rng.uniform(1, 3)),
         float(rng.choice([0, 1, 2])),
         float(rng.uniform(0, 2 * np.pi)),
         float(rng.uniform(-1, 1))),
        (float(rng.uniform(1, 3)),
         float(rng.choice([0, 1, 2])),
         float(rng.uniform(0, 2 * np.pi)),
         float(rng.uniform(-1, 1))),
    ]
    p['waviness_amp'] = r['waviness_ratio']
    return p


def build_surface(params, nx, ny):
    """按参数构造结构化曲面。

    返回
    ----
    mesh : ndarray (3, nx, ny, 1)  与 read_mesh 输出一致
    XX, YY : ndarray (nx, ny)     平面展开后的规则坐标(mm)，
                                  步长按弧长计算（与 read_mesh 同口径）
    """
    L, W = params['L'], params['W']
    iu = np.arange(nx)
    iv = np.arange(ny)
    u = iu / (nx - 1)
    v = iv / (ny - 1)
    UU = np.tile(u[:, None], (1, ny))
    VV = np.tile(v[None, :], (nx, 1))

    # 中弧线（抛物线弯度），u=0.5 处最大
    zc = 4.0 * params['camber'] * UU * (1.0 - UU)

    # 展向弯曲：边缘相对中部下垂
    w_bend = params['span_bend'] * ((2.0 * VV - 1.0) ** 2 - 1.0)

    # 随机波纹
    wav = np.zeros_like(UU)
    amp = params['waviness_amp'] * L
    for fu, fv, phase, sign in params['waviness']:
        if fv == 0:
            wav += sign * amp * np.sin(2 * np.pi * fu * UU + phase)
        else:
            wav += sign * amp * np.sin(2 * np.pi * fu * UU + phase) \
                   * np.cos(2 * np.pi * fv * VV)

    y0 = W * (VV - 0.5)
    w = w_bend + wav

    # 沿弦向线性变化的扭转角（绕中弧线轴旋转截面）
    theta = params['twist0'] + params['twist1'] * (UU - 0.5)
    ct, st = np.cos(theta), np.sin(theta)

    X = L * UU
    Y = y0 * ct - w * st
    Z = zc + y0 * st + w * ct

    mesh = np.stack([X, Y, Z], axis=0)[:, :, :, None]

    # 平面展开坐标：与 read_mesh._step_len 完全相同的弧长口径
    dx = float(np.linalg.norm(mesh[:, 0, 1, 0] - mesh[:, 0, 0, 0]))
    dy = float(np.linalg.norm(mesh[:, 1, 0, 0] - mesh[:, 0, 0, 0]))
    XX = np.broadcast_to((iv[None, :] + 1) * dx, (nx, ny)).copy()
    YY = np.broadcast_to((iu[:, None] + 1) * dy, (nx, ny)).copy()
    return mesh, XX, YY


def random_surface(rng, nx, ny, params=None):
    """随机参数 + 构造曲面，返回 dict(mesh, XX, YY, params)。"""
    if params is None:
        params = random_geometry_params(rng)
    mesh, XX, YY = build_surface(params, nx, ny)
    params_out = dict(params)
    params_out['nx'] = int(nx)
    params_out['ny'] = int(ny)
    params_out['dx'] = float(XX[0, 1] - XX[0, 0])
    params_out['dy'] = float(YY[1, 0] - YY[0, 0])
    return {'mesh': mesh, 'XX': XX, 'YY': YY, 'params': params_out}


def write_mesh_x(path, mesh):
    """按 mesh.x 的 Pointwise 导出格式写盘（read_mesh.read_mesh 可直接读回）。

    文件布局：
        第 1 行: 1
        第 2 行: nx ny 1
        其后: 先 X 分量(每行 4 个数, ny 行 x 每行 nx 个)，再 Y，再 Z
    """
    nx = mesh.shape[1]
    ny = mesh.shape[2]
    with open(path, 'w') as f:
        f.write('   1\n')
        f.write('%5d %5d %5d\n' % (nx, ny, 1))

        def _write_block(values_2d):
            flat = values_2d.reshape(-1)
            for k in range(0, flat.size, 4):
                chunk = flat[k:k + 4]
                f.write('    ' + '    '.join('%.15f' % v for v in chunk) + '\n')

        for c in range(3):
            # 与 read_mesh 一致：iv 行优先，每行 nx 个 iu
            for iv in range(ny):
                line_vals = mesh[c, :, iv, 0]
                # 保持与源文件同样的 4 列换行风格
                pos = 0
                while pos < nx:
                    chunk = line_vals[pos:pos + 4]
                    f.write('    ' + '    '.join('%.15f' % v for v in chunk) + '\n')
                    pos += 4
