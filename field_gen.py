# -*- coding: utf-8 -*-
"""平面物理标量场生成模块（训练目标 / ground truth）。

在曲面的 (u, v) 参数域（即"展开后的二维平面"）上生成光滑、正的物理标量场，
例如气膜冷却效率 / 传热系数这类叶片表面测量量：

- 低频高斯团：模拟前缘滞止、尾缘衰减等大尺度分布；
- 线性梯度：主流方向的系统变化；
- 气膜孔尾迹条带：随机孔排沿流向的射流贴壁条带（指数衰减+展向高斯+摆动），
  与数据目录中的"吸力面/压力面/吹风比"试验背景一致。

输出为 float32、无量纲、取值在 [0, 1]（可按真实物理量纲在训练时整体缩放，
缩放系数记录在 meta 中），不在此阶段做任何 8bit 量化，保证物理量可用于
物理模型的数值损失函数。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np


FIELD_RANGES = {
    'n_bump': (4, 9),
    'bump_sigma_u': (0.05, 0.25),
    'bump_sigma_v': (0.05, 0.25),
    'bump_amp': (0.08, 0.35),
    'grad_amp': 0.25,
    'n_streak_rows': (0, 3),          # 气膜孔排数
    'streak_holes_per_row': (3, 14),
    'streak_width': (0.008, 0.03),    # 展向初始半宽
    'streak_decay_u': (0.08, 0.4),    # 流向衰减长度
    'streak_spread': (1.5, 3.5),      # 沿程展向扩散
    'streak_amp': (0.25, 0.7),
    'out_low': (0.02, 0.20),
    'out_high': (0.70, 1.00),
}


def _add_streaks(rng, f, UU, VV, ranges):
    """在流场上叠加随机气膜孔排的射流尾迹条带。"""
    n_rows = rng.integers(*ranges['n_streak_rows'])
    rows_info = []
    for _ in range(n_rows):
        u0 = rng.uniform(0.08, 0.6)                 # 孔排弦向位置
        n_holes = rng.integers(*ranges['streak_holes_per_row'])
        dec = rng.uniform(*ranges['streak_decay_u'])
        w0 = rng.uniform(*ranges['streak_width'])
        spread = rng.uniform(*ranges['streak_spread'])
        amp = rng.uniform(*ranges['streak_amp'])
        meander_freq = rng.uniform(1.0, 4.0)
        meander_amp = rng.uniform(0.0, 0.015)

        margin = 1.0 / (n_holes + 1)
        vc = (np.arange(n_holes) + 1) * margin + rng.uniform(-0.3, 0.3, n_holes) * margin
        s = UU - u0                                    # 流向距离（孔后为正）
        # 周期性条带：取每个孔到最近孔心的展向距离
        dV = np.abs((VV[None, :, :] - vc[:, None, None]) + 0.5) % 1.0 - 0.5
        dV = np.min(np.abs(dV), axis=0)
        # 若孔数足够密，最近孔距离即近似条带展向位置
        center = meander_amp * np.sin(2 * np.pi * meander_freq * s)
        width = w0 * (1.0 + spread * s / max(dec, 1e-6))
        alive = s > 0
        streak = (amp * np.exp(-s / dec)
                  * np.exp(-((dV - center) / np.maximum(width, 1e-4)) ** 2)
                  * alive)
        f += streak
        rows_info.append({'u0': float(u0), 'n_holes': int(n_holes),
                          'decay_u': float(dec), 'width0': float(w0),
                          'amplitude': float(amp)})
    return f, rows_info


def random_field(rng, nx, ny, ranges=None):
    """生成 (nx, ny) 的平面物理标量场及生成参数记录。"""
    r = ranges or FIELD_RANGES
    u = np.linspace(0.0, 1.0, nx)
    v = np.linspace(0.0, 1.0, ny)
    VV, UU = np.meshgrid(v, u)            # 形状 (nx, ny)，与 mesh 索引一致

    f = np.zeros((nx, ny), dtype=np.float64)

    # 线性大尺度梯度
    gu = rng.uniform(-1, 1)
    gv = rng.uniform(-1, 1)
    f += r['grad_amp'] * 0.5 * (gu * (UU - 0.5) + gv * (VV - 0.5))

    # 高斯团
    n_bump = rng.integers(*r['n_bump'])
    bumps = []
    for _ in range(n_bump):
        cu = rng.uniform(0.0, 1.0)
        cv = rng.uniform(0.0, 1.0)
        su = rng.uniform(*r['bump_sigma_u'])
        sv = rng.uniform(*r['bump_sigma_v'])
        amp = rng.uniform(-1, 1) * rng.uniform(*r['bump_amp'])
        # 各向异性（含随机协方差旋转）
        theta = rng.uniform(0, np.pi)
        du, dv = UU - cu, VV - cv
        cr, sr = np.cos(theta), np.sin(theta)
        ru = cr * du + sr * dv
        rv = -sr * du + cr * dv
        f += amp * np.exp(-0.5 * ((ru / su) ** 2 + (rv / sv) ** 2))
        bumps.append({'center': [float(cu), float(cv)],
                      'sigma': [float(su), float(sv)],
                      'angle': float(theta), 'amp': float(amp)})

    # 气膜孔尾迹条带
    f, streak_rows = _add_streaks(rng, f, UU, VV, r)

    # 仿射映射到随机物理取值区间（保留真实相对大小），再裁剪到 [0, 1]
    lo = rng.uniform(*r['out_low'])
    hi = rng.uniform(*r['out_high'])
    fmin, fmax = float(f.min()), float(f.max())
    if fmax - fmin > 1e-12:
        f = (f - fmin) / (fmax - fmin)
    field = (lo + (hi - lo) * f).clip(0.0, 1.0).astype(np.float32)

    info = {
        'quantity': 'dimensionless_surface_scalar',
        'unit': '1',
        'n_bump': int(n_bump),
        'bumps': bumps,
        'grad': [float(gu), float(gv)],
        'streak_rows': streak_rows,
        'mapped_low': float(lo),
        'mapped_high': float(hi),
        'value_min': float(field.min()),
        'value_max': float(field.max()),
        'value_mean': float(field.mean()),
    }
    return field, info
