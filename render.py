# -*- coding: utf-8 -*-
"""相机视角渲染器。

把定义在结构化曲面 mesh 上的物理标量场，按针孔相机渲染成二维透视图像，
即"相机视角下拍摄到的三维物体二维曲面图像"（模型输入）。

方法：对每个像素发射光线，与分片双线性参数曲面 S(a, b) 求交——
    F(a, b, t) = S(a, b) - (o + t*d) = 0
用多初值网格 + 阻尼 Newton 迭代（全部 numpy 向量化），同一像素取深度最小的
交点，天然处理曲面自遮挡；未命中像素为背景(0)并给出 mask。

曲面分片与 file_output / read_mesh 使用同一张 mesh、同一套 (a, b) 索引，
因此渲染图与平面真值场在物理量上逐点对应。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np

from camera import pixel_rays


def _bilinear_patch(pts_flat, a, b, nx, ny):
    """在浮点索引 (a, b) 处双线性插值曲面点及其对索引的偏导。

    pts_flat : (3, nx*ny)，C 展平（index = iu*ny + iv）
    返回 S, Sa, Sb，形状与 a/b 相同 + 末维 3。
    """
    ia = np.floor(a).astype(np.int64)
    ib = np.floor(b).astype(np.int64)
    ia_c = np.clip(ia, 0, nx - 2)
    ib_c = np.clip(ib, 0, ny - 2)
    wa = np.clip(a - ia, -1.0, 2.0)[..., None]
    wb = np.clip(b - ib, -1.0, 2.0)[..., None]

    i00 = ia_c * ny + ib_c
    i10 = (ia_c + 1) * ny + ib_c
    i01 = ia_c * ny + (ib_c + 1)
    i11 = (ia_c + 1) * ny + (ib_c + 1)

    def g(idx):
        # pts_flat[:, idx] : (3, ...) -> (..., 3)
        return np.moveaxis(pts_flat[:, idx], 0, -1)

    p00, p10, p01, p11 = g(i00), g(i10), g(i01), g(i11)
    S = ((1 - wa) * (1 - wb) * p00 + wa * (1 - wb) * p10
         + (1 - wa) * wb * p01 + wa * wb * p11)
    Sa = (1 - wb) * (p10 - p00) + wb * (p11 - p01)
    Sb = (1 - wa) * (p01 - p00) + wa * (p11 - p10)
    return S, Sa, Sb


def _sample_field(field, a, b, nx, ny):
    """物理场在 (a, b) 处双线性采样。"""
    ia = np.clip(np.floor(a).astype(np.int64), 0, nx - 2)
    ib = np.clip(np.floor(b).astype(np.int64), 0, ny - 2)
    wa = np.clip(a - ia, 0.0, 1.0)
    wb = np.clip(b - ib, 0.0, 1.0)
    f00 = field[ia, ib]
    f10 = field[ia + 1, ib]
    f01 = field[ia, ib + 1]
    f11 = field[ia + 1, ib + 1]
    return ((1 - wa) * (1 - wb) * f00 + wa * (1 - wb) * f10
            + (1 - wa) * wb * f01 + wa * wb * f11)


def _trace(origins, dirs, mesh, center, n_start=5, n_iter=14):
    """光线求交核心。

    返回 (hit, a, b, depth) ：
      hit   bool (N,)         是否命中
      a, b  float (N,)        命中处的曲面浮点索引
      depth float (N,)        光线上的深度
    """
    _, nx, ny, _ = mesh.shape
    pts_flat = mesh[:, :, :, 0].reshape(3, nx * ny)
    L_scale = float(np.linalg.norm(pts_flat[:, ny - 1] - pts_flat[:, 0]))
    L_scale = max(L_scale, 1e-9)

    N = origins.shape[0]
    sa = np.linspace(0.0, nx - 1.0, n_start)
    sb = np.linspace(0.0, ny - 1.0, n_start)
    SA, SB = np.meshgrid(sa, sb, indexing='ij')
    nS = n_start * n_start

    o = origins[:, None, :]          # (N,1,3)
    d = dirs[:, None, :]
    a = np.broadcast_to(SA.ravel(), (N, nS)).copy()
    b = np.broadcast_to(SB.ravel(), (N, nS)).copy()
    tc0 = np.einsum('ij,ij->i', center - origins, dirs)
    tc = np.broadcast_to(tc0[:, None], (N, nS)).copy()
    alive = np.ones((N, nS), dtype=bool)

    eps = 1e-12
    for _ in range(n_iter):
        S, Sa, Sb = _bilinear_patch(pts_flat, a, b, nx, ny)
        g = S - o - tc[..., None] * d
        Ccol = -d
        # 3x3 方程组 [Sa, Sb, -d] x = -g 的 Cramer 解
        bc = np.cross(Sb, Ccol)
        rc = np.cross(-g, Ccol)
        br = np.cross(Sb, -g)
        det = np.einsum('ijk,ijk->ij', Sa, bc)
        bad = np.abs(det) < eps * L_scale ** 3
        da = np.einsum('ijk,ijk->ij', -g, bc) / det
        db = np.einsum('ijk,ijk->ij', Sa, rc) / det
        dt = np.einsum('ijk,ijk->ij', Sa, br) / det
        # 阻尼：单步索引位移不超过 2 个网格，避免跨区跳跃
        da = np.clip(da, -2.0, 2.0)
        db = np.clip(db, -2.0, 2.0)
        dt = np.clip(dt, -0.5 * L_scale, 0.5 * L_scale)
        move = alive & ~bad & np.isfinite(da) & np.isfinite(db) & np.isfinite(dt)
        a = np.where(move, a + da, a)
        b = np.where(move, b + db, b)
        tc = np.where(move, tc + dt, tc)
        alive &= move
        # 跑出参数域过远或跑到相机后方的初值判死
        alive &= (a > -0.25) & (a < nx - 1 + 0.25)
        alive &= (b > -0.25) & (b < ny - 1 + 0.25)
        alive &= tc > 0

    # 终值残差校验
    S, _, _ = _bilinear_patch(pts_flat, a, b, nx, ny)
    g = S - o - tc[..., None] * d
    res = np.linalg.norm(g, axis=-1)
    valid = (alive
             & (res < 1e-4 * L_scale)
             & (a >= -1e-3) & (a <= nx - 1 + 1e-3)
             & (b >= -1e-3) & (b <= ny - 1 + 1e-3)
             & (tc > 0))

    big = np.inf
    tc_sel = np.where(valid, tc, big)
    best = np.argmin(tc_sel, axis=1)              # 最近交点 -> 处理自遮挡
    row = np.arange(N)
    hit = valid[row, best]
    a_hit = a[row, best]
    b_hit = b[row, best]
    t_hit = tc[row, best]
    return hit, a_hit, b_hit, t_hit


def _pixel_rays(cam, jitter=None):
    """生成像素光线；jitter=(dy,dx)∈[0,1)^2 时用于超采样。"""
    K, R = cam['K'], cam['R']
    H, W = cam['height'], cam['width']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if jitter is not None:
        xx = xx + jitter[1]
        yy = yy + jitter[0]
    dir_cam = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], axis=-1)
    dirs = dir_cam @ R
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(-R.T @ cam['t'], dirs.shape)
    return origins.copy().reshape(-1, 3), dirs.reshape(-1, 3)


def render_view(mesh, field, cam, n_start=5, n_iter=14, supersample=1):
    """渲染相机视角图。

    Parameters
    ----------
    mesh  : (3, nx, ny, 1) 结构化曲面
    field : (nx, ny) float 曲面上的物理标量场（平面真值）
    cam   : camera.random_camera 返回的相机字典
    n_start, n_iter : Newton 初值网格边长 / 迭代次数
    supersample : 每像素超采样数（1 或 2）

    Returns
    -------
    image : float32 (H, W)，物理标量取值，背景为 0
    mask  : uint8   (H, W)，1=曲面可见
    depth : float32 (H, W)，命中处深度，背景 0
    """
    _, nx, ny, _ = mesh.shape
    H, W = cam['height'], cam['width']
    center = cam['center']

    image = np.zeros((H, W), dtype=np.float64)
    cover = np.zeros((H, W), dtype=np.float64)
    depth_acc = np.zeros((H, W), dtype=np.float64)

    offsets = [(0.5, 0.5)]
    if supersample > 1:
        s = supersample
        offsets = [(0.5 + (j + 0.5) / s - 0.5, 0.5 + (i + 0.5) / s - 0.5)
                   for j in range(s) for i in range(s)]

    for jy, jx in offsets:
        origins, dirs = _pixel_rays(cam, jitter=(jy - 0.5, jx - 0.5))
        hit, a, b, t = _trace(origins, dirs, mesh, center, n_start, n_iter)
        vals = np.zeros(H * W, dtype=np.float64)
        vals[hit] = _sample_field(field, a[hit], b[hit], nx, ny)
        image += vals.reshape(H, W)
        cover += hit.reshape(H, W).astype(np.float64)
        depth_acc += (t * hit).reshape(H, W)

    # mask 必须在安全除法之前确定，否则背景会被 1e-12 兜底值误判为命中
    mask = (cover >= 0.5).astype(np.uint8)
    safe_cover = np.maximum(cover, 1.0)
    image = (image / safe_cover).astype(np.float32) * mask
    depth = (depth_acc / safe_cover).astype(np.float32)
    return image, mask, depth
