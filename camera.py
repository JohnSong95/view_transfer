# -*- coding: utf-8 -*-
"""随机相机（针孔模型）模块。

用随机外参(方位角/俯仰角/滚转角/距离)和内参(视场角/主点)合成相机，
输出与 get_p.py 中手动标定结果形式完全一致的 3x4 投影矩阵 P：

    [x_img, y_img, 1]^T ~ P @ [X, Y, Z, 1]^T
    像素坐标 x 向右、y 向下（与 get_p / file_output 的口径一致）

因此采集到的每一个样本，都可以直接用现有 file_output 流水线反算展开，
用于检验训练数据的物理一致性。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np


CAM_RANGES = {
    'fov_deg': (35.0, 60.0),       # 垂直视场角
    'elevation_deg': (20.0, 70.0),
    'azimuth_deg': (0.0, 360.0),
    'roll_deg': (-15.0, 15.0),
    'frame_margin': (1.10, 1.35),  # 物体在画面中的留白倍数
    'principal_jitter': 0.02,      # 主点相对画面中心的抖动(画面宽高比例)
}


def _look_at(eye, center, world_up, roll):
    """构造 world->camera 旋转，相机坐标轴 x=右, y=下, z=光轴(指向场景)。"""
    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    right0 = np.cross(fwd, world_up)
    right0 /= np.linalg.norm(right0)
    up0 = np.cross(right0, fwd)          # 画面"上"方向

    cr, sr = np.cos(roll), np.sin(roll)
    right = cr * right0 - sr * up0
    up = sr * right0 + cr * up0
    down = -up
    return np.stack([right, down, fwd], axis=0)


def random_camera(rng, points3d, width, height, ranges=None):
    """随机摆放一个能完整看到点云的针孔相机。

    Parameters
    ----------
    points3d : ndarray (..., 3) 物体点云（用于包围球与取景距离）
    width, height : int          相机图像分辨率（像素）
    """
    r = ranges or CAM_RANGES
    pts = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1)))

    fov = np.deg2rad(rng.uniform(*r['fov_deg']))
    elev = np.deg2rad(rng.uniform(*r['elevation_deg']))
    azim = np.deg2rad(rng.uniform(*r['azimuth_deg']))
    roll = np.deg2rad(rng.uniform(*r['roll_deg']))

    # 焦距(像素)：图像高度方向
    fy = (height / 2.0) / np.tan(fov / 2.0)
    fx = fy
    cx = width / 2.0 + rng.uniform(-1, 1) * r['principal_jitter'] * width
    cy = height / 2.0 + rng.uniform(-1, 1) * r['principal_jitter'] * height

    ce, se = np.cos(elev), np.sin(elev)
    direction = np.array([ce * np.cos(azim), ce * np.sin(azim), se])

    # 先用包围球保证完整入画，再按投影包围盒自适应推近，使物体尽量充满画面
    fit_dist = radius / np.tan(fov / 2.0)
    dist = fit_dist * rng.uniform(*r['frame_margin'])
    fill = rng.uniform(0.70, 0.92)
    for _ in range(2):
        eye = center + dist * direction
        R = _look_at(eye, center, np.array([0.0, 0.0, 1.0]), roll)
        right, down, fwd = R[0], R[1], R[2]
        rel = pts - eye
        s = rel @ fwd
        bxp = fx * ((rel @ right) / s)
        byp = fy * ((rel @ down) / s)
        bw = bxp.max() - bxp.min()
        bh = byp.max() - byp.min()
        k = max(bw / (width * fill), bh / (height * fill))
        # 只能在"完整入画距离"之外推近，避免节点跑出画面或到相机后方
        dist = float(min(max(dist * k, fit_dist * 1.005), dist))

    eye = center + dist * direction
    R = _look_at(eye, center, np.array([0.0, 0.0, 1.0]), roll)
    t = -R @ eye
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])
    P = K @ np.hstack([R, t[:, None]])

    return {
        'K': K, 'R': R, 't': t, 'P': P,
        'eye': eye, 'center': center,
        'fov_deg': float(np.rad2deg(fov)),
        'azimuth_deg': float(np.rad2deg(azim) % 360.0),
        'elevation_deg': float(np.rad2deg(elev)),
        'roll_deg': float(np.rad2deg(roll)),
        'distance': float(dist),
        'width': int(width), 'height': int(height),
    }


def project_points(P, pts):
    """将 3D 点投影为像素坐标 (x=列, y=行)，返回 xy 与深度 zc。

    与 file_output.check_output 中的投影公式一致。
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    L = np.hstack([pts, np.ones((n, 1))])
    Rm = (P @ L.T).T
    zc = Rm[:, 2]
    xy = Rm[:, :2] / zc[:, None]
    return xy[:, 0], xy[:, 1], zc


def estimate_visibility(mesh, cam):
    """毫秒级估计曲面在画面中的像素覆盖率（不做光线追踪）。

    把所有网格节点投影到像素平面，逐四边形用鞋带公式算投影像素面积，
    并用法向朝向相机做背面剔除；忽略微弱自遮挡（单块曲面误差很小）。
    用于在昂贵的真实渲染之前快速筛选相机视角。
    """
    H, W = cam['height'], cam['width']
    _, nx, ny, _ = mesh.shape
    X = mesh[0, :, :, 0]
    Y = mesh[1, :, :, 0]
    Z = mesh[2, :, :, 0]
    L = np.stack([X, Y, Z, np.ones_like(X)], axis=-1)     # (nx,ny,4)
    Rp = L @ cam['P'].T
    px = Rp[..., 0] / Rp[..., 2]
    py = Rp[..., 1] / Rp[..., 2]

    x00, y00 = px[:-1, :-1], py[:-1, :-1]
    x10, y10 = px[1:, :-1], py[1:, :-1]
    x11, y11 = px[1:, 1:], py[1:, 1:]
    x01, y01 = px[:-1, 1:], py[:-1, 1:]
    area2 = (x00 * y10 - y00 * x10 + x10 * y11 - y10 * x11
             + x11 * y01 - y11 * x01 + x01 * y00 - y01 * x00)
    area = np.abs(area2) * 0.5

    pts = np.stack([X, Y, Z], axis=-1)
    p00 = pts[:-1, :-1]
    nrm = np.cross(pts[1:, :-1] - p00, pts[:-1, 1:] - p00)
    center = (p00 + pts[1:, :-1] + pts[:-1, 1:] + pts[1:, 1:]) * 0.25
    eye = cam['eye']
    dotv = np.einsum('ijk,ijk->ij', eye - center, nrm)
    orient = np.sign(dotv.mean())
    front = dotv * orient > 0.0
    cov = float(area[front].sum() / (H * W))
    return min(cov, 1.0)


def pixel_rays(cam):
    """生成每个像素对应的世界坐标光线 (origins, dirs)，形状 (H, W, 3)。"""
    K, R = cam['K'], cam['R']
    H, W = cam['height'], cam['width']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dir_cam = np.stack([(xx - cx) / fx,
                        (yy - cy) / fy,
                        np.ones_like(xx)], axis=-1)
    # R 的三行是相机轴在世界系中的表达，故 d_w = d_c @ R
    dirs = dir_cam @ R
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = -R.T @ cam['t']
    origins = np.broadcast_to(origins, dirs.shape)
    return origins.copy(), dirs
