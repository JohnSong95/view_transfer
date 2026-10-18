# -*- coding: utf-8 -*-
"""模型训练数据采集器（主程序）。

任务背景
--------
训练一个面向"未知三维物体几何参数"的物理模型：
    输入：相机视角下拍摄的三维物体二维曲面图像（含透视畸变/自遮挡）
    输出：展开后的二维平面物理量图像
本模块批量生成「曲面几何 + 表面物理场 + 相机」随机样本，并把成对数据
（输入透视图、平面真值图、mask、几何/相机标签）按训练友好的目录结构保存。

与原工程的关系
--------------
- mesh.x 由 geometry.write_mesh_x 按 Pointwise 导出格式写出，read_mesh 可读回；
- 相机投影矩阵 P 与 get_p.py 手动标定的 P 完全同口径（3x4, y 像素向下）；
- 平面真值 .plt 由 tecplot.tecplot 写出；
- 每个样本可用 file_output（原物理展开流程）从透视图反算平面场，
  与真值对比 RMSE，确保数据物理自洽、可直接用于物理模型训练。

输出结构
--------
out_dir/
  dataset_config.json          采集配置与坐标约定
  manifest.jsonl               每个样本一行（路径 + 标签 + 验证指标）
  dataset_stats.json           全数据集数值统计（供归一化）
  sample_000000/
      input_view.npy           float32 (H,W) 透视图物理量，背景 0（模型输入）
      input_view.png           同上的彩色可视化
      view_mask.png            曲面可见区域 mask（255/0）
      depth.npy                float32 (H,W) 命中深度标签
      target_planar.npy        float32 (nx,ny) 平面真值物理量（模型目标）
      target_planar.png        真值彩色可视化
      target_planar.plt        Tecplot 格式真值（X,Y,H）
      mesh.x                   该样本的三维曲面网格
      meta.json                几何参数 / 相机内外参 / P / 物理场参数
"""
import os
import sys
import json
import time
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np
from PIL import Image

from geometry import random_surface, write_mesh_x
from camera import random_camera, project_points, estimate_visibility
from field_gen import random_field
from render import render_view
from read_mesh import read_mesh, _mesh_cache
from file_output import file_output
from read_file import read_file
from tecplot import tecplot


# ----------------------------------------------------------------------------
# 可视化（不依赖 matplotlib，等价 jet colormap；背景为纯黑）
# ----------------------------------------------------------------------------
def jet_rgb(x):
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _save_color_png(path, values, vmin, vmax, mask=None):
    norm = np.zeros_like(values, dtype=np.float64)
    if vmax - vmin > 1e-12:
        norm = (values - vmin) / (vmax - vmin)
    rgb = jet_rgb(norm)
    if mask is not None:
        rgb = rgb * (mask[..., None] > 0)
    Image.fromarray((rgb * 255).astype(np.uint8), 'RGB').save(path)


# ----------------------------------------------------------------------------
# 统计累加
# ----------------------------------------------------------------------------
class _Stats:
    def __init__(self):
        self.n_t = 0
        self.sum_t = 0.0
        self.sumsq_t = 0.0
        self.min_t = np.inf
        self.max_t = -np.inf
        self.n_i = 0
        self.sum_i = 0.0
        self.sumsq_i = 0.0

    def update(self, target, image, mask):
        self.n_t += target.size
        self.sum_t += float(target.sum())
        self.sumsq_t += float((target.astype(np.float64) ** 2).sum())
        self.min_t = min(self.min_t, float(target.min()))
        self.max_t = max(self.max_t, float(target.max()))
        vis = image[mask > 0].astype(np.float64)
        self.n_i += vis.size
        self.sum_i += float(vis.sum())
        self.sumsq_i += float((vis ** 2).sum())

    def as_dict(self):
        def _mom(n, s, sq):
            mean = s / max(n, 1)
            var = max(sq / max(n, 1) - mean ** 2, 0.0)
            return mean, float(np.sqrt(var))
        tm, ts = _mom(self.n_t, self.sum_t, self.sumsq_t)
        im, is_ = _mom(self.n_i, self.sum_i, self.sumsq_i)
        return {
            'target_min': float(self.min_t), 'target_max': float(self.max_t),
            'target_mean': tm, 'target_std': ts,
            'input_mean_visible': im, 'input_std_visible': is_,
        }


# ----------------------------------------------------------------------------
# 用原有物理展开流程反算，检验"透视图 + P + mesh -> 平面场"能否还真
# ----------------------------------------------------------------------------
def verify_sample(image, mask, P, field, sample_dir, mesh_file, margin=5):
    """复用 file_output + read_file + read_mesh 做一致性反算。"""
    verify_dir = os.path.join(sample_dir, '.verify_tmp')
    os.makedirs(verify_dir, exist_ok=True)
    try:
        file_output(image, P, verify_dir, mesh_file)
        rec = read_file(os.path.join(verify_dir, 'new_h'))   # (nx, ny)
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)

    nx, ny = field.shape
    delta_x, delta_y, mesh = read_mesh(mesh_file)
    pts = mesh[:, :, :, 0].reshape(3, -1).T
    x_px, y_px, zc = project_points(P, pts)
    x_px = x_px.reshape(nx, ny)
    y_px = y_px.reshape(nx, ny)
    zc = zc.reshape(nx, ny)
    H, W = image.shape

    ii, jj = np.mgrid[0:nx, 0:ny]
    interior = ((ii >= margin) & (ii < nx - margin)
                & (jj >= margin) & (jj < ny - margin))
    inside = ((x_px >= margin) & (x_px < W - margin)
              & (y_px >= margin) & (y_px < H - margin)
              & (zc > 0))
    xi = np.clip(np.rint(x_px).astype(int), 0, W - 1)
    yi = np.clip(np.rint(y_px).astype(int), 0, H - 1)
    # 用渲染 mask 排除投影到背景（自遮挡/视场外）的节点
    visible = mask[yi, xi] > 0
    sel = interior & inside & visible

    err = (rec - field.astype(np.float64))[sel]
    truth = field.astype(np.float64)[sel]
    rng = float(truth.max() - truth.min()) + 1e-12
    rmse = float(np.sqrt((err ** 2).mean()))
    metrics = {
        'verify_nodes': int(sel.sum()),
        'verify_rmse': rmse,
        'verify_mae': float(np.abs(err).mean()),
        'verify_rmse_norm': rmse / rng,
        'verify_max_abs': float(np.abs(err).max()),
    }
    rec_sel = rec[sel]
    if truth.std() > 1e-8 and rec_sel.std() > 1e-12:
        metrics['verify_corr'] = float(np.corrcoef(truth, rec_sel)[0, 1])
    return metrics


# ----------------------------------------------------------------------------
# 主采集流程
# ----------------------------------------------------------------------------
def collect(n_samples=100, out_dir='dataset', image_size=256,
            grid_nx=192, grid_ny=192, seed=0,
            min_visibility=0.35, max_view_tries=12,
            save_plt=True, supersample=1, n_start=5, n_iter=12,
            verify=True):
    """批量采集训练样本。

    Parameters
    ----------
    n_samples : int        样本数量
    out_dir : str          输出数据集目录
    image_size : int       透视图分辨率（方形 H=W）
    grid_nx, grid_ny : int 曲面 mesh / 平面真值的网格尺寸
    seed : int             随机种子（每个样本 seed=seed+i，可复现）
    min_visibility : float 曲面像素占画面最小比例，过低（掠射）视角重采
    save_plt : bool        是否额外输出 Tecplot 真值 .plt
    verify : bool          是否用原 file_output 流程反算并记录误差
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'manifest.jsonl')

    config = {
        'n_samples': n_samples,
        'image_size': int(image_size),
        'grid_nx': int(grid_nx),
        'grid_ny': int(grid_ny),
        'seed': int(seed),
        'supersample': int(supersample),
        'conventions': {
            'input': 'input_view.npy: float32 (H,W) 相机透视物理量, 背景=0',
            'mask': 'view_mask.png: 255=曲面可见像素',
            'target': 'target_planar.npy: float32 (nx,ny) 展开平面物理量真值',
            'depth': 'depth.npy: float32 (H,W) 光线交点深度, 背景=0',
            'P': '3x4 针孔投影矩阵, [x,y,1]~P[X,Y,Z,1], 像素 y 向下; '
                  '与 get_p.py 标定结果同口径',
            'mesh': 'mesh.x 为 Pointwise 结构化网格, read_mesh.py 可读',
            'planar_coords': 'target XX/YY 为弧长规则网格(mm), 见 meta.json',
            'values': '物理量为 float32 原值, 未做 8bit 量化; 归一化统计见 '
                      'dataset_stats.json',
        },
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(out_dir, 'dataset_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    stats = _Stats()
    t0 = time.time()
    n_written = 0
    si = 0
    with open(manifest_path, 'w', encoding='utf-8') as manifest:
        while n_written < n_samples:
            rng = np.random.default_rng(seed + si)
            si += 1

            surf = random_surface(rng, grid_nx, grid_ny)
            mesh = surf['mesh']
            field, field_info = random_field(rng, grid_nx, grid_ny)
            pts = mesh[:, :, :, 0].reshape(3, -1).T

            # 廉价预筛：只用投影面积估覆盖率（毫秒级），攒够候选再做一次真实渲染
            candidates = []
            cheap_tries = max_view_tries * 4
            for _try in range(cheap_tries):
                cam_try = random_camera(rng, pts, image_size, image_size)
                est = estimate_visibility(mesh, cam_try)
                if est >= min_visibility + 0.03:
                    candidates.append((est, cam_try))
                    if len(candidates) >= 12:
                        break
            if not candidates:
                # 该几何在随机视角下普遍太小（过于细长/强弯），换一组几何
                continue
            # 在候选中随机取前半（保留视角多样性，同时偏向覆盖率高的）
            candidates.sort(key=lambda c: c[0])
            pick = int(rng.integers(len(candidates) // 2, len(candidates)))
            est, cam = candidates[pick]
            image, mask, depth = render_view(
                mesh, field, cam, n_start=n_start, n_iter=n_iter,
                supersample=supersample)
            vis_ratio = float(mask.mean())
            if vis_ratio < min_visibility * 0.85 and len(candidates) > 1:
                # 估算与真实遮挡有偏差时，用覆盖率最高的候选补渲一次
                _, cam = candidates[-1]
                image, mask, depth = render_view(
                    mesh, field, cam, n_start=n_start, n_iter=n_iter,
                    supersample=supersample)
                vis_ratio = float(mask.mean())
            if vis_ratio < min_visibility * 0.7:
                continue

            sample_id = 'sample_%06d' % n_written
            sdir = os.path.join(out_dir, sample_id)
            os.makedirs(sdir, exist_ok=True)
            mesh_file = os.path.join(sdir, 'mesh.x')
            write_mesh_x(mesh_file, mesh)

            np.save(os.path.join(sdir, 'input_view.npy'), image)
            np.save(os.path.join(sdir, 'depth.npy'), depth)
            np.save(os.path.join(sdir, 'target_planar.npy'), field)
            Image.fromarray((mask * 255).astype(np.uint8), 'L').save(
                os.path.join(sdir, 'view_mask.png'))

            vmin = float(field.min())
            vmax = float(field.max())
            _save_color_png(os.path.join(sdir, 'input_view.png'),
                            image, vmin, vmax, mask=mask)
            _save_color_png(os.path.join(sdir, 'target_planar.png'),
                            field, vmin, vmax)

            if save_plt:
                tecplot(surf['XX'], surf['YY'], field.astype(np.float64),
                        os.path.join(sdir, 'target_planar.plt'))

            meta = {
                'sample_id': sample_id,
                'seed': int(seed + si - 1),
                'geometry': surf['params'],
                'camera': {
                    'P': cam['P'].tolist(),
                    'K': cam['K'].tolist(),
                    'R': cam['R'].tolist(),
                    't': cam['t'].tolist(),
                    'eye': cam['eye'].tolist(),
                    'fov_deg': cam['fov_deg'],
                    'azimuth_deg': cam['azimuth_deg'],
                    'elevation_deg': cam['elevation_deg'],
                    'roll_deg': cam['roll_deg'],
                    'distance': cam['distance'],
                },
                'field': field_info,
                'planar_grid': {
                    'XX00': float(surf['XX'][0, 0]),
                    'YY00': float(surf['YY'][0, 0]),
                    'dx': float(surf['XX'][0, 1] - surf['XX'][0, 0]),
                    'dy': float(surf['YY'][1, 0] - surf['YY'][0, 0]),
                },
                'visibility': vis_ratio,
                'estimated_visibility': float(est),
                'image_size': [int(image_size), int(image_size)],
                'value_min': vmin,
                'value_max': vmax,
            }

            if verify:
                try:
                    meta['verification'] = verify_sample(
                        image, mask, cam['P'], field, sdir, mesh_file)
                except Exception as e:
                    meta['verification'] = {'verify_error': repr(e)}
                # read_mesh 按路径缓存 mesh，批量生成时主动释放，避免内存增长
                _mesh_cache.clear()

            with open(os.path.join(sdir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            stats.update(field, image, mask)

            rec = {
                'sample_id': sample_id,
                'dir': sdir,
                'input_view': os.path.join(sdir, 'input_view.npy'),
                'input_png': os.path.join(sdir, 'input_view.png'),
                'mask': os.path.join(sdir, 'view_mask.png'),
                'depth': os.path.join(sdir, 'depth.npy'),
                'target': os.path.join(sdir, 'target_planar.npy'),
                'target_png': os.path.join(sdir, 'target_planar.png'),
                'mesh': mesh_file,
                'meta': os.path.join(sdir, 'meta.json'),
                'visibility': vis_ratio,
                'value_min': vmin,
                'value_max': vmax,
            }
            if verify and 'verification' in meta:
                rec['verification'] = meta['verification']
            manifest.write(json.dumps(rec, ensure_ascii=False) + '\n')
            manifest.flush()

            n_written += 1
            if n_written % 5 == 0 or n_written == n_samples:
                el = time.time() - t0
                vinfo = ''
                if verify and 'verification' in meta and \
                        'verify_rmse_norm' in meta.get('verification', {}):
                    vinfo = ', 反算归一化RMSE=%.4f' % \
                            meta['verification']['verify_rmse_norm']
                print('[%d/%d] %s  可见率=%.2f%s  (%.1fs/样本, 已用 %.0fs)'
                      % (n_written, n_samples, sample_id, vis_ratio, vinfo,
                         el / n_written, el))
                sys.stdout.flush()

    st = stats.as_dict()
    st['count'] = n_written
    with open(os.path.join(out_dir, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    print('完成：%d 个样本 -> %s（耗时 %.1fs）'
          % (n_written, out_dir, time.time() - t0))
    print('全数据集统计：target mean=%.4f std=%.4f range=[%.4f, %.4f]'
          % (st['target_mean'], st['target_std'], st['target_min'], st['target_max']))
