# -*- coding: utf-8 -*-
"""主入口。

默认：模型训练数据采集器
    生成「相机视角二维曲面图像 -> 展开二维平面图像」的成对训练数据，
    用于面向未知三维物体几何参数的物理模型训练。

    python main.py collect --samples 200 --out dataset
    python main.py collect --samples 20 --image-size 256 --grid 192 --no-plt

原有功能（标定照片 + mesh + Eta.plt 的透视展开）保留在 legacy 模式：
    python main.py legacy
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np


def collect_main(argv=None):
    from collect_data import collect
    parser = argparse.ArgumentParser(description='曲面展开物理模型训练数据采集器')
    parser.add_argument('--samples', type=int, default=100, help='样本数量')
    parser.add_argument('--out', type=str, default='dataset', help='输出目录')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--image-size', type=int, default=256,
                        help='相机透视图分辨率（像素，方形）')
    parser.add_argument('--grid', type=int, nargs=2, default=[192, 192],
                        metavar=('NX', 'NY'), help='曲面网格/平面真值尺寸')
    parser.add_argument('--min-visibility', type=float, default=0.35,
                        help='最小画面可见率，低于此值更换视角')
    parser.add_argument('--supersample', type=int, default=1, choices=[1, 2],
                        help='渲染超采样（1=关闭, 2=2x2）')
    parser.add_argument('--no-plt', action='store_true',
                        help='不输出 Tecplot .plt 真值文件')
    parser.add_argument('--no-verify', action='store_true',
                        help='跳过原物理展开流程的反算一致性检验')
    args = parser.parse_args(argv)

    collect(
        n_samples=args.samples,
        out_dir=args.out,
        image_size=args.image_size,
        grid_nx=args.grid[0],
        grid_ny=args.grid[1],
        seed=args.seed,
        min_visibility=args.min_visibility,
        supersample=args.supersample,
        save_plt=not args.no_plt,
        verify=not args.no_verify,
    )


def legacy_main():
    """原有标定展开流程：对标定照片、mesh 文件和 Eta.plt 数据做透视变换。"""
    from pathlib import Path
    from curve_flat import curve_flat

    # PS_PT：6 个标定点的 3D 坐标（X, Y, Z），需与在图像上点击的顺序一致
    PS_PT = np.array([
        [30, 27.75, 99.76],
        [30, -13.38, 75.06],
        [-36, 27.75, 99.76],
        [30, 90.54, 120.89],
        [-36, -13.38, 75.06],
        [-36, 90.54, 120.89]
    ])  # 传热吸力面3

    fig_dir = r'*.tiff'
    mesh_file = 'mesh.x'
    parent = Path(r'*')
    folder_name = [str(d) for d in parent.rglob('*')
                   if d.is_dir()
                   and '吸力面' in str(d)
                   and not any(child for child in d.iterdir() if child.is_dir())]

    print('Starting curve_flat...')
    sys.stdout.flush()
    curve_flat(fig_dir, PS_PT, folder_name, mesh_file)
    print('Done.')
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description='曲面图像展开：训练数据采集器(collect) / 原标定流程(legacy)',
        add_help=True)
    parser.add_argument('mode', nargs='?', default='collect',
                        choices=['collect', 'legacy'],
                        help="collect=训练数据采集(默认), legacy=原标定展开流程")
    args, rest = parser.parse_known_args()

    if args.mode == 'legacy':
        legacy_main()
    else:
        collect_main(rest if rest else None)


if __name__ == '__main__':
    main()
