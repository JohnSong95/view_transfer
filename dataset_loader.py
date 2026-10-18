# -*- coding: utf-8 -*-
"""训练数据加载器。

- NumpySurfaceDataset：纯 numpy，按 manifest.jsonl 读取样本；
- TorchSurfaceDataset：若环境装有 torch，则提供 torch.utils.data.Dataset。

每个样本返回 dict：
    input  : (1, H, W) float32  相机透视物理量（normalize=True 时为标准化值，
                                背景始终为 0）
    mask   : (1, H, W) float32  曲面可见 mask
    target : (1, ny, nx) float32 平面展开物理量真值
    meta   : dict（含几何参数、相机 P 等标签，可用于物理约束损失/条件输入）

数据划分：按 manifest 顺序 8:1:1，也可自行传入 sample 列表。
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pylibs'))
import numpy as np
from PIL import Image


def _load_manifest(dataset_dir):
    path = os.path.join(dataset_dir, 'manifest.jsonl')
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_stats(dataset_dir):
    path = os.path.join(dataset_dir, 'dataset_stats.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


class NumpySurfaceDataset:
    """numpy 版数据集（HWC/CHW 均返回 CHW，便于直接转 torch tensor）。"""

    def __init__(self, dataset_dir, records=None, normalize=True,
                 load_depth=False, load_meta=True):
        self.dataset_dir = dataset_dir
        self.records = records if records is not None else _load_manifest(dataset_dir)
        stats = _load_stats(dataset_dir) if normalize else None
        self.normalize = normalize and stats is not None
        self.load_depth = load_depth
        self.load_meta = load_meta
        if self.normalize:
            self.in_mean = stats['input_mean_visible']
            self.in_std = max(stats['input_std_visible'], 1e-8)
            self.tg_mean = stats['target_mean']
            self.tg_std = max(stats['target_std'], 1e-8)
        else:
            self.in_mean = self.in_std = self.tg_mean = 0.0
            self.in_std = self.tg_std = 1.0

    def __len__(self):
        return len(self.records)

    def split(self, ratios=(0.8, 0.1, 0.1), seed=0):
        """按固定随机种子划分 train/val/test，返回三个新数据集。"""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.records))
        sizes = [int(round(r * len(self.records))) for r in ratios]
        sizes[-1] = len(self.records) - sum(sizes[:-1])
        cuts = np.cumsum(sizes[:-1])
        groups = np.split(idx, cuts)
        return [NumpySurfaceDataset(
            self.dataset_dir, records=[self.records[i] for i in g],
            normalize=self.normalize, load_depth=self.load_depth,
            load_meta=self.load_meta) for g in groups]

    def __getitem__(self, i):
        rec = self.records[i]
        image = np.load(rec['input_view']).astype(np.float32)      # (H,W)
        target = np.load(rec['target']).astype(np.float32)         # (ny,nx)
        mask = (np.asarray(Image.open(rec['mask'])) > 0).astype(np.float32)

        if self.normalize:
            image = np.where(mask > 0, (image - self.in_mean) / self.in_std, 0.0)
            target = (target - self.tg_mean) / self.tg_std

        item = {
            'input': image[None],
            'mask': mask[None],
            'target': target[None],
            'sample_id': rec['sample_id'],
        }
        if self.load_depth:
            item['depth'] = np.load(rec['depth']).astype(np.float32)[None]
        if self.load_meta:
            with open(rec['meta'], 'r', encoding='utf-8') as f:
                item['meta'] = json.load(f)
        return item


try:
    import torch
    from torch.utils.data import Dataset

    class TorchSurfaceDataset(Dataset):
        """torch 版：直接委托 numpy 数据集并转 tensor。"""

        def __init__(self, np_dataset):
            self.ds = np_dataset

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            item = self.ds[i]
            out = {
                'input': torch.from_numpy(item['input']),
                'mask': torch.from_numpy(item['mask']),
                'target': torch.from_numpy(item['target']),
            }
            if 'depth' in item:
                out['depth'] = torch.from_numpy(item['depth'])
            return out

except ImportError:
    TorchSurfaceDataset = None


if __name__ == '__main__':
    # 简单自检：python dataset_loader.py <dataset_dir>
    d = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    ds = NumpySurfaceDataset(d)
    train, val, test = ds.split()
    print('train/val/test = %d/%d/%d' % (len(train), len(val), len(test)))
    item = train[0]
    print('input ', item['input'].shape, item['input'].dtype,
          'range=[%.3f, %.3f]' % (item['input'].min(), item['input'].max()))
    print('mask  ', item['mask'].shape, 'coverage=%.3f' % item['mask'].mean())
    print('target', item['target'].shape,
          'range=[%.3f, %.3f]' % (item['target'].min(), item['target'].max()))
    print('sample:', item['sample_id'])
