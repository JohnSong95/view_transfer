# view_transfer —— 曲面透视图展开模型的训练数据采集器

[English](#english) | [中文](#中文)

当前版本：[v1.0.1](https://github.com/JohnSong95/view_transfer/releases/tag/v1.0.1)

<a name="中文"></a>
## 中文

把相机视角下拍摄到的**三维曲面二维图像**直接展开为**二维平面物理量场**，是一个面向
未知几何参数物体的学习问题（输入：透视图；输出：平面展开图）。本项目用纯物理前向
建模的方式批量生成严格配对的训练数据，并对每一条样本用独立的逆管线做物理一致性验证：

```
随机几何参数 ──> 参数化 3D 曲面 ──> 随机针孔相机投影 + 光线求交渲染 ──> 透视图 (模型输入 X)
                                        │
平面物理标量场真值 (低频团 + 梯度 + 气膜孔射流尾迹) ─ 曲面参数化映射 ──> 平面图 (训练目标 Y)
                                        │
                  独立逆管线（P 矩阵投影 + 插值，复刻实测处理流程）──> RMSE / 相关系数验证
```

在 1000 个随机样本上的实测指标：逆管线反算归一化 RMSE 均值约 **1.1%**，真值/反算
相关系数均值约 **0.997**（逐样本指标写在 `manifest.jsonl` 中）。

### 目录结构

```
view_transfer/
├── main.py              # 入口：collect（数据采集，默认）/ legacy（手工标定展开）
├── geometry.py          # 参数化随机 3D 曲面生成（弯度/扭转/波纹）
├── camera.py            # 随机针孔相机 + 3×4 DLT 投影矩阵
├── field_gen.py         # 平面物理标量场真值（含气膜孔射流尾迹）
├── render.py            # 逐像素光线-曲面求交渲染（含自遮挡）
├── collect_data.py      # 批量采集编排 + 逆管线验证 + 统计清单
├── dataset_loader.py    # NumPy / PyTorch 数据集加载器
├── examples/            # 单个已验证样本（可直接查看数据格式）
├── mesh.x / calibration_image.png   # legacy 流程示例：Pointwise 网格 + 标定照片
├── read_mesh.py / get_p.py / file_output.py / ...   # 旧实测处理管线
└── requirements.txt
```

### 安装

需要 **Python 3.10+**，仅依赖 numpy 与 pillow；PyTorch 为可选依赖（仅在使用
`TorchSurfaceDataset` 封装时需要）：

```bash
git clone https://github.com/JohnSong95/view_transfer.git
cd view_transfer
pip install -r requirements.txt        # numpy + pillow

# 可选：需要接入 torch.utils.data.DataLoader 时
pip install torch
```

不需要编译，克隆后即可运行。注意：数据采集（`collect`）只需 numpy 与 pillow；
**旧流程 `legacy` 与格式转换小工具另需 matplotlib**（legacy 交互取点还需要桌面
环境），见下文对应章节。

### 快速开始：三步上手

**第 1 步：试跑 10 个样本，确认环境正常**

```bash
python main.py collect --samples 10 --out dataset_demo --seed 0
```

运行中每 5 个样本打印一次进度（可见率、反算 RMSE），结束时打印全数据集统计。

**第 2 步：检查生成结果**

```bash
# 用加载器自检（打印 train/val/test 划分与张量形状）
python dataset_loader.py dataset_demo
```

也可以直接打开 `dataset_demo/sample_000000/input_view.png`（模型输入）与
`target_planar.png`（训练目标）肉眼比对；`examples/` 目录下有一份已验证的
单样本示例（为控制仓库体积，省略了 `mesh.x` 与 `target_planar.plt`，其余
文件齐全），可先浏览它了解数据长什么样。

**第 3 步：正式采集**

```bash
# 采集 1000 对样本（256×256 透视图，192×192 平面真值）
python main.py collect --samples 1000 --out dataset_run --image-size 256 --grid 192 192 --seed 0
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--samples N` | 100 | 样本数量 |
| `--out DIR` | dataset | 输出目录（建议每次运行用独立目录） |
| `--seed` | 0 | 随机种子；同参数重跑可复现（每样本实际种子记录在 `meta.json` 的 `seed` 字段） |
| `--image-size` | 256 | 透视图分辨率（像素，正方形） |
| `--grid NX NY` | 192 192 | 曲面网格/平面真值分辨率（NX=弦向, NY=展向） |
| `--min-visibility` | 0.35 | 曲面像素占画面最小比例，过低（掠射视角）自动重采机位 |
| `--supersample` | 1 | 抗锯齿超采样倍率（可选 2，渲染约慢 4 倍） |
| `--no-plt` | 关 | 不写 Tecplot `.plt` 真值文件（每样本省约 1 MB） |
| `--no-verify` | 关 | 跳过逆管线物理一致性验证（不建议） |

性能参考（256/192 分辨率，纯 CPU）：约 **8.5 秒/样本、4 MB/样本**。样本逐个落盘、
清单逐行 flush，**中途中断不损失已生成样本**；用相同 `--seed` 与 `--out` 重跑可
逐样本覆盖补齐。

#### 在 Python 中调用采集

不经过命令行，直接在脚本/Notebook 里调 `collect()`：

```python
from collect_data import collect

collect(
    n_samples=200,          # 样本数
    out_dir='dataset_200',  # 输出目录
    image_size=256,         # 透视图分辨率
    grid_nx=192, grid_ny=192,   # 平面真值网格（NX=弦向, NY=展向）
    seed=0,                 # 随机种子
    save_plt=False,         # False = 不写 Tecplot 文件，节省磁盘
    verify=True,            # 物理一致性反算验证
)
```

### 数据格式

每个样本一个目录 `sample_XXXXXX/`：

| 文件 | 形状/类型 | 含义 |
|---|---|---|
| `input_view.npy` | float32 (H,W) | **模型输入**：透视渲染的曲面物理量，背景为 0 |
| `view_mask.png` | uint8 | 可见曲面掩码（255=曲面） |
| `depth.npy` | float32 (H,W) | 深度标签：光线交点深度，背景=0（辅助任务/几何监督用） |
| `target_planar.npy` | float32 (NX,NY) | **训练目标**：展开后的平面物理量真值（float32 原值，未做 8-bit 量化） |
| `target_planar.plt` | Tecplot 文本格式 | 平面真值，可用 Tecplot 直接打开 |
| `input_view.png` / `target_planar.png` | 图片 | 可视化预览（共用同一色标范围） |
| `mesh.x` | 文本网格 | 该样本的 3D 曲面网格（与 `read_mesh.py` 互通） |
| `meta.json` | JSON | 几何参数 + 相机内外参 + 物理场参数 + 验证指标 |

数据集根目录另有三个文件：

- `manifest.jsonl`：每个样本一行，含所有文件路径与关键指标，例如：

```json
{"sample_id": "sample_000000", "dir": "dataset_run/sample_000000",
 "input_view": "dataset_run/sample_000000/input_view.npy",
 "input_png": ".../input_view.png", "mask": ".../view_mask.png",
 "depth": ".../depth.npy", "target": ".../target_planar.npy",
 "target_png": ".../target_planar.png", "mesh": ".../mesh.x",
 "meta": ".../meta.json", "visibility": 0.409, "value_min": 0.135, "value_max": 0.942,
 "verification": {"verify_nodes": 33095, "verify_rmse": 0.00806,
                  "verify_rmse_norm": 0.00999, "verify_corr": 0.99790}}
```

- `dataset_stats.json`：全数据集统计（`input_mean_visible` / `input_std_visible` /
  `target_mean` / `target_std` / `target_min` / `target_max`），供归一化使用；
- `dataset_config.json`：本次采集的配置与坐标约定。

`meta.json` 的主要字段：

| 字段 | 内容 |
|---|---|
| `geometry` | 几何标签：`L` 弦长、`W` 展宽、`camber` 弯度、`span_bend` 展向弯曲、`twist0/twist1` 扭转（弧度）、`waviness` 波纹模态（可作为条件输入/物理约束标签） |
| `camera` | `P`（3×4 DLT 投影矩阵，像素坐标 x 向右、y 向下）、`K` 内参、`R`/`t` 外参、`eye` 相机位置、`fov_deg`/`azimuth_deg`/`elevation_deg`/`roll_deg`/`distance` |
| `field` | 物理场参数：高斯团 `bumps`、线性梯度 `grad`、气膜孔排 `streak_rows`（孔排位置 `u0`、孔数 `n_holes`、衰减长度、条带宽度、幅值） |
| `planar_grid` | 平面真值网格的起点坐标与步长（`XX00`/`YY00`/`dx`/`dy`） |
| `visibility` | 该视角下曲面像素占画面比例 |
| `verification` | 逆管线验证指标（见下文） |

`camera.P` 与常规针孔标定（及本项目旧的手工标定流程 `get_p.py`）完全同口径：
`[x_img, y_img, 1]^T ~ P @ [X, Y, Z, 1]^T`。

### 加载数据

#### NumPy 数据集（推荐入口）

`NumpySurfaceDataset` 按 `manifest.jsonl` 读取，自动用 `dataset_stats.json` 做标准化。
`__getitem__` 返回 **dict**（不是元组）：

```python
from dataset_loader import NumpySurfaceDataset

ds = NumpySurfaceDataset('dataset_run')            # normalize=True（默认）
train, val, test = ds.split((0.8, 0.1, 0.1))       # 固定 seed=0 的随机划分

item = train[0]
item['input']    # (1, 256, 256) float32，仅曲面区域标准化、背景恒为 0
item['mask']     # (1, 256, 256) float32，曲面可见掩码
item['target']   # (1, 192, 192) float32，平面真值（已标准化）
item['sample_id']# 'sample_000042'
item['meta']     # 完整 meta.json（load_meta=True 默认开启）
```

需要还原物理量纲时：

```python
x_phys = item['input'] * ds.in_std + ds.in_mean    # 曲面区域
y_phys = item['target'] * ds.tg_std + ds.tg_mean
```

常用选项：

```python
# 不做标准化，直接拿 [0,1] 原值
ds_raw = NumpySurfaceDataset('dataset_run', normalize=False)

# 同时加载深度标签
ds_d = NumpySurfaceDataset('dataset_run', load_depth=True)
ds_d[0]['depth']    # (1, 256, 256) float32

# 不加载 meta.json（样本多时可加快 IO）
ds_fast = NumpySurfaceDataset('dataset_run', load_meta=False)
```

#### 接入 PyTorch 训练

装有 torch 时，`TorchSurfaceDataset` 直接提供 `torch.utils.data.Dataset`：

```python
from torch.utils.data import DataLoader
from dataset_loader import NumpySurfaceDataset, TorchSurfaceDataset

train_np = NumpySurfaceDataset('dataset_run', load_depth=True).split()[0]
train_ds = TorchSurfaceDataset(train_np)
loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

for batch in loader:
    x, y = batch['input'], batch['target']   # (B,1,256,256) / (B,1,192,192)
    m, d = batch['mask'], batch['depth']     # (B,1,256,256) / (B,1,256,256)
    ...
```

#### 典型训练循环（示意）

仓库本身不包含网络定义，下面是一个最小的回归训练骨架，展示输入/目标如何
接入任意 image-to-image 网络（如 U-Net）：

```python
import torch
from torch.utils.data import DataLoader
from dataset_loader import NumpySurfaceDataset, TorchSurfaceDataset

train_np = NumpySurfaceDataset('dataset_run', load_meta=False).split()[0]
loader = DataLoader(TorchSurfaceDataset(train_np), batch_size=16, shuffle=True)

model = MyUnwrapNet().cuda()         # 你的网络：输入 (B,1,256,256) -> 输出 (B,1,192,192)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in loader:
        x, y = batch['input'].cuda(), batch['target'].cuda()   # 均已标准化
        loss = torch.nn.functional.mse_loss(model(x), y)       # 背景已置 0，无需额外屏蔽
        opt.zero_grad(); loss.backward(); opt.step()
```

提示：`input` 的背景恒为 0（标准化仅作用于曲面区域），如需显式屏蔽背景损失，
可用 `batch['mask']`；如需物理量纲的损失（如相对误差），先把预测与目标乘回
`ds.tg_std`、加回 `ds.tg_mean`。`meta.json` 中的几何/相机参数也可取出来做
条件输入或物理约束项。

#### 按质量筛选样本

`manifest.jsonl` 逐样本记录了验证指标，可在加载前过滤掉个别极端样本：

```python
import json
from dataset_loader import NumpySurfaceDataset

with open('dataset_run/manifest.jsonl', encoding='utf-8') as f:
    recs = [json.loads(l) for l in f if l.strip()]

good = [r for r in recs
        if r.get('verification', {}).get('verify_rmse_norm', 0) < 0.03]
ds_good = NumpySurfaceDataset('dataset_run', records=good)
print('保留 %d / %d 个样本' % (len(good), len(recs)))
```

#### 不用加载器，直接读文件

`examples/` 下的单样本演示了最原始的读取方式：

```python
import json
import numpy as np

root = 'examples/sample_000000'
x = np.load(f'{root}/input_view.npy')       # (256, 256) float32，背景=0
y = np.load(f'{root}/target_planar.npy')    # (192, 192) float32 平面真值
d = np.load(f'{root}/depth.npy')            # (256, 256) float32 深度

with open(f'{root}/meta.json', encoding='utf-8') as f:
    meta = json.load(f)
P = np.asarray(meta['camera']['P'])         # 3×4 DLT 投影矩阵
```

### 物理一致性验证

采集器对每个样本复用**实测数据处理管线**（`file_output.py`：mesh 点经 `P` 投影到
照片 + 双线性插值 + Tecplot 读写），由透视图和 `P` 反推平面场，与真值比较，
指标写入 `meta.json` 与 `manifest.jsonl`：

| 指标 | 含义 | 参考基线 |
|---|---|---|
| `verify_rmse_norm` | 反算场与真值的 RMSE（按该样本值域归一） | 均值 ~0.011，个别大弯扭样本可达 ~0.06 |
| `verify_corr` | 反算场与真值的相关系数 | 均值 >0.99 |
| `verify_nodes` | 参与验证的网格节点数 | — |
| `verify_rmse` / `verify_mae` / `verify_max_abs` | 物理量纲下的误差 | — |

也可单独对已有数据集自检：

```bash
python dataset_loader.py dataset_run
```

期望输出 train/val/test 划分（默认 8:1:1）、`input (1,256,256)`、
`target (1,192,192)`，无异常。

### 适配你自己的几何与物理量

三个随机采样范围都定义在对应模块顶部，可直接改源码，也可在运行时覆盖：

```python
import geometry, camera, field_gen

# geometry.py 顶部 GEOM_RANGES：弦长/展宽/弯度/展向弯曲/扭转/波纹
geometry.GEOM_RANGES['L'] = (100.0, 140.0)        # 弦长限制在 100~140 mm
geometry.GEOM_RANGES['twist1_deg'] = (-15.0, 15.0)

# camera.py 顶部 CAM_RANGES：视场角/俯仰/方位/滚转/取景留白/主点抖动
camera.CAM_RANGES['fov_deg'] = (45.0, 55.0)

# field_gen.py 顶部 FIELD_RANGES：高斯团/梯度/气膜孔排/输出值域
field_gen.FIELD_RANGES['n_streak_rows'] = (1, 3)  # 每个样本至少 1 排气膜孔
field_gen.FIELD_RANGES['out_high'] = (0.8, 1.0)

# 之后再调用 collect / main.py collect，即按新范围采样
```

### 实用建议与常见问题

- **磁盘占用**：默认每样本约 4 MB（含 Tecplot 与 mesh）。只训练网络时加
  `--no-plt` 可降到约 3 MB。
- **掠射视角样本多**：调高 `--min-visibility`（如 0.40）重采。
- **输入图有锯齿**：加 `--supersample 2`（渲染约慢 4 倍）。
- **中断恢复**：直接用相同命令重跑即可，已生成样本会被同 seed 覆盖，不会错位。
- **训练输入要不要带 mask**：`input` 背景恒为 0，可只用 `input`；需要显式屏蔽
  背景损失时用 `mask`。
- **量化说明**：所有 `.npy` 均为 float32 物理原值，`.png` 仅供人眼预览，不用于训练。

### 旧流程：实测照片的手工标定展开

除数据采集外，本仓库保留了原有的实测处理流程（交互式选标定标记点）。
**依赖 matplotlib（交互取点窗口，需桌面环境）**：

```bash
python main.py legacy
```

`main.py` 的 `legacy_main()` 中有 4 个变量需要按你的试验配置：

```python
PS_PT = np.array([...])   # 6 个标定标记点的 3D 坐标 (X, Y, Z)，顺序须与图像上点击顺序一致
fig_dir = r'*.tiff'       # 标定照片路径
mesh_file = 'mesh.x'      # Pointwise 导出的曲面网格
parent = Path(r'*')       # 试验数据根目录；脚本自动扫描其中含"吸力面"的末级子目录
```

完整流程（对应原始 `readme.txt`）：

1. 在 Pointwise 中绘制曲面网格，导出为 `mesh.x`，并量取 6 个标记点的三维坐标
   填入 `PS_PT`；
2. 配置上述 4 个变量后运行 `python main.py legacy`；
3. 弹出标定照片窗口，按 `PS_PT` 的顺序依次点击 6 个对应标记点；
4. 终端提示 `Whether tranformation satisfied? 0 or 1:`——输入 `0` 重新取点，
   输入 `1` 确认变换并继续；
5. 程序对扫描到的每个试验文件夹读取场数据文件 `h.plt`（代码中传 `h`，
   `read_file` 自动补 `.plt` 后缀），将 mesh 各点经投影矩阵
   `P` 映射到照片、双线性插值取值，展开结果写为该文件夹下的 `new_h.plt`
   （可用 Tecplot 打开）。

`data/`、`data1/` 中的原始试验 `.plt/.tiff` 数据体积较大，未随仓库分发；
但仓库根目录已附带一个示例 `mesh.x`（Pointwise 结构化网格）与标定照片
`calibration_image.png`，可直接查看这两种文件的格式。

### 格式转换小工具

两个独立的命令行小工具（不在 `requirements.txt` 中，按需安装）：
`plt_to_tiff.py` 依赖 matplotlib；`csv_to_tiff.py` 的 gray 模式只需
numpy + pillow，rgb/float 模式另需 matplotlib 与 tifffile：

```bash
# Tecplot .plt -> 彩色 TIFF
python plt_to_tiff.py input.plt [output.tiff] [--cmap jet]
# output 省略时默认为 input_color.tiff

# CSV 网格数据 -> TIFF
python csv_to_tiff.py input.csv output.tiff [--mode rgb] [--cmap jet] [--vmin .. --vmax ..]
# --mode: gray=8bit 灰度 / float=float32 原值 / rgb=彩色（默认，色标自动取 1%~99% 分位）
```

`check_output.py` 不是命令行工具，它提供 `check_output(C, P, mesh_file)`
函数：把 mesh 点经 `P` 投影、从图像 `C` 双线性插值后用等值线图可视化，
用于人工检查标定质量（采集器验证管线内部也复用了同样的投影插值逻辑）。

### 模块说明

| 模块 | 职责 |
|---|---|
| `main.py` | 入口：`collect`（默认，数据采集）/ `legacy`（手工标定展开） |
| `geometry.py` | 参数化三维曲面随机生成（弯度/扭转/波纹），mesh.x 读写互通 |
| `camera.py` | 随机针孔相机、自适应取景、覆盖率预筛、3×4 DLT 投影矩阵 |
| `field_gen.py` | 平面物理标量场真值（含气膜冷却孔排射流尾迹条带） |
| `render.py` | 逐像素光线-分片双线性曲面 Newton 求交，含自遮挡处理 |
| `collect_data.py` | 批量采集编排、配对落盘、逆管线验证、统计与清单 |
| `dataset_loader.py` | NumPy/PyTorch 数据集、标准化、train/val/test 划分 |
| `curve_flat.py` | 旧流程主函数：标定 P 后逐文件夹展开（`main.py legacy` 调用） |
| `read_mesh.py` `get_p.py` `file_output.py` | 旧管线：mesh.x 读取、点选 6 点标定 P、投影插值展开 |
| `read_file.py` `tecplot.py` | Tecplot 文本场数据读写（`read_file` 读 / `tecplot` 写） |
| `plt_to_tiff.py` `csv_to_tiff.py` `check_output.py` | 格式转换与结果检查小工具 |

### 许可证

[MIT](LICENSE)

---

<a name="english"></a>
## English

A physics-based **training-data generator** for learning to unwrap a camera-view image of
an unknown 3D curved surface directly into a 2-D planar scalar field (e.g. film-cooling
effectiveness / heat transfer coefficient on a turbine airfoil).

For every sample it (1) samples random surface geometry (chord/span, camber, spanwise
bending, twist, waviness), (2) samples a random pinhole camera and renders the
perspective view by per-pixel ray–patch intersection with self-occlusion handling,
(3) generates the planar ground-truth scalar field (low-frequency blobs, gradients,
film-cooling jet wakes), and (4) verifies physical consistency through an independent
inverse DLT pipeline that mirrors the experimental calibration workflow. Measured on
1000 random samples: mean normalized inverse RMSE ≈ **1.1%**, mean ground-truth vs.
reconstructed correlation ≈ **0.997**.

### Install & quick start

```bash
pip install -r requirements.txt          # numpy + pillow (torch optional)
python main.py collect --samples 10 --out dataset_demo --seed 0     # smoke test
python main.py collect --samples 1000 --out dataset_run --image-size 256 --grid 192 192 --seed 0
python dataset_loader.py dataset_run     # self-check
```

Or call it from Python:

```python
from collect_data import collect
collect(n_samples=200, out_dir='dataset_200', image_size=256,
        grid_nx=192, grid_ny=192, seed=0, save_plt=False)
```

~8.5 s and ~4 MB per sample at 256/192 resolution on CPU. Samples are written one by
one, so an interrupted run loses nothing; rerunning with the same `--seed` and `--out`
overwrites/patches in place. Note: collection (`collect`) needs only numpy + pillow;
the legacy workflow and the conversion tools additionally need matplotlib (see the
respective sections below).

### Data format

Each sample directory contains `input_view.npy` (float32 perspective view with zero
background), `target_planar.npy` (float32 unwrapped planar target), `view_mask.png`,
`depth.npy`, `mesh.x`, and `meta.json` (geometry labels + camera K/R/t/P + field
parameters + verification metrics), plus Tecplot and PNG previews. The dataset root has
`manifest.jsonl` (per-sample paths and metrics), `dataset_stats.json` (global
normalization statistics), and `dataset_config.json`. A verified sample is included
under [`examples/`](examples/) (`mesh.x` and the Tecplot file are omitted there to
keep the repo small; the remaining files are complete).

### Loading

```python
from dataset_loader import NumpySurfaceDataset

train, val, test = NumpySurfaceDataset('dataset_run').split((0.8, 0.1, 0.1))
item = train[0]                # dict: input (1,H,W), mask (1,H,W), target (1,nx,ny), ...
```

With PyTorch installed:

```python
from torch.utils.data import DataLoader
from dataset_loader import NumpySurfaceDataset, TorchSurfaceDataset

train_np = NumpySurfaceDataset('dataset_run').split()[0]
loader = DataLoader(TorchSurfaceDataset(train_np), batch_size=16, shuffle=True)
```

Quality filtering (drop poorly verified samples) works directly on the manifest:

```python
import json
recs = [json.loads(l) for l in open('dataset_run/manifest.jsonl', encoding='utf-8') if l.strip()]
good = [r for r in recs if r.get('verification', {}).get('verify_rmse_norm', 0) < 0.03]
```

A minimal training-loop skeleton (the repo ships no network definition):

```python
model = MyUnwrapNet().cuda()   # any image-to-image net, (B,1,256,256) -> (B,1,192,192)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for batch in loader:
    x, y = batch['input'].cuda(), batch['target'].cuda()   # already standardized
    loss = torch.nn.functional.mse_loss(model(x), y)       # background is 0, no masking needed
    opt.zero_grad(); loss.backward(); opt.step()
```

### CLI conversion tools

```bash
python plt_to_tiff.py input.plt [output.tiff] [--cmap jet]   # Tecplot .plt -> color TIFF
python csv_to_tiff.py input.csv output.tiff [--mode rgb] [--cmap jet] [--vmin .. --vmax ..]
```

These need matplotlib (plus tifffile for rgb/float modes), which is not in
`requirements.txt`; the gray mode of `csv_to_tiff.py` needs only numpy + pillow.
`check_output.py` exposes a `check_output(C, P, mesh_file)` function for visually
inspecting calibration quality.

### Customization

Sampling ranges live at the top of `geometry.py` (`GEOM_RANGES`), `camera.py`
(`CAM_RANGES`), and `field_gen.py` (`FIELD_RANGES`) — edit them or override at runtime:

```python
import geometry
geometry.GEOM_RANGES['L'] = (100.0, 140.0)   # restrict chord length to 100–140 mm
```

### Legacy workflow

The interactive calibration/unwrapping workflow for real photographs is kept under
`python main.py legacy` (needs matplotlib and a desktop environment for the
point-picking window): configure `PS_PT` (six 3-D marker coordinates), `fig_dir`
(calibration photo), `mesh_file`, and `parent` (data root; subfolders are scanned
automatically) in `legacy_main()`, click the six markers in the photo in order,
confirm with `1` at the prompt, and every scanned folder reads its `h.plt` and
gets an unwrapped `new_h.plt`. The original experimental `.plt/.tiff` data are not
redistributed, but the repo root ships an example `mesh.x` (Pointwise structured
grid) and `calibration_image.png` for format reference. Licensed under the
[MIT License](LICENSE).
