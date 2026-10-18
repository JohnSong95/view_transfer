# view_transfer —— 曲面透视图展开模型的训练数据采集器

[English](#english) | [中文](#中文)

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
                  独立逆管线（P 矩阵 DLT 反算，复刻实测标定流程）──> RMSE / 相关系数验证
```

在 1000 个随机样本上的实测指标：逆管线反算归一化 RMSE 均值约 **1.1%**，真值/反算
相关系数均值约 **0.997**（逐样本指标写在 `manifest.jsonl` 中）。

### 安装

需要 Python 3.10+（仅依赖 numpy 与 pillow；PyTorch 为可选依赖，用于自带的 Dataset）：

```bash
pip install -r requirements.txt
# 可选：训练侧使用 torch.utils.data.Dataset 封装时再安装 torch
```

### 快速开始：批量采集

```bash
# 采集 1000 对样本（256×256 透视图，192×192 平面真值）
python main.py collect --samples 1000 --out dataset_run --image-size 256 --grid 192 192 --seed 0

# 先试跑 10 个样本
python main.py collect --samples 10 --out dataset_demo
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--samples N` | 100 | 样本数量 |
| `--out DIR` | dataset | 输出目录（建议用带时间戳的独立目录） |
| `--seed` | 0 | 随机种子；同 seed + 同输出目录可复现/覆盖补齐 |
| `--image-size` | 256 | 透视图边长（像素） |
| `--grid H W` | 192 192 | 平面真值网格分辨率 |
| `--min-visibility` | 0.35 | 曲面最小可见率，低于则重新随机机位 |
| `--supersample` | 1 | 抗锯齿超采样倍率（可选 2，渲染约慢 4 倍） |
| `--no-plt` | 关 | 不写 Tecplot `.plt` 文件（每样本省约 1 MB） |
| `--no-verify` | 关 | 跳过逆管线验证（不建议） |

性能参考（256/192 分辨率，纯 CPU）：约 **8.5 秒/样本、4 MB/样本**。样本逐个落盘，
中断后已生成部分全部可用。

### 样本与标签格式

每个样本一个目录 `sample_XXXXXX/`：

| 文件 | 形状/类型 | 含义 |
|---|---|---|
| `input_view.npy` | float32 (H,W) | **模型输入**：透视渲染的曲面物理量，背景为 0 |
| `view_mask.png` | uint8 | 可见曲面掩码（255=曲面） |
| `depth.npy` | float32 (H,W) | 深度标签（辅助任务/几何监督用） |
| `target_planar.npy` | float32 (Gh,Gw) | **训练目标**：展开后的平面物理量真值（未做 8-bit 量化） |
| `target_planar.plt` | Tecplot 文本格式 | 平面真值，可用 Tecplot 打开 |
| `target_planar.png` / `input_view.png` | 图片 | 可视化预览 |
| `mesh.x` | 文本网格 | 该样本的 3D 曲面网格（与 `read_mesh.py` 互通） |
| `meta.json` | JSON | 几何参数 + 相机内外参 + 物理场参数 + 验证指标 |

数据集根目录另有：`manifest.jsonl`（逐样本清单与验证指标）、`dataset_stats.json`
（全局均值/标准差）、`dataset_config.json`（坐标约定）。可参考
[`examples/`](examples/) 中的单样本示例。

`meta.json` 中的相机矩阵 `P` 为 3×4 DLT 矩阵（y 像素坐标向下），与常规针孔标定
（及本项目旧的手工标定流程）同口径；几何标签包括弦长、展宽、弯度、展向弯曲、
扭转、波纹幅值等。

### 加载数据

```python
from dataset_loader import NumpySurfaceDataset

trainval = NumpySurfaceDataset('dataset_run')          # 自动按 stats 标准化
train, val, test = trainval.split((0.8, 0.1, 0.1))
x, y = train[0]                                        # x: (1,H,W), y: (1,Gh,Gw)

# 若安装了 PyTorch：
from dataset_loader import TorchSurfaceDataset
train_ds = TorchSurfaceDataset(train)
```

### 物理一致性验证

采集器对每个样本复用实测数据处理管线（`file_output.py` 的 DLT 反算 + Tecplot
读写），由透视图和 `P` 反推平面场，与真值比较得到 `verify_rmse_norm` 与
`verify_corr`，写入 `meta.json` 与 `manifest.jsonl`。也可单独自检：

```bash
python dataset_loader.py dataset_run
```

### 适配你自己的几何与物理量

- 几何参数采样范围：`geometry.py` 顶部的 `RANGES`（弦长/展宽/弯度/扭转/波纹）。
- 物理场参数（高斯团数量、梯度、气膜孔排位置/尾迹宽度）：`field_gen.py` 顶部配置。
- 相机视场角/机位范围、主点与滚转：`camera.py` 顶部配置。

### 旧流程：实测照片的手工标定展开

除数据采集外，本仓库保留了原有的实测处理流程（交互式选标定标记点）：

```bash
python main.py legacy
```

流程说明（原始 `readme.txt`）：在 Pointwise 中绘制曲面网格导出 `mesh.x` → 将标记点
坐标写入 `main.py` 的 `PT` 变量 → `fig_dir` 指向标定照片 → 选取对应点并确认变换。
`data/`、`data1/` 中的原始试验 `.plt/.tiff` 数据体积较大，未随仓库分发。

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
| `read_mesh.py` `get_p.py` `file_output.py` | 旧管线：网格读取、手工标定、DLT 反算展开 |
| `read_file.py` `tecplot.py` | Tecplot 文本网格/场数据读写 |
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

```bash
pip install -r requirements.txt
python main.py collect --samples 1000 --out dataset_run --seed 0
```

Each sample contains `input_view.npy` (model input, float32 perspective view with zero
background), `target_planar.npy` (unwrapped planar target), `view_mask.png`,
`depth.npy`, `mesh.x`, `meta.json` (geometry labels + camera K/R/t/P + verification
metrics), plus Tecplot and PNG previews. Load with:

```python
from dataset_loader import NumpySurfaceDataset
train, val, test = NumpySurfaceDataset('dataset_run').split((0.8, 0.1, 0.1))
```

The legacy interactive calibration/unwrapping workflow for real photographs is kept
under `python main.py legacy`. Geometry/field/camera sampling ranges are configured at
the top of `geometry.py`, `field_gen.py`, and `camera.py`. See
[`examples/`](examples/) for a sample. Licensed under the [MIT License](LICENSE).
