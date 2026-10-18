import numpy as np

# 缓存已读取的 mesh，避免重复读取和重复提示 delta_y
_mesh_cache = {}


def read_mesh(mesh_file):
    """读取 mesh 文件，返回 (delta_x, delta_y, mesh)。

    对应 MATLAB: read_mesh.m
    mesh 形状为 (3, i_max, j_max, z_max)，mesh[coord, i, j, k] 为坐标分量
    （coord=0/1/2 对应 x/y/z）。
    对同一文件多次调用时使用缓存，避免重复交互输入。
    """
    if mesh_file in _mesh_cache:
        return _mesh_cache[mesh_file]
    # 模拟 MATLAB textread：所有数字按行读取，短行用 0 填充到 4 列
    rows = []
    with open(mesh_file, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                rows.append([0.0, 0.0, 0.0, 0.0])
                continue
            nums = [float(p) for p in parts]
            while len(nums) < 4:
                nums.append(0.0)
            rows.append(nums[:4])
    Mesh = np.array(rows)

    i_max = int(Mesh[1, 0])
    j_max = int(Mesh[1, 1])
    z_max = int(Mesh[1, 2])

    # MATLAB: Mesh = Mesh(3:end, 1:4); Mesh = Mesh'; Mesh = Mesh(:)
    # 在 numpy 中相当于 row-major 展平
    data = Mesh[2:, :4]               # shape (num_data_rows, 4)
    flat = data.flatten()             # 长度 = 4 * num_data_rows = 3 * i_max * j_max * z_max

    mesh = np.zeros((3, i_max, j_max, z_max))
    index = 0
    for j in range(j_max):
        mesh[0, :, j, 0] = flat[index:index + i_max]
        index += i_max
    for j in range(j_max):
        mesh[1, :, j, 0] = flat[index:index + i_max]
        index += i_max
    for j in range(j_max):
        mesh[2, :, j, 0] = flat[index:index + i_max]
        index += i_max

    # delta_x：j 方向步长（沿 j 变化，i 固定）
    # delta_y：i 方向步长（沿 i 变化，j 固定）
    # 用欧氏距离计算，适用于平面/曲面网格（不依赖单一坐标分量为零）
    def _step_len(coord_axis, vary_axis):
        """计算沿 vary_axis 方向相邻点的平均欧氏步长。"""
        if vary_axis == 1:  # 沿 i 变化（mesh[:, i, j, 0] 中 i 变）
            p1 = mesh[:, 0, 0, 0]
            p2 = mesh[:, 1, 0, 0]
        else:  # 沿 j 变化（mesh[:, i, j, 0] 中 j 变）
            p1 = mesh[:, 0, 0, 0]
            p2 = mesh[:, 0, 1, 0]
        return float(np.linalg.norm(p2 - p1))

    delta_x = _step_len(0, vary_axis=2)   # j 方向步长
    delta_y = _step_len(1, vary_axis=1)   # i 方向步长

    result = (delta_x, delta_y, mesh)
    _mesh_cache[mesh_file] = result
    return result
