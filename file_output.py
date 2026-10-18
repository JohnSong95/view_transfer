import os
import numpy as np

from read_mesh import read_mesh
from tecplot import tecplot


def file_output(C, P, folder_name, mesh_file):
    """对 mesh 上每个点应用 P 投影到 2D，用 C 做双线性插值，然后输出 .plt 文件。

    对应 MATLAB: file_output.m
    """
    delta_x, delta_y, mesh = read_mesh(mesh_file)
    max_i = mesh.shape[2]   # j_max
    max_j = mesh.shape[1]   # i_max
    H, W = C.shape

    new_eta = np.zeros((max_j, max_i))
    XX = np.zeros((max_j, max_i))
    YY = np.zeros((max_j, max_i))

    for i in range(max_i):
        for j in range(max_j):
            L = np.array([mesh[0, j, i, 0],
                          mesh[1, j, i, 0],
                          mesh[2, j, i, 0],
                          1.0])
            R = P @ L
            xx = R[0] / R[2]
            yy = R[1] / R[2]

            fx = np.floor(xx)
            cx = np.ceil(xx)
            fy = np.floor(yy)
            cy = np.ceil(yy)

            if (yy >= 0 and yy <= H - 1 and xx >= 0 and xx <= W - 1):
                fx_i = int(min(max(fx, 0), W - 1))
                cx_i = int(min(max(cx, 0), W - 1))
                fy_i = int(min(max(fy, 0), H - 1))
                cy_i = int(min(max(cy, 0), H - 1))

                eta_tl = C[fy_i, cx_i]
                eta_bl = C[fy_i, fx_i]
                eta_tr = C[cy_i, cx_i]
                eta_br = C[cy_i, fx_i]

                dx = cx - fx
                dy = cy - fy
                if dx == 0 and dy == 0:
                    new_eta[j, i] = eta_bl
                elif dx == 0:
                    new_eta[j, i] = (eta_bl * (cy - yy) +
                                     eta_br * (yy - fy)) / dy
                elif dy == 0:
                    new_eta[j, i] = (eta_bl * (cx - xx) +
                                     eta_tl * (xx - fx)) / dx
                else:
                    new_eta[j, i] = (eta_tl * (cy - yy) * (xx - fx) +
                                     eta_bl * (cy - yy) * (cx - xx) +
                                     eta_tr * (yy - fy) * (xx - fx) +
                                     eta_br * (yy - fy) * (cx - xx)) / (dy * dx)

            XX[j, i] = (i + 1) * delta_x
            YY[j, i] = (j + 1) * delta_y

    save_name = os.path.join(folder_name, 'new_h.plt')
    tecplot(XX, YY, new_eta, save_name)
    print('Saved ' + save_name)
