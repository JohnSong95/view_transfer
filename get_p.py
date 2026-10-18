import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 确保使用 Tk 后端
from check_output import check_output

def set_window_topmost_transparent(fig, alpha=0.7):
    """
    将 matplotlib Figure 窗口设为置顶 + 整体半透明
    alpha: 0.0(全透明) ~ 1.0(不透明)
    """
    backend = matplotlib.get_backend().lower()
    win = fig.canvas.manager.window

    try:
        # ---------- Qt 系列 (PyQt5/6, PySide2/6) ----------
        if 'qt' in backend:
            # 自动导入可用的 Qt 绑定
            try:
                from PyQt6.QtCore import Qt
            except ImportError:
                try:
                    from PyQt5.QtCore import Qt
                except ImportError:
                    try:
                        from PySide6.QtCore import Qt
                    except ImportError:
                        from PySide2.QtCore import Qt

            # 置顶标志
            win.setWindowFlags(win.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
            # 整体透明度
            win.setWindowOpacity(alpha)
            # 修改 flags 后必须重新 show，否则窗口会隐藏
            win.show()

        # ---------- TkAgg ----------
        elif 'tk' in backend:
            win.attributes('-alpha', alpha)      # 0.0 ~ 1.0
            win.attributes('-topmost', True)     # 置顶

        # ---------- wxWidgets ----------
        elif 'wx' in backend:
            import wx
            win.SetTransparent(int(alpha * 255))
            win.SetWindowStyle(win.GetWindowStyle() | wx.STAY_ON_TOP)

        else:
            print(f"[警告] 当前 backend '{backend}' 未适配，请手动置顶窗口。")

    except Exception as e:
        print(f"[错误] 设置窗口属性失败: {e}")


def get_p(eta, A, mesh_file):
    """通过 ginput 选取 6 个标定点，用 SVD 求解 3x4 投影矩阵 P。
    """
    check = False
    P = None
    while not check:
        plt.figure()
        plt.imshow(eta * 500, cmap='gray', origin='upper')
        plt.axis('image')
        plt.title('Click 6 calibration points (in order matching PS_PT rows)')
        # 在 ginput 前设置窗口属性
        #set_window_topmost_transparent(plt.gcf(), alpha=0.7)
        # ginput 在 matplotlib 中返回 0-based 坐标
        part1 = plt.ginput(6, timeout=0)
        plt.close()

        x = np.array([p[0] for p in part1])
        y = np.array([p[1] for p in part1])
        #y = 1207 - y    #坐标对齐

        X = A[:, 0]
        Y = A[:, 1]
        Z = A[:, 2]

        # 构建 12x12 矩阵 C（对应 MATLAB 的拼接方式）
        C = np.zeros((12, 12))
        for i in range(6):
            D = np.array([
                [0, 0, 0, 0, -X[i], -Y[i], -Z[i], -1,
                 y[i] * X[i], y[i] * Y[i], y[i] * Z[i], y[i]],
                [X[i], Y[i], Z[i], 1, 0, 0, 0, 0,
                 -x[i] * X[i], -x[i] * Y[i], -x[i] * Z[i], -x[i]]
            ])
            C[2 * i:2 * i + 2, :] = D

        # SVD：MATLAB [U,S,V] = svd(C)；numpy 返回 (U, S, Vt)，故 V = Vt.T
        U, S, Vt = np.linalg.svd(C)
        V = Vt.T
        P = np.zeros((3, 4))
        P[0, :] = V[0:4, -1]
        P[1, :] = V[4:8, -1]
        P[2, :] = V[8:12, -1]

        # --- 验证 P 矩阵：检查标定点回投影误差 ---
        print('\n--- P 矩阵验证 ---')
        for i in range(6):
            L_cal = np.array([X[i], Y[i], Z[i], 1.0])
            R_cal = P @ L_cal
            r2 = R_cal[2]
            if abs(r2) < 1e-12:
                print('  标定点 %d: R[2] 太小! r2=%.2e, 投影无效' % (i+1, r2))
                continue
            x_proj = R_cal[0] / r2
            y_proj = R_cal[1] / r2
            err_x = abs(x_proj - x[i])
            err_y = abs(y_proj - y[i])
            print('  标定点 %d: 目标像素=(%.1f, %.1f), 回投影=(%.1f, %.1f), 误差=(%.2f, %.2f), R[2]=%.4f' %
                  (i+1, x[i], y[i], x_proj, y_proj, err_x, err_y, r2))
        print('SVD 奇异值:', S)
        sys.stdout.flush()

        # 显示结果以供检查
        check_output(eta, P, mesh_file)
        plt.show(block=False)
        plt.pause(0.5)  # 让窗口有时间渲染

        while True:
            response = input('Whether tranformation satisfied? 0 or 1: ').strip()
            if response == '1':
                check = True
                plt.close('all')
                break
            elif response == '0':
                plt.close('all')
                break
            else:
                print('Wrong input! 0 or 1:')

    return P
