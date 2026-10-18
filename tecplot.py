import numpy as np


def tecplot(XX, YY, Eta, save_name):
    """将结果写入 Tecplot .plt 文件。

    与 check_output 的 plt.contourf(XX, YY, new_eta) 行为一致：
    - Eta[i, j] 对应坐标 (XX[i, j], YY[i, j])
    - i 对应 Y 方向（行），j 对应 X 方向（列）
    - 让 Tecplot 的 I 方向对应 X 方向（j），J 方向对应 Y 方向（i）
    - I（X）方向变化最快（内层循环），保证 Tecplot 中"沿 I 方向相邻"=X 相邻
    """
    m, n = Eta.shape   # m = Y 方向点数（第一维 i），n = X 方向点数（第二维 j）
    # 防止 NaN/Inf 写入
    Eta = np.where(np.isfinite(Eta), Eta, 0.0)

    with open(save_name, 'w') as fid:
        fid.write('TITLE = "h data " \n') #PSP
        fid.write('VARIABLES = "X", "Y", "H" ,\n') #Effectiveness
        # I = X 方向点数 (n)，J = Y 方向点数 (m)
        fid.write('ZONE I={}, J={}, F=POINT\n'.format(n, m))
        # 外层 J = Y 方向（行索引 i），内层 I = X 方向（列索引 j）
        # I (X) 方向变化最快，与 contourf 显示的"行"一致
        for i in range(m):
            for j in range(n):
                fid.write('%g %g %g\r\n' % (XX[i, j], YY[i, j], Eta[i, j]))
