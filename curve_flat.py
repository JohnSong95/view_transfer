import os
import sys
import numpy as np
from PIL import Image

from get_p import get_p
from read_file import read_file
from file_output import file_output


def curve_flat(fig_dir, PT, folder_name, mesh_file):
    """主流程：标定投影矩阵 P，然后对每个文件夹中的 Eta.plt 做变换并输出。

    对应 MATLAB: curve_flat.m
    """
    # 读取标定图像：rgb2gray + im2double
    print('Loading image: ' + fig_dir)
    sys.stdout.flush()
    img = Image.open(fig_dir).convert('RGB')
    ori_fig = np.asarray(img, dtype=np.float64) / 255.0
    # rgb2gray：MATLAB 用 0.2989*R + 0.5870*G + 0.1140*B
    ori_fig = (0.2989 * ori_fig[:, :, 0] +
               0.5870 * ori_fig[:, :, 1] +
               0.1140 * ori_fig[:, :, 2])
    print('Image loaded, shape: ' + str(ori_fig.shape))
    sys.stdout.flush()

    # 计算投影矩阵 P
    print('Computing P matrix (please click 6 points on the image)...')
    sys.stdout.flush()
    P = get_p(ori_fig, PT, mesh_file)
    print('P matrix computed.')
    sys.stdout.flush()

    # 逐个处理文件夹（兼容 str 或 list[str]）
    if isinstance(folder_name, str):
        folder_list = [folder_name]
    else:
        folder_list = list(folder_name)
    for folder in folder_list:
        file_name = os.path.join(folder, 'h') #Eta
        eta = read_file(file_name)
    #    eta = eta.T   # MATLAB: eta = eta'
        file_output(eta, P, folder, mesh_file)
    print('All folders processed.')
    sys.stdout.flush()
