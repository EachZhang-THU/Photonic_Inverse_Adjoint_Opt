import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os
from scipy.ndimage import distance_transform_cdt

def gen_filter(filter_R, kernel_type='mean'):
    kernel_length = 2 * filter_R + 1
    if kernel_type == 'conical':
        # conical_kernel
        filter_kernel = generate_conical_kernel(filter_R)
    elif kernel_type == 'mean':
        # mean kernel
        filter_kernel = np.ones((kernel_length, kernel_length)) / (kernel_length * kernel_length)
    else:
        # #empty kernel
        filter_kernel = np.zeros((kernel_length, kernel_length))
        filter_kernel[filter_R, filter_R] = 1

    # create kernel to calculate grad
    filter_kernel_inverse = np.zeros_like(filter_kernel)
    for i in range(kernel_length):
        for j in range(kernel_length):
            filter_kernel_inverse[2 * filter_R - i][2 * filter_R - j] = filter_kernel[i][j]

    return filter_kernel, filter_kernel_inverse

def generate_conical_kernel(filter_R):
    core_length = 2 * filter_R + 1
    kernel = np.ones((core_length, core_length))
    kernel[filter_R, filter_R] = 0
    # 计算距离变换
    distance = distance_transform_cdt(kernel)
    # 应用权重（这里使用距离的倒数）
    weights = 1 / (distance+1)  # 加1以避免除以零
    # 归一化权重
    weights /= np.sum(weights)

    return weights
