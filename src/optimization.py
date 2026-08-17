import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d

from src import quant


def interp_simtodesign_2d(original_data, region, method):
    interpolator = RegularGridInterpolator(
        (region.sim_y_pos, region.sim_x_pos),
        original_data,
        method=method,
        bounds_error=False,
        fill_value=None
    )

    yy, xx = np.meshgrid(region.y_pos, region.x_pos, indexing='ij')
    points = np.column_stack([yy.ravel(), xx.ravel()])

    result = interpolator(points).reshape(len(region.y_pos), len(region.x_pos))
    return result


def calculate_gradient_3d(obj, region):

    grad_eps_sim = np.mean(-2 * np.real(np.sum(obj.E_for * obj.E_adj, axis=3)), axis=2)
    grad_eps = interp_simtodesign_2d(grad_eps_sim, region, "linear")

    return grad_eps


def calculate_gradient_2d(obj, region):

    grad_eps_sim = -2 * np.real(np.sum(obj.E_for * obj.E_adj, axis=2))
    grad_eps = interp_simtodesign_2d(grad_eps_sim, region, "linear")

    return grad_eps


def calculate_gradient_2d_analog(obj, state, cfg, iteration):
    grad_eps = -2 * np.real(np.sum(obj.E_for * obj.E_adj, axis=2))

    R = state.filter_R
    R_max = cfg.drc.filter_R_max

    # 梯度相对于滤波输出 params_conv（设计区内部，即扩展场网格的中心区域）
    grad_conv = (
        grad_eps[R_max:-R_max, R_max:-R_max]
        * quant.d_quant_1bit(state.params_conv, state, cfg)
    )

    # 滤波的伴随：full 卷积后裁剪内部 R 区域
    grad_full = convolve2d(grad_conv, state.filter_kernel_inverse, mode="full")
    grad = grad_full[R:-R, R:-R]

    return grad


def adam_opt(i, m, v, params, grad, opt_cfg):

    # adam 超参数
    epsilon = 1e-8

    beta1 = opt_cfg.beta1
    beta2 = opt_cfg.beta2
    learning_rate = opt_cfg.learning_rate

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m_new / (1 - beta1 ** i)
    v_hat = v_new / (1 - beta2 ** i)
    params_opt = np.real(params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))
    # 保证优化后的结果在允许范围内
    params_opt = np.clip(params_opt, opt_cfg.params_min, opt_cfg.params_max)

    return m_new, v_new, params_opt


def adam_update(state, iteration, grad, cfg):
    """执行一次 Adam 更新并更新优化器状态。"""

    m_new, v_new, params_opt = adam_opt(
        iteration + state.iteration, state.m, state.v,
        state.params, grad, cfg.optimizer
    )
    state.m = m_new
    state.v = v_new
    state.params = params_opt
