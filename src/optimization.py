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

    # 将滤波后的参数写回带边界参数矩阵
    state.params_all[
        cfg.drc.filter_R_max:-cfg.drc.filter_R_max,
        cfg.drc.filter_R_max:-cfg.drc.filter_R_max,
    ] = state.params_conv

    deps_dparams_hat = quant.d_quant_1bit(
        state.params_all[
            state.delta_R:-state.delta_R,
            state.delta_R:-state.delta_R,
        ],
        state,
        cfg,
    )

    # 梯度从滤波边界区域裁剪到当前滤波半径对应的有效区域
    grad_params_hat = (
        grad_eps[
            state.delta_R:-state.delta_R,
            state.delta_R:-state.delta_R,
        ]
        * deps_dparams_hat
    )

    # 反卷积得到设计参数梯度
    grad = convolve2d(grad_params_hat, state.filter_kernel_inverse, mode="valid")

    # 在方法内部直接更新优化参数
    adam_update(state, iteration, grad, cfg)

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
