import os

import numpy as np


def convergence_judgment(cfg, state, history, ws, obj):
    if len(history.all_fom) < cfg.quant.test_length:
        print("The array does not have enough elements; "
              "it is impossible to make a judgment on convergence.")
    else:
        last_arrays = history.all_fom[-cfg.quant.test_length:]
        last_elements = [array[-1] for array in last_arrays]
        if (max(last_elements) - min(last_elements)) < cfg.quant.threshold:
            print("The current optimization has converged.")
            state.beta *= 1.2
            sim_file = "grey_complete"
            obj.model.save(os.path.join(ws.save_path, sim_file))
        else:
            print("The current optimization has not converged.")
    print(f"beta = {state.beta}")


def convergence_judgment_analog(cfg, state, history):
    """收敛升级：直接修改 state（与普通路径的 beta 更新保持一致）。

    调用时机应在当轮梯度更新之后，使升级只对下一轮生效，
    避免本轮正向结构与反向求导之间的滞后错位。
    """
    if len(history.all_fom) < cfg.quant.test_length:
        print("The array does not have enough elements; "
              "it is impossible to make a judgment on convergence.")
        return

    last_arrays = history.all_fom[-cfg.quant.test_length:]
    last_elements = [array[-1] for array in last_arrays]
    if ((max(last_elements) - min(last_elements)) < cfg.quant.threshold
            or history.non_convergence >= 100):
        print("The current optimization has converged.")
        if state.filter_R == (cfg.drc.filter_R_max - 1):
            state.beta *= 1.2
        else:
            state.filter_R += 1
    else:
        if state.beta > 1 or state.filter_R > cfg.drc.filter_R_min:
            history.non_convergence += 1
            print("The current optimization has not converged.")

    print(f"filter_R = {state.filter_R}")
    print(f"beta = {state.beta}")


def quant_1bit(rho_i, state, cfg):
    q, mat = cfg.quant, cfg.material

    if state.beta > q.beta_max:
        res = np.where((rho_i >= q.eta), mat.eps_max, mat.eps_min)
    else:
        res = mat.eps_min + (
                np.tanh(state.beta * q.eta)
                + np.tanh(state.beta * (rho_i - q.eta))) / (
                np.tanh(state.beta * q.eta)
                + np.tanh(state.beta * (1 - q.eta))) * (
                mat.eps_max - mat.eps_min)

    return res


def dq_drho(rho_i, state, cfg):
    const = 1e-12
    q, mat = cfg.quant, cfg.material

    rho_i = np.clip(rho_i, 0.0, 1.0)
    eta = np.clip(q.eta, 0.0, 1.0)

    u = state.beta * (rho_i - eta)
    t = np.tanh(u)

    denom = np.tanh(state.beta * eta) + np.tanh(state.beta * (1.0 - eta)) + const

    return state.beta * (mat.eps_max - mat.eps_min) * (1.0 - t * t) / denom


def d_quant_1bit(rho_i, state, cfg):
    if state.beta > cfg.quant.beta_max:
        res = np.where((rho_i >= cfg.quant.eta), 1, 0)
    else:
        res = dq_drho(rho_i, state, cfg)

    return res
