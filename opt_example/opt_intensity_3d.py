"""intensity_3d 任务主函数（参数见 configs/intensity_3d.py）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import src.environment as env

env.configure()

from configs.intensity_3d import make_config
from src.config import History, OptimState, build_object, build_region
import src.fom as fom
import src.optimization as opt
import src.plot as plot
import src.quant as quant
import src.setting as setting
import src.simulation as sim


def main():
    cfg = make_config()
    ws = env.create_workspace(cfg.output_dir)
    region = build_region(cfg.region)
    obj = build_object(cfg.objects[0])
    state = OptimState.create(region, cfg, npz_path=cfg.resume)
    history = History()

    setting.initialize_model_3d_rect(obj, ws, region, cfg.fdtd)

    for i in range(1, cfg.optimizer.max_epoch + 1):
        print(f"Interation {i}")
        setting.refresh_design_region_3d_rect(obj, region, state)

        # 进行正向仿真
        sim.make_forward_sim_3d(obj, cfg.fdtd, region)
        fom.get_fom_intensity(obj, region.V_cell)

        plot.fom_display(obj, state, history, ws, i)

        if state.beta < cfg.quant.beta_max:
            # 进行伴随仿真（此时 beta 与正向结构一致）
            sim.make_adjoint_sim_3d(obj, cfg.fdtd)
            grad_eps = opt.calculate_gradient_3d(obj, region)

            deps_dparams = quant.d_quant_1bit(state.params, state, cfg)
            grad = grad_eps * deps_dparams

            # 根据 Adam 优化算法进行梯度下降更新
            opt.adam_update(state, i, grad, cfg)

            # 梯度更新完成后统一升级，仅对下一轮生效
            quant.convergence_judgment(cfg, state, history, ws, obj)

            state.eps_opt = quant.quant_1bit(state.params, state, cfg)
            state.index_opt = np.sqrt(state.eps_opt)

            plot.plot_optresult(obj, history, ws, i)
        else:
            plot.plot_optresult(obj, history, ws, i)
            break


if __name__ == "__main__":
    main()
