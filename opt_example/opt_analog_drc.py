"""analog_drc 任务主函数（参数见 configs/analog_drc.py）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import src.environment as env

env.configure()

from configs.analog_drc import make_config
from src.config import History, OptimState, build_object, build_region
import src.drc as drc
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

    setting.initialize_model_import_analog(obj, state, ws, region, cfg.fdtd, cfg)

    for i in range(1, cfg.optimizer.max_epoch + 1):
        print(f"Interation {i}")
        setting.refresh_design_region_import_analog(obj, region, state, cfg)

        # 进行正向仿真
        sim.make_forward_sim_2d_analog(obj, cfg.fdtd, region)
        fom.get_fom_intensity(obj, region.V_cell)

        plot.fom_display(obj, state, history, ws, i)

        if state.beta < cfg.quant.beta_max:
            quant.convergence_judgment_analog(cfg, state, history)

            # 进行伴随仿真
            sim.make_adjoint_sim_2d_analog(obj, cfg.fdtd)
            # 计算梯度并在内部更新参数
            grad = opt.calculate_gradient_2d_analog(obj, state, cfg, i)
        else:
            params_opt, drc_passed = drc.run_drc_iteration(
                cfg, ws, region, obj, state.params, i)
            state.params = params_opt
            if drc_passed:
                break

        setting.update_params_conv_analog(state, cfg)

        # 执行量化操作
        state.eps_opt = quant.quant_1bit(state.params, state, cfg)
        state.index_opt = np.sqrt(state.eps_opt)

        plot.plot_optresult(obj, history, ws, i)


if __name__ == "__main__":
    main()
