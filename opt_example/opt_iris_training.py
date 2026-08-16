"""iris_training 任务主函数（参数见 configs/iris_training.py）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import src.environment as env

env.configure()

from configs.iris_training import make_config
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

    data = np.load(env.DATASET_PATH / cfg.dataset.path)
    keys = data.files
    data_batch_num = int(len(keys) / cfg.dataset.batch_size)

    setting.initialize_model_import(obj, state, ws, region, cfg.fdtd)

    for i in range(1, cfg.optimizer.max_epoch + 1):
        print(f"Interation {i}")

        grad_all = np.zeros((region.y_points, region.x_points))
        all_batch_fom = np.zeros((cfg.dataset.batch_size, len(obj.fom_name)))

        for num_batch in range(cfg.dataset.batch_size):
            data_index_factor = (i - 1) % data_batch_num
            key = keys[num_batch + cfg.dataset.batch_size * data_index_factor]
            sample = data[key]

            obj.forward_source_phase[:] = sample[:len(obj.forward_source_phase)] * 360
            obj.target_fom[:] = sample[-len(obj.target_fom):]

            setting.refresh_design_region_import(obj, region, state)

            # 进行正向仿真
            sim.make_forward_sim_2d(obj, cfg.fdtd, region)
            fom.get_fom_ce(obj, region.V_cell)

            all_batch_fom[num_batch, :] = np.real(obj.fom)

            # 进行伴随仿真
            sim.make_adjoint_sim_2d(obj, cfg.fdtd)
            grad_eps = opt.calculate_gradient_2d(obj, region)
            grad_all += grad_eps

        grad_all = grad_all / cfg.dataset.batch_size

        plot.fom_training_display(obj, cfg, state, history, ws, i, all_batch_fom)
        quant.convergence_judgment(cfg, state, history, ws, obj)

        deps_dparams = quant.d_quant_1bit(state.params, state, cfg)
        # 注意：与原脚本一致，此处仍使用最后一个 batch 的 grad_eps
        grad = grad_eps * deps_dparams

        # 根据 Adam 优化算法进行梯度下降更新
        opt.adam_update(state, i, grad, cfg)

        state.eps_opt = quant.quant_1bit(state.params, state, cfg)
        state.index_opt = np.sqrt(state.eps_opt)

        plot.plot_optresult(obj, history, ws, i)


if __name__ == "__main__":
    main()
