"""Lumerical 模型构建与设计区域刷新。"""

import os
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d

import lumapi

from src import environment as env
from src import filter


def load_from_lsf(script_file_name):
    with open(os.path.join(env.LSF_PATH, script_file_name), 'r') as text_file:
        lines = [line.strip().split(sep='#', maxsplit=1)[0] for line in text_file.readlines()]
    script = ''.join(lines)
    if not script:
        raise UserWarning('empty script.')
    return script


def initial_rect(obj, region):

    obj.model.eval(
        "addstructuregroup;" +
        "set('name', 'design_pixel');" +
        "set('construction group', 0);" +
        "set('x', 0);" +
        "set('y', 0);" +
        "set('z', 0);" +

        "set('script', \" " +
        "opt_size_x = {};".format(region.size_x) +
        "opt_size_y = {};".format(region.size_y) +
        "opt_size_z = {};".format(region.size_z) +
        "pixel_size = {};".format(region.pixel_size) +
        "begin_x = -opt_size_x / 2 + pixel_size / 2;" +
        "begin_y = opt_size_y / 2 - pixel_size / 2;" +
        "row = opt_size_y / pixel_size;" +
        "col = opt_size_x / pixel_size;" +
        "for (i=1:row){" +
        "for (j=1:col) {" +
        "   addrect;" +
        "set('name', 'pixel_' + num2str(i) + '_' + num2str(j));" +
        "set('x', begin_x + pixel_size * (j - 1));" +
        "set('x span', pixel_size);" +
        "set('y', begin_y - pixel_size * (i - 1));" +
        "set('y span', pixel_size);" +
        "set('z', 0);" +
        "set('z span', opt_size_z);" +
        "set('index', (3.47 - 1.44) * 0.5 + 1.44);" +
        "}" +
        "}" +
        "\");"
    )

    while True:
        time.sleep(1)
        number = obj.model.getnamednumber(f"design_pixel::pixel_{region.y_points}_{region.x_points}")
        if number == 1:
            print(f"Design Pixels Initialized successfully")
            break


def initialize_model_3d_rect(obj, ws, region, fdtd):

    model = lumapi.FDTD(hide=fdtd.fdtd_gui)
    script = load_from_lsf(obj.filename)
    model.eval(script)
    model.eval('setnamed("FDTD", "express mode", {});'.format(fdtd.express_mode))
    model.setnamed('FDTD', 'dimension', f'{fdtd.fdtd_dimension}')
    save_path = os.path.join(ws.sub_path, obj.filename)
    model.save(save_path)
    obj.model = model
    initial_rect(obj, region)


def refresh_design_region_3d_rect(obj, region, optstate):

    commands_str = ""
    for i in range(1, region.y_points + 1):
        for j in range(1, region.x_points + 1):
            command = f"setnamed('pixel_{i}_{j}','index',{optstate.index_opt[i-1, j-1]:.6f});\n"
            commands_str += command

    commands_str += f"addrect; \n set('name', 'flag');"

    obj.model.switchtolayout()

    obj.model.eval(
        "select('design_pixel');" +
        "set('script', \" " +
        "{}".format(commands_str) +
        "\");"
    )

    while True:
        time.sleep(1)
        number = obj.model.getnamednumber("design_pixel::flag")
        if number == 1:
            obj.model.eval(
                "select('design_pixel');" +
                "set('script', \" " +
                "\");"
            )
            time.sleep(0.1)
            obj.model.select("design_pixel::flag")
            obj.model.delete()
            number = obj.model.getnamednumber("design_pixel::flag")
            print(f"Design pixels updated successfully")
            break


def initialize_model_import(obj, state, ws, region, fdtd):

    model = lumapi.FDTD(hide=fdtd.fdtd_gui)
    script = load_from_lsf(obj.filename)
    model.eval(script)
    model.eval('setnamed("FDTD", "express mode", {});'.format(fdtd.express_mode))
    model.setnamed('FDTD', 'dimension', f'{fdtd.fdtd_dimension}')
    save_path = os.path.join(ws.sub_path, obj.filename)
    model.save(save_path)
    obj.model = model
    refresh_design_region_import(obj, region, state)


def interp_designtosim_2d(original_data, region, method):
    y_min, y_max = region.y_pos[0], region.y_pos[-1]
    x_min, x_max = region.x_pos[0], region.x_pos[-1]

    yy, xx = np.meshgrid(region.sim_y_pos, region.sim_x_pos, indexing='ij')

    yy_clipped = np.clip(yy, y_min, y_max)
    xx_clipped = np.clip(xx, x_min, x_max)

    points = np.column_stack([yy_clipped.ravel(), xx_clipped.ravel()])

    interpolator = RegularGridInterpolator(
        (region.y_pos, region.x_pos),
        original_data,
        method=method,
        bounds_error=False,
        fill_value=None
    )

    result = interpolator(points).reshape(len(region.sim_y_pos), len(region.sim_x_pos))
    return result


def refresh_design_region_import(obj, region, optstate):

    # 利用邻近插值生成对应的 index 矩阵
    index_opt_inter = interp_designtosim_2d(optstate.index_opt, region, "nearest")

    obj.model.switchtolayout()
    obj.model.select("import")
    obj.model.delete()
    tensor = np.tile(np.flipud(index_opt_inter), (len(region.z_pos), 1, 1))
    tensor = tensor.transpose(2, 1, 0)

    # 更新优化区域内的结构
    obj.model.addimport()
    obj.model.putv('tensor', tensor)
    obj.model.putv('x', region.sim_x_pos)
    obj.model.putv('y', region.sim_y_pos)
    obj.model.putv('z', region.z_pos)
    obj.model.eval("importnk2(tensor, x, y, z);")


def initialize_model_import_analog(obj, state, ws, region, fdtd, cfg):

    model = lumapi.FDTD(hide=fdtd.fdtd_gui)
    script = load_from_lsf(obj.filename)
    script = script.replace('filter_r = 3;', 'filter_r = {:d};'.format(cfg.drc.filter_R_max))
    model.eval(script)
    model.eval('select("fields_mesh");'
               'set("dx", {});'
               'set("dy", {});'
               'set("dz", {});'
               'setnamed("FDTD","min mesh step", {});'.format(region.dx, region.dy, region.dz, region.dx)
               )

    model.eval('setnamed("FDTD", "express mode", {});'.format(fdtd.express_mode))
    model.setnamed('FDTD', 'dimension', f'{fdtd.fdtd_dimension}')
    save_path = os.path.join(ws.sub_path, obj.filename)
    model.save(save_path)
    obj.model = model
    refresh_design_region_import_analog(obj, region, state, cfg)

    index_all = obj.model.getresult("opt_index", "index")
    eps_all = np.rot90(np.real(np.squeeze(index_all['index_x']) ** 2), k=1)

    eps_all[cfg.drc.filter_R_max:-cfg.drc.filter_R_max,
            cfg.drc.filter_R_max:-cfg.drc.filter_R_max] = state.eps_opt
    state.params_all = (eps_all - cfg.material.eps_min) / (cfg.material.eps_max - cfg.material.eps_min)
    state.params_conv = state.params


def refresh_design_region_import_analog(obj, region, state, cfg):

    state.delta_R = cfg.drc.filter_R_max - state.filter_R
    state.filter_kernel, state.filter_kernel_inverse = filter.gen_filter(state.filter_R, 'mean')

    # 利用邻近插值生成对应的 index 矩阵
    index_opt_inter = interp_designtosim_2d(state.index_opt, region, "nearest")

    obj.model.switchtolayout()
    obj.model.select("import")
    obj.model.delete()
    tensor = np.tile(np.flipud(index_opt_inter), (len(region.z_pos), 1, 1))
    tensor = tensor.transpose(2, 1, 0)

    # 更新优化区域内的结构
    obj.model.addimport()
    obj.model.putv('tensor', tensor)
    obj.model.putv('x', region.sim_x_pos)
    obj.model.putv('y', region.sim_y_pos)
    obj.model.putv('z', region.z_pos)
    obj.model.eval("importnk2(tensor, x, y, z);")


def update_params_conv_analog(state, cfg):
    """将优化参数写回带边界的参数矩阵，并更新滤波后的参数。"""

    state.params_all[
        cfg.drc.filter_R_max:-cfg.drc.filter_R_max,
        cfg.drc.filter_R_max:-cfg.drc.filter_R_max,
    ] = state.params

    params_conv_unclip = convolve2d(
        state.params_all[
            state.delta_R:-state.delta_R,
            state.delta_R:-state.delta_R,
        ],
        state.filter_kernel,
        mode='valid',
    )

    state.params_conv = np.clip(
        params_conv_unclip,
        cfg.optimizer.params_min,
        cfg.optimizer.params_max,
    )
