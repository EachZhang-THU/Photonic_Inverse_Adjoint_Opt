import numpy as np


def make_forward_sim_3d(obj, fdtd, region):
    # sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.adjoint_source_name)):
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", False)
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", True)
        obj.model.set("phase", obj.forward_source_phase[i])
        obj.model.set("amplitude", np.abs(obj.forward_source_amp[i]))

    obj.model.save(obj.filename)
    obj.model.run("FDTD", fdtd.device)

    # get forward fields
    forward_Ex = obj.model.getresult("opt_fields", "Ex")
    forward_Ey = obj.model.getresult("opt_fields", "Ey")
    forward_Ez = obj.model.getresult("opt_fields", "Ez")

    E_stack = np.stack([np.squeeze(forward_Ex), np.squeeze(forward_Ey), np.squeeze(forward_Ez)], axis=3)

    obj.E_for = np.rot90(E_stack, k=1, axes=(0, 1))

    region.sim_x_pos = np.squeeze(obj.model.getresult("opt_fields", "x"))
    region.sim_y_pos = np.squeeze(obj.model.getresult("opt_fields", "y"))


def make_adjoint_sim_3d(obj, fdtd):
    # sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", False)
    for i in range(len(obj.adjoint_source_name)):
        phase_radians = np.angle(obj.factor[i] * obj.weight[i])
        adj_phase = np.degrees(phase_radians)
        adj_amp = np.abs(obj.factor[i] * obj.weight[i])
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", True)
        obj.model.set("phase", adj_phase)
        obj.model.set("amplitude", adj_amp)
        obj.model.set('center wavelength', obj.wavelength)
        obj.model.set('wavelength span', 0)

    obj.model.run("FDTD", fdtd.device)

    # get adjoint fields
    adjoint_Ex = obj.model.getresult("opt_fields", "Ex")
    adjoint_Ey = obj.model.getresult("opt_fields", "Ey")
    adjoint_Ez = obj.model.getresult("opt_fields", "Ez")

    E_stack = np.stack([np.squeeze(adjoint_Ex), np.squeeze(adjoint_Ey), np.squeeze(adjoint_Ez)], axis=3)

    obj.E_adj = np.rot90(E_stack, k=1, axes=(0, 1))


def make_forward_sim_2d(obj, fdtd, region):
    # sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.adjoint_source_name)):
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", False)
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", True)
        obj.model.set("phase", obj.forward_source_phase[i])
        obj.model.set("amplitude", np.abs(obj.forward_source_amp[i]))

    obj.model.save(obj.filename)
    obj.model.run("FDTD", fdtd.device)

    # get forward fields
    forward_Ex = obj.model.getresult("opt_fields", "Ex")
    forward_Ey = obj.model.getresult("opt_fields", "Ey")
    forward_Ez = obj.model.getresult("opt_fields", "Ez")

    E_stack = np.stack([np.squeeze(forward_Ex), np.squeeze(forward_Ey), np.squeeze(forward_Ez)], axis=2)

    obj.E_for = np.rot90(E_stack, k=1, axes=(0, 1))

    region.sim_x_pos = np.squeeze(obj.model.getresult("opt_fields", "x"))
    region.sim_y_pos = np.squeeze(obj.model.getresult("opt_fields", "y"))


def make_adjoint_sim_2d(obj, fdtd):
    # sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", False)
    for i in range(len(obj.adjoint_source_name)):
        phase_radians = np.angle(obj.factor[i] * obj.weight[i])
        adj_phase = np.degrees(phase_radians)
        adj_amp = np.abs(obj.factor[i] * obj.weight[i])
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", True)
        obj.model.set("phase", adj_phase)
        obj.model.set("amplitude", adj_amp)
        obj.model.set('center wavelength', obj.wavelength)
        obj.model.set('wavelength span', 0)

    obj.model.run("FDTD", fdtd.device)

    # get adjoint fields
    adjoint_Ex = obj.model.getresult("opt_fields", "Ex")
    adjoint_Ey = obj.model.getresult("opt_fields", "Ey")
    adjoint_Ez = obj.model.getresult("opt_fields", "Ez")

    E_stack = np.stack([np.squeeze(adjoint_Ex), np.squeeze(adjoint_Ey), np.squeeze(adjoint_Ez)], axis=2)

    obj.E_adj = np.rot90(E_stack, k=1, axes=(0, 1))


def make_base_forward_sim_3d(obj, fdtd, region):

    E_for_base_list = []

    rows = len(obj.forward_source_name)
    cols = len(obj.fom_name)
    E_fom_base_list = [[0 for _ in range(cols)] for _ in range(rows)]

    # sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.adjoint_source_name)):
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", False)

    for j in range(len(obj.forward_source_name)):
        obj.model.switchtolayout()
        for k in range(len(obj.forward_source_name)):
            obj.model.select(obj.forward_source_name[k])
            obj.model.set("Enabled", False)

        obj.model.select(obj.forward_source_name[j])
        obj.model.set("Enabled", True)
        obj.model.set("phase", 0)
        obj.model.set("amplitude", 1)

        obj.model.save(obj.filename)
        obj.model.run("FDTD", fdtd.device)

        # get forward fields
        forward_Ex = obj.model.getresult("opt_fields", "Ex")
        forward_Ey = obj.model.getresult("opt_fields", "Ey")
        forward_Ez = obj.model.getresult("opt_fields", "Ez")

        E_stack = np.stack([np.squeeze(forward_Ex), np.squeeze(forward_Ey), np.squeeze(forward_Ez)], axis=3)

        E_for = np.rot90(E_stack, k=1, axes=(0, 1))

        for k in range(len(obj.fom_name)):
            E_dict = obj.model.getresult(obj.fom_name[k], "E")
            E_fom_base_list[j][k] = np.squeeze(E_dict['E'])

        E_for_base_list.append(E_for)

    # 收集所有数据
    region.sim_x_pos = np.squeeze(obj.model.getresult("opt_fields", "x"))
    region.sim_y_pos = np.squeeze(obj.model.getresult("opt_fields", "y"))
    obj.E_for = E_for_base_list
    obj.E_fom = E_fom_base_list


def make_base_adjoint_sim_3d(obj, fdtd):
    E_adj_base_list = []

    # sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", False)

    for j in range(len(obj.adjoint_source_name)):
        obj.model.switchtolayout()
        for k in range(len(obj.adjoint_source_name)):
            obj.model.select(obj.adjoint_source_name[k])
            obj.model.set("Enabled", False)

        obj.model.select(obj.adjoint_source_name[j])
        obj.model.set("Enabled", True)
        obj.model.set("phase", 0)
        obj.model.set("amplitude", 1)

        obj.model.save(obj.filename)
        obj.model.run("FDTD", fdtd.device)

        # get adjoint fields
        adjoint_Ex = obj.model.getresult("opt_fields", "Ex")
        adjoint_Ey = obj.model.getresult("opt_fields", "Ey")
        adjoint_Ez = obj.model.getresult("opt_fields", "Ez")

        E_stack = np.stack([np.squeeze(adjoint_Ex), np.squeeze(adjoint_Ey), np.squeeze(adjoint_Ez)], axis=3)

        E_adj = np.rot90(E_stack, k=1, axes=(0, 1))

        E_adj_base_list.append(E_adj)

    obj.E_adj = E_adj_base_list


def make_forward_sim_2d_analog(obj, fdtd, region):
    # sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.adjoint_source_name)):
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", False)
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", True)
        obj.model.set("phase", obj.forward_source_phase[i])
        obj.model.set("amplitude", np.abs(obj.forward_source_amp[i]))

    obj.model.save(obj.filename)
    obj.model.run("FDTD", fdtd.device)

    # get forward fields
    forward_Ex = obj.model.getresult("opt_fields", "Ex")
    forward_Ey = obj.model.getresult("opt_fields", "Ey")
    forward_Ez = obj.model.getresult("opt_fields", "Ez")

    E_stack = np.stack([np.squeeze(forward_Ex), np.squeeze(forward_Ey), np.squeeze(forward_Ez)], axis=2)

    obj.E_for = np.rot90(E_stack, k=1, axes=(0, 1))


def make_adjoint_sim_2d_analog(obj, fdtd):
    # sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", False)
    for i in range(len(obj.adjoint_source_name)):
        phase_radians = np.angle(obj.factor[i] * obj.weight[i])
        adj_phase = np.degrees(phase_radians)
        adj_amp = np.abs(obj.factor[i] * obj.weight[i])
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", True)
        obj.model.set("phase", adj_phase)
        obj.model.set("amplitude", adj_amp)
        obj.model.set('center wavelength', obj.wavelength)
        obj.model.set('wavelength span', 0)

    obj.model.run("FDTD", fdtd.device)

    # get adjoint fields
    adjoint_Ex = obj.model.getresult("opt_fields", "Ex")
    adjoint_Ey = obj.model.getresult("opt_fields", "Ey")
    adjoint_Ez = obj.model.getresult("opt_fields", "Ez")

    E_stack = np.stack([np.squeeze(adjoint_Ex), np.squeeze(adjoint_Ey), np.squeeze(adjoint_Ez)], axis=2)

    obj.E_adj = np.rot90(E_stack, k=1, axes=(0, 1))
