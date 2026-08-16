import numpy as np
import scipy.constants as const
from scipy.constants import epsilon_0

from src import optimization as opt


def get_source_power(obj, wavelengths):
    frequency = const.c / wavelengths
    source_power = obj.model.sourcepower(frequency)
    return np.asarray(source_power).flatten()


def get_fom_ce(obj, V_cell):

    T_fwd = np.zeros(len(obj.fom_name), dtype=np.complex128)
    temp_factor = np.zeros(len(obj.fom_name), dtype=np.complex128)
    for i in range(len(obj.fom_name)):
        mode_exp_result_name = 'expansion for ' + obj.mode_exp_monitor_name[i]
        mode_exp_data_set = obj.model.getresult(obj.mode_exp_monitor_name[i], mode_exp_result_name)
        wavelengths = mode_exp_data_set['lambda'].flatten()
        trans_coeff = (mode_exp_data_set['a'] * np.sqrt(mode_exp_data_set['N'].real)).flatten()

        omega = 2.0 * np.pi * const.c / wavelengths
        source_power = get_source_power(obj, wavelengths)
        phase_prefactors = trans_coeff / 4.0 / source_power
        T_fwd[i] = np.real(trans_coeff * trans_coeff.conj() / source_power)
        temp_factor[i] = np.conj(phase_prefactors) * omega * 1j / np.sqrt(source_power)
    T_sum = np.sum(T_fwd)

    for i in range(len(obj.fom_name)):
        obj.fom[i] = -obj.target_fom[i] * np.real(np.log((T_fwd[i]) / T_sum))
        obj.factor[i] = (obj.target_fom[i] * temp_factor[i] * T_sum / T_fwd[i]) * V_cell * epsilon_0


def get_fom_amp(obj):

    T_fwd = np.zeros(len(obj.fom_name), dtype=np.complex128)
    for i in range(len(obj.fom_name)):
        mode_exp_result_name = 'expansion for ' + obj.mode_exp_monitor_name[i]
        mode_exp_data_set = obj.model.getresult(obj.mode_exp_monitor_name[i], mode_exp_result_name)
        wavelengths = mode_exp_data_set['lambda'].flatten()
        trans_coeff = (mode_exp_data_set['a'] * np.sqrt(mode_exp_data_set['N'].real)).flatten()

        source_power = get_source_power(obj, wavelengths)
        T_fwd[i] = trans_coeff / np.sqrt(source_power)
        obj.fom[i] = np.abs(obj.target_fom[i] - T_fwd[i]) ** 2
        obj.factor[i] = -2 * np.conj(T_fwd[i] - obj.target_fom[i]) * 1j


def get_fom_intensity(obj, V_cell):

    for i in range(len(obj.fom_name)):
        mode_exp_result_name = 'expansion for ' + obj.mode_exp_monitor_name[i]
        mode_exp_data_set = obj.model.getresult(obj.mode_exp_monitor_name[i], mode_exp_result_name)
        wavelengths = mode_exp_data_set['lambda'].flatten()
        trans_coeff = (mode_exp_data_set['a'] * np.sqrt(mode_exp_data_set['N'].real)).flatten()

        omega = 2.0 * np.pi * const.c / wavelengths
        source_power = get_source_power(obj, wavelengths)
        phase_prefactors = trans_coeff / 4.0 / source_power
        T_fwd_vs_wavelength = np.real(trans_coeff * trans_coeff.conj() / source_power)
        obj.fom[i] = np.abs(obj.target_fom[i] - T_fwd_vs_wavelength.flatten()) ** 2
        obj.factor[i] = (2 * (obj.target_fom[i] - T_fwd_vs_wavelength.flatten()) * np.conj(
            phase_prefactors) * omega * 1j / np.sqrt(source_power)) * V_cell * epsilon_0


def get_fom_ce_gradient_parallel(obj, data, region):

    obj.E_fom_desire_mode = np.squeeze(obj.model.getresult(f"{obj.mode_exp_monitor_name[0]}", "mode profiles")["E1"])
    T_fwd = np.zeros(len(obj.target_fom), dtype=np.complex64)
    temp_factor = np.zeros_like(T_fwd)
    grad_eps_sim_all = np.zeros((len(region.sim_y_pos), len(region.sim_x_pos)))
    fom_all = np.zeros(len(obj.target_fom))

    for i, key in enumerate(data.files):

        for_amp = np.exp(1j * 2 * np.pi * data[key][:len(obj.forward_source_name)])
        target_fom = data[key][-len(obj.target_fom):]

        out_arr = np.tensordot(for_amp, np.asarray(obj.E_fom), axes=(0, 0))
        E_sim_list = [out_arr[j] for j in range(len(obj.target_fom))]

        for j in range(len(obj.target_fom)):
            input_mode = E_sim_list[j]
            overlap_mode = np.sum(np.conj(obj.E_fom_desire_mode) * input_mode)
            trans_coeff = overlap_mode / np.sum(np.conj(obj.E_fom_desire_mode) * obj.E_fom_desire_mode)
            T_fwd[j] = trans_coeff * np.conj(trans_coeff)
            temp_factor[j] = np.conj(trans_coeff) * 1j

        T_sum = np.sum(T_fwd)

        fom = -target_fom * np.real(np.log(T_fwd / T_sum))
        factor = target_fom * temp_factor * T_sum / T_fwd

        E_for_sim = np.tensordot(for_amp, np.asarray(obj.E_for), axes=(0, 0))
        E_adj_sim = np.tensordot(factor, np.asarray(obj.E_adj), axes=(0, 0))

        grad_eps_sim_all += np.mean(-2 * np.real(np.sum(E_for_sim * E_adj_sim, axis=3)), axis=2)

        fom_all += fom

    grad_eps_sim_all_mean = grad_eps_sim_all / len(data.files)

    grad_eps = opt.interp_simtodesign_2d(grad_eps_sim_all_mean, region, "linear")

    obj.fom = fom_all / len(data.files)

    return grad_eps
