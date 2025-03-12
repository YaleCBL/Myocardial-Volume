#!/usr/bin/env python3

import pdb
import pandas as pd
import matplotlib.pyplot as plt
import pysvzerod
import copy
import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.optimize import minimize, least_squares, Bounds
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

str_val = "zero_d_element_values"
str_bc = "boundary_conditions"

def read_data(csv_file):
    df = pd.read_csv(csv_file)
    t = df["t abs [s]"].to_numpy()
    pa = df["AoP [mmHg]"].to_numpy()
    pv = df["LVP [mmHg]"].to_numpy()
    v = df["tissue vol ischemic [ml]"].to_numpy()
    flow = df["LAD Flow [mL/s]"].to_numpy()
    return t, pa, pv, v, flow


def create_forward_config(t, pa, intype):
    if intype == "p":
        bc_type = "PRESSURE"
        bc_val = "P"
    elif intype == "q":
        bc_type = "FLOW"
        bc_val = "Q"
    with open("myocardium.json", "r") as f:
        config_estim = json.load(f)
    config_estim[str_bc][0]["bc_type"] = bc_type
    config_estim[str_bc][0]["bc_values"]["t"] = t.tolist()
    config_estim[str_bc][0]["bc_values"][bc_val] = pa.tolist()
    return config_estim


def create_inverse_config(config_estim, res_forward):
    inverse = copy.deepcopy(config_estim)
    inverse["calibration_parameters"] = {
        "tolerance_gradient": 1e-5,
        "tolerance_increment": 1e-10,
        "maximum_iterations": 100,
        "calibrate_stenosis_coefficient": False,
        "set_capacitance_to_zero": False,
    }
    for v in inverse["vessels"]:
        for k in v[str_val].keys():
            v[str_val][k] = 0.0

    names = ["pa:myocardium", "myocardium:gnd"]
    inverse["y"] = {}
    inverse["dy"] = {}
    for n in names:
        for f in ["flow", "pressure"]:
            key = f + ":" + n
            res = res_forward[res_forward["name"] == key]
            inverse["y"][key] = res["y"].tolist()
            inverse["dy"][key] = res["ydot"].tolist()
    return inverse


def get_v_sim(config):
    sim = pysvzerod.simulate(config)
    name = "flow:pa:RC1"
    y = sim[sim["name"] == name]["y"].to_numpy()
    tmax = config[str_bc][0]["bc_values"]["t"][-1]
    nt = config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"]
    return np.cumsum(y) * tmax / nt


def get_p_sim(config):
    sim = pysvzerod.simulate(config)
    name = "pressure:pa:RC1"
    return sim[sim["name"] == name]["y"].to_numpy()


def set_params(config, param_keys, p):
    for i, (id, k) in enumerate(param_keys):
        if id == -1:
            config[str_bc][1]["bc_values"][k] = np.exp(p[i])
        else:
            config["vessels"][id][str_val][k] = np.exp(p[i])
    return config


def print_params(config, param_keys):
    str = ""
    for id, k in param_keys:
        if id == -1:
            str += k + f" {config[str_bc][1]['bc_values'][k]:.1e} "
        else:
            str += k + repr(id) + f" {config['vessels'][id][str_val][k]:.1e} "
    print(str)


def optimize_zero_d(config, param_keys, p_ref, p0):
    def cost_function(p):
        set_params(config, param_keys, p)
        print(np.exp(p))
        p_sim = get_p_sim(config)
        cost = p_ref - p_sim
        # print(p)
        # plt.plot(p_ref, "k-", label="measured")
        # plt.plot(p_sim, "r:", label="simulated")
        # plt.show()
        # print(np.linalg.norm(cost))
        return cost

    res = least_squares(cost_function, np.log(p0), jac="2-point", method="lm")
    set_params(config, param_keys, res.x)
    return config


def main_ideal():
    t, pa, pv, v, flow = read_data("DSEA08_baseline.csv")
    config_estim = create_forward_config(t, pa, "p")
    with open("config_estim.json", "w") as f:
        json.dump(config_estim, f, indent=2)
    res_forward = pysvzerod.simulate(config_estim)

    reference = {"y": {}, "ydot": {}}
    names = ["pa:myocardium"]  # , "myocardium:gnd"]
    for n in names:
        for f in ["flow", "pressure"]:
            key = f + ":" + n
            res = res_forward[res_forward["name"] == key]
            for k in ["y", "ydot"]:
                reference[k][key] = res[k].tolist()

    config_opti = optimize_zero_d(config_estim, reference)
    print(config_opti["vessels"][0][str_val])


def estimate_rc(config, param_keys, p):
    t = config[str_bc][0]["bc_values"]["t"]
    q = config[str_bc][0]["bc_values"]["Q"]
    v = cumulative_trapezoid(q, t, initial=0.0)
    dp = np.gradient(p, t)
    # dv = q_smooth

    # pmax = p_smooth.max()
    # pmin = p_smooth.min()
    # dpmax = dp.max()
    # dpmin = dp.min()
    # dvmax = dv.max()
    # dvmin = dv.min()

    # c1 = -dvmax / dpmin
    # c2 = -dvmin / dpmax
    # r1 = (pmax - pmin) / dvmax
    # r2 = (pmin - pmax) / dvmin

    dp_dv = dp / q
    systole_idx = np.argmax(dp_dv)
    late_systole_idx = np.argmin(np.abs(dp_dv))
    early_diastole_idx = np.argmin(dp_dv)

    pmin = p.min()
    pmax = p.max()

    max_idx = np.argmax(dp)
    r1 = - dp.max() / q[max_idx]

    steady_idx = np.argmin(np.abs(p - pmin))
    r2 = (p[steady_idx] - pmin) / q[steady_idx]
    pdb.set_trace()
    # plt.plot(v, p, "k-")

    param = [pmin, c1, r1, c2, r2]
    plt.plot(v_smooth, p_smooth, "ro")
    plt.show()
    pdb.set_trace()
    set_params(config, param_keys, param)


def smooth(t, q, mode, cutoff=10):
    N = len(q)
    freqs = fftfreq(N, d=t[1] - t[0])
    q_fft = fft(q)
    q_fft[np.abs(freqs) > cutoff] = 0
    q_smooth = np.real(ifft(q_fft))
    if mode == "q":
        return q_smooth - np.trapz(q_smooth, t) / (t[-1] - t[0])
    elif mode == "p":
        return q_smooth


def estimate(fname):
    # read measurements
    t, pa, pv, v, flow = read_data(fname)

    # calculate and smooth flow rate
    q = np.gradient(v, t)
    q_smooth = - smooth(t, q, "q")
    p_smooth = smooth(t, pv, "p")
    v_smooth = cumulative_trapezoid(q_smooth, t, initial=0.0)

    # create 0D model
    config_estim = create_forward_config(t, q_smooth, "q")

    # interoplate to the same time points
    nt = config_estim["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"]
    t_interp = np.linspace(t[0], t[-1], nt)
    v_interp = np.interp(t_interp, t, v)
    p_interp = np.interp(t_interp, t, pv)

    # plt.plot(p_smooth, v_smooth, "k-")
    # plt.xlabel("Pressure [mmHg]")
    # plt.ylabel("Volume [mL]")
    # plt.show()

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.set_xlabel("Time [s]")
    # ax1.set_ylabel("Pressure [mmHg]")
    # ax1.plot(t, p_smooth, "r")
    # ax2.set_ylabel("Flow [mL/s]")
    # ax2.plot(t, q_smooth, "k")
    # plt.show()
    # pdb.set_trace()

    # parameters to be calibrated
    param_keys = [(-1, "Pd"), (0, "C"), (0, "R_poiseuille")]

    # initial guess
    p0 = [1.0e1, 1.0e-4, 1.0e3]
    # estimate parameters
    # p_estim = estimate_rc(config_estim, param_keys, p_smooth)
    # set_params(config_estim, param_keys, np.log(p_estim))
    # print_params(config_estim, param_keys)
    # p_estim = get_p_sim(config_estim)

    # plt.plot(p_interp, v_interp, "k-", label="measured")
    # plt.plot(p_estim, v_interp, "r:", label="estimated")
    # plt.xlabel("Pressure [mmHg]")
    # plt.ylabel("Volume [mL]")
    # plt.legend()
    # plt.show()

    # optimize parameters
    config_opti = optimize_zero_d(config_estim, param_keys, p_interp, p0)
    print_params(config_opti, param_keys)
    p_opti = get_p_sim(config_estim)

    plt.plot(p_interp, v_interp, "k-", label="measured")
    plt.plot(p_opti, v_interp, "y:", label="simulated")
    plt.legend()
    plt.show()
    plt.plot(t_interp, p_interp, "k-", label="measured")
    plt.plot(t_interp, p_opti, "r:", label="simulated")
    plt.legend()
    plt.show()
    # pdb.set_trace()


def main():
    studies = [
        "DSEA08_baseline.csv",
        # "DSEA08_mild_sten.csv",
        # "DSEA08_mild_sten_dob.csv",
        # "DSEA08_mod_sten.csv",
        # "DSEA08_mod_sten_dob.csv",
    ]
    for study in studies:
        estimate(study)


if __name__ == "__main__":
    main()
