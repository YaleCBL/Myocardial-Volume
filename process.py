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
        forward = json.load(f)
    forward["boundary_conditions"][0]["bc_type"] = bc_type
    forward["boundary_conditions"][0]["bc_values"]["t"] = t.tolist()
    forward["boundary_conditions"][0]["bc_values"][bc_val] = pa.tolist()
    return forward


def create_inverse_config(forward, res_forward):
    inverse = copy.deepcopy(forward)
    inverse["calibration_parameters"] = {
        "tolerance_gradient": 1e-5,
        "tolerance_increment": 1e-10,
        "maximum_iterations": 100,
        "calibrate_stenosis_coefficient": False,
        "set_capacitance_to_zero": False,
    }
    for v in inverse["vessels"]:
        for k in v["zero_d_element_values"].keys():
            v["zero_d_element_values"][k] = 0.0

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


def get_vol(config):
    sim = pysvzerod.simulate(config)
    name = "flow:pa:RC1"
    y = sim[sim["name"] == name]["y"].to_numpy()
    tmax = config["boundary_conditions"][0]["bc_values"]["t"][-1]
    nt = config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"]
    return np.cumsum(y) * tmax / nt

def set_params(config, param_keys, p):
    for i, (id, k) in enumerate(param_keys):
        config["vessels"][id]["zero_d_element_values"][k] = np.exp(p[i])
    return config

def optimize_zero_d(config, vol_ref):
    param_keys = [(0, "C"), (0, "R_poiseuille"), (1, "C"), (1, "R_poiseuille")]

    def cost_function(p):
        set_params(config, param_keys, p)
        print(np.exp(p))
        cost = vol_ref - vol_ref[0] - get_vol(config)
        print(np.linalg.norm(cost))
        return cost

    p0 = np.log(0.1*np.ones(len(param_keys)))
    # pdb.set_trace()
    res = least_squares(cost_function, p0, jac="2-point", method="lm")
    set_params(config, param_keys, res.x)
    return config


def main_ideal():
    t, pa, pv, v, flow = read_data("DSEA08_baseline.csv")
    forward = create_forward_config(t, pa, "p")
    with open("forward.json", "w") as f:
        json.dump(forward, f, indent=2)
    res_forward = pysvzerod.simulate(forward)

    reference = {"y": {}, "ydot": {}}
    names = ["pa:myocardium"]  # , "myocardium:gnd"]
    for n in names:
        for f in ["flow", "pressure"]:
            key = f + ":" + n
            res = res_forward[res_forward["name"] == key]
            for k in ["y", "ydot"]:
                reference[k][key] = res[k].tolist()

    optimized_config = optimize_zero_d(forward, reference)
    print(optimized_config["vessels"][0]["zero_d_element_values"])


def estimate_rc(t, p, v):
    p_smooth = savgol_filter(p, window_length=11, polyorder=2)
    v_smooth = savgol_filter(v, window_length=11, polyorder=2)

    dp = np.gradient(p_smooth, t)
    dv = np.gradient(v_smooth, t)

    pmax = p_smooth.max()
    pmin = p_smooth.min()
    dpmax = dp.max()
    dpmin = dp.min()
    dvmax = dv.max()
    dvmin = dv.min()

    c1 = - dvmax / dpmin
    c2 = - dvmin / dpmax
    r1 = (pmax - pmin) / dvmax
    r2 = (pmin - pmax) / dvmin
    # plt.plot(v, p, "k-")
    # plt.plot(v_smooth, p_smooth, "ro")
    # plt.show()
    return [c1, r1, c2, r2]

def smooth_inflow(t, q, cutoff=10):
    N = len(q)
    freqs = fftfreq(N, d=t[1] - t[0])
    q_fft = fft(q)
    q_fft[np.abs(freqs) > cutoff] = 0
    q_smooth = np.real(ifft(q_fft))
    q_zero_mean = q_smooth - np.trapz(q_smooth, t) / (t[-1] - t[0])
    return q_zero_mean

def estimate(fname):
    # read measurements
    t, pa, pv, v, flow = read_data(fname)

    # unit conversion
    mmHg_to_CGS = 1.33322e3
    pv *= mmHg_to_CGS
    pa *= mmHg_to_CGS

    # calculate and smooth flow rate
    q = np.gradient(v, t)
    q_smooth = smooth_inflow(t, q)

    # create 0D model
    forward = create_forward_config(t, q_smooth, "q")

    # interoplate to the same time points
    nt = forward["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"]
    t_sim = np.linspace(t[0], t[-1], nt)
    p_sim = np.interp(t_sim, t, pv)
    v_sim = np.interp(t_sim, t, v)

    p_estim = estimate_rc(t, pv, v)
    param_keys = [(0, "C"), (0, "R_poiseuille"), (1, "C"), (1, "R_poiseuille")]
    set_params(forward, param_keys, np.log(p_estim))
    v_sim = get_vol(forward)
    plt.plot(v, pv, "k-")
    plt.plot(v_sim + v[0], p_sim, "r:")
    plt.show()
    pdb.set_trace()

    optimized_config = optimize_zero_d(forward, v_sim)
    print(optimized_config["vessels"][0]["zero_d_element_values"])

    v_sim = get_vol(optimized_config)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax1.plot(pv, v, "b", label="measured")
    # ax1.plot(p_sim, v_sim, "y", label="simulated")
    ax1.plot(t, pv, "r", label="pv")
    ax2.plot(t_sim, v_sim, "r", label="pv")
    ax1.legend()
    # ax2.legend()
    plt.show()
    pdb.set_trace()


def main():
    studies = [
        "DSEA08_baseline.csv",
        "DSEA08_mild_sten.csv",
        "DSEA08_mild_sten_dob.csv",
        "DSEA08_mod_sten.csv",
        "DSEA08_mod_sten_dob.csv",
    ]
    for study in studies:
        estimate(study)


if __name__ == "__main__":
    main()
