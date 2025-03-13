#!/usr/bin/env python3

import pdb
import pandas as pd
import matplotlib.pyplot as plt
import pysvzerod
import copy
import numpy as np
import json
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.optimize import minimize, least_squares, Bounds
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

str_val = "zero_d_element_values"
str_bc = "boundary_conditions"
str_param = "simulation_parameters"


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
    nt = config[str_param]["number_of_time_pts_per_cardiac_cycle"]
    return np.cumsum(y) * tmax / nt


def get_p_sim(config, loc):
    sim = pysvzerod.simulate(config)
    name = "pressure:" + loc
    return sim[sim["name"] == name]["y"].to_numpy()

def get_valve_open(config):
    sim = pysvzerod.simulate(config)
    q = []
    for name in ["pressure:RC1:valve", "pressure:valve:BC"]:
        q += [sim[sim["name"] == name]["y"].to_numpy()]
    return q[1] - q[0] < 0



def set_params(config, p, fun=lambda x: x):
    out = []
    for id, k in p.keys():
        val = fun(p[(id, k)])
        out += [val]
        if "BC" in id:
            for bc in config[str_bc]:
                if bc["bc_name"] == id:
                    bc["bc_values"][k] = val
        else:
            for vs in config["vessels"]:
                if vs["vessel_name"] == id:
                    vs[str_val][k] = val
    return out


def print_params(config, param):
    str = ""
    for id, k in param.keys():
        if "BC" in id:
            for bc in config[str_bc]:
                if bc["bc_name"] == id:
                    val = bc["bc_values"][k]
        else:
            for vs in config["vessels"]:
                if vs["vessel_name"] == id:
                    val = vs[str_val][k]
        str += id + " " + k + f" {val:.1e} "
    print(str)


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


def plot_data(study, t_interp, p_smooth, v_smooth, q_smooth):
    plt.plot(p_smooth, v_smooth, "ko")
    plt.xlabel("Pressure [mmHg]")
    plt.ylabel("Volume [mL]")
    plt.savefig(study + "_pv.pdf")
    plt.close()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Time [s]")
    ax1.plot(t_interp, p_smooth, "r")
    ax1.set_ylabel("Pressure [mmHg]")
    ax2.plot(t_interp, v_smooth, "k")
    ax2.set_ylabel("Volume [mL]")
    plt.savefig(study + "_p_and_v.pdf")
    plt.close()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Time [s]")
    ax1.plot(t_interp, p_smooth, "r")
    ax1.set_ylabel("Pressure [mmHg]")
    ax2.plot(t_interp, q_smooth, "k")
    ax2.set_ylabel("Flow [mL/s]")
    plt.savefig(study + "_p_and_q.pdf")
    plt.close()


def optimize_zero_d(config, p0, t_interp, q_smooth, p_smooth, v_smooth):
    def cost_function(p):
        p = {k: p[i] for i, k in enumerate(p0.keys())}
        p_set = set_params(config, p, np.exp)
        # print(np.exp(p))
        # print(p_set)
        p_sim = get_p_sim(config, "pa:RC1")
        # t_fit = np.array(config[str_bc][0]["bc_values"]["t"]) < 0.3
        cost = p_smooth - p_sim
        # plt.plot(p_ref, "k-", label="measured")
        # plt.plot(p_sim, "r:", label="simulated")
        # plt.show()
        # print(np.linalg.norm(cost))
        return cost

    initial = set_params(config, p0, np.log)
    res = least_squares(cost_function, initial, jac="2-point", method="lm")
    set_params(config, {k: res.x[i] for i, k in enumerate(p0.keys())}, np.exp)
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


def estimate_rc(config, p0, ts, qs, ps, vs):
    set_params(config, p0)
    with open("forward.json", "w") as f:
        json.dump(config, f, indent=2)
    return config


def estimate(fname):
    # read measurements
    t, pa, pv, v, flow = read_data(fname)

    # simulation parameters
    nt = 201

    # interoplate to the same time points
    t_interp = np.linspace(t[0], t[-1], nt)
    v_interp = np.interp(t_interp, t, v)
    p_interp = np.interp(t_interp, t, pv)
    q_interp = np.gradient(v_interp, t_interp)

    # calculate and smooth flow rate
    q_smooth = -smooth(t_interp, q_interp, "q")
    p_smooth = smooth(t_interp, p_interp, "p")
    v_smooth = cumulative_trapezoid(q_smooth, t_interp, initial=0.0)
    plot_data(fname, t_interp, p_smooth, v_smooth, q_smooth)

    # create 0D model
    config = create_forward_config(t_interp, q_smooth, "q")
    config[str_param]["number_of_time_pts_per_cardiac_cycle"] = nt

    # initial guess
    p0 = OrderedDict()
    p0[("BC", "Pd")] = p_smooth[0]
    # p0[("RC1", "C")] = 1.0e-6
    p0[("RC1", "R_poiseuille")] = 1.0e1

    # p0[("BC_right", "Pd")] = 1.0e1
    # p0[("BC_left", "Pd")] = 1.0e1
    # p0[("BC_right", "Pd")] = 1.0e1
    # p0[("RCLS_left", "C")] = 2.0e-4
    # p0[("RCLS_right", "C")] = 2.0e-4
    # p0[("RCLS_left", "R_poiseuille")] = 1.0e0
    # p0[("RCLS_right", "R_poiseuille")] = 1.0e0
    for name, fun in zip(["estimated", "optimized"], [estimate_rc, optimize_zero_d]):
        config = fun(config, p0, t_interp, q_smooth, p_smooth, v_smooth)
        print_params(config, p0)
        p_estim = get_p_sim(config, "pa:RC1")
        p_out = get_p_sim(config, "RC1:valve")
        valve = get_valve_open(config)

        fig, axs = plt.subplots(4, 1, figsize=(12, 8))
        axs[0].plot(p_smooth, v_smooth, "k-", label="smooth")
        axs[0].plot(p_estim, v_smooth, "r:", label="simulated")
        axs[0].legend()
        axs[0].set_xlabel("Pressure")
        axs[0].set_ylabel("Volume")
        axs[1].plot(t_interp, p_smooth, "k-", label="smooth")
        axs[1].plot(t_interp, p_estim, "r:", label="simulated")
        axs[1].legend()
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Pressure")
        axs[2].plot(t_interp, p_out, "r:", label="outlet")
        axs[2].legend()
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Pressure")
        axs[3].plot(t_interp, valve, "k-")
        axs[3].set_yticks([0, 1])
        axs[3].set_yticklabels(['closed', 'open'])
        axs[3].set_xlabel("Time")
        plt.tight_layout()
        plt.show()
        # pdb.set_trace()

        with open("config_" + name + ".json", "w") as f:
            json.dump(config, f, indent=2)


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
