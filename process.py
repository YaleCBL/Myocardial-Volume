#!/usr/bin/env python3

import os
import pdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
import pysvzerod
import copy
import numpy as np
import json
from collections import OrderedDict
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.optimize import minimize, least_squares, Bounds
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

str_val = "zero_d_element_values"
str_bc = "boundary_conditions"
str_param = "simulation_parameters"
str_time = "number_of_time_pts_per_cardiac_cycle"

mmHg_to_Ba = 133.322
Ba_to_mmHg = 1 / mmHg_to_Ba

def read_data(animal, study):
    csv_file = os.path.join("data", animal + "_" + study + ".csv")
    df = pd.read_csv(csv_file)
    t = df["t abs [s]"].to_numpy()

    # convert pressure to cgs units
    for field in df.keys():
        if "mmHg" in field:
            df[field.replace("mmHg", "Ba")] = df[field] * mmHg_to_Ba

    pat = df["AoP [Ba]"].to_numpy()
    pven = df["LVP [Ba]"].to_numpy()
    vmyo = df["tissue vol ischemic [ml]"].to_numpy()
    qlad = df["LAD Flow [mL/s]"].to_numpy()

    csv_file = os.path.join("data", animal + "_microsphere.csv")
    df = pd.read_csv(csv_file)
    for k in df.keys():
        if study in k:
            dvcycle = df[k][0]
            break
    return t, pat, pven, vmyo, qlad, dvcycle


def read_lit_data(fname):
    lit_path = os.path.join("data", fname)
    lit_data = read_config(lit_path)
    for k, v in lit_data.items():
        for kk, vv in v.items():
            if kk[0] == "R":
                for kkk in vv.keys():
                    lit_data[k][kk][kkk] *= 1e3
            if kk[0] == "C":
                for kkk in vv.keys():
                    lit_data[k][kk][kkk] *= 1e-6
    return lit_data


def read_config(fname):
    with open(fname, "r") as f:
        return json.load(f)

def get_sim(config, loc):
    sim = pysvzerod.simulate(config)
    res = sim[sim["name"] == loc]["y"].to_numpy()
    if not res.size:
        raise ValueError(f"Result {loc} not found. Options are:\n" + ", ".join(np.unique(sim["name"]).tolist()))
    return res

def get_sim_p(config):
    return get_sim(config, "pressure:BC_AT:RC")


def get_sim_out(config):
    q_sim_in = get_sim(config, "flow:BC_AT:RC")
    q_sim_out = get_sim(config, "flow:RC:BC_LV")
    nt = config[str_param][str_time]
    tmax = config['boundary_conditions'][0]['bc_values']['t'][-1]
    ti = np.linspace(0.0, tmax, nt)
    q_diff = q_sim_out - q_sim_in
    # vol = cumulative_trapezoid(q_diff, ti, initial=0)
    vol = cumulative_trapezoid(q_sim_in, ti, initial=0)
    return vol


def get_valve_open(config):
    sim = pysvzerod.simulate(config)
    q = []
    for name in ["pressure:RC1:valve", "pressure:valve:BC"]:
        q += [sim[sim["name"] == name]["y"].to_numpy()]
    return q[1] - q[0] < 0


def set_params(config, p, x=None):
    out = []
    for i, (id, k) in enumerate(p.keys()):
        pval, pmin, pmax = p[(id, k)]
        if x is not None:
            xval = x[i]
            if xval > 100:
                pval = pmax
            elif xval < -100:
                pval = pmin
            else:
                pval = pmin + (pmax - pmin) * 1 / (1 / np.exp(xval) + 1)
        out += [pval]
        if "BC" in id:
            for bc in config[str_bc]:
                if bc["bc_name"] == id:
                    if k == "P" and isinstance(pval, float):
                        bc["bc_values"][k] = [pval, pval]
                        bc["bc_values"]["t"] = [0, 0.737]
                    else:
                        bc["bc_values"][k] = pval
        else:
            for vs in config["vessels"]:
                if vs["vessel_name"] == id:
                    vs[str_val][k] = pval
    return out


def get_params(p):
    out = []
    for k in p.keys():
        pval, pmin, pmax = p[k]
        out += [np.log((pval - pmin) / (pmax - pmin))]
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
        # convert units
        if k == "R":
            val *= Ba_to_mmHg
            unit = "mmHg*s/ml"
        elif k == "C":
            val *= 1 / Ba_to_mmHg
            unit = "ml/mmHg"
        else:
            unit = "cgs"
        str += id + " " + k[0] + f" {val:.1e} " + unit + "\n"
    print(str)


def smooth(t, ti, qt, cutoff=10):
    N = len(qt)
    freqs = fftfreq(N, d=t[1] - t[0])
    q_fft = fft(qt)
    q_fft[np.abs(freqs) > cutoff] = 0
    qs = np.real(ifft(q_fft))
    dqs_fft = q_fft * (2j * np.pi * freqs)
    dqs = np.real(ifft(dqs_fft))
    return np.interp(ti, t, qs), np.interp(ti, t, dqs)


def plot_data(study, data):
    _, ax = plt.subplots(2, 2, figsize=(15, 9))
    ax = ax.flatten()
    axt = [ax[i].twinx() for i in range(len(ax))]
    mk = {"original": ".", "smoothed": "-"}

    ax[0].set_xlabel("AT Pressure [mmHg]")
    ax[0].set_ylabel("Volume [mL]")
    ax[1].set_ylabel("Delta pressure [mmHg]")

    for j, loc in enumerate(["myo", "lad"]):
        ax[j + 2].set_ylabel(f"{loc} volume [mL]", color="b")
        axt[j + 2].set_ylabel(f"{loc} flow [mL/s]", color="m")

    for i in range(1, 4):
        ax[i].set_xlabel("Time [s]")

    for k in ["original", "smoothed"]:
        ax[0].plot(data[k]["pat"]*Ba_to_mmHg, data[k]["vmyo"], f"k{mk[k]}")
        ax[1].plot(data[k]["t"], (data[k]["pat"] - data[k]["pven"])*Ba_to_mmHg, f"r{mk[k]}")

        for j, loc in enumerate(["myo", "lad"]):
            ax[j + 2].plot(
                data[k]["t"],
                data[k]["v" + loc],
                f"b{mk[k]}"
            )
            axt[j + 2].plot(
                data[k]["t"],
                data[k]["q" + loc],
                f"m{mk[k]}"
            )

    plt.tight_layout()
    plt.savefig(study + "_data.pdf")
    plt.close()


def plot_results(animal, config, data):
    _, axs = plt.subplots(3, len(config), figsize=(16, 9), sharex="col", sharey="row")
    if len(config) == 1:
        axs = axs.reshape(-1,1)
    for j, study in enumerate(config.keys()):
        with open(f'{animal}_{study}.json', "w") as f:
            json.dump(config[study], f, indent=2)

        dats = OrderedDict()
        dats["pat"] = get_sim(config[study], "pressure:BC_AT:RC")
        dats["pven"] = get_sim(config[study], "pressure:RC:BC_LV")
        dats["vmyo"] = get_sim_out(config[study])
        labels = ["AT Pressure [mmHg]", "LV Pressure [mmHg]", "Myocardial Volume [ml]"]

        ti = config[study]["boundary_conditions"][0]["bc_values"]["t"]
        datm = data[study]["smoothed"]

        convert = {"p": Ba_to_mmHg, "v": 1.0}

        for i, k in enumerate(dats.keys()):
            axs[i, j].plot(ti, dats[k] * convert[k[0]], "r-", label="simulated")
            axs[i, j].plot(ti, datm[k] * convert[k[0]], "k--", label="measured (smoothed)")
            axs[i, j].set_xlim(ti[0], ti[-1])
            axs[i, j].grid(True)
            if i == 0:
                axs[i, j].set_title(study)
            if i == len(dats) - 1:
                axs[i, j].legend()
                axs[i, j].set_xlabel("Time [s]")
            if j == 0:
                axs[i, j].set_ylabel(labels[i])

    plt.tight_layout()
    plt.savefig(f'{animal}_simulated.pdf')
    plt.close()

def plot_all_optimized(animal, studies, optimized):
    _, axes = plt.subplots(2, 1, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param_val in enumerate(optimized[studies[0]].keys()):
        if param_val.endswith("R"):
            values = [optimized[s][param_val] * Ba_to_mmHg for s in studies]
            ylabel = 'Resistance [mmHg*s/ml]'
        elif param_val.endswith("C"):
            values = [optimized[s][param_val] / Ba_to_mmHg for s in studies]
            ylabel = 'Capacitance [ml/mmHg]'
        elif param_val.endswith("L"):
            values = [optimized[s][param_val] for s in studies]
            ylabel = 'Inductance [cgs]'
            
        axes[i].bar(range(len(studies)), values)
        axes[i].set_xticks(range(len(studies)))
        axes[i].set_xticklabels(studies, rotation=45)
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(f'{animal} {param_val}')

    plt.tight_layout()
    plt.savefig(f'{animal}_parameters.pdf')
    plt.close()


def optimize_zero_d(config, p0, qref, verbose=0):
    def cost_function(p):
        pset = set_params(config, p0, p)
        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")
        tmin = qref.argmin()
        qmin = qref[tmin] - get_sim_out(config)[tmin]
        qend = qref[-1] - get_sim_out(config)[-1]
        obj = [qmin, qend]
        obj = qref - get_sim_out(config)
        if verbose:
            print(f"{np.linalg.norm(obj):.1e}", end="\n")
        return obj

    initial = get_params(p0)
    if verbose:
        for k in p0.keys():
            print(f"{k[0][:5]} {k[1][0]}", end="\t")
        print("obj", end="\n")
    res = least_squares(cost_function, initial, jac="2-point")#, method="lm", diff_step=1e-3
    set_params(config, p0, res.x)
    return config


def estimate(animal, study, verb=0):
    # read measurements
    name = animal + "_" + study
    t, pat, pven, vmyo, qlad, dvcycle = read_data(animal, study)
    qmyo = np.gradient(vmyo, t)
    vlad = cumulative_trapezoid(qlad, t, initial=0)

    # scale the LAD inflow according to microsphere measurements
    dvlad = vlad.max() - vlad.min()
    dvmyo = vmyo.max() - vmyo.min()
    qlad *= dvcycle / dvlad
    vlad *= dvcycle / dvlad

    # interoplate to simulation time points and smooth flow rate
    nt = 201
    ti = np.linspace(t[0], t[-1], nt)
    pats, _ = smooth(t, ti, pat, cutoff=15)
    pvens, _ = smooth(t, ti, pven, cutoff=15)
    qlads, _ = smooth(t, ti, qlad, cutoff=15)
    vmyos, qmyos = smooth(t, ti, vmyo, cutoff=15)
    vmyos -= vmyos[0]
    vlads = cumulative_trapezoid(qlads, ti, initial=0)
    tv = [0.0, ti[-1]]
    pv = [0.0, 0.0]

    # create dictionary with all data
    data = {}
    data["original"] = {
        "t": t,
        "pat": pat,
        "pven": pven,
        "vmyo": vmyo,
        "qmyo": qmyo,
        "qlad": qlad,
        "vlad": vlad,
    }
    data["smoothed"] = {
        "t": ti,
        "pat": pats,
        "pven": pvens,
        "vmyo": vmyos,
        "qmyo": qmyos,
        "qlad": qlads,
        "vlad": vlads,
    }
    # plot_data(name, data)

    # read literature data
    lit_path = os.path.join("data", "kim10b_table3.json")
    lit_data = read_config(lit_path)

    # create 0D model
    config = read_config("WK1.json")
    config[str_param][str_time] = nt

    # set boundary conditions
    pini = {}
    pini[("BC_AT", "t")] = (ti.tolist(), None, None)
    pini[("BC_AT", "P")] = (pats.tolist(), None, None)
    pini[("BC_LV", "t")] = (ti.tolist(), None, None)
    pini[("BC_LV", "P")] = (pvens.tolist(), None, None)
    set_params(config, pini)

    # set initial values
    bounds = {"R": (1e3, 1e12), "C": (1e-12, 1e-3), "L": (1e-12, 1e12)}
    p0 = OrderedDict()
    p0[("RC", "C")] = (1e-5, *bounds["C"])
    p0[("RC", "R")] = (1e+4, *bounds["R"])
    # p0[("RL", "L")] = (1e-9, *bounds["L"])
    # p0[("RL", "R")] = (1e+5, *bounds["R"])
    set_params(config, p0)

    config_opt = optimize_zero_d(config, p0, vmyos, verbose=verb)
    print(name)
    print_params(config_opt, p0)

    return data, config_opt, p0


def main():
    animal = "DSEA08"
    studies = [
        "baseline",
        "mild_sten",
        "mild_sten_dob",
        "mod_sten", 
        "mod_sten_dob",
    ]
    optimized = defaultdict(dict)
    data = {}
    config = {}

    _, axes = plt.subplots(3, len(studies), figsize=(15, 10))
    for study in studies:
        data[study], config[study], p0 = estimate(animal, study, verb=1)
        params = [opt[0] for opt in p0]
        values = [opt[1] for opt in p0]
        for param in params:
            for val in values:
                for vessel in config[study]["vessels"]:
                    if vessel["vessel_name"] == param:
                        if val in vessel[str_val]:
                            optimized[study][f"{param}_{val}"] = vessel[str_val][val]
    
    plot_results(animal, config, data)
    plot_all_optimized(animal, studies, optimized)

if __name__ == "__main__":
    main()
