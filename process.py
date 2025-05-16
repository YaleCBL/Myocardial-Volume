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

n_in = "BC_AT:BV"
n_out = "BV:BC_COR"
model = "coronary"

def read_data(animal, study):
    data = {}
    csv_file = os.path.join("data", animal + "_" + study + ".csv")
    df = pd.read_csv(csv_file)
    data["t"] = df["t abs [s]"].to_numpy()

    # convert pressure to cgs units
    for field in df.keys():
        if "mmHg" in field:
            df[field.replace("mmHg", "Ba")] = df[field] * mmHg_to_Ba

    data["pat"] = df["AoP [Ba]"].to_numpy()
    data["pven"] = df["LVP [Ba]"].to_numpy()
    data["vmyo"] = df["tissue vol ischemic [ml]"].to_numpy()
    data["qlad"] = df["LAD Flow [mL/s]"].to_numpy()

    data["qmyo"] = np.gradient(data["vmyo"], data["t"])
    data["vlad"] = cumulative_trapezoid(data["qlad"], data["t"], initial=0)

    csv_file = os.path.join("data", animal + "_microsphere.csv")
    df = pd.read_csv(csv_file)
    for k in df.keys():
        if study in k:
            data["dvcycle"] = df[k][0]
            break
    return data


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
    return get_sim(config, "pressure:" + n_in)


def get_sim_out(config):
    q_sim_in = get_sim(config, "flow:" + n_in)
    q_sim_out = get_sim(config, "flow:" + n_out)
    nt = config[str_param][str_time]
    tmax = config['boundary_conditions'][0]['bc_values']['t'][-1]
    ti = np.linspace(0.0, tmax, nt)
    q_diff = q_sim_out - q_sim_in
    # vol = cumulative_trapezoid(q_diff, ti, initial=0)
    vol = cumulative_trapezoid(q_sim_in, ti, initial=0)
    return vol


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


def smooth_data(data_o, nt):
    data_s = {}
    data_s["t"] = np.linspace(data_o["t"][0], data_o["t"][-1], nt)
    for field in ["pat", "pven", "qlad"]:
        data_s[field], _ = smooth(data_o["t"], data_s["t"], data_o[field], cutoff=15)
    data_s["vmyo"], data_s["qmyo"] = smooth(data_o["t"], data_s["t"], data_o["vmyo"], cutoff=15)
    data_s["vmyo"] -= data_s["vmyo"][0]
    data_s["vlad"] = cumulative_trapezoid(data_s["qlad"], data_s["t"], initial=0)
    return data_s


def plot_data(animal, data):
    # pdb.set_trace()
    n_param = len(data.keys())
    _, ax = plt.subplots(4, n_param, figsize=(n_param*5, 10), sharex="col", sharey="row")
    # if n_param == 1:
    #     ax = [ax]
    # else:
    #     ax = ax.flatten()
    axt = np.array([[ax[i,j].twinx() for j in range(ax.shape[1])] for i in range(ax.shape[0])])
    mk = {"o": ".", "s": "-"}

    for i, s in enumerate(data.keys()):
        ax[0, i].set_title(s)
        ax[0, i].set_xlabel("AT Pressure [mmHg]")
        ax[0, i].set_ylabel("Volume [mL]")
        ax[1, i].set_ylabel("Delta pressure [mmHg]")
        for j, loc in enumerate(["myo", "lad"]):
            ax[j + 2, i].set_ylabel(f"{loc} volume [mL]", color="b")
            axt[j + 2, i].set_ylabel(f"{loc} flow [mL/s]", color="m")

        for j in range(1, 4):
            ax[j, i].set_xlabel("Time [s]")

        for k in ["o", "s"]:
            vol = data[s][k]["vmyo"]
            vol -= vol[0]
            # ax[0, i].plot(data[s][k]["pat"]*Ba_to_mmHg, vol, f"k{mk[k]}")
            ax[1, i].plot(data[s][k]["t"], (data[s][k]["pat"] - data[s][k]["pven"])*Ba_to_mmHg, f"r{mk[k]}")

            for j, loc in enumerate(["myo", "lad"]):
                ax[j + 2, i].plot(
                    data[s][k]["t"],
                    data[s][k]["v" + loc],
                    f"b{mk[k]}"
                )
                axt[j + 2, i].plot(
                    data[s][k]["t"],
                    data[s][k]["q" + loc],
                    f"m{mk[k]}"
                )

    plt.tight_layout()
    plt.savefig(f"{animal}_data.pdf")
    plt.close()


def plot_results(animal, config, data):
    _, axs = plt.subplots(3, len(config), figsize=(16, 9), sharex="col", sharey="row")
    if len(config) == 1:
        axs = axs.reshape(-1,1)
    for j, study in enumerate(config.keys()):
        with open(f'{animal}_{study}.json', "w") as f:
            json.dump(config[study], f, indent=2)

        dats = OrderedDict()
        dats["pat"] = get_sim(config[study], "pressure:" + n_in)
        dats["pven"] = get_sim(config[study], "pressure:" + n_out)
        vol = get_sim(config[study], "volume_im:BC_COR")
        vol -= vol[0]
        dats["vmyo"] = vol
        labels = ["AT Pressure [mmHg]", "LV Pressure [mmHg]", "Myocardial Volume [ml]"]

        ti = config[study]["boundary_conditions"][0]["bc_values"]["t"]
        datm = data[study]["s"]

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
    plt.savefig(f'{animal}_{model}_simulated.pdf')
    plt.close()

def plot_optimized(animal, studies, optimized):
    n_param = len(optimized[studies[0]].keys())
    _, axes = plt.subplots(1, n_param, figsize=(n_param*5, 10))
    if n_param == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, param_val in enumerate(optimized[studies[0]].keys()):
        if param_val[0] == "R":
            values = [optimized[s][param_val] * Ba_to_mmHg for s in studies]
            ylabel = 'Resistance [mmHg*s/ml]'
        elif param_val[0] == "C":
            values = [optimized[s][param_val] / Ba_to_mmHg for s in studies]
            ylabel = 'Capacitance [ml/mmHg]'
        elif param_val[0] == "L":
            values = [optimized[s][param_val] for s in studies]
            ylabel = 'Inductance [cgs]'
            
        axes[i].bar(range(len(studies)), values)
        axes[i].set_xticks(range(len(studies)))
        axes[i].set_xticklabels(studies, rotation=45)
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(f'{animal} {param_val}')

    plt.tight_layout()
    plt.savefig(f'{animal}_{model}_parameters.pdf')
    plt.close()


def optimize_zero_d(config, p0, data, verbose=0):
    pref = data["s"]["pat"]
    vref = data["s"]["vmyo"]
    def cost_function(p):
        pset = set_params(config, p0, p)
        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")
        vol = get_sim(config, "volume_im:BC_COR")
        vol -= vol[0]
        obj = pref - get_sim_p(config)
        # obj = np.concatenate([pref - get_sim_p(config), vref - vol])
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
    name = animal + "_" + study
    print(name)

    # read measurements
    data_o = read_data(animal, study)

    # scale the LAD inflow according to microsphere measurements
    dvlad = data_o["vlad"].max() - data_o["vlad"].min()
    dvmyo = data_o["vmyo"].max() - data_o["vmyo"].min()
    data_o["qlad"] *= data_o["dvcycle"] / dvlad
    data_o["vlad"] *= data_o["dvcycle"] / dvlad

    # interoplate to simulation time points and smooth flow rate
    data_s = smooth_data(data_o, 201)

    # read literature data
    lit_path = os.path.join("data", "kim10b_table3.json")
    lit_data = read_config(lit_path)

    # create 0D model
    config = read_config(f"{model}.json")
    config[str_param][str_time] = len(data_s["t"])

    # set boundary conditions
    pini = {}
    pini[("BC_AT", "t")] = (data_s["t"].tolist(), None, None)
    pini[("BC_AT", "Q")] = (data_s["qlad"].tolist(), None, None)
    pini[("BC_COR", "t")] = (data_s["t"].tolist(), None, None)
    pini[("BC_COR", "Pim")] = (data_s["pven"].tolist(), None, None)
    set_params(config, pini)

    # set initial values
    bounds = {"R": (1e3, 1e6), "C": (1e-12, 1e-3), "L": (1e-12, 1e12)}
    p0 = OrderedDict()
    p0[("BC_COR", "Ra1")] = (1e+4, *bounds["R"])
    p0[("BC_COR", "Ra2")] = (1e+4, *bounds["R"])
    p0[("BC_COR", "Rv1")] = (1e+4, *bounds["R"])
    p0[("BC_COR", "Ca")] = (1e-5, *bounds["C"])
    p0[("BC_COR", "Cc")] = (1e-5, *bounds["C"])
    set_params(config, p0)

    data = {"o": data_o, "s": data_s}
    config_opt = optimize_zero_d(config, p0, data, verbose=verb)
    # print_params(config_opt, p0)

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
    
    # plot_data(animal, data)
    plot_results(animal, config, data)
    plot_optimized(animal, studies, optimized)

if __name__ == "__main__":
    main()
