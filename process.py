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
from scipy.interpolate import interp1d
from scipy.optimize import minimize, least_squares, Bounds
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

str_val = "zero_d_element_values"
str_bc = "boundary_conditions"
str_param = "simulation_parameters"

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

    plad = df["AoP [Ba]"].to_numpy()
    pven = df["LVP [Ba]"].to_numpy()
    vmyo = df["tissue vol ischemic [ml]"].to_numpy()
    qlad = df["LAD Flow [mL/s]"].to_numpy()

    csv_file = os.path.join("data", animal + "_microsphere.csv")
    df = pd.read_csv(csv_file)
    for k in df.keys():
        if study in k:
            dvcycle = df[k][0]
            break
    return t, plad, pven, vmyo, qlad, dvcycle


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


def get_v_sim(config):
    sim = pysvzerod.simulate(config)
    name = "flow:pa:RC1"
    y = sim[sim["name"] == name]["y"].to_numpy()
    tmax = config[str_bc][0]["bc_values"]["t"][-1]
    nt = config[str_param]["number_of_time_pts_per_cardiac_cycle"]
    return np.cumsum(y) * tmax / nt


def get_sim(config, loc):
    sim = pysvzerod.simulate(config)
    res = sim[sim["name"] == loc]["y"].to_numpy()
    if not res.size:
        raise ValueError(f"Result {loc} not found. Options are:\n" + ", ".join(np.unique(sim["name"]).tolist()))
    return res


def get_sim_out(config):
    q_sim_in = get_sim(config, "flow:J1:Cim")
    # q_sim_out = get_sim(config, "flow:Cim:BC_PLV")
    # return q_sim_out - q_sim_in
    return q_sim_in
    # return get_sim(config, "flow:J1:Rv")


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
        str += id + " " + k[0] + f" {val:.1e} "
    print(str)


def smooth(t, ti, qt, cutoff=10):
    q = np.interp(ti, t, qt)
    N = len(q)
    freqs = fftfreq(N, d=t[1] - t[0])
    q_fft = fft(q)
    q_fft[np.abs(freqs) > cutoff] = 0
    qs = np.real(ifft(q_fft))
    return qs


def plot_data(study, data):
    _, ax = plt.subplots(2, 2, figsize=(15, 9))
    ax = ax.flatten()
    axt = [ax[i].twinx() for i in range(len(ax))]
    mk = {"original": ".", "smoothed": "-"}

    ax[0].set_xlabel("LAD Pressure [mmHg]")
    ax[0].set_ylabel("Volume [mL]")
    ax[1].set_ylabel("Delta pressure [mmHg]")

    for j, loc in enumerate(["myo", "lad"]):
        ax[j + 2].set_ylabel(f"{loc} volume [mL]", color="b")
        axt[j + 2].set_ylabel(f"{loc} flow [mL/s]", color="m")

    for i in range(1, 4):
        ax[i].set_xlabel("Time [s]")

    for k in ["original", "smoothed"]:
        ax[0].plot(data[k]["plad"]*Ba_to_mmHg, data[k]["vmyo"], f"k{mk[k]}")
        ax[1].plot(data[k]["t"], (data[k]["plad"] - data[k]["pven"])*Ba_to_mmHg, f"r{mk[k]}")

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


def plot_results(name, config, ti, plad, plv, qlads, fname, save=True):
    with open(name + ".json", "w") as f:
        json.dump(config, f, indent=2)

    plad_sim = get_sim(config, "pressure:BC_Par:Ra")
    plv_sim = get_sim(config, "pressure:Cim:BC_PLV")
    q_sim = get_sim_out(config)

    _, axs = plt.subplots(3, 1, figsize=(12, 9))

    i = 0
    axs[i].plot(ti, plad_sim*Ba_to_mmHg, "r-", label="simulated")
    axs[i].plot(ti, plad*Ba_to_mmHg, "k--", label="measured (smoothed)")
    axs[i].legend()
    axs[i].set_xlabel("Time [s]")
    axs[i].set_ylabel("Pressure [mmHg]")

    i = 1
    axs[i].plot(ti, plv_sim*Ba_to_mmHg, "r-", label="simulated")
    axs[i].plot(ti, plv*Ba_to_mmHg, "k--", label="measured (smoothed)")
    axs[i].legend()
    axs[i].set_xlabel("Time [s]")
    axs[i].set_ylabel("Pressure [mmHg]")

    i = 2
    axs[i].plot(ti, q_sim, "r-", label="simulated")
    axs[i].plot(ti, qlads, "k--", label="measured (smoothed)")
    axs[i].legend()
    axs[i].set_xlabel("Time [s]")
    axs[i].set_ylabel("Flow [ml/s]")

    plt.tight_layout()
    if save:
        plt.savefig(name + "_" + fname + ".pdf")
        plt.close()
    else:
        plt.show()


def optimize_zero_d(config, p0, ti, plads, qlads, verbose=0):
    def cost_function(p):
        pset = set_params(config, p0, p)
        if verbose:
            for val in pset:
                print(f"{val:.12e}", end="\t")
            if verbose > 1:
                plot_results("iter", config, ti, plads, qlads, save=False)
        obj = 1 - get_sim_out(config) / qlads / len(qlads)
        print(f"{np.linalg.norm(obj):.12e}", end="\n")
        return obj

    initial = get_params(p0)
    if verbose:
        for k in p0.keys():
            print(f"{k[0][:5]} {k[1][0]}", end="\t")
        print("obj", end="\n")
    res = least_squares(cost_function, initial, jac="2-point", method="lm", diff_step=1e-2)
    set_params(config, p0, res.x)
    return config


def estimate(animal, study):
    # read measurements
    name = animal + "_" + study
    t, plad, pven, vmyo, qlad, dvcycle = read_data(animal, study)
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
    plads = smooth(t, ti, plad, cutoff=15)
    pvens = smooth(t, ti, pven, cutoff=15)
    qlads = smooth(t, ti, qlad, cutoff=15)
    vmyos = smooth(t, ti, vmyo, cutoff=8)
    qmyos = np.gradient(vmyos, ti)
    vlads = cumulative_trapezoid(qlads, ti, initial=0)
    tv = [0.0, ti[-1]]
    pv = [0.0, 0.0]

    # create dictionary with all data
    data = {}
    data["original"] = {
        "t": t,
        "plad": plad,
        "pven": pven,
        "vmyo": vmyo,
        "qmyo": qmyo,
        "qlad": qlad,
        "vlad": vlad,
    }
    data["smoothed"] = {
        "t": ti,
        "plad": plads,
        "pven": pvens,
        "vmyo": vmyos,
        "qmyo": qmyos,
        "qlad": qlads,
        "vlad": vlads,
    }
    plot_data(name, data)

    # read literature data
    lit_path = os.path.join("data", "kim10b_table3.json")
    lit_data = read_config(lit_path)

    # create 0D model
    config = read_config("RCRCR_kim10b.json")
    config[str_param]["number_of_time_pts_per_cardiac_cycle"] = nt

    # set boundary conditions
    pini = {}
    pini[("BC_Par", "t")] = (ti.tolist(), None, None)
    pini[("BC_Par", "P")] = (plads.tolist(), None, None)
    pini[("BC_PLV", "t")] = (ti.tolist(), None, None)
    pini[("BC_PLV", "P")] = (pvens.tolist(), None, None)
    pini[("BC_Pv0", "t")] = (tv, None, None)
    pini[("BC_Pv0", "P")] = (pv, None, None)
    pini[("BC_Pv1", "t")] = (tv, None, None)
    pini[("BC_Pv1", "P")] = (pv, None, None)
    set_params(config, pini)

    # set initial parameter guess from kim10b
    lit_data = read_lit_data("kim10b_table3.json")
    lit_vessel = lit_data["a"]
    status = "R" # rest
    bounds = {"R": (1e0, 1e12), "C": (1e-12, 1e-3), "L": (1e-12, 1e0)}
    param = ["Ra", "Ca", "Ra-micro", "Cim", "Rv"]

    for p in param:
        for vessel in config["vessels"]:
            if vessel["vessel_name"] == p:
                pini[(p, p[0])] = (lit_vessel[p][status], *bounds[p[0]])
    set_params(config, pini)
    
    p0 = OrderedDict()
    optimize = [
        ("Ra", "R"),
        ("Ca", "C"),
        # ("Ra-micro", "R"),
        ("Cim", "C"),
        ("Rv", "R"),
        ]
    for o in optimize:
        p0[o] = pini[o]
    set_params(config, p0)
    plot_results(name, config, ti, plads, pvens, qlads, "literature")

    config_opt = optimize_zero_d(config, p0, ti, plads, qlads, verbose=1)
    plot_results(name, config_opt, ti, plads, pvens, qlads, "simulated")
    print(name)
    print_params(config_opt, p0)


def main():
    animal = "DSEA08"
    studies = [
        "baseline",
        # "mild_sten",
        # "mild_sten_dob",
        # "mod_sten",
        # "mod_sten_dob",
    ]
    for study in studies:
        estimate(animal, study)


if __name__ == "__main__":
    main()
