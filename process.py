#!/usr/bin/env python3

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


def read_data(animal, study):
    csv_file = animal + "_" + study + ".csv"
    df = pd.read_csv(csv_file)
    t = df["t abs [s]"].to_numpy()
    plad = df["AoP [mmHg]"].to_numpy()
    pven = df["LVP [mmHg]"].to_numpy()
    vmyo = df["tissue vol ischemic [ml]"].to_numpy()
    qlad = df["LAD Flow [mL/s]"].to_numpy()

    csv_file = animal + "_microsphere.csv"
    df = pd.read_csv(csv_file)
    for k in df.keys():
        if study in k:
            dvcycle = df[k][0]
            break
    return t, plad, pven, vmyo, qlad, dvcycle


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


def smooth(t, q, cutoff=10):
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
    markers = {"original": ".", "smoothed": "-"}

    ax[0].set_xlabel("Pressure [mmHg]")
    ax[0].set_ylabel("Volume [mL]")

    ax[1].set_ylabel("LAD Pressure [mmHg]")

    for j, loc in enumerate(["myo", "lad"]):
        ax[j + 2].set_ylabel(f"{loc} volume [mL]", color="b")
        axt[j + 2].set_ylabel(f"{loc} flow [mL/s]", color="m")

    for i in range(1, 4):
        ax[i].set_xlabel("Time [s]")

    for k in ["original", "smoothed"]:
        ax[0].plot(data[k]["p"], data[k]["vmyo"], f"k{markers[k]}", label=k)
        ax[1].plot(
            data[k]["t"], data[k]["p"], f"r{markers[k]}", label=f"Pressure ({k})"
        )

        for j, loc in enumerate(["myo", "lad"]):
            ax[j + 2].plot(
                data[k]["t"],
                data[k]["v" + loc],
                f"b{markers[k]}",
                label=f"Volume {loc} ({k})",
            )
            axt[j + 2].plot(
                data[k]["t"],
                data[k]["q" + loc],
                f"m{markers[k]}",
                label=f"Flow {loc} ({k})",
            )

    plt.tight_layout()
    plt.savefig(study + "_data.pdf")
    plt.close()


def plot_results(name, config, ti, plads, qlads, save=True):
    with open(name + ".json", "w") as f:
        json.dump(config, f, indent=2)

    p_estim = get_sim(config, "pressure:v_i_p:BC_LAD")
    q_estim = get_sim(config, "flow:v_i_p:BC_LAD")

    _, axs = plt.subplots(2, 1, figsize=(12, 9))

    i = 0
    axs[i].plot(ti, plads, "k", label="measured (smoothed)")
    axs[i].plot(ti, p_estim, "r--", label="simulated")
    axs[i].legend()
    axs[i].set_xlabel("Time [s]")
    axs[i].set_ylabel("Pressure [mmHg]")

    i = 1
    axs[i].plot(ti, qlads, "k", label="measured (smoothed)")
    axs[i].plot(ti, q_estim, "r--", label="simulated")
    axs[i].legend()
    axs[i].set_xlabel("Time [s]")
    axs[i].set_ylabel("Flow [mmHg]")

    plt.tight_layout()
    if save:
        plt.savefig(name + "_simulated.pdf")
        plt.close()
    else:
        plt.show()


def optimize_zero_d(config, p0, ti, plads, qlads, verbose=0):
    def cost_function(p):
        pset = set_params(config, p0, p)
        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")
            if verbose > 1:
                plot_results("iter", config, ti, plads, qlads, save=False)
        qsim = get_sim(config, "flow:v_i_p:BC_LAD")
        obj = qlads - qsim
        print(f"{np.linalg.norm(obj):.1e}", end="\n")
        return obj

    initial = get_params(p0)
    if verbose:
        for k in p0.keys():
            print(f"{k[0][:5]} {k[1][0]}", end="\t")
        print("obj", end="\n")
    res = least_squares(cost_function, initial, jac="2-point", method="lm")
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

    # simulation parameters
    nt = 201

    # interoplate to simulation time points
    ti = np.linspace(t[0], t[-1], nt)
    pladi = np.interp(ti, t, plad)
    vmyoi = np.interp(ti, t, vmyo)
    qladi = np.interp(ti, t, qlad)

    # calculate and smooth flow rate
    plads = smooth(ti, pladi, cutoff=15)
    qlads = smooth(ti, qladi, cutoff=15)
    vmyos = smooth(ti, vmyoi, cutoff=8)
    qmyos = np.gradient(vmyos, ti)
    vlads = cumulative_trapezoid(qlads, ti, initial=0)

    # create dictionary with all data
    data = {}
    data["original"] = {
        "t": t,
        "p": plad,
        "vmyo": vmyo,
        "qmyo": qmyo,
        "qlad": qlad,
        "vlad": vlad,
    }
    data["smoothed"] = {
        "t": ti,
        "p": plads,
        "vmyo": vmyos,
        "qmyo": qmyos,
        "qlad": qlads,
        "vlad": vlads,
    }
    plot_data(name, data)

    # create 0D model
    config = read_config("myocardium_bifurcation.json")
    config[str_param]["number_of_time_pts_per_cardiac_cycle"] = nt

    # set boundary conditions
    pini = {}
    pini[("BC_myo", "t")] = (ti.tolist(), None, None)
    pini[("BC_myo", "Q")] = (qmyos.tolist(), None, None)
    pini[("BC_LAD", "t")] = (ti.tolist(), None, None)
    pini[("BC_LAD", "P")] = (plads.tolist(), None, None)
    set_params(config, pini)

    # initial guess
    p0 = OrderedDict()
    p0[("BC_distal", "P")] = (1e1, 1e-2, 1e2)
    for i in ["v_myo", "v_i_p", "v_i_d"]:
        p0[(i, "R_poiseuille")] = (1.0e0, 1e-6, 1e4)
        # p0[(i, "C")] = (1.0e-3, 1e-12, 1e1)
        # p0[(i, "L")] = (1.0e-9, 1e-12, 1e1)
    # p0[("BC", "Rp")] = (1.0e1, 1e-6, 1e6)
    # p0[("BC", "Rd")] = (1.0e1, 1e-6, 1e6)
    # p0[("BC", "C")] = (1.0e-3, 1e-9, 1e1)
    set_params(config, p0)

    config_opt = optimize_zero_d(config, p0, ti, plads, qlads, verbose=1)
    plot_results(name, config_opt, ti, plads, qlads)
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
