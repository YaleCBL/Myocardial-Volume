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

from utils import read_config, mmHg_to_Ba, Ba_to_mmHg, str_val, str_bc, str_param, str_time, smooth, get_params, set_params, print_params, convert_units, units

n_in = "BC_AT:BV"
n_out = "BV:BC_COR"
model = "coronary"

def read_data(animal, study):
    csv_file = os.path.join("data", f"{get_name(animal)}_{study}.csv")
    if not os.path.isfile(csv_file):
        print(f"Data file {csv_file} not found.")
        return {}
    df = pd.read_csv(csv_file)
    data = {"t": df["t abs [s]"].to_numpy()}

    # convert pressure to cgs units
    for field in df.keys():
        if "mmHg" in field:
            df[field.replace("mmHg", "Ba")] = df[field] * mmHg_to_Ba

    data["pat"] = df["AoP [Ba]"].to_numpy()
    data["pven"] = df["LVP [Ba]"].to_numpy()
    data["vmyo"] = df["tissue vol ischemic [ml]"].to_numpy()
    data["qlad"] = df["LAD Flow [ml/s]"].to_numpy()

    data["qmyo"] = np.gradient(data["vmyo"], data["t"])
    data["vlad"] = cumulative_trapezoid(data["qlad"], data["t"], initial=0)
    dvlad = data["vlad"].max() - data["vlad"].min()

    csv_file = os.path.join("data", f"{get_name(animal)}_microsphere.csv")
    df = pd.read_csv(csv_file)
    data["dvcycle"] = df[f"{study} ischemic flow [ml/min/g]"].to_numpy()
    data["dvcycle"] /= 60.0  # convert from ml/min/g to ml/s/g
    data["dvcycle"] *= data["t"][-1] # convert from ml/s/g to ml/cycle/g

    # scale the LAD inflow according to microsphere measurements
    scale_vol = data["dvcycle"] / dvlad
    data["qlad"] *= scale_vol
    data["vlad"] *= scale_vol

    # read literature data
    lit_path = os.path.join(os.path.join("models", "kim10b_table3.json"))
    lit_data = read_config(lit_path)
    return data

def get_sim(config, loc):
    sim = pysvzerod.simulate(config)
    res = sim[sim["name"] == loc]["y"].to_numpy()
    if not res.size:
        raise ValueError(f"Result {loc} not found. Options are:\n" + ", ".join(np.unique(sim["name"]).tolist()))
    return res

def get_sim_p(config):
    return get_sim(config, "pressure:" + n_in)


def get_sim_v(config):
    if config['boundary_conditions'][1]['bc_type'] == "CORONARY":
        v_sim = get_sim(config, "volume_im:BC_COR")
    else:
        q_sim = get_sim(config, "flow:" + n_in)
        nt = config[str_param][str_time]
        tmax = config['boundary_conditions'][0]['bc_values']['t'][-1]
        ti = np.linspace(0.0, tmax, nt)
        v_sim = cumulative_trapezoid(q_sim, ti, initial=0)
    return v_sim - v_sim[0]

# smooths the data and interpolates it to a specific size nt
def smooth_data(data_o, nt):
    # Creates a dictionary with all the smoothed data 
    data_s = {}
    # Creates the time array of size nt with the high and low values of the time of the original data set 
    data_s["t"] = np.linspace(data_o["t"][0], data_o["t"][-1], nt)
    # goes through the 3 data sets 'pat' 'pven' and 'qlad' and uses the smooth function to smooth it without gathering the derivatives 
    for field in ["pat", "pven", "qlad"]:
        data_s[field], _ = smooth(data_o["t"], data_s["t"], data_o[field], cutoff=15)
    # smooths 'vmyo' and gets it's derivatives as 'qmyo'
    data_s["vmyo"], data_s["qmyo"] = smooth(data_o["t"], data_s["t"], data_o["vmyo"], cutoff=15)
    data_s["vmyo"] -= data_s["vmyo"][0]
    data_s["vlad"] = cumulative_trapezoid(data_s["qlad"], data_s["t"], initial=0)
    # returns the dictionary 
    return data_s


def plot_data(animal, data):
    n_param = len(data.keys())
    _, ax = plt.subplots(4, n_param, figsize=(n_param*5, 10), sharex="col", sharey="row")
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
    plt.savefig(f"plots/{get_name(animal)}_data.pdf")
    plt.close()

def plot_results(animal, config, data):
    labels = {"qlad": "LAD Flow (scaled) [ml/s]",
                "pat": "Aterial Pressure [mmHg]",
                "pven": "Left-Ventricular Pressure [mmHg]",
                "vmyo": "Myocardial Volume [ml]"}
    _, axs = plt.subplots(len(labels), len(config), figsize=(16, 9), sharex="col", sharey="row")
    if len(config) == 1:
        axs = axs.reshape(-1,1)
    for j, study in enumerate(config.keys()):
        with open(f'results/{animal}_{study}.json', "w") as f:
            json.dump(config[study], f, indent=2)

        ti = config[study]["boundary_conditions"][0]["bc_values"]["t"]
        datm = data[study]["s"]

        dats = OrderedDict()
        dats["qlad"] = get_sim(config[study], "flow:" + n_in)
        dats["pven"] = datm["pven"]
        dats["pat"] = get_sim(config[study], "pressure:" + n_in)
        dats["vmyo"] = get_sim_v(config[study])

        convert = {"p": Ba_to_mmHg, "v": 1.0, "q": 1.0}

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
                axs[i, j].set_ylabel(labels[k])

    plt.tight_layout()
    plt.savefig(f"plots/{get_name(animal)}_{model}_simulated.pdf")
    plt.close()

def plot_parameters(animal, optimized):
    studies = list(optimized.keys())
        
    # add Ra sum
    # for s in studies:
    #     total = ('BC_COR', 'Rx=Ra1+Ra2+Rv')
    #     optimized[s][total] = 0
    #     for res in ['Ra1', 'Ra2', 'Rv1']:
    #         optimized[s][total] += optimized[s][('BC_COR', res)]

    # add time constants
    # pairs = [('Ra1', 'Ca'), ('Ra2', 'Cc')]
    # for s in studies:
    #     for i, pair in enumerate(pairs):
    #         total = ('BC_COR', 'tau'+str(i))
    #         optimized[s][total] = 1
    #         for res in pair:
    #             optimized[s][total] *= optimized[s][('BC_COR', res)]

    n_param = len(optimized[studies[0]].keys())
    _, axes = plt.subplots(1, n_param, figsize=(n_param*3, 6))
    if n_param == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Group parameters by zerod type
    param_groups = defaultdict(list)
    for i, (param, val) in enumerate(optimized[studies[0]].keys()):
        param_groups[val[0]] += [(i, param, val)]

    # Plot parameters sharing y axes within groups
    for zerod, params in param_groups.items():
        first_ax = None
        for i, param, val in params:
            values = np.array([optimized[s][(param, val)] for s in studies])
            values, unit, name = convert_units(zerod, values, units)
            ylabel = f'{name} [{unit}]'
            print(f'{param} {zerod} {values.min():.1e} - {values.max():.1e} [{unit}]')
            
            if first_ax is None:
                first_ax = axes[i]
                axes[i].set_ylabel(ylabel)
            else:
                axes[i].sharey(first_ax)
                axes[i].set_ylabel('')
            
            axes[i].grid(True, axis='y')
            axes[i].bar(range(len(studies)), values)
            axes[i].set_xticks(range(len(studies)))
            axes[i].set_xticklabels(studies, rotation=45)
            axes[i].set_title(f'{animal} {val}')
            axes[i].tick_params(axis='both', which='major')

    plt.tight_layout()
    plt.savefig(f"plots/{get_name(animal)}_{model}_parameters.pdf")
    plt.close()

# The objective function used for the optimization of the parameters
def get_objective(ref, sim, t=None):
    # If a specifc time is given then tnat value is evaluated 
    if t is not None:
        ref = ref[t]
        sim = sim[t]
    # objective function which normalizes the difference using the mean 
    obj = (ref - sim) / np.mean(ref)
    return obj

# Optimizes the parameters using a least squares optimization and returns the edited config 
def optimize_zero_d(config, p0, data, verbose=0):
    # grabs the references for the pressure and volume from the actual data 
    pref = data["s"]["pat"]
    vref = data["s"]["vmyo"]
    # defines a cost function for the parameters 
    def cost_function(p):
        # sets parameters according to passed p array for the x in the set_params functions 
        pset = set_params(config, p0, p)
        # prints value that are set if verbose is 1 
        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")
        t_p = [0, np.argmax(pref), -1]
        t_q = [0, np.argmin(vref), -1]
        # runs the simulation and gets the objective function value and saves them for 
        # both the pressure and the volume 
        obj_p = get_objective(pref, get_sim_p(config))
        obj_v = get_objective(vref, get_sim_v(config))
        # concatenates them to return one object 
        obj = np.concatenate((obj_p, obj_v))
        # prints them if opted to 
        if verbose:
            print(f"{np.linalg.norm(obj):.1e}", end="\n")
        return obj
    # gets the initial parameter values 
    initial = get_params(p0)
    # prints them if opted to 
    if verbose:
        for k in p0.keys():
            print(f"{k[1]}", end="\t")
        print("obj", end="\n")
    # runs a leas squares on the initial values with the cost_function 
    res = least_squares(cost_function, initial, jac="2-point")#, method="lm", diff_step=1e-3
    # sets the parameters based on the results of least squares 
    set_params(config, p0, res.x)
    return config


def get_name(animal):
    return f"DSEA{animal:02d}"


def read_and_smooth_data(animal, study):
    print(f"{get_name(animal)}_{study}")

    # read measurements
    data_o = read_data(animal, study)
    if data_o == {}:
        return {}

    # interoplate to simulation time points and smooth flow rate
    data_s = smooth_data(data_o, 201)

    return {"o": data_o, "s": data_s}

def estimate(data, verb=0):
    # create 0D model
    config = read_config(f"models/{model}.json")
    config[str_param][str_time] = len(data["s"]["t"])

    # set boundary conditions
    pini = {}
    pini[("BC_AT", "t")] = (data["s"]["t"].tolist(), None, None)
    pini[("BC_AT", "Q")] = (data["s"]["qlad"].tolist(), None, None)
    pini[("BC_COR", "t")] = (data["s"]["t"].tolist(), None, None)
    pini[("BC_COR", "Pim")] = (data["s"]["pven"].tolist(), None, None)
    set_params(config, pini)

    # set initial values
    bounds = {"R": (1e4, 1e8), "Rv": (1e2, 1e6), "C": (1e-12, 1e-3), "L": (1e-12, 1e12)}
    p0 = OrderedDict()
    p0[("BC_COR", "Ra1")] = (1e+5, *bounds["R"])
    p0[("BC_COR", "Ra2")] = (1e+5, *bounds["R"])
    p0[("BC_COR", "Rv1")] = (1e+4, *bounds["Rv"])
    p0[("BC_COR", "Ca")] = (1e-5, *bounds["C"])
    p0[("BC_COR", "Cc")] = (1e-5, *bounds["C"])
    # p0[("BC_COR", "P_v")] = (1e3, *bounds["P"])
    set_params(config, p0)

    config_opt = optimize_zero_d(config, p0, data, verbose=verb)
    print_params(config_opt, p0)

    return config_opt, p0


def main():
    animals = [8]
    # animals = [8, 10, 15, 16] # clean
    # animals = [6, 7, 8, 10, 14, 15, 16] # all
    studies = [
        "baseline",
        "mild_sten",
        "mild_sten_dob",
        "mod_sten", 
        "mod_sten_dob",
    ]
    for animal in animals:
        process(animal, studies)
    

def process(animal, studies):
    optimized = defaultdict(dict)
    data = {}
    config = {}

    for study in studies:
        dat = read_and_smooth_data(animal, study)
        if dat != {}:
            data[study] = dat
    plot_data(animal, data)

    for study in studies:
        print(f"Estimating {study}...")
        config[study], p0 = estimate(data[study], verb=1)
        params = [opt[0] for opt in p0]
        values = [opt[1] for opt in p0]
        for param in params:
            for val in values:
                for vessel in config[study]["vessels"]:
                    if vessel["vessel_name"] == param:
                        if val in vessel[str_val]:
                            optimized[study][(param, val)] = vessel[str_val][val]
                for bc in config[study][str_bc]:
                    if bc["bc_name"] == param:
                        if val in bc["bc_values"]:
                            optimized[study][(param, val)] = bc["bc_values"][val]
    
    plot_results(animal, config, data)
    plot_parameters(animal, optimized)

if __name__ == "__main__":
    main()
