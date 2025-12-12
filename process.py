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
from scipy.optimize import minimize, least_squares, Bounds, differential_evolution
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

from utils import (
    read_config,
    mmHg_to_Ba,
    Ba_to_mmHg,
    str_val,
    bc_val,
    str_bc,
    str_param,
    str_time,
    smooth,
    get_params,
    set_params,
    print_params,
    convert_units,
    units,
    get_param_map,
)

n_in = "BC_AT:BV"
n_out = "BV:BC_COR"
model = "coronary_varres"


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
    data["dvcycle"] *= data["t"][-1]  # convert from ml/s/g to ml/cycle/g

    # scale the LAD inflow according to microsphere measurements
    scale_vol = data["dvcycle"] / dvlad
    data["qlad"] *= scale_vol
    data["vlad"] *= scale_vol

    # read literature data
    lit_path = os.path.join(os.path.join("models", "kim10b_table3.json"))
    lit_data = read_config(lit_path)
    return data


def get_sim(config, loc):
    try:
        sim = pysvzerod.simulate(config)
        res = sim[sim["name"] == loc]["y"].to_numpy()
        if not res.size:
            raise ValueError(
                f"Result {loc} not found. Options are:\n"
                + ", ".join(np.unique(sim["name"]).tolist())
            )
    except RuntimeError:
        print("Simulation failed")
        res = np.array([0.0])
    return res


def get_sim_pv(config):
    """Run simulation once and extract both pressure and volume.

    Returns:
        tuple: (pressure, volume) arrays
    """
    try:
        sim = pysvzerod.simulate(config)

        # Extract pressure
        p_sim = sim[sim["name"] == "pressure:" + n_in]["y"].to_numpy()
        if not p_sim.size:
            raise ValueError(f"Pressure result not found at pressure:{n_in}")

        # Extract volume
        if "CORONARY" in config["boundary_conditions"][1]["bc_type"]:
            v_sim = sim[sim["name"] == "volume_im:BC_COR"]["y"].to_numpy()
            if not v_sim.size:
                raise ValueError("Volume result not found at volume_im:BC_COR")
        else:
            q_sim = sim[sim["name"] == "flow:" + n_in]["y"].to_numpy()
            if not q_sim.size:
                raise ValueError(f"Flow result not found at flow:{n_in}")
            nt = config[str_param][str_time]
            tmax = config["boundary_conditions"][0]["bc_values"]["t"][-1]
            ti = np.linspace(0.0, tmax, nt)
            v_sim = cumulative_trapezoid(q_sim, ti, initial=0)

        v_sim = v_sim - v_sim[0]
        return p_sim, v_sim

    except RuntimeError:
        print("Simulation failed")
        nt = config[str_param][str_time]
        return np.zeros(nt), np.zeros(nt)


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
    data_s["vmyo"], data_s["qmyo"] = smooth(
        data_o["t"], data_s["t"], data_o["vmyo"], cutoff=15
    )
    data_s["vmyo"] -= data_s["vmyo"][0]
    data_s["vlad"] = cumulative_trapezoid(data_s["qlad"], data_s["t"], initial=0)
    # returns the dictionary
    return data_s


def plot_data(animal, data):
    n_param = len(data.keys())
    _, ax = plt.subplots(
        4, n_param, figsize=(n_param * 5, 10), sharex="col", sharey="row"
    )
    axt = np.array(
        [[ax[i, j].twinx() for j in range(ax.shape[1])] for i in range(ax.shape[0])]
    )
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
            ax[1, i].plot(
                data[s][k]["t"],
                (data[s][k]["pat"] - data[s][k]["pven"]) * Ba_to_mmHg,
                f"r{mk[k]}",
            )

            for j, loc in enumerate(["myo", "lad"]):
                ax[j + 2, i].plot(data[s][k]["t"], data[s][k]["v" + loc], f"b{mk[k]}")
                axt[j + 2, i].plot(data[s][k]["t"], data[s][k]["q" + loc], f"m{mk[k]}")

    plt.tight_layout()
    plt.savefig(f"plots/{get_name(animal)}_data.pdf")
    plt.close()


def plot_results(animal, config, data):
    labels = {
        "qlad": "LAD Flow (scaled) [ml/s]",
        "pat": "Aterial Pressure [mmHg]",
        "pven": "Left-Ventricular Pressure [mmHg]",
        "vmyo": "Myocardial Volume [ml]",
    }
    _, axs = plt.subplots(
        len(labels), len(config), figsize=(16, 9), sharex="col", sharey="row"
    )
    if len(config) == 1:
        axs = axs.reshape(-1, 1)
    for j, study in enumerate(config.keys()):
        with open(f"results/{get_name(animal)}_{study}_{model}.json", "w") as f:
            json.dump(config[study], f, indent=2)

        ti = config[study]["boundary_conditions"][0]["bc_values"]["t"]
        datm = data[study]["s"]

        dats = OrderedDict()
        dats["qlad"] = get_sim(config[study], "flow:" + n_in)
        dats["pven"] = datm["pven"]
        dats["pat"] = get_sim(config[study], "pressure:" + n_in)
        _, dats["vmyo"] = get_sim_pv(config[study])

        convert = {"p": Ba_to_mmHg, "v": 1.0, "q": 1.0}

        for i, k in enumerate(dats.keys()):
            axs[i, j].plot(ti, dats[k] * convert[k[0]], "r-", label="simulated")
            axs[i, j].plot(
                ti, datm[k] * convert[k[0]], "k--", label="measured (smoothed)"
            )
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
    _, axes = plt.subplots(1, n_param, figsize=(n_param * 3, 6))
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
            ylabel = f"{name} [{unit}]"
            print(f"{param} {zerod} {values.min():.1e} - {values.max():.1e} [{unit}]")

            if first_ax is None:
                first_ax = axes[i]
                axes[i].set_ylabel(ylabel)
            else:
                axes[i].sharey(first_ax)
                axes[i].set_ylabel("")

            axes[i].grid(True, axis="y")
            axes[i].bar(range(len(studies)), values)
            axes[i].set_xticks(range(len(studies)))
            axes[i].set_xticklabels(studies, rotation=45)
            axes[i].set_title(f"{get_name(animal)} {val}")
            axes[i].tick_params(axis="both", which="major")

    plt.tight_layout()
    plt.savefig(f"plots/{get_name(animal)}_{model}_parameters.pdf")
    plt.close()


def plot_parameters_multi(animals_optimized, studies=None):
    # Get first animal to determine studies and parameters
    first_animal = list(animals_optimized.keys())[0]
    if studies is None:
        studies = list(animals_optimized[first_animal].keys())

    # Get all parameters from first study of first animal
    first_study = studies[0]
    params = list(animals_optimized[first_animal][first_study].keys())

    n_param = len(params)
    n_studies = len(studies)

    _, axes = plt.subplots(1, n_param, figsize=(n_param * 3, 6))
    if n_param == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Group parameters by zerod type
    param_groups = defaultdict(list)
    for i, (param, val) in enumerate(params):
        param_groups[val[0]] += [(i, param, val)]

    # Plot parameters sharing y axes within groups
    for zerod, param_list in param_groups.items():
        first_ax = None
        for i, param, val in param_list:
            # Collect values for all animals and studies
            all_values = []
            for study_idx, study in enumerate(studies):
                study_values = []
                for animal in animals_optimized.keys():
                    if (
                        study in animals_optimized[animal]
                        and (param, val) in animals_optimized[animal][study]
                    ):
                        study_values.append(
                            animals_optimized[animal][study][(param, val)]
                        )
                all_values.append(study_values)

            # Convert units
            flat_values = [v for sublist in all_values for v in sublist]
            if flat_values:
                flat_values_array = np.array(flat_values)
                _, unit, name = convert_units(zerod, flat_values_array, units)
                ylabel = f"{name} [{unit}]"

                # Convert all values with the same conversion
                converted_values = []
                for study_values in all_values:
                    if study_values:
                        converted = convert_units(zerod, np.array(study_values), units)[
                            0
                        ]
                        converted_values.append(converted)
                    else:
                        converted_values.append(np.array([]))

                # Calculate mean and std for each study
                means = []
                stds = []
                for converted in converted_values:
                    if len(converted) > 0:
                        means.append(np.mean(converted))
                        stds.append(np.std(converted))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                # Set up axis
                if first_ax is None:
                    first_ax = axes[i]
                    axes[i].set_ylabel(ylabel)
                else:
                    axes[i].sharey(first_ax)
                    axes[i].set_ylabel("")

                axes[i].grid(True, axis="y")

                # Plot individual samples as scatter points
                for study_idx, converted in enumerate(converted_values):
                    if len(converted) > 0:
                        x_positions = np.full(len(converted), study_idx)
                        # Add small random jitter for visibility
                        # x_positions += np.random.normal(0, 0.05, len(converted))
                        axes[i].scatter(
                            x_positions,
                            converted,
                            alpha=0.6,
                            s=50,
                            color="gray",
                            zorder=2,
                        )

                # Plot mean as bar with error bars for std
                x_pos = range(n_studies)
                axes[i].bar(x_pos, means, alpha=0.5, color="steelblue", zorder=1)
                axes[i].errorbar(
                    x_pos,
                    means,
                    yerr=stds,
                    fmt="none",
                    ecolor="black",
                    capsize=5,
                    capthick=2,
                    zorder=3,
                )

                axes[i].set_xticks(range(n_studies))
                axes[i].set_xticklabels(studies, rotation=45)
                axes[i].set_title(f"{val}")
                axes[i].tick_params(axis="both", which="major")

                print(
                    f"{param} {zerod} mean: {np.nanmean(means):.1e} ± {np.nanmean(stds):.1e} [{unit}]"
                )

    plt.tight_layout()
    plt.savefig(f"plots/multi_animal_{model}_parameters.pdf")
    plt.close()


def plot_convergence(animal, study, p0, param_history, obj_history):
    """Plot parameter convergence during optimization.

    Args:
        animal: Animal identifier
        study: Study name
        p0: Parameter dictionary with (val, min, max, map_type) tuples
        param_history: List of parameter values at each iteration
        obj_history: List of objective function values at each iteration
    """
    if not param_history:
        print("No convergence history to plot")
        return

    # Convert history to numpy array for easier manipulation
    param_history = np.array(param_history)
    n_iter, n_params = param_history.shape

    # Get parameter names and bounds
    param_names = [f"{k[1]}" for k in p0.keys()]
    bounds = [(pmin, pmax) for _, pmin, pmax, _ in p0.values()]

    # Create figure with two subplots: parameters and objective
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Generate distinct colors for each parameter
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))
    if n_params > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_params))

    # Plot 1: Parameter evolution with bounds
    iterations = np.arange(n_iter)
    for i, (name, color) in enumerate(zip(param_names, colors)):
        # Plot parameter evolution
        ax1.semilogy(
            iterations, param_history[:, i], "-", color=color, label=name, linewidth=2
        )

        # Plot bounds as dashed horizontal lines
        ax1.axhline(bounds[i][0], color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax1.axhline(bounds[i][1], color=color, linestyle="--", alpha=0.5, linewidth=1)

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Parameter Value (log scale)", fontsize=12)
    ax1.set_title(
        f"{get_name(animal)} {study} - Parameter Convergence", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=10, ncol=2)

    # Plot 2: Objective function evolution
    ax2.semilogy(iterations, obj_history, "k-", linewidth=2)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Objective Function (log scale)", fontsize=12)
    ax2.set_title("Objective Function Convergence", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/{get_name(animal)}_{study}_{model}_convergence.pdf")
    plt.close()

    print(
        f"Convergence plot saved: plots/{get_name(animal)}_{study}_{model}_convergence.pdf"
    )


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

    # Track parameter history for convergence plot
    param_history = []
    obj_history = []

    # defines a cost function for the parameters
    def cost_function(p):
        # sets parameters according to passed p array for the x in the set_params functions
        pset = set_params(config, p0, p)
        # Store parameter values for convergence plot
        param_history.append(pset.copy())

        # prints value that are set if verbose is 1
        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")
        t_p = [0, np.argmax(pref), -1]
        t_v = [0, np.argmin(vref), -1]
        # runs the simulation ONCE and gets both pressure and volume
        p_sim, v_sim = get_sim_pv(config)
        # compute objective function for both the pressure and the volume
        obj_p = get_objective(pref, p_sim)
        obj_v = get_objective(vref, v_sim)
        # concatenates them to return one object
        obj = np.concatenate((obj_p, obj_v))
        obj_history.append(np.linalg.norm(obj))

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
    # Apply the appropriate map to bounds based on parameter type
    bounds = []
    for param_tuple in p0.values():
        _, pmin, pmax, map_type = param_tuple
        forward_map, _ = get_param_map(map_type)
        bounds.append((forward_map(pmin), forward_map(pmax)))
    res = least_squares(cost_function, initial, bounds=np.array(bounds).T)
    # sets the parameters based on the results of least squares
    set_params(config, p0, res.x)

    # Return config, error, and convergence history
    return config, np.linalg.norm(res.fun), param_history, obj_history


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


def estimate(data, animal, study, verb=0):
    # create 0D model
    config = read_config(f"models/{model}.json")
    config[str_param][str_time] = len(data["s"]["t"])

    # set boundary conditions
    pini = {}
    pini[("BC_AT", "t")] = (data["s"]["t"].tolist(), None, None, "lin")
    pini[("BC_AT", "Q")] = (data["s"]["qlad"].tolist(), None, None, "lin")
    pini[("BC_COR", "t")] = (data["s"]["t"].tolist(), None, None, "lin")
    pini[("BC_COR", "Pim")] = (data["s"]["pven"].tolist(), None, None, "lin")

    # set timing constants
    # t_v_min, t_v_dia = get_sys_dia(data)
    t = data["o"]["t"]
    V = data["o"]["vmyo"]
    t_v_min = t[np.argmin(V)]
    t_v_dia = t_v_min
    print(f"t_v_min: {t_v_min:.3f}, t_v_dia: {t_v_dia:.3f}")
    # pdb.set_trace()
    pini[("BC_COR", "T_vc")] = (t_v_min, None, None, "lin")
    pini[("BC_COR", "T_vr")] = (t_v_dia, None, None, "lin")

    set_params(config, pini)

    # set initial values (val, min, max, map_type)
    # map_type can be 'log' (logarithmic) or 'linear'
    # We optimize resistances and time constants directly
    # Ra2 parameterization: Ra2 (geometric mean) and ratio_Ra2 (max/min)
    #   Ra2_min = Ra2 / sqrt(ratio_Ra2)
    #   Ra2_max = Ra2 * sqrt(ratio_Ra2)
    p0 = OrderedDict()
    p0[("BC_COR", "Ra1")] = (1e6, 1e5, 1e8, "log")
    p0[("BC_COR", "Rv1")] = (1e5, 1e3, 1e6, "log")
    p0[("BC_COR", "Ra2")] = (1e6, 1e5, 1e8, "log")  # geometric mean
    p0[("BC_COR", "ratio_Ra2")] = (1.5, 1.0, 5.0, "lin")  # max/min ratio (must be >= 1)
    # Time constants: tc1 = Ra1 * Ca, tc2 = Rv1 * Cc
    p0[("BC_COR", "tc1")] = (0.05, 0.005, 0.5, "lin")  # wider range for robustness
    p0[("BC_COR", "tc2")] = (0.01, 0.01, 0.5, "lin")
    set_params(config, p0)

    # Use global optimization (differential evolution + local refinement)
    config_opt, err, param_history, obj_history = optimize_zero_d(
        config, p0, data, verbose=verb
    )
    # print_params(config_opt, p0)
    # plot_convergence(animal, study, p0, param_history, obj_history)

    return config_opt, p0, err


def main():
    # animals = [8]  # initial
    animals = [8, 10, 15, 16]  # clean
    # animals = [6, 7, 8, 10, 14, 15, 16] # all
    studies = [
        "baseline",
        "mild_sten",
        "mild_sten_dob",
        "mod_sten",
        "mod_sten_dob",
    ]

    # Collect optimized parameters from all animals
    all_animals_optimized = {}

    for animal in animals:
        optimized = process(animal, studies)
        if optimized:
            all_animals_optimized[animal] = optimized

    # Plot multi-animal comparison
    if len(all_animals_optimized) > 1:
        plot_parameters_multi(all_animals_optimized, studies)


def process(animal, studies):
    optimized = defaultdict(dict)
    data = {}
    config = {}
    err = {}

    for study in studies:
        dat = read_and_smooth_data(animal, study)
        if dat != {}:
            data[study] = dat

    plot_data(animal, data)

    for study in data.keys():
        print(f"Estimating {study}...")
        config[study], p0, err = estimate(data[study], animal, study, verb=1)

        # Extract optimized parameters from p0 (resistances and time constants)
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

        # Extract computed parameters (Ra2_min/max, Ca, Cc are computed and stored in config)
        for bc in config[study][str_bc]:
            if bc["bc_name"] == "BC_COR":
                if "Ra2_min" in bc["bc_values"]:
                    optimized[study][("BC_COR", "Ra2_min")] = bc["bc_values"]["Ra2_min"]
                if "Ra2_max" in bc["bc_values"]:
                    optimized[study][("BC_COR", "Ra2_max")] = bc["bc_values"]["Ra2_max"]
                if "Ca" in bc["bc_values"]:
                    optimized[study][("BC_COR", "Ca")] = bc["bc_values"]["Ca"]
                if "Cc" in bc["bc_values"]:
                    optimized[study][("BC_COR", "Cc")] = bc["bc_values"]["Cc"]

        # Compute derived parameters for tracking/plotting
        # Ra2 (geometric mean) and ratio_Ra2 from Ra2_min and Ra2_max
        if ("BC_COR", "Ra2_min") in optimized[study] and ("BC_COR", "Ra2_max") in optimized[study]:
            Ra2_min = optimized[study][("BC_COR", "Ra2_min")]
            Ra2_max = optimized[study][("BC_COR", "Ra2_max")]
            Ra2 = np.sqrt(Ra2_min * Ra2_max)  # geometric mean
            ratio_Ra2 = Ra2_max / Ra2_min
            optimized[study][("BC_COR", "Ra2")] = Ra2
            optimized[study][("BC_COR", "ratio_Ra2")] = ratio_Ra2

        # Time constants from R and C values
        if ("BC_COR", "Ra1") in optimized[study] and ("BC_COR", "Ca") in optimized[study]:
            tc1 = optimized[study][("BC_COR", "Ra1")] * optimized[study][("BC_COR", "Ca")]
            optimized[study][("BC_COR", "tc1")] = tc1

        if ("BC_COR", "Rv1") in optimized[study] and ("BC_COR", "Cc") in optimized[study]:
            tc2 = optimized[study][("BC_COR", "Rv1")] * optimized[study][("BC_COR", "Cc")]
            optimized[study][("BC_COR", "tc2")] = tc2

        optimized[study][("global", "residual")] = err

    plot_results(animal, config, data)
    plot_parameters(animal, optimized)

    return dict(optimized)


if __name__ == "__main__":
    main()
