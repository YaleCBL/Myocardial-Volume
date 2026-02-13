#!/usr/bin/env python3
"""Main processing script for coronary model parameter estimation."""

import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from scipy.integrate import cumulative_trapezoid

from utils import (
    read_config, mmHg_to_Ba, str_val, bc_val, str_bc, str_param, str_time,
    smooth, set_params, get_param_map
)
from simulation import get_sim_qv
from optimization import optimize
from plotting import (
    plot_data, plot_results, plot_parameters, plot_parameters_multi,
    plot_combined_results, plot_circuit_diagram
)

# Model configuration
MODEL_NAME = "coronary_varres_time"
NODE_IN = "BC_AT:BV_prox"


def get_name(animal):
    """Get animal name string."""
    return f"DSEA{animal:02d}"


def read_data(animal, study):
    """Read measurement data from CSV file."""
    csv_file = os.path.join("data", f"{get_name(animal)}_{study}.csv")
    if not os.path.isfile(csv_file):
        print(f"Data file {csv_file} not found.")
        return {}

    df = pd.read_csv(csv_file)
    data = {"t": df["t abs [s]"].to_numpy()}

    for field in df.keys():
        if "mmHg" in field:
            df[field.replace("mmHg", "Ba")] = df[field] * mmHg_to_Ba

    data["pat"] = df["AoP [Ba]"].to_numpy()
    data["pven"] = df["LVP [Ba]"].to_numpy()
    data["vmyo"] = df["tissue vol ischemic [ml]"].to_numpy()
    data["qlad"] = df["LAD Flow [ml/s]"].to_numpy()
    data["qmyo"] = np.gradient(data["vmyo"], data["t"])
    data["vlad"] = cumulative_trapezoid(data["qlad"], data["t"], initial=0)

    # Scale LAD flow using microsphere data
    csv_file = os.path.join("data", f"{get_name(animal)}_microsphere.csv")
    df = pd.read_csv(csv_file)
    data["dvcycle"] = df[f"{study} ischemic flow [ml/min/g]"].to_numpy()
    data["dvcycle"] /= 60.0
    data["dvcycle"] *= data["t"][-1]

    dvlad = data["vlad"].max() - data["vlad"].min()
    scale_vol = data["dvcycle"] / dvlad
    data["qlad"] *= scale_vol
    data["vlad"] *= scale_vol

    return data


def smooth_data(data_o, nt):
    """Smooth and interpolate data to simulation time points."""
    data_s = {}
    data_s["t"] = np.linspace(data_o["t"][0], data_o["t"][-1], nt)

    for field in ["pat", "pven", "qlad"]:
        data_s[field], _ = smooth(data_o["t"], data_s["t"], data_o[field], cutoff=15)

    data_s["vmyo"], data_s["qmyo"] = smooth(data_o["t"], data_s["t"], data_o["vmyo"], cutoff=15)
    data_s["vmyo"] -= data_s["vmyo"][0]
    data_s["vlad"] = cumulative_trapezoid(data_s["qlad"], data_s["t"], initial=0)

    return data_s


def setup_model(data):
    """Set up model configuration and parameters.

    Returns:
        tuple: (config, p0) - model config and parameter dict
    """
    config = read_config(f"models/{MODEL_NAME}.json")
    config[str_param][str_time] = len(data["s"]["t"])

    # Set boundary conditions
    pini = {}
    pini[("BC_AT", "t")] = (data["s"]["t"].tolist(), None, None, "lin")
    pini[("BC_AT", "P")] = (data["s"]["pat"].tolist(), None, None, "lin")
    pini[("BC_COR", "t")] = (data["s"]["t"].tolist(), None, None, "lin")

    # Compute Pim components
    pven = np.array(data["s"]["pven"])
    t = np.array(data["s"]["t"])
    dt = t[1] - t[0]
    dpdt = np.gradient(pven, dt)
    dpdt_norm = dpdt * (pven.max() / (np.abs(dpdt).max() + 1e-10))
    dpdt_pos = np.maximum(dpdt_norm, 0)

    pini[("BC_COR", "Pim")] = (pven.tolist(), None, None, "lin")
    pini[("BC_COR", "_Pim_base")] = (pven.tolist(), None, None, "lin")
    pini[("BC_COR", "_Pim_dpdt_base")] = (dpdt_pos.tolist(), None, None, "lin")

    # Set timing constants from data
    t_raw = data["o"]["t"]
    V = data["o"]["vmyo"]
    t_cycle = data["s"]["t"][-1]

    t_v_min = t_raw[np.argmin(V)]
    v_range = V.max() - V.min()
    v_threshold = V.min() + 0.9 * v_range
    t_after_min = t_raw[t_raw > t_v_min]
    v_after_min = V[t_raw > t_v_min]

    if len(t_after_min) > 0:
        idx_relaxed = np.where(v_after_min >= v_threshold)[0]
        if len(idx_relaxed) > 0:
            t_v_dia = t_after_min[idx_relaxed[0]] - t_v_min
        else:
            t_v_dia = t_raw[-1] - t_v_min
    else:
        t_v_dia = 0.3

    print(f"T_vc: {t_v_min:.3f}s, T_vr: {t_v_dia:.3f}s")
    pini[("BC_COR", "T_vc")] = (t_v_min, None, None, "lin")
    pini[("BC_COR", "T_vr")] = (t_v_dia, None, None, "lin")

    set_params(config, pini)

    # Set optimization parameters (val, min, max, map_type)
    p0 = OrderedDict()
    p0[("BV_prox", "L")] = (1e3, 1e1, 1e5, "log")
    p0[("BV_prox", "R_poiseuille")] = (1e5, 1e3, 1e7, "log")
    p0[("BC_COR", "Ra1")] = (1e6, 1e4, 1e8, "log")
    p0[("BC_COR", "Ra2")] = (1e6, 1e4, 1e8, "log")
    p0[("BC_COR", "_ratio_Ra2")] = (3.0, 1.5, 8.0, "lin")
    p0[("BC_COR", "Rv1")] = (1e5, 1e3, 1e7, "log")
    p0[("BC_COR", "tc1")] = (0.15, 0.05, 0.5, "lin")
    p0[("BC_COR", "tc2")] = (0.3, 0.1, 2.0, "lin")

    T_vc_init = np.clip(t_v_min, 0.05, 0.6 * t_cycle)
    T_vr_init = np.clip(t_v_dia, 0.05, 0.6 * t_cycle)
    p0[("BC_COR", "T_vc")] = (T_vc_init, 0.05, 0.6 * t_cycle, "lin")
    p0[("BC_COR", "T_vr")] = (T_vr_init, 0.05, 0.6 * t_cycle, "lin")
    p0[("BC_COR", "_Pim_scale")] = (1.5, 0.8, 2.5, "lin")
    p0[("BC_COR", "_Pim_dpdt")] = (0.5, 0.0, 1.5, "lin")

    set_params(config, p0)

    return config, p0


def extract_optimized_params(config, p0, optimized, study, err):
    """Extract optimized parameters from config."""
    params = [opt[0] for opt in p0]
    values = [opt[1] for opt in p0]

    for param in params:
        for val in values:
            for vessel in config["vessels"]:
                if vessel["vessel_name"] == param and val in vessel[str_val]:
                    optimized[study][(param, val)] = vessel[str_val][val]
            for bc in config[str_bc]:
                if bc["bc_name"] == param and val in bc["bc_values"]:
                    optimized[study][(param, val)] = bc["bc_values"][val]

    # Extract computed parameters
    for bc in config[str_bc]:
        if bc["bc_name"] == "BC_COR":
            for key in ["Ra2_min", "Ra2_max", "Ca", "Cc"]:
                if key in bc["bc_values"]:
                    optimized[study][("BC_COR", key)] = bc["bc_values"][key]

    optimized[study][("global", "residual")] = err


def process_animal(animal, studies, weight_flow=2.0, weight_volume=0.5, weight_flow_min=5.0):
    """Process all studies for one animal."""
    optimized = defaultdict(dict)
    data = {}
    config = {}

    # Read and smooth data
    for study in studies:
        print(f"{get_name(animal)}_{study}")
        data_o = read_data(animal, study)
        if data_o:
            data[study] = {"o": data_o, "s": smooth_data(data_o, 201)}

    if not data:
        return None, None, None

    plot_data(animal, data, get_name)

    # Fit each study
    for study in data.keys():
        print(f"Estimating {study}...")
        cfg, p0 = setup_model(data[study])

        cfg, err = optimize(cfg, p0, data[study], NODE_IN, verbose=1,
                           weight_flow=weight_flow, weight_volume=weight_volume,
                           weight_flow_min=weight_flow_min)

        config[study] = cfg
        extract_optimized_params(cfg, p0, optimized, study, err)

        # Save config
        with open(f"results/{get_name(animal)}_{study}_{MODEL_NAME}.json", "w") as f:
            json.dump(cfg, f, indent=2)

    plot_results(animal, config, data, get_sim_qv, NODE_IN, get_name, MODEL_NAME)
    plot_parameters(animal, optimized, get_name, MODEL_NAME)

    return dict(optimized), data, config


def main():
    """Main entry point."""
    animals = [8, 10, 15, 16]
    studies = ["baseline", "mild_sten", "mild_sten_dob", "mod_sten", "mod_sten_dob"]

    # Optimization weights
    weight_flow = 2.0
    weight_volume = 0.5
    weight_flow_min = 5.0

    all_optimized = {}
    all_data = {}
    all_config = {}

    for animal in animals:
        result = process_animal(animal, studies, weight_flow, weight_volume, weight_flow_min)
        if result[0]:
            optimized, data, config = result
            all_optimized[animal] = optimized
            all_data[animal] = data
            all_config[animal] = config

    # Generate summary plots
    if len(all_optimized) > 1:
        plot_parameters_multi(all_optimized, studies, MODEL_NAME)

    if all_data:
        plot_combined_results(all_data, all_config, MODEL_NAME, get_sim_qv, NODE_IN, get_name)

    plot_circuit_diagram(MODEL_NAME)


if __name__ == "__main__":
    main()
