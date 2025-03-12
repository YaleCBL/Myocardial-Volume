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
    forward = {
        "boundary_conditions": [
            {
                "bc_name": "pa",
                "bc_type": bc_type,
                "bc_values": {"t": t.tolist(), bc_val: pa.tolist()},
            },
            {
                "bc_name": "gnd",
                "bc_type": "RCR",
                "bc_values": {"C": 0.0, "Pd": 0.0, "Rd": 0.0, "Rp": 0.0},
            },
        ],
        "simulation_parameters": {
            "number_of_cardiac_cycles": 200,
            "number_of_time_pts_per_cardiac_cycle": 1001,
            "output_all_cycles": False,
            "output_derivative": True,
            "output_variable_based": True,
        },
        "vessels": [
            {
                "boundary_conditions": {"inlet": "pa", "outlet": "gnd"},
                "vessel_id": 0,
                "vessel_length": 1.0,
                "vessel_name": "myocardium",
                "zero_d_element_type": "BloodVessel",
                "zero_d_element_values": {
                    "C": 0.0,
                    "L": 0.0,
                    "R_poiseuille": 0.0,
                    "stenosis_coefficient": 0.0,
                },
            }
        ],
    }
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


def optimize_zero_d(config, reference):
    param_keys = ["R_poiseuille", "stenosis_coefficient"] # "C", "L", "stenosis_coefficient"]
    # param_keys = list(config["vessels"][0]["zero_d_element_values"].keys())
    p0 = [config["vessels"][0]["zero_d_element_values"][k] for k in param_keys]

    def cost_function(p):
        for i, k in enumerate(param_keys):
            config["vessels"][0]["zero_d_element_values"][k] = p[i]
        sim = pysvzerod.simulate(config)
        cost = []
        for k in reference.keys():
            for name, ref in reference[k].items():
                y = sim[sim["name"] == name][k]
                cost += [np.array(y) - np.array(ref)]
        return np.array(cost).flatten()

    res = least_squares(
        cost_function, np.ones(len(param_keys)), jac="2-point", method="lm"
    )
    for i, k in enumerate(param_keys):
        config["vessels"][0]["zero_d_element_values"][k] = res.x[i]

    # sim = pysvzerod.simulate(config)
    # rname = "pressure:pa:myocardium"
    # pa = sim[sim["name"] == rname]["y"]
    # t = sim[sim["name"] == rname]["time"]
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(t, reference["y"][rname], "b", label="measured")
    # ax1.plot(t, pa, "y", label="simulated")
    # ax2.plot(
    #     config["boundary_conditions"][0]["bc_values"]["t"],
    #     config["boundary_conditions"][0]["bc_values"]["Q"],
    #     "r",
    #     label="inflow",
    # )
    # ax1.legend()
    # ax2.legend()
    # plt.show()
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


def estimate(fname):
    t, pa, pv, v, flow = read_data(fname)
    forward = create_forward_config(t, flow, "q")
    time_points = forward["simulation_parameters"][
        "number_of_time_pts_per_cardiac_cycle"
    ]
    t_solve = np.linspace(t[0], t[-1], time_points)
    pa_solve = np.interp(t_solve, t, pa)
    reference = {"y": {"pressure:pa:myocardium": pa_solve.tolist()}}

    optimized_config = optimize_zero_d(forward, reference)
    print(optimized_config["vessels"][0]["zero_d_element_values"])


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
