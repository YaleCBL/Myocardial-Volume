#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import pysvzerod
import copy
import pdb
import numpy as np
import json
from scipy.interpolate import interp1d

df = pd.read_csv("DSEA08_baseline.csv")
# print(df.head())
# print(df.info())

t = df["t abs [s]"].to_numpy()
pa = df["AoP [mmHg]"].to_numpy()
pv = df["LVP [mmHg]"].to_numpy()
v = df["tissue vol ischemic [ml]"].to_numpy()
flow = df["LAD Flow [mL/s]"].to_numpy()
q = np.gradient(v, t)
dpa = np.gradient(pa, t)

# plt.plot(t, dpa, label='v')
# plt.plot(pa, v, "o", label="dv/dt")
# plt.plot(t, q, label='dv/dt')
# plt.plot(t, pa-pv, label='p')
# plt.plot(pa, v, 'o', label='pa')
# plt.plot(t, flow, label="flow")
# plt.plot(t, q, label="q")
# plt.plot(t, pa, label="p")
# plt.plot(t0d, q0d, ":", label="0d")
# plt.plot(t0d, v0d, ":", label="0d")
# plt.plot(pa, v0d_resampled, "o", label='v')
# plt.legend()
# plt.show()

forward = {
    "boundary_conditions": [
        {
            "bc_name": "pa",
            "bc_type": "PRESSURE",
            "bc_values": {"t": t.tolist(), "P": pa.tolist()},
        },
        {
            "bc_name": "gnd",
            "bc_type": "RCR",
            "bc_values": {"C": 0.0, "Pd": 0.0, "Rd": 0.0, "Rp": 0.0},
        },
    ],
    "simulation_parameters": {
        "number_of_cardiac_cycles": 10,
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
                "C": 0.0001,
                "L": 1.0,
                "R_poiseuille": 100.0,
                "stenosis_coefficient": 0.0,
            },
        }
    ],
}
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


# solver = pysvzerod.Solver(forward)
res_forward = pysvzerod.simulate(forward)

names = ["pa:myocardium", "myocardium:gnd"]

inverse["y"] = {}
inverse["dy"] = {}
for n in names:
    for f in ["flow", "pressure"]:
        key = f + ":" + n
        res = res_forward[res_forward["name"] == key]
        inverse["y"][key] = res["y"].tolist()
        inverse["dy"][key] = res["ydot"].tolist()

res_inverse = pysvzerod.calibrate(inverse)

print(res_inverse["vessels"][0]["zero_d_element_values"])
# solver = pysvzerod.
# pdb.set_trace()

# with open("forward.json", "w") as f:
#     json.dump(forward, f, indent=2)
# with open("inverse.json", "w") as f:
#     json.dump(inverse, f, indent=2)


# dt = t0d[1] - t0d[0]
# v0d = v[0] + np.cumsum(q0d) * dt


# f = interp1d(t0d, v0d, fill_value="extrapolate")
# v0d_resampled = f(t)

# flow_scale = np.mean(q) / np.mean(flow)
# pdb.set_trace()
