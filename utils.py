import os
import json
import pdb
import numpy as np
from scipy.fft import fft, ifft, fftfreq

str_val = "zero_d_element_values"
bc_val = "bc_values"
str_bc = "boundary_conditions"
str_param = "simulation_parameters"
str_time = "number_of_time_pts_per_cardiac_cycle"

unit_choices = ["cgs", "mmHg", "paper"]
units = unit_choices[0]
mmHg_to_Ba = 1333.22
Ba_to_mmHg = 1 / mmHg_to_Ba


# Reads in json file with given name and returns the object containing all information in it
# Parameter mapping functions
def get_param_map(map_type):
    if map_type == "log":
        return np.log, np.exp
    elif map_type == "lin":
        return lambda x: x, lambda x: x
    else:
        raise ValueError(f"Unknown map type: {map_type}. Use 'log' or 'lin'")


def read_config(fname):
    with open(fname, "r") as f:
        return json.load(f)


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

# Rewrites the passed config object to include whatever parameters are in p
# Either the boundary conditions for the time varying flow and pressures are inputted or the
# new values for the resistors and capacitors are inputted after they have been modified
# Time constants (tc1, tc2) are converted to capacitances using C = tau / R
def set_params(config, p, x=None):
    out = []
    # Store parameter values for later computation of capacitances from time constants
    param_values = {}
    time_constants = {}
    ra2_params = {}  # Store Ra2 and _ratio_Ra2 for later computation

    # Loops through all the parameters in the dict p
    for i, (id, k) in enumerate(p.keys()):
        param_tuple = p[(id, k)]
        pval, _, _, map_type = param_tuple

        # if x is not empty then it is the optmized value gathered after the least square optimization
        if x is not None:
            # Get the inverse map function for this parameter
            _, inverse_map = get_param_map(map_type)
            # Apply the inverse map to convert from optimization space back to parameter space
            pval = inverse_map(x[i])
        # Sets the output to the pvals that were modified/added to config
        out += [pval]

        # Store parameter value for later use
        param_values[(id, k)] = pval

        # Check if this is a time constant (starts with 'tc' or 'tau')
        if k.startswith('tc') or k.startswith('tau'):
            time_constants[(id, k)] = pval
        # Check if this is Ra2 or _ratio_Ra2 (internal parameters)
        elif k == 'Ra2' or k == '_ratio_Ra2':
            ra2_params[(id, k)] = pval
        else:
            # sets the parameter specifically checks if they are boundary conditions or vessel parameters
            if "BC" in id:
                # loops through all the boundary conditions
                for bc in config[str_bc]:
                    # checks if any of them match the id in the parameters dictionary
                    if bc["bc_name"] == id:
                        # if the parameter is the P (pressure) then it makes it constant
                        if k == "P" and isinstance(pval, float):
                            bc["bc_values"][k] = [pval, pval]
                            bc["bc_values"]["t"] = [0, 0.737]
                        # if the parameter is anything else then it updates that parameter's value with whatever was passed as pval
                        else:
                            bc["bc_values"][k] = pval
            else:
                # Similarly for the vessels just with different names
                for vs in config["vessels"]:
                    if vs["vessel_name"] == id:
                        vs[str_val][k] = pval

    # Compute Ra2_min and Ra2_max from Ra2 and _ratio_Ra2
    # Ra2 = geometric mean = sqrt(Ra2_min * Ra2_max)
    # _ratio_Ra2 = Ra2_max / Ra2_min
    # => Ra2_min = Ra2 / sqrt(_ratio_Ra2)
    # => Ra2_max = Ra2 * sqrt(_ratio_Ra2)
    for id in set([id for (id, k) in ra2_params.keys()]):
        if (id, 'Ra2') in ra2_params and (id, '_ratio_Ra2') in ra2_params:
            Ra2 = ra2_params[(id, 'Ra2')]
            _ratio_Ra2 = ra2_params[(id, '_ratio_Ra2')]
            sqrt_ratio = np.sqrt(_ratio_Ra2)

            Ra2_min = Ra2 / sqrt_ratio
            Ra2_max = Ra2 * sqrt_ratio

            # Store in config
            for bc in config[str_bc]:
                if bc["bc_name"] == id:
                    bc["bc_values"]['Ra2_min'] = Ra2_min
                    bc["bc_values"]['Ra2_max'] = Ra2_max

    # Compute capacitances from time constants and resistances
    for (id, tc_name), tc_value in time_constants.items():
        # Compute and store corresponding capacitance in config
        if tc_name == 'tc1':
            # tc1 = Ra1 * Ca
            if (id, 'Ra1') in param_values:
                Ca = tc_value / param_values[(id, 'Ra1')]
                # Set Ca in config
                for bc in config[str_bc]:
                    if bc["bc_name"] == id:
                        bc["bc_values"]['Ca'] = Ca
        elif tc_name == 'tc2':
            # tc2 = Rv1 * Cc
            if (id, 'Rv1') in param_values:
                Cc = tc_value / param_values[(id, 'Rv1')]
                # Set Cc in config
                for bc in config[str_bc]:
                    if bc["bc_name"] == id:
                        bc["bc_values"]['Cc'] = Cc
    return out


# Map parameters to optimization space using the specified mapping (log or linear)
def get_params(p):
    out = []
    for k in p.keys():
        param_tuple = p[k]
        # Handle both old format (val, min, max) and new format (val, min, max, map_type)
        if len(param_tuple) == 4:
            pval, _, _, map_type = param_tuple
        else:
            pval, _, _ = param_tuple
            map_type = "log"  # default to log for backward compatibility

        # Get the forward map function for this parameter
        forward_map, _ = get_param_map(map_type)
        # Apply the forward map to convert from parameter space to optimization space
        out += [forward_map(pval)]
    return out


def convert_units(k, val, units):
    unit = ""
    if k[0] == "_":
        name = "Ratio"
        valu = val
        unit = "-"
    elif k[0] == "R":
        name = "Resistance"
    elif k[0] == "C":
        name = "Capacitance"
    elif k[0] == "P":
        name = "Pressure"
    elif k[0] == "L":
        name = "Inductance"
    elif k[0] == "t":
        name = "Time constant"
        valu = val
        unit = "s"
    elif k[0] == "T":
        name = "Time"
        valu = val
        unit = "s"
    elif k[0] == "r":
        name = "Residual"
        valu = val
        unit = "-"
    else:
        raise ValueError(f"Unknown name {k}")

    if units == "cgs":
        valu = val
        if unit == "":
            unit = "cgs"
    elif units == "mmHg":
        if k == "R":
            valu = val * Ba_to_mmHg
            unit = "mmHg*s/ml"
        elif k == "C":
            valu = val * 1 / Ba_to_mmHg
            unit = "ml/mmHg"
        elif k == "P":
            valu = val * Ba_to_mmHg
            unit = "mmHg"
    elif units == "paper":
        if k == "R":
            valu = val * 1e-3
            unit = "10^3 dynes*s/cm^5"
        elif k == "C":
            valu = val * 1e6
            unit = "10^-6 cm^5/dynes"
        elif k == "P":
            valu = val * Ba_to_mmHg
            unit = "mmHg"
    else:
        raise ValueError(f"Unknown units {units}")

    return valu, unit, name


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
        val, unit, _ = convert_units(k, val, units)
        str += id + " " + k[0] + f" {val:.1e} " + unit + "\n"
    print(str)


# smooths the data by doing a fourier transform to get rid of certain frequencies
def smooth(t, ti, qt, cutoff=10):
    N = len(qt)
    freqs = fftfreq(N, d=t[1] - t[0])
    q_fft = fft(qt)
    q_fft[np.abs(freqs) > cutoff] = 0
    qs = np.real(ifft(q_fft))
    dqs_fft = q_fft * (2j * np.pi * freqs)
    dqs = np.real(ifft(dqs_fft))
    # Linearly interpolates the results to upscale the data set and outputs both the data set and it's derivatives
    return np.interp(ti, t, qs), np.interp(ti, t, dqs)
