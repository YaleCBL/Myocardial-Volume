import os
import json
import numpy as np
from scipy.fft import fft, ifft, fftfreq

str_val = "zero_d_element_values"
str_bc = "boundary_conditions"
str_param = "simulation_parameters"
str_time = "number_of_time_pts_per_cardiac_cycle"

unit_choices = ["cgs", "mmHg", "paper"]
units = unit_choices[2]
mmHg_to_Ba = 1333.22
Ba_to_mmHg = 1 / mmHg_to_Ba

# Reads in json file with given name and returns the object containing all information in it 
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
def set_params(config, p, x=None):
    out = []
    # Loops through all the parameters in the dict p 
    for i, (id, k) in enumerate(p.keys()):
        pval, pmin, pmax = p[(id, k)]
        
        # if x is not empty then it is the optmized value gathered after the least square optimization 
        if x is not None:
            # sets xval to ith parameter in the p dictionary 
            xval = x[i]
            # rearranges the pval, pmin, pmax depending on the value of x 
            # as the limits causes x to be too small or too large so manually overwrites the values 
            if xval > 100:
                pval = pmax
            elif xval < -100:
                pval = pmin
            else:
                # Inverse of log function used for the bounds 
                pval = pmin + (pmax - pmin) * 1 / (1 / np.exp(xval) + 1)
        # Sets the output to the pvals that were modified/added to config 
        out += [pval]

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
    return out

# Simulate bounds for the parameters bounds using log function, as they get closer to the bounds they start changing less and less so the 
# simulation avoids those values 
def get_params(p):
    out = []
    for k in p.keys():
        pval, pmin, pmax = p[k]
        out += [np.log((pval - pmin) / (pmax - pmin))]
    return out

def convert_units(k, val, units):
    if k == "R":
        name = "Resistance"
    elif k == "C":
        name = "Capacitance"
    elif k == "P":
        name = "Pressure"
    elif k == "L":
        name = "Inductance"
    elif k == "t":
        name = "Time constant"
        valu = val
        unit = "s"
    else:
        raise ValueError(f"Unknown name {k}")
    
    if units == "cgs":
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
        val, unit, _ = convert_units(k[0], val, units)
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