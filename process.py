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

# Alternative model with inlet Windkessel for pressure pulsatility
model_wk = "coronary_varres_wk"
n_in_wk = "BC_AT:R_ao"

# Pressure-driven model: prescribe aortic pressure, fit flow + volume
model_pdriven = "coronary_varres_pdriven"

# Intramyocardial pump model: explicit pressure source representing myocardial compression
model_im_pump = "coronary_im_pump"
n_in_im = "BC_AO:R_epi"  # inlet node for IM pump model

# Pressure-driven model with inductance for time delay between pressure and flow
# Uses standard CORONARY BC (not CORONARY_VAR_RES which is not in standard pysvzerod)
model_inductance = "coronary_inductance"
n_in_inductance = "BC_AT:BV_prox"  # inlet node for inductance model

# Time-varying resistance model: combines inductance with CORONARY_VAR_RES BC
# The microvascular resistance varies during the cardiac cycle:
# - Contraction (0 -> T_vc): Ra2 increases from Ra2_min to Ra2_max
# - Relaxation (T_vc -> T_vc + T_vr): Ra2 decreases from Ra2_max to Ra2_min
# - Rest (> T_vc + T_vr): Ra2 stays at Ra2_min
# This creates phase shifts in flow/volume that can address timing discrepancies
model_varres_time = "coronary_varres_time"
n_in_varres_time = "BC_AT:BV_prox"  # inlet node for time-varying resistance model

# Backpressure model: outlet pressure = scaled Pim creates direct flow opposition
# Simpler than CORONARY BC, allows stronger systolic flow reduction
model_backpressure = "coronary_backpressure"
n_in_backpressure = "BC_AO:V_prox"  # inlet node for backpressure model


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


def get_sim_pv(config, node_in=None):
    """Run simulation once and extract both pressure and volume.

    Args:
        config: svZeroDSolver configuration
        node_in: Node name for pressure extraction (default: uses global n_in)

    Returns:
        tuple: (pressure, volume) arrays
    """
    if node_in is None:
        node_in = n_in

    try:
        sim = pysvzerod.simulate(config)

        # Extract pressure
        p_sim = sim[sim["name"] == "pressure:" + node_in]["y"].to_numpy()
        if not p_sim.size:
            raise ValueError(f"Pressure result not found at pressure:{node_in}")

        # Extract volume
        if "CORONARY" in config["boundary_conditions"][1]["bc_type"]:
            v_sim = sim[sim["name"] == "volume_im:BC_COR"]["y"].to_numpy()
            if not v_sim.size:
                raise ValueError("Volume result not found at volume_im:BC_COR")
        else:
            q_sim = sim[sim["name"] == "flow:" + node_in]["y"].to_numpy()
            if not q_sim.size:
                raise ValueError(f"Flow result not found at flow:{node_in}")
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


def get_sim_qv(config, node_in=None):
    """Run simulation once and extract both flow and volume.

    For pressure-driven models where we prescribe aortic pressure and fit flow + volume.

    Args:
        config: svZeroDSolver configuration
        node_in: Node name for flow extraction (default: uses global n_in)

    Returns:
        tuple: (flow, volume) arrays
    """
    if node_in is None:
        node_in = n_in

    try:
        sim = pysvzerod.simulate(config)

        # Extract flow
        q_sim = sim[sim["name"] == "flow:" + node_in]["y"].to_numpy()
        if not q_sim.size:
            raise ValueError(f"Flow result not found at flow:{node_in}")

        # Extract volume
        if "CORONARY" in config["boundary_conditions"][1]["bc_type"]:
            v_sim = sim[sim["name"] == "volume_im:BC_COR"]["y"].to_numpy()
            if not v_sim.size:
                raise ValueError("Volume result not found at volume_im:BC_COR")
        else:
            nt = config[str_param][str_time]
            tmax = config["boundary_conditions"][0]["bc_values"]["t"][-1]
            ti = np.linspace(0.0, tmax, nt)
            v_sim = cumulative_trapezoid(q_sim, ti, initial=0)

        v_sim = v_sim - v_sim[0]
        return q_sim, v_sim

    except RuntimeError as e:
        print(f"Simulation failed: {e}")
        nt = config[str_param][str_time]
        return np.zeros(nt), np.zeros(nt)
    except Exception as e:
        print(f"Unexpected error in get_sim_qv: {type(e).__name__}: {e}")
        nt = config[str_param][str_time]
        return np.zeros(nt), np.zeros(nt)


def get_sim_qv_im_pump(config, node_in=None):
    """Run simulation and extract flow and volume for intramyocardial pump model.

    For the IM pump model, myocardial blood volume is computed from mass conservation:
        V = integral of (Q_in - Q_out)

    This measures the net blood stored in the coronary vasculature.

    Args:
        config: svZeroDSolver configuration
        node_in: Node name for flow extraction (default: uses n_in_im)

    Returns:
        tuple: (flow, volume) arrays
    """
    if node_in is None:
        node_in = n_in_im

    try:
        sim = pysvzerod.simulate(config)

        # Extract inlet flow
        q_in = sim[sim["name"] == "flow:" + node_in]["y"].to_numpy()
        if not q_in.size:
            raise ValueError(f"Flow result not found at flow:{node_in}")

        # Extract outlet flow (at outlet BC)
        q_out = sim[sim["name"] == "flow:R_ven:BC_IM"]["y"].to_numpy()
        if not q_out.size:
            raise ValueError("Outlet flow not found at flow:R_ven:BC_IM")

        # Compute volume from mass conservation: V = integral of (Q_in - Q_out)
        nt = config[str_param][str_time]
        tmax = config["boundary_conditions"][0]["bc_values"]["t"][-1]
        ti = np.linspace(0.0, tmax, nt)

        dv = q_in - q_out
        v_sim = cumulative_trapezoid(dv, ti, initial=0)
        v_sim = v_sim - v_sim[0]

        return q_in, v_sim

    except RuntimeError as e:
        print(f"Simulation failed: {e}")
        nt = config[str_param][str_time]
        return np.zeros(nt), np.zeros(nt)


def get_sim_qv_backpressure(config, node_in=None):
    """Run simulation and extract flow and volume for backpressure model.

    For the backpressure model:
    - Inlet: aortic pressure
    - Outlet: scaled intramyocardial pressure (creates back-pressure)
    - Volume computed from inlet flow integration

    Args:
        config: svZeroDSolver configuration
        node_in: Node name for flow extraction (default: uses n_in_backpressure)

    Returns:
        tuple: (flow, volume) arrays
    """
    if node_in is None:
        node_in = n_in_backpressure

    try:
        sim = pysvzerod.simulate(config)

        # Extract inlet flow
        q_in = sim[sim["name"] == "flow:" + node_in]["y"].to_numpy()
        if not q_in.size:
            raise ValueError(f"Flow result not found at flow:{node_in}")

        # Compute volume from inlet flow integration
        nt = config[str_param][str_time]
        tmax = config["boundary_conditions"][0]["bc_values"]["t"][-1]
        ti = np.linspace(0.0, tmax, nt)

        v_sim = cumulative_trapezoid(q_in, ti, initial=0)
        v_sim = v_sim - v_sim[0]

        return q_in, v_sim

    except RuntimeError as e:
        print(f"Simulation failed: {e}")
        nt = config[str_param][str_time]
        return np.zeros(nt), np.zeros(nt)
    except Exception as e:
        print(f"Unexpected error in get_sim_qv_backpressure: {type(e).__name__}: {e}")
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


def align_data(data_s, method="xcorr", reference="pven", target="vmyo", max_shift_fraction=0.2):
    """Align signals in time to correct for measurement delays.

    The coronary flow and myocardial volume should be synchronized with the
    cardiac cycle (LV pressure). This function shifts signals to align them.

    Args:
        data_s: Dictionary with smoothed data (modified in place)
        method: Alignment method:
            - "xcorr": Cross-correlation to find optimal lag
            - "landmarks": Align based on physiological landmarks
            - "manual": Use a fixed shift (set via max_shift_fraction as actual shift)
        reference: Signal to use as reference (typically "pven" for LV pressure)
        target: Signal to shift (typically "vmyo" or "qlad")
        max_shift_fraction: Maximum shift as fraction of cardiac cycle (for xcorr)
                           or actual shift in seconds (for manual)

    Returns:
        shift_samples: Number of samples shifted (positive = target delayed)
    """
    t = data_s["t"]
    dt = t[1] - t[0]
    T = t[-1] - t[0]  # cardiac cycle length
    n = len(t)

    if method == "xcorr":
        # Cross-correlation based alignment
        ref = data_s[reference]
        tgt = data_s[target]

        # Normalize signals
        ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-10)
        tgt_norm = (tgt - np.mean(tgt)) / (np.std(tgt) + 1e-10)

        # Compute cross-correlation
        max_shift = int(max_shift_fraction * n)
        correlations = []
        shifts = range(-max_shift, max_shift + 1)

        for shift in shifts:
            if shift >= 0:
                corr = np.sum(ref_norm[shift:] * tgt_norm[:n-shift])
            else:
                corr = np.sum(ref_norm[:n+shift] * tgt_norm[-shift:])
            correlations.append(corr)

        # Find optimal shift (maximum correlation)
        best_idx = np.argmax(correlations)
        shift_samples = shifts[best_idx]

    elif method == "landmarks":
        # Align based on physiological landmarks:
        # - Peak LV pressure should coincide with minimum myocardial volume
        # - (systole = myocardium compressed = minimum blood volume)
        ref = data_s[reference]
        tgt = data_s[target]

        # Find peak of reference (e.g., peak LV pressure)
        ref_peak_idx = np.argmax(ref)

        # Find minimum of target (e.g., minimum myocardial volume)
        tgt_min_idx = np.argmin(tgt)

        # Shift target so its minimum aligns with reference peak
        shift_samples = ref_peak_idx - tgt_min_idx

        # Limit shift to max_shift_fraction of cycle
        max_shift = int(max_shift_fraction * n)
        shift_samples = np.clip(shift_samples, -max_shift, max_shift)

    elif method == "manual":
        # Manual shift specified in seconds
        shift_samples = int(max_shift_fraction / dt)

    else:
        raise ValueError(f"Unknown alignment method: {method}")

    # Apply shift to target signal (circular shift for periodic signal)
    if shift_samples != 0:
        data_s[target] = np.roll(data_s[target], shift_samples)

        # Also shift related signals if they exist
        if target == "vmyo" and "qmyo" in data_s:
            data_s["qmyo"] = np.roll(data_s["qmyo"], shift_samples)
        if target == "qlad" and "vlad" in data_s:
            data_s["vlad"] = np.roll(data_s["vlad"], shift_samples)

    print(f"Time alignment: shifted {target} by {shift_samples} samples ({shift_samples * dt * 1000:.1f} ms)")

    return shift_samples


def align_flow_to_pressure(data_s, align_method="landmarks"):
    """Align LAD flow and myocardial volume to LV pressure.

    Physiological expectation:
    - During systole (high LV pressure): coronary flow decreases, myocardial volume decreases
    - During diastole (low LV pressure): coronary flow increases, myocardial volume increases

    Args:
        data_s: Dictionary with smoothed data (modified in place)
        align_method: "landmarks", "xcorr", or "none"

    Returns:
        shifts: Dictionary with shift values for each signal
    """
    if align_method == "none":
        return {"vmyo": 0, "qlad": 0}

    shifts = {}

    # Align myocardial volume: minimum should occur near peak LV pressure
    shifts["vmyo"] = align_data(
        data_s,
        method=align_method,
        reference="pven",
        target="vmyo",
        max_shift_fraction=0.15
    )

    # Align LAD flow: minimum should occur near peak LV pressure (systolic compression)
    # Use derivative of volume as reference for better flow alignment
    shifts["qlad"] = align_data(
        data_s,
        method=align_method,
        reference="pven",
        target="qlad",
        max_shift_fraction=0.15
    )

    return shifts


def plot_data(animal, data):
    n_param = len(data.keys())
    _, ax = plt.subplots(
        4, n_param, figsize=(max(n_param * 5, 5), 10), sharex="col", sharey="row"
    )
    # Handle single-column case (ax is 1D array when n_param=1)
    if n_param == 1:
        ax = ax.reshape(-1, 1)
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


def plot_results(animal, config, data, use_pdriven=False, use_im_pump=False, use_inductance=False,
                 use_backpressure=False, use_varres_time=False):
    """Plot simulation results vs measured data.

    Args:
        animal: Animal ID
        config: Dictionary of config per study
        data: Dictionary of data per study
        use_pdriven: If True, pressure-driven mode (prescribe P, fit Q+V)
        use_im_pump: If True, intramyocardial pump model
        use_inductance: If True, inductance model for time delay
        use_backpressure: If True, backpressure model (scaled Pim as outlet)
        use_varres_time: If True, time-varying resistance model
    """
    labels = {
        "qlad": "LAD Flow (scaled) [ml/s]",
        "pat": "Aterial Pressure [mmHg]",
        "pven": "Left-Ventricular Pressure [mmHg]",
        "vmyo": "Myocardial Volume [ml]",
    }

    # Determine which model name to use for file output
    if use_backpressure:
        model_out = model_backpressure
    elif use_varres_time:
        model_out = model_varres_time
    elif use_im_pump:
        model_out = model_im_pump
    elif use_inductance:
        model_out = model_inductance
    elif use_pdriven:
        model_out = model_pdriven
    else:
        model_out = model

    _, axs = plt.subplots(
        len(labels), len(config), figsize=(16, 9), sharex="col", sharey="row"
    )
    if len(config) == 1:
        axs = axs.reshape(-1, 1)
    for j, study in enumerate(config.keys()):
        with open(f"results/{get_name(animal)}_{study}_{model_out}.json", "w") as f:
            json.dump(config[study], f, indent=2)

        ti = config[study]["boundary_conditions"][0]["bc_values"]["t"]
        datm = data[study]["s"]

        dats = OrderedDict()

        if use_backpressure:
            # Backpressure model: flow and volume are simulated
            q_sim, v_sim = get_sim_qv_backpressure(config[study])
            dats["qlad"] = q_sim
            dats["pven"] = datm["pven"]  # Always from data
            dats["pat"] = datm["pat"]    # Prescribed (BC_AO)
            dats["vmyo"] = v_sim
        elif use_varres_time:
            # Time-varying resistance model: flow and volume are simulated
            q_sim, v_sim = get_sim_qv(config[study], node_in=n_in_varres_time)
            dats["qlad"] = q_sim
            dats["pven"] = datm["pven"]  # Always from data
            dats["pat"] = datm["pat"]    # Prescribed (from data)
            dats["vmyo"] = v_sim
        elif use_im_pump:
            # IM pump model: flow and volume are simulated, pressures are prescribed
            q_sim, v_sim = get_sim_qv_im_pump(config[study])
            dats["qlad"] = q_sim
            dats["pven"] = datm["pven"]  # Prescribed (BC_IM)
            dats["pat"] = datm["pat"]    # Prescribed (BC_AO)
            dats["vmyo"] = v_sim
        elif use_inductance:
            # Inductance model: flow is simulated with time delay, pressure is prescribed
            q_sim, v_sim = get_sim_qv(config[study], node_in=n_in_inductance)
            dats["qlad"] = q_sim
            dats["pven"] = datm["pven"]  # Always from data
            dats["pat"] = datm["pat"]    # Prescribed (from data)
            dats["vmyo"] = v_sim
        elif use_pdriven:
            # Pressure-driven: flow is simulated, pressure is prescribed
            q_sim, v_sim = get_sim_qv(config[study])
            dats["qlad"] = q_sim
            dats["pven"] = datm["pven"]  # Always from data
            dats["pat"] = datm["pat"]    # Prescribed (from data)
            dats["vmyo"] = v_sim
        else:
            # Flow-driven: pressure is simulated, flow is prescribed
            p_sim, v_sim = get_sim_pv(config[study])
            dats["qlad"] = datm["qlad"]  # Prescribed (from data)
            dats["pven"] = datm["pven"]  # Always from data
            dats["pat"] = p_sim
            dats["vmyo"] = v_sim

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
    plt.savefig(f"plots/{get_name(animal)}_{model_out}_simulated.pdf")
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
def optimize_zero_d(config, p0, data, verbose=0, weight_pressure=1.0, weight_volume=1.0, node_in=None):
    """Optimize 0D model parameters using least squares.

    Args:
        config: svZeroDSolver configuration
        p0: Parameter dictionary with (val, min, max, map_type) tuples
        data: Dictionary with smoothed data
        verbose: Verbosity level
        weight_pressure: Weight for pressure objective (default 1.0)
        weight_volume: Weight for volume objective (default 1.0)
        node_in: Node name for pressure extraction (default: uses global n_in)
    """
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

        # runs the simulation ONCE and gets both pressure and volume
        p_sim, v_sim = get_sim_pv(config, node_in=node_in)

        # compute objective function for both the pressure and the volume
        # Apply weights to each component
        obj_p = weight_pressure * get_objective(pref, p_sim)
        obj_v = weight_volume * get_objective(vref, v_sim)

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


def optimize_zero_d_pdriven(config, p0, data, verbose=0, weight_flow=1.0, weight_volume=1.0,
                            weight_derivative=0.0, weight_regularization=0.0, weight_flow_min=0.0,
                            node_in=None):
    """Optimize 0D model for pressure-driven case (prescribe P, fit Q + V).

    Args:
        config: svZeroDSolver configuration
        p0: Parameter dictionary with (val, min, max, map_type) tuples
        data: Dictionary with smoothed data
        verbose: Verbosity level
        weight_flow: Weight for flow objective (default 1.0)
        weight_volume: Weight for volume objective (default 1.0)
        weight_derivative: Weight for derivative matching (default 0.0, disabled)
            Higher values penalize loss of pulsatility/oscillations
        weight_regularization: Weight for bound regularization (default 0.0, disabled)
            Penalizes parameters approaching their bounds, pulling toward center
        weight_flow_min: Extra weight on matching flow minimum (default 0.0)
            Specifically targets the systolic flow dip
        node_in: Node name for flow extraction (default: uses global n_in)
    """
    # Reference data: flow and volume from measurements
    qref = data["s"]["qlad"]
    vref = data["s"]["vmyo"]

    # Precompute reference derivatives for derivative matching
    dqref = np.gradient(qref)
    dvref = np.gradient(vref)
    # Normalization factors (avoid division by zero)
    dq_norm = np.std(dqref) + 1e-10
    dv_norm = np.std(dvref) + 1e-10

    # Precompute flow minimum info for flow_min penalty
    q_min_idx = np.argmin(qref)
    q_min_val = qref[q_min_idx]
    q_mean = np.mean(np.abs(qref)) + 1e-10

    # Precompute bound centers for regularization (in transformed space)
    bound_centers = []
    bound_half_widths = []
    for _, pmin, pmax, map_type in p0.values():
        forward_map, _ = get_param_map(map_type)
        t_min = forward_map(pmin)
        t_max = forward_map(pmax)
        bound_centers.append((t_min + t_max) / 2)
        bound_half_widths.append((t_max - t_min) / 2)

    # Track parameter history for convergence plot
    param_history = []
    obj_history = []

    def cost_function(p):
        pset = set_params(config, p0, p)
        param_history.append(pset.copy())

        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")

        # Get flow and volume from simulation
        q_sim, v_sim = get_sim_qv(config, node_in=node_in)

        # Compute objective: fit flow and volume
        obj_q = weight_flow * get_objective(qref, q_sim)
        obj_v = weight_volume * get_objective(vref, v_sim)

        obj = np.concatenate((obj_q, obj_v))

        # Flow minimum penalty: specifically target the systolic flow dip
        # This is critical for capturing the intramyocardial pump effect
        if weight_flow_min > 0:
            q_sim_at_min = q_sim[q_min_idx] if len(q_sim) > q_min_idx else 0
            # Penalty is normalized by mean flow magnitude
            flow_min_penalty = weight_flow_min * (q_min_val - q_sim_at_min) / q_mean
            obj = np.concatenate((obj, [flow_min_penalty]))

        # Derivative matching: penalize differences in signal dynamics
        # This prevents "flat" solutions that average out oscillations
        if weight_derivative > 0:
            dq_sim = np.gradient(q_sim)
            dv_sim = np.gradient(v_sim)

            obj_dq = weight_derivative * (dqref - dq_sim) / dq_norm
            obj_dv = weight_derivative * (dvref - dv_sim) / dv_norm

            obj = np.concatenate((obj, obj_dq, obj_dv))

        # Regularization: penalize parameters approaching bounds
        # Uses normalized distance from center: 0 at center, 1 at bounds
        if weight_regularization > 0:
            reg_penalties = []
            for i, pi in enumerate(p):
                # Normalized distance from center (0 to 1)
                dist = (pi - bound_centers[i]) / bound_half_widths[i]
                # Quadratic penalty that grows toward bounds
                reg_penalties.append(weight_regularization * dist**2)
            obj = np.concatenate((obj, np.array(reg_penalties)))

        obj_history.append(np.linalg.norm(obj))

        if verbose:
            print(f"{np.linalg.norm(obj):.1e}", end="\n")
        return obj

    initial = get_params(p0)
    if verbose:
        for k in p0.keys():
            print(f"{k[1]}", end="\t")
        print("obj", end="\n")

    bounds = []
    for param_tuple in p0.values():
        _, pmin, pmax, map_type = param_tuple
        forward_map, _ = get_param_map(map_type)
        bounds.append((forward_map(pmin), forward_map(pmax)))

    res = least_squares(cost_function, initial, bounds=np.array(bounds).T)
    set_params(config, p0, res.x)

    return config, np.linalg.norm(res.fun), param_history, obj_history


def optimize_zero_d_im_pump(config, p0, data, verbose=0, weight_flow=1.0, weight_volume=1.0, node_in=None):
    """Optimize intramyocardial pump model parameters.

    The IM pump model has explicit compliance elements whose pressure determines volume.

    Args:
        config: svZeroDSolver configuration
        p0: Parameter dictionary with (val, min, max, map_type) tuples
        data: Dictionary with smoothed data
        verbose: Verbosity level
        weight_flow: Weight for flow objective (default 1.0)
        weight_volume: Weight for volume objective (default 1.0)
        node_in: Node name for flow extraction (default: uses n_in_im)
    """
    qref = data["s"]["qlad"]
    vref = data["s"]["vmyo"]

    param_history = []
    obj_history = []

    def cost_function(p):
        pset = set_params(config, p0, p)
        param_history.append(pset.copy())

        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")

        # Get flow and volume from IM pump model
        q_sim, v_sim = get_sim_qv_im_pump(config, node_in=node_in)

        # Compute objective: fit flow and volume
        obj_q = weight_flow * get_objective(qref, q_sim)
        obj_v = weight_volume * get_objective(vref, v_sim)

        obj = np.concatenate((obj_q, obj_v))
        obj_history.append(np.linalg.norm(obj))

        if verbose:
            print(f"{np.linalg.norm(obj):.1e}", end="\n")
        return obj

    initial = get_params(p0)
    if verbose:
        for k in p0.keys():
            print(f"{k[1]}", end="\t")
        print("obj", end="\n")

    bounds = []
    for param_tuple in p0.values():
        _, pmin, pmax, map_type = param_tuple
        forward_map, _ = get_param_map(map_type)
        bounds.append((forward_map(pmin), forward_map(pmax)))

    res = least_squares(cost_function, initial, bounds=np.array(bounds).T)
    set_params(config, p0, res.x)

    return config, np.linalg.norm(res.fun), param_history, obj_history


def optimize_zero_d_backpressure(config, p0, data, verbose=0, weight_flow=1.0, weight_volume=1.0,
                                  weight_flow_min=0.0, node_in=None):
    """Optimize backpressure model parameters.

    The backpressure model uses scaled Pim as outlet pressure, creating direct
    opposition to flow during systole. This can produce sharper systolic flow dips
    than the CORONARY BC.

    Args:
        config: svZeroDSolver configuration
        p0: Parameter dictionary with (val, min, max, map_type) tuples
        data: Dictionary with smoothed data
        verbose: Verbosity level
        weight_flow: Weight for flow objective (default 1.0)
        weight_volume: Weight for volume objective (default 1.0)
        weight_flow_min: Extra weight on matching flow minimum (default 0.0)
        node_in: Node name for flow extraction (default: uses n_in_backpressure)
    """
    qref = data["s"]["qlad"]
    vref = data["s"]["vmyo"]

    # Find the flow minimum for extra weighting
    q_min_idx = np.argmin(qref)
    q_min_val = qref[q_min_idx]

    param_history = []
    obj_history = []

    def cost_function(p):
        pset = set_params(config, p0, p)
        param_history.append(pset.copy())

        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")

        # Get flow and volume from backpressure model
        q_sim, v_sim = get_sim_qv_backpressure(config, node_in=node_in)

        # Compute objective: fit flow and volume
        obj_q = weight_flow * get_objective(qref, q_sim)
        obj_v = weight_volume * get_objective(vref, v_sim)

        obj = np.concatenate((obj_q, obj_v))

        # Extra penalty for not matching flow minimum
        if weight_flow_min > 0:
            # Penalize the difference at the flow minimum point
            q_sim_at_min = q_sim[q_min_idx] if len(q_sim) > q_min_idx else 0
            flow_min_penalty = weight_flow_min * (q_min_val - q_sim_at_min) / (np.mean(np.abs(qref)) + 1e-10)
            obj = np.concatenate((obj, [flow_min_penalty]))

        obj_history.append(np.linalg.norm(obj))

        if verbose:
            print(f"{np.linalg.norm(obj):.1e}", end="\n")
        return obj

    initial = get_params(p0)
    if verbose:
        for k in p0.keys():
            print(f"{k[1]}", end="\t")
        print("obj", end="\n")

    bounds = []
    for param_tuple in p0.values():
        _, pmin, pmax, map_type = param_tuple
        forward_map, _ = get_param_map(map_type)
        bounds.append((forward_map(pmin), forward_map(pmax)))

    res = least_squares(cost_function, initial, bounds=np.array(bounds).T)
    set_params(config, p0, res.x)

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


def estimate(data, animal, study, verb=0, use_wk_model=False, use_pdriven=False, use_im_pump=False,
              use_inductance=False, use_backpressure=False, use_varres_time=True, weight_pressure=1.0, weight_volume=0.5,
              weight_flow=2.0, weight_derivative=0.0, weight_regularization=0.0, weight_flow_min=5.0):
    """Estimate coronary model parameters.

    Args:
        data: Dictionary with smoothed data
        animal: Animal ID
        study: Study name
        verb: Verbosity level
        use_wk_model: If True, use model with inlet Windkessel for pressure pulsatility
        use_pdriven: If True, prescribe aortic pressure and fit flow + volume
        use_im_pump: If True, use intramyocardial pump model with explicit compliances
        use_inductance: If True, use model with inductance for time delay (standard CORONARY BC)
        use_backpressure: If True, use backpressure model (scaled Pim as outlet pressure)
        use_varres_time: If True, use time-varying resistance model (CORONARY_VAR_RES BC with inductance)
        weight_pressure: Weight for pressure in objective function (flow-driven)
        weight_volume: Weight for volume in objective function
        weight_flow: Weight for flow in objective function (pressure-driven)
        weight_derivative: Weight for derivative matching (pressure-driven)
            Penalizes loss of pulsatility/oscillations in fitted curves
        weight_regularization: Weight for bound regularization (pressure-driven)
            Penalizes parameters approaching their bounds
        weight_flow_min: Extra weight on matching flow minimum (backpressure model)
    """
    # create 0D model
    if use_backpressure:
        model_name = model_backpressure
    elif use_varres_time:
        model_name = model_varres_time
    elif use_im_pump:
        model_name = model_im_pump
    elif use_inductance:
        model_name = model_inductance
    elif use_pdriven:
        model_name = model_pdriven
    elif use_wk_model:
        model_name = model_wk
    else:
        model_name = model

    config = read_config(f"models/{model_name}.json")
    config[str_param][str_time] = len(data["s"]["t"])

    # set boundary conditions based on model type
    pini = {}

    if use_backpressure:
        # Backpressure model: aortic pressure at inlet, scaled Pim at outlet
        # The outlet pressure creates direct opposition to flow during systole
        pini[("BC_AO", "t")] = (data["s"]["t"].tolist(), None, None, "lin")
        pini[("BC_AO", "P")] = (data["s"]["pat"].tolist(), None, None, "lin")
        pini[("BC_PIM", "t")] = (data["s"]["t"].tolist(), None, None, "lin")
        # Store base Pim for scaling optimization
        pini[("BC_PIM", "_Pim_base")] = (data["s"]["pven"].tolist(), None, None, "lin")
        # Initial Pim = 1.0 × LVP (will be scaled by optimizer)
        pini[("BC_PIM", "P")] = (data["s"]["pven"].tolist(), None, None, "lin")
        # Venous outlet at ~0 pressure (already set in model file)
        pini[("BC_VEN", "t")] = (data["s"]["t"].tolist(), None, None, "lin")
        pini[("BC_VEN", "P")] = ([0.0] * len(data["s"]["t"]), None, None, "lin")
    elif use_im_pump:
        # IM pump model: prescribe aortic pressure at inlet, Pim at outlet
        pini[("BC_AO", "t")] = (data["s"]["t"].tolist(), None, None, "lin")
        pini[("BC_AO", "P")] = (data["s"]["pat"].tolist(), None, None, "lin")
        pini[("BC_IM", "t")] = (data["s"]["t"].tolist(), None, None, "lin")
        # Outlet pressure = intramyocardial pressure (scaled from LV pressure)
        # The Pim acts as a "pump" - high during systole, low during diastole
        pini[("BC_IM", "P")] = (data["s"]["pven"].tolist(), None, None, "lin")
    else:
        pini[("BC_AT", "t")] = (data["s"]["t"].tolist(), None, None, "lin")

        if use_pdriven or use_inductance or use_varres_time:
            # Prescribe aortic pressure (measured) for pressure-driven models
            pini[("BC_AT", "P")] = (data["s"]["pat"].tolist(), None, None, "lin")
        else:
            # Prescribe LAD flow (measured)
            pini[("BC_AT", "Q")] = (data["s"]["qlad"].tolist(), None, None, "lin")

        pini[("BC_COR", "t")] = (data["s"]["t"].tolist(), None, None, "lin")

        # Compute composite Pim based on intramyocardial pump literature:
        # IMP = CEP + VE = α×LVP + β×dLVP/dt
        # CEP: Cavity-induced extracellular pressure (LV pressure transmission)
        # VE: Varying elastance (myocardial stiffness, peaks before LV pressure)
        # Reference: Kerckhoffs et al. 2010, van den Broek et al. 2022
        pven = np.array(data["s"]["pven"])
        t = np.array(data["s"]["t"])
        dt = t[1] - t[0]

        # Compute dP/dt (rate of pressure change)
        dpdt = np.gradient(pven, dt)

        # Normalize dP/dt to have same peak magnitude as pven for scaling
        dpdt_norm = dpdt * (pven.max() / (np.abs(dpdt).max() + 1e-10))

        # Only use positive dP/dt (early systolic rise)
        dpdt_pos = np.maximum(dpdt_norm, 0)

        # Store components for optimization
        # Pim = α×LVP + β×dLVP/dt (α and β are optimized as _Pim_scale and _Pim_dpdt)
        pini[("BC_COR", "Pim")] = (pven.tolist(), None, None, "lin")
        pini[("BC_COR", "_Pim_base")] = (pven.tolist(), None, None, "lin")
        pini[("BC_COR", "_Pim_dpdt_base")] = (dpdt_pos.tolist(), None, None, "lin")

        # For inductance model (without time-varying resistance): set fixed Ra2 value
        if use_inductance and not use_varres_time:
            pini[("BC_COR", "Ra2")] = (1e6, None, None, "lin")  # Fixed Ra2

        # Set timing constants for CORONARY_VAR_RES BC (time-varying resistance models)
        # T_vc: Time of ventricular contraction end (Ra2 increases from Ra2_min to Ra2_max during [0, T_vc])
        # T_vr: Duration of ventricular relaxation (Ra2 decreases from Ra2_max to Ra2_min during [T_vc, T_vc+T_vr])
        if use_varres_time or (not use_inductance and not use_im_pump and not use_backpressure):
            t = data["o"]["t"]
            V = data["o"]["vmyo"]
            pven = data["o"]["pven"]

            # Estimate T_vc from time of minimum volume (end of systolic compression)
            t_v_min = t[np.argmin(V)]
            # Estimate relaxation duration from time to reach 90% of max volume
            v_range = V.max() - V.min()
            v_threshold = V.min() + 0.9 * v_range
            t_after_min = t[t > t_v_min]
            v_after_min = V[t > t_v_min]
            if len(t_after_min) > 0:
                idx_relaxed = np.where(v_after_min >= v_threshold)[0]
                if len(idx_relaxed) > 0:
                    t_relaxed = t_after_min[idx_relaxed[0]]
                    t_v_dia = t_relaxed - t_v_min
                else:
                    t_v_dia = t[-1] - t_v_min  # Use remaining cycle time
            else:
                t_v_dia = 0.3  # Default relaxation duration

            print(f"T_vc (contraction end): {t_v_min:.3f}s, T_vr (relaxation duration): {t_v_dia:.3f}s")
            pini[("BC_COR", "T_vc")] = (t_v_min, None, None, "lin")
            pini[("BC_COR", "T_vr")] = (t_v_dia, None, None, "lin")

    set_params(config, pini)

    # set initial values (val, min, max, map_type)
    # map_type can be 'log' (logarithmic) or 'linear'
    p0 = OrderedDict()

    if use_backpressure:
        # Backpressure model parameters
        # V_prox: proximal vessel with inductance for time delay
        # V_mid: main coronary resistance and compliance
        # V_dist: distal resistance to Pim outlet
        # BC_PIM: outlet pressure = _Pim_scale × LVP (creates back-pressure)
        p0[("V_prox", "L")] = (1e3, 1e1, 1e5, "log")  # Inductance for time delay
        p0[("V_prox", "R_poiseuille")] = (1e5, 1e3, 1e7, "log")  # Proximal resistance
        p0[("V_mid", "R_poiseuille")] = (1e6, 1e4, 1e8, "log")  # Main resistance
        p0[("V_mid", "C")] = (1e-6, 1e-8, 1e-4, "log")  # Main compliance
        p0[("V_dist", "R_poiseuille")] = (1e5, 1e3, 1e7, "log")  # Distal resistance
        # Pim scaling factor: higher values create stronger back-pressure during systole
        # When _Pim_scale × Pim > P_ao, flow reverses
        # Literature suggests Pim can exceed LVP in subendocardium
        p0[("BC_PIM", "_Pim_scale")] = (1.0, 0.5, 2.0, "lin")
    elif use_im_pump:
        # Intramyocardial pump model parameters
        # R_epi: epicardial resistance (proximal)
        # C_art: arterial compliance (epicardial)
        # R_im: intramyocardial resistance (varies with compression)
        # C_im: intramyocardial compliance (main blood storage)
        # R_ven: venous resistance (distal)
        # Note: Compliance bounds widened to allow larger volume storage
        # V ~ C * P, with P ~ 80000 Ba and V ~ 0.1 ml, need C ~ 1e-6
        p0[("R_epi", "R_poiseuille")] = (1e5, 1e3, 1e7, "log")
        p0[("C_art", "C")] = (1e-6, 1e-9, 1e-3, "log")
        p0[("R_im", "R_poiseuille")] = (1e6, 1e4, 1e8, "log")
        p0[("C_im", "C")] = (1e-5, 1e-8, 1e-2, "log")
        p0[("R_ven", "R_poiseuille")] = (1e5, 1e3, 1e7, "log")
    elif use_varres_time:
        # Time-varying resistance model parameters - uses CORONARY_VAR_RES BC with inductance
        # BV_prox: proximal vessel with inductance for time delay
        p0[("BV_prox", "L")] = (1e3, 1e1, 1e5, "log")
        p0[("BV_prox", "R_poiseuille")] = (1e5, 1e3, 1e7, "log")
        # CORONARY_VAR_RES BC parameters
        # Ra1: small artery (epicardial) resistance
        p0[("BC_COR", "Ra1")] = (1e6, 1e4, 1e8, "log")
        # Ra2: microvascular resistance (geometric mean of Ra2_min and Ra2_max)
        # The actual resistance varies between Ra2_min and Ra2_max during cardiac cycle
        p0[("BC_COR", "Ra2")] = (1e6, 1e4, 1e8, "log")
        # _ratio_Ra2: ratio of Ra2_max to Ra2_min (controls amplitude of resistance variation)
        # Higher ratio = larger resistance change during cardiac cycle
        # Physiological range: 2-6 based on intramyocardial pressure studies
        p0[("BC_COR", "_ratio_Ra2")] = (3.0, 1.5, 8.0, "lin")
        # Rv1: venous resistance
        p0[("BC_COR", "Rv1")] = (1e5, 1e3, 1e7, "log")
        # Time constants for capacitances
        p0[("BC_COR", "tc1")] = (0.15, 0.05, 0.5, "lin")
        p0[("BC_COR", "tc2")] = (0.3, 0.1, 2.0, "lin")
        # Timing parameters for resistance variation (can be optimized or fixed from data)
        # T_vc: time of ventricular contraction end (resistance increases during [0, T_vc])
        # T_vr: duration of ventricular relaxation (resistance decreases during [T_vc, T_vc+T_vr])
        # Initial values computed from data (minimum volume time)
        t = data["o"]["t"]
        V = data["o"]["vmyo"]
        t_cycle = data["s"]["t"][-1]

        # Estimate T_vc from time of minimum volume (end of systolic compression)
        t_v_min = t[np.argmin(V)]
        # Estimate relaxation duration from time to reach 90% of max volume
        v_range = V.max() - V.min()
        v_threshold = V.min() + 0.9 * v_range
        t_after_min = t[t > t_v_min]
        v_after_min = V[t > t_v_min]
        if len(t_after_min) > 0:
            idx_relaxed = np.where(v_after_min >= v_threshold)[0]
            if len(idx_relaxed) > 0:
                t_relaxed = t_after_min[idx_relaxed[0]]
                t_v_dia = t_relaxed - t_v_min
            else:
                t_v_dia = 0.5 * (t_cycle - t_v_min)  # Use half remaining cycle time
        else:
            t_v_dia = 0.3  # Default relaxation duration

        # Ensure initial values are within bounds
        T_vc_init = np.clip(t_v_min, 0.05, 0.6 * t_cycle)
        T_vr_init = np.clip(t_v_dia, 0.05, 0.6 * t_cycle)
        p0[("BC_COR", "T_vc")] = (T_vc_init, 0.05, 0.6 * t_cycle, "lin")
        p0[("BC_COR", "T_vr")] = (T_vr_init, 0.05, 0.6 * t_cycle, "lin")
        # Intramyocardial pressure scaling
        p0[("BC_COR", "_Pim_scale")] = (1.5, 0.8, 2.5, "lin")
        p0[("BC_COR", "_Pim_dpdt")] = (0.5, 0.0, 1.5, "lin")
    elif use_inductance:
        # Inductance model parameters - uses standard CORONARY BC
        # BV_prox: proximal vessel with inductance for time delay
        # L: inductance creates phase lag (time delay) between pressure and flow
        #    Time constant tau_L = L/R, with R ~ 1e5 and tau ~ 10ms, L ~ 1e3
        # C: set to 0 (removed from fit - was hitting lower bound)
        # R_poiseuille: proximal resistance (epicardial)
        p0[("BV_prox", "L")] = (1e3, 1e1, 1e5, "log")
        p0[("BV_prox", "R_poiseuille")] = (1e5, 1e3, 1e7, "log")
        # Standard CORONARY BC parameters (not variable resistance)
        # Ra2 removed from fit (was hitting lower bound) - set fixed below
        p0[("BC_COR", "Ra1")] = (1e6, 1e4, 1e8, "log")
        p0[("BC_COR", "Rv1")] = (1e5, 1e3, 1e7, "log")
        # Use time constants instead of Ca/Cc directly to constrain physiological range
        # tc1 = Ra1 * Ca (arteriolar time constant, ~50-500 ms physiologically)
        # tc2 = Rv1 * Cc (venous/capillary time constant, ~100-2000 ms physiologically)
        # Note: wider bounds than typical physiological values to accommodate data variability
        p0[("BC_COR", "tc1")] = (0.15, 0.05, 0.5, "lin")
        p0[("BC_COR", "tc2")] = (0.3, 0.1, 2.0, "lin")
        # Intramyocardial pump parameters based on literature (Kerckhoffs 2010, van den Broek 2022)
        # Pim = _Pim_scale × LVP + _Pim_dpdt × dLVP/dt
        #
        # _Pim_scale: Scales CEP (cavity-induced extracellular pressure)
        #   - Values >1 needed to capture deep systolic flow dips
        #   - Sensitivity analysis: 1.5-2.0× LVP needed for realistic flow dip
        #
        # _Pim_dpdt: Scales VE (varying elastance / early activation)
        #   - Captures early systolic compression (peaks before LV pressure)
        #   - Sensitivity analysis: 0.5-1.0× dP/dt helps match flow timing
        p0[("BC_COR", "_Pim_scale")] = (1.5, 0.8, 2.5, "lin")
        p0[("BC_COR", "_Pim_dpdt")] = (0.5, 0.0, 1.5, "lin")
    elif use_wk_model:
        # Inlet Windkessel parameters for aortic pressure pulsatility
        p0[("R_ao", "R_poiseuille")] = (1e4, 1e3, 1e6, "log")
        p0[("R_ao", "C")] = (1e-6, 1e-8, 1e-4, "log")
        # Coronary BC parameters
        p0[("BC_COR", "Ra1")] = (1e6, 1e4, 1e8, "log")
        p0[("BC_COR", "Rv1")] = (1e5, 1e3, 1e7, "log")
        p0[("BC_COR", "Ra2")] = (1e6, 1e4, 1e8, "log")
        p0[("BC_COR", "_ratio_Ra2")] = (2.0, 1.2, 10.0, "lin")
        p0[("BC_COR", "tc1")] = (0.05, 0.001, 1.0, "lin")
        p0[("BC_COR", "tc2")] = (0.05, 0.001, 1.0, "lin")
    else:
        # Standard coronary model parameters
        p0[("BC_COR", "Ra1")] = (1e7, 1e6, 1e8, "log")
        p0[("BC_COR", "Rv1")] = (1e6, 1e5, 1e7, "log")
        p0[("BC_COR", "Ra2")] = (1e7, 1e6, 1e8, "log")
        p0[("BC_COR", "_ratio_Ra2")] = (2.0, 1.0, 4.0, "lin")
        p0[("BC_COR", "tc1")] = (0.05, 0.01, 0.5, "lin")
        p0[("BC_COR", "tc2")] = (0.05, 0.01, 0.5, "lin")

    set_params(config, p0)

    # Determine which node to read from
    if use_backpressure:
        node_in = n_in_backpressure
    elif use_varres_time:
        node_in = n_in_varres_time
    elif use_im_pump:
        node_in = n_in_im
    elif use_inductance:
        node_in = n_in_inductance
    elif use_wk_model:
        node_in = n_in_wk
    else:
        node_in = n_in

    if use_backpressure:
        # Backpressure model: fit flow + volume with scaled Pim as outlet pressure
        config_opt, err, param_history, obj_history = optimize_zero_d_backpressure(
            config, p0, data, verbose=verb,
            weight_flow=weight_flow,
            weight_volume=weight_volume,
            weight_flow_min=weight_flow_min,
            node_in=node_in
        )
    elif use_im_pump:
        # IM pump model: fit flow + volume
        config_opt, err, param_history, obj_history = optimize_zero_d_im_pump(
            config, p0, data, verbose=verb,
            weight_flow=weight_flow,
            weight_volume=weight_volume,
            node_in=node_in
        )
    elif use_varres_time or use_inductance or use_pdriven:
        # Pressure-driven (with time-varying resistance, inductance, or plain): fit flow + volume
        config_opt, err, param_history, obj_history = optimize_zero_d_pdriven(
            config, p0, data, verbose=verb,
            weight_flow=weight_flow,
            weight_volume=weight_volume,
            weight_derivative=weight_derivative,
            weight_regularization=weight_regularization,
            weight_flow_min=weight_flow_min,
            node_in=node_in
        )
    else:
        # Flow-driven: fit pressure + volume
        config_opt, err, param_history, obj_history = optimize_zero_d(
            config, p0, data, verbose=verb,
            weight_pressure=weight_pressure,
            weight_volume=weight_volume,
            node_in=node_in
        )

    return config_opt, p0, err


def main():
    # animals = [8]  # use DSEA08 for testing new models
    animals = [8, 10, 15, 16]  # clean
    # animals = [15]  # test problem case
    # animals = [6, 7, 8, 10, 14, 15, 16] # all
    studies = [
        "baseline",
        "mild_sten",
        "mild_sten_dob",
        "mod_sten",
        "mod_sten_dob",
    ]

    # Configuration options
    # Model selection (choose ONE):
    #   use_varres_time=True: Time-varying resistance (CORONARY_VAR_RES BC) with inductance
    #   use_backpressure=True: Custom model with scaled Pim as outlet back-pressure
    #   use_inductance=True: Pressure-driven + inductance for time delay (standard CORONARY BC)
    #   use_im_pump=True: Intramyocardial pump model with explicit compliances
    #   use_pdriven=True: Prescribe aortic pressure, fit flow + volume
    #   use_wk_model=True: Add inlet Windkessel for pressure pulsatility
    #   All False: Original model (prescribe flow, fit pressure + volume)
    use_varres_time = True  # Use time-varying resistance to address volume time shifts
    use_backpressure = False
    use_inductance = False  # Superseded by use_varres_time
    use_im_pump = False
    use_pdriven = False
    use_wk_model = False

    # Time alignment options
    # align_method: "landmarks", "xcorr", or "none"
    align_method = "none"  # Disable alignment to see raw discrepancies

    # Objective weights - prioritize flow matching over volume
    weight_pressure = 1.0
    weight_volume = 0.5  # Reduce volume weight to allow more focus on flow
    weight_flow = 2.0  # Increase flow weight
    weight_derivative = 0.0  # Derivative matching to preserve pulsatility (try 0.1-1.0)
    weight_regularization = 0.0  # Disable regularization to allow extremes
    weight_flow_min = 5.0  # Strong emphasis on matching flow minimum

    # Collect optimized parameters from all animals
    all_animals_optimized = {}

    for animal in animals:
        optimized = process(animal, studies,
                          use_wk_model=use_wk_model,
                          use_pdriven=use_pdriven,
                          use_im_pump=use_im_pump,
                          use_inductance=use_inductance,
                          use_backpressure=use_backpressure,
                          use_varres_time=use_varres_time,
                          align_method=align_method,
                          weight_pressure=weight_pressure,
                          weight_volume=weight_volume,
                          weight_flow=weight_flow,
                          weight_derivative=weight_derivative,
                          weight_regularization=weight_regularization,
                          weight_flow_min=weight_flow_min)
        if optimized:
            all_animals_optimized[animal] = optimized

    # Plot multi-animal comparison
    if len(all_animals_optimized) > 1:
        plot_parameters_multi(all_animals_optimized, studies)


def process(animal, studies, use_wk_model=False, use_pdriven=False, use_im_pump=False,
            use_inductance=False, use_backpressure=False, use_varres_time=True, align_method="none",
            weight_pressure=1.0, weight_volume=0.5, weight_flow=2.0, weight_derivative=0.0,
            weight_regularization=0.0, weight_flow_min=5.0):
    """Process all studies for an animal.

    Args:
        animal: Animal ID
        studies: List of study names
        use_wk_model: If True, use Windkessel inlet model
        use_pdriven: If True, prescribe aortic pressure and fit flow + volume
        use_im_pump: If True, use intramyocardial pump model
        use_inductance: If True, use inductance model for time delay (standard CORONARY BC)
        use_backpressure: If True, use backpressure model (scaled Pim as outlet pressure)
        use_varres_time: If True, use time-varying resistance model (CORONARY_VAR_RES with inductance)
        align_method: Time alignment method ("landmarks", "xcorr", or "none")
        weight_pressure: Weight for pressure objective
        weight_volume: Weight for volume objective
        weight_flow: Weight for flow objective (pressure-driven mode)
        weight_derivative: Weight for derivative matching (pressure-driven mode)
        weight_regularization: Weight for bound regularization (pressure-driven mode)
        weight_flow_min: Extra weight on matching flow minimum (backpressure model)
    """
    optimized = defaultdict(dict)
    data = {}
    config = {}
    err = {}

    for study in studies:
        dat = read_and_smooth_data(animal, study)
        if dat != {}:
            data[study] = dat
            # Apply time alignment if requested
            if align_method != "none":
                print(f"Aligning data for {study}...")
                align_flow_to_pressure(dat["s"], align_method=align_method)

    plot_data(animal, data)

    for study in data.keys():
        print(f"Estimating {study}...")
        config[study], p0, err = estimate(
            data[study], animal, study, verb=1,
            use_wk_model=use_wk_model,
            use_pdriven=use_pdriven,
            use_im_pump=use_im_pump,
            use_inductance=use_inductance,
            use_backpressure=use_backpressure,
            use_varres_time=use_varres_time,
            weight_pressure=weight_pressure,
            weight_volume=weight_volume,
            weight_flow=weight_flow,
            weight_derivative=weight_derivative,
            weight_regularization=weight_regularization,
            weight_flow_min=weight_flow_min
        )

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
        # Ra2 (geometric mean) and _ratio_Ra2 from Ra2_min and Ra2_max
        if ("BC_COR", "Ra2_min") in optimized[study] and ("BC_COR", "Ra2_max") in optimized[study]:
            Ra2_min = optimized[study][("BC_COR", "Ra2_min")]
            Ra2_max = optimized[study][("BC_COR", "Ra2_max")]
            Ra2 = np.sqrt(Ra2_min * Ra2_max)  # geometric mean
            _ratio_Ra2 = Ra2_max / Ra2_min
            optimized[study][("BC_COR", "Ra2")] = Ra2
            optimized[study][("BC_COR", "_ratio_Ra2")] = _ratio_Ra2

        # Time constants from R and C values
        if ("BC_COR", "Ra1") in optimized[study] and ("BC_COR", "Ca") in optimized[study]:
            tc1 = optimized[study][("BC_COR", "Ra1")] * optimized[study][("BC_COR", "Ca")]
            optimized[study][("BC_COR", "tc1")] = tc1

        if ("BC_COR", "Rv1") in optimized[study] and ("BC_COR", "Cc") in optimized[study]:
            tc2 = optimized[study][("BC_COR", "Rv1")] * optimized[study][("BC_COR", "Cc")]
            optimized[study][("BC_COR", "tc2")] = tc2

        optimized[study][("global", "residual")] = err

    plot_results(animal, config, data, use_pdriven=use_pdriven, use_im_pump=use_im_pump,
                 use_inductance=use_inductance, use_backpressure=use_backpressure,
                 use_varres_time=use_varres_time)
    plot_parameters(animal, optimized)

    return dict(optimized)


if __name__ == "__main__":
    main()
