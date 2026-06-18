#!/usr/bin/env python3
"""Main processing script for coronary model parameter estimation."""

import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from scipy.integrate import cumulative_trapezoid

from utils import (
    read_config, mmHg_to_Ba, Ba_to_mmHg, str_val, bc_val, str_bc, str_param, str_time,
    smooth, set_params, get_param_map
)
from simulation import get_sim_qv, get_sim_internal_flows, get_sim_detailed
from optimization import optimize
from plotting import (
    plot_data, plot_results, plot_parameters, plot_parameters_multi,
    plot_combined_results, plot_circuit_diagram, plot_parameters_normalized,
    plot_internal_flows, plot_volume_metrics, plot_volume_balance,
    plot_sloshing_capacity, plot_volume_vs_inflow, plot_pim_parameters
)

# Model configuration
MODEL_NAME = "coronary_varres_time"
NODE_IN = "BC_AT:BV_prox"

# Intramyocardial-pressure driver for the main pipeline. "hybrid" = a*P_LV +
# beta*[dS/dt]+ (cavity-induced + shortening-rate) was the best model overall
# (lowest residual and AIC) and self-tunes to deformation under moderate stenosis;
# see benchmark_pim.py. Studies without strain fall back to LVP automatically.
PIM_MODE = "hybrid"


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
    data["qlad"] = df["LAD Flow [ml/s]"].to_numpy().copy()
    data["qmyo"] = np.gradient(data["vmyo"], data["t"])
    data["vlad"] = cumulative_trapezoid(data["qlad"], data["t"], initial=0).copy()

    # End-diastolic tissue volume (EDV): per Stendahl et al., ED (t=0) is the
    # LV-pressure upstroke, i.e. the first sample of the acquired cycle. The paper
    # normalizes all per-cycle volumes to EDV, and derives tissue mass = EDV * 1.05 g/mL.
    data["edv"] = float(df["tissue vol ischemic [ml]"].to_numpy()[0])
    data["tissue_mass_real"] = data["edv"] * 1.05  # myocardial density 1.05 g/mL

    # Regional strains (deformation), used for diagnostics / deformation-driven Pim
    for s in ["rad strain", "circ strain", "long strain"]:
        if s in df.keys():
            data[s.replace(" ", "_")] = df[s].to_numpy()

    # Scale LAD flow using microsphere data (ml/min/g)
    # Assume tissue mass = 1 g, so we work directly in per-gram units
    csv_file = os.path.join("data", f"{get_name(animal)}_microsphere.csv")
    df = pd.read_csv(csv_file)
    flow_ml_min_g = df[f"{study} ischemic flow [ml/min/g]"].to_numpy()[0]

    # Convert ml/min/g to ml/cycle/g using cardiac cycle duration
    t_cycle = data["t"][-1] - data["t"][0]  # seconds
    cycles_per_min = 60.0 / t_cycle
    target_flow_per_cycle_g = flow_ml_min_g / cycles_per_min  # ml/cycle/g

    # Current LAD volume change over the cardiac cycle
    dvlad = data["vlad"].max() - data["vlad"].min()
    scale = target_flow_per_cycle_g / dvlad
    data["qlad"] *= scale  # now in ml/s/g
    data["vlad"] *= scale  # now in ml/g

    # Store tissue mass as 1 g (assumed)
    data["tissue_mass"] = 1.0

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

    # Carry over tissue mass and EDV for normalization
    for key in ["tissue_mass", "edv", "tissue_mass_real"]:
        if key in data_o:
            data_s[key] = data_o[key]

    # Carry over strains, interpolated onto the simulation time grid
    for s in ["rad_strain", "circ_strain", "long_strain"]:
        if s in data_o:
            data_s[s] = np.interp(data_s["t"], data_o["t"], data_o[s])

    return data_s


def build_deformation_signal(data):
    """Build a normalized [0,1] contraction signal from regional myocardial strain.

    Used as a deformation-based driver for intramyocardial pressure (Pim) -- the
    mechanism Stendahl et al. argue dominates over LV-pressure transmission.

    Circumferential shortening (-E_cc, positive in systole) is used: across all
    animals/conditions it tracks the phasic myocardial volume (the ideal Pim shape,
    ~ -volume) at corr 0.91-0.98, far better than LVP (0.13-0.68, occasionally
    inverted), and -- unlike radial strain -- it remains robust under moderate
    stenosis. Falls back to radial strain, then None, if -E_cc is unavailable.

    Returns the normalized signal, or None if no usable strain is available.
    """
    def _norm(sig):
        sig = np.asarray(sig, dtype=float)
        if not np.all(np.isfinite(sig)):
            return None
        rng = sig.max() - sig.min()
        return (sig - sig.min()) / rng if rng > 0 else None

    circ = data["s"].get("circ_strain")
    if circ is not None:
        out = _norm(-np.asarray(circ, dtype=float))  # circumferential shortening
        if out is not None:
            return out
    rad = data["s"].get("rad_strain")
    if rad is not None:
        return _norm(np.asarray(rad, dtype=float))    # fallback: wall thickening
    return None


def setup_model(data, pim_mode="lvp"):
    """Set up model configuration and parameters.

    Args:
        data: smoothed/original data dict
        pim_mode: intramyocardial-pressure driver:
            - "lvp": Pim = LV pressure (transmission / intramyocardial-pump model)
            - "deformation": Pim = alpha * S(t), S = circ-shortening contraction
              (shortening-induced pump, amplitude only)
            - "deformation_rate": Pim = alpha*S(t) + beta*[dS/dt]+ (varying-elastance
              / shortening + shortening-rate; the rate term gives the sharp early-
              systolic flow impediment)
            - "hybrid": Pim = a*LVP + beta*[dS/dt]+ (cavity-induced CEP + shortening-
              rate impediment; Algranati-Kassab-Lanir CEP+SIP decomposition)
          Deformation modes fall back to "lvp" if strains are unavailable.

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

    # Deformation signal S(t) and its (positive) rate, both normalized to [0,1]
    deform_modes = ("deformation", "deformation_rate", "hybrid")
    deform = build_deformation_signal(data) if pim_mode in deform_modes else None
    if pim_mode in deform_modes and deform is None:
        print("  [Pim] strain unavailable -> falling back to LVP-driven Pim")
        pim_mode = "lvp"

    deform_rate = None
    if deform is not None:
        dS = np.maximum(np.gradient(deform, dt), 0.0)  # active shortening rate
        rng = dS.max() - dS.min()
        deform_rate = (dS - dS.min()) / rng if rng > 0 else dS * 0.0

    pini[("BC_COR", "_Pim_base")] = (pven.tolist(), None, None, "lin")
    pini[("BC_COR", "_Pim_dpdt_base")] = (dpdt_pos.tolist(), None, None, "lin")
    if deform is not None:
        pini[("BC_COR", "_Pim_strain_base")] = (deform.tolist(), None, None, "lin")
        pini[("BC_COR", "_Pim_strain_rate_base")] = (deform_rate.tolist(), None, None, "lin")
    # Initial Pim placeholder (overwritten by fitted components during optimization)
    if pim_mode in ("deformation", "deformation_rate"):
        pini[("BC_COR", "Pim")] = ((deform * pven.max()).tolist(), None, None, "lin")
    else:  # lvp, hybrid
        pini[("BC_COR", "Pim")] = (pven.tolist(), None, None, "lin")

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

    _, _ = set_params(config, pini)

    # Set blood vessel parameters to zero (no proximal vessel)
    pini_bv = {}
    pini_bv[("BV_prox", "L")] = (0.0, None, None, "lin")
    pini_bv[("BV_prox", "R_poiseuille")] = (0.0, None, None, "lin")
    _, _ = set_params(config, pini_bv)

    # Set optimization parameters (val, min, max, map_type)
    p0 = OrderedDict()
    p0[("BC_COR", "Ra1")] = (2e6, 1e5, 1e8, "log")
    p0[("BC_COR", "Ra2")] = (2e6, 1e5, 1e8, "log")
    # Ra2_max/Ra2_min contrast of the elastance-modulated micro-resistance.
    # NOTE: weakly identified. Tested bounds 5/8/12 -> the optimizer pins at the
    # upper bound for most baselines, yet the residual is essentially unchanged
    # (e.g. mod_sten ~2.8x baseline regardless). Above ~5 the systolic resistance
    # already fully impedes micro-flow, so the fit is insensitive to further
    # contrast. We therefore cap at a physiologically reasonable value rather than
    # let a non-identifiable parameter drift; the data only constrains ratio >~ 5.
    p0[("BC_COR", "_ratio_Ra2")] = (2.0, 1.5, 8.0, "lin")
    p0[("BC_COR", "Rv1")] = (2e5, 1e4, 1e7, "log")
    p0[("BC_COR", "tc1")] = (0.05, 0.03, 0.15, "lin")       # Narrowed to prevent Ca outliers
    p0[("BC_COR", "tc2")] = (0.15, 0.08, 0.35, "lin")       # Narrowed: was 0.1-2.0

    T_vc_init = np.clip(t_v_min, 0.05, 0.6 * t_cycle)
    T_vr_init = np.clip(t_v_dia, 0.05, 0.6 * t_cycle)
    p0[("BC_COR", "T_vc")] = (T_vc_init, 0.05, 0.6 * t_cycle, "lin")
    p0[("BC_COR", "T_vr")] = (T_vr_init, 0.05, 0.6 * t_cycle, "lin")
    # p0[("BC_COR", "_Pim_scale")] = (1.5, 0.8, 2.5, "lin")
    # p0[("BC_COR", "_Pim_dpdt")] = (0.5, 0.0, 1.5, "lin")

    # Pim component amplitudes [Ba]. Linear bounds starting at 0 so a component can
    # be switched off by the fit (a pinned-at-zero amplitude means it doesn't help).
    a_strain = (60 * mmHg_to_Ba, 0.0, 250 * mmHg_to_Ba, "lin")   # alpha: shortening
    a_rate = (40 * mmHg_to_Ba, 0.0, 300 * mmHg_to_Ba, "lin")     # beta: shortening-rate
    if pim_mode == "deformation":
        p0[("BC_COR", "_Pim_strain_amp")] = a_strain
    elif pim_mode == "deformation_rate":
        p0[("BC_COR", "_Pim_strain_amp")] = a_strain
        p0[("BC_COR", "_Pim_strain_rate")] = a_rate
    elif pim_mode == "hybrid":
        # a*LVP (CEP, dimensionless fraction) + beta*[dS/dt]+ (impediment)
        p0[("BC_COR", "_Pim_scale")] = (0.6, 0.0, 1.5, "lin")
        p0[("BC_COR", "_Pim_strain_rate")] = a_rate

    _, _ = set_params(config, p0)

    return config, p0


def extract_optimized_params(config, p0, optimized, study, err, param_values):
    """Extract optimized parameters from config and param_values dict.

    Args:
        config: Model configuration
        p0: Parameter dictionary
        optimized: Dict to store results
        study: Study name
        err: Residual error
        param_values: Dict of {(bc_name, param_name): value} from set_params
    """
    # Store all optimized parameter values
    for key, value in param_values.items():
        optimized[study][key] = value

    # Extract computed/derived parameters from config
    for bc in config[str_bc]:
        if bc["bc_name"] == "BC_COR":
            for key in ["Ra2_min", "Ra2_max", "Ca", "Cc"]:
                if key in bc["bc_values"]:
                    optimized[study][("BC_COR", key)] = bc["bc_values"][key]

    optimized[study][("global", "residual")] = err


def process_animal(animal, studies, weight_flow=2.0, weight_volume=0.5, weight_flow_min=5.0,
                   weight_flow_mean=1.0, weight_vol_min=2.0, reg_scale=1.0, volume_mode="minmax",
                   loss="linear", pim_mode="lvp"):
    """Process all studies for one animal.

    Args:
        animal: Animal ID number
        studies: List of study names to process
        weight_flow: Weight for flow objective
        weight_volume: Weight for volume objective
        weight_flow_min: Weight for flow minimum matching
        weight_flow_mean: Weight for mean LAD flow matching
        weight_vol_min: Weight for volume at minimum (from zero to min)
        reg_scale: Global regularization scale (0=none, 1=default, >1=stronger)
        volume_mode: "minmax", "rates", "curve", or "features"
        loss: Loss function ("linear", "soft_l1", "huber", "cauchy")
    """
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
        cfg, p0 = setup_model(data[study], pim_mode=pim_mode)

        cfg, err, param_values = optimize(cfg, p0, data[study], NODE_IN, verbose=1,
                           weight_flow=weight_flow, weight_volume=weight_volume,
                           weight_flow_min=weight_flow_min, weight_flow_mean=weight_flow_mean,
                           weight_vol_min=weight_vol_min, reg_scale=reg_scale,
                           volume_mode=volume_mode, loss=loss)

        config[study] = cfg
        extract_optimized_params(cfg, p0, optimized, study, err, param_values)

        # Save config
        with open(f"results/{get_name(animal)}_{study}_{MODEL_NAME}.json", "w") as f:
            json.dump(cfg, f, indent=2)

    plot_results(animal, config, data, get_sim_qv, NODE_IN, get_name, MODEL_NAME)
    plot_parameters(animal, optimized, get_name, MODEL_NAME)

    return dict(optimized), data, config


def compute_normalized_statistics(all_optimized, studies):
    """Compute statistics for parameters normalized by baseline.

    Args:
        all_optimized: Dict of {animal: {study: {param: value}}}
        studies: List of study names

    Returns:
        dict: Statistics for each parameter and condition
    """
    # Parameters to report
    params_to_report = [
        ("BC_COR", "Ra1"),
        ("BC_COR", "Ra2"),
        ("BC_COR", "Rv1"),
        ("BC_COR", "Ca"),
        ("BC_COR", "Cc"),
        ("BC_COR", "Ra2_min"),
        ("BC_COR", "Ra2_max"),
        ("BC_COR", "_ratio_Ra2"),
        ("BC_COR", "tc1"),
        ("BC_COR", "tc2"),
        ("BC_COR", "T_vc"),
        ("BC_COR", "T_vr"),
        ("BC_COR", "_Pim_scale"),
        ("BC_COR", "_Pim_dpdt"),
        ("global", "residual"),
    ]

    # Collect normalized values for each parameter and condition
    normalized_values = {study: {p: [] for p in params_to_report} for study in studies}

    for animal, animal_data in all_optimized.items():
        if "baseline" not in animal_data:
            continue

        baseline = animal_data["baseline"]

        for study in studies:
            if study not in animal_data:
                continue

            study_params = animal_data[study]

            for param in params_to_report:
                if param in baseline and param in study_params:
                    baseline_val = baseline[param]
                    study_val = study_params[param]
                    if baseline_val != 0:
                        normalized = study_val / baseline_val
                        normalized_values[study][param].append(normalized)

    # Compute statistics
    stats = {}
    for study in studies:
        stats[study] = {}
        for param in params_to_report:
            values = normalized_values[study][param]
            if len(values) > 0:
                stats[study][param] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "n": len(values),
                }

    return stats


MYO_DENSITY = 1.05  # myocardial density [g/mL] (Stendahl et al.)


def compute_volume_metrics(config, node_in, edv=None):
    """Compute volume change metrics from internal flows.

    All flows and volumes are normalized per gram of myocardial tissue (ml/cycle/g).
    This normalization is applied upstream in read_data() when scaling LAD flow.

    When the end-diastolic tissue volume (edv) is supplied, metrics are also
    expressed as a fraction of EDV to match the convention in Stendahl et al.
    (Fig. 4: per-cycle blood/tissue volumes normalized to EDV_tissue). Because our
    per-gram flows are intrinsically per gram and tissue mass = EDV * density, the
    conversion from ml/g to fraction-of-EDV is simply multiplication by density:
        (X ml/g) * (mass g) / (EDV mL) = X * (mass/EDV) = X * density.

    Args:
        config: svZeroDSolver configuration
        node_in: Node name for inlet flow extraction
        edv: End-diastolic tissue volume [mL]; enables fraction-of-EDV metrics
             and the "sloshing" (recirculated) volume quantification.

    Returns:
        dict: Volume and flow metrics with units:
            - Volume metrics (dV_total, dV_a, dV_c): ml/g
            - Flow volumes (V_in_forward, etc.): ml/cycle/g
            - Mean flows (Q_in_mean, etc.): ml/s/g
            - *_frac_edv: corresponding quantities as a fraction of EDV (if edv given)
            - Sloshing metrics (V_slosh_*, sloshing_fraction, Cc_required): see below
    """
    flows = get_sim_internal_flows(config, node_in)

    metrics = {}
    t = flows['t']
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    # Total intramyocardial volume change
    V_im = flows['V_im']
    metrics['dV_total'] = V_im.max() - V_im.min()

    # Mean flows
    metrics['Q_in_mean'] = np.mean(flows['Q_in'])
    metrics['Q_Ra2_mean'] = np.mean(flows['Q_mid'])  # Q_mid approximates Q_Ra2
    metrics['Q_Rv1_mean'] = np.mean(flows['Q_out'])  # Q_out approximates Q_Rv1

    # Flow amplitudes (max - min)
    metrics['Q_in_amp'] = flows['Q_in'].max() - flows['Q_in'].min()
    metrics['Q_Ra2_amp'] = flows['Q_mid'].max() - flows['Q_mid'].min()
    metrics['Q_Rv1_amp'] = flows['Q_out'].max() - flows['Q_out'].min()

    # Net flow over cardiac cycle (integral of flow rate)
    # Using trapezoidal integration
    metrics['Q_in_net'] = np.trapezoid(flows['Q_in'], t)
    metrics['Q_Ra2_net'] = np.trapezoid(flows['Q_mid'], t)
    metrics['Q_Rv1_net'] = np.trapezoid(flows['Q_out'], t)

    # Volume from positive flow (forward volume) - integral of positive flow
    Q_in_pos = np.maximum(flows['Q_in'], 0)
    Q_Ra2_pos = np.maximum(flows['Q_mid'], 0)
    Q_Rv1_pos = np.maximum(flows['Q_out'], 0)

    metrics['V_in_forward'] = np.trapezoid(Q_in_pos, t)
    metrics['V_Ra2_forward'] = np.trapezoid(Q_Ra2_pos, t)
    metrics['V_Rv1_forward'] = np.trapezoid(Q_Rv1_pos, t)

    # Volume from negative flow (backflow volume) - integral of negative flow (stored as positive)
    Q_in_neg = np.minimum(flows['Q_in'], 0)
    Q_Ra2_neg = np.minimum(flows['Q_mid'], 0)
    Q_Rv1_neg = np.minimum(flows['Q_out'], 0)

    metrics['V_in_backflow'] = -np.trapezoid(Q_in_neg, t)  # Make positive for easier interpretation
    metrics['V_Ra2_backflow'] = -np.trapezoid(Q_Ra2_neg, t)
    metrics['V_Rv1_backflow'] = -np.trapezoid(Q_Rv1_neg, t)

    # Intramyocardial volume rate of change (dV_im/dt)
    # Positive = filling, Negative = emptying
    dVim_dt = np.gradient(V_im, dt)
    dVim_dt_pos = np.maximum(dVim_dt, 0)
    dVim_dt_neg = np.minimum(dVim_dt, 0)

    metrics['V_im_forward'] = np.trapezoid(dVim_dt_pos, t)   # Filling volume
    metrics['V_im_backflow'] = -np.trapezoid(dVim_dt_neg, t)  # Emptying volume (positive)
    metrics['Q_im_net'] = np.trapezoid(dVim_dt, t)  # Net volume change (should be ~0 for periodic)

    # Verify: forward - backflow should equal net
    # Q_in_net = V_in_forward - V_in_backflow (this is a check)

    # Also keep the old names for backward compatibility
    metrics['Q_in_forward'] = metrics['V_in_forward']
    metrics['Q_Ra2_forward'] = metrics['V_Ra2_forward']
    metrics['Q_Rv1_forward'] = metrics['V_Rv1_forward']
    metrics['Q_in_backflow'] = metrics['V_in_backflow']
    metrics['Q_Ra2_backflow'] = metrics['V_Ra2_backflow']
    metrics['Q_Rv1_backflow'] = metrics['V_Rv1_backflow']

    # Estimate individual compliance volume changes using compliance ratio
    # Get Ca and Cc from config
    Ca, Cc = 1.0, 1.0
    for bc in config[str_bc]:
        if bc["bc_name"] == "BC_COR":
            Ca = bc["bc_values"].get("Ca", 1.0)
            Cc = bc["bc_values"].get("Cc", 1.0)
            break

    # Approximate volume split based on compliance ratio
    total_C = Ca + Cc
    if total_C > 0:
        ca_fraction = Ca / total_C
        cc_fraction = Cc / total_C
        metrics['dV_a'] = metrics['dV_total'] * ca_fraction
        metrics['dV_c'] = metrics['dV_total'] * cc_fraction
    else:
        metrics['dV_a'] = metrics['dV_total'] * 0.5
        metrics['dV_c'] = metrics['dV_total'] * 0.5

    # ---- "Sloshing" / recirculated volume quantification (per Stendahl et al.) ----
    # The paper's central observation: per-cycle arterial inflow (~1% EDV) is far
    # smaller than the phasic tissue-volume swing (~6-10% EDV). The difference is
    # blood that recirculates ("sloshes") into and out of the intramyocardial
    # compartment from the venous side (and, physiologically, retrograde to
    # epicardial arteries) rather than being delivered anew through the LAD.
    #
    #   V_slosh = (phasic IM volume swing) - (net arterial inflow per cycle)
    #
    # The compliance that must hold this oscillating volume is C ~ V_slosh / dPim.
    net_arterial_in = metrics['Q_in_net']            # ml/cycle/g (net LAD throughput)
    metrics['V_slosh'] = metrics['dV_total'] - net_arterial_in   # ml/g
    metrics['sloshing_fraction'] = (
        metrics['V_slosh'] / metrics['dV_total'] if metrics['dV_total'] > 0 else 0.0
    )

    # Intramyocardial pressure swing dPim [mmHg] (Pim drives the compliances)
    Pim_swing_mmHg = None
    for bc in config[str_bc]:
        if bc["bc_name"] == "BC_COR" and "Pim" in bc["bc_values"]:
            Pim = np.array(bc["bc_values"]["Pim"], dtype=float)
            Pim_swing_mmHg = (Pim.max() - Pim.min()) * Ba_to_mmHg
            break
    metrics['Pim_swing'] = Pim_swing_mmHg
    # Required compliance to store the sloshing volume against the Pim swing
    # [ml/mmHg/g]; compare to the fitted Cc.
    if Pim_swing_mmHg and Pim_swing_mmHg > 0:
        metrics['Cc_required'] = metrics['V_slosh'] / Pim_swing_mmHg
    else:
        metrics['Cc_required'] = np.nan
    # Cc is in CGS [ml/Ba] per gram; ml/mmHg = ml/Ba * (Ba/mmHg) = Cc * mmHg_to_Ba
    metrics['Cc_fitted'] = Cc * mmHg_to_Ba  # CGS -> ml/mmHg/g

    # ---- Fraction-of-EDV metrics (paper Fig. 4 convention) ----
    if edv and edv > 0:
        metrics['edv'] = edv
        for key in ['dV_total', 'V_slosh', 'V_in_forward', 'Q_in_net',
                    'V_Rv1_backflow', 'V_im_forward', 'dV_a', 'dV_c']:
            metrics[key + '_frac_edv'] = metrics[key] * MYO_DENSITY

    return metrics


def compute_all_volume_metrics(all_data, all_config, node_in):
    """Compute volume metrics for all animals and conditions.

    Args:
        all_data: Dict of {animal: {study: {"o": orig_data, "s": smooth_data}}}
        all_config: Dict of {animal: {study: config}}
        node_in: Node name for inlet flow extraction

    Returns:
        dict: {animal: {study: {metric: value}}}
    """
    all_metrics = {}

    for animal in all_config:
        all_metrics[animal] = {}
        for study in all_config[animal]:
            print(f"Computing volume metrics for DSEA{animal:02d} {study}...")
            edv = all_data.get(animal, {}).get(study, {}).get("s", {}).get("edv")
            metrics = compute_volume_metrics(all_config[animal][study], node_in, edv=edv)
            all_metrics[animal][study] = metrics

    return all_metrics


def print_normalized_statistics(stats, studies):
    """Print normalized parameter statistics in a formatted table.

    Args:
        stats: Statistics dict from compute_normalized_statistics
        studies: List of study names
    """
    print("\n" + "=" * 100)
    print("PARAMETER CHANGES NORMALIZED BY BASELINE")
    print("=" * 100)

    # Parameters to report
    params_to_report = [
        ("BC_COR", "Ra1"),
        ("BC_COR", "Ra2"),
        ("BC_COR", "Rv1"),
        ("BC_COR", "Ca"),
        ("BC_COR", "Cc"),
        ("BC_COR", "Ra2_min"),
        ("BC_COR", "Ra2_max"),
        ("BC_COR", "_ratio_Ra2"),
        ("BC_COR", "tc1"),
        ("BC_COR", "tc2"),
        ("BC_COR", "T_vc"),
        ("BC_COR", "T_vr"),
        ("BC_COR", "_Pim_scale"),
        ("BC_COR", "_Pim_dpdt"),
        ("global", "residual"),
    ]

    # Print header
    header = f"{'Parameter':<20}"
    for study in studies:
        if study != "baseline":
            header += f"{study:<20}"
    print(header)
    print("-" * 100)

    # Print each parameter
    for param in params_to_report:
        param_name = param[1]
        row = f"{param_name:<20}"

        for study in studies:
            if study == "baseline":
                continue
            if study in stats and param in stats[study]:
                s = stats[study][param]
                row += f"{s['mean']:.2f} +/- {s['std']:.2f}  "
            else:
                row += f"{'N/A':<20}"

        print(row)

    print("=" * 100)
    print("Values shown as mean +/- std (ratio to baseline, 1.0 = no change)")
    print()


def print_literature_comparison(all_optimized):
    """Print our baseline parameters vs Kim et al. 2010, being explicit about units.

    Our parameters are PER GRAM. Kim Table 3 values are PER TERRITORY in paper units
    (R: 1e3 dyn.s/cm^5, C: 1e-6 cm^5/dyn). We scale Kim to per gram with a territory
    mass M (R_per_g = R_terr*M; C_per_g = C_terr/M) and show a sensitivity range,
    since M is the dominant unit uncertainty and is not reported by Kim.
    """
    from utils import convert_units
    kim_path = "models/kim10b_table3.json"
    if not os.path.isfile(kim_path):
        print("Kim et al. data not found, skipping literature comparison.")
        return
    kim = json.load(open(kim_path))
    normal = [k for k in kim if len(k) == 1]  # exclude stenosis cases (c*, d*, ...)

    # (our_param, kim_field, paper->cgs scale)
    pmap = [("Ra1", "Ra", 1e3), ("Ra2_mean", "Ra-micro", 1e3), ("Rv1", "Rv", 1e3),
            ("Ca", "Ca", 1e-6), ("Cc", "Cim", 1e-6)]

    # Collect our baseline values (cgs, per gram)
    ours = defaultdict(list)
    for animal, adata in all_optimized.items():
        b = adata.get("baseline", {})
        if ("BC_COR", "Ra2_min") in b and ("BC_COR", "Ra2_max") in b:
            b = dict(b)
            b[("BC_COR", "Ra2_mean")] = np.sqrt(b[("BC_COR", "Ra2_min")] * b[("BC_COR", "Ra2_max")])
        for p, _, _ in pmap:
            if ("BC_COR", p) in b:
                ours[p].append(b[("BC_COR", p)])

    masses = sorted({KIM_LIT_MASS_LO, 10.0, KIM_LIT_MASS_HI})
    print("\n" + "=" * 100)
    print("LITERATURE COMPARISON (baseline, per gram of myocardium) vs Kim et al. 2010")
    print("=" * 100)
    print("Resistances in Wood units*g [mmHg*min/L per g]; compliances in ml/mmHg per g.")
    hdr = f"{'Param':<10}{'Ours (mean±std)':<24}"
    for m in masses:
        hdr += f"{'Kim/M=' + str(int(m)) + 'g':<18}"
    print(hdr)
    print("-" * 100)
    for p, kf, pscale in pmap:
        zerod = "R" if p.startswith("R") else "C"
        disp_unit = "wood" if zerod == "R" else "wood"  # wood maps C to ml/mmHg
        row = f"{p:<10}"
        if ours.get(p):
            ov = np.array(ours[p])
            oc = convert_units(zerod, ov, disp_unit)[0]
            row += f"{np.mean(oc):.3g} ± {np.std(oc):.2g}".ljust(24)
        else:
            row += f"{'N/A':<24}"
        kim_terr_cgs = np.array([kim[v][kf]["R"] * pscale for v in normal if kf in kim[v]])
        for m in masses:
            scale = m if zerod == "R" else 1.0 / m
            kim_pg = convert_units(zerod, kim_terr_cgs * scale, disp_unit)[0]
            row += f"{np.mean(kim_pg):.3g}".ljust(18)
        print(row)
    print("=" * 100)
    print(f"Kim territory mass M is unknown; shown for M in {masses} g. "
          f"Our Ra2_mean ~ Kim Ra-micro; Ra1 ~ Kim Ra; Rv1 ~ Kim Rv; Cc ~ Kim Cim.\n")


# Territory-mass sensitivity bounds for the literature comparison (see plotting.py)
KIM_LIT_MASS_LO = 5.0
KIM_LIT_MASS_HI = 20.0


def main():
    """Main entry point."""
    animals = [8, 10, 15, 16]
    studies = ["baseline", "mild_sten", "mild_sten_dob", "mod_sten", "mod_sten_dob"]

    # Optimization weights for least squares on flow and volume curves
    weight_flow = 1.0
    weight_volume = 1.0
    weight_flow_min = 0.0    # Disabled - using full curve matching
    weight_flow_mean = 0.0   # Disabled - using full curve matching
    weight_vol_min = 0.0     # Disabled - using full curve matching

    # Volume matching mode: "curve" for least squares on full flow and volume curves
    volume_mode = "curve"

    # Loss function: "linear" for standard least squares
    loss = "linear"

    # Disable regularization for unbiased fits
    reg_scale = 0.0

    all_optimized = {}
    all_data = {}
    all_config = {}

    for animal in animals:
        result = process_animal(animal, studies, weight_flow, weight_volume, weight_flow_min,
                               weight_flow_mean=weight_flow_mean, weight_vol_min=weight_vol_min,
                               reg_scale=reg_scale, volume_mode=volume_mode, loss=loss,
                               pim_mode=PIM_MODE)
        if result[0]:
            optimized, data, config = result
            all_optimized[animal] = optimized
            all_data[animal] = data
            all_config[animal] = config

    # Generate summary plots
    if len(all_optimized) > 1:
        plot_parameters_multi(all_optimized, studies, MODEL_NAME)
        plot_parameters_normalized(all_optimized, studies, MODEL_NAME)
        plot_pim_parameters(all_optimized, studies, MODEL_NAME)

    if all_data:
        plot_combined_results(all_data, all_config, MODEL_NAME, get_sim_qv, NODE_IN, get_name)
        plot_internal_flows(all_data, all_config, MODEL_NAME, get_sim_internal_flows, NODE_IN, get_name)

        # Compute and plot volume metrics
        all_volume_metrics = compute_all_volume_metrics(all_data, all_config, NODE_IN)
        if all_volume_metrics:
            plot_volume_metrics(all_volume_metrics, studies, MODEL_NAME)
            plot_volume_balance(all_volume_metrics, studies, MODEL_NAME)
            plot_volume_vs_inflow(all_volume_metrics, studies, MODEL_NAME)
            plot_sloshing_capacity(all_volume_metrics, studies, MODEL_NAME)

    plot_circuit_diagram(MODEL_NAME)

    # Compute and print normalized parameter statistics
    if all_optimized:
        stats = compute_normalized_statistics(all_optimized, studies)
        print_normalized_statistics(stats, studies)
        print_literature_comparison(all_optimized)


if __name__ == "__main__":
    main()
