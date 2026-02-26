#!/usr/bin/env python3
"""Optimization functions for coronary model parameter fitting."""

import numpy as np
from scipy.optimize import least_squares

from utils import set_params, get_params, get_param_map
from simulation import get_sim_qv


def get_objective(ref, sim):
    """Compute normalized objective function."""
    return (ref - sim) / np.mean(ref)


def extract_flow_features(t, q):
    """Extract features from flow curve (legacy version).

    Args:
        t: Time array
        q: Flow array

    Returns:
        dict with flow features
    """
    return {
        "mean": np.mean(q),
        "max": np.max(q),
        "min": np.min(q),
        "range": np.max(q) - np.min(q),
        "time_to_max": t[np.argmax(q)] - t[0],
        "time_to_min": t[np.argmin(q)] - t[0],
        "integral": np.trapz(q, t),
        "std": np.std(q),
    }


def extract_robust_flow_features(t, q, T_sys):
    """Extract clinically-validated robust flow features.

    Based on literature review of coronary flow assessment metrics:
    - DSFR: Diastolic-to-Systolic Flow Ratio (validated against FFR)
    - PI: Pulsatility Index (standard Doppler metric)
    - SV: Stroke Volume / Flow Integral (robust to noise)
    - DFF: Diastolic Flow Fraction (normally ~70-80% in LAD)

    Args:
        t: Time array
        q: Flow array
        T_sys: Systolic duration (time to minimum volume)

    Returns:
        dict with robust flow features (all dimensionless except SV)
    """
    t0 = t[0]
    T_cycle = t[-1] - t0
    eps = 1e-10

    # Identify systolic and diastolic phases
    idx_sys = (t - t0) < T_sys
    idx_dia = (t - t0) >= T_sys

    # Mean flows in each phase
    q_mean = np.mean(q)
    q_mean_sys = np.mean(q[idx_sys]) if np.any(idx_sys) else q_mean
    q_mean_dia = np.mean(q[idx_dia]) if np.any(idx_dia) else q_mean

    # Integrals
    sv_total = np.trapz(q, t)
    sv_dia = np.trapz(q[idx_dia], t[idx_dia]) if np.any(idx_dia) else 0.0

    return {
        # Diastolic-to-Systolic Flow Ratio (normal LAD ~2.0, stenotic <1.5)
        "DSFR": q_mean_dia / (q_mean_sys + eps),
        # Pulsatility Index (dimensionless)
        "PI": (np.max(q) - np.min(q)) / (np.abs(q_mean) + eps),
        # Stroke Volume / Flow Integral
        "SV": sv_total,
        # Diastolic Flow Fraction (normal ~0.7-0.8)
        "DFF": sv_dia / (sv_total + eps),
    }


def extract_volume_features(t, v, T_vc=None, T_vr=None):
    """Extract features from volume curve (legacy version).

    Args:
        t: Time array
        v: Volume array
        T_vc: Ventricular contraction time (systole duration)
        T_vr: Ventricular relaxation time (diastole duration)

    Returns:
        dict with volume features
    """
    v_min = np.min(v)
    v_max = np.max(v)
    v_range = v_max - v_min

    t0 = t[0]
    dt = t[1] - t[0]

    if T_vc is None:
        T_vc = t[np.argmin(v)] - t0
    if T_vr is None:
        T_vr = t[-1] - t[np.argmin(v)]

    idx_sys_end = min(int(T_vc / dt), len(v) - 1)
    if idx_sys_end > 1:
        dvdt_sys = (v[idx_sys_end] - v[0]) / T_vc
    else:
        dvdt_sys = 0.0

    idx_dia_start = idx_sys_end
    idx_dia_end = min(int((T_vc + T_vr) / dt), len(v) - 1)
    if idx_dia_end > idx_dia_start and T_vr > 0:
        dvdt_dia = (v[idx_dia_end] - v[idx_dia_start]) / T_vr
    else:
        dvdt_dia = 0.0

    return {
        "v_range": v_range,
        "dvdt_sys": dvdt_sys,
        "dvdt_dia": dvdt_dia,
    }


def extract_robust_volume_features(t, v, T_sys):
    """Extract clinically-validated robust volume features.

    Based on literature review of myocardial volume assessment:
    - VR: Volume Ratio (systolic/diastolic, typically 0.85-0.95)
    - FVC: Fractional Volume Change (analogous to ejection fraction)
    - Rate_sys: Normalized systolic contraction rate
    - Rate_dia: Normalized diastolic relaxation rate

    All features are dimensionless for cross-animal comparison.

    Args:
        t: Time array
        v: Volume array (with baseline at 0)
        T_sys: Systolic duration (time to minimum volume)

    Returns:
        dict with robust volume features (all dimensionless)
    """
    t0 = t[0]
    T_cycle = t[-1] - t0
    T_dia = T_cycle - T_sys
    eps = 1e-10

    v_0 = v[0]  # Initial volume (should be ~0 after baseline subtraction)
    v_min = np.min(v)  # Minimum volume (most negative)
    v_max = np.max(v)  # Maximum volume
    v_end = v[-1]  # End-of-cycle volume

    # For volume starting at 0 and going negative:
    # v_max ~ 0, v_min < 0
    # Volume range is |v_min - v_max| = |v_min|

    v_range = np.abs(v_min - v_max)

    return {
        # Volume Ratio: min/max (for negative volumes, this is >1, so use abs)
        # Represents fractional compression
        "VR": np.abs(v_min) / (np.abs(v_max) + eps) if v_max != 0 else np.abs(v_min) / (v_range + eps),
        # Fractional Volume Change (analogous to strain)
        "FVC": v_range / (np.abs(v_0) + v_range + eps),
        # Normalized systolic rate: how fast does volume decrease relative to range
        "Rate_sys_norm": (v_0 - v_min) / (v_range * T_sys + eps),
        # Normalized diastolic rate: how fast does volume recover
        "Rate_dia_norm": (v_end - v_min) / (v_range * T_dia + eps) if T_dia > 0 else 0.0,
    }


# Default regularization weights - stronger for structural parameters, weaker for physiological
# Format: parameter_name -> (reference_value, weight)
# Weight of 0 means no regularization
# Reference values updated based on fitted results from 4 animals x 5 studies
DEFAULT_REGULARIZATION = {
    # Structural parameters - should be consistent across studies
    "L": (1e2, 0.5),              # Inductance: fitted mean ~100, already consistent
    "R_poiseuille": (1e4, 1.0),   # Proximal resistance: fitted mean ~4.5e4, increased weight

    # Physiological parameters - allow more variation
    "Ra1": (3e6, 0.1),            # Arterial resistance: fitted mean ~3e6
    "Ra2": (3e6, 0.1),            # Mid-arterial resistance: fitted mean ~3e6
    "Rv1": (3e5, 0.1),            # Venous resistance: fitted mean ~3e5
    "_ratio_Ra2": (1.5, 0.2),     # Ra2 ratio: moderate regularization

    # Time constants - Ca = tc1/Ra1, Cc = tc2/Rv1
    # Strong regularization to reduce Ca/Cc variability
    "tc1": (0.05, 1.0),           # tc1 = Ra1*Ca: target ~50ms, strong regularization
    "tc2": (0.15, 0.5),           # tc2 = Rv1*Cc: target ~150ms, moderate-strong weight

    # Timing parameters - derived from data, weak regularization
    "T_vc": (0.35, 0.1),          # Contraction time: fitted mean ~0.37s
    "T_vr": (0.3, 0.1),           # Relaxation time: fitted mean ~0.32s

    # Pim parameters - allow variation
    "_Pim_scale": (1.0, 0.1),     # CEP component
    "_Pim_dpdt": (0.2, 0.1),      # VE component
}


def optimize(config, p0, data, node_in, verbose=0, weight_flow=2.0, weight_volume=0.5,
             weight_flow_min=5.0, weight_flow_mean=1.0, weight_vol_min=2.0,
             regularization=None, reg_scale=1.0, volume_mode="robust",
             loss="linear"):
    """Optimize model parameters using least squares with optional regularization.

    Args:
        config: svZeroDSolver configuration
        p0: Parameter dictionary with (val, min, max, map_type) tuples
        data: Dictionary with smoothed data
        node_in: Node name for flow extraction
        verbose: Verbosity level
        weight_flow: Weight for flow objective
        weight_volume: Weight for volume objective
        weight_flow_min: Weight for flow minimum matching
        weight_flow_mean: Weight for mean LAD flow matching
        weight_vol_min: Weight for volume at minimum (from zero to min)
        regularization: Dict of {param_name: (ref_value, weight)} for regularization.
                       If None, uses DEFAULT_REGULARIZATION.
                       Set to {} to disable regularization.
        reg_scale: Global scaling factor for all regularization weights
        volume_mode: Volume matching mode:
                    - "robust": Clinically-validated features (DSFR, PI, VR, FVC) - recommended
                    - "minmax": Match only min and max volume (2 features)
                    - "rates": Match volume range + mean sys/dia dV/dt rates (3 features)
                    - "curve": Match full volume curve (~201 points)
                    - "features": Legacy feature-based matching
        loss: Loss function for least_squares:
                    - "linear": Standard L2 loss (default)
                    - "soft_l1": Smooth L1 loss, robust to outliers
                    - "huber": Huber loss, robust to outliers
                    - "cauchy": Cauchy loss, very robust to outliers

    Returns:
        tuple: (config, residual_norm)
    """
    qref = data["s"]["qlad"]
    vref = data["s"]["vmyo"]
    tref = data["s"]["t"]

    q_min_idx = np.argmin(qref)
    q_min_val = qref[q_min_idx]
    q_mean_ref = np.mean(qref)
    q_mean_abs = np.mean(np.abs(qref)) + 1e-10

    # Volume at minimum (from zero baseline)
    v_min_ref = np.min(vref)

    # Cycle time (always needed)
    t_cycle = tref[-1] - tref[0]

    # Get initial timing estimates from p0
    T_vc_init = p0.get(("BC_COR", "T_vc"), (0.35, None, None, None))[0]
    T_vr_init = p0.get(("BC_COR", "T_vr"), (0.30, None, None, None))[0]

    # Extract reference volume features for feature-based matching
    if volume_mode in ("minmax", "rates", "features"):
        vref_features = extract_volume_features(tref, vref, T_vc_init, T_vr_init)
        v_range_ref = vref_features["v_range"] + 1e-10  # normalization factor
        v_mean_ref = np.mean(np.abs(vref)) + 1e-10

    # Extract reference flow features for feature-based matching
    if volume_mode == "features":
        qref_features = extract_flow_features(tref, qref)

    # Extract robust features for "robust" mode (clinically-validated metrics)
    if volume_mode == "robust":
        qref_robust = extract_robust_flow_features(tref, qref, T_vc_init)
        vref_robust = extract_robust_volume_features(tref, vref, T_vc_init)

    # Set up regularization
    if regularization is None:
        regularization = DEFAULT_REGULARIZATION

    # Build regularization info for each parameter in p0
    reg_info = []  # List of (ref_mapped, weight, map_type) for each parameter
    param_names = list(p0.keys())
    for (bc_name, param_name) in param_names:
        if param_name in regularization:
            ref_val, weight = regularization[param_name]
            _, _, _, map_type = p0[(bc_name, param_name)]
            forward_map, _ = get_param_map(map_type)
            ref_mapped = forward_map(ref_val)
            reg_info.append((ref_mapped, weight * reg_scale, map_type))
        else:
            reg_info.append((None, 0, None))

    def cost_function(p):
        pset, _ = set_params(config, p0, p)

        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")

        q_sim, v_sim = get_sim_qv(config, node_in=node_in)

        # Check for simulation failure (returns zeros)
        if np.allclose(q_sim, 0) and np.allclose(v_sim, 0):
            # Return large penalty to discourage these parameters
            n_features = 8 if volume_mode == "robust" else 10  # Approximate number of features
            n_penalties = 3  # calibration targets
            n_reg = sum(1 for _, w, _ in reg_info if w > 0)
            penalty = 100.0 * np.ones(n_features + n_penalties + n_reg)
            if verbose:
                print("SIM_FAIL")
            return penalty

        # Flow and volume objectives depend on mode
        if volume_mode == "robust":
            # Clinically-validated robust features (dimensionless)
            # Get current T_vc from config
            T_vc_curr = T_vc_init
            for bc in config.get("boundary_conditions", []):
                if bc.get("bc_name") == "BC_COR":
                    T_vc_curr = bc.get("bc_values", {}).get("T_vc", T_vc_init)
                    break

            qsim_robust = extract_robust_flow_features(tref, q_sim, T_vc_curr)
            vsim_robust = extract_robust_volume_features(tref, v_sim, T_vc_curr)

            # Flow features (all dimensionless except SV)
            # DSFR: target ~2.0 for normal LAD
            # PI: target ~1.5-2.5
            # DFF: target ~0.7-0.8
            flow_penalties = [
                (qref_robust["DSFR"] - qsim_robust["DSFR"]) / (qref_robust["DSFR"] + 1e-10),
                (qref_robust["PI"] - qsim_robust["PI"]) / (qref_robust["PI"] + 1e-10),
                (qref_robust["SV"] - qsim_robust["SV"]) / (np.abs(qref_robust["SV"]) + 1e-10),
                (qref_robust["DFF"] - qsim_robust["DFF"]) / (qref_robust["DFF"] + 1e-10),
            ]
            obj_q = weight_flow * np.array(flow_penalties)

            # Volume features (all dimensionless)
            vol_penalties = [
                (vref_robust["VR"] - vsim_robust["VR"]) / (vref_robust["VR"] + 1e-10),
                (vref_robust["FVC"] - vsim_robust["FVC"]) / (vref_robust["FVC"] + 1e-10),
                (vref_robust["Rate_sys_norm"] - vsim_robust["Rate_sys_norm"]) / (np.abs(vref_robust["Rate_sys_norm"]) + 1e-10),
                (vref_robust["Rate_dia_norm"] - vsim_robust["Rate_dia_norm"]) / (np.abs(vref_robust["Rate_dia_norm"]) + 1e-10),
            ]
            obj_v = weight_volume * np.array(vol_penalties)

        elif volume_mode == "features":
            # Legacy feature-based matching
            qsim_features = extract_flow_features(tref, q_sim)
            vsim_features = extract_volume_features(tref, v_sim, T_vc_init, T_vr_init)

            # Flow features (normalized by mean absolute flow)
            flow_penalties = [
                (qref_features["mean"] - qsim_features["mean"]) / q_mean_abs,
                (qref_features["max"] - qsim_features["max"]) / q_mean_abs,
                (qref_features["min"] - qsim_features["min"]) / q_mean_abs,
                (qref_features["integral"] - qsim_features["integral"]) / (q_mean_abs * t_cycle),
                (qref_features["time_to_max"] - qsim_features["time_to_max"]) / t_cycle,
            ]
            obj_q = weight_flow * np.array(flow_penalties)

            # Volume features (normalized by volume range)
            vol_penalties = [
                (vref_features["v_range"] - vsim_features["v_range"]) / v_range_ref,
                (vref_features["dvdt_sys"] - vsim_features["dvdt_sys"]) / (v_range_ref / t_cycle),
                (vref_features["dvdt_dia"] - vsim_features["dvdt_dia"]) / (v_range_ref / t_cycle),
            ]
            obj_v = weight_volume * np.array(vol_penalties)

        else:
            # Curve-based flow matching
            obj_q = weight_flow * get_objective(qref, q_sim)

        # Volume objective: different modes (for non-feature modes)
        if volume_mode in ("robust", "features"):
            pass  # Already handled above
        elif volume_mode == "minmax":
            # Simple min/max matching (2 features)
            v_min_sim_mm = np.min(v_sim)
            v_max_sim_mm = np.max(v_sim)
            v_min_ref_mm = np.min(vref)
            v_max_ref_mm = np.max(vref)

            vol_penalties = [
                (v_min_ref_mm - v_min_sim_mm) / v_mean_ref,
                (v_max_ref_mm - v_max_sim_mm) / v_mean_ref,
            ]
            obj_v = weight_volume * np.array(vol_penalties)

        elif volume_mode == "rates":
            # Volume range + mean sys/dia rates (3 features)
            # Get current T_vc, T_vr from the config (updated by set_params)
            T_vc_curr = None
            T_vr_curr = None
            for bc in config.get("boundary_conditions", []):
                if bc.get("bc_name") == "BC_COR":
                    T_vc_curr = bc.get("bc_values", {}).get("T_vc", T_vc_init)
                    T_vr_curr = bc.get("bc_values", {}).get("T_vr", T_vr_init)
                    break
            if T_vc_curr is None:
                T_vc_curr = T_vc_init
            if T_vr_curr is None:
                T_vr_curr = T_vr_init

            # Extract features using current timing parameters
            vsim_features = extract_volume_features(tref, v_sim, T_vc_curr, T_vr_curr)

            # Characteristic rate for normalization (volume range / cycle time)
            rate_scale = v_range_ref / t_cycle

            vol_penalties = [
                # Volume range (insensitive to baseline offset)
                (vref_features["v_range"] - vsim_features["v_range"]) / v_range_ref,
                # Mean systolic dV/dt (contraction, typically negative)
                (vref_features["dvdt_sys"] - vsim_features["dvdt_sys"]) / rate_scale,
                # Mean diastolic dV/dt (relaxation, typically positive)
                (vref_features["dvdt_dia"] - vsim_features["dvdt_dia"]) / rate_scale,
            ]
            obj_v = weight_volume * np.array(vol_penalties)

        else:  # volume_mode == "curve"
            obj_v = weight_volume * get_objective(vref, v_sim)

        obj = np.concatenate((obj_q, obj_v))

        if weight_flow_min > 0:
            q_sim_at_min = q_sim[q_min_idx] if len(q_sim) > q_min_idx else 0
            flow_min_penalty = weight_flow_min * (q_min_val - q_sim_at_min) / q_mean_abs
            obj = np.concatenate((obj, [flow_min_penalty]))

        # Mean LAD flow matching
        if weight_flow_mean > 0:
            q_mean_sim = np.mean(q_sim)
            flow_mean_penalty = weight_flow_mean * (q_mean_ref - q_mean_sim) / q_mean_abs
            obj = np.concatenate((obj, [flow_mean_penalty]))

        # Volume at minimum (from zero baseline)
        if weight_vol_min > 0:
            v_min_sim = np.min(v_sim)
            v_range = np.abs(v_min_ref) + 1e-10  # normalize by absolute min (volume range from zero)
            vol_min_penalty = weight_vol_min * (v_min_ref - v_min_sim) / v_range
            obj = np.concatenate((obj, [vol_min_penalty]))

        # Add regularization penalties
        reg_penalties = []
        for i, (ref_mapped, weight, _) in enumerate(reg_info):
            if weight > 0 and ref_mapped is not None:
                # Penalty in mapped (optimization) space
                penalty = weight * (p[i] - ref_mapped)
                reg_penalties.append(penalty)

        if reg_penalties:
            obj = np.concatenate((obj, reg_penalties))

        if verbose:
            print(f"{np.linalg.norm(obj):.1e}")

        return obj

    initial = get_params(p0)

    if verbose:
        for k in p0.keys():
            print(f"{k[1]}", end="\t")
        print("obj")

    bounds = []
    for param_tuple in p0.values():
        _, pmin, pmax, map_type = param_tuple
        forward_map, _ = get_param_map(map_type)
        bounds.append((forward_map(pmin), forward_map(pmax)))

    res = least_squares(cost_function, initial, bounds=np.array(bounds).T, loss=loss)
    _, param_values = set_params(config, p0, res.x)

    return config, np.linalg.norm(res.fun), param_values
