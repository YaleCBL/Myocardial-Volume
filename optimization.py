#!/usr/bin/env python3
"""Optimization functions for coronary model parameter fitting."""

import numpy as np
from scipy.optimize import least_squares

from utils import set_params, get_params, get_param_map
from simulation import get_sim_qv


def get_objective(ref, sim):
    """Compute normalized objective function."""
    return (ref - sim) / np.mean(ref)


def optimize(config, p0, data, node_in, verbose=0, weight_flow=2.0, weight_volume=0.5, weight_flow_min=5.0):
    """Optimize model parameters using least squares.

    Args:
        config: svZeroDSolver configuration
        p0: Parameter dictionary with (val, min, max, map_type) tuples
        data: Dictionary with smoothed data
        node_in: Node name for flow extraction
        verbose: Verbosity level
        weight_flow: Weight for flow objective
        weight_volume: Weight for volume objective
        weight_flow_min: Weight for flow minimum matching

    Returns:
        tuple: (config, residual_norm)
    """
    qref = data["s"]["qlad"]
    vref = data["s"]["vmyo"]

    q_min_idx = np.argmin(qref)
    q_min_val = qref[q_min_idx]
    q_mean = np.mean(np.abs(qref)) + 1e-10

    def cost_function(p):
        pset = set_params(config, p0, p)

        if verbose:
            for val in pset:
                print(f"{val:.1e}", end="\t")

        q_sim, v_sim = get_sim_qv(config, node_in=node_in)

        obj_q = weight_flow * get_objective(qref, q_sim)
        obj_v = weight_volume * get_objective(vref, v_sim)
        obj = np.concatenate((obj_q, obj_v))

        if weight_flow_min > 0:
            q_sim_at_min = q_sim[q_min_idx] if len(q_sim) > q_min_idx else 0
            flow_min_penalty = weight_flow_min * (q_min_val - q_sim_at_min) / q_mean
            obj = np.concatenate((obj, [flow_min_penalty]))

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

    res = least_squares(cost_function, initial, bounds=np.array(bounds).T)
    set_params(config, p0, res.x)

    return config, np.linalg.norm(res.fun)
