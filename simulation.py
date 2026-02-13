#!/usr/bin/env python3
"""Simulation wrapper functions for coronary models."""

import numpy as np
import pysvzerod
from scipy.integrate import cumulative_trapezoid

from utils import str_param, str_time


def get_sim_qv(config, node_in):
    """Run simulation and extract flow and volume.

    Args:
        config: svZeroDSolver configuration
        node_in: Node name for flow extraction

    Returns:
        tuple: (flow, volume) arrays
    """
    try:
        sim = pysvzerod.simulate(config)

        q_sim = sim[sim["name"] == "flow:" + node_in]["y"].to_numpy()
        if not q_sim.size:
            raise ValueError(f"Flow result not found at flow:{node_in}")

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
        print(f"Unexpected error: {type(e).__name__}: {e}")
        nt = config[str_param][str_time]
        return np.zeros(nt), np.zeros(nt)
