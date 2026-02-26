#!/usr/bin/env python3
"""Simulation wrapper functions for coronary models."""

import numpy as np
import pysvzerod
from scipy.integrate import cumulative_trapezoid

from utils import str_param, str_time, str_bc


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

        bc_type = config["boundary_conditions"][1]["bc_type"]
        if bc_type == "CORONARY_DETAILED":
            # For CORONARY_DETAILED, use V_c as the intramyocardial volume
            v_sim = sim[sim["name"] == "V_c:BC_COR"]["y"].to_numpy()
            if not v_sim.size:
                raise ValueError("Volume result not found at V_c:BC_COR")
        elif "CORONARY" in bc_type:
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


def get_sim_internal_flows(config, node_in):
    """Run simulation and extract flows at different locations in the coronary model.

    For the CORONARY_VAR_RES model circuit:
        P_in --> Ra1 --> [Ca] --> Ra2 --> [Cc] --> Rv1 --> Pv

    Since pysvzerod doesn't output internal BC flows, we compute them:
        Q_in: Inlet flow (directly from simulation)
        Q_out: Outlet flow = Q_in - dV_im/dt (mass conservation)
        Q_mid: Mid flow = Q_in - dV_Ca/dt (estimated from compliance ratio)

    Args:
        config: svZeroDSolver configuration
        node_in: Node name for inlet flow extraction

    Returns:
        dict: {
            't': time array,
            'Q_in': inlet flow (through Ra1),
            'Q_mid': mid flow (through Ra2, estimated),
            'Q_out': outlet flow (through Rv1),
            'V_im': intramyocardial volume
        }
    """
    try:
        sim = pysvzerod.simulate(config)

        # Extract time
        nt = config[str_param][str_time]
        tmax = config["boundary_conditions"][0]["bc_values"]["t"][-1]
        t = np.linspace(0.0, tmax, nt)
        dt = t[1] - t[0]

        # Inlet flow
        q_in = sim[sim["name"] == "flow:" + node_in]["y"].to_numpy()
        if not q_in.size:
            raise ValueError(f"Flow result not found at flow:{node_in}")

        # Intramyocardial volume
        v_im = sim[sim["name"] == "volume_im:BC_COR"]["y"].to_numpy()
        if not v_im.size:
            raise ValueError("Volume result not found at volume_im:BC_COR")

        # Get model parameters for compliance ratio
        Ca, Cc = 1.0, 1.0
        for bc in config[str_bc]:
            if bc["bc_name"] == "BC_COR":
                Ca = bc["bc_values"].get("Ca", 1.0)
                Cc = bc["bc_values"].get("Cc", 1.0)
                break

        # Compute dV_im/dt
        dVim_dt = np.gradient(v_im, dt)

        # Outlet flow: Q_out = Q_in - dV_im/dt
        q_out = q_in - dVim_dt

        # Mid flow: estimate dV_Ca/dt assuming volume splits by compliance ratio
        ca_fraction = Ca / (Ca + Cc) if (Ca + Cc) > 0 else 0.5
        q_mid = q_in - ca_fraction * dVim_dt

        return {
            't': t,
            'Q_in': q_in,
            'Q_mid': q_mid,
            'Q_out': q_out,
            'V_im': v_im - v_im[0],
        }

    except Exception as e:
        print(f"Simulation error: {e}")
        nt = config[str_param][str_time]
        return {
            't': np.zeros(nt),
            'Q_in': np.zeros(nt),
            'Q_mid': np.zeros(nt),
            'Q_out': np.zeros(nt),
            'V_im': np.zeros(nt),
        }


def get_sim_detailed(config, node_in):
    """Run simulation with CORONARY_DETAILED BC and extract all internal quantities.

    For the CORONARY_DETAILED model circuit:
        P_in --> Ra1 --> P_a --> Ra2 --> P_c --> Rv1 --> Pv
                         |             |
                         Ca            Cc
                         |             |
                        GND           Pim

    Internal variables directly from solver:
        V_a, V_c: Compliance volumes
        P_a, P_c: Internal pressures

    Computed flows (using Ohm's law):
        Q_in: Inlet flow (through Ra1)
        Q_Ra2: Flow through Ra2 = (P_a - P_c) / Ra2
        Q_Rv1: Flow through Rv1 = (P_c - P_v) / Rv1

    Args:
        config: svZeroDSolver configuration (must use CORONARY_DETAILED BC type)
        node_in: Node name for inlet flow extraction

    Returns:
        dict: {
            't': time array,
            'Q_in': inlet flow (through Ra1),
            'Q_Ra2': flow through Ra2 (between Ca and Cc),
            'Q_Rv1': flow through Rv1 (outflow to venous),
            'V_a': arterial compliance volume,
            'V_c': capillary compliance volume,
            'P_a': pressure at arterial compliance,
            'P_c': pressure at capillary compliance,
        }
    """
    try:
        sim = pysvzerod.simulate(config)

        # Extract time
        nt = config[str_param][str_time]
        tmax = config["boundary_conditions"][0]["bc_values"]["t"][-1]
        t = np.linspace(0.0, tmax, nt)

        # Inlet flow
        q_in = sim[sim["name"] == "flow:" + node_in]["y"].to_numpy()
        if not q_in.size:
            raise ValueError(f"Flow result not found at flow:{node_in}")

        # Internal pressures and volumes from CORONARY_DETAILED
        V_a = sim[sim["name"] == "V_a:BC_COR"]["y"].to_numpy()
        V_c = sim[sim["name"] == "V_c:BC_COR"]["y"].to_numpy()
        P_a = sim[sim["name"] == "P_a:BC_COR"]["y"].to_numpy()
        P_c = sim[sim["name"] == "P_c:BC_COR"]["y"].to_numpy()

        if not V_a.size or not V_c.size or not P_a.size or not P_c.size:
            raise ValueError("Internal variables not found. Ensure bc_type is CORONARY_DETAILED")

        # Get model parameters for flow calculation
        Ra2, Rv1, P_v = 1.0, 1.0, 0.0
        for bc in config[str_bc]:
            if bc["bc_name"] == "BC_COR":
                Ra2 = bc["bc_values"].get("Ra2", 1.0)
                Rv1 = bc["bc_values"].get("Rv1", 1.0)
                P_v = bc["bc_values"].get("P_v", 0.0)
                break

        # Compute internal flows using Ohm's law
        Q_Ra2 = (P_a - P_c) / Ra2
        Q_Rv1 = (P_c - P_v) / Rv1

        return {
            't': t,
            'Q_in': q_in,
            'Q_Ra2': Q_Ra2,
            'Q_Rv1': Q_Rv1,
            'V_a': V_a - V_a[0],
            'V_c': V_c - V_c[0],
            'P_a': P_a,
            'P_c': P_c,
        }

    except Exception as e:
        print(f"Simulation error: {e}")
        nt = config[str_param][str_time]
        zeros = np.zeros(nt)
        return {
            't': zeros,
            'Q_in': zeros,
            'Q_Ra2': zeros,
            'Q_Rv1': zeros,
            'V_a': zeros,
            'V_c': zeros,
            'P_a': zeros,
            'P_c': zeros,
        }
