#!/usr/bin/env python3
"""Compare different fitting strategies for robustness."""

import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict

from utils import read_config, mmHg_to_Ba, set_params
from simulation import get_sim_qv
from optimization import optimize
from process import read_data, smooth_data, setup_model, get_name

# Configuration
MODEL_NAME = "coronary_varres_time"
NODE_IN = "BC_AT:BV_prox"

# Test cases - focus on problematic ones
TEST_CASES = [
    (8, "baseline"),      # Reference (relatively clean)
    (8, "mod_sten"),      # Problematic: 39 flow peaks
    (16, "mod_sten"),     # Problematic: 46 flow peaks, small volume
    (15, "mod_sten"),     # Problematic: 50 flow peaks
]

# Strategies to compare
STRATEGIES = {
    "baseline": {"volume_mode": "curve", "loss": "linear"},
    "soft_l1": {"volume_mode": "curve", "loss": "soft_l1"},
    "features": {"volume_mode": "features", "loss": "linear"},
    "features_soft_l1": {"volume_mode": "features", "loss": "soft_l1"},
}

def compute_fit_quality(data, config, node_in):
    """Compute fit quality metrics."""
    qref = data["s"]["qlad"]
    vref = data["s"]["vmyo"]

    q_sim, v_sim = get_sim_qv(config, node_in=node_in)

    # Flow metrics
    q_rmse = np.sqrt(np.mean((qref - q_sim)**2))
    q_mean_err = abs(np.mean(qref) - np.mean(q_sim))
    q_max_err = abs(np.max(qref) - np.max(q_sim))
    q_min_err = abs(np.min(qref) - np.min(q_sim))

    # Volume metrics
    v_rmse = np.sqrt(np.mean((vref - v_sim)**2))
    v_range_err = abs((vref.max() - vref.min()) - (v_sim.max() - v_sim.min()))
    v_min_err = abs(np.min(vref) - np.min(v_sim))

    return {
        "q_rmse": q_rmse,
        "q_mean_err": q_mean_err,
        "q_max_err": q_max_err,
        "q_min_err": q_min_err,
        "v_rmse": v_rmse,
        "v_range_err": v_range_err,
        "v_min_err": v_min_err,
    }


def run_comparison():
    """Run comparison of fitting strategies."""
    results = defaultdict(dict)

    for animal, study in TEST_CASES:
        case_name = f"DSEA{animal:02d}_{study}"
        print(f"\n{'='*60}")
        print(f"Processing {case_name}")
        print(f"{'='*60}")

        # Load data
        data_o = read_data(animal, study)
        if not data_o:
            print(f"  Skipping - no data")
            continue

        data = {"o": data_o, "s": smooth_data(data_o, 201)}

        for strategy_name, strategy_params in STRATEGIES.items():
            print(f"\n  Strategy: {strategy_name}")
            print(f"  {strategy_params}")

            # Setup model
            config, p0 = setup_model(data)

            # Run optimization
            try:
                config, err = optimize(
                    config, p0, data, NODE_IN, verbose=0,
                    weight_flow=2.0, weight_volume=0.5,
                    weight_flow_min=5.0, weight_flow_mean=5.0, weight_vol_min=10.0,
                    reg_scale=1.0,
                    **strategy_params
                )

                # Compute fit quality
                quality = compute_fit_quality(data, config, NODE_IN)
                quality["residual"] = err

                results[case_name][strategy_name] = quality

                print(f"    Residual: {err:.3f}")
                print(f"    Flow RMSE: {quality['q_rmse']:.3f}, Mean err: {quality['q_mean_err']:.3f}")
                print(f"    Vol RMSE: {quality['v_rmse']:.4f}, Range err: {quality['v_range_err']:.4f}")

                # Save config
                out_file = f"results/{case_name}_{MODEL_NAME}_{strategy_name}.json"
                with open(out_file, "w") as f:
                    json.dump(config, f, indent=2)

            except Exception as e:
                print(f"    ERROR: {e}")
                results[case_name][strategy_name] = {"error": str(e)}

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Fit Quality Comparison")
    print("="*80)

    metrics = ["residual", "q_rmse", "q_mean_err", "v_rmse", "v_range_err", "v_min_err"]

    for metric in metrics:
        print(f"\n{metric.upper()}")
        print("-" * 70)
        header = f"{'Case':<25}"
        for strategy in STRATEGIES.keys():
            header += f"{strategy:>12}"
        print(header)

        for case_name in results.keys():
            row = f"{case_name:<25}"
            for strategy in STRATEGIES.keys():
                if strategy in results[case_name]:
                    val = results[case_name][strategy].get(metric, np.nan)
                    if isinstance(val, float):
                        row += f"{val:>12.4f}"
                    else:
                        row += f"{'ERR':>12}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

    # Compute improvement percentages
    print("\n" + "="*80)
    print("IMPROVEMENT vs BASELINE (%)")
    print("="*80)

    for metric in ["q_rmse", "q_mean_err", "v_rmse", "v_min_err"]:
        print(f"\n{metric}")
        print("-" * 70)

        for case_name in results.keys():
            baseline_val = results[case_name].get("baseline", {}).get(metric, np.nan)
            if np.isnan(baseline_val) or baseline_val == 0:
                continue

            row = f"{case_name:<25}"
            for strategy in STRATEGIES.keys():
                if strategy == "baseline":
                    row += f"{'---':>12}"
                elif strategy in results[case_name]:
                    val = results[case_name][strategy].get(metric, np.nan)
                    if isinstance(val, float) and not np.isnan(val):
                        improvement = (baseline_val - val) / baseline_val * 100
                        row += f"{improvement:>+11.1f}%"
                    else:
                        row += f"{'ERR':>12}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

    return results


if __name__ == "__main__":
    results = run_comparison()
