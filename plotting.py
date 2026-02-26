#!/usr/bin/env python3
"""Plotting functions for coronary model fitting."""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict

from utils import Ba_to_mmHg, convert_units, units


def plot_data(animal, data, get_name_func):
    """Plot raw and smoothed measurement data."""
    n_param = len(data.keys())
    _, ax = plt.subplots(
        4, n_param, figsize=(max(n_param * 5, 5), 10)
    )
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
            ax[1, i].plot(
                data[s][k]["t"],
                (data[s][k]["pat"] - data[s][k]["pven"]) * Ba_to_mmHg,
                f"r{mk[k]}",
            )
            for j, loc in enumerate(["myo", "lad"]):
                ax[j + 2, i].plot(data[s][k]["t"], data[s][k]["v" + loc], f"b{mk[k]}")
                axt[j + 2, i].plot(data[s][k]["t"], data[s][k]["q" + loc], f"m{mk[k]}")

    plt.tight_layout()
    plt.savefig(f"plots/{get_name_func(animal)}_data.pdf")
    plt.close()


def plot_results(animal, config, data, get_sim_qv_func, node_in, get_name_func, model_name):
    """Plot simulation results vs measured data."""
    labels = {
        "qlad": "LAD Flow (scaled) [ml/s]",
        "pat": "Arterial Pressure [mmHg]",
        "pven": "Left-Ventricular Pressure [mmHg]",
        "vmyo": "Myocardial Volume [ml]",
    }

    _, axs = plt.subplots(
        len(labels), len(config), figsize=(16, 9)
    )
    if len(config) == 1:
        axs = axs.reshape(-1, 1)

    for j, study in enumerate(config.keys()):
        ti = config[study]["boundary_conditions"][0]["bc_values"]["t"]
        datm = data[study]["s"]
        dats = OrderedDict()

        q_sim, v_sim = get_sim_qv_func(config[study], node_in=node_in)
        dats["qlad"] = q_sim
        dats["pven"] = datm["pven"]
        dats["pat"] = datm["pat"]
        dats["vmyo"] = v_sim

        convert = {"p": Ba_to_mmHg, "v": 1.0, "q": 1.0}

        for i, k in enumerate(dats.keys()):
            axs[i, j].plot(ti, dats[k] * convert[k[0]], "r-", label="simulated")
            axs[i, j].plot(ti, datm[k] * convert[k[0]], "k--", label="measured")
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
    plt.savefig(f"plots/{get_name_func(animal)}_{model_name}_simulated.pdf")
    plt.close()


def plot_parameters(animal, optimized, get_name_func, model_name):
    """Plot optimized parameters for a single animal with independent y-axes."""
    studies = list(optimized.keys())
    n_param = len(optimized[studies[0]].keys())
    _, axes = plt.subplots(1, n_param, figsize=(n_param * 3, 6))
    if n_param == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (param, val) in enumerate(optimized[studies[0]].keys()):
        values = np.array([optimized[s][(param, val)] for s in studies])
        zerod = val[0]
        values, unit, name = convert_units(zerod, values, units)
        ylabel = f"{name} [{unit}]"

        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, axis="y")
        axes[i].bar(range(len(studies)), values)
        axes[i].set_xticks(range(len(studies)))
        axes[i].set_xticklabels(studies, rotation=45)
        axes[i].set_title(f"{get_name_func(animal)} {val}")

    plt.tight_layout()
    plt.savefig(f"plots/{get_name_func(animal)}_{model_name}_parameters.pdf")
    plt.close()


def plot_parameters_multi(animals_optimized, studies, model_name):
    """Plot parameters across multiple animals, grouped by physical quantity with shared y-axes.

    Uses clinical units (mmHg·s/ml for resistance, ml/mmHg for compliance) and includes
    comparison with Kim et al. 2010 (doi:10.1007/s10439-010-0083-6) reference values.
    """
    import json

    first_animal = list(animals_optimized.keys())[0]
    if studies is None:
        studies = list(animals_optimized[first_animal].keys())

    first_study = studies[0]
    n_studies = len(studies)

    # Load Kim et al. 2010 reference data
    kim_data = None
    try:
        with open("models/kim10b_table3.json") as f:
            kim_data = json.load(f)
    except FileNotFoundError:
        print("Warning: kim10b_table3.json not found, skipping reference comparison")

    # Compute Kim et al. ranges (mean ± std across vessels, excluding stenosis cases)
    # Kim values are for total vessel territories (~10g each), our values are per gram
    # To compare: R_per_gram = R_total * territory_mass (resistances in parallel)
    #             C_per_gram = C_total / territory_mass (capacitances in parallel)
    kim_ranges = {}
    KIM_TERRITORY_MASS = 10.0  # Estimated mass of coronary territory in grams
    if kim_data:
        # Only use normal vessels (single letter keys a-k), not stenosis cases (c*, d*, etc.)
        normal_vessels = [k for k in kim_data.keys() if len(k) == 1]

        # Map Kim parameters to our parameters with scaling
        # Kim Table 3 units: R in 10^3 dynes s/cm^5, C in 10^-6 cm^5/dynes
        # First convert to CGS (R *= 1e3, C *= 1e-6), then apply territory mass scaling
        # (our_param, kim_param, paper_to_cgs_scale, territory_mass_scale)
        kim_param_map = {
            'Ra1': ('Ra', 1e3, KIM_TERRITORY_MASS),           # R_per_g = R_cgs * M
            'Ra2_mean': ('Ra-micro', 1e3, KIM_TERRITORY_MASS), # R_per_g = R_cgs * M
            'Rv1': ('Rv', 1e3, KIM_TERRITORY_MASS),           # R_per_g = R_cgs * M
            'Ca': ('Ca', 1e-6, 1.0 / KIM_TERRITORY_MASS),     # C_per_g = C_cgs / M
            'Cc': ('Cim', 1e-6, 1.0 / KIM_TERRITORY_MASS),    # C_per_g = C_cgs / M
        }

        for our_param, (kim_param, paper_scale, mass_scale) in kim_param_map.items():
            # Get rest values (R) from all normal vessels - keep individual values for plotting
            # Apply paper->CGS conversion, then territory mass scaling
            values_cgs = [kim_data[v][kim_param]['R'] * paper_scale * mass_scale for v in normal_vessels if kim_param in kim_data[v]]
            if values_cgs:
                # Convert from CGS to clinical units (same as our data)
                values_clinical, _, _ = convert_units(our_param[0], np.array(values_cgs), "mmHg")
                kim_ranges[our_param] = {
                    'values': values_clinical.tolist(),  # Individual vessel values for scatter plot
                    'mean': np.mean(values_clinical),
                    'std': np.std(values_clinical),
                    'min': np.min(values_clinical),
                    'max': np.max(values_clinical),
                }

    # Compute derived parameters (Ra2_mean, Ra2_ratio) from Ra2_min/Ra2_max
    animals_optimized_extended = {}
    for animal in animals_optimized:
        animals_optimized_extended[animal] = {}
        for study in animals_optimized[animal]:
            animals_optimized_extended[animal][study] = dict(animals_optimized[animal][study])
            study_data = animals_optimized_extended[animal][study]

            # Compute Ra2_mean = sqrt(Ra2_min * Ra2_max) and Ra2_ratio = Ra2_max / Ra2_min
            if ("BC_COR", "Ra2_min") in study_data and ("BC_COR", "Ra2_max") in study_data:
                ra2_min = study_data[("BC_COR", "Ra2_min")]
                ra2_max = study_data[("BC_COR", "Ra2_max")]
                study_data[("BC_COR", "Ra2_mean")] = np.sqrt(ra2_min * ra2_max)
                study_data[("BC_COR", "Ra2_ratio")] = ra2_max / ra2_min if ra2_min > 0 else 1.0

    params = list(animals_optimized_extended[first_animal][first_study].keys())

    # Define parameter groups by physical quantity
    # Each group: (group_name, [param_names], share_y)
    param_groups = [
        ("Resistances", ["Ra1", "Ra2_mean", "Rv1", "R_poiseuille"], False),
        ("Resistance Ratio", ["Ra2_ratio"], False),
        ("Compliances", ["Ca", "Cc"], False),
        ("Inductance", ["L"], False),
        ("Time Constants", ["tc1", "tc2"], False),
        ("Timing", ["T_vc", "T_vr"], False),
        ("Pim Scaling", ["_Pim_scale", "_Pim_dpdt"], False),
        ("Fit Quality", ["residual"], False),
    ]

    # Map each parameter to its group
    param_to_group = {}
    for group_idx, (group_name, param_names, _) in enumerate(param_groups):
        for pname in param_names:
            param_to_group[pname] = group_idx

    # Sort parameters by group, then by order within group
    def get_sort_key(param_tuple):
        param, val = param_tuple
        group_idx = param_to_group.get(val, 99)
        # Get position within group
        for gidx, (_, param_names, _) in enumerate(param_groups):
            if val in param_names:
                return (gidx, param_names.index(val))
        return (99, 0)

    sorted_params = sorted(params, key=get_sort_key)

    # Filter to only include parameters that exist in groups
    sorted_params = [p for p in sorted_params if p[1] in param_to_group]
    n_param = len(sorted_params)

    # Define colors for each animal
    animals = list(animals_optimized_extended.keys())
    animal_colors = plt.cm.tab10(np.linspace(0, 1, len(animals)))
    animal_color_map = {animal: animal_colors[idx] for idx, animal in enumerate(animals)}

    # Determine grid layout based on groups
    # Count params per group
    group_counts = defaultdict(int)
    for param, val in sorted_params:
        group_idx = param_to_group.get(val, 99)
        group_counts[group_idx] += 1

    # Create figure with grouped subplots
    n_rows = len([g for g in group_counts.keys() if group_counts[g] > 0])
    max_cols = max(group_counts.values()) if group_counts else 1

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(max_cols * 3.5, n_rows * 3.5),
                              squeeze=False)

    # Track which axes belong to which group for shared y-axis
    group_axes = defaultdict(list)
    group_data_ranges = defaultdict(lambda: [float('inf'), float('-inf')])

    # First pass: collect data and determine y-ranges for shared axes
    param_data = {}
    for param, val in sorted_params:
        zerod = val[0]
        all_values = []
        all_animals_per_study = []
        for study in studies:
            study_values = []
            study_animals = []
            for animal in animals_optimized_extended.keys():
                if study in animals_optimized_extended[animal] and (param, val) in animals_optimized_extended[animal][study]:
                    study_values.append(animals_optimized_extended[animal][study][(param, val)])
                    study_animals.append(animal)
            all_values.append(study_values)
            all_animals_per_study.append(study_animals)

        flat_values = [v for sublist in all_values for v in sublist]
        if flat_values:
            # Use clinical units (mmHg-based) for this plot
            converted_flat, unit, name = convert_units(zerod, np.array(flat_values), "mmHg")

            converted_values = []
            for study_values in all_values:
                if study_values:
                    converted = convert_units(zerod, np.array(study_values), "mmHg")[0]
                    converted_values.append(converted)
                else:
                    converted_values.append(np.array([]))

            param_data[(param, val)] = {
                'converted_values': converted_values,
                'all_animals_per_study': all_animals_per_study,
                'unit': unit,
                'name': name,
                'zerod': zerod,
                'param_name': val,  # Store parameter name for Kim comparison
            }

            # Update group range
            group_idx = param_to_group.get(val, 99)
            all_conv = np.concatenate([c for c in converted_values if len(c) > 0])
            if len(all_conv) > 0:
                group_data_ranges[group_idx][0] = min(group_data_ranges[group_idx][0], np.min(all_conv))
                group_data_ranges[group_idx][1] = max(group_data_ranges[group_idx][1], np.max(all_conv))

    # Second pass: plot with shared y-axes
    current_row = -1
    current_col = 0
    last_group = -1

    for param, val in sorted_params:
        if (param, val) not in param_data:
            continue

        group_idx = param_to_group.get(val, 99)

        # Move to new row if group changes
        if group_idx != last_group:
            current_row += 1
            current_col = 0
            last_group = group_idx

        ax = axes[current_row, current_col]
        group_axes[group_idx].append(ax)

        data = param_data[(param, val)]
        converted_values = data['converted_values']
        all_animals_per_study = data['all_animals_per_study']
        unit = data['unit']
        name = data['name']
        zerod = data['zerod']
        param_name = data['param_name']

        means = [np.mean(c) if len(c) > 0 else np.nan for c in converted_values]
        stds = [np.std(c) if len(c) > 0 else np.nan for c in converted_values]

        # Only show y-label on first column
        if current_col == 0:
            group_name = param_groups[group_idx][0]
            ax.set_ylabel(f"{name} [{unit}]")

        ax.grid(True, axis="y", alpha=0.3)

        # Check if Kim et al. data available for this parameter
        has_kim = kim_ranges and param_name in kim_ranges

        # Plot individual points for our data
        for study_idx, (converted, study_animals) in enumerate(zip(converted_values, all_animals_per_study)):
            if len(converted) > 0:
                for val_pt, animal in zip(converted, study_animals):
                    ax.scatter(study_idx, val_pt, alpha=0.8, s=50,
                              color=animal_color_map[animal], zorder=2)

        # Plot bars and error bars for our data
        x_pos = list(range(n_studies))
        ax.bar(x_pos, means, alpha=0.3, color="steelblue", zorder=1)
        ax.errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="black", capsize=3, capthick=1.5, zorder=3)

        # Add Kim et al. as additional bar with individual points
        if has_kim:
            kim_ref = kim_ranges[param_name]
            kim_x = n_studies  # Position after all studies
            # Plot Kim bar
            ax.bar(kim_x, kim_ref['mean'], alpha=0.3, color="gray", zorder=1)
            ax.errorbar(kim_x, kim_ref['mean'], yerr=kim_ref['std'], fmt="none",
                       ecolor="black", capsize=3, capthick=1.5, zorder=3)
            # Plot individual Kim vessel points in black
            for kim_val in kim_ref['values']:
                ax.scatter(kim_x, kim_val, alpha=0.7, s=40, color='black', zorder=2)

        # Set x-axis labels
        x_labels = [s.replace('_', '\n') for s in studies]
        if has_kim:
            x_labels.append('Kim\net al.')
            ax.set_xticks(range(n_studies + 1))
        else:
            ax.set_xticks(range(n_studies))
        ax.set_xticklabels(x_labels, rotation=0, fontsize=8)
        ax.set_title(f"{val}", fontsize=10)

        print(f"{param} {zerod} mean: {np.nanmean(means):.1e} +/- {np.nanmean(stds):.1e} [{unit}]")

        current_col += 1

    # Apply shared y-axes within groups (linear scale)
    for group_idx, (group_name, param_names, share_y) in enumerate(param_groups):
        if share_y and group_idx in group_axes and len(group_axes[group_idx]) > 1:
            # Get the range for this group
            ymin, ymax = group_data_ranges[group_idx]
            if ymin != float('inf') and ymax != float('-inf'):
                # Add padding in linear space
                padding = (ymax - ymin) * 0.1
                for ax in group_axes[group_idx]:
                    ax.set_ylim(max(0, ymin - padding), ymax + padding)

    # Hide unused axes
    for row in range(n_rows):
        for col in range(max_cols):
            if axes[row, col] not in [ax for axlist in group_axes.values() for ax in axlist]:
                axes[row, col].set_visible(False)

    # Add global legend at top of figure
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=animal_color_map[animal],
                                  markersize=10, label=f'DSEA{animal:02d}') for animal in animals]
    # Add Kim et al. reference to legend if data was loaded
    if kim_ranges:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                                          markersize=10, label='Kim et al. 2010'))
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(animals) + (1 if kim_ranges else 0),
               bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(f"plots/multi_animal_{model_name}_parameters.pdf", bbox_inches='tight')
    print(f"Parameters plot saved: plots/multi_animal_{model_name}_parameters.pdf")
    plt.close()


def plot_parameters_normalized(animals_optimized, studies, model_name):
    """Plot parameters normalized by baseline across multiple animals.

    Layout matches plot_parameters_multi exactly, with same parameter groups.

    Args:
        animals_optimized: Dict of {animal: {study: {param: value}}}
        studies: List of study names (must include 'baseline')
        model_name: Model name for output filename
    """
    if "baseline" not in studies:
        print("Cannot plot normalized parameters: 'baseline' not in studies")
        return

    first_animal = list(animals_optimized.keys())[0]
    n_studies = len(studies)

    # Compute derived parameters (Ra2_mean, Ra2_ratio) from Ra2_min/Ra2_max
    animals_optimized_extended = {}
    for animal in animals_optimized:
        animals_optimized_extended[animal] = {}
        for study in animals_optimized[animal]:
            animals_optimized_extended[animal][study] = dict(animals_optimized[animal][study])
            study_data = animals_optimized_extended[animal][study]

            # Compute Ra2_mean = sqrt(Ra2_min * Ra2_max) and Ra2_ratio = Ra2_max / Ra2_min
            if ("BC_COR", "Ra2_min") in study_data and ("BC_COR", "Ra2_max") in study_data:
                ra2_min = study_data[("BC_COR", "Ra2_min")]
                ra2_max = study_data[("BC_COR", "Ra2_max")]
                study_data[("BC_COR", "Ra2_mean")] = np.sqrt(ra2_min * ra2_max)
                study_data[("BC_COR", "Ra2_ratio")] = ra2_max / ra2_min if ra2_min > 0 else 1.0

    # Define parameter groups by physical quantity (same as plot_parameters_multi)
    param_groups = [
        ("Resistances", ["Ra1", "Ra2_mean", "Rv1"], False),
        ("Resistance Ratio", ["Ra2_ratio"], False),
        ("Compliances", ["Ca", "Cc"], False),
        ("Time Constants", ["tc1", "tc2"], False),
        ("Timing", ["T_vc", "T_vr"], False),
        ("Pim Scaling", ["_Pim_scale", "_Pim_dpdt"], False),
        ("Fit Quality", ["residual"], False),
    ]

    # Map parameter names to full keys
    param_name_to_key = {
        "Ra1": ("BC_COR", "Ra1"),
        "Ra2_mean": ("BC_COR", "Ra2_mean"),
        "Rv1": ("BC_COR", "Rv1"),
        "Ra2_ratio": ("BC_COR", "Ra2_ratio"),
        "Ca": ("BC_COR", "Ca"),
        "Cc": ("BC_COR", "Cc"),
        "tc1": ("BC_COR", "tc1"),
        "tc2": ("BC_COR", "tc2"),
        "T_vc": ("BC_COR", "T_vc"),
        "T_vr": ("BC_COR", "T_vr"),
        "_Pim_scale": ("BC_COR", "_Pim_scale"),
        "_Pim_dpdt": ("BC_COR", "_Pim_dpdt"),
        "residual": ("global", "residual"),
    }

    # Compute normalized values for each animal, study, and parameter
    # normalized_data[param_name][study] = [(animal, normalized_value), ...]
    normalized_data = {pname: {study: [] for study in studies} for pname in param_name_to_key.keys()}

    animals_with_baseline = []
    for animal, animal_data in animals_optimized_extended.items():
        if "baseline" not in animal_data:
            continue
        animals_with_baseline.append(animal)
        baseline = animal_data["baseline"]

        for study in studies:
            if study not in animal_data:
                continue
            study_params = animal_data[study]

            for pname, pkey in param_name_to_key.items():
                if pkey in baseline and pkey in study_params:
                    baseline_val = baseline[pkey]
                    study_val = study_params[pkey]
                    if baseline_val != 0:
                        normalized = study_val / baseline_val
                        normalized_data[pname][study].append((animal, normalized))

    # Define colors for each animal
    animals = sorted(animals_with_baseline)
    animal_colors = plt.cm.tab10(np.linspace(0, 1, len(animals)))
    animal_color_map = {animal: animal_colors[idx] for idx, animal in enumerate(animals)}

    # Determine grid layout
    n_rows = len(param_groups)
    max_cols = max(len(params) for _, params, _ in param_groups)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(max_cols * 3.5, n_rows * 3.5), squeeze=False)

    for row_idx, (group_name, param_names, _) in enumerate(param_groups):
        for col_idx, param_name in enumerate(param_names):
            ax = axes[row_idx, col_idx]

            # Plot individual points for each study
            for study_idx, study in enumerate(studies):
                study_data = normalized_data[param_name].get(study, [])
                for animal, val in study_data:
                    ax.scatter(study_idx, val, alpha=0.8, s=50,
                              color=animal_color_map[animal], zorder=2)

            # Compute means and stds for bar plot
            means = []
            stds = []
            for study in studies:
                vals = [v for _, v in normalized_data[param_name].get(study, [])]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            # Plot bars and error bars
            x_pos = range(n_studies)
            ax.bar(x_pos, means, alpha=0.3, color="steelblue", zorder=1)
            ax.errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="black",
                       capsize=3, capthick=1.5, zorder=3)

            # Add reference line at 1.0 (no change from baseline)
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

            ax.set_xticks(range(n_studies))
            ax.set_xticklabels([s.replace('_', '\n') for s in studies], rotation=0, fontsize=8)
            ax.set_title(f"{param_name}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)

            # Only show y-label on first column
            if col_idx == 0:
                ax.set_ylabel(f"{group_name}\n(ratio to baseline)")

        # Hide unused axes in this row
        for col_idx in range(len(param_names), max_cols):
            axes[row_idx, col_idx].set_visible(False)

    # Add global legend at top of figure
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=animal_color_map[animal],
                                  markersize=10, label=f'DSEA{animal:02d}') for animal in animals]
    # Add reference line to legend
    legend_handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                                      label='Baseline (1.0)'))
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(animals) + 1,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(f"plots/multi_animal_{model_name}_parameters_normalized.pdf", bbox_inches='tight')
    plt.close()
    print(f"Normalized parameters plot saved: plots/multi_animal_{model_name}_parameters_normalized.pdf")


def plot_combined_results(all_data, all_config, model_name, get_sim_qv_func, node_in, get_name_func):
    """Create combined plot of all animals and conditions."""
    animals = sorted(all_data.keys())
    studies = list(all_data[animals[0]].keys())
    n_animals = len(animals)
    n_studies = len(studies)

    fig, axes = plt.subplots(n_animals * 2, n_studies, figsize=(4 * n_studies, 3 * n_animals * 2),
                              dpi=150)
    if n_studies == 1:
        axes = axes.reshape(-1, 1)

    for i, animal in enumerate(animals):
        for j, study in enumerate(studies):
            if study not in all_data[animal]:
                continue

            data = all_data[animal][study]["s"]
            config = all_config[animal][study]
            ti = config["boundary_conditions"][0]["bc_values"]["t"]

            q_sim, v_sim = get_sim_qv_func(config, node_in=node_in)

            ax_flow = axes[i * 2, j]
            ax_flow.plot(ti, data["qlad"], 'k-', linewidth=1.5, label='Measured')
            ax_flow.plot(ti, q_sim, 'r-', linewidth=1.5, label='Simulated')
            ax_flow.set_ylabel(f'{get_name_func(animal)}\nFlow [ml/s]', fontsize=9)
            ax_flow.grid(True, alpha=0.3)
            if i == 0:
                ax_flow.set_title(study.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            if i == 0 and j == n_studies - 1:
                ax_flow.legend(loc='upper right', fontsize=8)

            ax_vol = axes[i * 2 + 1, j]
            ax_vol.plot(ti, data["vmyo"], 'k-', linewidth=1.5, label='Measured')
            ax_vol.plot(ti, v_sim, 'r-', linewidth=1.5, label='Simulated')
            ax_vol.set_ylabel('Volume [ml]', fontsize=9)
            ax_vol.grid(True, alpha=0.3)
            if i == n_animals - 1:
                ax_vol.set_xlabel('Time [s]', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'plots/combined_all_animals_{model_name}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved: plots/combined_all_animals_{model_name}.pdf")


def plot_internal_flows(all_data, all_config, model_name, get_internal_flows_func, node_in, get_name_func):
    """Create plot of internal flows at different locations in the coronary model.

    Shows flows at three locations in the CORONARY_VAR_RES circuit:
        P_in --> Ra1 --> [Ca] --> Ra2 --> [Cc] --> Rv1 --> Pv

    Flows:
        Q_Ra1: Flow through Ra1 (inlet, after arterial resistance)
        Q_Ra2: Flow through Ra2 (mid-arterial, between Ca and Cc)
        Q_Rv1: Flow through Rv1 (venous outflow)

    Args:
        all_data: Dict of {animal: {study: {"o": orig_data, "s": smooth_data}}}
        all_config: Dict of {animal: {study: config}}
        model_name: Model name for output filename
        get_internal_flows_func: Function to compute internal flows from config
        node_in: Node name for inlet flow extraction
        get_name_func: Function to get animal name string
    """
    animals = sorted(all_data.keys())
    studies = list(all_data[animals[0]].keys())
    n_animals = len(animals)
    n_studies = len(studies)

    # Create figure: 3 rows per animal (Q_Ra1, Q_Ra2, Q_Rv1), columns for studies
    fig, axes = plt.subplots(n_animals * 3, n_studies, figsize=(4 * n_studies, 2.5 * n_animals * 3),
                              dpi=150)
    if n_studies == 1:
        axes = axes.reshape(-1, 1)

    flow_labels = ['Q_in (Ra1→Ca)', 'Q_mid (Ra2→Cc)', 'Q_out (Rv1→Pv)']
    flow_keys = ['Q_in', 'Q_mid', 'Q_out']
    flow_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    for i, animal in enumerate(animals):
        for j, study in enumerate(studies):
            if study not in all_data[animal]:
                continue

            data = all_data[animal][study]["s"]
            config = all_config[animal][study]

            # Get internal flows
            flows = get_internal_flows_func(config, node_in=node_in)
            t = flows['t']

            # Plot each flow location
            for k, (flow_key, flow_label, flow_color) in enumerate(zip(flow_keys, flow_labels, flow_colors)):
                ax = axes[i * 3 + k, j]

                # Plot simulated internal flow
                ax.plot(t, flows[flow_key], color=flow_color, linewidth=1.5, label=flow_label)

                # For Q_Ra1 (inlet), also show measured LAD flow for comparison
                if k == 0:
                    ax.plot(data["t"], data["qlad"], 'k--', linewidth=1, alpha=0.7, label='Measured')

                # Labels
                if j == 0:
                    ax.set_ylabel(f'{get_name_func(animal)}\n{flow_label}\n[ml/s]', fontsize=8)
                else:
                    ax.set_ylabel(f'{flow_label}\n[ml/s]', fontsize=8)

                ax.grid(True, alpha=0.3)

                # Title on top row only
                if i == 0 and k == 0:
                    ax.set_title(study.replace('_', ' ').title(), fontsize=10, fontweight='bold')

                # Legend on first plot
                if i == 0 and j == n_studies - 1 and k == 0:
                    ax.legend(loc='upper right', fontsize=7)

                # X-axis label on bottom row only
                if i == n_animals - 1 and k == 2:
                    ax.set_xlabel('Time [s]', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'plots/internal_flows_{model_name}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Internal flows plot saved: plots/internal_flows_{model_name}.pdf")


def plot_volume_metrics(all_volume_metrics, studies, model_name):
    """Plot volume change metrics across multiple animals.

    Shows volume metrics computed from internal flows:
        - Normalized metrics (ratio to baseline) for most quantities
        - Absolute values for backflow metrics

    Args:
        all_volume_metrics: Dict of {animal: {study: {metric: value}}}
        studies: List of study names (must include 'baseline')
        model_name: Model name for output filename
    """
    if "baseline" not in studies:
        print("Cannot plot volume metrics: 'baseline' not in studies")
        return

    n_studies = len(studies)

    # Define metric groups - separate normalized and absolute metrics
    # (group_name, [metrics], use_absolute_values, share_y_group)
    # Note: All flows/volumes are normalized per gram of myocardial tissue (ml/cycle/g)
    # share_y_group: rows with same group number will share y-axis limits
    metric_groups = [
        ("Volume Changes\n(ratio to baseline)", ["dV_total", "dV_a", "dV_c"], False, None),
        ("Inlet (Ra1)\n[ml/cycle/g]", ["V_in_forward", "V_in_backflow", "Q_in_net"], True, "volume"),
        ("Mid (Ra2)\n[ml/cycle/g]", ["V_Ra2_forward", "V_Ra2_backflow", "Q_Ra2_net"], True, "volume"),
        ("Venous (Rv1)\n[ml/cycle/g]", ["V_Rv1_forward", "V_Rv1_backflow", "Q_Rv1_net"], True, "volume"),
    ]

    # Collect all animals with baseline
    animals_with_baseline = []
    for animal, animal_data in all_volume_metrics.items():
        if "baseline" in animal_data:
            animals_with_baseline.append(animal)
    animals = sorted(animals_with_baseline)

    # Define colors for each animal
    animal_colors = plt.cm.tab10(np.linspace(0, 1, len(animals)))
    animal_color_map = {animal: animal_colors[idx] for idx, animal in enumerate(animals)}

    # Determine grid layout
    n_rows = len(metric_groups)
    max_cols = max(len(metrics) for _, metrics, _, _ in metric_groups)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(max_cols * 3.5, n_rows * 3.5), squeeze=False)

    # First pass: collect all data and find shared y-axis limits
    all_plot_data = {}
    shared_y_limits = {}

    for row_idx, (group_name, metrics, use_absolute, share_y_group) in enumerate(metric_groups):
        for col_idx, metric in enumerate(metrics):
            # Collect data for this metric
            plot_data = {study: [] for study in studies}

            for animal in animals:
                animal_data = all_volume_metrics[animal]
                if "baseline" not in animal_data:
                    continue
                baseline = animal_data["baseline"]

                for study in studies:
                    if study not in animal_data:
                        continue
                    study_metrics = animal_data[study]

                    if metric in study_metrics:
                        if use_absolute:
                            val = study_metrics[metric]
                        else:
                            if metric in baseline and baseline[metric] != 0:
                                val = study_metrics[metric] / baseline[metric]
                            else:
                                continue
                        plot_data[study].append((animal, val))

            all_plot_data[(row_idx, col_idx)] = plot_data

            # Track y limits for shared groups
            if share_y_group is not None:
                all_vals = [v for study_vals in plot_data.values() for _, v in study_vals]
                if all_vals:
                    if share_y_group not in shared_y_limits:
                        shared_y_limits[share_y_group] = [float('inf'), float('-inf')]
                    shared_y_limits[share_y_group][0] = min(shared_y_limits[share_y_group][0], min(all_vals))
                    shared_y_limits[share_y_group][1] = max(shared_y_limits[share_y_group][1], max(all_vals))

    # Add padding to shared limits
    for group in shared_y_limits:
        ymin, ymax = shared_y_limits[group]
        padding = (ymax - ymin) * 0.1
        shared_y_limits[group] = [max(0, ymin - padding), ymax + padding]

    # Second pass: plot data
    for row_idx, (group_name, metrics, use_absolute, share_y_group) in enumerate(metric_groups):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            plot_data = all_plot_data[(row_idx, col_idx)]

            # Check if we have data
            has_data = any(len(plot_data.get(s, [])) > 0 for s in studies)
            if not has_data:
                ax.set_visible(False)
                continue

            # Plot individual points for each study
            for study_idx, study in enumerate(studies):
                for animal, val in plot_data.get(study, []):
                    ax.scatter(study_idx, val, alpha=0.8, s=50,
                              color=animal_color_map[animal], zorder=2)

            # Compute means and stds for bar plot
            means = []
            stds = []
            for study in studies:
                vals = [v for _, v in plot_data.get(study, [])]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            # Plot bars and error bars
            x_pos = range(n_studies)
            ax.bar(x_pos, means, alpha=0.3, color="steelblue", zorder=1)
            ax.errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="black",
                       capsize=3, capthick=1.5, zorder=3)

            # Add reference line at 1.0 for normalized metrics only
            if not use_absolute:
                ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

            # Apply shared y-axis limits
            if share_y_group is not None and share_y_group in shared_y_limits:
                ax.set_ylim(shared_y_limits[share_y_group])

            ax.set_xticks(range(n_studies))
            ax.set_xticklabels([s.replace('_', '\n') for s in studies], rotation=0, fontsize=8)
            ax.set_title(metric.replace('_', ' '), fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)

            # Only show y-label on first column
            if col_idx == 0:
                ax.set_ylabel(group_name)

        # Hide unused axes in this row
        for col_idx in range(len(metrics), max_cols):
            axes[row_idx, col_idx].set_visible(False)

    # Add global legend at top of figure
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=animal_color_map[animal],
                                  markersize=10, label=f'DSEA{animal:02d}') for animal in animals]
    legend_handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                                      label='Baseline (1.0)'))
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(animals) + 1,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(f"plots/multi_animal_{model_name}_volume_metrics.pdf", bbox_inches='tight')
    plt.close()
    print(f"Volume metrics plot saved: plots/multi_animal_{model_name}_volume_metrics.pdf")

    # Print statistics tables
    _print_volume_metrics_tables(all_volume_metrics, studies, metric_groups)


def _print_volume_metrics_tables(all_volume_metrics, studies, metric_groups):
    """Print volume metrics statistics tables."""

    # Collect animals with baseline
    animals = sorted([a for a in all_volume_metrics if "baseline" in all_volume_metrics[a]])

    print("\n" + "=" * 110)
    print("VOLUME METRICS SUMMARY")
    print("=" * 110)

    for group_name, metrics, use_absolute, *_ in metric_groups:
        print(f"\n{group_name.replace(chr(10), ' ')}:")
        print("-" * 110)

        header = f"{'Metric':<25}"
        for study in studies:
            header += f"{study:<17}"
        print(header)

        for metric in metrics:
            row = f"{metric:<25}"
            for study in studies:
                vals = []
                for animal in animals:
                    animal_data = all_volume_metrics[animal]
                    if study not in animal_data or metric not in animal_data[study]:
                        continue
                    if use_absolute:
                        vals.append(animal_data[study][metric])
                    else:
                        baseline = animal_data.get("baseline", {})
                        if metric in baseline and baseline[metric] != 0:
                            vals.append(animal_data[study][metric] / baseline[metric])

                if vals:
                    if use_absolute:
                        row += f"{np.mean(vals):.4f}±{np.std(vals):.4f} "
                    else:
                        row += f"{np.mean(vals):.2f}±{np.std(vals):.2f}    "
                else:
                    row += f"{'N/A':<17}"
            print(row)

    print("\n" + "=" * 110)


def plot_circuit_diagram(model_name):
    """Compile LaTeX/TikZ circuit diagram to PDF."""
    import subprocess
    import os

    tex_file = "plots/circuit_diagram.tex"
    if not os.path.exists(tex_file):
        print(f"LaTeX source not found: {tex_file}")
        return

    # Compile LaTeX to PDF
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=plots", tex_file],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        # Clean up auxiliary files
        for ext in [".aux", ".log"]:
            aux_file = f"plots/circuit_diagram{ext}"
            if os.path.exists(aux_file):
                os.remove(aux_file)
        print(f"Circuit diagram saved: plots/circuit_diagram.pdf")
    else:
        print(f"LaTeX compilation failed: {result.stderr}")


def plot_averaged_conditions(scaled_data, individual_scaled, sim_results, all_optimized, studies, model_name):
    """Plot averaged conditions with individual scaled curves overlay.

    Uses the same color scheme as plot_parameters_multi for consistency.
    Shows inputs (pressures) and outputs (flow, volume) with their scaling.

    Circuit layout:
        BC_AT (pat) --> BV_prox --> BC_COR (Pim from pven)
        Flow measured at inlet, Volume at BC_COR

    Scaling:
        - Pressures (pat, pven): NOT scaled (intensive quantities)
        - Flow (qlad): Scaled by q_range ratio
        - Volume (vmyo): Scaled by v_range ratio

    Args:
        scaled_data: Dict of averaged scaled data per study
        individual_scaled: Dict of individual animal scaled curves per study
        sim_results: Dict of simulation results per study
        all_optimized: Dict of dicts - optimized parameters for each study
        studies: List of study names
        model_name: Model name for file naming
    """
    # Define colors for each condition
    condition_colors = {
        "baseline": "#1f77b4",        # Blue
        "mild_sten": "#ff7f0e",       # Orange
        "mild_sten_dob": "#2ca02c",   # Green
        "mod_sten": "#d62728",        # Red
        "mod_sten_dob": "#9467bd",    # Purple
    }

    # Colors for individual animals (using tab10)
    animal_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Create figure with 4 rows (pat, pven, flow, volume) x 5 columns (conditions)
    fig, axes = plt.subplots(4, len(studies), figsize=(4 * len(studies), 10), dpi=150)

    row_labels = [
        'Arterial P\n[mmHg]\n(input, not scaled)',
        'Ventricular P\n[mmHg]\n(input, not scaled)',
        'Scaled Flow\n[ml/s]\n(output)',
        'Scaled Volume\n[ml]\n(output)'
    ]

    for j, study in enumerate(studies):
        if study not in scaled_data:
            continue

        color = condition_colors.get(study, "gray")
        data_s = scaled_data[study]["s"]
        t_ref = data_s["t"]

        # Plot individual animal curves (thin, semi-transparent)
        for idx, ind_data in enumerate(individual_scaled.get(study, [])):
            animal = ind_data["animal"]
            ind_t = ind_data["data"]["t"]
            animal_color = animal_colors[idx % 10]

            # Pressures (not scaled) - convert to mmHg
            ind_pat = ind_data["data"]["pat"] * Ba_to_mmHg
            ind_pven = ind_data["data"]["pven"] * Ba_to_mmHg
            axes[0, j].plot(ind_t, ind_pat, color=animal_color, alpha=0.3, linewidth=0.8,
                           label=f'DSEA{animal:02d}' if j == 0 else None)
            axes[1, j].plot(ind_t, ind_pven, color=animal_color, alpha=0.3, linewidth=0.8)

            # Flow and Volume (scaled)
            ind_q = ind_data["data"]["qlad"]
            ind_v = ind_data["data"]["vmyo"]
            axes[2, j].plot(ind_t, ind_q, color=animal_color, alpha=0.3, linewidth=0.8)
            axes[3, j].plot(ind_t, ind_v, color=animal_color, alpha=0.3, linewidth=0.8)

        # Plot averaged measured data (thick dashed line)
        axes[0, j].plot(t_ref, data_s["pat"] * Ba_to_mmHg, 'k--', linewidth=2,
                       label='Mean' if j == 0 else None)
        axes[1, j].plot(t_ref, data_s["pven"] * Ba_to_mmHg, 'k--', linewidth=2)
        axes[2, j].plot(t_ref, data_s["qlad"], 'k--', linewidth=2,
                       label='Mean measured' if j == 0 else None)
        axes[3, j].plot(t_ref, data_s["vmyo"], 'k--', linewidth=2)

        # Plot simulation result (thick solid line in condition color)
        if study in sim_results:
            t_sim = sim_results[study]["t"]
            q_sim = sim_results[study]["q"]
            v_sim = sim_results[study]["v"]
            axes[2, j].plot(t_sim, q_sim, color=color, linewidth=2.5,
                           label='Simulated' if j == 0 else None)
            axes[3, j].plot(t_sim, v_sim, color=color, linewidth=2.5)

        # Labels and titles
        axes[0, j].set_title(study.replace('_', '\n'), fontsize=10, fontweight='bold')
        for i in range(4):
            axes[i, j].grid(True, alpha=0.3)
            if i == 3:
                axes[i, j].set_xlabel('Time [s]', fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(row_labels[i], fontsize=8)

    # Add legends
    axes[0, 0].legend(loc='upper right', fontsize=6)
    axes[2, 0].legend(loc='upper right', fontsize=6)

    plt.tight_layout()
    plt.savefig(f'plots/averaged_conditions_{model_name}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Averaged conditions plot saved: plots/averaged_conditions_{model_name}.pdf")

    # Create parameter summary plot showing all conditions
    _plot_averaged_parameters_multi(all_optimized, studies, model_name, condition_colors)


def _plot_averaged_parameters_multi(all_optimized, studies, model_name, condition_colors):
    """Plot optimized parameters across all conditions.

    Shows parameters in a format similar to plot_parameters_multi, with bars for each condition.

    Args:
        all_optimized: Dict of dicts - {study: {param: value}}
        studies: List of study names
        model_name: Model name for file naming
        condition_colors: Dict mapping study names to colors
    """
    # Compute derived parameters for each study
    for study in all_optimized:
        opt = all_optimized[study]
        if ("BC_COR", "Ra2_min") in opt and ("BC_COR", "Ra2_max") in opt:
            ra2_min = opt[("BC_COR", "Ra2_min")]
            ra2_max = opt[("BC_COR", "Ra2_max")]
            opt[("BC_COR", "Ra2_mean")] = np.sqrt(ra2_min * ra2_max)
            opt[("BC_COR", "Ra2_ratio")] = ra2_max / ra2_min if ra2_min > 0 else 1.0

    # Get reference study for parameter list
    first_study = list(all_optimized.keys())[0]

    # Define parameter groups (same as in plot_parameters_multi)
    param_groups = [
        ("Resistances", ["Ra1", "Ra2_mean", "Rv1", "R_poiseuille"], False),
        ("Resistance Ratio", ["Ra2_ratio"], False),
        ("Compliances", ["Ca", "Cc"], False),
        ("Inductance", ["L"], False),
        ("Time Constants", ["tc1", "tc2"], False),
        ("Timing", ["T_vc", "T_vr"], False),
        ("Pim Scaling", ["_Pim_scale", "_Pim_dpdt"], False),
        ("Fit Quality", ["residual"], False),
    ]

    # Map parameters to groups
    param_to_group = {}
    for group_idx, (group_name, param_names, _) in enumerate(param_groups):
        for pname in param_names:
            param_to_group[pname] = group_idx

    # Get list of parameters to plot
    params_to_plot = [p for p in all_optimized[first_study].keys() if p[1] in param_to_group]

    def get_sort_key(param):
        _, val = param
        group_idx = param_to_group.get(val, 99)
        for gidx, (_, param_names, _) in enumerate(param_groups):
            if val in param_names:
                return (gidx, param_names.index(val))
        return (99, 0)

    sorted_params = sorted(params_to_plot, key=get_sort_key)

    # Count params per group for layout
    group_counts = defaultdict(int)
    for param, val in sorted_params:
        group_idx = param_to_group.get(val, 99)
        group_counts[group_idx] += 1

    n_rows = len([g for g in group_counts.keys() if group_counts[g] > 0])
    max_cols = max(group_counts.values()) if group_counts else 1
    n_studies = len(studies)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(max_cols * 3.5, n_rows * 3),
                              squeeze=False)

    current_row = -1
    current_col = 0
    last_group = -1
    used_axes = []

    for param, val in sorted_params:
        group_idx = param_to_group.get(val, 99)

        if group_idx != last_group:
            current_row += 1
            current_col = 0
            last_group = group_idx

        ax = axes[current_row, current_col]
        used_axes.append(ax)

        # Get values for each study
        values = []
        colors = []
        for study in studies:
            if study in all_optimized and (param, val) in all_optimized[study]:
                values.append(all_optimized[study][(param, val)])
                colors.append(condition_colors.get(study, "gray"))
            else:
                values.append(0)
                colors.append("gray")

        # Convert units
        zerod = val[0]
        converted_values, unit, name = convert_units(zerod, np.array(values), units)

        # Plot bars
        x_pos = range(n_studies)
        ax.bar(x_pos, converted_values, color=colors, alpha=0.8)
        ax.set_xticks(range(n_studies))
        ax.set_xticklabels([s.replace('_', '\n') for s in studies], rotation=0, fontsize=7)
        ax.set_title(f"{val}", fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        if current_col == 0:
            ax.set_ylabel(f"{name} [{unit}]", fontsize=9)

        # Print values
        for study, cv in zip(studies, converted_values):
            print(f"{study} {val}: {cv:.3e} [{unit}]")

        current_col += 1

    # Hide unused axes
    for row in range(n_rows):
        for col in range(max_cols):
            if axes[row, col] not in used_axes:
                axes[row, col].set_visible(False)

    plt.suptitle(f'Averaged Fit Parameters by Condition ({model_name})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(f'plots/averaged_parameters_{model_name}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Averaged parameters plot saved: plots/averaged_parameters_{model_name}.pdf")
