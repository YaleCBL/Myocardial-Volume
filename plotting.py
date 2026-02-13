#!/usr/bin/env python3
"""Plotting functions for coronary model fitting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from collections import OrderedDict, defaultdict

from utils import Ba_to_mmHg, convert_units, units


def plot_data(animal, data, get_name_func):
    """Plot raw and smoothed measurement data."""
    n_param = len(data.keys())
    _, ax = plt.subplots(
        4, n_param, figsize=(max(n_param * 5, 5), 10), sharex="col", sharey="row"
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
        len(labels), len(config), figsize=(16, 9), sharex="col", sharey="row"
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
    """Plot optimized parameters for a single animal."""
    studies = list(optimized.keys())
    n_param = len(optimized[studies[0]].keys())
    _, axes = plt.subplots(1, n_param, figsize=(n_param * 3, 6))
    if n_param == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    param_groups = defaultdict(list)
    for i, (param, val) in enumerate(optimized[studies[0]].keys()):
        param_groups[val[0]] += [(i, param, val)]

    for zerod, params in param_groups.items():
        first_ax = None
        for i, param, val in params:
            values = np.array([optimized[s][(param, val)] for s in studies])
            values, unit, name = convert_units(zerod, values, units)
            ylabel = f"{name} [{unit}]"

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
            axes[i].set_title(f"{get_name_func(animal)} {val}")

    plt.tight_layout()
    plt.savefig(f"plots/{get_name_func(animal)}_{model_name}_parameters.pdf")
    plt.close()


def plot_parameters_multi(animals_optimized, studies, model_name):
    """Plot parameters across multiple animals."""
    first_animal = list(animals_optimized.keys())[0]
    if studies is None:
        studies = list(animals_optimized[first_animal].keys())

    first_study = studies[0]
    params = list(animals_optimized[first_animal][first_study].keys())
    n_param = len(params)
    n_studies = len(studies)

    _, axes = plt.subplots(1, n_param, figsize=(n_param * 3, 6))
    if n_param == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    param_groups = defaultdict(list)
    for i, (param, val) in enumerate(params):
        param_groups[val[0]] += [(i, param, val)]

    for zerod, param_list in param_groups.items():
        first_ax = None
        for i, param, val in param_list:
            all_values = []
            for study in studies:
                study_values = []
                for animal in animals_optimized.keys():
                    if study in animals_optimized[animal] and (param, val) in animals_optimized[animal][study]:
                        study_values.append(animals_optimized[animal][study][(param, val)])
                all_values.append(study_values)

            flat_values = [v for sublist in all_values for v in sublist]
            if flat_values:
                _, unit, name = convert_units(zerod, np.array(flat_values), units)
                ylabel = f"{name} [{unit}]"

                converted_values = []
                for study_values in all_values:
                    if study_values:
                        converted = convert_units(zerod, np.array(study_values), units)[0]
                        converted_values.append(converted)
                    else:
                        converted_values.append(np.array([]))

                means = [np.mean(c) if len(c) > 0 else np.nan for c in converted_values]
                stds = [np.std(c) if len(c) > 0 else np.nan for c in converted_values]

                if first_ax is None:
                    first_ax = axes[i]
                    axes[i].set_ylabel(ylabel)
                else:
                    axes[i].sharey(first_ax)
                    axes[i].set_ylabel("")

                axes[i].grid(True, axis="y")

                for study_idx, converted in enumerate(converted_values):
                    if len(converted) > 0:
                        x_positions = np.full(len(converted), study_idx)
                        axes[i].scatter(x_positions, converted, alpha=0.6, s=50, color="gray", zorder=2)

                x_pos = range(n_studies)
                axes[i].bar(x_pos, means, alpha=0.5, color="steelblue", zorder=1)
                axes[i].errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="black", capsize=5, capthick=2, zorder=3)
                axes[i].set_xticks(range(n_studies))
                axes[i].set_xticklabels(studies, rotation=45)
                axes[i].set_title(f"{val}")

                print(f"{param} {zerod} mean: {np.nanmean(means):.1e} +/- {np.nanmean(stds):.1e} [{unit}]")

    plt.tight_layout()
    plt.savefig(f"plots/multi_animal_{model_name}_parameters.pdf")
    plt.close()


def plot_combined_results(all_data, all_config, model_name, get_sim_qv_func, node_in, get_name_func):
    """Create combined plot of all animals and conditions."""
    animals = sorted(all_data.keys())
    studies = list(all_data[animals[0]].keys())
    n_animals = len(animals)
    n_studies = len(studies)

    fig, axes = plt.subplots(n_animals * 2, n_studies, figsize=(4 * n_studies, 3 * n_animals * 2),
                              sharex='col', dpi=150)
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
            ax_flow.plot(ti, q_sim, 'r--', linewidth=1.5, label='Simulated')
            ax_flow.set_ylabel(f'{get_name_func(animal)}\nFlow [ml/s]', fontsize=9)
            ax_flow.grid(True, alpha=0.3)
            if i == 0:
                ax_flow.set_title(study.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            if i == 0 and j == n_studies - 1:
                ax_flow.legend(loc='upper right', fontsize=8)

            ax_vol = axes[i * 2 + 1, j]
            ax_vol.plot(ti, data["vmyo"], 'k-', linewidth=1.5, label='Measured')
            ax_vol.plot(ti, v_sim, 'r--', linewidth=1.5, label='Simulated')
            ax_vol.set_ylabel('Volume [ml]', fontsize=9)
            ax_vol.grid(True, alpha=0.3)
            if i == n_animals - 1:
                ax_vol.set_xlabel('Time [s]', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'plots/combined_all_animals_{model_name}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved: plots/combined_all_animals_{model_name}.pdf")


def plot_circuit_diagram(model_name):
    """Create circuit diagram showing model structure and parameters."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    c_input = '#2ecc71'
    c_output = '#e74c3c'
    c_param = '#3498db'
    c_element = '#f39c12'

    ax.text(7, 9.5, 'CORONARY_VAR_RES Model with Time-Varying Resistance',
            fontsize=14, fontweight='bold', ha='center')

    # Input data
    ax.text(0.5, 8, 'INPUT DATA', fontsize=11, fontweight='bold', color=c_input)
    ax.text(0.5, 7.4, 'Aortic Pressure P_ao(t)', fontsize=10, color=c_input)
    ax.text(0.5, 6.9, 'LV Pressure P_lv(t) -> P_im', fontsize=10, color=c_input)
    ax.add_patch(Rectangle((0.3, 6.6), 3.5, 1.8, fill=False, edgecolor=c_input, linewidth=2))

    y_circuit = 4.5

    # Circuit elements
    ax.annotate('', xy=(2.5, y_circuit), xytext=(1.5, y_circuit),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(1.2, y_circuit + 0.3, 'P_ao', fontsize=10, fontweight='bold')

    ax.add_patch(Rectangle((2.5, y_circuit - 0.3), 1.5, 0.6, fill=True,
                            facecolor='lightyellow', edgecolor=c_element, linewidth=2))
    ax.text(3.25, y_circuit, 'BV_prox', fontsize=9, ha='center', va='center')
    ax.text(3.25, y_circuit + 0.8, 'L, R', fontsize=9, ha='center', color=c_param, fontweight='bold')

    ax.plot([4, 4.5], [y_circuit, y_circuit], 'k-', lw=2)

    ax.add_patch(Rectangle((4.5, y_circuit - 0.3), 1.2, 0.6, fill=True,
                            facecolor='lightblue', edgecolor=c_element, linewidth=2))
    ax.text(5.1, y_circuit, 'Ra1', fontsize=9, ha='center', va='center', color=c_param, fontweight='bold')

    ax.plot([5.7, 6.2], [y_circuit, y_circuit], 'k-', lw=2)
    ax.plot([5.95, 5.95], [y_circuit, y_circuit - 1.5], 'k-', lw=2)
    ax.plot([5.7, 6.2], [y_circuit - 1.5, y_circuit - 1.5], 'k-', lw=3)
    ax.plot([5.7, 6.2], [y_circuit - 1.7, y_circuit - 1.7], 'k-', lw=3)
    ax.text(5.95, y_circuit - 2.1, 'Ca', fontsize=9, ha='center', color=c_param, fontweight='bold')

    ax.add_patch(FancyBboxPatch((6.2, y_circuit - 0.4), 1.6, 0.8, fill=True,
                                 facecolor='lightyellow', edgecolor=c_element, linewidth=2,
                                 boxstyle='round,pad=0.05'))
    ax.text(7, y_circuit, 'Ra2(t)', fontsize=9, ha='center', va='center', color=c_param, fontweight='bold')
    ax.text(7, y_circuit + 0.7, 'Ra2, ratio', fontsize=8, ha='center', color=c_param)
    ax.text(7, y_circuit + 1.1, 'T_vc, T_vr', fontsize=8, ha='center', color=c_param)

    ax.plot([7.8, 8.3], [y_circuit, y_circuit], 'k-', lw=2)
    ax.plot([8.05, 8.05], [y_circuit, y_circuit - 1.5], 'k-', lw=2)
    ax.plot([7.8, 8.3], [y_circuit - 1.5, y_circuit - 1.5], 'k-', lw=3)
    ax.plot([7.8, 8.3], [y_circuit - 1.7, y_circuit - 1.7], 'k-', lw=3)
    ax.text(8.05, y_circuit - 2.1, 'Cc', fontsize=9, ha='center', color=c_param, fontweight='bold')
    ax.annotate('', xy=(8.05, y_circuit - 2.8), xytext=(8.05, y_circuit - 3.3),
                arrowprops=dict(arrowstyle='->', color=c_input, lw=2))
    ax.text(8.05, y_circuit - 3.6, 'P_im', fontsize=9, ha='center', color=c_input, fontweight='bold')
    ax.text(8.05, y_circuit - 4.0, '_Pim_scale, _Pim_dpdt', fontsize=8, ha='center', color=c_param)

    ax.add_patch(Rectangle((8.3, y_circuit - 0.3), 1.2, 0.6, fill=True,
                            facecolor='lightblue', edgecolor=c_element, linewidth=2))
    ax.text(8.9, y_circuit, 'Rv1', fontsize=9, ha='center', va='center', color=c_param, fontweight='bold')

    ax.plot([9.5, 10.5], [y_circuit, y_circuit], 'k-', lw=2)
    ax.text(10.8, y_circuit, 'P_v=0', fontsize=10)

    # Equation
    ax.text(7, 7.5, 'Ra2(t) = [(sqrt(Ra2_max) - sqrt(Ra2_min)) * e(t) + sqrt(Ra2_min)]^2', fontsize=9,
            ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Parameters list
    ax.text(11, 8, 'OPTIMIZED', fontsize=11, fontweight='bold', color=c_param)
    ax.text(11, 7.5, 'PARAMETERS', fontsize=11, fontweight='bold', color=c_param)
    params = ['L', 'R_poiseuille', 'Ra1', 'Ra2', '_ratio_Ra2', 'Rv1',
              'tc1 -> Ca', 'tc2 -> Cc', 'T_vc', 'T_vr', '_Pim_scale', '_Pim_dpdt']
    for k, p in enumerate(params):
        ax.text(11, 7.0 - k * 0.35, f'* {p}', fontsize=9, color=c_param)
    ax.add_patch(Rectangle((10.8, 2.5), 3, 5.3, fill=False, edgecolor=c_param, linewidth=2))

    # Objective
    ax.text(4, 1.2, 'OBJECTIVE FUNCTION', fontsize=11, fontweight='bold', color=c_output)
    ax.text(4, 0.7, 'Minimize: w_flow * ||Q_sim - Q_meas|| + w_vol * ||V_sim - V_meas||', fontsize=10, color=c_output)
    ax.add_patch(Rectangle((0.3, 0.4), 9, 1.2, fill=False, edgecolor=c_output, linewidth=2))

    plt.tight_layout()
    plt.savefig(f'plots/circuit_diagram_{model_name}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Circuit diagram saved: plots/circuit_diagram_{model_name}.pdf")
