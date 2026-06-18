#!/usr/bin/env python3
"""Benchmark LV-pressure-driven vs deformation-driven intramyocardial pressure (Pim).

Stendahl et al. argue that phasic myocardial volume is driven by myocardial
deformation rather than by transmitted LV pressure. This script tests that claim
two ways:

  1. STATIC (model-free): does a deformation signal track the phasic volume better
     than LVP? We correlate candidate Pim drivers with the "ideal" Pim shape
     (-volume, which peaks at the volume minimum) for every animal/condition.

  2. DYNAMIC (in the 0D model): refit the coronary model with Pim driven by LVP vs
     by circumferential-shortening strain, and compare flow+volume residuals.

Outputs a comparison figure and prints summary tables. The main pipeline
(process.py) keeps the better-fitting driver as default.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from process import read_data, smooth_data, setup_model, NODE_IN, build_deformation_signal
from optimization import optimize, get_objective
from simulation import get_sim_qv

ANIMALS = [8, 10, 15, 16]
STUDIES = ["baseline", "mild_sten", "mild_sten_dob", "mod_sten", "mod_sten_dob"]
MODES = ["lvp", "deformation", "deformation_rate", "hybrid"]
MODE_LABEL = {"lvp": "LVP", "deformation": r"$\alpha S$",
              "deformation_rate": r"$\alpha S+\beta\dot S^+$",
              "hybrid": r"$aP_{LV}+\beta\dot S^+$"}
FIT_KW = dict(weight_flow=1.0, weight_volume=1.0, weight_flow_min=0.0,
              weight_flow_mean=0.0, weight_vol_min=0.0, reg_scale=0.0,
              volume_mode="curve", loss="linear")


def _norm(x):
    x = np.asarray(x, float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else x * 0.0


def static_correlations():
    """Correlate candidate Pim drivers with the ideal shape (-volume)."""
    rows = {}  # driver -> {study -> [corr per animal]}
    drivers = ["LVP", "-circ strain", "rad strain", "-long strain"]
    for d in drivers:
        rows[d] = {s: [] for s in STUDIES}
    for a in ANIMALS:
        for s in STUDIES:
            do = read_data(a, s)
            if not do:
                continue
            d = smooth_data(do, 201)
            vol = np.asarray(d["vmyo"], float)
            ideal = -(vol - vol[0])  # peaks at volume minimum
            circ = d.get("circ_strain"); rad = d.get("rad_strain"); lon = d.get("long_strain")
            cand = {
                "LVP": np.asarray(d["pven"], float),
                "-circ strain": -np.asarray(circ, float) if circ is not None else None,
                "rad strain": np.asarray(rad, float) if rad is not None else None,
                "-long strain": -np.asarray(lon, float) if lon is not None else None,
            }
            for k, v in cand.items():
                if v is None or not np.all(np.isfinite(v)) or np.std(v) == 0:
                    continue
                rows[k][s].append(np.corrcoef(_norm(v), _norm(ideal))[0, 1])
    return drivers, rows


def _aic(err, n, k):
    """Approximate AIC for a (normalized) least-squares fit: N*ln(RSS/N) + 2k.

    Used only for relative model selection across Pim drivers (same objective, so
    N is identical); penalizes the extra Pim parameters of the richer models.
    """
    rss = err ** 2
    return n * np.log(rss / n) + 2 * k


def dynamic_benchmark():
    """Refit the 0D model with each Pim driver; collect residuals and AIC."""
    results = {m: {s: [] for s in STUDIES} for m in MODES}
    aics = {m: {s: [] for s in STUDIES} for m in MODES}
    detail = []
    for a in ANIMALS:
        for s in STUDIES:
            do = read_data(a, s)
            if not do:
                continue
            data = {"o": do, "s": smooth_data(do, 201)}
            qref, vref = data["s"]["qlad"], data["s"]["vmyo"]
            n_obs = 2 * len(qref)  # flow + volume curves (curve mode, reg off)
            has_strain = build_deformation_signal(data) is not None
            line = f"DSEA{a:02d} {s:14s}"
            for mode in MODES:
                cfg, p0 = setup_model(data, pim_mode=mode)
                cfg, err, pv = optimize(cfg, p0, data, NODE_IN, verbose=0, **FIT_KW)
                q, v = get_sim_qv(cfg, NODE_IN)
                results[mode][s].append(err)
                aics[mode][s].append(_aic(err, n_obs, len(p0)))
                detail.append((a, s, mode, err, len(p0),
                               pv.get(("BC_COR", "_Pim_strain_amp")),
                               pv.get(("BC_COR", "_Pim_strain_rate")),
                               pv.get(("BC_COR", "_Pim_scale"))))
                line += f" {mode[:6]}={err:5.2f}"
            print(line + ("" if has_strain else "  (no strain->LVP)"))
    return results, aics, detail


def make_plot(drivers, corr_rows, results):
    x = np.arange(len(STUDIES))
    labels = [s.replace("_", "\n") for s in STUDIES]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (a) static correlation of drivers with -volume
    ax = axes[0]
    w = 0.8 / len(drivers)
    for i, d in enumerate(drivers):
        m = [np.mean(corr_rows[d][s]) if corr_rows[d][s] else np.nan for s in STUDIES]
        e = [np.std(corr_rows[d][s]) if corr_rows[d][s] else np.nan for s in STUDIES]
        ax.bar(x + (i - (len(drivers) - 1) / 2) * w, m, w, yerr=e, capsize=2, label=d)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("corr( driver , -volume )")
    ax.set_title("(a) Static: which signal tracks phasic volume?")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(-0.7, 1.05)

    # (b) dynamic residuals across all four Pim drivers
    ax = axes[1]
    w = 0.8 / len(MODES)
    for i, mode in enumerate(MODES):
        m = [np.mean(results[mode][s]) if results[mode][s] else np.nan for s in STUDIES]
        e = [np.std(results[mode][s]) if results[mode][s] else np.nan for s in STUDIES]
        ax.bar(x + (i - (len(MODES) - 1) / 2) * w, m, w, yerr=e, capsize=2,
               label=MODE_LABEL[mode])
    ax.set_ylabel("fit residual (flow + volume)")
    ax.set_title("(b) Dynamic: 0D model fit quality (lower = better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9, title="Pim driver"); ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Intramyocardial pressure driver: LV pressure vs myocardial deformation",
                 fontsize=14)
    plt.tight_layout()
    out = "plots/benchmark_pim_lvp_vs_deformation.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\nBenchmark figure saved: {out}")


def main():
    print("=" * 78)
    print("STATIC correlation of candidate Pim drivers with -volume (mean over animals)")
    print("=" * 78)
    drivers, corr_rows = static_correlations()
    hdr = f"{'driver':<14}" + "".join(f"{s[:10]:>12}" for s in STUDIES)
    print(hdr)
    for d in drivers:
        row = f"{d:<14}"
        for s in STUDIES:
            row += f"{np.mean(corr_rows[d][s]):>12.2f}" if corr_rows[d][s] else f"{'-':>12}"
        print(row)

    print("\n" + "=" * 78)
    print("DYNAMIC 0D model refit: residual per animal/condition (4 Pim drivers)")
    print("=" * 78)
    results, aics, detail = dynamic_benchmark()

    def overall(d):
        return {m: np.mean([e for s in STUDIES for e in d[m][s]]) for m in MODES}
    res_all, aic_all = overall(results), overall(aics)

    print("\n" + "-" * 78)
    print("Mean residual (flow+volume), lower is better:")
    print(f"{'condition':<16}" + "".join(f"{MODE_LABEL[m]:>16}" for m in MODES))
    for s in STUDIES:
        row = f"{s:<16}"
        best = min(MODES, key=lambda m: np.mean(results[m][s]) if results[m][s] else 1e9)
        for m in MODES:
            v = np.mean(results[m][s]) if results[m][s] else np.nan
            mark = "*" if m == best else " "
            row += f"{v:>14.2f}{mark} "
        print(row)
    print("-" * 78)
    row = f"{'ALL (residual)':<16}"
    best = min(MODES, key=lambda m: res_all[m])
    for m in MODES:
        row += f"{res_all[m]:>14.2f}{'*' if m == best else ' '} "
    print(row)
    row = f"{'ALL (AIC)':<16}"
    bestaic = min(MODES, key=lambda m: aic_all[m])
    for m in MODES:
        row += f"{aic_all[m]:>14.1f}{'*' if m == bestaic else ' '} "
    print(row)
    print("-" * 78)
    print("AIC penalizes the extra Pim parameters (k: lvp=8, alphaS=9, +rate=10, hybrid=10).")
    print(f"Best by residual: {MODE_LABEL[min(MODES, key=lambda m: res_all[m])]}; "
          f"best by AIC: {MODE_LABEL[bestaic]}")

    make_plot(drivers, corr_rows, results)


if __name__ == "__main__":
    main()
