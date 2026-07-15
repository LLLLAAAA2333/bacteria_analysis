"""Visualise cross-date response consistency for the 6 stimuli measured on >1 date.

For each multi-date stimulus we plot, per neuron, the response traces separately
by date — same bacteria, same neuron, different experimental day.  Differences
directly reveal batch/date effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}

ALL_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)

OUTPUT_DIR = Path("results/cross_date_replicates")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stimuli measured on >1 date (from data inspection)
MULTI_DATE_STIMULI = ["b58_0", "b80_0", "b81_0", "b82_0", "b85_0", "b91_0"]


# ---------------------------------------------------------------------------
# data extraction
# ---------------------------------------------------------------------------

def _neuron_raw_names(neuron: str) -> list[str]:
    merged = LR_MERGE.get(neuron)
    return list(merged) if merged else [neuron]


def _trial_traces(raw: pd.DataFrame, stimulus: str, neuron: str) -> pd.DataFrame:
    """Return trial-level traces for one (stimulus, neuron) pair.

    Columns: trial_id, date, t00..t44
    Each row is one trial's mean trace (L/R merged if applicable).
    """
    raw_names = _neuron_raw_names(neuron)
    sub = raw[(raw["stimulus"] == stimulus) & (raw["neuron"].isin(raw_names))].copy()
    sub["trial_id"] = (
        sub["date"].astype(str) + "__"
        + sub["worm_key"].astype(str) + "__"
        + sub["segment_index"].astype(str)
    )
    # Mean across L/R within each trial, then pivot
    trial_avg = (
        sub.groupby(["trial_id", "date", "time_point"])["delta_F_over_F0"]
        .mean().reset_index()
    )
    wide = trial_avg.pivot(index=["trial_id", "date"], columns="time_point",
                           values="delta_F_over_F0")
    wide.columns = [f"t{int(c):02d}" for c in wide.columns]
    return wide.reset_index()


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_one_stimulus_traces(raw: pd.DataFrame, stimulus: str):
    """For one stimulus, plot all 13 neurons with date-separated traces."""
    dates = sorted(raw[raw["stimulus"] == stimulus]["date"].unique())
    date_colors = {dates[0]: "#3498db", dates[1]: "#e74c3c"}

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flat

    for idx, neuron in enumerate(ALL_NEURONS):
        ax = axes[idx]
        trials = _trial_traces(raw, stimulus, neuron)
        time_cols = [c for c in trials.columns if c.startswith("t") and len(c) == 3 and c[1:].isdigit()]
        timepoints = [int(c[1:]) for c in time_cols]

        for date in dates:
            date_trials = trials[trials["date"] == date]
            color = date_colors[date]
            traces = date_trials[time_cols].values

            if len(date_trials) == 0:
                continue

            # Individual trial traces (thin, semi-transparent)
            for trace in traces:
                ax.plot(timepoints, trace, color=color, alpha=0.25, lw=0.5)

            # Mean trace (thick)
            mean_trace = traces.mean(axis=0)
            ax.plot(timepoints, mean_trace, color=color, lw=2.0,
                    label=f"{date} (n={len(date_trials)})")

        ax.axvspan(5, 15, alpha=0.1, color="#2c3e50")
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.35)
        ax.set_title(neuron, fontsize=10, fontweight="bold")
        if idx >= 12:  # last row
            ax.set_xlabel("time (s)")
        if idx % 4 == 0:
            ax.set_ylabel("ΔF/F₀")
        ax.legend(fontsize=6, loc="upper right")

    # Hide unused last panel (13 neurons → 16 subplots)
    for idx in range(len(ALL_NEURONS), len(axes)):
        axes[idx].set_visible(False)

    info = raw[raw["stimulus"] == stimulus].iloc[0]
    genus = info.get("genus", "?")
    species = info.get("species", "?")
    fig.suptitle(f"{stimulus}  |  {genus} {species}  |  {dates[0]} vs {dates[1]}",
                 fontsize=14, y=1.01)
    fig.savefig(OUTPUT_DIR / f"{stimulus}_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUTPUT_DIR / f'{stimulus}_traces.png'}")


def plot_date_difference_heatmaps(raw: pd.DataFrame):
    """Neuron × time heatmap of date2 − date1 difference, one per stimulus.

    Red = date2 response stronger, Blue = date1 response stronger.
    Uses GridSpec with a dedicated colorbar column to avoid overlap.
    """
    from matplotlib.gridspec import GridSpec

    n_stim = len(MULTI_DATE_STIMULI)
    n_rows, n_cols = 2, 3

    fig = plt.figure(figsize=(21, 12))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[1, 1, 1, 0.04],
                  left=0.05, right=0.93, top=0.92, bottom=0.06,
                  hspace=0.45, wspace=0.35)

    global_vmax = 0

    # First pass: compute global vmax
    diffs_all = {}
    for stimulus in MULTI_DATE_STIMULI:
        dates = sorted(raw[raw["stimulus"] == stimulus]["date"].unique())
        d1, d2 = dates[0], dates[1]
        diff_mat = np.full((len(ALL_NEURONS), 45), np.nan)
        for ni, neuron in enumerate(ALL_NEURONS):
            trials = _trial_traces(raw, stimulus, neuron)
            time_cols = [c for c in trials.columns if c.startswith("t") and len(c) == 3 and c[1:].isdigit()]
            m1 = trials[trials["date"] == d1][time_cols].values.mean(axis=0)
            m2 = trials[trials["date"] == d2][time_cols].values.mean(axis=0)
            if len(m1) and len(m2):
                diff_mat[ni] = m2 - m1
        diffs_all[stimulus] = diff_mat
        global_vmax = max(global_vmax, np.nanmax(np.abs(diff_mat)))

    global_vmax = global_vmax or 0.3

    for si, stimulus in enumerate(MULTI_DATE_STIMULI):
        r, c = divmod(si, n_cols)
        ax = fig.add_subplot(gs[r, c])
        dates = sorted(raw[raw["stimulus"] == stimulus]["date"].unique())
        diff_mat = diffs_all[stimulus]

        im = ax.imshow(diff_mat, aspect="auto", cmap="RdBu_r",
                       vmin=-global_vmax, vmax=global_vmax, interpolation="nearest")

        ax.set_yticks(range(len(ALL_NEURONS)))
        ax.set_yticklabels(ALL_NEURONS, fontsize=8)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
        ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40], fontsize=7)
        ax.axvline(5, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(15, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("time (s)")
        ax.set_title(f"{stimulus}\n{dates[0]} → {dates[1]}", fontsize=10)

    # Dedicated colorbar axis spanning all rows in the last column
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Δ ΔF/F₀ (date2 − date1)", fontsize=10)

    fig.suptitle("Date difference heatmaps (date2 − date1 mean response per neuron)",
                 fontsize=14)
    fig.savefig(OUTPUT_DIR / "date_difference_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUTPUT_DIR / 'date_difference_heatmaps.png'}")


def plot_combined_neuron_heatmaps(raw: pd.DataFrame):
    """One figure per stimulus: side-by-side date1 / date2 heatmaps (neuron × time).

    This follows the plot_pair_heatmaps convention — raw data, not difference.
    """
    from bacteria_analysis._data_loaders import enrich_neural_dataframe

    enriched = enrich_neural_dataframe(raw)

    from matplotlib.gridspec import GridSpec

    for stimulus in MULTI_DATE_STIMULI:
        dates = sorted(raw[raw["stimulus"] == stimulus]["date"].unique())
        d1, d2 = dates[0], dates[1]

        # Build neuron × time matrices for each date
        mats = {}
        for date in [d1, d2]:
            mat = np.full((len(ALL_NEURONS), 45), np.nan)
            for ni, neuron in enumerate(ALL_NEURONS):
                raw_names = _neuron_raw_names(neuron)
                sub = raw[(raw["stimulus"] == stimulus)
                          & (raw["neuron"].isin(raw_names))
                          & (raw["date"] == date)]
                if sub.empty:
                    continue
                trial_ids = (sub["date"].astype(str) + "__"
                             + sub["worm_key"].astype(str) + "__"
                             + sub["segment_index"].astype(str))
                sub = sub.copy()
                sub["trial_id"] = trial_ids
                trial_avg = (sub.groupby(["trial_id", "time_point"])["delta_F_over_F0"]
                             .mean().reset_index())
                trace = trial_avg.groupby("time_point")["delta_F_over_F0"].mean()
                for tp, val in trace.items():
                    mat[ni, tp] = val
            mats[date] = mat

        # Global vmax for this stimulus
        all_vals = np.concatenate([m[~np.isnan(m)] for m in mats.values()])
        vmax = float(np.percentile(np.abs(all_vals), 98)) if len(all_vals) else 0.5
        vmax = max(vmax, 0.1)

        info = enriched[enriched["stimulus"] == stimulus].iloc[0]
        genus = info.get("genus", "?")
        species = info.get("species", "?")

        fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.04],
                      left=0.06, right=0.92, top=0.90, bottom=0.10,
                      wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        for ax, (date, mat) in zip([ax1, ax2], mats.items()):
            n_trials = raw[(raw["stimulus"] == stimulus)
                           & (raw["date"] == date)]["worm_key"].nunique()
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_yticks(range(len(ALL_NEURONS)))
            ax.set_yticklabels(ALL_NEURONS, fontsize=8)
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
            ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40], fontsize=6)
            ax.axvline(5, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.axvline(15, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.set_xlabel("time (s)", fontsize=9)
            ax.set_title(f"{date}  (n_worms≈{n_trials})", fontsize=10)

        fig.suptitle(f"{stimulus}  |  {genus} {species}  |  {d1} vs {d2}", fontsize=13)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("ΔF/F₀", fontsize=9)
        fig.savefig(OUTPUT_DIR / f"{stimulus}_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {OUTPUT_DIR / f'{stimulus}_heatmap.png'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading data …")
    raw = pd.read_parquet("data/106bac.parquet")

    print(f"Multi-date stimuli: {MULTI_DATE_STIMULI}\n")

    print("=== Trace overlays (per stimulus, all neurons) ===")
    for stimulus in MULTI_DATE_STIMULI:
        plot_one_stimulus_traces(raw, stimulus)

    print("\n=== Side-by-side neuron × time heatmaps ===")
    plot_combined_neuron_heatmaps(raw)

    print("\n=== Date-difference heatmaps ===")
    plot_date_difference_heatmaps(raw)

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
