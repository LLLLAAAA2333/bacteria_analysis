"""Static PNG/SVG of 106bac neural trajectories — heatmap + overlay per neuron.

Generates two views:
  1. Per-neuron heatmaps (strains × time, ordered by neural RDM clustering)
  2. Per-neuron overlay of all 106 traces

Much faster than subplot-per-strain grid.
"""

from __future__ import annotations

import sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bacteria_analysis.features.neural import build_trial_feature_matrix, neural_feature_columns
from bacteria_analysis._data_loaders import enrich_neural_dataframe

OUTPUT_DIR = Path("results/chemical_pca_ward")

MERGED_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)
LR_MERGE = {
    "ADFL": "ADF", "ADFR": "ADF", "ADLL": "ADL", "ADLR": "ADL",
    "ASGL": "ASG", "ASGR": "ASG", "ASHL": "ASH", "ASHR": "ASH",
    "ASIL": "ASI", "ASIR": "ASI", "ASJL": "ASJ", "ASJR": "ASJ",
    "ASKL": "ASK", "ASKR": "ASK", "AWAL": "AWA", "AWAR": "AWA",
    "AWBL": "AWB", "AWBR": "AWB",
}

WSTART, WSTOP = 5, 25
ACTIVE_THRESHOLD = 0.2


def sid(stim): return str(stim).strip().split()[0]


def main():
    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid", "stim_color"]].first()

    # ── Build neural RDM (vectorized) ──────────────────────────────────
    print("Building neural RDM...")
    features = build_trial_feature_matrix(raw, view="full_trajectory", merge_lr=True)
    all_cols = neural_feature_columns(features)
    wcols = [c for c in all_cols if c.rsplit("__", 1)[-1] in {f"t{t:02d}" for t in range(WSTART, WSTOP)}]

    def agg_proto(fr, gcols):
        rows = []
        for gk, grp in fr.groupby(gcols, sort=True, dropna=False):
            if not isinstance(gk, tuple): gk = (gk,)
            vals = grp.loc[:, wcols].to_numpy(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                proto = np.nanmedian(vals, axis=0)
            r = dict(zip(gcols, gk, strict=True))
            r.update(dict(zip(wcols, proto, strict=True)))
            rows.append(r)
        return pd.DataFrame(rows)

    date_stim = agg_proto(features, ["date", "stim_name"])
    scales = {}
    for n in MERGED_NEURONS:
        ncols = [c for c in wcols if c.startswith(f"{n}__")]
        vals = date_stim.loc[:, ncols].to_numpy(float).ravel()
        fin = vals[np.isfinite(vals)]
        act = fin[np.abs(fin) >= ACTIVE_THRESHOLD] if fin.size else np.array([])
        s = float(np.mean(np.abs(act))) if act.size else 1.0
        scales[n] = s if np.isfinite(s) and s > 0 else 1.0

    sp = agg_proto(features, ["stim_name"])
    sp["sample_id"] = sp["stim_name"].map(sid)
    for col in wcols:
        n = col.split("__", 1)[0]
        sp[col] = sp[col].astype(float) / scales[n]

    vals = sp.loc[:, wcols].to_numpy(float)
    rm = np.nanmean(vals, axis=1, keepdims=True)
    rs = np.nanstd(vals, axis=1, ddof=0, keepdims=True)
    vz = np.where(rs > 0, (vals - rm) / rs, 0.0)
    col_ok = np.isfinite(vz).any(axis=0)
    vz = vz[:, col_ok]
    r_mat = np.corrcoef(vz)
    distances = np.clip(1.0 - r_mat, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    labels = sp["sample_id"].astype(str).tolist()

    matrix = distances.copy()
    off = distances[~np.eye(len(labels), dtype=bool)]
    fin_off = off[np.isfinite(off)]
    fill_val = float(np.max(fin_off)) if fin_off.size else 1.0
    matrix = np.where(np.isfinite(matrix), matrix, fill_val)
    np.fill_diagonal(matrix, 0.0)
    order = leaves_list(linkage(squareform(matrix, checks=False), method="average"))
    cluster_order = [labels[i] for i in order]
    print(f"  {len(cluster_order)} strains in cluster order")

    # ── Build per-neuron × per-strain median traces ────────────────────
    raw_c = raw.copy()
    raw_c["neuron_display"] = raw_c["neuron"].replace(LR_MERGE)
    raw_c["sample_id"] = raw_c["stim_name"].map(sid)
    raw_c["rel_time"] = raw_c["time_point"].astype(float) - raw_c["start_time"].astype(float)

    traces = (
        raw_c.groupby(["neuron_display", "sample_id", "rel_time"], sort=False)["delta_F_over_F0"]
        .median().reset_index()
    )
    time_pts = sorted(traces["rel_time"].unique())
    tp_array = np.array(time_pts)

    # ── 1. Heatmap grid (13 rows × 1 column of heatmaps) ──────────────
    print("Generating heatmaps...")
    fig, axes = plt.subplots(len(MERGED_NEURONS), 1, figsize=(16, 14),
                              gridspec_kw={"hspace": 0.06})
    if len(MERGED_NEURONS) == 1:
        axes = [axes]

    for ri, neuron in enumerate(MERGED_NEURONS):
        ax = axes[ri]
        nd = traces[traces["neuron_display"].eq(neuron)]
        # Build matrix: strains (cluster order) × time points
        heat = np.full((len(cluster_order), len(time_pts)), np.nan)
        for ci, sid_name in enumerate(cluster_order):
            sd = nd[nd["sample_id"].eq(sid_name)].set_index("rel_time")
            for ti, tp in enumerate(time_pts):
                if tp in sd.index:
                    heat[ci, ti] = float(sd.loc[tp, "delta_F_over_F0"])

        vmax = float(np.nanquantile(np.abs(heat), 0.98))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        im = ax.imshow(
            heat, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, interpolation="bilinear",
            extent=[time_pts[0], time_pts[-1], len(cluster_order), 0],
        )
        ax.axvspan(0, 15, alpha=0.12, color="black", zorder=2)
        ax.set_ylabel(neuron, fontsize=8, fontweight="bold", rotation=0,
                      ha="right", va="center", labelpad=20)
        ax.set_yticks([])
        if ri == len(MERGED_NEURONS) - 1:
            ax.set_xlabel("Time (s)", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=6)

    fig.suptitle(
        "106bac neural trajectories — heatmap (RdBu_r, ±98th pct clip)  "
        "ordered by neural shape RDM clustering",
        fontsize=10, y=1.01,
    )
    fig.text(0.5, -0.01, f"{len(cluster_order)} strains  ×  {len(MERGED_NEURONS)} neurons  |  "
             "grey bar = stimulus interval", ha="center", fontsize=7, color="#666")

    fig.savefig(OUTPUT_DIR / "neural_trajectory_heatmap.svg", dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "neural_trajectory_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: neural_trajectory_heatmap.{svg,png}")

    # ── 2. Overlay per neuron (all 106 traces + cluster means) ─────────
    print("Generating overlays...")
    # ASH cluster labels for coloring
    ash_data = traces[traces["neuron_display"].eq("ASH")]
    ash_pivot = ash_data.pivot_table(
        index="sample_id", columns="rel_time", values="delta_F_over_F0", aggfunc="median"
    )
    ash_z = ash_pivot.subtract(ash_pivot.mean(axis=1), axis=0).div(ash_pivot.std(axis=1), axis=0)
    ash_r = np.corrcoef(ash_z.values)
    ash_d = np.clip(1 - ash_r, 0, None); np.fill_diagonal(ash_d, 0); ash_d = (ash_d + ash_d.T) / 2
    from scipy.cluster.hierarchy import fcluster
    ash_cl = fcluster(linkage(squareform(ash_d), "ward"), 2, criterion="maxclust")
    ash_cl_map = dict(zip(ash_pivot.index.tolist(), ash_cl))

    fig2, axes2 = plt.subplots(len(MERGED_NEURONS), 1, figsize=(14, 18),
                                gridspec_kw={"hspace": 0.10})
    if len(MERGED_NEURONS) == 1:
        axes2 = [axes2]

    for ri, neuron in enumerate(MERGED_NEURONS):
        ax = axes2[ri]
        nd = traces[traces["neuron_display"].eq(neuron)]
        for sid_name in cluster_order:
            sd = nd[nd["sample_id"].eq(sid_name)].sort_values("rel_time")
            if sd.empty: continue
            cl = ash_cl_map.get(sid_name, 0)
            color = "#e74c3c" if cl == 1 else "#3498db"
            ax.plot(sd["rel_time"], sd["delta_F_over_F0"], color=color, lw=0.35, alpha=0.30)
        # Cluster means
        for cl, color in [(1, "#e74c3c"), (2, "#3498db")]:
            cl_sids = [s for s in cluster_order if ash_cl_map.get(s, 0) == cl]
            cl_data = nd[nd["sample_id"].isin(cl_sids)]
            if cl_data.empty: continue
            means = cl_data.groupby("rel_time")["delta_F_over_F0"].median()
            ax.plot(means.index, means.values, color=color, lw=2.2, alpha=0.95)
        ax.axvspan(0, 15, alpha=0.08, color="black")
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.3)
        ax.set_ylabel(neuron, fontsize=9, fontweight="bold", rotation=0,
                      ha="right", va="center", labelpad=25)
        ax.set_xlim(time_pts[0], time_pts[-1])
        if ri == len(MERGED_NEURONS) - 1:
            ax.set_xlabel("Time (s)", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig2.suptitle(
        "106bac neural trajectory overlay  —  red/blue = ASH cluster 1/2  "
        "(bold = cluster median)",
        fontsize=10, y=1.01,
    )
    fig2.savefig(OUTPUT_DIR / "neural_trajectory_overlay.svg", dpi=150, bbox_inches="tight", facecolor="white")
    fig2.savefig(OUTPUT_DIR / "neural_trajectory_overlay.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print("Saved: neural_trajectory_overlay.{svg,png}")

    print("Done.")


if __name__ == "__main__":
    main()
