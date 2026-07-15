"""ASH excitatory vs inhibitory response — metabolite association.

For each metabolite in matrix.xlsx: compare log2-fold-change levels between
ASH-excitatory and ASH-inhibitory stimuli.  Rank by effect size / significance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis._data_loaders import enrich_neural_dataframe

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

RESPONSE_WINDOW = (5, 15)
EXC_THRESHOLD = 0.05      # mean ΔF/F₀ above → excitatory
INH_THRESHOLD = -0.05     # mean ΔF/F₀ below → inhibitory
FDR_ALPHA = 0.10          # FDR threshold for significance

NEURON = "ASH"
OUTPUT_DIR = Path(f"results/ash_metabolite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_LR_MERGE = {"ASH": ("ASHL", "ASHR")}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_prototype(raw: pd.DataFrame, neuron: str) -> tuple[pd.DataFrame, list[int]]:
    raw_names = list(_LR_MERGE.get(neuron, [neuron]))
    subset = raw[raw["neuron"].isin(raw_names)].copy()
    subset["trial_id"] = (
        pd.to_datetime(subset["date"]).dt.strftime("%Y%m%d")
        + "__"
        + subset["worm_key"].astype(str)
        + "__"
        + subset["segment_index"].astype(str)
    )
    trial_avg = (
        subset.groupby(["trial_id", "stimulus", "time_point"])["delta_F_over_F0"]
        .mean()
        .reset_index()
    )
    proto = (
        trial_avg.groupby(["stimulus", "time_point"])["delta_F_over_F0"]
        .median()
        .reset_index()
    )
    mat = proto.pivot(index="stimulus", columns="time_point", values="delta_F_over_F0")
    timepoints = mat.columns.astype(int).tolist()
    return mat, timepoints


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # ---- 1. ASH response classification ----
    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()

    mat_ash, tp = _build_prototype(raw, NEURON)
    resp_ash = mat_ash.loc[:, RESPONSE_WINDOW[0]:RESPONSE_WINDOW[1]].mean(axis=1)

    exc_mask = resp_ash > EXC_THRESHOLD
    inh_mask = resp_ash < INH_THRESHOLD
    neu_mask = ~(exc_mask | inh_mask)

    # Map stimulus → AID
    stim_to_aid = {s: stim_info.loc[s, "aid"] for s in mat_ash.index}

    print(f"ASH: exc={exc_mask.sum()}, inh={inh_mask.sum()}, neu={neu_mask.sum()}")
    print(f"  Exc range: [{resp_ash[exc_mask].min():.3f}, {resp_ash[exc_mask].max():.3f}]")
    print(f"  Inh range: [{resp_ash[inh_mask].min():.3f}, {resp_ash[inh_mask].max():.3f}]")

    # ---- 2. Load metabolite matrix ----
    met_raw = pd.read_excel("data/matrix.xlsx")
    met_raw = met_raw.rename(columns={"Unnamed: 0": "aid"})
    met_raw["aid"] = met_raw["aid"].astype(str)

    # log2(1 + value) transform on metabolite columns
    met_cols = [c for c in met_raw.columns if c != "aid"]
    met_vals = met_raw[met_cols].values.astype(float)
    met_log = np.log2(1 + met_vals)
    met_df = pd.DataFrame(met_log, columns=met_cols)
    met_df["aid"] = met_raw["aid"].values

    # ---- 3. Map stimuli to metabolite rows ----
    exc_aids = [stim_to_aid[s] for s in mat_ash.index[exc_mask]]
    inh_aids = [stim_to_aid[s] for s in mat_ash.index[inh_mask]]

    exc_met = met_df[met_df["aid"].isin(exc_aids)][met_cols].values
    inh_met = met_df[met_df["aid"].isin(inh_aids)][met_cols].values

    print(f"Metabolite matrix: {len(met_df)} AIDs, {len(met_cols)} metabolites")
    print(f"  Exc with metabolite data: {len(exc_met)}")
    print(f"  Inh with metabolite data: {len(inh_met)}")

    # ---- 4. Per-metabolite test (Mann-Whitney U) ----
    results = []
    for j, col in enumerate(met_cols):
        exc_vals = exc_met[:, j]
        inh_vals = inh_met[:, j]

        # Mann-Whitney U (robust to non-normality)
        try:
            u_stat, p_val = stats.mannwhitneyu(exc_vals, inh_vals, alternative="two-sided")
        except ValueError:
            p_val = 1.0
            u_stat = np.nan

        # Effect size: Cliff's delta (probability that exc > inh minus reverse)
        # Cliff's d = (2*U / (n1*n2)) - 1
        n1, n2 = len(exc_vals), len(inh_vals)
        cliff_d = (2 * u_stat / (n1 * n2)) - 1 if n1 * n2 > 0 else np.nan

        # Direction: positive = higher in exc, negative = higher in inh
        exc_mean = np.mean(exc_vals)
        inh_mean = np.mean(inh_vals)
        log2_fc = exc_mean - inh_mean  # difference in log2 space

        results.append({
            "metabolite": col,
            "exc_mean": exc_mean,
            "inh_mean": inh_mean,
            "log2_diff": log2_fc,
            "cliff_delta": cliff_d,
            "p_value": p_val,
            "u_stat": u_stat,
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("p_value")

    # FDR correction
    reject, p_adj, _, _ = multipletests(res_df["p_value"].values, alpha=FDR_ALPHA,
                                         method="fdr_bh")
    res_df["p_adj"] = p_adj
    res_df["significant"] = reject

    n_sig = reject.sum()
    print(f"\nSignificant metabolites (FDR < {FDR_ALPHA}): {n_sig} / {len(res_df)}")

    # ---- 5. Top hits ----
    print("\n=== Top 20 metabolites (by p-value) ===")
    top20 = res_df.head(20)
    for _, row in top20.iterrows():
        sig_mark = "*" if row["significant"] else " "
        direction = "↑ exc" if row["log2_diff"] > 0 else "↓ exc"
        print(f"  {sig_mark} {row['metabolite'][:60]:60s} "
              f"log2_diff={row['log2_diff']:+.4f}  "
              f"cliff_d={row['cliff_delta']:+.3f}  "
              f"p={row['p_value']:.2e}  p_adj={row['p_adj']:.2e}  {direction}")

    # ---- 6. Plots ----
    plot_volcano(res_df)
    plot_top_bars(res_df, n=25)
    plot_heatmap(exc_met, inh_met, exc_aids, inh_aids, met_cols, res_df)

    print(f"\nDone → {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# volcano plot
# ---------------------------------------------------------------------------

def plot_volcano(res_df: pd.DataFrame):
    """Cliff's delta vs -log10(p), FDR-significant highlighted."""
    fig, ax = plt.subplots(figsize=(9, 6))

    sig = res_df["significant"]
    ns = ~sig

    ax.scatter(res_df.loc[ns, "cliff_delta"], -np.log10(res_df.loc[ns, "p_value"]),
               c="#bdc3c7", s=12, alpha=0.6, label=f"NS (n={ns.sum()})")
    ax.scatter(res_df.loc[sig, "cliff_delta"], -np.log10(res_df.loc[sig, "p_value"]),
               c="#e74c3c", s=25, alpha=0.85, edgecolors="#c0392b", linewidths=0.5,
               label=f"FDR<{FDR_ALPHA} (n={sig.sum()})")

    # Label top significant hits
    top_sig = res_df[sig].head(15)
    for _, row in top_sig.iterrows():
        name = row["metabolite"]
        if len(name) > 40:
            name = name[:38] + "…"
        ax.annotate(name, (row["cliff_delta"], -np.log10(row["p_value"])),
                    fontsize=5.5, alpha=0.8, textcoords="offset points",
                    xytext=(4, 3))

    ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel("Cliff's delta (ASH exc > inh)")
    ax.set_ylabel("-log₁₀(p)")
    ax.set_title(f"ASH excitatory vs inhibitory — metabolite association\n"
                 f"(n_exc={res_df['exc_mean'].notna().sum()}, "
                 f"Mann-Whitney U, FDR={FDR_ALPHA})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "volcano.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# top-N bar plot
# ---------------------------------------------------------------------------

def plot_top_bars(res_df: pd.DataFrame, n: int = 25):
    """Horizontal bar plot of top-N metabolites by effect size."""
    top = res_df.sort_values("cliff_delta", key=abs, ascending=False).head(n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top["cliff_delta"]]
    bars = ax.barh(range(len(top)), top["cliff_delta"].values, color=colors, alpha=0.85)

    # p-value annotations
    for i, (_, row) in enumerate(top.iterrows()):
        p_str = f"p={row['p_value']:.1e}" if row["p_value"] >= 0.001 else "p<0.001"
        sig_str = " *" if row["significant"] else ""
        offset = 0.02 if row["cliff_delta"] >= 0 else -0.02
        ha = "left" if row["cliff_delta"] >= 0 else "right"
        ax.text(row["cliff_delta"] + offset, i, f"{p_str}{sig_str}",
                va="center", ha=ha, fontsize=6.5, alpha=0.8)

    # Truncate long names
    labels = []
    for name in top["metabolite"]:
        labels.append(name[:55] + "…" if len(name) > 55 else name)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Cliff's delta (positive = higher in ASH-exc)")
    ax.set_title(f"Top {n} metabolites associated with ASH response phenotype\n"
                 f"(blue = higher in ASH-inh, red = higher in ASH-exc)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_bars.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# metabolite heatmap (exc vs inh stimuli)
# ---------------------------------------------------------------------------

def plot_heatmap(exc_met: np.ndarray, inh_met: np.ndarray,
                 exc_aids: list[str], inh_aids: list[str],
                 met_cols: list[str], res_df: pd.DataFrame):
    """Heatmap of top metabolites × stimuli, sorted by ASH response group."""
    # Top 40 metabolites by significance
    top_met_idx = res_df.head(40).index
    top_met_names = [met_cols[i] for i in range(len(met_cols)) if i in top_met_idx]

    # Re-extract top metabolites
    top_idx_list = list(top_met_idx)
    exc_data = exc_met[:, top_idx_list]
    inh_data = inh_met[:, top_idx_list]

    # Row order: excitatory group first, then inhibitory
    all_data = np.vstack([exc_data, inh_data])
    n_exc = len(exc_data)
    n_inh = len(inh_data)
    n_total = n_exc + n_inh

    # Sort within each group by mean response
    exc_order = np.argsort(exc_data.mean(axis=1))[::-1]
    inh_order = np.argsort(inh_data.mean(axis=1))

    ordered_data = np.vstack([exc_data[exc_order], inh_data[inh_order]])
    ordered_aids = [exc_aids[i] for i in exc_order] + [inh_aids[i] for i in inh_order]

    # Z-score columns for visualization
    col_mean = ordered_data.mean(axis=0, keepdims=True)
    col_std = ordered_data.std(axis=0, keepdims=True)
    col_std[col_std == 0] = 1
    ordered_z = (ordered_data - col_mean) / col_std

    fig, ax = plt.subplots(figsize=(max(12, n_total * 0.25), max(8, 40 * 0.3)))

    vmax = max(2.5, np.percentile(np.abs(ordered_z), 98))
    im = ax.imshow(ordered_z.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Group separator
    ax.axvline(n_exc - 0.5, color="black", lw=2)

    # Labels
    ax.set_xticks(range(n_total))
    ax.set_xticklabels(ordered_aids, fontsize=5, rotation=90)
    ax.xaxis.tick_top()

    # Group markers at top
    ax.text(n_exc / 2 - 0.5, -1.5, f"ASH exc (n={n_exc})", ha="center",
            fontsize=8, fontweight="bold", color="#e74c3c",
            transform=ax.get_xaxis_transform())
    ax.text(n_exc + n_inh / 2 - 0.5, -1.5, f"ASH inh (n={n_inh})", ha="center",
            fontsize=8, fontweight="bold", color="#3498db",
            transform=ax.get_xaxis_transform())

    # Short metabolite names for y-axis
    short_names = [n[:50] + "…" if len(n) > 50 else n for n in top_met_names]
    ax.set_yticks(range(len(top_met_names)))
    ax.set_yticklabels(short_names, fontsize=7)

    plt.colorbar(im, ax=ax, shrink=0.8, label="log2(1+FC) z-score")
    ax.set_title("Top 40 metabolites: ASH exc vs inh stimuli\n"
                 "(z-scored per metabolite, sorted by ASH response)",
                 fontsize=11, pad=25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
