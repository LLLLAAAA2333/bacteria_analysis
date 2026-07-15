"""ASH–AWA response coupling: when ASH is excitatory, what does AWA do?

Simple exploratory analysis — no test suite, no stage, no reusable API.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis._data_loaders import enrich_neural_dataframe

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

RESPONSE_WINDOW = (5, 15)          # time points for "response"
EXCITATORY_THRESHOLD = 0.05        # mean ΔF/F₀ above this → excitatory
INHIBITORY_THRESHOLD = -0.05       # mean ΔF/F₀ below this → inhibitory

PAIR = ("ASH", "AWA")
OUTPUT_DIR = Path("results/ash_awa_coupling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# data prep (same logic as inspect_neural_dimensionality._prepare_neuron_prototypes)
# ---------------------------------------------------------------------------

_LR_MERGE = {
    "ASH": ("ASHL", "ASHR"),
    "AWA": ("AWAL", "AWAR"),
}


def _build_prototype(raw: pd.DataFrame, neuron: str) -> tuple[pd.DataFrame, list[int]]:
    merged = _LR_MERGE.get(neuron)
    raw_names = list(merged) if merged else [neuron]

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


def _window_response(mat: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    """Mean ΔF/F₀ in the response window, per stimulus."""
    return mat.loc[:, window[0]:window[1]].mean(axis=1)


def _classify(series: pd.Series) -> pd.Series:
    """Classify each stimulus as excitatory / inhibitory / neutral."""
    def _label(v):
        if v > EXCITATORY_THRESHOLD:
            return "excitatory"
        if v < INHIBITORY_THRESHOLD:
            return "inhibitory"
        return "neutral"
    return series.apply(_label)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def main():
    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()

    # ---- build prototypes ----
    n1, n2 = PAIR
    mat1, tp1 = _build_prototype(raw, n1)
    mat2, tp2 = _build_prototype(raw, n2)

    common_stim = mat1.index.intersection(mat2.index)
    mat1 = mat1.loc[common_stim]
    mat2 = mat2.loc[common_stim]
    print(f"Common stimuli: {len(common_stim)}")

    resp1 = _window_response(mat1, RESPONSE_WINDOW)
    resp2 = _window_response(mat2, RESPONSE_WINDOW)
    cls1 = _classify(resp1)
    cls2 = _classify(resp2)

    # ---- contingency table ----
    print(f"\n=== Contingency table ({n1} rows × {n2} columns) ===")
    ctab = pd.crosstab(cls1, cls2, margins=True)
    print(ctab)

    # Fisher exact for excitatory co-occurrence
    both_exc = ((cls1 == "excitatory") & (cls2 == "excitatory")).sum()
    n1_exc = (cls1 == "excitatory").sum()
    n2_exc = (cls2 == "excitatory").sum()
    n_total = len(common_stim)
    # 2×2: ASH exc/not-exc × AWA exc/not-exc
    table_2x2 = np.array([
        [both_exc, n1_exc - both_exc],
        [n2_exc - both_exc, n_total - n1_exc - n2_exc + both_exc],
    ])
    odds_ratio, p_fisher = stats.fisher_exact(table_2x2)
    print(f"\nFisher exact (excitatory co-occurrence): p={p_fisher:.4g}, OR={odds_ratio:.3f}")
    print(f"  ASH exc & AWA exc: {both_exc}/{n1_exc} = {both_exc/n1_exc:.1%} of ASH-exc stimuli")
    print(f"  ASH exc & AWA not-exc: {n1_exc - both_exc}/{n1_exc} = {(n1_exc-both_exc)/n1_exc:.1%}")

    # ---- full correlation ----
    r, p_corr = stats.pearsonr(resp1, resp2)
    rho, p_spear = stats.spearmanr(resp1, resp2)
    print(f"\nPearson r = {r:.4f} (p={p_corr:.4g})")
    print(f"Spearman ρ = {rho:.4f} (p={p_spear:.4g})")

    # ---- per-class breakdown ----
    print(f"\n=== AWA response when ASH is {n1} category ===")
    for cat in ["excitatory", "neutral", "inhibitory"]:
        mask = cls1 == cat
        if mask.sum() == 0:
            continue
        sub = resp2[mask]
        print(f"  ASH {cat} (n={mask.sum()}): AWA mean={sub.mean():.4f}, "
              f"AWA exc={((cls2[mask]=='excitatory').sum())}, "
              f"AWA inh={((cls2[mask]=='inhibitory').sum())}, "
              f"AWA neu={((cls2[mask]=='neutral').sum())}")

    # ---- plots ----
    plot_scatter(resp1, resp2, cls1, cls2, stim_info, r, rho)
    plot_top_pairs(mat1, mat2, tp1, resp1, resp2, stim_info)
    plot_full_traces(mat1, mat2, tp1, resp1, resp2, stim_info, common_stim)

    print(f"\nDone → {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# plot 1: scatter
# ---------------------------------------------------------------------------

def plot_scatter(resp1, resp2, cls1, cls2, stim_info, r, rho):
    n1, n2 = PAIR
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {"excitatory": "#e74c3c", "inhibitory": "#3498db", "neutral": "#95a5a6"}
    for cat in ["excitatory", "inhibitory", "neutral"]:
        mask = cls1 == cat
        if mask.sum() == 0:
            continue
        ax.scatter(
            resp1[mask], resp2[mask],
            c=colors[cat], alpha=0.7, s=50, edgecolors="white", linewidths=0.5,
            label=f"{n1} {cat} (n={mask.sum()})",
        )

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(EXCITATORY_THRESHOLD, color="#e74c3c", ls=":", lw=0.6, alpha=0.5)
    ax.axhline(INHIBITORY_THRESHOLD, color="#3498db", ls=":", lw=0.6, alpha=0.5)
    ax.axvline(EXCITATORY_THRESHOLD, color="#e74c3c", ls=":", lw=0.6, alpha=0.5)
    ax.axvline(INHIBITORY_THRESHOLD, color="#3498db", ls=":", lw=0.6, alpha=0.5)

    # Annotate top-N strongest ASH-excitatory stimuli
    top_idx = resp1.nlargest(10).index
    for s in top_idx:
        aid = stim_info.loc[s, "aid"] if s in stim_info.index else s
        ax.annotate(
            str(aid), (resp1[s], resp2[s]),
            fontsize=6, alpha=0.7, textcoords="offset points", xytext=(4, 4),
        )

    ax.set_xlabel(f"{n1} response (mean ΔF/F₀, window {RESPONSE_WINDOW[0]}-{RESPONSE_WINDOW[1]})")
    ax.set_ylabel(f"{n2} response (mean ΔF/F₀, window {RESPONSE_WINDOW[0]}-{RESPONSE_WINDOW[1]})")
    ax.set_title(f"{n1}–{n2} response coupling\nPearson r={r:.3f}, Spearman ρ={rho:.3f}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{n1}_{n2}_scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot 2: top-5 congruent vs anti-congruent trajectory pairs
# ---------------------------------------------------------------------------

def plot_top_pairs(mat1, mat2, timepoints, resp1, resp2, stim_info):
    """Show trajectory pairs where ASH/AWA most agree or disagree."""
    n1, n2 = PAIR

    # Congruent: both strongly excitatory
    both_exc_score = resp1 + resp2  # both high → high score
    top_congruent = both_exc_score.nlargest(5).index

    # Anti-congruent: ASH excitatory, AWA inhibitory
    anti_score = resp1 - resp2  # ASH high, AWA low → high score
    top_anti = anti_score.nlargest(5).index

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for col, stim in enumerate(top_congruent):
        ax1, ax2 = axes[0, col], axes[1, col]
        aid = stim_info.loc[stim, "aid"] if stim in stim_info.index else stim

        ax1.plot(timepoints, mat1.loc[stim].values, color="#e74c3c", lw=1.5, label=n1)
        ax1.plot(timepoints, mat2.loc[stim].values, color="#3498db", lw=1.5, label=n2)
        ax1.axvspan(*RESPONSE_WINDOW, alpha=0.1, color="#e74c3c")
        ax1.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
        ax1.set_title(f"{aid}\n{n1}:{resp1[stim]:.3f}  {n2}:{resp2[stim]:.3f}", fontsize=8)
        ax1.legend(fontsize=7)
        ax1.tick_params(labelsize=7)

        # Anti-congruent
        anti_stim = top_anti[col]
        ax2.plot(timepoints, mat1.loc[anti_stim].values, color="#e74c3c", lw=1.5, label=n1)
        ax2.plot(timepoints, mat2.loc[anti_stim].values, color="#3498db", lw=1.5, label=n2)
        ax2.axvspan(*RESPONSE_WINDOW, alpha=0.1, color="#e74c3c")
        ax2.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
        anti_aid = stim_info.loc[anti_stim, "aid"] if anti_stim in stim_info.index else anti_stim
        ax2.set_title(f"{anti_aid}\n{n1}:{resp1[anti_stim]:.3f}  {n2}:{resp2[anti_stim]:.3f}", fontsize=8)
        ax2.legend(fontsize=7)
        ax2.tick_params(labelsize=7)

    axes[0, 0].set_ylabel("ΔF/F₀ (congruent)", fontsize=9)
    axes[1, 0].set_ylabel("ΔF/F₀ (anti-congruent)", fontsize=9)
    fig.suptitle(f"Top-5 congruent (both exc) vs anti-congruent ({n1} exc, {n2} inh) trajectories",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{n1}_{n2}_top_pairs.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot 3: all traces sorted by ASH response, with AWA overlay
# ---------------------------------------------------------------------------

def plot_full_traces(mat1, mat2, timepoints, resp1, resp2, stim_info, common_stim):
    """Heatmap-style view: all stimuli sorted by ASH response, AWA shown alongside."""
    n1, n2 = PAIR

    order = resp1.sort_values().index
    n_stim = len(order)

    data1 = mat1.loc[order].values
    data2 = mat2.loc[order].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, n_stim * 0.25)),
                                    gridspec_kw={"width_ratios": [0.85, 0.15]})

    vmax = max(np.percentile(np.abs(data1), 99), np.percentile(np.abs(data2), 99))
    vmin = -vmax

    for ax, data, name in [(ax1, data1, n1), (ax2, data2, n2)]:
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       extent=[-0.5, data.shape[1] - 0.5, n_stim - 0.5, -0.5])
        ax.axvspan(-0.5, RESPONSE_WINDOW[0] - 0.5, alpha=0.06, color="gray")
        ax.axvspan(RESPONSE_WINDOW[0] - 0.5, RESPONSE_WINDOW[1] - 0.5, alpha=0.08, color="#e74c3c")
        ax.set_yticks(range(n_stim))
        ax.set_yticklabels([str(stim_info.loc[s, "aid"]) for s in order], fontsize=5)
        ax.set_xlabel("Time point")
        ax.set_title(f"{name} (sorted by {n1} response)", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{n1}_{n2}_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
