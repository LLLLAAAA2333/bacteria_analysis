"""Highlight frequent neural cluster patterns (n>4) on chemical HMDS.

Loads the 106bac neural data, computes per-neuron Ward cluster labels,
identifies unique global cluster-state combinations that occur in >4 stimuli,
then overlays those frequent patterns as coloured groups on the chemical
PCA Euclidean-distance HMDS (Poincaré ball).

Uses the EXACT same HMDS pipeline as chemical_pca_hmds.py:
  - Precomputed PCA Euclidean RDM (results/chemical_pca_ward/)
  - normalize_to_max_two → scipy_hyperbolic_mds → Poincaré projection

Rare patterns (n≤4) are shown in light grey as context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "exploratory"))

from bacteria_analysis._data_loaders import enrich_neural_dataframe
from compare_86bac_chord_hmds import (
    normalize_to_max_two,
    scipy_hyperbolic_mds,
    lorentz_to_poincare,
    recenter_poincare,
    preservation_metrics,
    upper_triangle,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
CHEMICAL_RDM_DIR = Path("results/chemical_pca_ward")
OUTPUT_DIR = Path("results/correlation_analysis/awa_k2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ_THRESHOLD = 4  # highlight patterns with > this many stimuli

ALL_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)

_LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}

K_MAP: dict[str, int] = {
    "ASG": 1, "AWCOFF": 1, "ASER": 1,
    "AWA": 2,  # override: k=2
}
# default k=2 for others

FILTERED_NEURONS = tuple(n for n in ALL_NEURONS if K_MAP.get(n, 2) > 1)

# ── HMDS parameters (exactly matching chemical_pca_hmds.py) ──
HMDS_DIM = 3
HMDS_STARTS = 8
HMDS_MAXITER = 900
HMDS_SEED = 42

# ---------------------------------------------------------------------------
# neural cluster patterns (from neuron_cluster_cooccurrence)
# ---------------------------------------------------------------------------

def _neuron_raw_names(neuron: str) -> list[str]:
    merged = _LR_MERGE.get(neuron)
    if merged is not None:
        return list(merged)
    return [neuron]


def _build_prototype_matrix(raw: pd.DataFrame, neuron: str) -> pd.DataFrame:
    raw_names = _neuron_raw_names(neuron)
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
    return proto.pivot(index="stimulus", columns="time_point",
                       values="delta_F_over_F0")


def _cluster_labels(mat: pd.DataFrame, k: int) -> np.ndarray:
    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    r = np.corrcoef(mat_z.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    Z = linkage(squareform(d), method="ward")
    labels = fcluster(Z, k, criterion="maxclust") if k > 1 else np.ones(mat.shape[0], dtype=int)
    return labels


def compute_cluster_patterns(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (labels_kmapped, combo_df, mats).

    labels_kmapped: stimulus × neuron, cluster labels per neuron
    combo_df: unique combinations with n_stimuli and neuron label columns
    mats: neuron → stimulus×timepoint prototype DataFrames
    """
    # build prototype matrices
    mats: dict[str, pd.DataFrame] = {}
    for neuron in ALL_NEURONS:
        mats[neuron] = _build_prototype_matrix(raw, neuron)

    common_stimuli = sorted(set.intersection(
        *[set(m.index) for m in mats.values()]
    ))
    print(f"Common stimuli (neural): {len(common_stimuli)}")

    labels_kmapped = pd.DataFrame(
        index=common_stimuli, columns=list(ALL_NEURONS), dtype=int
    )
    for neuron in ALL_NEURONS:
        mat = mats[neuron].loc[common_stimuli]
        k_opt = K_MAP.get(neuron, 2)
        labels_kmapped[neuron] = _cluster_labels(mat, k=k_opt)

    # build unique combinations (k>1 neurons only)
    filtered = labels_kmapped[list(FILTERED_NEURONS)]
    label_strs = filtered.astype(str).agg("|".join, axis=1)
    combo_counts = label_strs.value_counts()

    combo_labels_list = [
        list(map(int, s.split("|"))) for s in combo_counts.index
    ]
    combo_df = pd.DataFrame(
        combo_labels_list,
        columns=list(FILTERED_NEURONS),
        index=combo_counts.index,
    )
    combo_df.insert(0, "n_stimuli", combo_counts.values)

    print(f"Unique cluster combinations (k>1): {len(combo_df)}")
    print(f"Frequent patterns (n>{FREQ_THRESHOLD}): "
          f"{(combo_df['n_stimuli'] > FREQ_THRESHOLD).sum()}")

    return labels_kmapped, combo_df, mats


# ---------------------------------------------------------------------------
# chemical HMDS — exactly mirrors chemical_pca_hmds.py pipeline
# ---------------------------------------------------------------------------

def load_chemical_rdm() -> tuple[np.ndarray, list[str]]:
    """Load the precomputed PCA Euclidean RDM, matching chemical_pca_hmds.py.

    Returns (dist_clean, aids) where dist_clean is a cleaned, symmetrized
    numpy array and aids is the ordered list of AID labels.
    """
    rdm = pd.read_csv(CHEMICAL_RDM_DIR / "chemical_pca_euclidean_rdm.csv", index_col=0)
    colormap = pd.read_csv(CHEMICAL_RDM_DIR / "aid_colormap.csv", index_col="aid")

    # Ensure consistent order (same as chemical_pca_hmds.load_data)
    aids = sorted(set(rdm.index) & set(colormap.index))
    dist_matrix = rdm.loc[aids, aids].to_numpy(dtype=float)

    # Clean
    dist_clean = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_clean, 0.0)
    dist_clean = np.maximum(dist_clean, 0.0)

    return dist_clean, aids


def embed_chemical_hmds(
    dist_clean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run 2D hyperbolic MDS, exactly matching chemical_pca_hmds.py main().

    Returns (poincare_2d, dist_norm, predicted_2d, metrics_2d).
    """
    dist_norm = normalize_to_max_two(dist_clean)
    n_params = dist_clean.shape[0] * HMDS_DIM

    lorentz, embedded, lam, _ = scipy_hyperbolic_mds(
        dist_norm, dim=HMDS_DIM, starts=HMDS_STARTS, maxiter=HMDS_MAXITER, seed=HMDS_SEED,
    )
    poincare = recenter_poincare(lorentz_to_poincare(lorentz))
    predicted = embedded / lam
    metrics = preservation_metrics(dist_norm, embedded, predicted, n_params=n_params)
    return poincare, dist_norm, predicted, metrics


# ---------------------------------------------------------------------------
# neural HMDS — stimulus × (neurons × timepoints) correlation distance
# ---------------------------------------------------------------------------

def build_neural_rdm_from_prototypes(
    mats: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, list[str]]:
    """Build neural Pearson distance RDM from per-neuron prototype matrices.

    Concatenates all neuron time-courses into one feature matrix,
    z-scores per neuron, then computes correlation distance.
    """
    common_stimuli = sorted(set.intersection(
        *[set(m.index) for m in mats.values()]
    ))

    # Concatenate: one row per stimulus, columns = neuron×timepoint
    blocks = []
    for neuron in ALL_NEURONS:
        mat = mats[neuron].loc[common_stimuli]
        # z-score per stimulus (row-wise, matching cluster pipeline)
        mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
        mat_z.columns = [f"{neuron}_t{t}" for t in mat_z.columns]
        blocks.append(mat_z.values)

    feature_matrix = np.column_stack(blocks)  # (n_stim × n_features)

    # Pearson correlation → distance
    r = np.corrcoef(feature_matrix)
    dist = np.clip(1 - r, 0, None)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2

    return dist, common_stimuli


# ---------------------------------------------------------------------------
# mapping: neural stimulus → AID → HMDS coordinate
# ---------------------------------------------------------------------------

def map_stimulus_to_aid(raw: pd.DataFrame) -> pd.DataFrame:
    """Build stimulus→AID mapping from enriched neural data."""
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()
    return stim_info


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def plot_3d_poincare_with_highlights(
    coords: np.ndarray,
    aids: list[str],
    aid_to_pattern: dict[str, int],
    pattern_order: list[int],
    pattern_info: dict[int, dict],
    metrics: dict,
    output_path: Path,
    *,
    title_prefix: str = "Chemical PCA",
):
    """3D Poincaré ball: frequent patterns coloured, rare ones grey.

    Matches the style of chemical_pca_hmds.plot_3d_poincare.
    """
    n_frequent = sum(
        1 for pid in pattern_order if pattern_info[pid]["n"] > FREQ_THRESHOLD
    )

    # --- colour assignment ---
    if n_frequent <= 10:
        base_cmap = plt.get_cmap("tab10")
        freq_colors = [base_cmap(i) for i in range(n_frequent)]
    else:
        base_cmap = plt.get_cmap("tab20")
        freq_colors = [base_cmap(i % 20) for i in range(n_frequent)]

    point_colors = []
    point_sizes = []
    point_alphas = []
    point_edges = []
    for aid in aids:
        pid = aid_to_pattern.get(aid, -1)
        if pid >= 0 and pattern_info[pid]["n"] > FREQ_THRESHOLD:
            idx = pattern_order.index(pid)
            point_colors.append(freq_colors[idx])
            point_sizes.append(64)
            point_alphas.append(0.92)
            point_edges.append("white")
        else:
            point_colors.append("#c0c0c0")
            point_sizes.append(36)
            point_alphas.append(0.45)
            point_edges.append("#a0a0a0")

    # --- draw ---
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Wireframe unit sphere (matching chemical_pca_hmds style)
    u, v = np.mgrid[0:2 * np.pi:32j, 0:np.pi:16j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="#4b5563", linewidth=0.85, alpha=0.40,
                      rstride=3, cstride=3)

    # Draw rare points first (behind)
    for i, aid in enumerate(aids):
        if point_colors[i] == "#c0c0c0":
            ax.scatter(
                coords[i, 0], coords[i, 1], coords[i, 2],
                c=point_colors[i], s=point_sizes[i], alpha=point_alphas[i],
                edgecolors=point_edges[i], linewidths=0.3, zorder=2,
            )

    # Draw frequent points on top
    for i, aid in enumerate(aids):
        if point_colors[i] != "#c0c0c0":
            ax.scatter(
                coords[i, 0], coords[i, 1], coords[i, 2],
                c=[point_colors[i]], s=point_sizes[i], alpha=point_alphas[i],
                edgecolors=point_edges[i], linewidths=0.6, zorder=4,
            )
            # Label all frequent-pattern points
            ax.text(
                coords[i, 0], coords[i, 1], coords[i, 2],
                str(aid), fontsize=4, alpha=0.55, ha="center",
            )

    # Axis setup (matching chemical_pca_hmds.set_poincare_3d_axis)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_zlim(-1.02, 1.02)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("Poincare 1", fontsize=9)
    ax.set_ylabel("Poincare 2", fontsize=9)
    ax.set_zlabel("Poincare 3", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(
        f"{title_prefix} HMDS 3D — frequent neural cluster patterns (n>{FREQ_THRESHOLD})\n"
        f"ρ={metrics['distance_spearman']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )

    # --- legend (2D proxy on figure) ---
    from matplotlib.patches import Patch
    legend_elements = []
    for pid in pattern_order:
        info = pattern_info[pid]
        n_stim = info["n"]
        if n_stim <= FREQ_THRESHOLD:
            break
        idx = pattern_order.index(pid)
        label_parts = info["label"].split("|")
        neuron_assignments = ", ".join(
            f"{n}={label_parts[i]}" for i, n in enumerate(FILTERED_NEURONS)
        )
        legend_elements.append(
            Patch(
                facecolor=freq_colors[idx], edgecolor="white",
                label=f"P{pid} (n={n_stim})\n{neuron_assignments}",
            )
        )
    legend_elements.append(
        Patch(
            facecolor="#c0c0c0", edgecolor="#a0a0a0",
            label=f"rare (n≤{FREQ_THRESHOLD})",
        )
    )
    fig.legend(
        handles=legend_elements, fontsize=5.5, loc="upper right",
        bbox_to_anchor=(1.02, 0.92), ncol=1, frameon=True,
        title="Neural cluster patterns", title_fontsize=6.5,
        borderpad=0.5, labelspacing=0.3,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# interactive 3D HTML (plotly)
# ---------------------------------------------------------------------------

def plot_3d_html(
    coords: np.ndarray,
    aids: list[str],
    aid_to_pattern: dict[str, int],
    pattern_order: list[int],
    pattern_info: dict[int, dict],
    stim_to_genus: dict[str, str],
    metrics: dict,
    output_path: Path,
    *,
    title_prefix: str = "Chemical PCA",
):
    """Interactive 3D Poincaré ball via plotly — rotate/zoom/hover."""
    import plotly.graph_objects as go

    n_frequent = sum(
        1 for pid in pattern_order if pattern_info[pid]["n"] > FREQ_THRESHOLD
    )

    # --- per-point hover info ---
    hover_texts = []
    for aid in aids:
        pid = aid_to_pattern.get(aid, -1)
        if pid >= 0 and pattern_info[pid]["n"] > FREQ_THRESHOLD:
            info = pattern_info[pid]
            label_parts = info["label"].split("|")
            neuron_detail = "<br>".join(
                f"  {n}: cluster {label_parts[i]}"
                for i, n in enumerate(FILTERED_NEURONS)
            )
            genus = stim_to_genus.get(aid, "?")
            hover = (
                f"<b>{aid}</b> ({genus})<br>"
                f"Pattern P{pid} (n={info['n']})<br>"
                f"{neuron_detail}"
            )
        else:
            genus = stim_to_genus.get(aid, "?")
            hover = f"<b>{aid}</b> ({genus})<br>rare pattern (n≤{FREQ_THRESHOLD})"
        hover_texts.append(hover)

    # --- colour assignment ---
    FREQ_HEX = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5"]

    fig = go.Figure()

    # --- rare points (grey, smaller) ---
    rare_mask = np.array([
        aid_to_pattern.get(a, -1) < 0
        or pattern_info[aid_to_pattern[a]]["n"] <= FREQ_THRESHOLD
        for a in aids
    ])
    if rare_mask.any():
        fig.add_trace(go.Scatter3d(
            x=coords[rare_mask, 0],
            y=coords[rare_mask, 1],
            z=coords[rare_mask, 2],
            mode="markers",
            marker=dict(
                size=3, color="#b0b0b0", opacity=0.40,
                line=dict(color="#999999", width=0.3),
            ),
            text=[hover_texts[i] for i in range(len(aids)) if rare_mask[i]],
            hoverinfo="text",
            name=f"rare (n≤{FREQ_THRESHOLD})",
            showlegend=True,
        ))

    # --- frequent patterns, one trace each ---
    for pid in pattern_order:
        info = pattern_info[pid]
        if info["n"] <= FREQ_THRESHOLD:
            break
        idx = pattern_order.index(pid)
        color = FREQ_HEX[idx % len(FREQ_HEX)]

        pids = np.array([aid_to_pattern.get(a, -1) for a in aids])
        mask = pids == pid

        label_parts = info["label"].split("|")
        neuron_short = ",".join(
            f"{FILTERED_NEURONS[i]}:{label_parts[i]}" for i in range(len(FILTERED_NEURONS))
        )
        legend_label = f"P{pid} n={info['n']}"

        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=coords[mask, 2],
            mode="markers+text",
            marker=dict(
                size=5, color=color, opacity=0.92,
                line=dict(color="white", width=0.6),
            ),
            text=[str(a) for a in np.array(aids)[mask]],
            textposition="top center",
            textfont=dict(size=7, color=color),
            hovertext=[hover_texts[i] for i in range(len(aids)) if mask[i]],
            hoverinfo="text",
            name=legend_label,
            showlegend=True,
        ))

    # --- unit sphere wireframe (lines, not dots) ---
    # Latitude circles
    n_lat, n_lon = 16, 32
    u = np.linspace(0, 2 * np.pi, n_lon)
    v = np.linspace(0, np.pi, n_lat)

    # Longitude lines
    for i in range(0, n_lon, 4):
        theta = np.full(n_lat, u[i])
        phi = v
        sx = np.cos(theta) * np.sin(phi)
        sy = np.sin(theta) * np.sin(phi)
        sz = np.cos(phi)
        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz, mode="lines",
            line=dict(color="#6b7280", width=0.5),
            hoverinfo="skip", showlegend=False,
        ))

    # Latitude lines
    for j in range(1, n_lat - 1):
        theta = u
        phi = np.full(n_lon, v[j])
        sx = np.cos(theta) * np.sin(phi)
        sy = np.sin(theta) * np.sin(phi)
        sz = np.full(n_lon, np.cos(phi[0]))
        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz, mode="lines",
            line=dict(color="#6b7280", width=0.5),
            hoverinfo="skip", showlegend=False,
        ))

    # Add a single legend entry for the sphere
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode="lines",
        line=dict(color="#6b7280", width=0.5),
        name="unit sphere", showlegend=True,
    ))

    # --- layout ---
    rho = metrics['distance_spearman']
    stress = metrics['normalized_raw_stress']
    fig.update_layout(
        title=dict(
            text=f"{title_prefix} HMDS 3D — frequent neural cluster patterns (n>{FREQ_THRESHOLD})<br>"
                 f"<sup>ρ={rho:.3f}  stress={stress:.3f}</sup>",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis=dict(range=[-1.05, 1.05], title="Poincaré 1"),
            yaxis=dict(range=[-1.05, 1.05], title="Poincaré 2"),
            zaxis=dict(range=[-1.05, 1.05], title="Poincaré 3"),
            aspectmode="cube",
        ),
        legend=dict(
            font=dict(size=10),
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        width=1100, height=900,
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # ── 1. Load neural data & compute cluster patterns ──
    print("=" * 60)
    print("Step 1: Computing neural cluster patterns...")
    print("=" * 60)
    raw = pd.read_parquet("data/106bac.parquet")
    labels_kmapped, combo_df, mats = compute_cluster_patterns(raw)

    # Build stimulus→pattern mapping
    filtered = labels_kmapped[list(FILTERED_NEURONS)]
    stim_to_label_str = filtered.astype(str).agg("|".join, axis=1)

    # Assign pattern IDs (int) to unique label strings
    unique_labels = sorted(combo_df.index, key=lambda x: combo_df.loc[x, "n_stimuli"], reverse=True)
    label_to_pid = {label: i for i, label in enumerate(unique_labels)}
    pattern_info = {
        label_to_pid[label]: {
            "n": combo_df.loc[label, "n_stimuli"],
            "label": label,
        }
        for label in unique_labels
    }
    pattern_order = [label_to_pid[l] for l in unique_labels]

    # Map stimulus → AID
    stim_to_aid = map_stimulus_to_aid(raw)
    aid_to_stimulus = {}
    for stim, row in stim_to_aid.iterrows():
        aid_to_stimulus[row["aid"]] = stim

    # ── 2. Load precomputed chemical PCA RDM & embed (matching chemical_pca_hmds.py) ──
    print("\n" + "=" * 60)
    print("Step 2: Loading chemical PCA Euclidean RDM & running HMDS...")
    print("=" * 60)
    dist_clean, aids = load_chemical_rdm()
    print(f"Loaded: {len(aids)} strains, distance matrix {dist_clean.shape}")

    poincare, dist_norm, predicted, metrics = embed_chemical_hmds(dist_clean)
    print(f"  ρ={metrics['distance_spearman']:.3f}  "
          f"stress={metrics['normalized_raw_stress']:.3f}")

    # ── 3. Map AID → pattern ──
    aid_to_pattern: dict[str, int] = {}
    aid_to_label: dict[str, str] = {}
    for aid in aids:
        stim = aid_to_stimulus.get(aid)
        if stim is not None and stim in stim_to_label_str.index:
            label = stim_to_label_str.loc[stim]
            aid_to_pattern[aid] = label_to_pid.get(label, -1)
            aid_to_label[aid] = label
        else:
            aid_to_pattern[aid] = -1
            aid_to_label[aid] = ""

    n_mapped = sum(1 for v in aid_to_pattern.values() if v >= 0)
    print(f"AIDs mapped to patterns: {n_mapped}/{len(aids)}")

    # ── 4. Report frequent patterns ──
    print("\n" + "=" * 60)
    print(f"Frequent patterns (n > {FREQ_THRESHOLD}):")
    print("=" * 60)
    for pid in pattern_order:
        info = pattern_info[pid]
        n_stim = info["n"]
        if n_stim <= FREQ_THRESHOLD:
            break
        label_parts = info["label"].split("|")
        neuron_str = "  ".join(
            f"{n}={label_parts[i]}" for i, n in enumerate(FILTERED_NEURONS)
        )
        pattern_aids = [a for a, p in aid_to_pattern.items() if p == pid]
        print(f"  P{pid}: n={n_stim}  {neuron_str}")
        print(f"        AIDs: {', '.join(str(a) for a in sorted(pattern_aids))}")

    # ── 5. Plot 3D HMDS with pattern highlights ──
    print("\n" + "=" * 60)
    print("Step 3: Plotting 3D HMDS with pattern highlights...")
    print("=" * 60)

    plot_3d_poincare_with_highlights(
        poincare, aids, aid_to_pattern,
        pattern_order, pattern_info, metrics,
        OUTPUT_DIR / "chemical_hmds_3d_frequent_patterns.png",
    )

    # ── 5b. Interactive 3D HTML ──
    # Build aid→genus mapping for hover info
    stim_to_aid_df = map_stimulus_to_aid(raw)
    stim_to_genus = {}
    for stim, row in stim_to_aid_df.iterrows():
        stim_to_genus[row["aid"]] = row.get("genus", "?")

    plot_3d_html(
        poincare, aids, aid_to_pattern,
        pattern_order, pattern_info, stim_to_genus, metrics,
        OUTPUT_DIR / "chemical_hmds_3d_frequent_patterns.html",
    )

    # ── 6. Shepard diagram (matching chemical_pca_hmds.plot_shepard) ──
    from scipy.stats import spearmanr, pearsonr
    orig = upper_triangle(dist_norm)
    pred = upper_triangle(predicted)
    limit = float(max(orig.max(), pred.max()) * 1.04)

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.scatter(orig, pred, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit], [0, limit], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Input Euclidean distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"Shepard diagram\n"
        f"ρ={metrics['distance_spearman']:.3f}  "
        f"r={metrics['distance_pearson']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )
    ax.grid(True, color="#e5e7eb", lw=0.4)
    fig.savefig(OUTPUT_DIR / "chemical_hmds_3d_frequent_patterns_shepard.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → shepard diagram")

    # ── 7. Neural HMDS ──
    print("\n" + "=" * 60)
    print("Step 4: Building neural HMDS (Pearson distance, all neurons)...")
    print("=" * 60)

    neural_dist, neural_stimuli = build_neural_rdm_from_prototypes(mats)
    print(f"Neural RDM: {len(neural_stimuli)} stimuli")

    neural_poincare, neural_dist_norm, neural_predicted, neural_metrics = \
        embed_chemical_hmds(neural_dist)
    print(f"  ρ={neural_metrics['distance_spearman']:.3f}  "
          f"stress={neural_metrics['normalized_raw_stress']:.3f}")

    # Map neural stimuli → AID for pattern lookup & hover
    stim_to_aid_lookup: dict[str, str] = {}
    for stim, row in stim_to_aid_df.iterrows():
        stim_to_aid_lookup[stim] = row["aid"]

    # Build AID→pattern for neural HMDS (AID-keyed, matching chemical approach)
    neural_aid_to_pattern: dict[str, int] = {}
    neural_aid_labels: list[str] = []
    for stim in neural_stimuli:
        aid = stim_to_aid_lookup.get(stim, stim)
        neural_aid_labels.append(aid)
        if stim in stim_to_label_str.index:
            label = stim_to_label_str.loc[stim]
            neural_aid_to_pattern[aid] = label_to_pid.get(label, -1)
        else:
            neural_aid_to_pattern[aid] = -1

    # Plot neural 3D PNG
    plot_3d_poincare_with_highlights(
        neural_poincare, neural_aid_labels, neural_aid_to_pattern,
        pattern_order, pattern_info, neural_metrics,
        OUTPUT_DIR / "neural_hmds_3d_frequent_patterns.png",
        title_prefix="Neural",
    )

    # Plot neural 3D HTML
    plot_3d_html(
        neural_poincare, neural_aid_labels, neural_aid_to_pattern,
        pattern_order, pattern_info, stim_to_genus, neural_metrics,
        OUTPUT_DIR / "neural_hmds_3d_frequent_patterns.html",
        title_prefix="Neural",
    )

    # Neural Shepard
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    orig_n = upper_triangle(neural_dist_norm)
    pred_n = upper_triangle(neural_predicted)
    limit_n = float(max(orig_n.max(), pred_n.max()) * 1.04)
    ax.scatter(orig_n, pred_n, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit_n], [0, limit_n], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit_n); ax.set_ylim(0, limit_n)
    ax.set_aspect("equal")
    ax.set_xlabel("Input correlation distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"Neural HMDS Shepard\n"
        f"ρ={neural_metrics['distance_spearman']:.3f}  "
        f"r={neural_metrics['distance_pearson']:.3f}  "
        f"stress={neural_metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )
    ax.grid(True, color="#e5e7eb", lw=0.4)
    fig.savefig(OUTPUT_DIR / "neural_hmds_3d_frequent_patterns_shepard.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → neural shepard diagram")

    print(f"\nDone → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
