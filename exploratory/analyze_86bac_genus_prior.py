from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
NEURAL_RDM = "neural_shape_rdm__active_scaled_flattened_correlation.csv"
CHEMICAL_RDM = "chemical_rdm__qc20_missing50_log2_zscore_pca10_euclidean.csv"
TAXONOMY = "taxonomy_from_GM300/86bac_sample_species_mapping.csv"
MDS_COORDINATES = "rdm_mds_3d_coordinates.csv"
SEED = 20260603


def aid_sort_key(aid: str) -> tuple[str, int, str]:
    text = str(aid)
    prefix = "".join(char for char in text if not char.isdigit())
    digits = "".join(char for char in text if char.isdigit())
    numeric = int(digits) if digits else -1
    return prefix, numeric, text


def color_to_hex(color: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(round(color[0] * 255)),
        int(round(color[1] * 255)),
        int(round(color[2] * 255)),
    )


def read_rdm(path: Path) -> pd.DataFrame:
    rdm = pd.read_csv(path, index_col=0)
    rdm.index = rdm.index.astype(str)
    rdm.columns = rdm.columns.astype(str)
    if not rdm.index.equals(rdm.columns):
        raise ValueError(f"RDM index/columns do not match: {path}")
    values = rdm.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"RDM contains NaN values: {path}")
    return values


def align_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = root / "tables"
    neural = read_rdm(tables_dir / NEURAL_RDM)
    chemical = read_rdm(tables_dir / CHEMICAL_RDM)
    taxonomy = pd.read_csv(tables_dir / TAXONOMY)
    taxonomy["AID"] = taxonomy["AID"].astype(str)
    taxonomy = taxonomy.set_index("AID", drop=False)
    shared = [sample for sample in neural.index if sample in chemical.index and sample in taxonomy.index]
    if len(shared) < 4:
        raise ValueError(f"need at least 4 shared samples; found {len(shared)}")
    return neural.loc[shared, shared], chemical.loc[shared, shared], taxonomy.loc[shared].copy()


def upper_pairs(labels: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    left_idx, right_idx = np.triu_indices(len(labels), k=1)
    pairs = [(labels[i], labels[j]) for i, j in zip(left_idx, right_idx)]
    return left_idx, right_idx, pairs


def binary_spearman(binary: np.ndarray, distances: np.ndarray) -> float:
    return float(np.corrcoef(binary.astype(float), rankdata(distances))[0, 1])


def build_pair_table(neural: pd.DataFrame, chemical: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    samples = neural.index.astype(str).tolist()
    left_idx, right_idx, pairs = upper_pairs(samples)
    genus = taxonomy["genus"].astype(str).to_numpy()
    species = taxonomy["species"].astype(str).to_numpy()
    neural_values = neural.to_numpy(float)
    chemical_values = chemical.to_numpy(float)
    rows = []
    for pair_index, (left_sample, right_sample) in enumerate(pairs):
        i = left_idx[pair_index]
        j = right_idx[pair_index]
        rows.append(
            {
                "sample_left": left_sample,
                "sample_right": right_sample,
                "genus_left": genus[i],
                "genus_right": genus[j],
                "species_left": species[i],
                "species_right": species[j],
                "same_genus": bool(genus[i] == genus[j]),
                "different_genus": bool(genus[i] != genus[j]),
                "neural_distance": float(neural_values[i, j]),
                "chemical_distance": float(chemical_values[i, j]),
            }
        )
    return pd.DataFrame(rows)


def permutation_rsa(
    genus: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    distances: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[float, np.ndarray, float, float]:
    observed_model = (genus[left_idx] != genus[right_idx]).astype(float)
    observed = binary_spearman(observed_model, distances)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    ranked_distances = rankdata(distances)
    for permutation_index in range(n_permutations):
        shuffled = rng.permutation(genus)
        model = (shuffled[left_idx] != shuffled[right_idx]).astype(float)
        null[permutation_index] = float(np.corrcoef(model, ranked_distances)[0, 1])
    p_greater = float((np.count_nonzero(null >= observed) + 1) / (n_permutations + 1))
    p_abs = float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1))
    return observed, null, p_greater, p_abs


def distance_group_summary(pair_table: pd.DataFrame, distance_column: str) -> dict[str, float]:
    same = pair_table.loc[pair_table["same_genus"], distance_column].to_numpy(float)
    different = pair_table.loc[~pair_table["same_genus"], distance_column].to_numpy(float)
    test = mannwhitneyu(same, different, alternative="less")
    return {
        "same_genus_pairs": int(same.size),
        "different_genus_pairs": int(different.size),
        "same_mean": float(np.mean(same)),
        "different_mean": float(np.mean(different)),
        "different_minus_same_mean": float(np.mean(different) - np.mean(same)),
        "same_median": float(np.median(same)),
        "different_median": float(np.median(different)),
        "different_minus_same_median": float(np.median(different) - np.median(same)),
        "mannwhitney_u_less_p_uncorrected": float(test.pvalue),
    }


def genus_within_between(pair_table: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for genus, group in taxonomy.groupby("genus", sort=True):
        sample_ids = set(group["AID"].astype(str))
        if len(sample_ids) < 2:
            continue
        within = pair_table[
            pair_table["sample_left"].isin(sample_ids)
            & pair_table["sample_right"].isin(sample_ids)
            & pair_table["same_genus"]
        ]
        related_between = pair_table[
            (pair_table["sample_left"].isin(sample_ids) | pair_table["sample_right"].isin(sample_ids))
            & ~pair_table["same_genus"]
        ]
        rows.append(
            {
                "genus": genus,
                "n_strains": len(sample_ids),
                "within_pairs": len(within),
                "between_pairs_touching_genus": len(related_between),
                "neural_within_mean": float(within["neural_distance"].mean()),
                "neural_between_touching_mean": float(related_between["neural_distance"].mean()),
                "neural_between_minus_within": float(
                    related_between["neural_distance"].mean() - within["neural_distance"].mean()
                ),
                "chemical_within_mean": float(within["chemical_distance"].mean()),
                "chemical_between_touching_mean": float(related_between["chemical_distance"].mean()),
                "chemical_between_minus_within": float(
                    related_between["chemical_distance"].mean() - within["chemical_distance"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_strains", "genus"], ascending=[False, True])


def build_genus_model_rdm(taxonomy: pd.DataFrame) -> pd.DataFrame:
    labels = taxonomy.index.astype(str).tolist()
    genus = taxonomy["genus"].astype(str).to_numpy()
    values = (genus[:, None] != genus[None, :]).astype(int)
    return pd.DataFrame(values, index=labels, columns=labels)


def plot_distance_distributions(pair_table: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, distance_column, title in zip(
        axes,
        ["neural_distance", "chemical_distance"],
        ["Neural RDM", "Chemical RDM"],
    ):
        same = pair_table.loc[pair_table["same_genus"], distance_column].to_numpy(float)
        different = pair_table.loc[~pair_table["same_genus"], distance_column].to_numpy(float)
        parts = axis.violinplot([same, different], positions=[0, 1], showmeans=False, showextrema=False)
        for body, color in zip(parts["bodies"], ["#2b6cb0", "#dd6b20"]):
            body.set_facecolor(color)
            body.set_alpha(0.45)
            body.set_edgecolor("none")
        axis.boxplot(
            [same, different],
            positions=[0, 1],
            widths=0.18,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            boxprops={"color": "#333333"},
            whiskerprops={"color": "#333333"},
            capprops={"color": "#333333"},
        )
        row = summary.loc[summary["rdm"].eq(distance_column.replace("_distance", ""))].iloc[0]
        axis.set_title(
            f"{title}\n"
            f"RSA rho={row['genus_rdm_spearman']:.3f}, "
            f"p={row['label_shuffle_p_greater']:.4f}",
            fontsize=10,
        )
        axis.set_xticks([0, 1])
        axis.set_xticklabels([f"same genus\nn={same.size}", f"different genus\nn={different.size}"], fontsize=9)
        axis.set_ylabel("RDM distance", fontsize=9)
        axis.tick_params(axis="y", labelsize=8)
    figure.suptitle("Genus prior: same-genus pairs should be closer if taxonomy predicts geometry", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_mds_by_genus(root: Path, taxonomy: pd.DataFrame, output_path: Path) -> None:
    coordinates_path = root / "figures" / MDS_COORDINATES
    if not coordinates_path.exists():
        return
    coordinates = pd.read_csv(coordinates_path)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    merged = coordinates.merge(
        taxonomy[["AID", "genus"]].rename(columns={"AID": "sample_id"}),
        on="sample_id",
        how="inner",
    )
    top_genera = taxonomy["genus"].value_counts().head(8).index.tolist()
    merged["plot_genus"] = np.where(merged["genus"].isin(top_genera), merged["genus"], "Other")
    labels = top_genera + ["Other"]
    color_map = dict(zip(labels, plt.cm.tab10(np.linspace(0, 1, len(labels)))))

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for axis, prefix, title in zip(axes, ["neural", "chemical"], ["Neural MDS", "Chemical MDS"]):
        for label in labels:
            group = merged[merged["plot_genus"].eq(label)]
            if group.empty:
                continue
            axis.scatter(
                group[f"{prefix}_mds1"],
                group[f"{prefix}_mds2"],
                s=34,
                alpha=0.82,
                label=f"{label} (n={len(group)})",
                color=color_map[label],
                edgecolor="white",
                linewidth=0.4,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("MDS1", fontsize=9)
        axis.set_ylabel("MDS2", fontsize=9)
        axis.tick_params(labelsize=8)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=8, frameon=False)
    figure.suptitle("MDS overview colored by major genus labels", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def genus_color_table(taxonomy: pd.DataFrame) -> pd.DataFrame:
    genus_counts = taxonomy["genus"].value_counts()
    aid_ordered = taxonomy.copy()
    aid_ordered["_aid_sort_key"] = aid_ordered["AID"].map(aid_sort_key)
    aid_ordered = aid_ordered.sort_values("_aid_sort_key")
    labels = list(dict.fromkeys(aid_ordered["genus"].astype(str)))
    first_aid = aid_ordered.groupby("genus", sort=False)["AID"].first()
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, len(labels)))
    return pd.DataFrame(
        {
            "genus": labels,
            "genus_index": np.arange(len(labels)),
            "first_AID_in_color_order": first_aid.loc[labels].to_numpy(str),
            "n_strains": genus_counts.loc[labels].to_numpy(int),
            "color_hex": [color_to_hex(color) for color in colors],
        }
    )


def closest_major_genus(
    pair_table: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    major_min_strains: int = 4,
) -> pd.DataFrame:
    counts = taxonomy["genus"].value_counts()
    major = counts[counts >= major_min_strains].index.tolist()
    minor = counts[counts < major_min_strains].index.tolist()
    rows = []
    for minor_genus in minor:
        for rdm, distance_column in [("neural", "neural_distance"), ("chemical", "chemical_distance")]:
            candidates = []
            for major_genus in major:
                cross = pair_table[
                    (
                        pair_table["genus_left"].eq(minor_genus)
                        & pair_table["genus_right"].eq(major_genus)
                    )
                    | (
                        pair_table["genus_left"].eq(major_genus)
                        & pair_table["genus_right"].eq(minor_genus)
                    )
                ]
                if cross.empty:
                    continue
                candidates.append(
                    {
                        "genus": minor_genus,
                        "n_strains": int(counts.loc[minor_genus]),
                        "rdm": rdm,
                        "closest_major_genus": major_genus,
                        "mean_distance_to_major": float(cross[distance_column].mean()),
                        "median_distance_to_major": float(cross[distance_column].median()),
                        "n_cross_pairs": int(len(cross)),
                    }
                )
            if candidates:
                rows.append(
                    min(
                        candidates,
                        key=lambda row: (row["mean_distance_to_major"], row["median_distance_to_major"]),
                    )
                )
    return pd.DataFrame(rows).sort_values(["genus", "rdm"])


def plot_mds_3d_all_genera(root: Path, taxonomy: pd.DataFrame, output_path: Path) -> None:
    coordinates_path = root / "figures" / MDS_COORDINATES
    if not coordinates_path.exists():
        return
    coordinates = pd.read_csv(coordinates_path)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    color_table = genus_color_table(taxonomy)
    merged = coordinates.merge(
        taxonomy[["AID", "genus", "species"]].rename(columns={"AID": "sample_id"}),
        on="sample_id",
        how="inner",
    ).merge(color_table[["genus", "genus_index"]], on="genus", how="left")

    figure = plt.figure(figsize=(13.4, 6.0), constrained_layout=True)
    axes = [
        figure.add_subplot(1, 2, 1, projection="3d"),
        figure.add_subplot(1, 2, 2, projection="3d"),
    ]
    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=-0.5, vmax=len(color_table) - 0.5)
    last_scatter = None
    for axis, prefix, title in zip(axes, ["neural", "chemical"], ["Neural 3D MDS", "Chemical 3D MDS"]):
        last_scatter = axis.scatter(
            merged[f"{prefix}_mds1"],
            merged[f"{prefix}_mds2"],
            merged[f"{prefix}_mds3"],
            c=merged["genus_index"],
            cmap=cmap,
            norm=norm,
            s=42,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.45,
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("MDS1", fontsize=9)
        axis.set_ylabel("MDS2", fontsize=9)
        axis.set_zlabel("MDS3", fontsize=9)
        axis.tick_params(labelsize=8)

    if last_scatter is not None:
        colorbar = figure.colorbar(last_scatter, ax=axes, fraction=0.035, pad=0.02)
        colorbar.set_ticks(color_table["genus_index"])
        colorbar.set_ticklabels(
            [f"{row.genus} ({row.n_strains})" for row in color_table.itertuples(index=False)]
        )
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label("Genus", fontsize=9)
    figure.suptitle("86bac 3D MDS colored by all genus labels", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_mds_3d_all_genera_html(root: Path, taxonomy: pd.DataFrame, output_path: Path) -> None:
    coordinates_path = root / "figures" / MDS_COORDINATES
    if not coordinates_path.exists():
        return
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    coordinates = pd.read_csv(coordinates_path)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    color_table = genus_color_table(taxonomy)
    merged = coordinates.merge(
        taxonomy[["AID", "genus", "species"]].rename(columns={"AID": "sample_id"}),
        on="sample_id",
        how="inner",
    ).merge(color_table[["genus", "color_hex", "n_strains"]], on="genus", how="left")

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Neural 3D MDS", "Chemical 3D MDS"),
    )
    for col, prefix in [(1, "neural"), (2, "chemical")]:
        for genus in color_table["genus"]:
            group = merged[merged["genus"].eq(genus)]
            if group.empty:
                continue
            label = f"{genus} (n={int(group['n_strains'].iloc[0])})"
            figure.add_trace(
                go.Scatter3d(
                    x=group[f"{prefix}_mds1"],
                    y=group[f"{prefix}_mds2"],
                    z=group[f"{prefix}_mds3"],
                    mode="markers",
                    name=label,
                    legendgroup=genus,
                    showlegend=(col == 1),
                    marker={
                        "size": 5,
                        "color": group["color_hex"].iloc[0],
                        "line": {"width": 0.5, "color": "white"},
                    },
                    text=group["sample_id"],
                    customdata=np.column_stack([group["genus"], group["species"]]),
                    hovertemplate=(
                        "%{text}<br>"
                        "genus=%{customdata[0]}<br>"
                        "species=%{customdata[1]}<br>"
                        "MDS1=%{x:.3f}<br>MDS2=%{y:.3f}<br>MDS3=%{z:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    figure.update_layout(
        title="86bac 3D MDS colored by all genus labels",
        width=1320,
        height=680,
        margin={"l": 10, "r": 10, "t": 65, "b": 10},
        legend={"font": {"size": 10}},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def plot_neural_mds_3d_aid_order_genera(root: Path, taxonomy: pd.DataFrame, output_path: Path) -> None:
    coordinates_path = root / "figures" / MDS_COORDINATES
    if not coordinates_path.exists():
        return
    coordinates = pd.read_csv(coordinates_path)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    color_table = genus_color_table(taxonomy)
    merged = coordinates.merge(
        taxonomy[["AID", "genus", "species"]].rename(columns={"AID": "sample_id"}),
        on="sample_id",
        how="inner",
    ).merge(color_table[["genus", "genus_index"]], on="genus", how="left")
    merged["_aid_sort_key"] = merged["sample_id"].map(aid_sort_key)
    merged = merged.sort_values("_aid_sort_key")

    figure = plt.figure(figsize=(8.8, 6.8), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    norm = plt.Normalize(vmin=-0.5, vmax=len(color_table) - 0.5)
    scatter = axis.scatter(
        merged["neural_mds1"],
        merged["neural_mds2"],
        merged["neural_mds3"],
        c=merged["genus_index"],
        cmap=plt.cm.turbo,
        norm=norm,
        s=48,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.45,
    )
    axis.set_title("Neural 3D MDS", fontsize=12)
    axis.set_xlabel("MDS1", fontsize=9)
    axis.set_ylabel("MDS2", fontsize=9)
    axis.set_zlabel("MDS3", fontsize=9)
    axis.tick_params(labelsize=8)

    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_ticks(color_table["genus_index"])
    colorbar.set_ticklabels(
        [
            f"{row.genus} ({row.n_strains}, first {row.first_AID_in_color_order})"
            for row in color_table.itertuples(index=False)
        ]
    )
    colorbar.ax.invert_yaxis()
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("Genus color order by first AID", fontsize=9)
    figure.suptitle("86bac neural MDS colored by genus", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_neural_mds_3d_aid_order_genera_html(root: Path, taxonomy: pd.DataFrame, output_path: Path) -> None:
    coordinates_path = root / "figures" / MDS_COORDINATES
    if not coordinates_path.exists():
        return
    import plotly.graph_objects as go

    coordinates = pd.read_csv(coordinates_path)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    color_table = genus_color_table(taxonomy)
    merged = coordinates.merge(
        taxonomy[["AID", "genus", "species"]].rename(columns={"AID": "sample_id"}),
        on="sample_id",
        how="inner",
    ).merge(color_table[["genus", "color_hex", "n_strains", "first_AID_in_color_order"]], on="genus", how="left")
    merged["_aid_sort_key"] = merged["sample_id"].map(aid_sort_key)
    merged = merged.sort_values("_aid_sort_key")

    figure = go.Figure()
    for genus in color_table["genus"]:
        group = merged[merged["genus"].eq(genus)]
        if group.empty:
            continue
        label = f"{genus} (n={int(group['n_strains'].iloc[0])})"
        figure.add_trace(
            go.Scatter3d(
                x=group["neural_mds1"],
                y=group["neural_mds2"],
                z=group["neural_mds3"],
                mode="markers",
                name=label,
                marker={
                    "size": 5,
                    "color": group["color_hex"].iloc[0],
                    "line": {"width": 0.5, "color": "white"},
                },
                text=group["sample_id"],
                customdata=np.column_stack(
                    [group["genus"], group["species"], group["first_AID_in_color_order"]]
                ),
                hovertemplate=(
                    "%{text}<br>"
                    "genus=%{customdata[0]}<br>"
                    "species=%{customdata[1]}<br>"
                    "genus color starts at %{customdata[2]}<br>"
                    "MDS1=%{x:.3f}<br>MDS2=%{y:.3f}<br>MDS3=%{z:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="86bac neural 3D MDS colored by genus",
        width=920,
        height=720,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        legend={"font": {"size": 10}},
        scene={
            "xaxis_title": "MDS1",
            "yaxis_title": "MDS2",
            "zaxis_title": "MDS3",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def run_analysis(root: Path, n_permutations: int, seed: int) -> dict[str, object]:
    neural, chemical, taxonomy = align_inputs(root)
    samples = neural.index.astype(str).tolist()
    left_idx, right_idx, _ = upper_pairs(samples)
    genus = taxonomy["genus"].astype(str).to_numpy()

    pair_table = build_pair_table(neural, chemical, taxonomy)
    outputs_dir = root / "tables" / "taxonomy_from_GM300" / "genus_prior"
    figures_dir = root / "figures"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    neural_distances = pair_table["neural_distance"].to_numpy(float)
    chemical_distances = pair_table["chemical_distance"].to_numpy(float)
    summary_rows = []
    null_frames = []
    for name, distances, perm_seed in [
        ("neural", neural_distances, seed),
        ("chemical", chemical_distances, seed + 1),
    ]:
        observed, null, p_greater, p_abs = permutation_rsa(
            genus,
            left_idx,
            right_idx,
            distances,
            n_permutations=n_permutations,
            seed=perm_seed,
        )
        group_summary = distance_group_summary(pair_table, f"{name}_distance")
        summary_rows.append(
            {
                "rdm": name,
                "genus_rdm_spearman": observed,
                "label_shuffle_permutations": n_permutations,
                "label_shuffle_p_greater": p_greater,
                "label_shuffle_p_abs": p_abs,
                "null_mean": float(np.mean(null)),
                "null_std": float(np.std(null, ddof=1)),
                "null_q95": float(np.quantile(null, 0.95)),
                "null_q99": float(np.quantile(null, 0.99)),
                **group_summary,
            }
        )
        null_frames.append(pd.DataFrame({"rdm": name, "permutation": np.arange(n_permutations), "spearman": null}))

    summary = pd.DataFrame(summary_rows)
    within_between = genus_within_between(pair_table, taxonomy)
    closest_major = closest_major_genus(pair_table, taxonomy, major_min_strains=4)
    color_table = genus_color_table(taxonomy)
    genus_counts = (
        taxonomy.groupby("genus")
        .agg(n_strains=("AID", "count"), n_species=("species", "nunique"), AIDs=("AID", lambda x: ", ".join(x)))
        .reset_index()
        .sort_values(["n_strains", "genus"], ascending=[False, True])
    )
    model_rdm = build_genus_model_rdm(taxonomy)

    pair_table.to_csv(outputs_dir / "genus_prior_pair_distances.csv", index=False)
    summary.to_csv(outputs_dir / "genus_prior_rsa_summary.csv", index=False)
    pd.concat(null_frames, ignore_index=True).to_csv(outputs_dir / "genus_prior_label_shuffle_null.csv", index=False)
    within_between.to_csv(outputs_dir / "genus_within_between_summary.csv", index=False)
    closest_major.to_csv(outputs_dir / "minor_genus_closest_major_genus.csv", index=False)
    color_table.to_csv(outputs_dir / "genus_color_map.csv", index=False)
    genus_counts.to_csv(outputs_dir / "genus_counts.csv", index=False)
    model_rdm.to_csv(outputs_dir / "genus_model_rdm__different_genus.csv")

    plot_distance_distributions(pair_table, summary, figures_dir / "genus_prior_distance_distributions.png")
    plot_mds_by_genus(root, taxonomy, figures_dir / "genus_prior_mds_by_genus.png")
    plot_neural_mds_3d_aid_order_genera(root, taxonomy, figures_dir / "genus_prior_neural_mds_3d_aid_order_genera.png")
    plot_neural_mds_3d_aid_order_genera_html(
        root,
        taxonomy,
        figures_dir / "genus_prior_neural_mds_3d_aid_order_genera.html",
    )

    run_summary: dict[str, object] = {
        "input": {
            "root": str(root),
            "taxonomy": str(root / "tables" / TAXONOMY),
            "neural_rdm": str(root / "tables" / NEURAL_RDM),
            "chemical_rdm": str(root / "tables" / CHEMICAL_RDM),
        },
        "n_samples": len(samples),
        "n_pairs": int(len(pair_table)),
        "n_genera": int(taxonomy["genus"].nunique()),
        "n_species": int(taxonomy["species"].nunique()),
        "same_genus_pairs": int(pair_table["same_genus"].sum()),
        "different_genus_pairs": int((~pair_table["same_genus"]).sum()),
        "seed": seed,
        "n_permutations": n_permutations,
        "summary": summary.to_dict(orient="records"),
        "outputs": {
            "tables_dir": str(outputs_dir),
            "figures": [
                str(figures_dir / "genus_prior_distance_distributions.png"),
                str(figures_dir / "genus_prior_mds_by_genus.png"),
                str(figures_dir / "genus_prior_neural_mds_3d_aid_order_genera.png"),
                str(figures_dir / "genus_prior_neural_mds_3d_aid_order_genera.html"),
            ],
        },
    }
    (outputs_dir / "genus_prior_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze genus-level taxonomy prior against neural and chemical RDMs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    summary = run_analysis(args.root, n_permutations=args.permutations, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
