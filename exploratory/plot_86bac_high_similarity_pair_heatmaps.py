from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from plot_86bac_neural_trajectory_distance_audit import (
    neuron_cluster_order,
    pair_table,
    parse_feature_columns,
    trajectory_matrix,
)


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
PROTOTYPES = "active_scaled_flattened_neural_prototypes.csv"
NEURAL_RDM = "neural_correlation_distance_matrix.csv"
TAXONOMY = "taxonomy_from_GM300/86bac_sample_species_mapping.csv"


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = root / "tables"
    prototypes = pd.read_csv(tables_dir / PROTOTYPES)
    rdm = pd.read_csv(tables_dir / NEURAL_RDM, index_col=0)
    rdm.index = rdm.index.astype(str)
    rdm.columns = rdm.columns.astype(str)
    taxonomy = pd.read_csv(tables_dir / TAXONOMY)
    taxonomy["AID"] = taxonomy["AID"].astype(str)
    return prototypes, rdm, taxonomy


def annotate_pairs(pairs: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    meta = taxonomy.set_index("AID")[["genus", "species"]]
    annotated = pairs.copy()
    for side in ["left", "right"]:
        annotated[f"genus_{side}"] = annotated[f"sample_{side}"].map(meta["genus"])
        annotated[f"species_{side}"] = annotated[f"sample_{side}"].map(meta["species"])
    annotated["same_genus"] = annotated["genus_left"].eq(annotated["genus_right"])
    annotated["same_species"] = annotated["species_left"].eq(annotated["species_right"])
    return annotated


def complete_linkage_groups(
    rdm: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    max_distance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = rdm.index.astype(str).tolist()
    distance = rdm.loc[labels, labels].to_numpy(float)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    clusters = fcluster(linkage(squareform(distance, checks=False), method="complete"), t=max_distance, criterion="distance")
    meta = taxonomy.set_index("AID")[["genus", "species"]]

    group_rows = []
    member_rows = []
    group_index = 0
    for cluster_id in sorted(set(clusters)):
        member_indices = np.where(clusters == cluster_id)[0]
        if len(member_indices) < 2:
            continue
        member_labels = [labels[index] for index in member_indices]
        sub_distance = distance[np.ix_(member_indices, member_indices)]
        pair_values = sub_distance[np.triu_indices(len(member_indices), k=1)]
        group_index += 1
        genera = [str(meta.loc[label, "genus"]) for label in member_labels]
        species = [str(meta.loc[label, "species"]) for label in member_labels]
        group_rows.append(
            {
                "group_id": group_index,
                "n_members": int(len(member_labels)),
                "min_pair_distance": float(pair_values.min()),
                "mean_pair_distance": float(pair_values.mean()),
                "max_pair_distance": float(pair_values.max()),
                "min_pair_correlation": float(1.0 - pair_values.max()),
                "members": ", ".join(member_labels),
                "genera": ", ".join(dict.fromkeys(genera)),
                "species": "; ".join(dict.fromkeys(species)),
                "n_genera": int(len(set(genera))),
                "n_species": int(len(set(species))),
            }
        )
        for member_order, sample_id in enumerate(member_labels):
            member_rows.append(
                {
                    "group_id": group_index,
                    "member_order": member_order + 1,
                    "sample_id": sample_id,
                    "genus": str(meta.loc[sample_id, "genus"]),
                    "species": str(meta.loc[sample_id, "species"]),
                }
            )

    groups = pd.DataFrame(group_rows)
    members = pd.DataFrame(member_rows)
    if not groups.empty:
        groups = groups.sort_values(
            ["n_members", "max_pair_distance", "mean_pair_distance"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        group_id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(groups["group_id"])}
        groups["group_id"] = groups["group_id"].map(group_id_map)
        members["group_id"] = members["group_id"].map(group_id_map)
        members = members[members["group_id"].notna()].copy()
        members["group_id"] = members["group_id"].astype(int)
        members = members.sort_values(["group_id", "member_order"], kind="stable").reset_index(drop=True)
    return groups, members


def select_high_similarity_pairs(
    rdm: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    max_distance: float,
    max_pairs: int,
) -> pd.DataFrame:
    pairs = pair_table(rdm)
    pairs = pairs[pairs["distance"] <= max_distance].copy()
    if max_pairs > 0:
        pairs = pairs.head(max_pairs).copy()
    return annotate_pairs(pairs, taxonomy)


def plot_pair_heatmaps(
    prototypes: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    output_path: Path,
) -> dict[str, object]:
    neurons, timepoints = parse_feature_columns(prototypes.columns.astype(str).tolist())
    neuron_order = neuron_cluster_order(prototypes, neurons=neurons, timepoints=timepoints)
    feature_columns = [f"{neuron}__t{timepoint:02d}" for neuron in neurons for timepoint in timepoints]
    values = prototypes.loc[:, feature_columns].to_numpy(float)
    finite = values[np.isfinite(values)]
    vmax = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    n_pairs = len(selected)
    if n_pairs == 0:
        raise ValueError("No high-similarity pairs selected")
    figure, axes = plt.subplots(
        n_pairs,
        2,
        figsize=(12.4, 2.0 * n_pairs + 1.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    tick_positions = [index for index, timepoint in enumerate(timepoints) if timepoint in {5, 10, 15, 20, 24}]
    tick_labels = [str(timepoints[index]) for index in tick_positions]
    stimulus_off = timepoints.index(15) + 0.5 if 15 in timepoints else None
    last_image = None

    for row_index, pair in selected.reset_index(drop=True).iterrows():
        distance = float(pair["distance"])
        correlation = float(pair["correlation"])
        row_label = (
            f"pair {row_index + 1}\n"
            f"d={distance:.3f}; r={correlation:.3f}\n"
            f"{'same genus' if pair['same_genus'] else 'diff genus'}"
        )
        for column_index, sample_column in enumerate(["sample_left", "sample_right"]):
            sample_id = str(pair[sample_column])
            side = "left" if sample_column.endswith("left") else "right"
            matrix = trajectory_matrix(
                prototypes,
                sample_id=sample_id,
                neurons=neuron_order,
                timepoints=timepoints,
            )
            axis = axes[row_index, column_index]
            last_image = axis.imshow(
                matrix,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
                interpolation="nearest",
            )
            if stimulus_off is not None:
                axis.axvline(stimulus_off, color="#303030", linewidth=0.8, linestyle="--")
            axis.set_title(
                f"{sample_id} | {pair[f'genus_{side}']}\n{pair[f'species_{side}']}",
                fontsize=8,
                pad=4,
            )
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(tick_labels, fontsize=8)
            axis.set_yticks(range(len(neuron_order)))
            axis.set_yticklabels(neuron_order, fontsize=7)
            axis.tick_params(length=0)
        axes[row_index, 0].set_ylabel(f"{row_label}\n\nNeuron", fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=9)
    figure.suptitle(
        "High-similarity neural trajectory pairs\n"
        "13 x 20 heatmaps; dashed line = stimulus offset",
        fontsize=12,
    )
    if last_image is not None:
        colorbar = figure.colorbar(last_image, ax=axes, fraction=0.018, pad=0.012)
        colorbar.set_label("active-scaled response", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return {
        "n_pairs_plotted": int(n_pairs),
        "neuron_order": neuron_order,
        "timepoints": timepoints,
        "vmax_abs_quantile_0.98": vmax,
    }


def plot_group_heatmaps(
    prototypes: pd.DataFrame,
    groups: pd.DataFrame,
    members: pd.DataFrame,
    *,
    output_path: Path,
    max_groups: int,
) -> dict[str, object]:
    neurons, timepoints = parse_feature_columns(prototypes.columns.astype(str).tolist())
    neuron_order = neuron_cluster_order(prototypes, neurons=neurons, timepoints=timepoints)
    feature_columns = [f"{neuron}__t{timepoint:02d}" for neuron in neurons for timepoint in timepoints]
    values = prototypes.loc[:, feature_columns].to_numpy(float)
    finite = values[np.isfinite(values)]
    vmax = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    displayed_groups = groups.head(max_groups).copy() if max_groups > 0 else groups.copy()
    displayed_members = members[members["group_id"].isin(displayed_groups["group_id"])].copy()
    n_groups = len(displayed_groups)
    max_members = int(displayed_members.groupby("group_id")["sample_id"].count().max())
    figure, axes = plt.subplots(
        n_groups,
        max_members,
        figsize=(2.35 * max_members + 1.6, 1.55 * n_groups + 1.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(n_groups, max_members)

    tick_positions = [index for index, timepoint in enumerate(timepoints) if timepoint in {5, 10, 15, 20, 24}]
    tick_labels = [str(timepoints[index]) for index in tick_positions]
    stimulus_off = timepoints.index(15) + 0.5 if 15 in timepoints else None
    last_image = None

    for row_index, group in enumerate(displayed_groups.itertuples(index=False)):
        group_members = displayed_members[displayed_members["group_id"].eq(group.group_id)].reset_index(drop=True)
        for column_index in range(max_members):
            axis = axes[row_index, column_index]
            if column_index >= len(group_members):
                axis.axis("off")
                continue
            member = group_members.iloc[column_index]
            matrix = trajectory_matrix(
                prototypes,
                sample_id=str(member["sample_id"]),
                neurons=neuron_order,
                timepoints=timepoints,
            )
            last_image = axis.imshow(
                matrix,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
                interpolation="nearest",
            )
            if stimulus_off is not None:
                axis.axvline(stimulus_off, color="#303030", linewidth=0.7, linestyle="--")
            axis.set_title(
                f"{member['sample_id']} | {member['genus']}",
                fontsize=7,
                pad=3,
            )
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(tick_labels, fontsize=7)
            if column_index == 0:
                axis.set_yticks(range(len(neuron_order)))
                axis.set_yticklabels(neuron_order, fontsize=6)
            else:
                axis.set_yticks([])
            axis.tick_params(length=0)
        axes[row_index, 0].set_ylabel(
            f"group {group.group_id}\n"
            f"n={group.n_members}; max d={group.max_pair_distance:.3f}\n"
            f"{group.n_genera} genera\n\nNeuron",
            fontsize=7,
        )

    for axis in axes[-1, :]:
        if axis.axison:
            axis.set_xlabel("Time (s)", fontsize=8)
    figure.suptitle(
        "Mutually high-similarity neural trajectory groups\n"
        "complete-linkage groups; all within-group pair distances <= threshold",
        fontsize=11,
    )
    if last_image is not None:
        colorbar = figure.colorbar(last_image, ax=axes, fraction=0.014, pad=0.010)
        colorbar.set_label("active-scaled response", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return {
        "n_groups_plotted": int(n_groups),
        "max_members_plotted": int(max_members),
        "neuron_order": neuron_order,
        "timepoints": timepoints,
        "vmax_abs_quantile_0.98": vmax,
    }


def run(root: Path, max_distance: float, max_pairs: int) -> dict[str, object]:
    prototypes, rdm, taxonomy = load_inputs(root)
    all_pairs = annotate_pairs(pair_table(rdm), taxonomy)
    selected = select_high_similarity_pairs(
        rdm,
        taxonomy,
        max_distance=max_distance,
        max_pairs=max_pairs,
    )
    groups, members = complete_linkage_groups(rdm, taxonomy, max_distance=max_distance)
    output_dir = root / "tables" / "high_similarity_pairs"
    figures_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    threshold_tag = str(max_distance).replace(".", "p")
    selected_path = output_dir / f"neural_high_similarity_pairs_d_le_{threshold_tag}.csv"
    groups_path = output_dir / f"neural_high_similarity_complete_groups_d_le_{threshold_tag}.csv"
    members_path = output_dir / f"neural_high_similarity_complete_group_members_d_le_{threshold_tag}.csv"
    all_summary_path = output_dir / "neural_high_similarity_pair_summary.json"
    figure_path = figures_dir / f"neural_high_similarity_pair_heatmaps_d_le_{threshold_tag}.png"
    group_figure_path = figures_dir / f"neural_high_similarity_complete_group_heatmaps_d_le_{threshold_tag}.png"
    selected.to_csv(selected_path, index=False)
    groups.to_csv(groups_path, index=False)
    members.to_csv(members_path, index=False)
    plot_summary = plot_pair_heatmaps(prototypes, selected, output_path=figure_path)
    group_plot_summary = plot_group_heatmaps(
        prototypes,
        groups,
        members,
        output_path=group_figure_path,
        max_groups=max_pairs,
    )

    summary = {
        "input": {
            "root": str(root),
            "prototypes": str(root / "tables" / PROTOTYPES),
            "neural_rdm": str(root / "tables" / NEURAL_RDM),
            "taxonomy": str(root / "tables" / TAXONOMY),
        },
        "threshold": {
            "max_distance": max_distance,
            "min_correlation": 1.0 - max_distance,
        },
        "all_pairs": int(len(all_pairs)),
        "n_pairs_at_or_below_threshold": int((all_pairs["distance"] <= max_distance).sum()),
        "fraction_pairs_at_or_below_threshold": float((all_pairs["distance"] <= max_distance).mean()),
        "n_pairs_plotted": int(len(selected)),
        "n_same_genus_plotted": int(selected["same_genus"].sum()),
        "n_same_species_plotted": int(selected["same_species"].sum()),
        "complete_linkage_groups": {
            "n_groups": int(len(groups)),
            "n_members_in_groups": int(members["sample_id"].nunique()) if not members.empty else 0,
            "max_group_size": int(groups["n_members"].max()) if not groups.empty else 0,
            "n_groups_plotted": int(group_plot_summary["n_groups_plotted"]),
        },
        "plot": plot_summary,
        "group_plot": group_plot_summary,
        "outputs": {
            "selected_pairs": str(selected_path),
            "complete_groups": str(groups_path),
            "complete_group_members": str(members_path),
            "figure": str(figure_path),
            "group_figure": str(group_figure_path),
        },
    }
    all_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 13 x 20 heatmaps for high-similarity neural RDM pairs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-distance", type=float, default=0.2)
    parser.add_argument("--max-pairs", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.max_distance, args.max_pairs), indent=2))


if __name__ == "__main__":
    main()
