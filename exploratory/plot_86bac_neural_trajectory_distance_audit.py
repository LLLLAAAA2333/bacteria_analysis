from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


DEFAULT_INPUT_DIR = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1" / "tables"
DEFAULT_OUTPUT = (
    Path("results")
    / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
    / "figures"
    / "neural_shape_distance_audit_preview.png"
)


def parse_feature_columns(columns: list[str]) -> tuple[list[str], list[int]]:
    feature_columns = [column for column in columns if "__t" in column]
    neurons = list(dict.fromkeys(column.split("__", 1)[0] for column in feature_columns))
    timepoints = sorted({int(column.rsplit("__t", 1)[1]) for column in feature_columns})
    return neurons, timepoints


def neuron_cluster_order(
    prototypes: pd.DataFrame,
    *,
    neurons: list[str],
    timepoints: list[int],
) -> list[str]:
    profiles = []
    for neuron in neurons:
        columns = [f"{neuron}__t{timepoint:02d}" for timepoint in timepoints]
        profiles.append(prototypes.loc[:, columns].to_numpy(float).ravel())

    distances = np.zeros((len(neurons), len(neurons)), dtype=float)
    for left_index in range(len(neurons)):
        for right_index in range(left_index + 1, len(neurons)):
            left = profiles[left_index]
            right = profiles[right_index]
            valid = np.isfinite(left) & np.isfinite(right)
            if valid.sum() < 2 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
                distance = 1.0
            else:
                distance = float(np.clip(1.0 - np.corrcoef(left[valid], right[valid])[0, 1], 0.0, 2.0))
            distances[left_index, right_index] = distance
            distances[right_index, left_index] = distance

    order = leaves_list(linkage(squareform(distances, checks=False), method="average"))
    return [neurons[index] for index in order]


def pair_table(rdm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = rdm.index.astype(str).tolist()
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            distance = pd.to_numeric(rdm.loc[left, right], errors="coerce")
            if np.isfinite(distance):
                rows.append(
                    {
                        "sample_left": left,
                        "sample_right": right,
                        "distance": float(distance),
                        "correlation": float(1.0 - distance),
                    }
                )
    return pd.DataFrame(rows).sort_values("distance", kind="stable").reset_index(drop=True)


def trajectory_matrix(
    prototypes: pd.DataFrame,
    *,
    sample_id: str,
    neurons: list[str],
    timepoints: list[int],
) -> np.ndarray:
    row = prototypes.loc[prototypes["sample_id"].astype(str).eq(sample_id)]
    if len(row) != 1:
        raise ValueError(f"expected exactly one prototype for {sample_id!r}; found {len(row)}")
    return np.asarray(
        [
            [float(row.iloc[0][f"{neuron}__t{timepoint:02d}"]) for timepoint in timepoints]
            for neuron in neurons
        ],
        dtype=float,
    )


def plot_audit_preview(
    prototypes: pd.DataFrame,
    rdm: pd.DataFrame,
    *,
    output_path: Path,
    pairs_per_group: int,
) -> pd.DataFrame:
    neurons, timepoints = parse_feature_columns(prototypes.columns.astype(str).tolist())
    neuron_order = neuron_cluster_order(prototypes, neurons=neurons, timepoints=timepoints)
    all_pairs = pair_table(rdm)
    selected = pd.concat(
        [
            all_pairs.head(pairs_per_group).assign(group="closest"),
            all_pairs.tail(pairs_per_group).sort_values("distance", ascending=False).assign(group="furthest"),
        ],
        ignore_index=True,
    )

    feature_columns = [f"{neuron}__t{timepoint:02d}" for neuron in neurons for timepoint in timepoints]
    values = prototypes.loc[:, feature_columns].to_numpy(float)
    finite = values[np.isfinite(values)]
    vmax = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    figure, axes = plt.subplots(
        len(selected),
        2,
        figsize=(12.4, 2.15 * len(selected) + 1.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    last_image = None
    tick_positions = [index for index, timepoint in enumerate(timepoints) if timepoint in {10, 15, 20, 25, 29}]
    tick_labels = [str(timepoints[index]) for index in tick_positions]
    post_start = timepoints.index(16) - 0.5 if 16 in timepoints else None

    for row_index, pair in selected.iterrows():
        group = str(pair["group"])
        distance = float(pair["distance"])
        correlation = float(pair["correlation"])
        for column_index, sample_column in enumerate(["sample_left", "sample_right"]):
            sample_id = str(pair[sample_column])
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
            if post_start is not None:
                axis.axvline(post_start, color="#303030", linewidth=0.8, linestyle="--")
            axis.set_title(sample_id, fontsize=9, pad=4)
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(tick_labels, fontsize=8)
            axis.set_yticks(range(len(neuron_order)))
            axis.set_yticklabels(neuron_order, fontsize=8)
            axis.tick_params(length=0)

        axes[row_index, 0].set_ylabel(
            f"{group} {row_index % pairs_per_group + 1}\n"
            f"d={distance:.3f}; r={correlation:.3f}\n\nNeuron",
            fontsize=8,
        )

    axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=9)
    figure.suptitle(
        "86bac neural-shape correlation distance audit\n"
        "global neuron order; dashed line = start of post-stimulus window",
        fontsize=12,
    )
    if last_image is not None:
        colorbar = figure.colorbar(last_image, ax=axes, fraction=0.018, pad=0.012)
        colorbar.set_label("active-scaled response", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)

    selected = selected.copy()
    selected["neuron_order"] = ", ".join(neuron_order)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a preview audit of closest and furthest neural-shape pairs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs-per-group", type=int, default=4)
    args = parser.parse_args()

    prototypes = pd.read_csv(args.input_dir / "active_scaled_flattened_neural_prototypes.csv")
    rdm = pd.read_csv(
        args.input_dir / "neural_shape_rdm__active_scaled_flattened_correlation.csv",
        index_col=0,
    )
    selected = plot_audit_preview(
        prototypes,
        rdm,
        output_path=args.output,
        pairs_per_group=args.pairs_per_group,
    )
    selected.to_csv(args.output.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
