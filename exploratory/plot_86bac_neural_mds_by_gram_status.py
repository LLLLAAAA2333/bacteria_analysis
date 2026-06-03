from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
TAXONOMY = "taxonomy_from_GM300/86bac_sample_species_mapping.csv"
MDS_COORDINATES = "rdm_mds_3d_coordinates.csv"


GRAM_STATUS_BY_GENUS: dict[str, tuple[str, str]] = {
    "Anaerostipes": (
        "Gram-positive",
        "A. caccae is reported Gram-variable; assigned to positive-side for binary plotting.",
    ),
    "Bacillus": ("Gram-positive", "Bacillus spp. are treated as Gram-positive spore-forming rods."),
    "Bacteroides": ("Gram-negative", "Bacteroides spp. are treated as Gram-negative anaerobic rods."),
    "Bifidobacterium": ("Gram-positive", "Bifidobacterium spp. are treated as Gram-positive actinobacteria."),
    "Catenibacterium": ("Gram-positive", "Catenibacterium mitsuokai is reported Gram-positive."),
    "Citrobacter": ("Gram-negative", "Enterobacterales/Enterobacteriaceae genus."),
    "Clostridium": (
        "Gram-positive",
        "Clostridium spp. are treated as Gram-positive anaerobic spore-forming rods.",
    ),
    "Enterococcus": ("Gram-positive", "Lactic-acid-bacteria genus."),
    "Escherichia": ("Gram-negative", "Enterobacterales/Enterobacteriaceae genus."),
    "Klebsiella": ("Gram-negative", "Enterobacterales/Enterobacteriaceae genus."),
    "Kluyvera": ("Gram-negative", "Enterobacterales/Enterobacteriaceae genus."),
    "Lacticaseibacillus": ("Gram-positive", "Lactobacillus-reclassified lactic-acid-bacteria genus."),
    "Lactiplantibacillus": ("Gram-positive", "Lactobacillus-reclassified lactic-acid-bacteria genus."),
    "Lactobacillus": ("Gram-positive", "Lactic-acid-bacteria genus."),
    "Lactococcus": ("Gram-positive", "Lactic-acid-bacteria genus."),
    "Leuconostoc": ("Gram-positive", "Lactic-acid-bacteria genus."),
    "Ligilactobacillus": ("Gram-positive", "Lactobacillus-reclassified lactic-acid-bacteria genus."),
    "Limosilactobacillus": ("Gram-positive", "Lactobacillus-reclassified lactic-acid-bacteria genus."),
    "Megasphaera": ("Gram-negative", "Megasphaera spp. are treated as Gram-negative anaerobic cocci."),
    "Odoribacter": ("Gram-negative", "Odoribacter splanchnicus is reported Gram-negative."),
    "Parabacteroides": ("Gram-negative", "Parabacteroides is treated as a Gram-negative anaerobic genus."),
    "Pediococcus": ("Gram-positive", "Lactic-acid-bacteria genus."),
    "Proteus": ("Gram-negative", "Enterobacterales/Enterobacteriaceae genus."),
    "Raoultella": ("Gram-negative", "Enterobacterales/Enterobacteriaceae genus."),
    "Streptococcus": ("Gram-positive", "Lactic-acid-bacteria genus."),
    "Weissella": ("Gram-positive", "Lactic-acid-bacteria genus."),
}

GRAM_COLORS = {
    "Gram-negative": "#e11d48",
    "Gram-positive": "#6d28d9",
    "Unknown": "#9ca3af",
}
GRAM_ORDER = ["Gram-negative", "Gram-positive", "Unknown"]


def aid_sort_key(aid: str) -> tuple[str, int, str]:
    text = str(aid)
    prefix = "".join(char for char in text if not char.isdigit())
    digits = "".join(char for char in text if char.isdigit())
    numeric = int(digits) if digits else -1
    return prefix, numeric, text


def load_taxonomy(root: Path) -> pd.DataFrame:
    taxonomy_path = root / "tables" / TAXONOMY
    taxonomy = pd.read_csv(taxonomy_path)
    taxonomy["AID"] = taxonomy["AID"].astype(str)
    taxonomy["_aid_sort_key"] = taxonomy["AID"].map(aid_sort_key)
    taxonomy = taxonomy.sort_values("_aid_sort_key").drop(columns="_aid_sort_key")
    missing = sorted(set(taxonomy["genus"]) - set(GRAM_STATUS_BY_GENUS))
    if missing:
        raise ValueError(f"Missing Gram-status mapping for genera: {missing}")
    return taxonomy


def build_gram_tables(taxonomy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_rows = []
    for row in taxonomy.itertuples(index=False):
        gram_status, gram_note = GRAM_STATUS_BY_GENUS[row.genus]
        sample_rows.append(
            {
                "AID": row.AID,
                "species": row.species,
                "genus": row.genus,
                "gram_status": gram_status,
                "gram_note": gram_note,
            }
        )
    sample_table = pd.DataFrame(sample_rows)

    genus_table = (
        sample_table.groupby(["genus", "gram_status", "gram_note"], sort=False)
        .agg(
            n_strains=("AID", "count"),
            n_species=("species", "nunique"),
            first_AID=("AID", "first"),
            AIDs=("AID", lambda values: ", ".join(values)),
        )
        .reset_index()
    )
    genus_table["_aid_sort_key"] = genus_table["first_AID"].map(aid_sort_key)
    genus_table = genus_table.sort_values("_aid_sort_key").drop(columns="_aid_sort_key")

    summary = (
        sample_table.groupby("gram_status")
        .agg(
            n_strains=("AID", "count"),
            n_genera=("genus", "nunique"),
            n_species=("species", "nunique"),
        )
        .reset_index()
    )
    summary["gram_status"] = pd.Categorical(summary["gram_status"], GRAM_ORDER, ordered=True)
    summary = summary.sort_values("gram_status").reset_index(drop=True)
    summary["gram_status"] = summary["gram_status"].astype(str)
    return sample_table, genus_table, summary


def load_neural_mds(root: Path, sample_table: pd.DataFrame) -> pd.DataFrame:
    coordinates_path = root / "figures" / MDS_COORDINATES
    coordinates = pd.read_csv(coordinates_path)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    merged = coordinates.merge(sample_table, left_on="sample_id", right_on="AID", how="inner")
    if len(merged) != len(sample_table):
        missing = sorted(set(sample_table["AID"]) - set(merged["sample_id"]))
        raise ValueError(f"Missing MDS coordinates for samples: {missing}")
    merged["_aid_sort_key"] = merged["AID"].map(aid_sort_key)
    return merged.sort_values("_aid_sort_key").drop(columns="_aid_sort_key")


def plot_static(merged: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(8.6, 6.4), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    for status in GRAM_ORDER:
        group = merged[merged["gram_status"].eq(status)]
        if group.empty:
            continue
        row = summary[summary["gram_status"].eq(status)].iloc[0]
        axis.scatter(
            group["neural_mds1"],
            group["neural_mds2"],
            group["neural_mds3"],
            s=52,
            color=GRAM_COLORS[status],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.55,
            label=f"{status} (n={int(row.n_strains)}, genera={int(row.n_genera)})",
        )
    axis.set_title("Neural 3D MDS", fontsize=12)
    axis.set_xlabel("MDS1", fontsize=9)
    axis.set_ylabel("MDS2", fontsize=9)
    axis.set_zlabel("MDS3", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9, frameon=False)
    figure.suptitle("86bac neural MDS colored by Gram status", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_html(merged: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    import plotly.graph_objects as go

    figure = go.Figure()
    for status in GRAM_ORDER:
        group = merged[merged["gram_status"].eq(status)]
        if group.empty:
            continue
        row = summary[summary["gram_status"].eq(status)].iloc[0]
        figure.add_trace(
            go.Scatter3d(
                x=group["neural_mds1"],
                y=group["neural_mds2"],
                z=group["neural_mds3"],
                mode="markers",
                name=f"{status} (n={int(row.n_strains)}, genera={int(row.n_genera)})",
                marker={
                    "size": 5.8,
                    "color": GRAM_COLORS[status],
                    "line": {"width": 0.5, "color": "white"},
                },
                text=group["AID"],
                customdata=np.column_stack([group["genus"], group["species"], group["gram_note"]]),
                hovertemplate=(
                    "%{text}<br>"
                    "Gram=%{fullData.name}<br>"
                    "genus=%{customdata[0]}<br>"
                    "species=%{customdata[1]}<br>"
                    "%{customdata[2]}<br>"
                    "MDS1=%{x:.3f}<br>MDS2=%{y:.3f}<br>MDS3=%{z:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="86bac neural 3D MDS colored by Gram status",
        width=900,
        height=700,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        legend={"font": {"size": 11}},
        scene={
            "xaxis_title": "MDS1",
            "yaxis_title": "MDS2",
            "zaxis_title": "MDS3",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def run(root: Path) -> dict[str, object]:
    taxonomy = load_taxonomy(root)
    sample_table, genus_table, summary = build_gram_tables(taxonomy)
    merged = load_neural_mds(root, sample_table)

    output_dir = root / "tables" / "taxonomy_from_GM300" / "gram_status_prior"
    figures_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sample_table.to_csv(output_dir / "gram_status_sample_mapping.csv", index=False)
    genus_table.to_csv(output_dir / "gram_status_genus_mapping.csv", index=False)
    summary.to_csv(output_dir / "gram_status_summary.csv", index=False)

    png_path = figures_dir / "gram_status_neural_mds_3d.png"
    html_path = figures_dir / "gram_status_neural_mds_3d.html"
    plot_static(merged, summary, png_path)
    plot_html(merged, summary, html_path)

    run_summary = {
        "input": {
            "root": str(root),
            "taxonomy": str(root / "tables" / TAXONOMY),
            "mds_coordinates": str(root / "figures" / MDS_COORDINATES),
        },
        "n_samples": int(len(sample_table)),
        "n_genera": int(sample_table["genus"].nunique()),
        "n_species": int(sample_table["species"].nunique()),
        "gram_status_counts": summary.to_dict(orient="records"),
        "outputs": {
            "tables_dir": str(output_dir),
            "figures": [str(png_path), str(html_path)],
        },
    }
    (output_dir / "gram_status_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 86bac neural 3D MDS colored by Gram status.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))


if __name__ == "__main__":
    main()
