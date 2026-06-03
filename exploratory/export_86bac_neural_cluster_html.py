from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


DEFAULT_NEURAL_PARQUET = Path("data") / "86bac.parquet"
DEFAULT_RDM = (
    Path("results")
    / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
    / "tables"
    / "neural_shape_rdm__active_scaled_flattened_correlation.csv"
)
DEFAULT_OUTPUT = (
    Path("results")
    / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
    / "figures"
    / "neural_trajectory_by_neural_rdm_cluster_order.html"
)

MERGED_NEURON_ORDER = (
    "ADF",
    "ADL",
    "ASEL",
    "ASER",
    "ASG",
    "ASH",
    "ASI",
    "ASJ",
    "ASK",
    "AWA",
    "AWB",
    "AWCOFF",
    "AWCON",
)

LR_MERGE_MAP = {
    "ADFL": "ADF",
    "ADFR": "ADF",
    "ADLL": "ADL",
    "ADLR": "ADL",
    "ASGL": "ASG",
    "ASGR": "ASG",
    "ASHL": "ASH",
    "ASHR": "ASH",
    "ASIL": "ASI",
    "ASIR": "ASI",
    "ASJL": "ASJ",
    "ASJR": "ASJ",
    "ASKL": "ASK",
    "ASKR": "ASK",
    "AWAL": "AWA",
    "AWAR": "AWA",
    "AWBL": "AWB",
    "AWBR": "AWB",
}

REQUIRED_COLUMNS = {
    "neuron",
    "time_point",
    "delta_F_over_F0",
    "start_time",
    "end_time",
    "stim_name",
    "stim_color",
}


def sample_id_from_stim_name(stim_name: object) -> str:
    parts = str(stim_name).strip().split()
    return parts[0] if parts else ""


def neural_cluster_order(rdm: pd.DataFrame) -> list[str]:
    labels = rdm.index.astype(str).tolist()
    if labels != rdm.columns.astype(str).tolist():
        raise ValueError("neural RDM row and column labels must match in the same order")
    if len(labels) < 3:
        return labels

    matrix = rdm.to_numpy(dtype=float, copy=True)
    off_diagonal = matrix[~np.eye(len(labels), dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    fill_value = float(np.max(finite)) if finite.size else 1.0
    matrix = np.where(np.isfinite(matrix), matrix, fill_value)
    np.fill_diagonal(matrix, 0.0)
    order = leaves_list(linkage(squareform(matrix, checks=False), method="average"))
    return [labels[index] for index in order]


def prepare_plot_frame(raw: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"neural parquet is missing required columns: {missing}")

    frame = raw.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    frame["sample_id"] = frame["stim_name"].map(sample_id_from_stim_name)
    frame["neuron_display"] = frame["neuron"].replace(LR_MERGE_MAP)
    frame["rel_time"] = frame["time_point"].astype(float) - frame["start_time"].astype(float)
    frame["stimulus_end_rel"] = frame["end_time"].astype(float) - frame["start_time"].astype(float)
    return frame


def validate_sample_scope(frame: pd.DataFrame, stimulus_order: list[str]) -> pd.DataFrame:
    metadata = (
        frame.loc[:, ["sample_id", "stim_name", "stim_color", "stimulus_end_rel"]]
        .drop_duplicates()
        .sort_values("sample_id", kind="stable")
    )
    counts = metadata.groupby("sample_id").size()
    ambiguous = counts[counts.ne(1)]
    if len(ambiguous):
        raise ValueError(f"sample metadata is not one-to-one for: {ambiguous.index.astype(str).tolist()[:10]}")

    available = set(metadata["sample_id"].astype(str))
    expected = set(stimulus_order)
    if available != expected:
        raise ValueError(
            "RDM/data stimulus mismatch: "
            f"missing_from_data={sorted(expected - available)}, extra_in_data={sorted(available - expected)}"
        )

    display_neurons = set(frame["neuron_display"].astype(str))
    if display_neurons != set(MERGED_NEURON_ORDER):
        raise ValueError(
            "unexpected merged neuron scope: "
            f"missing={sorted(set(MERGED_NEURON_ORDER) - display_neurons)}, "
            f"extra={sorted(display_neurons - set(MERGED_NEURON_ORDER))}"
        )
    return metadata.set_index("sample_id").loc[stimulus_order]


def rgba(hex_color: str, alpha: float) -> str:
    cleaned = str(hex_color).strip().lstrip("#")
    if len(cleaned) != 6:
        return f"rgba(128,128,128,{alpha})"
    try:
        red, green, blue = (int(cleaned[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return f"rgba(128,128,128,{alpha})"
    return f"rgba({red},{green},{blue},{alpha})"


def neuron_y_ranges(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for neuron in MERGED_NEURON_ORDER:
        values = pd.to_numeric(
            frame.loc[frame["neuron_display"].eq(neuron), "delta_F_over_F0"],
            errors="coerce",
        ).dropna()
        if values.empty:
            ranges[neuron] = (-0.5, 0.5)
            continue
        lower = float(values.quantile(0.05))
        upper = float(values.quantile(0.95))
        buffer = (upper - lower) * 0.15
        if not np.isfinite(buffer) or buffer <= 0:
            buffer = 0.1
        ranges[neuron] = (lower - buffer, upper + buffer)
    return ranges


def create_trajectory_figure(
    frame: pd.DataFrame,
    *,
    stimulus_order: list[str],
    metadata: pd.DataFrame,
) -> go.Figure:
    stats = (
        frame.groupby(["neuron_display", "sample_id", "rel_time"], sort=False)["delta_F_over_F0"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    y_ranges = neuron_y_ranges(frame)
    row_count = len(MERGED_NEURON_ORDER)
    column_count = len(stimulus_order)

    figure = make_subplots(
        rows=row_count,
        cols=column_count,
        shared_yaxes=False,
        horizontal_spacing=0.0,
        vertical_spacing=0.0,
    )

    for row_index, neuron in enumerate(MERGED_NEURON_ORDER, start=1):
        y_range = y_ranges[neuron]
        figure.add_annotation(
            text=f"<b>{neuron}</b>",
            xref="x domain",
            yref="y domain",
            x=-0.05,
            y=0.5,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            row=row_index,
            col=1,
            font=dict(size=12, color="black"),
        )

        for column_index, sample_id in enumerate(stimulus_order, start=1):
            sample_stats = stats.loc[
                stats["neuron_display"].eq(neuron) & stats["sample_id"].eq(sample_id)
            ].sort_values("rel_time", kind="stable")
            if sample_stats.empty:
                continue

            color = str(metadata.loc[sample_id, "stim_color"])
            end_time_rel = float(metadata.loc[sample_id, "stimulus_end_rel"])
            error = sample_stats["sem"].fillna(0.0)
            x = sample_stats["rel_time"].to_numpy(dtype=float)
            mean = sample_stats["mean"].to_numpy(dtype=float)
            sem = error.to_numpy(dtype=float)

            figure.add_shape(
                type="rect",
                x0=0.0,
                x1=end_time_rel,
                y0=y_range[0],
                y1=y_range[1],
                fillcolor=color,
                opacity=0.15,
                layer="below",
                line_width=0,
                row=row_index,
                col=column_index,
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertemplate=(
                        f"{sample_id}<br>{neuron}<br>"
                        "x: %{x}<br>mean: %{y:.3f}<br>N: %{customdata}<extra></extra>"
                    ),
                    customdata=sample_stats["count"].to_numpy(dtype=int),
                ),
                row=row_index,
                col=column_index,
            )
            figure.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                    fill="toself",
                    fillcolor=rgba(color, 0.30),
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_index,
                col=column_index,
            )
            figure.update_yaxes(
                range=y_range,
                showgrid=False,
                showticklabels=column_index == 1,
                fixedrange=True,
                row=row_index,
                col=column_index,
            )
            figure.update_xaxes(
                showgrid=False,
                showticklabels=row_index == row_count,
                tickfont=dict(size=7),
                fixedrange=True,
                row=row_index,
                col=column_index,
            )

            if row_index == 1:
                figure.add_annotation(
                    text=f"<b>{sample_id}</b>",
                    xref="x domain",
                    yref="y domain",
                    x=0.5,
                    y=1.08,
                    textangle=-90,
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    row=row_index,
                    col=column_index,
                    font=dict(size=9, color="#334155"),
                )

    figure.update_layout(
        title=dict(
            text=(
                "86bac neural trajectories ordered by corrected neural-shape RDM clustering"
                "<br><sup>raw trace mean +/- SEM; non-ASE L/R merged; shaded area = stimulus interval</sup>"
            ),
            x=0.005,
            xanchor="left",
        ),
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
        template="plotly_white",
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Inter', sans-serif", color="#334155", size=13),
        showlegend=False,
    )
    return figure


def export_html(
    *,
    neural_parquet: Path,
    neural_rdm_csv: Path,
    output_html: Path,
) -> dict[str, object]:
    raw = pd.read_parquet(neural_parquet)
    rdm = pd.read_csv(neural_rdm_csv, index_col=0)
    stimulus_order = neural_cluster_order(rdm)
    frame = prepare_plot_frame(raw)
    metadata = validate_sample_scope(frame, stimulus_order)
    figure = create_trajectory_figure(frame, stimulus_order=stimulus_order, metadata=metadata)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.layout.width = None
    figure.layout.height = None
    figure.update_layout(autosize=True, template="plotly_white", margin=dict(l=50, r=50, t=80, b=50))
    figure.write_html(
        output_html,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "scrollZoom": False, "displayModeBar": False},
    )

    order_output = output_html.with_suffix(".csv")
    pd.DataFrame(
        {
            "cluster_position": np.arange(1, len(stimulus_order) + 1),
            "sample_id": stimulus_order,
            "stim_name": metadata["stim_name"].astype(str).tolist(),
        }
    ).to_csv(order_output, index=False)

    summary = {
        "input_neural_parquet": str(neural_parquet),
        "input_neural_rdm": str(neural_rdm_csv),
        "output_html": str(output_html),
        "output_cluster_order": str(order_output),
        "trace_value": "raw delta_F_over_F0",
        "display": "mean +/- SEM",
        "lr_merge": "non-ASE L/R merge matching visweb.py",
        "time_axis": "full relative time: time_point - start_time",
        "column_order": (
            "average-linkage clustering of corrected [5,25) active-scaled neural shape RDM "
            "with silent-neuron scale=1.0"
        ),
        "responsive_export": True,
        "zoom_enabled": False,
        "n_raw_rows": int(len(raw)),
        "n_raw_neurons": int(raw["neuron"].nunique()),
        "n_display_neurons": int(len(MERGED_NEURON_ORDER)),
        "n_stimuli": int(len(stimulus_order)),
        "n_plotly_traces": int(len(figure.data)),
        "n_stimulus_shapes": int(len(figure.layout.shapes)),
        "first_stimulus": stimulus_order[0],
        "last_stimulus": stimulus_order[-1],
    }
    summary_output = output_html.with_suffix(".summary.json")
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 86bac neural trajectories ordered by neural RDM clustering.")
    parser.add_argument("--neural-parquet", type=Path, default=DEFAULT_NEURAL_PARQUET)
    parser.add_argument("--neural-rdm", type=Path, default=DEFAULT_RDM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = export_html(
        neural_parquet=args.neural_parquet,
        neural_rdm_csv=args.neural_rdm,
        output_html=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
