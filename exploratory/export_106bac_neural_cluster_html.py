"""Export 106bac neural trajectory HTML, columns ordered by neural shape RDM clustering.

Same logic as export_86bac_neural_cluster_html.py, adapted for 106bac.
Neural RDM built inline: active-scaled flattened trace (t05:t25), correlation distance.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis.features.neural import build_trial_feature_matrix, neural_feature_columns
from bacteria_analysis._data_loaders import enrich_neural_dataframe

DEFAULT_NEURAL_PARQUET = Path("data") / "106bac.parquet"
DEFAULT_OUTPUT = Path("results") / "chemical_pca_ward" / "neural_trajectory_by_neural_rdm_cluster_order.html"

MERGED_NEURON_ORDER = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)

LR_MERGE_MAP = {
    "ADFL": "ADF", "ADFR": "ADF", "ADLL": "ADL", "ADLR": "ADL",
    "ASGL": "ASG", "ASGR": "ASG", "ASHL": "ASH", "ASHR": "ASH",
    "ASIL": "ASI", "ASIR": "ASI", "ASJL": "ASJ", "ASJR": "ASJ",
    "ASKL": "ASK", "ASKR": "ASK", "AWAL": "AWA", "AWAR": "AWA",
    "AWBL": "AWB", "AWBR": "AWB",
}

REQUIRED_COLUMNS = {
    "neuron", "time_point", "delta_F_over_F0",
    "start_time", "end_time", "stim_name", "stim_color",
}

# Active-scaling parameters (matching 86bac pipeline)
WINDOW_START = 5
WINDOW_STOP = 25
ACTIVE_THRESHOLD = 0.2
SILENT_SCALE = 1.0


# ── neural RDM builder (replicates compute_86bac_shape_pca_rsa logic) ──


def sample_id_from_stim_name(stim_name: object) -> str:
    parts = str(stim_name).strip().split()
    return parts[0] if parts else ""


def feature_window_columns(columns: list[str], *, window_start: int, window_stop: int) -> list[str]:
    wanted = {f"t{t:02d}" for t in range(window_start, window_stop)}
    return [c for c in columns if c.rsplit("__", 1)[-1] in wanted]


def build_106bac_neural_rdm(raw_neural: pd.DataFrame) -> pd.DataFrame:
    """Build sample-level neural shape RDM for 106bac (active-scaled, correlation distance)."""
    features = build_trial_feature_matrix(raw_neural, view="full_trajectory", merge_lr=True)
    all_cols = neural_feature_columns(features)
    window_cols = feature_window_columns(all_cols, window_start=WINDOW_START, window_stop=WINDOW_STOP)

    # Aggregate: date×stim prototypes for scale computation
    def agg_prototypes(frame, group_cols):
        rows = []
        for gk, grp in frame.groupby(group_cols, sort=True, dropna=False):
            if not isinstance(gk, tuple):
                gk = (gk,)
            vals = grp.loc[:, window_cols].to_numpy(dtype=float, copy=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                proto = np.nanmedian(vals, axis=0)
            row = dict(zip(group_cols, gk, strict=True))
            row["n_trials"] = int(grp["trial_id"].nunique())
            row.update(dict(zip(window_cols, proto, strict=True)))
            rows.append(row)
        return pd.DataFrame(rows)

    date_stim = agg_prototypes(features, ["date", "stim_name"])

    # Active scales per neuron
    merged_neurons = (
        "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
        "AWA", "AWB", "AWCOFF", "AWCON",
    )
    scale_rows = []
    for neuron in merged_neurons:
        ncols = [c for c in window_cols if c.startswith(f"{neuron}__")]
        vals = date_stim.loc[:, ncols].to_numpy(dtype=float, copy=False).ravel()
        fin = vals[np.isfinite(vals)]
        if fin.size == 0:
            scale = 1.0
        else:
            active = fin[np.abs(fin) >= ACTIVE_THRESHOLD]
            scale = float(np.mean(np.abs(active))) if active.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        scale_rows.append({"neuron": neuron, "scale": scale})
    scales_df = pd.DataFrame(scale_rows).set_index("neuron")["scale"]

    # Sample-level prototypes
    sample_prototypes = agg_prototypes(features, ["stim_name"])
    sample_prototypes["sample_id"] = sample_prototypes["stim_name"].map(sample_id_from_stim_name)

    # Apply active scaling
    scaled = sample_prototypes.copy()
    for col in window_cols:
        neuron = col.split("__", 1)[0]
        scaled[col] = scaled[col].astype(float) / float(scales_df[neuron])

    # Correlation RDM (1 - Pearson r)
    values = scaled.loc[:, window_cols].to_numpy(dtype=float, copy=False)
    row_means = np.nanmean(values, axis=1, keepdims=True)
    row_stds = np.nanstd(values, axis=1, ddof=0, keepdims=True)
    values_z = np.where(row_stds > 0, (values - row_means) / row_stds, 0.0)

    n = len(values_z)
    distances = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(distances, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            valid = np.isfinite(values_z[i]) & np.isfinite(values_z[j])
            if valid.sum() < 2:
                continue
            left = values_z[i, valid]; right = values_z[j, valid]
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            distances[i, j] = distances[j, i] = float(np.clip(
                1.0 - np.corrcoef(left, right)[0, 1], 0.0, 2.0))

    labels = scaled["sample_id"].astype(str).tolist()
    return pd.DataFrame(distances, index=labels, columns=labels)


# ── HTML export (same logic as export_86bac_neural_cluster_html.py) ─────


def neural_cluster_order(rdm: pd.DataFrame) -> list[str]:
    labels = rdm.index.astype(str).tolist()
    if len(labels) < 3:
        return labels
    matrix = rdm.to_numpy(dtype=float, copy=True)
    off_diag = matrix[~np.eye(len(labels), dtype=bool)]
    finite = off_diag[np.isfinite(off_diag)]
    fill = float(np.max(finite)) if finite.size else 1.0
    matrix = np.where(np.isfinite(matrix), matrix, fill)
    np.fill_diagonal(matrix, 0.0)
    order = leaves_list(linkage(squareform(matrix, checks=False), method="average"))
    return [labels[idx] for idx in order]


def prepare_plot_frame(raw: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
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
        raise ValueError(f"ambiguous sample metadata for: {ambiguous.index.astype(str).tolist()[:10]}")
    available = set(metadata["sample_id"].astype(str))
    expected = set(stimulus_order)
    if available != expected:
        raise ValueError(
            f"RDM/data mismatch: missing={sorted(expected-available)}, extra={sorted(available-expected)}"
        )
    return metadata.set_index("sample_id").loc[stimulus_order]


def rgba(hex_color: str, alpha: float) -> str:
    c = str(hex_color).strip().lstrip("#")
    if len(c) != 6:
        return f"rgba(128,128,128,{alpha})"
    try:
        r, g, b = (int(c[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return f"rgba(128,128,128,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"


def neuron_y_ranges(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ranges = {}
    for neuron in MERGED_NEURON_ORDER:
        vals = pd.to_numeric(
            frame.loc[frame["neuron_display"].eq(neuron), "delta_F_over_F0"],
            errors="coerce",
        ).dropna()
        if vals.empty:
            ranges[neuron] = (-0.5, 0.5)
            continue
        lo, hi = float(vals.quantile(0.05)), float(vals.quantile(0.95))
        buf = (hi - lo) * 0.15
        if not np.isfinite(buf) or buf <= 0:
            buf = 0.1
        ranges[neuron] = (lo - buf, hi + buf)
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
    n_rows = len(MERGED_NEURON_ORDER)
    n_cols = len(stimulus_order)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_yaxes=False, horizontal_spacing=0.0, vertical_spacing=0.0,
    )

    for ri, neuron in enumerate(MERGED_NEURON_ORDER, start=1):
        yr = y_ranges[neuron]
        fig.add_annotation(
            text=f"<b>{neuron}</b>", xref="x domain", yref="y domain",
            x=-0.05, y=0.5, showarrow=False, xanchor="right", yanchor="middle",
            row=ri, col=1, font=dict(size=12, color="black"),
        )
        for ci, sid in enumerate(stimulus_order, start=1):
            sdat = stats.loc[
                stats["neuron_display"].eq(neuron) & stats["sample_id"].eq(sid)
            ].sort_values("rel_time", kind="stable")
            if sdat.empty:
                continue
            color = str(metadata.loc[sid, "stim_color"])
            etr = float(metadata.loc[sid, "stimulus_end_rel"])
            x = sdat["rel_time"].to_numpy(dtype=float)
            mean = sdat["mean"].to_numpy(dtype=float)
            sem = sdat["sem"].fillna(0.0).to_numpy(dtype=float)

            fig.add_shape(
                type="rect", x0=0.0, x1=etr, y0=yr[0], y1=yr[1],
                fillcolor=color, opacity=0.15, layer="below", line_width=0,
                row=ri, col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=mean, mode="lines",
                    line=dict(color=color, width=2), showlegend=False,
                    hovertemplate=f"{sid}<br>{neuron}<br>"
                    "x: %{x}<br>mean: %{y:.3f}<br>N: %{customdata}<extra></extra>",
                    customdata=sdat["count"].to_numpy(dtype=int),
                ), row=ri, col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                    fill="toself", fillcolor=rgba(color, 0.30),
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False, hoverinfo="skip",
                ), row=ri, col=ci,
            )
            fig.update_yaxes(
                range=yr, showgrid=False,
                showticklabels=ci == 1, fixedrange=True,
                row=ri, col=ci,
            )
            fig.update_xaxes(
                showgrid=False, showticklabels=ri == n_rows,
                tickfont=dict(size=7), fixedrange=True,
                row=ri, col=ci,
            )
            if ri == 1:
                fig.add_annotation(
                    text=f"<b>{sid}</b>", xref="x domain", yref="y domain",
                    x=0.5, y=1.08, textangle=-90, showarrow=False,
                    xanchor="left", yanchor="bottom",
                    row=ri, col=ci, font=dict(size=9, color="#334155"),
                )

    fig.update_layout(
        title=dict(
            text=(
                "106bac neural trajectories ordered by active-scaled neural shape RDM clustering"
                "<br><sup>raw trace mean ± SEM; non-ASE L/R merged; shaded = stimulus; columns = average-linkage order</sup>"
            ),
            x=0.005, xanchor="left",
        ),
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
        template="plotly_white", hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Inter', sans-serif", color="#334155", size=13),
        showlegend=False,
    )
    return fig


def export_html(
    *,
    neural_parquet: Path,
    output_html: Path,
) -> dict[str, object]:
    raw = pd.read_parquet(neural_parquet)
    print("Building 106bac neural RDM...")
    rdm = build_106bac_neural_rdm(raw)
    stimulus_order = neural_cluster_order(rdm)
    print(f"  {len(stimulus_order)} stimuli in cluster order")

    frame = prepare_plot_frame(raw)
    metadata = validate_sample_scope(frame, stimulus_order)
    figure = create_trajectory_figure(frame, stimulus_order=stimulus_order, metadata=metadata)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    figure.layout.width = None
    figure.layout.height = None
    figure.update_layout(autosize=True, template="plotly_white", margin=dict(l=50, r=50, t=80, b=50))
    figure.write_html(
        output_html, include_plotlyjs="cdn", full_html=True,
        config={"responsive": True, "scrollZoom": False, "displayModeBar": False},
    )
    print(f"Saved: {output_html}")

    order_csv = output_html.with_suffix(".csv")
    pd.DataFrame({
        "cluster_position": np.arange(1, len(stimulus_order) + 1),
        "sample_id": stimulus_order,
        "stim_name": metadata["stim_name"].astype(str).tolist(),
    }).to_csv(order_csv, index=False)

    summary = {
        "input_neural_parquet": str(neural_parquet),
        "output_html": str(output_html),
        "output_cluster_order": str(order_csv),
        "trace_value": "raw delta_F_over_F0",
        "display": "mean ± SEM",
        "lr_merge": "non-ASE L/R merge",
        "rdm": "active-scaled flattened trace [5,25), 1-Pearson correlation, average-linkage clustering",
        "n_raw_rows": int(len(raw)),
        "n_display_neurons": int(len(MERGED_NEURON_ORDER)),
        "n_stimuli": int(len(stimulus_order)),
        "n_plotly_traces": int(len(figure.data)),
    }
    summary_path = output_html.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export 106bac neural trajectory HTML.")
    parser.add_argument("--neural-parquet", type=Path, default=DEFAULT_NEURAL_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = export_html(neural_parquet=args.neural_parquet, output_html=args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
