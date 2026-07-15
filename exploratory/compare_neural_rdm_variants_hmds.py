"""Compare HMDS embeddings across 4 neural RDM construction variants.

Builds 4 RDMs by crossing:
  - time window: response_window (t05:t24) vs full_trajectory (t00:t44)
  - scaling: active_scale vs raw (no scaling)

Each variant → correlation RDM → chord transform → normalize → HMDS 2D → outputs.
Outputs are saved to separate subdirectories (no comparison table).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = str(PROJECT_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
EXPLORATORY = str(Path(__file__).resolve().parent)
if EXPLORATORY not in sys.path:
    sys.path.insert(0, EXPLORATORY)

import compare_86bac_chord_hmds as base

from bacteria_analysis.features.neural import (
    build_trial_feature_matrix,
    neural_feature_columns,
)

# ---------------------------------------------------------------------------
# helpers ported from compute_86bac_shape_pca_rsa (avoid importing main scripts)
# ---------------------------------------------------------------------------

MERGED_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)


def _sample_id_from_stim_name(stim_name: object) -> str:
    parts = str(stim_name).strip().split()
    return parts[0] if parts else ""


def _feature_window_columns(
    columns: list[str], *, window_start: int, window_stop: int
) -> list[str]:
    wanted = {f"t{t:02d}" for t in range(window_start, window_stop)}
    return [c for c in columns if c.rsplit("__", 1)[-1] in wanted]


# ── aggregation ────────────────────────────────────────────────────────────

def _aggregate_features(
    features: pd.DataFrame,
    *,
    group_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group in features.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prototype = np.nanmedian(values, axis=0)
        row = dict(zip(group_columns, group_key, strict=True))
        row["n_trials"] = int(group["trial_id"].nunique())
        row.update(dict(zip(feature_columns, prototype, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


# ── active scale ───────────────────────────────────────────────────────────

def _compute_active_scales(
    date_stim_prototypes: pd.DataFrame,
    *,
    feature_columns: list[str],
    active_threshold: float = 0.2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for neuron in MERGED_NEURONS:
        neuron_cols = [c for c in feature_columns if c.startswith(f"{neuron}__")]
        vals = date_stim_prototypes.loc[:, neuron_cols].to_numpy(dtype=float, copy=False).ravel()
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            scale, n_active, method = 1.0, 0, "fallback_no_finite_values"
        else:
            active = finite[np.abs(finite) >= active_threshold]
            n_active = int(active.size)
            if active.size:
                scale = float(np.mean(np.abs(active)))
                method = "mean_abs_active_frames"
            else:
                scale, method = 1.0, "silent_neuron_unit_scale"
        if not np.isfinite(scale) or scale <= 0:
            scale, method = 1.0, "fallback_unit_scale"
        rows.append({
            "neuron": neuron,
            "active_scale": scale,
            "n_active_frames_for_scale": n_active,
            "scaling_method": method,
        })
    return pd.DataFrame(rows)


def _apply_active_scale(
    prototypes: pd.DataFrame,
    *,
    feature_columns: list[str],
    scales: pd.DataFrame,
) -> pd.DataFrame:
    scale_map = scales.set_index("neuron")["active_scale"].to_dict()
    scaled = prototypes.copy()
    for col in feature_columns:
        neuron = col.split("__", 1)[0]
        scaled[col] = scaled[col].astype(float) / float(scale_map[neuron])
    return scaled


# ── RDM builder ────────────────────────────────────────────────────────────

def _build_correlation_rdm(
    values: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    labels = values[label_column].astype(str).tolist()
    arr = values.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
    dist = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(dist, 0.0)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            valid = np.isfinite(arr[i]) & np.isfinite(arr[j])
            if valid.sum() < 2:
                continue
            left, right = arr[i, valid], arr[j, valid]
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            d = float(np.clip(1.0 - np.corrcoef(left, right)[0, 1], 0.0, 2.0))
            dist[i, j] = dist[j, i] = d
    rdm = pd.DataFrame(dist, index=labels, columns=labels)
    rdm.index.name = label_column
    return rdm


# ── HMDS pipeline (reuses plot_86bac_hmds_2d logic) ────────────────────────

def _fit_hmds_2d(
    distance: np.ndarray,
    *,
    method: str,
    input_distance_name: str,
    analysis_label: str,
    hmds_distance_label: str,
    seed: int,
    hmds_starts: int,
    hmds_maxiter: int,
    bayesian_hmds: Path | None,
) -> base.EmbeddingResult:
    """Fit 2D hyperbolic MDS and return an EmbeddingResult."""
    # try Bayesian first, fall back to scipy
    bayesian = None
    if bayesian_hmds is not None:
        bayesian = base.run_bayesian_hmds_if_available(
            distance, dim=2, trials=3, bayesian_hmds=bayesian_hmds,
        )
    if bayesian is None:
        coords_lorentz, embedded_hyp, lambda_value, metadata = base.scipy_hyperbolic_mds(
            distance, dim=2, starts=hmds_starts, maxiter=hmds_maxiter, seed=seed,
        )
    else:
        coords_lorentz, embedded_hyp, lambda_value, metadata = bayesian

    poincare = base.recenter_poincare(base.lorentz_to_poincare(coords_lorentz))
    predicted = embedded_hyp / lambda_value
    n_params = distance.shape[0] * 2 + 1 - 2 * (2 - 1) / 2
    metrics = base.preservation_metrics(distance, embedded_hyp, predicted, n_params=n_params)

    # radius summary
    radius = np.linalg.norm(poincare, axis=1)
    hyp_radius = 2.0 * np.arctanh(np.clip(radius, 0.0, 0.999999))
    metrics.update({
        "lambda": float(lambda_value),
        "analysis_label": analysis_label,
        "hmds_distance_label": hmds_distance_label,
        "poincare_radius_min": float(radius.min()),
        "poincare_radius_median": float(np.median(radius)),
        "poincare_radius_mean": float(radius.mean()),
        "poincare_radius_q90": float(np.quantile(radius, 0.90)),
        "poincare_radius_q95": float(np.quantile(radius, 0.95)),
        "poincare_radius_max": float(radius.max()),
        "hyperbolic_radius_median": float(np.median(hyp_radius)),
        "fraction_radius_gt_0_90": float(np.mean(radius > 0.90)),
        "fraction_radius_gt_0_95": float(np.mean(radius > 0.95)),
        **metadata,
    })
    return base.EmbeddingResult(
        method=method,
        input_distance_name=input_distance_name,
        coordinates=poincare,
        embedded_distance=embedded_hyp,
        predicted_distance=predicted,
        metrics=metrics,
    )


# ── plot helpers ───────────────────────────────────────────────────────────

def _plot_static_disk(
    result: base.EmbeddingResult,
    sample_table: pd.DataFrame,
    color_table: pd.DataFrame,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    merged = sample_table.merge(
        color_table[["species", "species_index"]], on="species", how="left"
    )
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1.0)
    ax.add_artist(circle)
    norm = plt.Normalize(vmin=-0.5, vmax=len(color_table) - 0.5)
    ax.scatter(
        result.coordinates[:, 0],
        result.coordinates[:, 1],
        c=merged["species_index"],
        cmap=plt.cm.turbo,
        norm=norm,
        s=44,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Poincare 1", fontsize=9)
    ax.set_ylabel("Poincare 2", fontsize=9)
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    label = f"{base.analysis_label(result).title()} {base.hmds_distance_label(result)} HMDS 2D"
    ax.set_title(f"{title_prefix}{label}", fontsize=11)
    fig.suptitle(
        f"stress={result.metrics['normalized_raw_stress']:.3f};  "
        f"rho={result.metrics['distance_spearman']:.3f};  "
        f"lambda={result.metrics['lambda']:.3f}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_html_disk(
    result: base.EmbeddingResult,
    sample_table: pd.DataFrame,
    color_table: pd.DataFrame,
    output_path: Path,
    title_prefix: str = "",
) -> None:
    import plotly.graph_objects as go

    merged = sample_table.merge(
        color_table[["species", "species_index", "color_hex", "first_AID"]],
        on="species",
        how="left",
    )
    n_colors = len(color_table)
    colorscale: list[list[float | str]] = []
    for idx, ch in enumerate(color_table["color_hex"]):
        colorscale.append([idx / n_colors, ch])
        colorscale.append([(idx + 1) / n_colors, ch])

    theta = np.linspace(0, 2 * np.pi, 240)
    customdata = np.column_stack([
        merged["genus"].astype(str),
        merged["species"].astype(str),
        (merged["species_index"].astype(int) + 1).astype(str),
        merged["first_AID"].astype(str),
        np.linalg.norm(result.coordinates, axis=1),
    ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        line={"color": "rgba(75,85,99,0.55)", "width": 1.4},
        hoverinfo="skip", showlegend=False, name="boundary",
    ))
    fig.add_trace(go.Scatter(
        x=result.coordinates[:, 0], y=result.coordinates[:, 1],
        mode="markers",
        marker={
            "size": 8, "color": merged["species_index"],
            "colorscale": colorscale, "cmin": -0.5, "cmax": n_colors - 0.5,
            "line": {"width": 0.6, "color": "white"},
            "colorbar": {
                "title": "Species",
                "tickmode": "array",
                "tickvals": np.linspace(0, n_colors - 1, 6, dtype=int),
                "ticktext": [f"S{t + 1:02d}" for t in np.linspace(0, n_colors - 1, 6, dtype=int)],
                "tickfont": {"size": 10},
            },
        },
        text=merged["sample_id"].astype(str),
        customdata=customdata,
        hovertemplate=(
            "name=%{text}<br>genus=%{customdata[0]}<br>species=%{customdata[1]}<br>"
            "species color=S%{customdata[2]}<br>first AID=%{customdata[3]}<br>"
            "radius=%{customdata[4]:.4f}<br>P1=%{x:.4f}<br>P2=%{y:.4f}<extra></extra>"
        ),
        showlegend=False,
    ))
    label = f"{base.analysis_label(result)} {base.hmds_distance_label(result).lower()} HMDS 2D"
    fig.update_layout(
        title=f"{title_prefix}{label}<br>"
              f"stress={result.metrics['normalized_raw_stress']:.3f}, "
              f"rho={result.metrics['distance_spearman']:.3f}, "
              f"lambda={result.metrics['lambda']:.3f}",
        width=820, height=760,
        margin={"l": 30, "r": 120, "t": 70, "b": 40},
        xaxis={"title": "Poincare 1", "range": [-1.03, 1.03], "zeroline": False},
        yaxis={"title": "Poincare 2", "range": [-1.03, 1.03], "scaleanchor": "x", "scaleratio": 1, "zeroline": False},
        plot_bgcolor="white",
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


# ── per-variant pipeline ───────────────────────────────────────────────────

def _build_variant_rdm(
    trial_features: pd.DataFrame,
    feature_columns: list[str],
    *,
    apply_scale: bool,
    active_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """Build one variant's correlation RDM + metadata."""
    # sample-level prototypes
    sample_prototypes = _aggregate_features(
        trial_features,
        group_columns=["stim_name"],
        feature_columns=feature_columns,
    )
    sample_prototypes["sample_id"] = sample_prototypes["stim_name"].map(_sample_id_from_stim_name)

    extra = {"n_features": len(feature_columns)}

    if apply_scale:
        date_stim = _aggregate_features(
            trial_features,
            group_columns=["date", "stim_name"],
            feature_columns=feature_columns,
        )
        scales = _compute_active_scales(
            date_stim, feature_columns=feature_columns, active_threshold=active_threshold,
        )
        extra["active_scales"] = scales.set_index("neuron")["active_scale"].to_dict()
        scaled = _apply_active_scale(
            sample_prototypes, feature_columns=feature_columns, scales=scales,
        )
        rdm = _build_correlation_rdm(scaled, label_column="sample_id", feature_columns=feature_columns)
    else:
        rdm = _build_correlation_rdm(sample_prototypes, label_column="sample_id", feature_columns=feature_columns)

    # drop rows/cols with all-NaN
    valid = rdm.notna().all(axis=1)
    rdm = rdm.loc[valid, valid]
    extra["n_samples_in_rdm"] = int(valid.sum())
    extra["n_nan_samples"] = int((~valid).sum())
    return rdm, extra


def _run_one_variant(
    *,
    variant_id: str,
    variant_label: str,
    title_prefix: str,
    feature_columns: list[str],
    apply_scale: bool,
    trial_features: pd.DataFrame,
    taxonomy: pd.DataFrame,
    output_dir: Path,
    active_threshold: float,
    seed: int,
    hmds_starts: int,
    hmds_maxiter: int,
    bayesian_hmds: Path | None,
) -> dict:
    """Full pipeline for one variant: RDM → HMDS → save outputs."""
    print(f"\n{'='*60}")
    print(f"  {variant_label}")
    print(f"  features={len(feature_columns)}, active_scale={apply_scale}")
    print(f"{'='*60}")

    var_dir = output_dir / variant_id
    fig_dir = var_dir / "figures"
    tbl_dir = var_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. build RDM
    rdm, rdm_meta = _build_variant_rdm(
        trial_features, feature_columns,
        apply_scale=apply_scale, active_threshold=active_threshold,
    )
    print(f"  RDM: {rdm.shape[0]} samples, {rdm_meta['n_features']} features")
    if "active_scales" in rdm_meta:
        scales = rdm_meta["active_scales"]
        print(f"  active_scales range: [{min(scales.values()):.4f}, {max(scales.values()):.4f}]")

    rdm.to_csv(tbl_dir / "neural_correlation_rdm.csv")
    with open(tbl_dir / "rdm_build_metadata.json", "w") as fh:
        json.dump({k: v for k, v in rdm_meta.items() if k != "active_scales"}, fh, indent=2, default=str)

    # 2. align with taxonomy
    samples = [s for s in rdm.index.astype(str) if s in taxonomy.index]
    if len(samples) < 4:
        msg = f"Only {len(samples)} samples with taxonomy; need ≥4. Skipping HMDS."
        print(f"  {msg}")
        return {"variant_id": variant_id, "status": "skipped", "reason": msg}
    rdm = rdm.loc[samples, samples]
    tax = taxonomy.loc[samples].copy()
    sample_table = base.build_sample_table(samples, tax)
    color_table = base.species_color_table(tax)
    color_table.to_csv(tbl_dir / "species_color_map.csv", index=False)

    # 3. chord + normalize
    linear = base.clean_distance_matrix(rdm)
    chord = base.chord_from_linear(linear)
    distance = base.normalize_to_max_two(chord)
    pd.DataFrame(distance, index=samples, columns=samples).to_csv(
        tbl_dir / "distance__chord_normalized.csv"
    )

    # 4. HMDS
    try:
        result = _fit_hmds_2d(
            distance,
            method=f"{variant_id}_chord_hmds_poincare_2d",
            input_distance_name=f"{variant_id}_chord_normalized",
            analysis_label=variant_id,
            hmds_distance_label="Chord",
            seed=seed,
            hmds_starts=hmds_starts,
            hmds_maxiter=hmds_maxiter,
            bayesian_hmds=bayesian_hmds,
        )
    except Exception:
        print(f"  HMDS FAILED:\n{traceback.format_exc()}")
        return {"variant_id": variant_id, "status": "hmds_failed"}

    # 5. metrics
    print(f"  stress={result.metrics['normalized_raw_stress']:.4f}  "
          f"rho={result.metrics['distance_spearman']:.4f}  "
          f"lambda={result.metrics['lambda']:.4f}  "
          f"backend={result.metrics.get('backend', '?')}")
    print(f"  radius: median={result.metrics['poincare_radius_median']:.3f}  "
          f"q90={result.metrics['poincare_radius_q90']:.3f}  "
          f"frac>0.90={result.metrics['fraction_radius_gt_0_90']:.3f}")

    with open(tbl_dir / "metrics.json", "w") as fh:
        json.dump(result.metrics, fh, indent=2, default=str)

    # 6. save outputs
    coord_out = tbl_dir / "coordinates.csv"
    tbl = sample_table.copy()
    tbl["poincare1"] = result.coordinates[:, 0]
    tbl["poincare2"] = result.coordinates[:, 1]
    tbl["poincare_radius"] = np.linalg.norm(result.coordinates, axis=1)
    tbl.to_csv(coord_out, index=False)

    base.save_pair_distances(result, distance, samples, tbl_dir / "pair_distances.csv")
    base.plot_shepard(result, distance, fig_dir / "shepard.png")
    _plot_static_disk(result, sample_table, color_table, fig_dir / "poincare_disk.png", title_prefix)
    _plot_html_disk(result, sample_table, color_table, fig_dir / "poincare_disk.html", title_prefix)

    print(f"  Done → {var_dir}")
    return {"variant_id": variant_id, "status": "ok", "metrics": result.metrics}


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HMDS across 4 neural RDM variants: "
                    "window × active_scale."
    )
    parser.add_argument("--neural-parquet", type=Path, default=Path("data/86bac.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/neural_rdm_variants_hmds"))
    parser.add_argument("--taxonomy-csv", type=Path,
                        default=Path("results/86bac_shape_pca_rsa_t05_t24_silent_scale1/tables/"
                                     "taxonomy_from_GM300/86bac_sample_species_mapping.csv"))
    parser.add_argument("--window-start", type=int, default=5)
    parser.add_argument("--window-stop", type=int, default=25)
    parser.add_argument("--active-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--hmds-starts", type=int, default=8)
    parser.add_argument("--hmds-maxiter", type=int, default=900)
    parser.add_argument("--bayesian-hmds", type=Path, default=None)
    args = parser.parse_args()

    # ── load data ──────────────────────────────────────────────────────
    print("Loading data...")
    raw_neural = pd.read_parquet(args.neural_parquet)
    trial_features = build_trial_feature_matrix(
        raw_neural, view="full_trajectory", merge_lr=True,
    )
    all_feature_columns = neural_feature_columns(trial_features)
    print(f"  trial_features: {trial_features.shape[0]} trials, "
          f"{len(all_feature_columns)} neural feature columns")

    response_window_cols = _feature_window_columns(
        all_feature_columns,
        window_start=args.window_start,
        window_stop=args.window_stop,
    )
    print(f"  response_window (t{args.window_start:02d}:t{args.window_stop:02d}): "
          f"{len(response_window_cols)} columns")
    print(f"  full_trajectory: {len(all_feature_columns)} columns")

    # ── taxonomy ───────────────────────────────────────────────────────
    taxonomy = pd.read_csv(args.taxonomy_csv)
    taxonomy["AID"] = taxonomy["AID"].astype(str)
    taxonomy = taxonomy.set_index("AID", drop=False)
    print(f"  taxonomy: {len(taxonomy)} entries")

    # ── define variants ────────────────────────────────────────────────
    variants: list[dict] = [
        {
            "variant_id": "response_window_active",
            "variant_label": "Response Window + Active Scale",
            "title_prefix": "86bac RW+AS  ",
            "feature_columns": response_window_cols,
            "apply_scale": True,
        },
        {
            "variant_id": "response_window_raw",
            "variant_label": "Response Window  (no active scale)",
            "title_prefix": "86bac RW Raw  ",
            "feature_columns": response_window_cols,
            "apply_scale": False,
        },
        {
            "variant_id": "full_trajectory_active",
            "variant_label": "Full Trajectory + Active Scale",
            "title_prefix": "86bac FT+AS  ",
            "feature_columns": all_feature_columns,
            "apply_scale": True,
        },
        {
            "variant_id": "full_trajectory_raw",
            "variant_label": "Full Trajectory  (no active scale)",
            "title_prefix": "86bac FT Raw  ",
            "feature_columns": all_feature_columns,
            "apply_scale": False,
        },
    ]

    # ── run ────────────────────────────────────────────────────────────
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for v in variants:
        result = _run_one_variant(
            trial_features=trial_features,
            taxonomy=taxonomy,
            output_dir=output_dir,
            active_threshold=args.active_threshold,
            seed=args.seed,
            hmds_starts=args.hmds_starts,
            hmds_maxiter=args.hmds_maxiter,
            bayesian_hmds=args.bayesian_hmds,
            **{k: v[k] for k in [
                "variant_id", "variant_label", "title_prefix",
                "feature_columns", "apply_scale",
            ]},
        )
        results.append(result)

    # ── summary ────────────────────────────────────────────────────────
    summary = {
        "parameters": {
            "neural_parquet": str(args.neural_parquet),
            "window_start": args.window_start,
            "window_stop": args.window_stop,
            "active_threshold": args.active_threshold,
            "response_window_n_features": len(response_window_cols),
            "full_trajectory_n_features": len(all_feature_columns),
            "seed": args.seed,
            "hmds_starts": args.hmds_starts,
            "hmds_maxiter": args.hmds_maxiter,
        },
        "results": [],
    }
    for r in results:
        entry = {"variant_id": r["variant_id"], "status": r["status"]}
        if "metrics" in r:
            m = r["metrics"]
            entry.update({
                "n_samples": m.get("n_samples"),
                "stress": m.get("normalized_raw_stress"),
                "spearman_rho": m.get("distance_spearman"),
                "lambda": m.get("lambda"),
                "poincare_radius_median": m.get("poincare_radius_median"),
                "poincare_radius_q90": m.get("poincare_radius_q90"),
                "fraction_radius_gt_0_90": m.get("fraction_radius_gt_0_90"),
                "backend": m.get("backend"),
            })
        elif "reason" in r:
            entry["reason"] = r["reason"]
        summary["results"].append(entry)

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary → {summary_path}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
