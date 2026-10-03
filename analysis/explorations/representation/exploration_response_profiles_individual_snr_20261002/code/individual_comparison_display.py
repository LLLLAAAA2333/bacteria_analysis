"""Redraw saved RDMs and checked HMDS fits without rerunning response analysis."""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_hex
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

STYLE = {"font.family": "DejaVu Sans", "font.size": 11,
         "axes.spines.top": False, "axes.spines.right": False,
         "svg.fonttype": "none"}


def _save(fig, folder, name):
    for ext in ("png", "svg"):
        fig.savefig(folder / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix(path):
    return pd.read_csv(path, index_col=0)


def plot_rdm(output_dir):
    """Keep the response-heatmap palette; the notebook-color exception is for points."""
    out = Path(output_dir)
    tables, folder = out / "tables", out / "figures"
    ids = pd.read_csv(tables / "figure_row_order.csv").strain.tolist()
    neural = _matrix(tables / "rdm_filtered.csv").loc[ids, ids].to_numpy(float)
    chemical = _matrix(tables / "rdm_chemical.csv").loc[ids, ids].to_numpy(float)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.7, 5.2), layout="constrained")
        for ax, values, title, upper, label in (
                (axes[0], neural, "Neural RDM", 2., "1 − cosine"),
                (axes[1], chemical, "Chemical RDM", float(np.nanmax(chemical)), "RMS Δlog₂FC")):
            im = ax.imshow(values, cmap=cmap, vmin=0, vmax=upper, interpolation="nearest")
            ax.set(title=title, xlabel="Sample", ylabel="Sample", xticks=[], yticks=[])
            fig.colorbar(im, ax=ax, shrink=.78, label=label)
        tri = np.triu_indices(len(ids), 1)
        xx, yy = chemical[tri], neural[tri]
        valid = np.isfinite(xx) & np.isfinite(yy)
        im = axes[2].hexbin(xx[valid], yy[valid], gridsize=27, mincnt=1,
                            cmap="RdBu_r", linewidths=0)
        axes[2].set(xlabel="Chemical RMS Δlog₂FC", ylabel="Neural 1 − cosine",
                    ylim=(0, 2), title="Matched sample pairs")
        fig.colorbar(im, ax=axes[2], shrink=.78, label="Pairs per bin")
        _save(fig, folder, "03_neural_chemical_rdm")
    caption = (f"03_neural_chemical_rdm: Both matrices use the same {len(ids)} samples and ordering. "
               "The RdBu_r palette is retained; neural distances span 0–2 and chemical distances use 0–observed maximum. "
               "Neural distance is 1 − cosine of individual-SNR 0.5 template amplitudes over 0–40 s; "
               "chemical distance is RMS difference across the existing 380 log₂FC features. "
               f"The hexbin contains {int(valid.sum())} finite pairs. Pair sharing prevents treating these "
               "as independent biological replicates. Gray denotes unavailable distances.")
    return dict(filename="03_neural_chemical_rdm", cmap="RdBu_r", neural_limits=[0, 2],
                chemical_limits=[0, float(np.nanmax(chemical))], n_samples=len(ids),
                n_pairs=int(valid.sum()), caption=caption)


def _checked_inputs(out, dimension, scope="all"):
    if scope not in ("all", "bootstrap_subset"):
        raise ValueError("Scope must be all or bootstrap_subset")
    base_2d = out / "hmds_full106/2d" if scope == "all" else out / "hmds"
    base_3d = out / "hmds_full106/3d" if scope == "all" else out / "hmds3d"
    if dimension == 2:
        root = base_2d
        coordinate_paths = {d: root / f"{d}_coordinates.csv" for d in ("neural", "chemical")}
        pair_paths = {"neural": root / "neural/fitted_pairs.csv",
                      "chemical": root / "chemical/fitted_pairs_2d.csv"}
    elif dimension == 3:
        root = base_3d
        coordinate_paths = {d: root / d / "coordinates.csv" for d in ("neural", "chemical")}
        pair_paths = {d: root / d / "pairs.csv" for d in ("neural", "chemical")}
    else:
        raise ValueError("HMDS display must be 2D or 3D")
    result = json.loads((root / "result.json").read_text())
    for domain in ("neural", "chemical"):
        if not result[domain].get("converged", False):
            raise ValueError(f"{domain} {dimension}D HMDS lacks checked convergence")
    coordinates = {d: _matrix(p) for d, p in coordinate_paths.items()}
    ids = _matrix(base_2d / "neural_coordinates.csv").index
    if scope == "all" and set(ids) != set(_matrix(out / "tables/rdm_filtered.csv").index):
        raise ValueError("Full-sample embedding must include every sample in the neural RDM")
    if not coordinates["neural"].index.equals(ids):
        raise ValueError("Embedding dimensions must retain the same neural sample order")
    cols = ["x", "y", "z"][:dimension]
    for domain, frame in coordinates.items():
        if not frame.index.is_unique or not set(ids) <= set(frame.index):
            raise ValueError(f"{domain} coordinates do not cover the matched sample set")
        coordinates[domain] = frame.loc[ids, cols]
        values = coordinates[domain].to_numpy(float)
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) > 1 + 1e-12):
            raise ValueError("Invalid native Poincare coordinates")
    pairs = {}
    for domain, path in pair_paths.items():
        frame = pd.read_csv(path)
        frame = frame.loc[frame.sample_i.isin(ids) & frame.sample_j.isin(ids)].copy()
        keys = [tuple(sorted(p)) for p in zip(frame.sample_i, frame.sample_j)]
        if len(keys) != len(set(keys)) or any(a == b for a, b in keys):
            raise ValueError("Duplicate or diagonal Shepard pairs")
        if scope == "all" and len(keys) != len(ids)*(len(ids)-1)//2:
            raise ValueError("The full-sample Shepard diagram must include every observed pair")
        values = frame[["input_distance", "fitted_distance"]].to_numpy(float)
        if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid Shepard distances")
        # Both dimensions must show precisely the same input pairs in each domain.
        reference_path = base_2d / ("neural/fitted_pairs.csv" if domain == "neural"
                                    else "chemical/fitted_pairs_2d.csv")
        reference = pd.read_csv(reference_path)
        reference = reference.loc[reference.sample_i.isin(ids) & reference.sample_j.isin(ids)]
        ref = {tuple(sorted((r.sample_i, r.sample_j))): r.input_distance
               for r in reference.itertuples()}
        actual = dict(zip(keys, frame.input_distance))
        if set(ref) != set(actual) or not all(np.isclose(ref[k], actual[k], rtol=1e-12, atol=1e-12) for k in ref):
            raise ValueError("Shepard pair support or input distance changed between dimensions")
        pairs[domain] = frame
    return coordinates, pairs, {str(p): _hash(p) for p in list(coordinate_paths.values()) + list(pair_paths.values())}, result


PROJECTION_VIEWS = (("x", "y"), ("x", "z"), ("y", "z"))


def _orthographic_views(fig, grid, row, domain, frame, scatter_args):
    """Project the same saved 3D coordinates; do not rotate, scale or refit."""
    axes = []
    for column, (horizontal, vertical) in enumerate(PROJECTION_VIEWS):
        ax = fig.add_subplot(grid[row, column])
        ax.add_patch(Circle((0, 0), 1, fill=False, color="#7b8994", lw=.8))
        ax.axhline(0, color="#e1e5e8", lw=.6, zorder=0)
        ax.axvline(0, color="#e1e5e8", lw=.6, zorder=0)
        ax.scatter(frame[horizontal], frame[vertical], **scatter_args)
        ax.set(xlim=(-1.04, 1.04), ylim=(-1.04, 1.04), aspect="equal",
               xticks=[], yticks=[], xlabel=horizontal.upper(), ylabel=vertical.upper())
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title(f"{horizontal.upper()}{vertical.upper()}", fontsize=14, pad=12)
        if column == 0:
            ax.text(-.22, .5, domain.title(), transform=ax.transAxes,
                    rotation=90, ha="center", va="center", fontsize=15, fontweight="bold")
        axes.append(ax)
    return axes


def plot_hmds_dimension(output_dir, dimension, scope="all"):
    """One dimensionality per figure: matched embeddings and their Shepard plots."""
    out = Path(output_dir)
    coordinates, pairs, hashes, fit_result = _checked_inputs(out, dimension, scope=scope)
    ids = coordinates["neural"].index
    color_fit_root = out / "hmds_full106/2d" if scope == "all" else out / "hmds"
    score = _matrix(color_fit_root / "sample_colors.csv").chemical_PCo1
    color_source = Path(fit_result["chemical"]["source_directory"]) / "color_reference"
    color_table_path = color_source / "aid_to_chemical_color.csv"
    color_parameters_path = color_source / "color_parameters.json"
    color_table = _matrix(color_table_path)
    color_parameters = json.loads(color_parameters_path.read_text())
    if (not color_table.index.is_unique or set(color_table.index) != set(score.index)
            or not np.allclose(color_table.loc[score.index, "chemical_PCo1"], score, rtol=1e-12, atol=1e-12)):
        raise ValueError("Notebook point colors do not match the frozen chemical reference")
    norm = Normalize(vmin=color_parameters["vmin"], vmax=color_parameters["vmax"])
    point_cmap = color_parameters["cmap_name"]
    continuous_cmap = LinearSegmentedColormap.from_list(
        f"{point_cmap}_continuous", plt.get_cmap(point_cmap)(np.linspace(0, 1, 256)), N=4096)
    reconstructed_hex = [to_hex(c) for c in continuous_cmap(norm(color_table.chemical_PCo1))]
    if reconstructed_hex != color_table.color_hex.tolist():
        raise ValueError("Notebook colorbar recipe differs from the saved point colors")
    colorbar_mappable = ScalarMappable(norm=norm, cmap=continuous_cmap)
    hashes.update({str(p): _hash(p) for p in (color_table_path, color_parameters_path)})
    diagnostics = {}
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(17.2, 8.4) if dimension == 3 else (12.8, 10.2),
                         layout="constrained")
        grid = (fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.08]) if dimension == 3
                else fig.add_gridspec(2, 2, height_ratios=[1.18, 1]))
        embedding_axes = []
        for position, domain in enumerate(("neural", "chemical")):
            frame = coordinates[domain]
            scatter_args = dict(c=color_table.loc[ids, "color_hex"].tolist(),
                                s=31, edgecolors="white", linewidths=.4)
            if dimension == 2:
                ax = fig.add_subplot(grid[0, position])
                embedding_axes.append(ax)
                ax.add_patch(Circle((0, 0), 1, fill=False, color="#7b8994", lw=.8))
                ax.scatter(frame.x, frame.y, **scatter_args)
                ax.set(xlim=(-1.04, 1.04), ylim=(-1.04, 1.04), aspect="equal")
                ax.set_axis_off()
                ax.set_title(f"{domain.title()} HMDS · {dimension}D", fontsize=14)
                shepard = fig.add_subplot(grid[1, position])
            else:
                embedding_axes.extend(_orthographic_views(
                    fig, grid, position, domain, frame, scatter_args))
                shepard = fig.add_subplot(grid[position, 3])
            x, y = pairs[domain][["input_distance", "fitted_distance"]].to_numpy().T
            maximum = float(max(x.max(), y.max()) * 1.04)
            shepard.scatter(x, y, s=6, c="#32658b", alpha=.19, linewidths=0, rasterized=True)
            shepard.plot([0, maximum], [0, maximum], color="#444444", lw=1, linestyle="--")
            unit = "chord distance" if domain == "neural" else "RMS Δlog₂FC"
            relative_rmse = float(np.linalg.norm(y-x) / np.linalg.norm(x))
            shepard.set(xlim=(0, maximum), ylim=(0, maximum), aspect="equal",
                        xlabel=f"Input {unit}", ylabel=f"Fitted {unit}",
                        title=f"Shepard · relative RMSE {relative_rmse:.3f}")
            shepard.text(.04, .96, f"{len(x):,} pairs", transform=shepard.transAxes,
                          va="top", fontsize=10, color="#555555")
            fit_diagnostics = fit_result[domain]["diagnostics"]
            fixed_lambda = fit_diagnostics.get("fixed_lambda")
            if fixed_lambda is not None and not np.isfinite(fixed_lambda):
                fixed_lambda = None
            diagnostics[domain] = dict(n_displayed_pairs=len(x), displayed_relative_rmse=relative_rmse,
                                       lambda_value=fit_diagnostics["lambda_value"],
                                       fixed_lambda=fixed_lambda, lambda_estimated=fixed_lambda is None)
        if dimension == 3:
            fig.colorbar(colorbar_mappable, ax=embedding_axes, orientation="horizontal",
                         shrink=.52, aspect=45, pad=.05, label="Chemical PCo1")
        else:
            fig.colorbar(colorbar_mappable, ax=embedding_axes, shrink=.72, pad=.035,
                         label="Chemical PCo1")
        view_label = " · orthographic views" if dimension == 3 else ""
        fig.suptitle(f"{dimension}D HMDS{view_label} · {len(ids)} matched samples", fontsize=17)
        name = f"03{'b' if dimension == 2 else 'c'}_neural_chemical_hmds_{dimension}d"
        _save(fig, out / "figures", name)
    coverage_note = (
        f"All {len(ids)} samples and all {len(pairs['neural'])} observed neural pairs are fitted. "
        "The previous 80% bootstrap-coverage exclusion is removed; coverage remains a quality record. "
        "Original neural distances and bootstrap variances are reused without imputation or resampling. "
        "Variance is conditional on valid fixed-support draws, and lower-coverage pairs have less secure uncertainty estimates. "
        if scope == "all" else
        f"The {len(ids)} samples satisfy the earlier 80% bootstrap-coverage graph rule. ")
    layout_note = ("Rows: neural and chemical. Columns 1–3: XY, XZ and YZ orthographic views "
                   "of the same fitted 3D Poincaré coordinates, not separate 2D fits. "
                   "All six projections have equal aspect and limits −1.04 to 1.04; circles mark "
                   "the projected unit-ball boundary. No camera selection or coordinate rotation is applied. "
                   if dimension == 3 else "Top: 2D Poincaré disk displays. ")
    caption = (f"{name}: " + layout_note
               + coverage_note +
               "Coordinates are native isometrically centered fits, with no Euclidean shrinkage; "
               "Point colors exactly reuse notebook 03's saved per-sample color_hex mapping: turbo, "
               "with linear min–max normalization over the full fixed 106-sample chemical PCo1 reference. "
               "No subset rescaling, zero centering or new color grouping is applied. "
               + ("Right: " if dimension == 3 else "Bottom: ") +
               "input versus fitted distances for pairs within the displayed sample set; "
               f"the dashed line is identity. Neural uses {len(pairs['neural'])} observed pairs; chemical uses "
               f"all {len(pairs['chemical'])} pairs among these {len(ids)} samples, taken from its checked full-reference fit. "
               "The same pairs are used in both dimensionalities. Relative RMSE is recomputed on the shown "
               "pairs as norm(fitted − input) / norm(input). Fitted distances are restored to input units. "
               "Neural variance values and SNR 0.5 inputs are unchanged. Orientation is arbitrary and coordinates "
               "across domains cannot be directly compared; radius is not a biological hierarchy. "
               + ("Orientations are not aligned across domains. Distances between projected points are not "
                  "hyperbolic distances; Shepard plots use unchanged full-3D fitted distances. "
                  if dimension == 3 else "")
               + "Chemical 2D fixes λ = 10, whereas chemical 3D estimates λ; their error difference is not "
               "a pure comparison of dimension with both scale parameters freely fitted. Neural λ is estimated "
               "in both dimensions. Coverage and gradient checks are recorded with the fits.")
    return dict(filename=name, dimension=dimension, n_samples=len(ids), scope=scope,
                sample_ids=ids.tolist(), color_cmap=point_cmap,
                color_limits=[norm.vmin, norm.vmax], color_normalization="linear full-reference min-max",
                color_interpolation="Notebook continuous palette: 256 turbo knots, 4096 levels",
                camera_views=None, geometry_only_view_selection=False,
                projection_views=[list(view) for view in PROJECTION_VIEWS] if dimension == 3 else None,
                projection_limits=[-1.04, 1.04] if dimension == 3 else None,
                layout="domains by rows; XY, XZ, YZ, Shepard by columns" if dimension == 3 else "2x2",
                point_colors=color_table.loc[ids, "color_hex"].to_dict(),
                shepard=diagnostics, source_sha256=hashes, caption=caption)


def plot_comparisons(output_dir, dimensions=(2, 3), scope="all"):
    out = Path(output_dir)
    outputs = [plot_rdm(out)] + [plot_hmds_dimension(out, d, scope=scope) for d in dimensions]
    (out / "figures/comparison_captions.txt").write_text("\n\n".join(item["caption"] for item in outputs)+"\n")
    result = dict(figures=outputs, code_sha256=_hash(__file__), response_analysis_rerun=False)
    (out / "figures/comparison_display_parameters.json").write_text(json.dumps(result, indent=2)+"\n")
    return result
