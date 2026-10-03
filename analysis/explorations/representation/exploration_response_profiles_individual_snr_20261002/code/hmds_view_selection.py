"""Choose static HMDS camera views by projected marker visibility only."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


CURRENT = dict(elev=20., azim=35.)
MARKER_SIZE, LINE_WIDTH, DPI = 31., .4, 180


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _camera(ax, elev, azim, wireframe=False):
    if wireframe:
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 21), np.linspace(0, np.pi, 13))
        ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                          color="#c7cdd3", alpha=.28, linewidth=.45, zorder=0)
    ax.set(xlim=(-1.04, 1.04), ylim=(-1.04, 1.04), zlim=(-1.04, 1.04))
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_proj_type("persp", focal_length=1.)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def score_view(ax, coordinates, elev, azim):
    """Sum equal-circle intersection areas, each divided by one marker area.

    The marker includes the white edge. Projection uses the actual Matplotlib
    perspective matrix and the target panel's screen transform, not world-space
    distances. Color and sample identity are not inputs to this function.
    """
    ax.view_init(elev=elev, azim=azim)
    x, y, _ = proj3d.proj_transform(*np.asarray(coordinates).T, ax.get_proj())
    xy = ax.transData.transform(np.column_stack([x, y]))
    diameter = (np.sqrt(MARKER_SIZE) + LINE_WIDTH) * ax.figure.dpi / 72.
    distances = pdist(xy)
    relative = distances / diameter
    intersect = relative < 1.
    t = np.clip(relative[intersect], 0, 1)
    overlap = (2*np.arccos(t) - 2*t*np.sqrt(1-t*t)) / np.pi
    # Soft crowding is only a deterministic tie-breaker for equal total overlap.
    soft = np.exp(-relative**2).sum()
    bbox = ax.get_window_extent()
    margins = np.column_stack([xy[:, 0]-bbox.x0, bbox.x1-xy[:, 0],
                               xy[:, 1]-bbox.y0, bbox.y1-xy[:, 1]])
    return dict(elev=float(elev), azim=float(azim), overlap_penalty=float(overlap.sum()),
                overlapping_pairs=int(intersect.sum()), soft_crowding=float(soft),
                marker_diameter_px=float(diameter),
                clipped_markers=int((margins.min(axis=1) < diameter/2).sum()),
                hull_area_fraction=float(ConvexHull(xy).volume / (bbox.width*bbox.height)),
                rms_spread_px=float(np.sqrt(np.mean(np.sum((xy-xy.mean(axis=0))**2, axis=1)))))


def _view_vector(view):
    e, a = np.radians([view["elev"], view["azim"]])
    return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])


def select_distinct_views(scores, number=3, minimum_degrees=30.):
    """Select low-overlap viewing axes, treating reverse views as the same axis."""
    ranked = sorted(scores, key=lambda r: (r["clipped_markers"], r["overlap_penalty"],
                                          r["soft_crowding"], r["elev"], r["azim"]))
    selected = []
    for row in ranked:
        vector = _view_vector(row)
        separations = [np.degrees(np.arccos(np.clip(abs(vector @ _view_vector(v)), 0, 1)))
                       for v in selected]
        if not separations or min(separations) >= minimum_degrees - 1e-10:
            selected.append(row)
        if len(selected) == number:
            return selected
    raise ValueError("The grid has too few distinct candidate viewing axes")


def run_selection(output_dir):
    """Scan coordinates only; save selected views, all scores and a contact sheet."""
    out = Path(output_dir).resolve()
    figures = out / "figures"
    coordinate_paths = {d: out / f"hmds_full106/3d/{d}/coordinates.csv"
                        for d in ("neural", "chemical")}
    frames = {d: pd.read_csv(p, index_col=0) for d, p in coordinate_paths.items()}
    ids = frames["neural"].index
    if len(ids) != 106 or not ids.is_unique:
        raise ValueError("The requested static views require all 106 unique samples")
    for domain, frame in frames.items():
        if not frame.index.is_unique or set(frame.index) != set(ids):
            raise ValueError("Neural and chemical views must retain the same samples")
        frames[domain] = frame.loc[ids, ["x", "y", "z"]]
        if not np.isfinite(frames[domain].to_numpy()).all():
            raise ValueError("Saved coordinates contain nonfinite values")

    # Read the actual top-panel rectangle from the previous saved full figure.
    # SVG units are points; unlike an approximate viewport this preserves s=31's
    # screen footprint relative to the final publication panel.
    geometry_source = figures / "previous_3d_view/03c_neural_chemical_hmds_3d.svg"
    if not geometry_source.exists():
        geometry_source = figures / "03c_neural_chemical_hmds_3d.svg"
    svg = ET.parse(geometry_source).getroot()
    rects = svg.findall('.//{http://www.w3.org/2000/svg}clipPath/{http://www.w3.org/2000/svg}rect')
    rects = sorted(rects, key=lambda r: (float(r.attrib["y"]), float(r.attrib["x"])))
    top_rects = rects[:2]
    sides = [float(r.attrib["width"]) for r in top_rects]
    if len(sides) != 2 or not np.allclose(sides, sides[0], atol=1e-5):
        raise ValueError("Cannot identify equal-size top-panel viewports in the saved figure")
    side_inches = sides[0] / 72.
    fit_result_path = out / "hmds_full106/3d/result.json"
    fit_result = json.loads(fit_result_path.read_text())
    color_path = Path(fit_result["chemical"]["source_directory"]) / "color_reference/aid_to_chemical_color.csv"
    color_frame = pd.read_csv(color_path, index_col=0)
    if not set(ids) <= set(color_frame.index):
        raise ValueError("Frozen notebook colors do not cover every sample")
    colors = color_frame.loc[ids, "color_hex"].tolist()
    paths = list(coordinate_paths.values()) + [geometry_source, fit_result_path, color_path]
    hashes = {str(p): _hash(p) for p in paths}

    results = {}
    figure = Figure(figsize=(side_inches, side_inches), dpi=DPI)
    FigureCanvasAgg(figure)
    axis = figure.add_axes([0, 0, 1, 1], projection="3d")
    _camera(axis, **CURRENT)
    figure.canvas.draw()
    for domain, frame in frames.items():
        coordinates = frame.to_numpy(float)
        baseline = score_view(axis, coordinates, **CURRENT)
        scores = [score_view(axis, coordinates, elev, azim)
                  for elev in range(-60, 61, 10) for azim in range(0, 360, 10)]
        candidates = select_distinct_views(scores)
        results[domain] = dict(current=baseline, candidates=candidates, grid_scores=scores)
    plt.close(figure)

    # Keep every contact-sheet viewport the same physical size as the scored one.
    sheet = Figure(figsize=(20.4, 10.8), dpi=DPI)
    FigureCanvasAgg(sheet)
    for row, domain in enumerate(("neural", "chemical")):
        candidates = [results[domain]["current"], *results[domain]["candidates"]]
        xyz = frames[domain].to_numpy(float)
        for column, view in enumerate(candidates):
            left_inches = .18 + column * 5.1
            bottom_inches = 5.45 if row == 0 else .18
            ax = sheet.add_axes([left_inches/20.4, bottom_inches/10.8,
                                 side_inches/20.4, side_inches/10.8], projection="3d")
            _camera(ax, view["elev"], view["azim"], wireframe=True)
            ax.scatter(*xyz.T, c=colors, s=MARKER_SIZE, edgecolors="white",
                       linewidths=LINE_WIDTH, depthshade=False)
            label = "current" if column == 0 else f"candidate {column}"
            ax.set_title(f"{domain.title()} · {label}\nElevation {view['elev']:g}° · azimuth {view['azim']:g}°",
                         fontsize=12, pad=0)
    sheet_paths = []
    for suffix in ("png", "svg"):
        path = figures / f"3d_view_candidates.{suffix}"
        sheet.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
        sheet_paths.append(str(path))
    plt.close(sheet)
    if any(_hash(p) != hashes[str(p)] for p in paths):
        raise ValueError("An input changed during the geometry-only view scan")
    result = dict(
        views={d: {k: results[d]["candidates"][0][k] for k in ("elev", "azim")}
               for d in frames},
        selected_by="Geometry only; lowest projected marker overlap on the specified grid",
        objective="Sum pairwise equal-circle intersection areas divided by one marker area; "
                  "first reject clipped markers, then minimize overlap, then soft crowding as tie-breaker",
        soft_crowding_definition="Sum exp(-(projected center distance / marker diameter)^2)",
        geometry=dict(projection="Matplotlib perspective, focal_length=1, default camera distance=10",
                      axis_limits=[-1.04, 1.04], box_aspect=[1, 1, 1], zoom=1.35,
                      viewport_side_points=sides[0], viewport_side_pixels=side_inches*DPI,
                      dpi=DPI, marker_area_points_squared=MARKER_SIZE, marker_edge_points=LINE_WIDTH,
                      marker_diameter_definition="(sqrt(31) + 0.4) points, converted to pixels",
                      projection_method="proj3d.proj_transform with ax.get_proj(), then ax.transData"),
        grid=dict(elevations=list(range(-60, 61, 10)), azimuths=list(range(0, 360, 10)),
                  n_views_per_domain=468),
        candidate_minimum_axis_separation_degrees=30., original_view=CURRENT,
        results=results, input_sha256=hashes, code_sha256=_hash(__file__),
        sample_ids=ids.tolist(), point_colors=dict(zip(ids, colors)),
        colors_used_for_scoring=False, cross_domain_agreement_used_for_scoring=False,
        refit=False, coordinates_changed=False, files=sheet_paths,
        interpretation="This improves static marker visibility only; it does not measure neural "
                       "discriminability, fit quality, biological separation or agreement with chemistry.",
    )
    destination = figures / "hmds_3d_view_selection.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return result
