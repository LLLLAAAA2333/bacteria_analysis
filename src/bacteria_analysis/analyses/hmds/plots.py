"""HMDS visualisation: Shepard diagrams, Poincaré disk (2D) and ball (3D).

All plot functions accept generic ``labels: list[str]`` and
``colors: list[str]`` (hex strings) — they are not coupled to any
specific taxonomy, species table, or dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bacteria_analysis.analyses.hmds._distance import upper_triangle
from bacteria_analysis.analyses.hmds._result import EmbeddingResult


# ---------------------------------------------------------------------------
# Shepard diagram
# ---------------------------------------------------------------------------

def plot_shepard(
    result: EmbeddingResult,
    original_distance: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Shepard diagram",
    dpi: int = 200,
) -> None:
    """Shepard diagram: input distance vs. embedded predicted distance.

    Parameters
    ----------
    result : EmbeddingResult
        HMDS result (uses ``predicted_distance``).
    original_distance : (n, n) ndarray
        Input distance matrix.
    output_path : str or Path
        Output PNG path.
    title : str
        Plot title.
    dpi : int
        Output resolution.
    """
    orig = upper_triangle(original_distance)
    pred = upper_triangle(result.predicted_distance)
    limit = float(max(orig.max(), pred.max()) * 1.04)
    rho = result.metrics.get("distance_spearman", float("nan"))
    stress = result.metrics.get("normalized_raw_stress", float("nan"))

    fig, ax = plt.subplots(figsize=(6.0, 5.5), constrained_layout=True)
    ax.scatter(orig, pred, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit], [0, limit], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Input distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"{title}\nρ={rho:.3f}  stress={stress:.3f}", fontsize=10,
    )
    ax.grid(True, color="#e5e7eb", lw=0.4)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2D Poincaré disk (static PNG)
# ---------------------------------------------------------------------------

def plot_poincare_2d(
    coords: np.ndarray,
    labels: list[str],
    colors: list[str],
    output_path: str | Path,
    *,
    title: str = "HMDS 2D",
    rho: float | None = None,
    stress: float | None = None,
    dpi: int = 200,
) -> None:
    """Static 2D Poincaré disk with AID labels.

    Parameters
    ----------
    coords : (n, 2) ndarray
        Poincaré coordinates.
    labels : list of str
        Point labels (e.g. AID strings).
    colors : list of str
        Hex colour strings, one per point.
    output_path : str or Path
        Output PNG path.
    title : str
        Plot title.
    rho : float or None
        If given, annotated in the subtitle.
    stress : float or None
        If given, annotated in the subtitle.
    dpi : int
        Output resolution.
    """
    _validate_plot_inputs(coords, labels, colors)

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1)
    ax.add_artist(circle)

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=60, alpha=0.90, edgecolors="white", linewidths=0.6,
    )
    for i, label in enumerate(labels):
        ax.annotate(
            str(label), (coords[i, 0], coords[i, 1]),
            fontsize=4, alpha=0.55, ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 3),
        )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("Poincaré 1", fontsize=9)
    ax.set_ylabel("Poincaré 2", fontsize=9)
    ax.grid(True, color="#e5e7eb", linewidth=0.4, alpha=0.5)

    subtitle_parts = []
    if rho is not None:
        subtitle_parts.append(f"ρ={rho:.3f}")
    if stress is not None:
        subtitle_parts.append(f"stress={stress:.3f}")
    subtitle = "  ".join(subtitle_parts)
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=10)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2D Poincaré disk (interactive HTML via plotly)
# ---------------------------------------------------------------------------

def plot_poincare_2d_html(
    coords: np.ndarray,
    labels: list[str],
    colors: list[str],
    output_path: str | Path,
    *,
    title: str = "HMDS 2D",
    rho: float | None = None,
    stress: float | None = None,
) -> None:
    """Interactive 2D Poincaré disk (plotly).

    Parameters
    ----------
    coords : (n, 2) ndarray
        Poincaré coordinates.
    labels : list of str
        Point labels.
    colors : list of str
        Hex colour strings, one per point.
    output_path : str or Path
        Output HTML path.
    title : str
        Plot title.
    rho : float or None
        If given, shown in subtitle.
    stress : float or None
        If given, shown in subtitle.
    """
    import plotly.graph_objects as go

    _validate_plot_inputs(coords, labels, colors)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers+text",
        marker={
            "size": 8, "color": colors, "opacity": 0.92,
            "line": {"color": "white", "width": 0.5},
        },
        text=[str(lbl) for lbl in labels],
        textposition="top center",
        textfont={"size": 7, "color": "#333333"},
        hovertext=[f"<b>{lbl}</b>" for lbl in labels],
        hoverinfo="text",
        showlegend=False,
    ))

    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines",
        line={"color": "#6b7280", "width": 0.8},
        hoverinfo="skip", showlegend=False,
    ))

    subtitle_parts = []
    if rho is not None:
        subtitle_parts.append(f"ρ={rho:.3f}")
    if stress is not None:
        subtitle_parts.append(f"stress={stress:.3f}")
    subtitle = "  ".join(subtitle_parts)

    fig.update_layout(
        title={
            "text": f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            "font": {"size": 14},
        },
        xaxis={"range": [-1.08, 1.08], "constrain": "domain", "title": "Poincaré 1"},
        yaxis={
            "range": [-1.08, 1.08], "scaleanchor": "x", "scaleratio": 1,
            "title": "Poincaré 2",
        },
        width=950, height=850,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# 3D Poincaré ball (interactive HTML via plotly)
# ---------------------------------------------------------------------------

def plot_poincare_3d_html(
    coords: np.ndarray,
    labels: list[str],
    colors: list[str],
    output_path: str | Path,
    *,
    title: str = "HMDS 3D",
    rho: float | None = None,
    stress: float | None = None,
) -> None:
    """Interactive 3D Poincaré ball (plotly).

    Parameters
    ----------
    coords : (n, 3) ndarray
        Poincaré coordinates.
    labels : list of str
        Point labels.
    colors : list of str
        Hex colour strings, one per point.
    output_path : str or Path
        Output HTML path.
    title : str
        Plot title.
    rho : float or None
        If given, shown in subtitle.
    stress : float or None
        If given, shown in subtitle.
    """
    import plotly.graph_objects as go

    _validate_plot_inputs(coords, labels, colors)
    if coords.shape[1] < 3:
        raise ValueError(f"plot_poincare_3d_html requires 3D coords, got shape {coords.shape}")

    fig = go.Figure()

    radii = np.linalg.norm(coords, axis=1)
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text",
        marker={
            "size": 5, "color": colors, "opacity": 0.90,
            "line": {"color": "white", "width": 0.4},
        },
        text=[str(lbl) for lbl in labels],
        textposition="top center",
        textfont={"size": 7, "color": "#444444"},
        hovertext=[
            f"<b>{lbl}</b><br>r={radii[i]:.4f}"
            for i, lbl in enumerate(labels)
        ],
        hoverinfo="text",
        showlegend=False,
    ))

    # wireframe sphere — three great circles
    theta = np.linspace(0, 2 * np.pi, 120)
    zero = np.zeros_like(theta)
    for circle_xs, circle_ys, circle_zs, circle_color in [
        (np.cos(theta), np.sin(theta), zero, "#6b7280"),
        (np.cos(theta), zero, np.sin(theta), "#9ca3af"),
        (zero, np.cos(theta), np.sin(theta), "#9ca3af"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=circle_xs, y=circle_ys, z=circle_zs,
            mode="lines",
            line={"color": circle_color, "width": 0.8},
            hoverinfo="skip", showlegend=False,
        ))

    subtitle_parts = []
    if rho is not None:
        subtitle_parts.append(f"ρ={rho:.3f}")
    if stress is not None:
        subtitle_parts.append(f"stress={stress:.3f}")
    subtitle = "  ".join(subtitle_parts)

    fig.update_layout(
        title={
            "text": f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            "font": {"size": 14},
        },
        scene={
            "xaxis": {"title": "P1", "range": [-1.05, 1.05]},
            "yaxis": {"title": "P2", "range": [-1.05, 1.05]},
            "zaxis": {"title": "P3", "range": [-1.05, 1.05]},
            "aspectmode": "cube",
        },
        width=900, height=850,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _validate_plot_inputs(
    coords: np.ndarray, labels: list[str], colors: list[str],
) -> None:
    n = coords.shape[0]
    if len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != coords {n}")
    if len(colors) != n:
        raise ValueError(f"colors length {len(colors)} != coords {n}")
