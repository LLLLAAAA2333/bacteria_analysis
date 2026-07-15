"""Circular dendrograms with polished styling.

Two outputs:
  1. circular_dendrogram_metabolite.png — 106 strains, Ward+Euclidean on log2(FC)
  2. circular_dendrogram_taxonomy.png — 299 strains, Newick tree
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Patch, Wedge
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bacteria_analysis._data_loaders import enrich_neural_dataframe

OUTPUT_DIR = Path("results/genus_cluster")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# colour palettes
# ---------------------------------------------------------------------------

PHYLUM_COLORS = {
    "Bacillota":        "#d62728",
    "Bacteroidota":     "#1f77b4",
    "Pseudomonadota":   "#2ca02c",
    "Actinomycetota":   "#ff7f0e",
    "Fusobacteriota":   "#9467bd",
    "Verrucomicrobiota": "#17becf",
}

_GENUS_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20))
_GENUS_TAB20B = plt.cm.tab20b(np.linspace(0, 1, 20))


def _build_genus_palette(genera: list[str]) -> dict[str, tuple]:
    all_colors = np.vstack([_GENUS_TAB20, _GENUS_TAB20B])
    return {g: all_colors[i % len(all_colors)] for i, g in enumerate(genera)}


# ---------------------------------------------------------------------------
# Newick parser (same as before)
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("name", "length", "children", "_angle", "_depth")
    def __init__(self):
        self.name = None
        self.length = 0.0
        self.children: list[Node] = []
        self._angle = 0.0
        self._depth = 0.0


def parse_newick(text: str) -> Node:
    text = text.strip().rstrip(";")
    root = Node()
    _parse_node(text, 0, root)
    return root


def _parse_node(text, pos, node):
    if text[pos] == "(":
        pos += 1
        while pos < len(text):
            child = Node()
            pos = _parse_node(text, pos, child)
            node.children.append(child)
            if pos >= len(text) or text[pos] == ")":
                break
            if text[pos] == ",":
                pos += 1
        pos += 1
        pos = _parse_label_len(text, pos, node)
    else:
        pos = _parse_label_len(text, pos, node)
    return pos


def _parse_label_len(text, pos, node):
    if pos < len(text) and text[pos] not in (":", ",", ")", ";"):
        end = pos
        while end < len(text) and text[end] not in (":", ",", ")", ";"):
            end += 1
        node.name = text[pos:end]
        pos = end
    if pos < len(text) and text[pos] == ":":
        pos += 1
        end = pos
        while end < len(text) and text[end] not in (",", ")", ";"):
            end += 1
        node.length = float(text[pos:end])
        pos = end
    return pos


def count_leaves(node):
    if not node.children:
        return 1
    return sum(count_leaves(c) for c in node.children)


# ---------------------------------------------------------------------------
# circular layout (shared between Newick and Ward)
# ---------------------------------------------------------------------------

def _tree_layout_newick(root, n_total):
    """Walk Newick tree; return (leaf_angles, leaf_depths, internal_nodes)."""
    leaf_angles: dict[str, float] = {}
    leaf_depths: dict[str, float] = {}
    internal: list[dict] = []

    def walk(node, parent_depth, counter):
        node._depth = parent_depth + node.length
        if not node.children:
            idx = counter[0]; counter[0] += 1
            node._angle = 2 * np.pi * idx / n_total
            leaf_angles[node.name] = node._angle
            leaf_depths[node.name] = node._depth
        else:
            for c in node.children:
                walk(c, node._depth, counter)
            xs = sum(np.cos(c._angle) for c in node.children)
            ys = sum(np.sin(c._angle) for c in node.children)
            node._angle = np.arctan2(ys, xs)
            internal.append({
                "depth": node._depth,
                "angle": node._angle,
                "child_angles": [c._angle for c in node.children],
                "child_depths": [c._depth for c in node.children],
            })

    walk(root, 0.0, [0])
    return leaf_angles, leaf_depths, internal


def _tree_layout_ward(Z, n):
    """Convert scipy Ward linkage to internal-node list."""
    node_ang = np.zeros(2 * n - 1)
    node_dep = np.zeros(2 * n - 1)
    leaf_order = leaves_list(Z)
    for i in range(n):
        node_ang[i] = float(leaf_order[i])
    internal = []
    for i, (a, b, dist, _) in enumerate(Z):
        ni = n + i; a, b = int(a), int(b)
        node_ang[ni] = (node_ang[a] + node_ang[b]) / 2.0
        node_dep[ni] = dist
        internal.append({
            "depth": dist,
            "angle": 2 * np.pi * node_ang[ni] / n,
            "child_angles": [2 * np.pi * node_ang[a] / n,
                             2 * np.pi * node_ang[b] / n],
            "child_depths": [node_dep[a], node_dep[b]],
        })
    return leaf_order, node_ang[:n], node_dep.max(), internal


# ---------------------------------------------------------------------------
# drawing engine
# ---------------------------------------------------------------------------

def _draw_circular_tree(ax, internal, max_depth, n_leaves,
                        leaf_labels, leaf_colors, label_size,
                        color_ring_groups=None):
    """
    ax : polar Axes
    internal : list of {"depth","angle","child_angles","child_depths"}
    leaf_labels / leaf_colors : aligned to leaf angular order (0..2pi)
    color_ring_groups : optional list-of-lists for outer color strips
    """
    toff = np.pi / 2  # start from top

    ax.set_ylim(0, max_depth * 1.45)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.set_facecolor("white")

    # ---- branch lines ----
    lw_base = 1.0
    lw_min = 0.15
    segments_radial = []
    segments_arc = []

    for nd in internal:
        rp = nd["depth"]
        tp = nd["angle"] - toff
        frac = 1.0 - rp / (max_depth + 1e-6)
        lw = lw_min + (lw_base - lw_min) * frac * frac  # quadratic taper

        for tc, rc in zip(nd["child_angles"], nd["child_depths"]):
            tc = tc - toff
            # radial segment (child -> parent)
            segments_radial.append(([tc, tc], [rc, rp], lw))
            # arc segment at parent level
            t0, t1 = min(tc, tp), max(tc, tp)
            theta_arc = np.linspace(t0, t1, max(8, int(abs(t1 - t0) / (2 * np.pi / n_leaves) * 3)))
            segments_arc.append((theta_arc, np.full_like(theta_arc, rp), lw))

    # Draw arcs as LineCollection segments
    all_seg_colors = ["#555555"] * (len(segments_radial) + len(segments_arc))

    # radial lines
    for (xs, ys, lw) in segments_radial:
        ax.plot(xs, ys, color="#555555", lw=lw, solid_capstyle="round", alpha=0.7)

    # arcs
    for (xs, ys, lw) in segments_arc:
        ax.plot(xs, ys, color="#555555", lw=lw, solid_capstyle="round", alpha=0.7)

    # ---- color rings (outer bands) ----
    if color_ring_groups is not None:
        ring_start = max_depth * 1.04
        ring_height = max_depth * 0.018
        for group_idx, (indices, color) in enumerate(color_ring_groups):
            r = ring_start + group_idx * ring_height * 1.5
            for idx in indices:
                theta0 = 2 * np.pi * idx / n_leaves - toff
                theta1 = 2 * np.pi * (idx + 1) / n_leaves - toff
                wedge = Wedge((0, 0), r + ring_height, np.degrees(theta0),
                              np.degrees(theta1), width=ring_height,
                              facecolor=color, edgecolor="none",
                              alpha=0.9, transform=ax.transData._b)
                ax.add_patch(wedge)

    # ---- leaf labels ----
    for i in range(n_leaves):
        theta = 2 * np.pi * i / n_leaves - toff
        color = leaf_colors[i]
        label = leaf_labels[i]

        rot = np.degrees(theta) % 360
        if 90 < rot < 270:
            rot -= 180
            ha = "right"
        else:
            ha = "left"

        r_label = max_depth * 1.08
        if color_ring_groups is not None:
            r_label += max_depth * 0.022 * len(color_ring_groups)

        ax.annotate(label, xy=(theta, r_label), fontsize=label_size,
                    color="#333333", ha=ha, va="center",
                    rotation=rot, rotation_mode="anchor")

        # tiny tick mark at leaf tip
        ax.plot([theta, theta], [max_depth, max_depth * 1.015],
                color=color, lw=0.5, alpha=0.7)


# ---------------------------------------------------------------------------
# figure A: metabolite tree (106 strains)
# ---------------------------------------------------------------------------

def make_metabolite_tree():
    met_df = pd.read_parquet("results/metabolite_modules/metabolites_reduced.parquet")
    met_log2 = np.log2(met_df.values.astype(float))
    all_aids = list(met_df.index)

    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()
    stim_info = stim_info.drop_duplicates(subset="aid").set_index("aid")
    neural_aids = set(stim_info.index)

    mask = [a in neural_aids for a in all_aids]
    aids_106 = [a for a, m in zip(all_aids, mask) if m]
    met_106 = met_log2[mask]

    # ---- cluster ----
    dist = pdist(met_106, metric="euclidean")
    Z = linkage(dist, method="ward")
    n = len(aids_106)
    leaf_order, _, max_depth, internal = _tree_layout_ward(Z, n)

    # ---- labels & colours ----
    genera = sorted(set(
        stim_info.loc[a, "genus"] if a in stim_info.index else "unknown"
        for a in aids_106
    ))
    genus_pal = _build_genus_palette(genera)

    # species numbering
    sp_count: dict[str, int] = {}
    sp_seen: dict[str, int] = {}

    leaf_labels = []
    leaf_colors = []
    genus_indices = {g: [] for g in genera}  # for color ring

    for i_leaf in range(n):
        idx = leaf_order[i_leaf]
        aid = aids_106[idx]
        sp = stim_info.loc[aid, "species"] if aid in stim_info.index else "unknown"
        genus = stim_info.loc[aid, "genus"] if aid in stim_info.index else "unknown"
        sp_count[sp] = sp_count.get(sp, 0) + 1

    sp_seen = {}
    for i_leaf in range(n):
        idx = leaf_order[i_leaf]
        aid = aids_106[idx]
        sp = stim_info.loc[aid, "species"] if aid in stim_info.index else "unknown"
        genus = stim_info.loc[aid, "genus"] if aid in stim_info.index else "unknown"
        sp_seen[sp] = sp_seen.get(sp, 0) + 1

        leaf_labels.append(f"{_short_sp(sp)} st{sp_seen[sp]} ({aid})")
        leaf_colors.append(genus_pal.get(genus, (0.5, 0.5, 0.5, 1.0)))
        genus_indices[genus].append(i_leaf)

    # ---- color ring by genus ----
    color_ring = [(genus_indices[g], genus_pal[g]) for g in genera
                  if len(genus_indices[g]) > 0]

    # ---- draw ----
    fig, ax = plt.subplots(figsize=(24, 24), subplot_kw={"projection": "polar"})
    _draw_circular_tree(ax, internal, max_depth, n,
                        leaf_labels, leaf_colors, label_size=5.5,
                        color_ring_groups=color_ring)

    # genus legend
    els = [Patch(facecolor=genus_pal[g], label=g, edgecolor="none")
           for g in genera]
    fig.legend(handles=els, fontsize=7.5, loc="upper left",
               bbox_to_anchor=(0.01, 0.99), ncol=2, frameon=True,
               title="Genus", title_fontsize=9, facecolor="white",
               edgecolor="#cccccc")

    fig.suptitle("106 bacterial strains — metabolic similarity\n"
                 "Ward linkage · Euclidean distance · log2(fold change)",
                 fontsize=14, y=0.97, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "circular_dendrogram_metabolite.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {OUTPUT_DIR / 'circular_dendrogram_metabolite.png'}")


def _short_sp(name):
    parts = name.split(" ", 1)
    return f"{parts[0][0]}. {parts[1]}" if len(parts) == 2 else name


# ---------------------------------------------------------------------------
# figure B: taxonomy tree (299 strains, Newick)
# ---------------------------------------------------------------------------

def make_taxonomy_tree():
    tree_text = Path("data/taxonomy_dendrogram_tree_foldchange_log2+1.txt").read_text()
    root = parse_newick(tree_text)
    n_total = count_leaves(root)
    print(f"Taxonomy tree: {n_total} leaves")

    leaf_angles, leaf_depths, internal = _tree_layout_newick(root, n_total)
    max_depth = max(leaf_depths.values())

    # Sort leaves by angle
    sorted_items = sorted(leaf_angles.items(), key=lambda x: x[1])

    leaf_labels = []
    leaf_colors = []
    phylum_indices = {p: [] for p in PHYLUM_COLORS}

    for i, (name, angle) in enumerate(sorted_items):
        phylum = name.split("_")[0] if "_" in name else "Unknown"
        remainder = name.split("_", 1)[1] if "_" in name else name
        m = re.match(r"(.+?)_(strain\d+)", remainder)
        if m:
            label = f"{m.group(1).replace('_', ' ')} {m.group(2)}"
        else:
            label = remainder.replace("_", " ")

        leaf_labels.append(label)
        c = PHYLUM_COLORS.get(phylum, "#888888")
        leaf_colors.append(c)
        if phylum in phylum_indices:
            phylum_indices[phylum].append(i)

    # color ring
    color_ring = []
    for p, indices in phylum_indices.items():
        if indices:
            color_ring.append((indices, PHYLUM_COLORS[p]))
    # Sort rings by phylum order
    color_ring.sort(key=lambda x: list(PHYLUM_COLORS.keys()).index(
        [k for k, v in phylum_indices.items() if v == x[0]][0]))

    # draw
    fig, ax = plt.subplots(figsize=(26, 26), subplot_kw={"projection": "polar"})
    _draw_circular_tree(ax, internal, max_depth, n_total,
                        leaf_labels, leaf_colors, label_size=3.6,
                        color_ring_groups=color_ring)

    # phylum legend
    els = [Patch(facecolor=c, label=p, edgecolor="none")
           for p, c in PHYLUM_COLORS.items()]
    fig.legend(handles=els, fontsize=8, loc="upper left",
               bbox_to_anchor=(0.01, 0.99), ncol=1, frameon=True,
               title="Phylum", title_fontsize=9, facecolor="white",
               edgecolor="#cccccc")

    fig.suptitle(f"{n_total} bacterial strains — taxonomy tree (log2 FC+1)",
                 fontsize=14, y=0.97, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "circular_dendrogram_taxonomy.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {OUTPUT_DIR / 'circular_dendrogram_taxonomy.png'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    make_metabolite_tree()
    make_taxonomy_tree()
