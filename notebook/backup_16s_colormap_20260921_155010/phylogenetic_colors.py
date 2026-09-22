"""Categorical colors for an explicit partition of the supplied 16S tree."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from Bio import Phylo
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.ticker import FixedFormatter

# Display resolution selected by the investigator; not a taxonomic rank.
PHYLOGENETIC_N_GROUPS = 12


def load_phylogenetic_colors(tree_path=None, n_groups=PHYLOGENETIC_N_GROUPS):
    """Cut the longest internal edges into tip-containing connected components.

    An internal edge has no terminal endpoint. Skip a cut if either resulting
    component would contain no tips. Terminal branches are never cut directly.
    Keep input branch lengths and do not infer a biological root. Groups are
    connected parts of this unrooted tree, not asserted monophyletic taxa.
    Support labels are recorded; they are not used to validate the partition.
    """
    if tree_path is None:
        tree_path = Path(__file__).resolve().parents[1] / 'data/16S.aln.trim.fa.treefile'
    tree_path = Path(tree_path)
    tree = Phylo.read(tree_path, 'newick')
    tips = tree.get_terminals()
    ids = pd.Index([tip.name for tip in tips], name='sample_id')
    if len(ids) < 2 or not ids.is_unique or ids.isna().any():
        raise ValueError('16S tree needs at least two unique, named tips')
    if not isinstance(n_groups, int) or not 1 <= n_groups <= min(20, len(tips)):
        raise ValueError('n_groups must be an integer from 1 to min(20, number of tips)')
    for clade in tree.find_clades():
        if clade is not tree.root and (
            clade.branch_length is None or not np.isfinite(clade.branch_length)
            or clade.branch_length < 0
        ):
            raise ValueError('16S tree needs finite, nonnegative branch lengths')

    nodes = list(tree.find_clades())
    adjacency = {node: {} for node in nodes}
    edges = []
    for parent in nodes:
        for child in parent.clades:
            adjacency[parent][child] = adjacency[child][parent] = child.branch_length
            if not child.is_terminal():
                edges.append((parent, child))

    def component(start):
        seen, stack = {start}, [start]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return seen

    cut_edges = []
    for parent, child in sorted(edges, key=lambda edge: -edge[1].branch_length):
        if len(cut_edges) == n_groups-1:
            break
        length = adjacency[parent].pop(child)
        adjacency[child].pop(parent)
        sides = (component(parent), component(child))
        if all(any(node.is_terminal() for node in side) for side in sides):
            cut_edges.append((parent, child))
        else:
            adjacency[parent][child] = adjacency[child][parent] = length
    if len(cut_edges) != n_groups-1:
        raise ValueError('Not enough internal edges to create this many tip-containing groups')
    groups, visited = [], set()
    # First appearance in the original leaf order fixes reproducible group IDs.
    for tip in tips:
        if tip not in visited:
            part = component(tip)
            groups.append(part)
            visited.update(part)

    def diameter(part):
        def farthest(start):
            distances, stack = {start: 0.0}, [start]
            while stack:
                node = stack.pop()
                for neighbor, length in adjacency[node].items():
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + length
                        stack.append(neighbor)
            end = max((node for node in distances if node.is_terminal()), key=distances.get)
            return end, distances[end]
        first = min((node for node in part if node.is_terminal()), key=lambda node: node.name)
        return farthest(farthest(first)[0])[1]

    # Preserve leaf order. Qualitative palette; no numerical ordering of colors.
    palette_indices = list(range(0, 20, 2)) + list(range(1, 20, 2))
    palette = [to_hex(colormaps['tab20'](i)) for i in palette_indices[:n_groups]]
    assignments = pd.Series(index=ids, dtype='object', name='phylogenetic_group')
    records, node_groups = [], {}
    for i, part in enumerate(groups):
        group = f'G{i+1:02d}'
        members = [node.name for node in part if node.is_terminal()]
        assignments.loc[members] = group
        node_groups.update({node: group for node in part})
        records.append(dict(group=group, n_strains=len(members),
                            max_tree_distance=diameter(part), color_hex=palette[i]))
    group_table = pd.DataFrame(records).set_index('group')
    cut_table = pd.DataFrame([
        dict(group_a=node_groups[parent], group_b=node_groups[child],
             branch_length=child.branch_length, input_node_label=child.confidence)
        for parent, child in cut_edges
    ], columns=['group_a', 'group_b', 'branch_length', 'input_node_label'])
    note = ('Colors identify 16S tree groups; color differences are not distances. '
            'Groups are a display partition, not taxonomic ranks.')
    return dict(
        groups=assignments, group_table=group_table, palette=palette,
        tree=tree, node_groups=node_groups, cut_edges=cut_edges, cut_table=cut_table,
        cmap=ListedColormap(palette, name='16S_groups'),
        vmin=-0.5, vmax=n_groups-0.5, n_groups=n_groups, label='16S tree group',
        method='Cut longest internal branches; keep only cuts with at least one tip on both sides',
        rooting='Unrooted connected groups; input root retained for drawing only',
        palette_name='tab20: dark colors then light colors',
        tree_path=str(tree_path.resolve()), tree_sha256=hashlib.sha256(tree_path.read_bytes()).hexdigest(),
        note=note,
    )


def color_values(mapping, sample_ids):
    """Integer category codes are for plotting only, not a phylogenetic score."""
    groups = mapping['groups'].reindex(sample_ids)
    if groups.isna().any():
        raise ValueError(f"Sample IDs missing from 16S tree: {groups.index[groups.isna()].tolist()}")
    return mapping['group_table'].index.get_indexer(groups)


def mpl_colors(mapping, sample_ids):
    n = mapping['n_groups']
    return dict(c=color_values(mapping, sample_ids), cmap=mapping['cmap'],
                norm=BoundaryNorm(np.arange(n+1)-.5, n))


def mpl_colorbar_kwargs(mapping):
    """Discrete category key with group names, never a continuous score bar."""
    return dict(ticks=np.arange(mapping['n_groups']),
                format=FixedFormatter(mapping['group_table'].index.tolist()), label=mapping['label'])


def plotly_colors(mapping, sample_ids):
    n = mapping['n_groups']
    steps = []
    for i, color in enumerate(mapping['palette']):
        steps.extend([[i/n, color], [(i+1)/n, color]])
    return dict(color=color_values(mapping, sample_ids).tolist(), cmin=-.5, cmax=n-.5,
                colorscale=steps, showscale=True,
                colorbar=dict(title=dict(text=mapping['label'], side='right'),
                              tickmode='array', tickvals=list(range(n)),
                              ticktext=mapping['group_table'].index.tolist()))


def color_provenance(mapping):
    return {key: mapping[key] for key in
            ('tree_path', 'tree_sha256', 'n_groups', 'method', 'rooting', 'palette_name', 'note')}
