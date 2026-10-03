"""Draw the categorical 16S tree partition and all actual AID labels."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_phylogenetic_groups(mapping):
    """Return a group overview, full branch-length tree, and AID-to-group table."""
    summary = mapping['group_table']
    colors = summary.color_hex.to_dict()
    table = mapping['groups'].to_frame()
    table['color_hex'] = table.phylogenetic_group.map(colors)
    table['leaf_order'] = np.arange(len(table))
    n = mapping['n_groups']
    overview = plt.figure(figsize=(15, 8.6), facecolor='white')
    grid = overview.add_gridspec(1, 2, width_ratios=[1.2, 1], left=.04, right=.97,
                                 top=.8, bottom=.16, wspace=.12)
    overview.suptitle(f'{n} groups from the 16S tree', x=.04, y=.96,
                      ha='left', fontsize=23, fontweight='bold')
    overview.text(.04, .903, f'Cut the {n-1} longest eligible internal branches. '
                  'Each remaining tip-containing component is one group.', fontsize=12, color='#475569')
    network = overview.add_subplot(grid[0])
    angles = np.linspace(np.pi/2, np.pi/2-2*np.pi, n, endpoint=False)
    positions = {group: np.array([np.cos(angle), np.sin(angle)])
                 for group, angle in zip(summary.index, angles)}
    for cut in mapping['cut_table'].itertuples():
        a, b = positions[cut.group_a], positions[cut.group_b]
        network.plot([a[0], b[0]], [a[1], b[1]], color='#94a3b8', lw=1.4, zorder=1)
    for group, point in positions.items():
        network.add_patch(Circle(point, .07, facecolor=colors[group], edgecolor='white', lw=1, zorder=3))
        network.text(*(point*1.23), group, ha='center', va='center', fontweight='bold', fontsize=10)
    network.set(xlim=(-1.45, 1.45), ylim=(-1.35, 1.4), aspect='equal')
    network.axis('off')
    network.set_title('Group connections through cut branches\nSchematic layout; lengths are not to scale',
                       fontsize=12, loc='left', pad=15)
    info = overview.add_subplot(grid[1])
    info.axis('off')
    info.set(xlim=(0, 1), ylim=(n+.9, -1.6))
    info.text(.02, -1.05, 'Group', weight='bold', fontsize=11)
    info.text(.5, -1.05, 'Strains', weight='bold', fontsize=11, ha='right')
    info.text(.99, -1.05, 'Max. within-group distance', weight='bold', fontsize=10.5, ha='right')
    for i, (group, row) in enumerate(summary.iterrows()):
        if i % 2 == 0:
            info.add_patch(plt.Rectangle((0, i-.4), 1, .85, color='#f1f5f9', lw=0))
        info.scatter(.04, i, s=90, color=colors[group], marker='s')
        info.text(.1, i, group, va='center', fontsize=11)
        info.text(.5, i, str(int(row.n_strains)), ha='right', va='center', fontsize=11)
        info.text(.99, i, f'{row.max_tree_distance:.4f}', ha='right', va='center', fontsize=11)
    overview.text(.04, .083, 'Colors represent membership, not a continuous genetic distance. '
                  'These groups are not asserted genera, species, or monophyletic taxa.', fontsize=11, color='#475569')
    overview.text(.04, .045, 'Group sizes need not be equal. The full tree below shows every AID and the exact cut branches.',
                  fontsize=11, color='#475569')

    tree = mapping['tree']
    tips = tree.get_terminals()
    depths = tree.depths()
    y = {tip: i for i, tip in enumerate(tips)}
    for node in tree.find_clades(order='postorder'):
        if not node.is_terminal():
            y[node] = (y[node.clades[0]] + y[node.clades[-1]]) / 2
    xmax = max(depths.values())
    full, ax = plt.subplots(figsize=(15, max(12, len(tips)*.135)), facecolor='white')
    full.subplots_adjust(left=.055, right=.95, top=.97, bottom=.035)
    ax.set_title(f'Full 16S tree: {len(tips)} actual AIDs, {n} categorical groups\n'
                 'Dashed gray branches are the cuts; all supplied branch lengths are retained.',
                 loc='left', fontsize=15, pad=20)
    cuts = set(mapping['cut_edges'])
    for parent in tree.find_clades():
        color = colors[mapping['node_groups'][parent]]
        if parent.clades:
            ax.plot([depths[parent]]*2, [y[parent.clades[0]], y[parent.clades[-1]]], color=color, lw=.8)
        for child in parent.clades:
            cut = (parent, child) in cuts
            edge_color = '#64748b' if cut else colors[mapping['node_groups'][child]]
            ax.plot([depths[parent], depths[child]], [y[child]]*2,
                    color=edge_color, lw=1 if cut else .8, ls='--' if cut else '-')
            if cut:
                ax.plot((depths[parent]+depths[child])/2, y[child], marker='|', color='#111827', ms=9)
    # Aligned actual AID labels, with leaders from their real terminal positions.
    label_x = xmax*1.025
    for tip in tips:
        group = mapping['groups'].loc[tip.name]
        color = colors[group]
        ax.plot([depths[tip], label_x-.01*xmax], [y[tip]]*2, color='#cbd5e1', lw=.3, zorder=0)
        ax.scatter(label_x, y[tip], marker='s', s=16, color=color, edgecolors='none')
        ax.text(label_x+.014*xmax, y[tip], f'{tip.name}  {group}', va='center', fontsize=8.3, color='#1e293b')
    ax.set(xlim=(-.005*xmax, xmax*1.2), ylim=(len(tips), -1.5), yticks=[],
           xlabel='Path length from the input display root (no evolutionary direction implied)')
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.grid(axis='x', color='#e2e8f0', lw=.5)
    ax.set_axisbelow(True)
    full.text(.055, .014, 'Grouping uses internal tree branches only; neural and chemical embeddings do not enter the partition. '
              'Input node labels and cut lengths are exported separately.', fontsize=10, color='#475569')
    return overview, full, table
