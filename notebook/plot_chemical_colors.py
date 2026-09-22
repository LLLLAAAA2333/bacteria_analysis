"""Inspect continuous chemical PCo1 scores and actual bacterial AID colors."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_chemical_colors(mapping):
    members = mapping['samples'].sort_values('chemical_PCo1', kind='stable')
    norm = Normalize(mapping['vmin'], mapping['vmax'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor='white')
    fig.subplots_adjust(left=.07, right=.95, bottom=.29, top=.82, wspace=.33)
    fig.suptitle('Chemical reference defines color; each RDM defines position',
                 x=.07, y=.96, ha='left', fontsize=19, weight='bold')
    fig.text(.07, .90, f"{len(members)} bacteria | continuous chemical PCo1 | "
             f"{mapping['positive_inertia_fraction']:.1%} of positive inertia | {mapping['cmap_name']}",
             fontsize=12, color='#475569')
    values = mapping['reference_rdm'].loc[members.index, members.index].to_numpy()
    matrix = axes[0].imshow(values, cmap='magma', vmin=0, interpolation='nearest')
    axes[0].set(xticks=[], yticks=[], xlabel='Bacteria ordered by chemical PCo1',
                ylabel='Chemical reference RDM (same order)')
    fig.colorbar(matrix, ax=axes[0], fraction=.04, pad=.03, label='Chemical distance')
    counts, edges, bars = axes[1].hist(members.chemical_PCo1, bins=24)
    for bar, center in zip(bars, (edges[:-1] + edges[1:]) / 2):
        bar.set_facecolor(mapping['cmap'](norm(center)))
    axes[1].set(xlabel='Chemical PCo1 score', ylabel='Number of bacteria',
                title='Score distribution (bins do not define colors)')
    axes[1].spines[['top', 'right']].set_visible(False)
    # All sample positions on the bar; representative real AIDs use spaced labels.
    bar_ax = fig.add_axes([.09, .20, .82, .025])
    fig.colorbar(ScalarMappable(norm=norm, cmap=mapping['cmap']), cax=bar_ax, orientation='horizontal')
    bar_ax.scatter(members.chemical_PCo1, np.ones(len(members)) * .5,
                   marker='|', s=55, color='white', linewidth=.7)
    selected = np.linspace(0, len(members) - 1, min(9, len(members))).round().astype(int)
    for position, index in zip(np.linspace(.03, .97, len(selected)), selected):
        aid, row = members.index[index], members.iloc[index]
        bar_ax.annotate(aid, xy=(row.chemical_PCo1, .5), xycoords='data',
                        xytext=(position, -2.9), textcoords='axes fraction', ha='center', va='top',
                        fontsize=10, arrowprops=dict(arrowstyle='-', color='#64748b', lw=.7))
    fig.text(.07, .045, 'Chemical RDM -> classical PCoA -> PCo1 score -> fixed linear color scale for every sample.',
             fontsize=11, color='#475569')
    fig.text(.07, .018, mapping['note'], fontsize=10, color='#475569')

    columns = 4 if len(members) <= 160 else 6
    rows = int(np.ceil(len(members) / columns))
    full, key_axes = plt.subplots(1, columns, figsize=(14, max(7, rows * .28 + 1.4)), facecolor='white')
    full.subplots_adjust(left=.025, right=.985, bottom=.06, top=.89, wspace=.15)
    full.suptitle('Continuous bacterial colors from the chemical reference', x=.025, y=.97,
                  ha='left', fontsize=19, weight='bold')
    full.text(.025, .93, 'Actual AIDs ordered by PCo1; the same AID keeps the same color in every embedding.',
              fontsize=11, color='#475569')
    for col, ax in enumerate(key_axes):
        ax.axis('off')
        ax.set(xlim=(0, 1), ylim=(rows - .25, -1.5))
        ax.text(.025, -1, 'Color   AID', fontsize=11, weight='bold')
        ax.text(.98, -1, 'PCo1', ha='right', fontsize=11, weight='bold')
        for y, (aid, row) in enumerate(members.iloc[col * rows:(col + 1) * rows].iterrows()):
            if y % 2 == 0:
                ax.add_patch(plt.Rectangle((0, y - .42), 1, .85, color='#f1f5f9', lw=0))
            ax.add_patch(plt.Rectangle((.025, y - .27), .14, .54, facecolor=row.color_hex, lw=0))
            ax.text(.21, y, aid, va='center', fontsize=11)
            ax.text(.98, y, f'{row.chemical_PCo1:.3f}', ha='right', va='center', fontsize=11)
    full.text(.025, .023, 'Equal or close scores have equal or similar colors; no clustering or AID-based ordering defines color.',
              fontsize=10, color='#475569')
    return fig, full, members
