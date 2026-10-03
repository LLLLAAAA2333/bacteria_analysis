"""Poster and individual-support figures from saved results only.

All heatmap values are observed unit coefficients centered on the genus mean.
The scatter ordinate is explicitly the selected, fitted 13-neuron direction.
Calling make_figures never refits the chemical or neural models.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

NEURONS = ['ASK', 'ADL', 'ASI', 'AWA', 'AWB', 'ASG', 'ADF', 'ASH',
           'ASJ', 'ASEL', 'ASER', 'AWCON', 'AWCOFF']
GENERA = ['Bacteroides', 'Bifidobacterium']
GROUPS = ['Low', 'Mid', 'High']
GROUP_COLORS = {'Low': '#347FA3', 'Mid': '#9CA5A6', 'High': '#C66B36'}


def style():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 11, 'axes.titlesize': 13,
        'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })


def save(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text().strip():
            bb = item.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1:
                outside.append(item.get_text())
    if outside:
        raise ValueError(f'Text outside figure canvas: {outside}')
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(path.with_suffix('.' + ext), dpi=320, facecolor='white')
    plt.close(fig)


def load_tables(out):
    t = Path(out) / 'figure_data'
    return {name: pd.read_csv(t / f'{name}.csv') for name in
            ['strain_scores', 'neural_slopes', 'selected_members',
             'group_neural_means', 'selected_chemical_z']}


def neural_limit(tables):
    # Round upward: no clipping and exactly the same scale for both genera.
    maximum = tables['group_neural_means'].centered_mean.abs().max()
    return float(np.ceil(maximum / .05) * .05)


def make_poster(out, tables):
    out = Path(out)
    limit = neural_limit(tables)
    fig = plt.figure(figsize=(12.0, 9.0))
    fig.suptitle('Local chemical states and neural response patterns',
                 x=.5, y=.973, fontsize=18, weight='bold')
    positions = [(.105, .54, .35, .26), (.60, .54, .35, .26)]
    hpositions = [(.105, .153, .35, .285), (.60, .153, .35, .285)]
    display = {'neural_heatmap_limit': limit, 'neural_row_order': NEURONS,
               'group_order': GROUPS, 'group_colors': GROUP_COLORS,
               'neural_values': 'Observed group mean minus observed genus mean; no row scaling',
               'scatter_y': {'Bacteroides': 'Previously fixed unit ADF minus ASH',
                             'Bifidobacterium': 'Projection onto fitted 13-neuron direction'},
               'genera': {}}
    for k, genus in enumerate(GENERA):
        s = tables['strain_scores'].query('genus == @genus').copy()
        members = tables['selected_members'].query('genus == @genus')
        n_species = s.species.nunique()
        cx = positions[k][0] + positions[k][2] / 2
        fig.text(cx, .917, genus, ha='center', fontsize=16, fontstyle='italic')
        fig.text(cx, .885, f'{len(s)} strains · {n_species} recorded species',
                 ha='center', fontsize=11, color='#4C5357')
        fig.text(cx, .855, f'{len(members)} co-varying chemical annotations',
                 ha='center', fontsize=11, color='#4C5357')
        examples = ('Includes p-Cresol, N-acetylleucine and ADP' if genus == 'Bacteroides'
                    else 'Includes succinic acid, choline and vitamin B1')
        fig.text(cx, .826, examples, ha='center', fontsize=9.5, color='#60686C')
        ax = fig.add_axes(positions[k])
        for group in GROUPS:
            part = s[s.rank_group == group]
            ax.scatter(part.chemical_score, part.plot_response,
                       color=GROUP_COLORS[group], s=44, edgecolor='white',
                       linewidth=.6, zorder=3, label=group)
        x, y = s.chemical_score.to_numpy(), s.plot_response.to_numpy()
        beta = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, np.polyval(beta, xx), color='#51595D', lw=1.25, zorder=2)
        ax.axhline(0, color='#D6D9DB', lw=.6, zorder=1)
        ylabel = 'ADF − ASH (unit coefficients)' if genus == 'Bacteroides' else 'Projection onto fitted\n13-neuron direction'
        ax.set(xlabel='Chemical-state score', ylabel=ylabel)
        ax.text(-.22, 1.02, 'AB'[k], transform=ax.transAxes,
                fontsize=16, weight='bold')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#ECEEF0', lw=.55)
        ax.margins(x=.09, y=.12)
        ax = fig.add_axes(hpositions[k])
        g = tables['group_neural_means'].query('genus == @genus')
        mat = g.pivot(index='neuron', columns='rank_group', values='centered_mean').loc[NEURONS, GROUPS]
        ax.pcolormesh(np.arange(4)-.5, np.arange(14)-.5, mat.to_numpy(),
                      cmap='RdBu_r', vmin=-limit, vmax=limit, shading='flat')
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(12.5, -.5)
        counts = s.rank_group.value_counts()
        ax.set_yticks(range(len(NEURONS)), NEURONS)
        ax.set_xticks(range(3), [f'{group}\n(n = {counts[group]})' for group in GROUPS])
        for tick, group in zip(ax.get_xticklabels(), GROUPS):
            tick.set_color(GROUP_COLORS[group])
        ax.tick_params(length=0, axis='both', pad=5)
        ax.set_title('Observed neural deviations', pad=11, fontsize=12)
        ax.set_xlabel('Chemical-state thirds', labelpad=8)
        ax.text(-.22, 1.02, 'CD'[k], transform=ax.transAxes,
                fontsize=16, weight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 13, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.7)
        ax.tick_params(which='minor', length=0)
        display['genera'][genus] = {'n': len(s), 'n_species': n_species,
                                  'n_annotations': len(members),
                                  'counts': {g: int(counts[g]) for g in GROUPS},
                                  'scatter_strains': s.strain.tolist()}
    cax = fig.add_axes([.365, .055, .30, .013])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap='RdBu_r'),
                      cax=cax, orientation='horizontal', ticks=[-limit, 0, limit])
    cb.set_label('Observed unit coefficient − genus mean', fontsize=10, labelpad=4)
    cb.solids.set_rasterized(False)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=2)
    save(fig, out / 'figures/poster_final')
    (out / 'figures/display_parameters.json').write_text(json.dumps(display, indent=2))


def make_individuals(out, tables):
    """Full chemical membership and all 13 coordinates, in the same strain order."""
    out = Path(out)
    for genus in GENERA:
        s = tables['strain_scores'].query('genus == @genus').sort_values(['chemical_score', 'strain'])
        chem = tables['selected_chemical_z'].query('genus == @genus')
        members = tables['selected_members'].query('genus == @genus').metabolite.tolist()
        z = chem.pivot(index='metabolite', columns='strain', values='z').loc[members, s.strain]
        neural = s.set_index('strain')[[f'unit_{n}' for n in NEURONS]].T
        neural = neural.sub(neural.mean(axis=1), axis=0)
        h = max(8.6, 4.8 + .19 * len(members))
        fig = plt.figure(figsize=(13.8, h))
        gs = fig.add_gridspec(3, 1, left=.32, right=.895, bottom=.18, top=.90,
                              height_ratios=[1.5, len(members), 13], hspace=.16)
        top = fig.add_subplot(gs[0])
        top.plot(range(len(s)), s.chemical_score, color='#3A4247', lw=1)
        top.scatter(range(len(s)), s.chemical_score, c=s.rank_group.map(GROUP_COLORS), s=20)
        top.set_xlim(-.5, len(s)-.5)
        top.margins(y=.35)
        top.set_ylabel('Score')
        top.set_xticks([])
        top.spines['bottom'].set_visible(False)
        for slot, values, names, label in [
            (1, z.to_numpy(), members, 'Chemical log2 concentration (z)'),
            (2, neural.to_numpy(), NEURONS, 'Unit coefficient − genus mean')]:
            ax = fig.add_subplot(gs[slot])
            vmax = np.ceil(np.max(np.abs(values)) * 10) / 10
            im = ax.pcolormesh(np.arange(values.shape[1]+1)-.5,
                              np.arange(values.shape[0]+1)-.5, values,
                              cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='flat')
            ax.set_xlim(-.5, values.shape[1]-.5)
            ax.set_ylim(values.shape[0]-.5, -.5)
            visible_names = [name.replace('（', '(').replace('）', ')') for name in names]
            ax.set_yticks(range(len(names)), visible_names, fontsize=8 if slot==1 else 9)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            boundary = 0
            for group in GROUPS[:-1]:
                boundary += (s.rank_group == group).sum()
                ax.axvline(boundary-.5, color='#3B4146', lw=1.2)
            if slot == 2:
                labels = [f'{row.strain}  {row.species.split(" ", 1)[-1]}' for row in s.itertuples()]
                ax.set_xticks(range(len(s)), labels, rotation=65, ha='right', fontsize=8)
                ax.set_xlabel('Every strain, ordered by chemical-state score')
            else:
                ax.set_xticks([])
            box = ax.get_position()
            cax = fig.add_axes([.913, box.y0 + box.height * .1, .01, box.height * .8])
            cb = fig.colorbar(im, cax=cax)
            cb.solids.set_rasterized(False)
            cb.set_label(label, fontsize=9)
            cb.ax.tick_params(labelsize=8)
            cb.outline.set_visible(False)
        fig.suptitle(f'{genus}: individual support', fontsize=16, weight='bold', y=.968)
        fig.text(.625, .932, f'All {len(s)} strains · all selected chemical annotations · all 13 neurons',
                 ha='center', fontsize=11)
        save(fig, out / f'figures/support_individuals_{genus.lower()}')


def make_figures(out):
    out = Path(out)
    style()
    tables = load_tables(out)
    make_poster(out, tables)
    make_individuals(out, tables)


if __name__ == '__main__':
    make_figures(Path(__file__).resolve().parents[1])
