"""Save figures from frozen local-model results; no fitting or model selection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LABEL_OFFSETS = {
    'A001': (5, 7, 'left'), 'A002': (-5, -12, 'right'),
    'A005': (5, 7, 'left'), 'A006': (5, 7, 'left'),
    'A007': (5, 7, 'left'), 'A008': (5, -12, 'left'),
    'A009': (-5, 8, 'right'), 'A010': (5, -12, 'left'),
    'A011': (-5, -12, 'right'), 'A013': (5, 7, 'left'),
    'A014': (5, 7, 'left'), 'A015': (5, 7, 'left'),
    'A016': (-5, 7, 'right'), 'A017': (5, 7, 'left'),
    'A019': (-4, -14, 'right'), 'A020': (5, -12, 'left'),
    'A021': (5, 7, 'left'), 'A022': (-4, 12, 'right'),
    'A023': (8, 2, 'left'), 'A024': (5, 7, 'left'),
    'A025': (5, 7, 'left'), 'A026': (-5, 7, 'right'),
    'A038': (5, 7, 'left'), 'A040': (5, 7, 'left'),
    'A041': (5, 7, 'left'), 'A044': (5, 7, 'left'),
    'A045': (5, 7, 'left'), 'A048': (5, 7, 'left'),
    'A049': (-5, -12, 'right'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def date_label(value):
    """Preserve each exact date-set, formatting every YYYYMMDD date in full."""
    return ' + '.join(f'{v[:4]}-{v[4:6]}-{v[6:]}' for v in str(value).split(';'))


def categorical_styles(values, kind):
    groups = sorted(values.unique())
    if kind == 'species':
        palette = plt.get_cmap('tab20').colors
        color_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11]
        markers = ['o', 's', '^', 'D']
        return {g: {'color': palette[color_order[i]], 'marker': markers[i % len(markers)]} for i, g in enumerate(groups)}
    palette = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<']
    return {g: {'color': palette[i], 'marker': markers[i]} for i, g in enumerate(groups)}


def scatter_panel(ax, legend_ax, frame, summary, grouping, title):
    styles = categorical_styles(frame[grouping], 'species' if grouping == 'species' else 'dates')
    handles = []
    for name, style in styles.items():
        block = frame.loc[frame[grouping].eq(name)]
        ax.scatter(block.chemical_axis_score, block.neural_PC1, s=70, marker=style['marker'],
                   color=style['color'], edgecolors='white', linewidths=.75, zorder=3)
        label = name.replace('Bacteroides ', 'B. ') if grouping == 'species' else date_label(name)
        label += f' (n={len(block)})'
        handles.append(Line2D([0], [0], marker=style['marker'], color='none', markerfacecolor=style['color'],
                              markeredgecolor='white', markersize=7.5, label=label))
    xline = np.array([-1.23, 1.87])
    ax.plot(xline, summary['intercept'] + summary['slope'] * xline, color='#303030', linewidth=1.45, zorder=2)
    for row in frame.itertuples():
        flag = pd.notna(row.taxonomy_note) and bool(str(row.taxonomy_note).strip())
        label = row.strain + ('*' if flag else '')
        dx, dy, ha = LABEL_OFFSETS[row.strain]
        ax.annotate(label, (row.chemical_axis_score, row.neural_PC1), xytext=(dx, dy), textcoords='offset points',
                    ha=ha, va='center', fontsize=8.3, color='#252525', zorder=4)
    ax.set_xlim(-1.35, 1.98)
    ax.set_ylim(-.68, .50)
    ax.set_xticks([-1, -.5, 0, .5, 1, 1.5])
    ax.set_yticks([-.6, -.4, -.2, 0, .2, .4])
    ax.set_xlabel('Local chemical axis L06 score', fontsize=11)
    ax.set_ylabel('Neural PC1 projection', fontsize=11)
    ax.set_title(title, loc='left', fontsize=12, pad=10)
    ax.grid(axis='y', color='#E6E6E6', linewidth=.65, zorder=0)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color('#999999'); ax.spines['bottom'].set_color('#999999')
    ax.tick_params(labelsize=9, color='#999999')
    legend_ax.set_axis_off()
    legend_ax.legend(handles=handles, loc='upper left', ncol=2, frameon=False,
                     fontsize=8.8, handletextpad=.4, columnspacing=1.15, labelspacing=.58, borderaxespad=0)
    return styles


def save_plots(report_root):
    """Save the two requested plots from existing tables; never fit or display."""
    report = Path(report_root).resolve()
    tables = report / 'tables'
    figures = report / 'figures'
    sources = {name: tables / name for name in ['strain_model_scores.csv', 'model_summary.json',
                                               'selected_chemical_axis_members.csv', 'heldout_pooled_performance.csv']}
    frame = pd.read_csv(sources['strain_model_scores.csv'], dtype={'strain': str, 'dates': str})
    summary = json.loads(sources['model_summary.json'].read_text())
    members = pd.read_csv(sources['selected_chemical_axis_members.csv'])
    cv = pd.read_csv(sources['heldout_pooled_performance.csv']).set_index('scheme')
    assert len(frame) == 29 and frame.strain.is_unique and set(frame.strain) == set(LABEL_OFFSETS)
    assert summary['selected_axis'] == 'L06' and summary['n_chemical_candidates'] == 12
    assert set(members.metabolite) == set(summary['selected_members'])
    assert frame.species.nunique() == 16 and frame.dates.nunique() == 8
    np.testing.assert_allclose(frame.neural_PC1_predicted_apparent,
                               summary['intercept'] + summary['slope'] * frame.chemical_axis_score, rtol=1e-10, atol=1e-10)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'axes.unicode_minus': True})
    fig = plt.figure(figsize=(14.6, 9.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[5.4, 2.45], hspace=.23, wspace=.18,
                           left=.065, right=.99, bottom=.065, top=.91)
    left = fig.add_subplot(grid[0, 0]); right = fig.add_subplot(grid[0, 1], sharex=left, sharey=left)
    left_legend = fig.add_subplot(grid[1, 0]); right_legend = fig.add_subplot(grid[1, 1])
    species_styles = scatter_panel(left, left_legend, frame, summary, 'species', 'A  Recorded species')
    date_styles = scatter_panel(right, right_legend, frame, summary, 'dates', 'B  Recorded date set')
    fig.suptitle('Chemical–neural correspondence within Bacteroides', x=.065, ha='left', y=.974, fontsize=16)
    fig.text(.065, .025, '* Source taxonomy-note flag', fontsize=9, color='#555555')
    fig.savefig(figures / '03_chemical_neural_correspondence.png', dpi=180)
    fig.savefig(figures / '03_chemical_neural_correspondence.svg')
    plt.close(fig)

    conditions = ['All data (apparent)', 'Leave one strain out', 'Leave one recorded species out']
    values = np.array([summary['full_apparent_vector_error_improvement'],
                       cv.loc['leave_one_strain_out', 'vector_error_improvement'],
                       cv.loc['leave_one_recorded_species_out', 'vector_error_improvement']]) * 100
    fig, ax = plt.subplots(figsize=(9.3, 3.7))
    ypos = np.arange(3)
    ax.barh(ypos, values, height=.48, color=['#497CA6', '#C07B60', '#C07B60'], zorder=3)
    ax.axvline(0, color='#4A4A4A', lw=1)
    for y, value in zip(ypos, values):
        ax.text(value + (.15 if value >= 0 else -.15), y, f'{value:+.2f}%',
                ha='left' if value >= 0 else 'right', va='center', fontsize=11)
    ax.set_yticks(ypos, conditions, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-5.8, 4.2)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel('Reduction in squared 13-neuron prediction error (%)', fontsize=11, labelpad=10)
    ax.set_title('Prediction error improvement over the mean', loc='left', fontsize=14, pad=15)
    ax.grid(axis='x', color='#E6E6E6', linewidth=.65, zorder=0)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#999999')
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', color='#999999')
    fig.tight_layout()
    fig.savefig(figures / '04_prediction_check.png', dpi=180)
    fig.savefig(figures / '04_prediction_check.svg')
    plt.close(fig)

    member_text = ', '.join(summary['selected_members'])
    notes = f'''# Figure notes

## 03 — Chemical–neural correspondence within Bacteroides

Both panels show the same 29 strains and the same two scores: the full-data selected chemical axis L06 (x) and the full-data neural PC1 projection (y). Panel A uses the recorded species labels; panel B uses each complete recorded date-set, without assigning multi-date strains to a single date. Legends retain all 16 species and all 8 date-sets with their counts. B. abbreviates Bacteroides. Asterisks next to six strain IDs mark nonempty source taxonomy_note entries; they do not indicate statistical significance. All strains and flags are retained.

The dark line is the selected apparent fit already saved by the model calculation, not a new regression performed for this plot. Its full-data Pearson correlation is r={summary['full_selected_pearson_r']:.3f}. L06 was selected as the best of {summary['n_chemical_candidates']} candidates in these same 29 strains; this apparent relationship must not be interpreted as an independent test or held-out predictive performance. No significance stars, p-values, confidence bands, subgroup regressions or parameter search are added. Recorded dates are context labels, not an assertion of chemical culture or assay batches.

L06 contains six report annotations: {member_text}. Each member is standardized using its training mean and sample SD, and axis scores average members with equal family weights and equal within-family weights. The final weighted score is not subsequently divided by its own SD, so its axis label is score, not SD. Neural PC1 is a projection in the supplied normalized neural representation, not a standardized neural score or response amplitude. L06 is a statistical chemical co-variation group, not an established pathway or causal mechanism.

## 04 — Prediction error improvement over the mean

Bars reproduce the saved reduction in squared 13-neuron vector prediction error relative to the corresponding mean-only baseline: all-data apparent {values[0]:+.2f}%, leave-one-strain-out {values[1]:+.2f}%, and leave-one-recorded-species-out {values[2]:+.2f}%. Positive values mean less squared error than the mean baseline; negative values mean greater error. The apparent result uses the full-data neural mean, while held-out results use the training-fold neural mean. Each held-out scheme pools predictions for all 29 strains. This metric concerns the complete 13-neuron representation, not the fraction of PC1 variance explained.

The held-out results are copied from the saved analysis, whose fold-wise reconstruction and model selection are described in the parent protocol. The plotting function does not alter folds, select axes or refit any model. No uncertainty interval is implied by a bar. The three bars deliberately separate the apparent fit from held-out performance.

## Rendering and provenance

Only strain_model_scores.csv, model_summary.json, selected_chemical_axis_members.csv and heldout_pooled_performance.csv are read. Point coordinates and the saved fit are identical between scatter panels. Fixed per-strain text offsets keep ID labels readable without moving data points. Species/date styles, offsets, bounds, exact plotted values and file hashes are stored in plot_parameters.json. The script defines a save-only save_plots(report_root) function and uses existing matplotlib/pandas/numpy libraries.
'''
    (figures / 'figure_notes.md').write_text(notes)
    outputs = ['03_chemical_neural_correspondence.png', '03_chemical_neural_correspondence.svg',
               '04_prediction_check.png', '04_prediction_check.svg', 'figure_notes.md']
    parameters = {'inputs': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in sources.items()},
                  'plot_code': {'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
                  'point_count_per_panel': len(frame), 'taxonomy_flag_count': int(frame.taxonomy_note.fillna('').astype(str).str.strip().ne('').sum()),
                  'x_field': 'chemical_axis_score', 'y_field': 'neural_PC1', 'selected_axis': summary['selected_axis'],
                  'line': {'intercept': summary['intercept'], 'slope': summary['slope'], 'kind': 'saved selected apparent fit'},
                  'selected_apparent_r': summary['full_selected_pearson_r'], 'selection_candidate_count': summary['n_chemical_candidates'],
                  'axis_limits': {'x': [-1.35,1.98], 'y': [-.68,.50]}, 'scatter_figsize_inches': [14.6,9.6],
                  'prediction_figsize_inches': [9.3,3.7], 'dpi': 180, 'label_offsets_points': LABEL_OFFSETS,
                  'species_styles': species_styles, 'exact_date_set_styles': date_styles,
                  'prediction_metric': '100 * (1 - squared_vector_error_model / squared_vector_error_mean_baseline)',
                  'prediction_conditions': conditions, 'prediction_values_percent': values.tolist(),
                  'outputs': {name: {'path': str(figures / name), 'sha256': sha256(figures / name)} for name in outputs}}
    (figures / 'plot_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
    return [figures / name for name in outputs]


if __name__ == '__main__':
    save_plots(Path(__file__).resolve().parents[1])
