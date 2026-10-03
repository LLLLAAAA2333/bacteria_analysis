# %% Read-only entry for an existing notebook. No fitting or file writes.
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

report = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_neural_reliability_20261003')
geometry = json.loads((report / 'subspace/summary.json').read_text())
repeat_reliability = json.loads((report / 'repeats/summary.json').read_text())

# %% Deletion sensitivity is distinct from experimental repeat reliability.
deletions = pd.read_csv(report / 'subspace/tables/deletion_subspace_metrics.csv')
display(deletions.groupby('scheme')[['max_principal_angle_deg', 'pc1_acute_angle_deg']]
        .agg(['min', 'median', 'max']))
display(pd.read_csv(report / 'subspace/tables/pre_gate_subspace_comparison.csv'))

# %% Display saved figures only; each branch's captions explain scope and units.
for branch in ['subspace', 'repeats']:
    for figure in sorted((report / branch / 'figures').glob('*.png')):
        display(Image(filename=str(figure)))

# %% Compact repeat summaries; ratios refer to each table's stated reference.
animal_metrics = pd.read_csv(report / 'repeats/tables/animal_metric_summary.csv')
main = animal_metrics.loc[
    animal_metrics.representation.eq('gated')
    & animal_metrics.metric.isin(['unit_ADF_minus_ASH', 'unit_AWB', 'fixed_PC12', 'unit_13D'])
    & animal_metrics.quantity.isin(['same_RMS', 'between_RMS', 'same_to_between_ratio', 'pearson'])
]
display(main.pivot(index='metric', columns='quantity', values='median'))
date_metrics = pd.read_csv(report / 'repeats/tables/date_summary.csv')
display(date_metrics.loc[date_metrics.representation.eq('gated')])
display(pd.read_csv(report / 'repeats/tables/date_auxiliary_summary.csv'))
print('Strains never supported for complete-13 animal comparisons:',
      repeat_reliability['strains_never_joint_complete13'])
