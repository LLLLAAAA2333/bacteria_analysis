# %% Load the saved steps 1/2 comparison; this does not rerun the analysis.
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
OUT = ROOT / 'reports/exploration_genus_within_between_20261003'
summary = pd.read_csv(OUT / 'tables/genus_summary.csv')
results = json.loads((OUT / 'results.json').read_text())
display(Image(filename=str(OUT / 'figures/01_within_between_by_genus.png')))

# %% Exact descriptive summaries. Ratios are within/between distance medians.
display(summary[['genus', 'n_strains', 'n_within_pairs',
                 'chemical_within_median', 'chemical_between_median',
                 'chemical_median_ratio', 'chemical_prob_within_smaller',
                 'neural_within_median', 'neural_between_median',
                 'neural_median_ratio', 'neural_prob_within_smaller']])

# %% Two bounded checks: external-reference coverage and neural zeroing.
display(pd.read_csv(OUT / 'tables/sensitivity_external_multistrain.csv'))
display(pd.read_csv(OUT / 'tables/sensitivity_neural_pre_gate.csv'))

# %% Optional display-only rerender (uncomment as needed).
# import sys
# sys.path.insert(0, str(OUT / 'code'))
# from plot_genus_comparison import make_figure
# make_figure(OUT)

# To replay the analysis, use run_analysis(ROOT, NEW_OUT) with a fresh directory.
# Existing scientific tables/results are protected against replacement.
