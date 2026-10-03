# %% Read saved poster output; no model fitting and no data processing.
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

report = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/poster_local_chemical_neural_20261003')
display(Image(filename=str(report / 'figures/poster_final.png')))

# %% Exact poster data: prior Bacteroides anchor plus new Bifidobacterium result.
poster_strains = pd.read_csv(report / 'figure_data/strain_scores.csv')
poster_means = pd.read_csv(report / 'figure_data/group_neural_means.csv')
poster_members = pd.read_csv(report / 'figure_data/selected_members.csv')
display(poster_means)

# %% Individual support (including every selected chemical annotation).
display(Image(filename=str(report / 'figures/support_individuals_bacteroides.png')))
display(Image(filename=str(report / 'figures/support_individuals_bifidobacterium.png')))

# %% Full bounded extension, including the negative Bacteroides all-vector fit.
checks = pd.read_csv(report / 'tables/heldout_performance.csv')
display(checks[checks.neuron.eq('all13')])

# %% Optional redraw only; explicit function call, no refitting.
# import sys
# sys.path.insert(0, str(report / 'code'))
# from prepare_display import prepare_display
# from plot_poster import make_figures
# prepare_display(report)
# make_figures(report)
