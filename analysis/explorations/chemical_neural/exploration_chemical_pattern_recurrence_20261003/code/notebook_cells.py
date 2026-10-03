# %% Read the saved exploration without rerunning discovery or holdout.
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
OUT = ROOT / 'reports/exploration_chemical_pattern_recurrence_20261003'
sys.path.insert(0, str(OUT / 'code'))

results = json.loads((OUT / 'results.json').read_text())
members = pd.read_csv(OUT / 'tables/selected_members.csv')
neural_slopes = pd.read_csv(OUT / 'tables/neural_slopes.csv')
samples = pd.read_csv(OUT / 'tables/sample_scores.csv')
results

# %% Fixed chemical membership and complete neural change vectors.
members[['metabolite', 'effective_weight', 'discovery_rho_to_score', 'holdout_rho_to_score']]

# %%
neural_slopes

# %% Display existing figures.
from IPython.display import Image, display

display(Image(filename=str(OUT / 'figures/01_candidate_validation.png')))
display(Image(filename=str(OUT / 'figures/02_holdout_profiles.png')))

# %% Optional display-only rerender; no selection, fitting, or heldout testing.
# from plot_recurrence import make_figures
# make_figures(OUT)

# To reproduce the frozen protocol from scratch, use a NEW output directory:
# from chemical_recurrence import run_discovery, run_holdout
# run_discovery(ROOT, new_output_directory)
# run_holdout(ROOT, new_output_directory)
# The original run deliberately refuses to overwrite its frozen candidate/results.
