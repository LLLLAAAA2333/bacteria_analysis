# %% Load newly calculated results. This does not rerun the analysis.
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
OUT = ROOT / 'reports/exploration_chemical_pattern_direct_report_20261003'
sys.path.insert(0, str(OUT / 'code'))

results = json.loads((OUT / 'results.json').read_text())
feature_audit = pd.read_csv(OUT / 'tables/fresh_feature_audit.csv')
members = pd.read_csv(OUT / 'tables/selected_members.csv')
slopes = pd.read_csv(OUT / 'tables/neural_slopes.csv')
results

# %% Report annotations and newly fitted chemical score weights.
members[['metabolite', 'unit', 'effective_weight', 'discovery_rho_to_score', 'holdout_rho_to_score']]

# %% Complete neural change vector.
slopes

# %% Main draft: observed means in five chemistry-defined groups, all 106 strains.
from IPython.display import Image, display

display(Image(filename=str(OUT / 'figures/05_grouped_story.png')))

# %% Supporting views: every strain, followed by the existing split comparison.
display(Image(filename=str(OUT / 'figures/05_grouped_story_individuals.png')))
display(Image(filename=str(OUT / 'figures/06_holdout_prediction_explained.png')))
display(Image(filename=str(OUT / 'figures/01_candidate_correspondence.png')))
display(Image(filename=str(OUT / 'figures/02_holdout_profiles.png')))

# %% Reverse check: all 13 neural coefficients predict the fixed chemical targets.
REVERSE = OUT / 'reverse_prediction_20261003'
reverse_metrics = pd.read_csv(REVERSE / 'heldout_metrics.csv', index_col='target')
display(reverse_metrics[['relative_error_reduction', 'pearson_r', 'rmse', 'baseline_rmse']])
display(Image(filename=str(REVERSE / 'reverse_prediction.png')))

# %% Bounded reduced-input check: same chemical score and same 70/36 split.
REDUCED = OUT / 'reduced_input_comparison_20261003'
reduced_results = json.loads((REDUCED / 'results.json').read_text())
display(pd.DataFrame(reduced_results['models']).T)
display(pd.read_csv(REDUCED / 'selection_stability.csv'))
display(Image(filename=str(REDUCED / 'reduced_input_comparison.png')))

# %% Bounded interaction/curvature check within the five-input selection procedure.
NONLINEAR = OUT / 'nonlinear_input_comparison_20261003'
nonlinear_results = json.loads((NONLINEAR / 'results.json').read_text())
display(pd.DataFrame(nonlinear_results['models']).T)
display(pd.DataFrame(nonlinear_results['comparisons']).T)
display(Image(filename=str(NONLINEAR / 'nonlinear_paired_error_comparison.png')))
display(Image(filename=str(NONLINEAR / 'nonlinear_prediction_comparison.png')))

# %% Optional display-only rerender.
# from plot_nonlinear_input_comparison import make_figures as make_nonlinear_figures
# make_nonlinear_figures(NONLINEAR)
# from plot_reduced_input_comparison import make_figure as make_reduced_figure
# make_reduced_figure(REDUCED)
# from plot_grouped_story import make_figures as make_grouped_figures
# make_grouped_figures(OUT)
# from plot_holdout_explainer import make_figure as make_holdout_explainer
# make_holdout_explainer(OUT)
# from plot_direct_report import make_figures
# make_figures(OUT)

# For an exact replay, use a new output directory, without changing settings:
# from direct_report_recurrence import run_discovery, run_holdout
# run_discovery(ROOT, new_output_directory)
# run_holdout(ROOT, new_output_directory)
# The saved original candidate/results cannot be overwritten by these functions.
