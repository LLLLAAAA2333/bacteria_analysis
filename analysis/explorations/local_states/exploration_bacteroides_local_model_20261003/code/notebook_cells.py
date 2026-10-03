# %% Read existing results only; no fitting or file writes.
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown

report = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_local_model_20261003')
summary = json.loads((report / 'tables/model_summary.json').read_text())
scores = pd.read_csv(report / 'tables/strain_model_scores.csv', index_col='strain')
candidates = pd.read_csv(report / 'tables/full_candidate_associations.csv')
validation = pd.read_csv(report / 'tables/heldout_pooled_performance.csv')
display(pd.DataFrame([{
    'strains': summary['n_strains'],
    'recorded_species': summary['n_recorded_species'],
    'neural_PC1_variance': summary['neural_pc1_variance_ratio'],
    'neural_PC2_variance': summary['neural_pc2_variance_ratio'],
    'selected_chemical_axis': summary['selected_axis'],
    'selected_apparent_r': summary['full_selected_pearson_r'],
    'apparent_13D_explained_fraction': summary['full_apparent_vector_error_improvement'],
}]))
display(validation[['scheme', 'n_predictions', 'vector_error_improvement']])

# %% Neural directions and all 29 strains; species and complete dates retained.
display(Image(filename=str(report / 'neural/figures/01_neural_pc1_and_variance.png')))
display(Image(filename=str(report / 'neural/figures/02_neural_strains_by_pc1.png')))

# %% Selected apparent association; all chemical candidates remain inspectable.
display(Image(filename=str(report / 'figures/03_chemical_neural_correspondence.png')))
display(Image(filename=str(report / 'figures/04_prediction_check.png')))
display(candidates[['axis', 'pearson_r', 'r_squared', 'n_members', 'selected_full_data']])
display(scores[['species', 'dates', 'taxonomy_note', 'chemical_axis_score', 'neural_PC1']])

# %% Optional, deliberate recomputation into a NEW report directory only.
# import importlib.util
# spec = importlib.util.spec_from_file_location('local_model', report / 'code/local_model.py')
# model = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(model)
# recomputed = model.run_analysis(report=report.parent / 'bacteroides_local_model_recompute_new')
