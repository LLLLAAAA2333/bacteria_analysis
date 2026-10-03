# Bacteroides: a fixed neural contrast and simple chemical states

Protocol frozen before inspecting any new ADF–ASH / chemical association on 2026-10-03. This is a bounded exploratory analysis authorized by the investigator, not an independent confirmatory study.

## Cohort and response

Use all 29 recorded Bacteroides strains and all 162 previously QC-screened chemical features. Retain all 16 recorded species, recording-date sets, and six taxonomy-note flags. Do not exclude the eight strains lacking sufficient animal-split reliability coverage.

Primary response: `(coefficient_ADF - coefficient_ASH) / ||coefficient_13||2`, taken from the saved SNR-gated unit profiles. It is a relative configuration contrast, with a denominator involving all 13 neurons. The prespecified sensitivity response is the difference of the unnormalized SNR-gated template coefficients, in the original fluorescence-response coefficient units. Also describe the pre-gate unit contrast with the same chemical state. Do not change the primary response according to chemical association.

The saved temporal templates and neural representation are frozen from the earlier full atlas. Validation below measures strain-level chemical prediction conditional on that representation; it does not refit temporal templates or establish independent animal/culture replication.

## Chemical states and simple models

Reuse the pure `chemical/code/local_chemical_axes.py` from `exploration_bacteroides_local_model_20261003`, without changing its parameters: training-feature z scores (sample SD), Pearson distance `1-r`, average linkage cut at 0.5, at least three annotations and three Mass-column families per group, equal family weights and equal member weights within family. No individual-feature search, alternate cut search, nonlinear models, or post hoc neural-target search.

Fit each eligible state to the primary contrast with an OLS intercept and one slope. Select the state with largest training Pearson r squared; break ties by the sorted member-name tuple. Report every full-cohort candidate and its actual members. If no state exists in a training fold, predict its training mean.

Use the selected state, without secondary selection, to fit the unnormalized contrast, pre-gate unit contrast, and each of the 13 unit neural coordinates. These are representation checks and response-pattern descriptions, not additional independently selected discoveries. Increasing ADF–ASH does not by itself establish ADF increase with ASH decrease.

## Held-out validation

Run 29 leave-one-strain-out folds and 16 leave-recorded-species-out folds. In each fold, refit chemical means, SDs, correlations, clustering, memberships, weights, state selection, and OLS using only training strains. Transform held-out chemistry with those fitted parameters. The neural contrast is fixed, so no neural PCA is estimated or selected.

For each target report pooled held-out `1 - SSE_model / SSE_training_mean_baseline`, RMSE of both models, and prediction versus observation. Also save full 13-coordinate predictions, per-neuron scores, and pooled 13D scores. A negative value means that this model predicts worse than the corresponding training-mean baseline. LOO and species folds are overlapping estimates, not independent replicates. Save every fold's parameters and identify state-selection stability by member overlap, not local group IDs.

The target was prioritized using this cohort's preceding neural reliability results; held-out chemical validation does not erase that history or validate an entirely independent discovery pipeline.

## Continuity, composition, and influence

With the full-cohort selected state held fixed, show all 29 strain points and the complete 13-neuron responses, with recorded species/date information and taxonomy flags. Describe low, middle, and high chemistry using outcome-independent rank thirds (first 10, next 10, final 9 after chemical-score then strain-ID ordering). Save the species/date composition, individual values, means, medians, ranges, and simple response slopes. These thirds summarize support and do not prove a continuous biological dose response.

Report Pearson and Spearman associations, observed chemistry IQR and its fitted contrast change, fixed-axis one-strain and one-species deletion sensitivity, and within-repeated-species and within-identical-recorded-date-set demeaned associations. The latter are limited descriptive checks: small within-group support and confounded dates prevent treating them as complete adjustment for species or batch. Contrast ADF and ASH separately and keep all 13 coordinates visible. Quantify whether the apparent chemical effect is driven by extreme points and whether it survives held-out strain/species prediction.

Do not assign a post hoc threshold for success. Judge the investigator's requested clear, continuous, simple pattern using effect size, actual low/middle/high support, held-out improvement, and sensitivity together. If evidence is weak or unstable, state that the present prespecified candidate family does not support the desired model; do not expand the search within this run.

## Reproducibility and scope

Write only within this new report directory. Preserve source files and previous reports. Save source paths and SHA-256, cohort/targets, fold parameters, numerical tables, small notebook-callable functions, and English scientific figures. Independently reconstruct validation metrics and inspect figures. Do not run or modify notebooks, recollect raw data, or infer compound potency, dose sensitivity, causality, or behavioral meaning from this association.
