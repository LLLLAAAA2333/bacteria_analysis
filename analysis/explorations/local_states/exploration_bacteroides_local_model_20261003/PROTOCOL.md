# Bacteroides local chemical–neural model

Fixed before the root joint analysis, 2026-10-03. Scope: all 29 recorded Bacteroides strains, 13 neural coefficients and 162 fresh-report chemical annotations. Strain is the analysis unit. Recorded species, all recording dates, and taxonomy flags remain visible; none is a primary exclusion rule.

## Neural target

Use the saved unit-length, SNR-gated 13-neuron coefficients. Center each neuron by the training-strain mean; do not scale neurons by their standard deviations. The sole main target is PC1, specified before inspecting chemical associations. Orient PC1 so its largest absolute loading is positive. Keep the full spectrum, PC2 loadings, deletion stability and pre-gate comparison as diagnostics. Similar PC1/PC2 eigenvalues limit a unique one-dimensional biological interpretation; no alternative target will be selected to improve chemical correlation.

## Chemical candidates

Independently discover local chemical modules from training-strain log2 concentrations, using training means/sample SDs, Pearson distance 1-r, average linkage and a fixed cut of 0.5. Exclude features with SD <= 1e-12. Require >=3 annotations and >=3 Mass-column families. Score each module by equal family weights and equal member weights within family. These scores are averages of feature z scores, not themselves unit-SD scores. No old intergenus modules, neural-informed clustering, class selection, nonlinear model or threshold search.

## One-axis model

Choose the candidate with the largest squared training Pearson correlation with training PC1. Ties use alphabetical sorted member tuples. Fit ordinary least squares with an intercept:

`predicted neural vector = training neural mean + (intercept + slope * chemical axis score) * training PC1 loading`.

Do not renormalize predictions. If no eligible varying candidate exists, predict the training mean. The full-data winning correlation is selected/apparent, not a validated association. Retain all candidate correlations and the selected members/scales/weights.

## Validation

Leave one strain out (29 folds) and leave one recorded-species label out (16 folds). In EVERY fold refit neural centering/PCA, chemical scaling/clustering/members/weights, chemical candidate selection and regression using only training strains. Project a held-out neural vector with the training mean/loading only. The primary outcome is pooled squared error across all held-out 13-dimensional neural vectors, relative to that fold's training-mean prediction: `1 - SSE_model / SSE_training_mean`. Also report the fold-PC1 component error relative to zero; these fold axes may differ and must not be portrayed as one fixed biological direction. Pool strains, not fold-average R2. Retain individual predictions, errors, training/test IDs and fitted parameters. Compare selected member sets rather than assuming L IDs match across folds.

## Influence and metadata diagnostics

For the full-data selected chemical score and PC1 only, recompute Pearson correlations after deleting each strain and each recorded species, holding axes fixed. These are conditional influence summaries, not cross-validation. For repeated-species strains, remove each recorded-species mean from both scores and report within-species residual association and counts. Separately demean by the exact full recorded-date-set string, reporting the residual association and effective residual dimension. Date-set adjustment is only a descriptive batch diagnostic: dates can overlap across sets and cannot separate batch from biology. Do not use dates to choose the main model or filter the 29 strains. Keep all six taxonomy flags in the main analysis.

## Interpretation boundary

The target is relative neural composition, not overall response magnitude, sensitivity, inhibition or causality. Existing neural template estimation and chemical feature QC predate this local validation and used the broader dataset; this is internal validation of the local model, not an independent end-to-end experiment. Shared recordings and small recorded-species groups remain limitations. Failure to obtain a robust chemical explanation will be reported without expanding the model search.
