# Figure notes

## 03 — Chemical–neural correspondence within Bacteroides

Both panels show the same 29 strains and the same two scores: the full-data selected chemical axis L06 (x) and the full-data neural PC1 projection (y). Panel A uses the recorded species labels; panel B uses each complete recorded date-set, without assigning multi-date strains to a single date. Legends retain all 16 species and all 8 date-sets with their counts. B. abbreviates Bacteroides. Asterisks next to six strain IDs mark nonempty source taxonomy_note entries; they do not indicate statistical significance. All strains and flags are retained.

The dark line is the selected apparent fit already saved by the model calculation, not a new regression performed for this plot. Its full-data Pearson correlation is r=-0.295. L06 was selected as the best of 12 candidates in these same 29 strains; this apparent relationship must not be interpreted as an independent test or held-out predictive performance. No significance stars, p-values, confidence bands, subgroup regressions or parameter search are added. Recorded dates are context labels, not an assertion of chemical culture or assay batches.

L06 contains six report annotations: Sebacic acid, cis-Aconitic acid, Methionine, Leucyl-Glycine, Proline, Glucose 1-phosphate. Each member is standardized using its training mean and sample SD, and axis scores average members with equal family weights and equal within-family weights. The final weighted score is not subsequently divided by its own SD, so its axis label is score, not SD. Neural PC1 is a projection in the supplied normalized neural representation, not a standardized neural score or response amplitude. L06 is a statistical chemical co-variation group, not an established pathway or causal mechanism.

## 04 — Prediction error improvement over the mean

Bars reproduce the saved reduction in squared 13-neuron vector prediction error relative to the corresponding mean-only baseline: all-data apparent +2.93%, leave-one-strain-out -2.92%, and leave-one-recorded-species-out -4.33%. Positive values mean less squared error than the mean baseline; negative values mean greater error. The apparent result uses the full-data neural mean, while held-out results use the training-fold neural mean. Each held-out scheme pools predictions for all 29 strains. This metric concerns the complete 13-neuron representation, not the fraction of PC1 variance explained.

The held-out results are copied from the saved analysis, whose fold-wise reconstruction and model selection are described in the parent protocol. The plotting function does not alter folds, select axes or refit any model. No uncertainty interval is implied by a bar. The three bars deliberately separate the apparent fit from held-out performance.

## Rendering and provenance

Only strain_model_scores.csv, model_summary.json, selected_chemical_axis_members.csv and heldout_pooled_performance.csv are read. Point coordinates and the saved fit are identical between scatter panels. Fixed per-strain text offsets keep ID labels readable without moving data points. Species/date styles, offsets, bounds, exact plotted values and file hashes are stored in plot_parameters.json. The script defines a save-only save_plots(report_root) function and uses existing matplotlib/pandas/numpy libraries.
