# Independent review of neuron transfer and poster figures

Scope: read-only inspection of 03_neuron_patterns.py and 04_poster_figures.py, reconstruction of the saved transfer values from trial_curves.parquet, checking exported figure data, and visual inspection of poster_neighborhood_atlas.png. No analysis code or result table was changed during this review.

## Overall assessment

No computational blocker was found in the leave-animal/presentation transfer analysis. The figure contributions sum to their population values, preserve negative estimates, and display unsupported cells as missing. Two presentation clarifications were sent to the root agent: ensure negative numerical ticks are visible on the asymmetric color scale, and explicitly state that two of the 28 rows average 10 eligible neurons rather than 13. These are important for interpretation but do not require another analysis.

## Whole-animal exclusion and presentation transfer

- The target animal is identified by the combination of acquisition block and worm key. Its responses to **every strain** are excluded before fitting every training-presentation cell scale. Same worm labels in other blocks are different animals.
- Cell selection uses the source presentation in other animals. Held alignment uses the target presentation in the held animal. The target animal is absent from all 20,498 exported cell-level training lists; each of 1,776 folds has exactly one selected cell and no duplicate pair/presentation/animal key.
- The top cell is selected by training energy among cells observed in the held response. The candidate panel therefore conditions on measurement availability, but not on the numerical held response. This is acceptable as an observed-panel sensitivity; it does not establish performance for unmeasured cells or a deployable fixed-panel predictor.
- First and later presentations are derived independently per cell from segment order. Inspection of all 607 animal-by-strain groups found no discrepancy in first segment index between the measured neuron classes. Thus the first-presentation tables are not accidentally mixing different presentation numbers across cells in this dataset.
- Six scale fits spanning both presentation modes were independently reconstructed after whole-animal exclusion; maximum absolute difference from saved scales was 1.11e-16. All saved transfer scale joins also agree.
- All 20,498 cell-level training energies were independently recomputed using explicit sums over distinct training-animal pairs. Maximum absolute error was 6.76e-14. Recomputed held alignment differed by at most 1.71e-13.
- All 1,776 selected cells match the maximum training energy; selected, all-cell and selected-cell-removed fold means were recomputed with maximum absolute error 1.34e-15.

The transfer is a joint animal/presentation stress test for the recorded stimulus catalogue. Pairs, folds and presentations overlap. It is not independent strain/date replication or an independent validation dataset. Fixed sequence and preceding-stimulus effects can persist in both presentations. The code does not use transfer p-values or count these repeated rows as independent animals.

The positive-part contribution fractions and effective number of positive contributors in 03 are descriptive summaries. They do not change the signed population statistic. Negative cell contributions remain separately exported. The descriptive top cell from full data is distinct from the top cell re-selected in each transfer training fold.

## Figure values and meaning

- For all 147 pairs, the sum of per-cell fixed-scale contributions equals the displayed population score to 8.89e-16. Each contribution is the neuron-specific cross-animal energy divided by that pair's eligible neuron count. Missing cells are not filled with zero; this means incomplete rows have a different averaging panel and require a coverage statement.
- Of the 28 displayed nonautomatic chemical-neighbor pairs, 26 have 13 eligible cells. A238/A240 and A246/A248 each have 10; ASI, ASEL and AWCON have fewer than 3 paired animals and are marked with a cross.
- Raw-unit contributions and totals have units (delta F/F0)^2. Fixed-scale scores are dimensionless squared cell-scaled response units: both numerator and scale originate in delta F/F0. These are average cross-animal inner products, not nonnegative metric distances. A negative value is possible and cannot mean biological equivalence.
- Cell colors express contribution to a reproducible response difference, not activation versus inhibition. The figure explicitly states this; the per-cell sign must not be given a physiological activation interpretation.
- The chemical coordinate is RMS of differences in log2 fold change over the same 380 features. It is not measured stimulus concentration. A chemical neighbor is relative to the same-genus, same-reference, same-block candidate set, not chemically identical.
- The 28 rows are selected by chemistry-only nearest-either status and a candidate-set size of at least three, then sorted by the observed neural score. Both ordering and heatmap are descriptive; neither is evidence for a discovered cluster. The figure states the ordering and does not claim validation from it.
- All 28 chemical coordinates lie within the atlas x limits (observed 1.758--3.164); all delete-animal ranges lie within the displayed neural limits (observed -0.240--3.022). All 147 background chemical values lie within the landscape limits (observed 1.758--3.569). No inspected observation or range is clipped.
- Displayed deletion ranges remove an entire animal while holding scales and the initial cell panel fixed; they are correctly labeled sensitivity ranges rather than confidence intervals.

## Presentation clarifications sent for final rendering

The inspected image used TwoSlopeNorm with unequal negative and positive limits. Equal color intensity on opposite sides therefore does not imply equal numerical magnitude. Its initial automatically chosen colorbar showed no negative ticks. The root agent was asked to add explicit negative/zero/positive numerical ticks or use a symmetric linear normalization, and to retain the negative-estimate explanation. This review does not alter the renderer.

The root agent was also asked to make the two incomplete panels explicit in the caption: 26 pairs have 13 cells, 2 pairs have 10; the population statistic averages eligible cells. The crosses already distinguish missing values from negative gray values. The common 10-cell sensitivity is available in the tables for checking this coverage choice.
