# Local chemical–neural patterns for the final poster figure

Written before executing the extension on 2026-10-03. The user requests a visible local correspondence, preferably using all 13 neurons and broader bacterial coverage, without forcing a universal rule or emphasizing recording dates.

## Bounded analysis

- Use the current fresh 106 × 162 log2 chemical panel and saved 106 × 13 unit neural coefficients. Never re-estimate shared templates or alter raw data.
- Eligibility depends only on coverage: genera with at least 10 strains. This gives Bacteroides (29) and Bifidobacterium (11). Keep every member, species label and taxonomy flag. This pragmatic threshold permits at least three observations in each descriptive chemical-rank third; it is not a statistical power guarantee. No search across arbitrary subsets, no small-genus models.
- Reuse `local_chemical_axes.fit_axes` and `transform_axes` unchanged: training mean and sample SD, Pearson distance 1-r, average linkage at 0.5, at least three annotations and three Mass-column families, equal-family/equal-member weights.
- For each chemical state, fit an intercept and one slope to each of all 13 unit coordinates. Select one state by smallest total training SSE across the 13 coordinates; ties by sorted member names. Neural coordinates are not standardized. This describes correspondence in the existing representation, not equal biological importance of each neuron. No additional chemical-state, component-count, regularization or nonlinear search.
- Retain all full-data candidate results. Full-data selected relationships, projections and grouped means are exploratory descriptions after selection, not independently estimated association strengths.
- Check leave-one-strain-out and leave-one-recorded-species-out by refitting chemical scales, grouping, selection and all response coefficients within each training fold. Compare pooled 13-coordinate SSE with each fold's training neural mean. Folds remain internal checks conditional on previously estimated neural templates and chemical QC; they are not independent experimental validation. If training data cannot support an axis, predict its mean.
- Supplement the total-vector check with per-coordinate errors and membership overlap. Positive total-vector improvement does not establish that all coordinates carry chemical information.
- Preserve the previously selected Bacteroides ADF–ASH relationship as a separately identified anchor. The new complete-vector objective differs from the old contrast objective; do not compare their percentages as equivalent tasks.

## Figure rules

Use observed values. A candidate display is three chemical-rank groups per genus with all 13 mean unit coordinates centered on the genus mean, on one common neural color scale without row z-scores. Equal-width rank groups do not represent equal chemical intervals. Retain every strain in a supporting plot and tables. Any projected neural direction learned from full-data chemical association is disclosed as fitted, not treated as independent evidence. Main figure scope will follow the results without further model-family or subset searches. Dates remain in the metadata and are not a display axis.

Export editable SVG, PDF and high-resolution PNG, with English figure text and external caption. Save small notebook-callable code, plotting values and source hashes in this new output directory. Preserve existing notebooks, reports and raw inputs.
