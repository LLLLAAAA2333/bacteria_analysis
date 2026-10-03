# Fixed-selected-axis diagnostics

Saved before receiving fitted model outputs, 2026-10-03. This diagnostic subtree implements the already frozen parent protocol and does not select another chemical state, response, or subset.

Read the model worker's saved full-cohort scores/targets, selected state members with source annotations, and held-out predictions/performance. Preserve all 29 strain IDs and labels, including taxonomy flags and strains without complete animal-split reliability support. Full chemical state definition and fold fitting are owned by `model/`; no chemical clustering or held-out model fitting is repeated here.

For the fixed full-cohort selected chemical score x, save Pearson/Spearman and centered OLS slope/intercept for the primary unit ADF−ASH, prespecified raw and pre-gate alternatives, and all 13 unit coordinates. Save x quartiles and fitted change over its observed IQR. These are descriptive full-fit associations, without significance tests or confidence intervals.

Order by x ascending then strain ID. Assign the first 10 strains LOW, next 10 MID, final 9 HIGH. Keep individual values and labels; summarize n, mean, median, min/max, species/date-set composition and taxonomy flags. Plot individual overlap, not only group averages. These groups are score-rank descriptions, not concentration doses or validated neural categories.

With the full-data chemical state and its score values frozen, delete one strain at a time, then each recorded species, and recompute only primary-response descriptive slope/correlations. This is fixed-axis influence, distinct from the model worker's fully refitted held-out validation. Save deleted identities, train n, association and IQR-scaled slope. Report leverage and individual extreme-score deletions rather than introduce a new extreme-removal cutoff.

For recorded species, then exact original recorded-date-set labels, keep groups having at least 2 strains, subtract each group's x and primary-y mean, and calculate pooled demeaned Pearson/Spearman and slope. Save individual residuals, group membership/size, retained group/strain n and sum(n_group−1). This is a limited within-group description, not complete species/date adjustment; correlations treat no new independent experimental units.

Keep all 13 unit coordinates in the fixed chemical order using a shared zero-centered color scale without row/column SD scaling. Show ADF and ASH separately on that same x and summarize all 13 coordinate slopes over the same x IQR. Increasing the unit contrast alone does not establish simultaneous opposite biological effects. Same-axis raw coefficients are unnormalized template coefficients, not original calcium curves.

Describe the selected state's actual annotation and Mass-column family count, score weights, and available superclass/class composition. A heterogeneous group is not renamed a pathway. It is an observational chemical state; do not infer dose sensitivity, potency, causality or behavior.

Create three English PNG/SVG figures: full-fit and held-out primary relationship; separate ADF/ASH plus rank thirds and all13 slope context; chemical-ordered all13 heatmap with direct species/date labels. Preserve strain/date context and source flags. Personally inspect images. Save input SHA-256, numeric checks and callable functions. Refuse overwrite of existing scientific outputs; plotting from saved diagnostic tables may use a new empty directory. Do not create or run notebooks.

For clarity about normalization, also describe each saved unnormalized ADF and ASH template coefficient against the same frozen chemical x. This is an auxiliary two-coordinate description, without chemical state selection, another held-out model, or changes to the primary unit contrast. An opposing direction in unit space alone is not evidence of opposing absolute response amplitudes.

## Wording clarification after independent review

The earlier term “outcome-independent” refers only to the rank-third cutpoint rule conditional on an already selected x. The full-cohort axis itself was selected using the full primary response. Therefore rank thirds, fixed-axis deletion diagnostics and component associations remain same-data descriptions; none is an independent validation. No membership, cutpoint, numerical result or analysis method changes with this clarification.
