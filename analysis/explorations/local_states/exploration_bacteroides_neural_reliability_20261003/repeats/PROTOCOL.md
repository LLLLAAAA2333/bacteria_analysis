# Fixed-template repeat reliability within Bacteroides

Protocol fixed before response comparisons, 2026-10-03. The user authorized neural two-dimensional stability and repeated-recording reliability, stopping at interpretation. No chemical input/model or new threshold/model search is used.

## Frozen inputs and estimands

- All 29 Bacteroides strains and original 13 neural coordinates from the existing saved unit-profile/context tables; preserve current species/date labels and taxonomy flags.
- Existing SNR=0.5 13 × 8 shared templates are frozen. Existing full-29 neural mean and PC1/PC2 loadings are frozen for displayed positions. The templates were fitted on the full existing atlas; these checks estimate reliability conditional on that representation, not end-to-end independent validation.
- Two fixed readable unit-coordinate measures: `ADF − ASH` and `AWB`. Keep full13 Euclidean differences, fixed PC1/PC2 individual coordinates, and joint two-dimensional positions. No half-specific z-scoring or score normalization.
- Source animal curves are the saved animal-mean 0–40 s curves, indexed by strain, date, animal (`date|worm_key`), and neuron. Trials are already averaged within animal. Whole animals retain all their strains/neurons in one half.

## Six cross-date pairs

For A011, A013, A014, A024, A025 and A044, retain both recorded dates in chronological order. Use the existing date-level gated and same-template pre-gate coefficients; all 13 coordinates observed and finite. Normalize each full date coefficient vector by its own L2 norm, then project using the frozen full-29 mean/loadings. Save all coefficients, norm, unit vectors, fixed measures, two-dimensional scores and the 13-dimensional difference.

For each measure, `same_RMS = sqrt(mean_i ||later_i − earlier_i||²)`. The descriptive between-strain reference is the RMS of all 15 unique differences between the six strains' two-date mean profiles, computed in the same representation. Different measurement precision, nonrandom six-strain selection, overlapping calendar dates, shared animals and unverified culture independence prevent this ratio being interpreted as a reliability coefficient or noise ceiling. Retain individual paired values and changes, species, flags and both dates.

## Existing 100 animal assignments and coverage

Reuse exactly the 100 saved date-balanced global whole-animal A/B assignments; do not redraw or pick splits. For each half, recompute condition means and individual-SNR from complete animal 40-s curves. Gate: `P=mean_t(mean_curve²)`, `V=mean_t(sample variance across animal curves, ddof=1)`, `SNR=sqrt(max(P−V/n,0)/V)`; zero-scatter positive signal gives infinity and zero signal gives zero. At least two animals are required per date × neuron. Project 8-bin means onto the frozen templates. SNR failures with adequate observations become zero; n<2 / missing observations remain NaN. Same-template pre-gate comparison requires identical n>=2 support.

Before aggregating dates, keep a date × neuron only when **both** halves have n>=2 there. Both halves then use the same shared dates, equally weighted within each strain × neuron. This matched-support intersection may omit an original date; retain shared-date counts and an `all_original_dates_complete13` flag. It is a conditional subset representation, not exactly the original all-date full-cohort estimand. Do not substitute different dates between halves. A coverage-only check before inspecting responses found common complete-13 strains per split min/median/max=13/20/21; complete support at every original date gives 9/15/15. No additional strict-support analysis is added.

Only strains with all 13 common coordinates and nonzero full coefficient vectors in both halves and both gated/pre-gate versions enter the primary unit metrics. Missing coordinates are never filled with zero for norm or projection. Save coverage for all 29 strains on every split, including missing/zero cases and per-cell date intersections. Save condition-level n, SNR, gate status and the gate disagreement fraction on shared eligible conditions.

## Repeat difference versus between-strain reference

Within each split, use exactly the same complete-strain cohort for all primary measures and both representations. For arrays `a_i`, `b_i`:

- `same_RMS = sqrt(mean_i ||a_i − b_i||²)`.
- `between_RMS = sqrt(mean_(i≠j) ||a_i − b_j||²)`, over all ordered different-strain pairs.
- Save `same_RMS / between_RMS`, the n strains, pair count, and Pearson/Spearman for scalar measures (correlation is secondary; undefined correlations remain missing).
- As a fixed diagnostic only, also save between_RMS and pair count restricted to strains with the same original complete recorded-date label set. Call this the **same-original-date-label reference**, not date/batch-controlled: cell-specific shared-date intersections may differ. No-pair cases are NaN.
- Full13 and fixed-PC1/PC2 joint distances are Euclidean norms; scalar differences keep their natural unit. Do not divide by sqrt(2), infer a full-sample noise ceiling, or equate between-strain distances with noise-free biological signal.

For each half separately, refit centered/unscaled PCA on the same jointly complete strains. Compare their top-two loading subspaces using both principal angles and save each half's first-two variance share. This tests plane orientation, separately from positions projected onto the frozen full-29 plane. Do not interpret near-origin cosine as a main reliability metric.

## Summaries, plots and checks

Summarize the 100 overlapping splits by median/5th/95th percentiles plus valid support counts; these percentiles are split sensitivity, **not confidence intervals or 100 independent experiments**. Also save per-strain distributions of paired differences and valid split counts, without forcing every strain into every split.

Use up to three English figures: six cross-date pairs plus their descriptive same/between reference; animal-split same/between RMS for fixed measures and 13D/2D; animal half-plane angle and coverage diagnostics. Preserve raw coefficients/norm as auxiliary fields for interpreting unit normalization. Save full results, source hashes and small callable functions; refuse to overwrite existing scientific result directories. Validate the frozen-template reconstruction against the existing full-condition table, independently recompute representative split metrics/support/gates, check labels and coverage, and visually inspect all figures. No raw data or earlier reports are edited.

## Explicit auxiliary summary clarification

The investigator-authorized raw-coefficient/norm check was made explicit before the auxiliary summaries were calculated: use unnormalized ADF−ASH, AWB and full13 coefficient norm, on exactly the same joint-complete13 animal cohort as all primary metrics. Save the same within/between RMS definitions and scalar Pearson/Spearman, plus the same six cross-date comparisons. This adds no threshold, model, alternative support cohort or selection search. Read the already saved half/date coefficients; do not refit anything. Units are ΔF/F0 coefficient space. Compare patterns, ratios and correlations to the unit-coordinate results, not absolute RMS magnitudes across these scales. Retain the primary unit/2D results unchanged.
