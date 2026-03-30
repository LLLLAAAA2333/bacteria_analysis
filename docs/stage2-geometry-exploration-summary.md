# Stage 2 Geometry Exploration Summary

Date: 2026-03-30  
Status: Exploratory notes for next-step decision making, not a final conclusion document

## Current Scope

This note summarizes the current Stage 2 dual-view MVP outputs for:

- `response_window`
- `full_trajectory`

The goal here is not to claim that the neural geometry is already fully validated. The goal is to record what currently looks real, what is still uncertain, and what should be checked next.

This summary is based on the latest Stage 2 outputs after:

- correcting `b5_2` label from `A137 stationary` to `A138 stationary`
- updating pooled RDM figures to use `stim_name`
- adding clustered pooled RDM display figures

Current inspected output root:

- `.tmp_after_label_fix/stage2_geometry/`

## Outputs Reviewed

Main figures reviewed:

- `figures/rdm_matrix__response_window__pooled.png`
- `figures/rdm_matrix__response_window__pooled__clustered.png`
- `figures/rdm_matrix__full_trajectory__pooled.png`
- `figures/rdm_matrix__full_trajectory__pooled__clustered.png`
- `figures/rdm_stability_by_individual.png`
- `figures/rdm_stability_by_date.png`
- `figures/rdm_view_comparison.png`

Main tables reviewed:

- `tables/rdm_pairs__response_window__pooled.parquet`
- `tables/rdm_pairs__full_trajectory__pooled.parquet`
- `tables/rdm_stability_by_individual.parquet`
- `tables/rdm_stability_by_date.parquet`
- `tables/rdm_view_comparison.parquet`
- `qc/rdm_group_coverage.parquet`

## Core Readout

### 1. Pooled geometry is the strongest current signal

The pooled RDMs show visible structure in both views. The pattern does not look like a random unstructured matrix.

The strongest quantitative support is the pooled cross-view similarity:

- pooled `response_window` vs pooled `full_trajectory`: `0.919574` Spearman

This is high enough to treat the pooled geometry as a real and view-stable signal at the current stage.

There is also qualitative agreement in the nearest stimulus pairs. Among the 15 closest off-diagonal stimulus pairs, 13 are shared between the two views. Repeated examples include:

- `A160 stationary` and `A138 stationary`
- `A179 stationary` and `A138 stationary`
- `A085 stationary` and `A160 stationary`
- `A085 stationary` and `A138 stationary`
- `A085 stationary` and `A137 stationary`

This does not prove biological meaning yet, but it does support the claim that the pooled geometry is not arbitrary.

### 2. Individual-level geometry is only moderately reproducible

The individual-level stability summary is clearly weaker than the pooled result.

Within-individual-type pairwise RDM similarity:

- `response_window`: median `0.500000`
- `full_trajectory`: median `0.500000`

Pooled-vs-individual similarity:

- `response_window`: median `0.671429`
- `full_trajectory`: median `0.714286`

Current interpretation:

- many individual RDMs still resemble the pooled geometry when overlap exists
- direct individual-to-individual geometry agreement is only moderate
- this is not strong enough to claim that individual geometry is highly stable

However, these numbers should not be read in isolation. Coverage is sparse enough that many comparisons are only weakly supported.

### 3. Date-level numbers look better than individual-level numbers, but are much less secure than they first appear

Pooled-vs-date similarity:

- `response_window`: median `0.903571`
- `full_trajectory`: median `0.746429`

Date-to-date within-type similarity:

- `response_window`: median `0.000000`
- `full_trajectory`: median `0.750000`

At first glance, this could look like date-level geometry is as good as or better than individual-level geometry. I do not think that is the right reading.

The reason is support. There are only 6 dates total, and only 2 valid date-to-date comparisons in each view:

- valid date-date comparisons: `2`
- invalid date-date comparisons: `13`

So the date-level summary is not stable enough yet to support a strong claim.

## Why The Group-Level Stability Is Hard To Read

### 1. Stimulus coverage is highly fragmented across groups

Per-individual stimulus coverage is narrow:

- number of individuals: `37`
- median distinct stimuli per individual: `4`
- lower quartile: `2`
- upper quartile: `6`
- minimum: `1`
- maximum: `10`

Per-date stimulus coverage is also uneven:

- number of dates: `6`
- median distinct stimuli per date: `3.5`
- minimum: `1`
- maximum: `10`

Actual date stimulus panels are:

- `20260106`: `4` stimuli
- `20260114`: `1` stimulus
- `20260115`: `3` stimuli
- `20260126`: `3` stimuli
- `20260127`: `6` stimuli
- `20260128`: `10` stimuli

This means many group-level RDM comparisons are structurally underpowered before biological variability is even considered.

### 2. Many stability scores are based on very few shared entries

Median shared upper-triangle entries for valid comparisons:

- within-individual comparisons: `3`
- within-date comparisons: `3`
- pooled-vs-individual: `6`
- pooled-vs-date: `6`

This is a major limitation. A similarity score computed from only 3 shared entries is better interpreted as a weak indication than as a stable estimate.

### 3. Invalid comparisons are common, especially within-group comparisons

For individual-to-individual comparisons:

- valid: `119`
- invalid: `547`

For date-to-date comparisons:

- valid: `2`
- invalid: `13`

So the group-level instability story is partly about true variability, but it is also strongly driven by incomplete overlap.

## How I Would Interpret Stage 2 Right Now

### Strongest supported statements

- There is a visible and quantitatively reproducible pooled neural geometry.
- `response_window` and `full_trajectory` tell a very similar pooled geometry story.
- The pooled RDM is likely a useful descriptive object for Stage 2 and later Stage 3 planning.

### Supported but only cautiously

- Many individual/date-specific RDMs are directionally consistent with the pooled RDM.
- `full_trajectory` does not break the geometry story; it broadly agrees with `response_window`.

### Not yet supported

- A strong claim that geometry is robustly reproducible across individuals.
- A strong claim that geometry is robustly reproducible across dates.
- A biological interpretation of clustered blocks in the pooled heatmap.
- A claim that low group-level similarity mainly reflects biology rather than sampling/coverage structure.

My current working judgment is:

> Stage 2 currently supports a pooled descriptive geometry much more strongly than it supports a clean group-level stability claim.

That is still a useful Stage 2 outcome. It just means Stage 3 should be planned with that limitation in mind.

## About The Diagonal

The diagonal entries in the current RDM figures are not `1`, and they are not forced to `0`.

That is expected in the current exploratory setup because:

- the matrix is a dissimilarity matrix, not a similarity matrix
- distances are based on `1 - correlation`
- diagonal entries summarize same-stimulus distances across different trials, not a trial compared to itself

So the diagonal is currently carrying within-stimulus trial variability information. For the exploration phase, keeping that information visible is reasonable.

## What I Do Not Want To Over-Read

- The clustered pooled heatmaps are display aids, not new evidence.
- The visible blocks in the pooled RDM are worth noticing, but they should not yet be named as biological clusters.
- The better-looking date pooled-vs-group scores should not be taken as proof that date effects are small.
- The weaker individual-to-individual scores should not be taken as proof that individual differences are large.

At the moment, both of those latter patterns are confounded by sparse stimulus overlap.

## Main Risks Carrying Forward

### 1. Stage 3 could over-inherit pooled structure without enough group-level validation

If RSA is run directly on the pooled RDM, it may produce interpretable-looking results even if individual/date stability remains only partially established.

### 2. Group-level instability may be partly a design artifact

The stimulus panel is not well balanced across dates and individuals. This makes Stage 2 stability look weaker even before asking the biology question.

### 3. Heatmap readability can create false confidence

The pooled matrices are visually convincing. That makes it easy to over-weight them relative to the much weaker group-level support.

## Suggested Next Steps

### Priority 1: make the overlap problem explicit

Write one small QC summary that shows stimulus coverage overlap across:

- date x stimulus
- individual x stimulus

The current `rdm_group_coverage.parquet` shows trial counts, but not the overlap structure that actually determines whether group RDM comparisons are meaningful.

### Priority 2: define an interpretation threshold for RDM similarity

Before drawing stronger conclusions, set a rule for how many shared upper-triangle entries are required before a similarity score is considered interpretable.

Right now, many valid scores are based on only `3` shared entries, which is too fragile for strong claims.

### Priority 3: separate pooled-useful from group-validated

For later planning, it may help to explicitly split the Stage 2 conclusion into:

- pooled geometry is usable as a descriptive summary
- group-level stability is only partially validated

That framing will make Stage 3 discussions cleaner.

### Priority 4: decide whether to do a restricted-overlap follow-up

One possible follow-up is to compare only groups that share a minimally adequate stimulus subset. The point would not be to maximize sample size, but to test whether stability improves once overlap is less pathological.

This may still be limited by the data, but it would answer whether the current instability is mostly a geometry problem or mostly a support problem.

## Bottom Line

My current Stage 2 read is:

- pooled neural geometry looks real
- pooled geometry is highly consistent across `response_window` and `full_trajectory`
- group-level stability is not yet strong enough to read cleanly because stimulus overlap is too sparse
- Stage 2 is already scientifically useful as an exploratory geometry description
- Stage 2 is not yet a strong proof of cross-group geometry stability

That is enough to continue, but the next step should focus on overlap-aware interpretation rather than immediately escalating to strong biological claims.
