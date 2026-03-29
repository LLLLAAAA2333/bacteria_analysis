# Stage 1 Reliability Design

Date: 2026-03-28
Status: Drafted after user-approved design
Topic: Validate whether bacterial metabolites evoke reliable chemosensory population representations

## Goal

Establish that the dataset contains a reproducible neural representation of stimulus identity.

This stage is not about clustering and not yet about biochemical meaning. It is the reliability gate for all later stages.

The core claim to test is:

> trials from the same stimulus should be more similar to each other than trials from different stimuli, and this relationship should generalize across individuals and dates.

## Current Context

Stage 0 is complete. Preprocessing outputs already exist:

- `data/processed/clean/neuron_segments_clean.parquet`
- `data/processed/trial_level/trial_metadata.parquet`
- `data/processed/trial_level/trial_wide_baseline_centered.parquet`
- `data/processed/trial_level/trial_tensor_baseline_centered.npz`
- `data/processed/qc/preprocessing_report.json`
- `data/processed/qc/preprocessing_report.md`

Current verified dataset facts:

- `678` trials
- `22` neurons
- `45` time points
- mild filtering already applied
- fully-NaN traces removed, partially missing traces preserved

## Scientific Question

Does the worm chemosensory population encode stimulus identity in a way that is reproducible across repeated trials, across individuals, and across experiment dates?

## Scope

This stage covers:

- definition of trial-level representations for reliability testing
- same-stimulus versus different-stimulus distance analysis
- cross-individual generalization
- date-level generalization
- split-half reliability as supplementary evidence
- permutation nulls and grouped bootstrap uncertainty

This stage does not cover:

- stimulus clustering as a primary result
- RDM interpretation
- biochemical model matching
- neuron contribution analysis
- behavior prediction

## Individual Definition

The project must not treat `worm_key` alone as an individual identifier.

For Stage 1:

`individual_id = "{date}__{worm_key}"`

This is the unit for:

- grouped bootstrap
- leave-one-individual-out validation
- individual-level summaries

## Trial Representation Views

Stage 1 uses a multi-view design. Reliability must be evaluated in all of the following representations:

1. **Full trajectory**
   - time points `0..44`
2. **ON window**
   - time points `6..15`
3. **Response window**
   - time points `6..20`
4. **Post window**
   - time points `16..44`

These are not alternative projects. They are parallel views of the same Stage 1 question.

### Formal Reporting Rule

For the formal Stage 1 conclusion:

- use a single primary view, with **`response_window`** as the default
- allow explicit manual override to `full_trajectory`, `on_window`, or `post_window` when needed
- treat `full_trajectory`, `ON window`, and `post window` as supplementary robustness checks

This rule is chosen to keep the formal evidence chain short and biologically interpretable while still preserving the other views for sensitivity analysis. Any manual override must be recorded in the run summary and stated in the report text.

## Primary Analysis Unit

The primary analysis unit is one trial-level neural trajectory:

`trial_id x stimulus x neuron x time`

All Stage 1 statistics should be based on trial-level representations, not early stimulus-averaged prototypes.

## Distance Metric

Primary distance:

- `correlation distance`

Sensitivity analysis:

- optional Euclidean distance as a secondary check

Correlation distance is the default because Stage 1 is asking whether the population response pattern is reproducible, not whether absolute magnitude alone is stable.

## Missing-Neuron Handling

Missing neurons are not imputed.

When comparing two trials:

- only neurons observed in both trials are used
- distance is computed on the overlapping neuron set only
- the overlap neuron count must be recorded as a QC quantity for every comparison

This is the agreed default rule for Stage 1.

## Core Reliability Analyses

### A. Same vs Different Distance Test

For each representation view:

- compute pairwise trial distances
- separate comparisons into:
  - same-stimulus
  - different-stimulus
- compare the distributions

Primary expectation:

- same-stimulus distances are smaller than different-stimulus distances

### B. Leave-One-Individual-Out Generalization

This is the primary validation analysis.

For each held-out `individual_id`:

- use the remaining individuals as the reference set
- compare held-out trial representations against stimulus-specific reference representations built from the training individuals
- quantify whether held-out trials are closer to their own stimulus than to other stimuli

Primary expectation:

- stimulus reliability remains above chance on held-out individuals

### C. Leave-One-Date-Out Generalization

Repeat the same logic at the date level:

- hold out one `date`
- use all remaining dates as the reference set
- test whether the held-out date preserves the same stimulus structure

Primary expectation:

- reliability is not driven by one experimental date

### D. Split-Half Reliability

This is supplementary, not the main result.

For each representation view:

- randomly split eligible trials into two balanced halves
- build half-level stimulus representations
- compare same-stimulus versus different-stimulus distances across halves
- repeat many times

Primary expectation:

- split-half reliability is consistent with the cross-individual result

### Supplementary Diagnostics Kept Out of the Formal Evidence Chain

The following analyses may be useful for interpretation, but they are not part of the formal Stage 1 pass/fail evidence chain:

- same-vs-different analysis restricted to different individuals within the same `date`
- per-date LOIO analyses
- stimulus-by-stimulus distance matrices

### E. Label-Shuffle Null

This is the main significance framework.

For each main Stage 1 statistic:

- shuffle stimulus labels under a structure-preserving scheme
- recompute the reliability statistic many times
- compare the observed value to the null distribution

Primary expectation:

- the observed statistic exceeds the shuffled null

## Statistical Framework

### Permutation

Use permutation-based null testing for the main reliability statistics.

The label-shuffle procedure must preserve:

- the trial structure
- the individual grouping structure
- the date structure as much as possible

The implementation should avoid destroying the nested design.

### Grouped Bootstrap

Use grouped bootstrap to quantify uncertainty.

Bootstrap unit:

- `individual_id = date + worm_key`

For each bootstrap iteration:

- resample individuals with replacement
- carry all of each sampled individual's trials with them
- recompute the Stage 1 statistic

Primary output:

- bootstrap mean
- confidence interval

## Main Statistics to Report

For the formal Stage 1 conclusion, the primary reporting set should be:

1. same-stimulus mean distance
2. different-stimulus mean distance
3. distance gap:
   - `different - same`
4. leave-one-individual-out reliability score
5. leave-one-date-out reliability score
6. permutation p-value or equivalent null percentile
7. grouped-bootstrap confidence interval
8. overlap-neuron-count summary

`split-half` should still be reported, but only as supplementary support.

## QC Requirements

Every Stage 1 analysis must record:

- number of trials used
- number of individuals used
- number of dates used
- overlap neuron count distribution
- any stimuli excluded due to insufficient comparable trials

If a representation view fails due to insufficient overlap or insufficient held-out coverage, that failure must be reported explicitly rather than silently dropped.

## Visualization Requirements

Stage 1 figures should be descriptive but targeted.

Formal figure families:

1. same-stimulus vs different-stimulus distance distributions
2. leave-one-individual-out reliability summary
3. leave-one-date-out reliability summary
4. overlap-neuron-count QC summary

Formal figures should use the selected primary view as the main display view. The default primary view is `response_window`.

For the main same-vs-different figure:

- include permutation-based `p` value annotation
- do not overlay all pairwise points by default, because pairwise distances are numerous and not independent observations

Supplementary figure families:

1. split-half reliability summary
2. reliability comparison across:
   - full trajectory
   - ON window
   - response window
   - post window
3. stimulus-by-stimulus distance matrix, if later requested

Optional figure:

- PCA trajectory plot for qualitative illustration only

PCA is explicitly supplementary and not the main statistical evidence for Stage 1.

## Table Outputs

Stage 1 should produce at least:

1. a trial-comparison results table
2. a per-view summary table
3. a leave-one-individual-out summary table
4. a leave-one-date-out summary table
5. a permutation/bootstrap summary table

## Exit Criteria

Stage 1 is considered successful if:

- same-stimulus distances are consistently smaller than different-stimulus distances
- the effect is present in at least one biologically meaningful representation view
- the result survives leave-one-individual-out validation
- the result is not abolished by leave-one-date-out validation
- the observed statistics exceed shuffled-label nulls
- grouped-bootstrap intervals support a stable positive reliability effect

Stage 1 is considered inconclusive if:

- results exist only in split-half but disappear across held-out individuals
- results depend entirely on one date
- overlap-neuron constraints make most comparisons unstable

## Recommended Interpretation Rules

- If full trajectory and response window are both reliable, prioritize the simpler and more interpretable view in the later stages.
- If only one view is reliable, carry that view forward as the Stage 2 main input and keep the others as supplementary.
- If no view is reliable, do not advance to Stage 2 as if reliability had already been established.

## Non-Goals

Stage 1 should not claim:

- that stimuli form biologically meaningful clusters
- that a particular biochemical property is encoded
- that any one neuron is causally responsible

Those belong to later stages.

## Immediate Next Step After This Spec

The next document should be a Stage 1 implementation plan that specifies:

- exact reliability statistic formulas
- exact permutation scheme
- exact grouped-bootstrap procedure
- exact output file paths
- exact figure and table filenames
