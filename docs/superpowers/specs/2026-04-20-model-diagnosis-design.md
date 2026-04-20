# Model Diagnosis Beyond Global Profile Design

Date: 2026-04-20
Status: Draft for user review
Topic: Diagnose neural and chemical representation mismatch after treating global profile as a weak baseline

## Goal

Build a focused diagnostic package for the filtered 202604 batch that explains
where existing chemical/model spaces align with the neural RDM and where they
fail.

The main shift is interpretive:

- `global_profile` is no longer the primary explanatory model
- `global_profile` remains a weak baseline and control
- the next analysis should estimate neural RDM reliability, select reliable
  stimulus pairs, and diagnose residual mismatches between neural and chemical
  spaces

The first implementation should prioritize pair-level diagnosis over searching
for a newly optimized model.

## Fixed Analysis Contract

Use one primary reference for this work:

- dataset: filtered `202604_without_20260331`
- preprocess root: `data/202604/202604_preprocess_without_20260331`
- output root: `results/202604_without_20260331/model_diagnosis`
- neural reference: non-ASE L/R merge plus trial-median aggregation
- views: `response_window` and `full_trajectory`
- weak baseline: `global_profile_default_correlation`
- broad chemical baseline: `QCRSD <= 0.2 + log2(matrix) + Euclidean`

The full 202604 batch should stay out of the primary analysis. It may be
mentioned as background, but the main conclusions should use the filtered
`202604_without_20260331` reference.

The filtered batch does not support a clean LODO result. Date-level claims must
therefore be phrased as date-controlled or date-pair diagnostics, not as strict
cross-date generalization.

## Non-Goals

- Do not change production RSA defaults in this slice.
- Do not replace the main `rsa` command or Stage 3 output contract.
- Do not add heavy dependencies.
- Do not treat supervised embedding as a main result in the first pass.
- Do not claim that any learned embedding is the correct biological chemical
  model.
- Do not promote global profile back to the main explanatory model.

## Design Summary

The diagnostic package has two sections.

Section 1 estimates neural RDM reliability and builds a reliable-pair map. Its
main output is a table that identifies which stimulus pairs have stable neural
distances and can be interpreted in model residual analysis. It also reports an
overall neural RDM ceiling.

Section 2 uses the reliable-pair map to diagnose model residuals. It identifies
neural-far/model-near and neural-near/model-far stimulus pairs, summarizes
whether residuals concentrate by date pair or annotation group, and compares
fixed chemical models and a small unsupervised chemical embedding scan.

## Section 1: Neural Reliability Ceiling and Reliable-Pair Selection

### Purpose

Estimate which neural RDM pair distances are stable enough to interpret.

The primary output is not only one ceiling number. The primary output is a
pair-level reliability map:

- reliable pairs can support residual interpretation
- unreliable pairs should not drive model mismatch conclusions
- the overall ceiling contextualizes current model RSA values

### Inputs

Use filtered preprocess outputs:

- `trial_level/trial_metadata.parquet`
- `trial_level/trial_tensor_baseline_centered.npz`
- optional existing helper outputs if they match the current neural reference

The neural construction must match the current reference:

- merge non-ASE L/R neurons
- keep ASEL and ASER separate
- aggregate trials with median, not mean
- build RDMs for `response_window` and `full_trajectory`

### Split Strategy

Run repeated split-half RDM construction.

Trial split:

- split trials within each stimulus into A/B halves
- build one prototype per stimulus per half
- generate `rdm_A` and `rdm_B`
- repeat with deterministic seeded random splits

Individual or worm split:

- split by recording individual where feasible
- avoid placing trials from the same individual into both halves for a given
  stimulus when the data support it
- fall back conservatively when a stimulus has too few individuals

Date split:

- record date labels and date-pair support
- do not use date split as the main ceiling because stimulus and date are mostly
  confounded in this filtered batch

### Overall Ceiling Metrics

For each view and split family, calculate:

- split-half RDM Spearman correlation
- Spearman-Brown corrected reliability
- number of valid stimulus pairs
- number of valid split iterations
- observed RSA divided by the reliability estimate for selected model RDMs

The ceiling should be reported separately for `response_window` and
`full_trajectory`.

### Pair-Level Reliability Metrics

For each stimulus pair, calculate:

- `stimulus_left`
- `stimulus_right`
- `date_left`
- `date_right`
- `date_pair`
- `date_pair_type`
- mean neural distance across splits
- split distance standard deviation
- split distance confidence interval
- rank stability across splits
- valid split count
- reliability tier

### Reliability Tiers

Use simple, auditable first-pass tiering.

Recommended tiers:

- `high`: stable rank, low split variance, enough valid split iterations
- `medium`: usable for primary residual analysis with cautious interpretation
- `low`: excluded from the main residual list, kept for supplementary diagnosis
- `insufficient`: not enough support to interpret

Thresholds should be quantile based in the first implementation. For example:

- top 25 percent of pair stability becomes `high`
- top 50 percent becomes the main `high + medium` pool
- pairs below the support threshold become `low` or `insufficient`

This avoids hard-coding arbitrary biological cutoffs before seeing the
distribution.

### Section 1 Outputs

Write:

- `tables/neural_rdm_ceiling_summary.csv`
- `tables/neural_pair_reliability.csv`
- `tables/neural_pair_reliability_by_date_pair.csv`
- `figures/neural_ceiling_by_view.png`
- `figures/pair_reliability_vs_distance.png`
- `run_summary.md`

### Section 1 Interpretation

This section should answer:

- which neural pair distances are reliable enough to interpret
- how high the neural RDM ceiling appears to be
- whether current model RSA values are close to or far below that ceiling
- which date pairs have weak neural support

## Section 2: Residual Pair Analysis and Model Diagnosis

### Purpose

Diagnose where model RDMs fail against reliable neural pair distances.

The analysis should identify two mismatch classes:

- neural-far/model-near pairs
- neural-near/model-far pairs

It should also identify whether residuals concentrate by date pair, base odor,
taxonomy, metabolite feature group, or model family.

### Model Set

First-pass models:

- `global_profile_default_correlation` as the weak baseline
- `global_qc20_log2_euclidean` as the broad chemical baseline
- selected taxonomy or subspace RDMs that were already scored under the current
  neural reference
- current weighted fusion RDM as an exploratory comparator

Weighted fusion should not be presented as the main biological explanation. It
is useful as a diagnostic comparator for whether mixed subspace structure is
more aligned with the neural RDM than a single global geometry.

### Residual Definition

Use rank residuals rather than raw distance residuals because RDM scales differ
across model families.

Global residual:

```text
residual = neural_rank_pct - model_rank_pct
```

Interpretation:

- positive residual: neural-far/model-near
- negative residual: neural-near/model-far
- near-zero residual: neural and model rank order agree

Date-pair-stratified residual:

```text
date_pair_stratified_residual =
  neural_rank_within_date_pair - model_rank_within_date_pair
```

This reduces the effect of date-pair composition when diagnosing pair-level
mismatch.

### Pair Pools

Use Section 1 reliability tiers to define interpretation pools:

- primary pool: `high + medium`
- high-confidence supplement: `high`
- excluded diagnostic pool: `low + insufficient`

The main residual tables and figures should default to the primary pool. Low
and insufficient pairs can be listed, but should not drive the main conclusion.

### Residual Pair Fields

Each residual pair table should include:

- `view`
- `model_id`
- `stimulus_left`
- `stimulus_right`
- `date_left`
- `date_right`
- `date_pair`
- `date_pair_type`
- `neural_distance`
- `neural_reliability_tier`
- `model_distance`
- `neural_rank_pct`
- `model_rank_pct`
- `residual`
- `date_pair_stratified_residual`
- `residual_direction`
- available base odor annotation
- available taxonomy or metabolite group annotation
- whether the pair is also extreme under other models

### Section 2 Outputs

Write:

- `tables/residual_pairs__<view>__<model>.csv`
- `tables/top_residual_pairs__<view>__<model>.csv`
- `tables/residual_summary_by_date_pair.csv`
- `tables/residual_summary_by_taxonomy_or_feature_group.csv`
- `figures/residual_scatter__<view>__<model>.png`
- `figures/top_residual_pair_heatmap__<view>__<model>.png`
- `figures/residual_by_date_pair.png`
- `run_summary.md`

### Section 2 Interpretation

This section should answer:

- where `global_profile` fails as a weak baseline
- whether `QCRSD <= 0.2 + log2(matrix) + Euclidean` reduces those failures
- whether residuals concentrate by date pair
- whether residuals concentrate by annotation group
- whether top residual pairs are reliable enough to interpret

## Embedding Diagnosis Scope

The first pass should include conservative fixed-model comparison and a small
unsupervised embedding scan.

### Level 1: Fixed Model Comparison

Compare fixed RDMs under one reporting contract:

- all pairs
- reliable pairs only
- high-reliability pairs only
- within-date pairs
- cross-date pairs
- date-pair-stratified ranks
- date-preserving permutation

This is the main model comparison layer.

### Level 2: Unsupervised Chemical Embedding Scan

Run PCA on the QC-filtered log2 matrix without using neural labels.

Recommended grid:

- dimensions: `2`, `3`, `5`, `10`, `20`
- distance: Euclidean in embedding space

For each embedding RDM, run the same reporting contract as fixed models.

Only interpret an embedding as helpful if improvement is consistent across:

- both neural views
- reliable-pair analysis
- date-controlled summaries

Do not over-read a pooled all-pairs improvement if it disappears under reliable
pairs or date-pair stratification.

### Level 3: Supervised Embedding

Do not implement supervised embedding in the first pass.

Potential later methods include:

- PLS
- ridge projection
- kernel alignment

If added later, they must use held-out pair, held-out stimulus, or held-out
date-pair validation plus date-preserving permutation. They should remain
exploratory unless they generalize under those controls.

## Output Organization

Use one diagnostic output root:

```text
results/202604_without_20260331/model_diagnosis/
|-- run_summary.md
|-- tables/
|-- figures/
`-- qc/
```

The script may write additional intermediate files under `qc/` if they help
audit split construction and pair support.

## Implementation Recommendation

Keep this as a focused diagnostic review package.

Recommended first implementation:

- one script: `scripts/model_diagnosis_review.py`
- small local helper functions inside the script unless reuse becomes necessary
- no production RSA default changes
- no broad refactor
- no new heavy dependencies

Reusable code may be extracted later if this workflow becomes part of the main
pipeline.

## Testing Strategy

Add focused tests for the core math with small synthetic inputs:

- split-half RDM correlation
- Spearman-Brown correction
- pair reliability tier assignment
- global rank residual direction
- date-pair-stratified residual
- reliable-pair filtering
- PCA embedding RDM construction

Tests should not depend on the full real dataset.

## Acceptance Criteria

The diagnostic package is complete when:

- filtered `202604_without_20260331` is the only primary dataset used
- Section 1 writes neural ceiling and pair reliability outputs for both views
- Section 2 writes residual outputs for the weak baseline and broad chemical
  baseline
- fixed model comparison reports all-pairs, reliable-pair, high-reliability,
  within-date, cross-date, and date-pair-stratified summaries
- PCA embedding scan reports whether unsupervised chemical denoising improves
  alignment under the same controls
- `run_summary.md` states which pairs are reliable, where the models fail, and
  whether any embedding improvement is robust enough to discuss

## Interpretation Contract

The final write-up should avoid claiming that the analysis found the correct
chemical model.

It should answer:

1. Which neural pair distances are reliable enough to interpret?
2. How high is the neural RDM reliability ceiling?
3. Does current model RSA sit near or far below that ceiling?
4. Where does global profile fail as a weak baseline?
5. Does the QC20 log2 Euclidean chemical baseline reduce those failures?
6. Are residual mismatches concentrated by date pair, taxonomy, base odor, or
   metabolite group?
7. Does unsupervised chemical embedding improve alignment robustly, or only in
   pooled all-pairs RSA?

## Recommended First Implementation Slice

Implement in this order:

1. Build neural split RDMs for both views using the fixed neural reference.
2. Write ceiling summary and pair reliability table.
3. Build residual tables for `global_profile_default_correlation` and
   `global_qc20_log2_euclidean`.
4. Add residual summaries by date pair and annotation group.
5. Add fixed-model comparison on reliable-pair subsets.
6. Add PCA embedding scan.
7. Add figures and final `run_summary.md`.

This order keeps the work diagnosis-first and prevents embedding search from
running ahead of the reliability evidence.
