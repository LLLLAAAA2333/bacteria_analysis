# Aggressive Analysis Refactor Design

Date: 2026-04-30
Status: Draft for user review
Topic: Replace the stage-oriented analysis workflow with a function-first scientific analysis library

## Goal

Refactor the current analysis code around the actual scientific sequence:

1. use anchor stimuli to assess batch/date effects;
2. compare neural and chemical RDMs, including significance and sample stability;
3. test chemical-class RDMs and their significance/stability.

The new code should be directly callable from Python and notebook-friendly. It
should not require Stage 1/2/3 outputs, review-script outputs, or large
intermediate tables as normal inputs.

## Motivation

The current repository contains useful results, but the useful path is hidden
inside exploratory scripts and old stage-derived artifacts. Several current
figures depend on prior review outputs such as `neural_median_peak_review_lr_merge`,
`neural_common_response_residualization_review`, `joint_consensus_rdm_review`,
or archived backup folders. That makes it difficult to answer a simple question:
"which code generated this figure, and how do I rerun only that analysis from
the source data?"

The project has also moved beyond the original `stage1 -> stage2 -> stage3`
framing. The current analysis logic is progressive and scientific rather than
pipeline-stage oriented. The code should reflect that.

## Non-Goals

- Do not preserve CLI ergonomics as a primary design goal.
- Do not maintain Stage 1/2/3 as the conceptual frame for new work.
- Do not generate large pair-level or permutation-null tables by default.
- Do not rewrite every legacy script in the first implementation slice.
- Do not treat `docs/ChatGPT-Guidance.md` as current project policy.
- Do not add heavy workflow engines, DAG frameworks, databases, or new large
  dependencies.
- Do not make paper-ready claims from this refactor. The outputs remain
  exploratory until the scientific dataset and preprocessing interpretation are
  finalized.

## Design Principles

- Functions first, notebooks second, CLI last.
- Source data and explicit parameters should be enough to run each main analysis.
- Intermediate data should live in memory unless explicitly requested.
- Saved outputs should be small, final, and inspectable.
- Each analysis function should return a result object that can be inspected,
  plotted, or saved.
- Shared utilities should be simple and boring: RDM alignment, distance metrics,
  permutations, resampling, plotting helpers.
- Legacy scripts may remain as historical references, but new code must not
  depend on their generated output directories.

## Proposed Package Structure

```text
src/bacteria_analysis/
  analysis_dataset.py
  neural_features.py
  chemical_features.py
  rdm.py
  stats.py
  analysis_results.py
  analysis_plotting.py

  analyses/
    __init__.py
    anchor_batch_effect.py
    rdm_alignment.py
    chemical_class_rsa.py
```

### `analysis_dataset.py`

Owns source-data loading and dataset filtering.

Primary public API:

```python
dataset = build_analysis_dataset(
    neural_path="data/202604/202604_data.parquet",
    matrix_path="data/matrix.xlsx",
    metadata_path="data/metabolism_raw_data.xlsx",
    exclude_dates=["20260331"],
)

anchor_dataset = build_anchor_dataset(
    neural_path="data/202604/202604_data_withbaseodor.parquet",
    anchor_stimuli=("s3_0", "s6_1", "s8_2"),
)
```

The returned `AnalysisDataset` should contain:

- filtered neural trial data loaded from the raw neural parquet;
- metabolite matrix;
- metabolite metadata;
- stimulus-to-sample mapping;
- dataset metadata such as included dates, excluded dates, number of stimuli,
  and default stimulus order.

The default dataset path should read directly from raw neural parquet and build
the needed trial-level features in memory. Preprocessing is usually fast enough
that new analysis code should not require existing preprocess output directories
as the public input contract. Preprocessed artifacts may still be used as an
internal optimization later, but they should not be required for the main API.

Anchor-stimulus data should be accepted as a separate dataset because it may use
different raw files, date filters, or stimulus panels from the main neural-
chemical analysis.

This module should replace scattered path defaults in review scripts.

### `neural_features.py`

Owns neural feature construction from raw/preprocessed neural rows.

Responsibilities:

- baseline-center if needed;
- merge non-ASE left/right neuron pairs;
- keep `ASEL`, `ASER`, `AWCOFF`, and `AWCON` separate;
- build view-specific response vectors;
- aggregate trials by stimulus using median by default;
- build neural RDMs.

Primary public API:

```python
neural = build_neural_rdm(
    dataset,
    view="response_window",
    aggregation="median",
    merge_lr=True,
    distance="correlation",
)
```

The function should return a compact object containing the RDM and the minimal
metadata needed to interpret it. It should not write files by default.

### `chemical_features.py`

Owns chemical matrix filtering, transformation, taxonomy grouping, and chemical
RDM construction.

Primary public APIs:

```python
chemical = build_chemical_rdm(
    dataset,
    qc_threshold=0.2,
    transform="log2",
    distance="euclidean",
)

class_rdms = build_chemical_class_rdms(
    dataset,
    taxonomy_level="Class",
    qc_threshold=0.2,
    min_features=3,
    transform="log2",
    distance="euclidean",
)
```

This module should compute the current broad baseline directly from
`matrix.xlsx` and `metabolism_raw_data.xlsx`; it should not read model RDMs from
prior results.

### `rdm.py`

Small shared RDM utilities:

- align square RDMs by shared labels;
- extract upper triangles;
- compute Spearman/Pearson similarity;
- rank-normalize values where needed for display;
- cluster or preserve display order.

This should absorb duplicated helpers currently repeated across review scripts.

### `stats.py`

Small shared statistical utilities:

- empirical one-sided p-values;
- label-shuffle nulls;
- date-preserving label-shuffle nulls;
- sample/stimulus subset resampling;
- fixed-model permutation summaries;
- Benjamini-Hochberg adjustment where needed.

This module should return arrays or summary frames in memory. Saving full null
draws should be opt-in.

### `analysis_results.py`

Defines result dataclasses and output saving.

Suggested result fields:

```python
@dataclass
class AnalysisResult:
    analysis_id: str
    parameters: dict[str, object]
    summary: pd.DataFrame | dict[str, object]
    tables: dict[str, pd.DataFrame]
    figures: dict[str, matplotlib.figure.Figure]
    diagnostics: dict[str, object]
```

Saving should be explicit:

```python
save_analysis_result(result, output_root, include_debug=False)
```

Default saved outputs:

- `summary.md`
- `summary.json`
- selected final tables;
- selected final figures;
- `parameters.json`.

Debug-only outputs:

- full pair-level tables;
- full permutation-null draws;
- large resampling draw tables.

### `analysis_plotting.py`

Shared plotting primitives for concise scientific figures:

- RDM heatmap panels;
- null distribution panels;
- class score/rank panels;
- sample-stability panels;
- anchor-stimulus date-effect panels.

Plotting functions should accept data frames/RDMs and return `Figure` objects.
They should not save files directly except through `save_analysis_result`.

## Main Analysis APIs

### 1. Anchor Batch Effect

Module: `bacteria_analysis.analyses.anchor_batch_effect`

Purpose:

Use repeated anchor stimuli to assess whether batch/date effects dominate the
neural representation.

Primary API:

```python
anchor = run_anchor_batch_effect(
    anchor_dataset,
    anchor_stimuli=("s3_0", "s6_1", "s8_2"),
    views=("response_window", "full_trajectory"),
    aggregation="median",
)
```

Main outputs:

- anchor stimulus coverage by date;
- date-by-anchor prototype RDM;
- same-anchor cross-date distance summary;
- date-effect versus stimulus-effect contrast;
- compact figures for prototype RDM, MDS, and neuron activity heatmaps.

Interpretation boundary:

This analysis does not prove there is no batch effect. It estimates whether
anchor stimuli show a dominant global date artifact and identifies where date
drift may be localized.

### 2. RDM Alignment

Module: `bacteria_analysis.analyses.rdm_alignment`

Purpose:

Build neural and chemical RDMs directly from source data, compare them, and test
whether the alignment is above relevant nulls.

Primary API:

```python
alignment = run_rdm_alignment(
    dataset,
    neural_view="response_window",
    neural_aggregation="median",
    chemical_qc_threshold=0.2,
    chemical_transform="log2",
    chemical_distance="euclidean",
    permutations=2000,
    subset_count=200,
    subset_fraction=0.8,
    seed=20260430,
)
```

Main outputs:

- neural RDM;
- broad chemical RDM;
- all-pair RSA;
- within-date and cross-date RSA;
- label-shuffle significance;
- date-preserving significance where date labels are available;
- sample/stimulus subset stability;
- compact RDM and null-distribution figures.

Current caution:

The current RDM rank similarity is not high, so the first implementation should
not freeze one alignment metric as final. It should expose a small set of
comparison summaries that can be inspected side by side:

- raw upper-triangle Spearman RSA;
- within-date and cross-date RSA;
- sample/stimulus subset stability;
- label-shuffle and date-preserving null context.

Within-date and cross-date RSA should be the primary date-structure summaries
because they are direct and easy to interpret. They are descriptive summaries,
not clean controls, because stimulus identity and date are confounded in the
current dataset. Cross-date RSA is therefore a stress test for cross-date
stability, not proof of or against generalization.

Date-pair-stratified rank RSA should not be implemented in the new core
workflow. It is a post-hoc sensitivity analysis with an unclear primary
scientific estimand, and it risks rank-normalizing away date/sample-composition
structure. It can be reconsidered only if a future question explicitly needs
that sensitivity analysis.

This keeps the analysis honest while leaving room to explore whether another
alignment comparison better captures the shared structure.

Default saved outputs should omit full pair-level and null-draw tables. Those
large tables are useful only under `debug=True`.

### 3. Chemical Class RSA

Module: `bacteria_analysis.analyses.chemical_class_rsa`

Purpose:

Ask which chemical taxonomy classes have stable neural-RDM alignment.

Primary API:

```python
classes = run_chemical_class_rsa(
    dataset,
    neural_rdm=alignment.rdms["neural"],
    taxonomy_level="Class",
    qc_threshold=0.2,
    min_features=3,
    fixed_permutations=2000,
    resamples=500,
    search_permutations=2000,
    top_k=5,
    seed=20260430,
)
```

Main outputs:

- observed class RSA scores;
- fixed-class permutation summary;
- reselection stability summary;
- search-corrected diagnostic summary;
- top-class RDM comparison;
- class-to-class chemical RDM similarity;
- class-versus-full chemical RDM similarity.

Interpretation boundary:

Fixed-class significance is conditional on the class being evaluated. Full
search permutation is a diagnostic for selection/search effects, not a reason
to overstate current exploratory results.

## Notebook Workflow

The preferred user-facing workflow is a small notebook sequence:

```text
notebook/
  01_anchor_batch_effect.ipynb
  02_rdm_alignment.ipynb
  03_chemical_class_rsa.ipynb
```

The notebooks should:

- import the function APIs;
- define dataset paths and key parameters near the top;
- display returned summaries and figures inline;
- save final outputs only when explicitly requested.

Notebook code should be thin. Scientific logic belongs in `src/`.

## Output Policy

The new default should not create large intermediate output trees.

Default output root for a saved run:

```text
results/<run_id>/
  anchor_batch_effect/
    summary.md
    summary.json
    parameters.json
    figures/
    tables/

  rdm_alignment/
    summary.md
    summary.json
    parameters.json
    audit/
    rdms/
    figures/
    tables/

  chemical_class_rsa/
    summary.md
    summary.json
    parameters.json
    audit/
    rdms/
    figures/
    tables/
```

Each analysis should save only final, likely-to-be-read tables and figures by
default. Large intermediate artifacts go under `debug/` only when explicitly
requested.

Default saved audit artifacts should be compact and sufficient to reproduce the
analysis:

- source manifest with input paths, file hashes when cheap to compute, git
  commit, package version when available, seeds, permutation counts, and key
  parameters;
- aligned stimulus order for every final RDM comparison;
- `n_pairs` by scope for all-pair, within-date, and cross-date summaries;
- retained metabolite lists for full chemical RDMs;
- retained feature lists for reported chemical classes;
- final aligned RDM matrices used in headline figures or summaries.

Final aligned RDM matrices are considered audit outputs, not disposable
intermediates. Save the RDMs used for the broad alignment and the reported
top-class/final-shortlist comparisons by default. Do not save every candidate
class RDM by default.

Examples of debug-only files:

- pair-level distance tables;
- every permutation draw;
- every resampling draw;
- all non-reported candidate RDM matrices.

## Relationship to Existing Code

Legacy code remains useful as reference but should no longer drive new analysis.

Current scripts to mine for reusable logic:

- `scripts/base_odor_date_effect_review.py`
- `scripts/plot_neural_chemical_rdm_foundation.py`
- `scripts/date_controlled_rsa_review.py`
- `scripts/date_effect_sampling_rsa_review.py`
- `scripts/taxonomy_class_stability_review.py`

Existing modules to reuse where possible:

- `src/bacteria_analysis/model_space.py`
- `src/bacteria_analysis/rsa.py`
- `src/bacteria_analysis/rsa_aggregated_responses.py`
- `src/bacteria_analysis/reliability.py`

The first implementation should copy or move only the stable, needed pieces.
Avoid preserving incidental exploratory behavior just because it exists.

After the new notebook workflow is validated, `scripts/run_rsa.py` may be frozen
as a legacy compatibility entry point. It should not receive new feature work
unless a specific comparison against the old workflow is needed.

## Migration Plan

### Slice 1: Build the new foundation

- Add `AnalysisDataset`.
- Make raw neural parquet the default input for the new dataset builder.
- Add `AnchorDataset` or an equivalent separate anchor-data input object.
- Add RDM utilities.
- Add neural feature/RDM construction for the current reference:
  non-ASE L/R merge, trial median, response-window and full-trajectory views.
- Add chemical RDM construction for the current broad baseline:
  `QCRSD <= 0.2`, `log2(matrix)`, Euclidean distance.

### Slice 2: Rebuild RDM alignment

- Implement `run_rdm_alignment` from source data.
- Recreate the current broad-baseline RSA values within a tolerance based on the
  relevant 99th-percentile/null or resampling context, rather than a brittle
  exact-match threshold for stochastic summaries.
- Keep alignment-comparison methods inspectable because the current rank
  similarity is modest and may not be the final best comparison.
- Add compact figures and explicit saving.
- Keep large pair/null tables debug-only.

### Slice 3: Rebuild anchor batch effect

- Implement `run_anchor_batch_effect`.
- Recreate the useful anchor figures from source data.
- Keep only the figures/tables needed to judge batch/date effects.

### Slice 4: Rebuild chemical class RSA

- Implement `run_chemical_class_rsa`.
- Recreate the current taxonomy class stability summary and main figures.
- Remove dependencies on `plot_biological_subspace_rdm_panel.py` and prior
  result directories.

### Slice 5: Notebook entry points

- Add three notebooks that call the new functions.
- Keep notebooks explanatory but thin.
- Link final notebooks to the current interpretation summary.

### Slice 6: Legacy demotion

- Add a short legacy note for old stage/review scripts.
- Stop using old output directories as inputs in new docs.
- Keep old results for audit only.

## Testing Strategy

Unit tests should cover:

- date filtering and stimulus/sample mapping;
- raw neural parquet loading for the new default dataset path;
- separate anchor-dataset loading;
- neural feature aggregation and L/R merge behavior;
- chemical QC filtering and log2 transform;
- RDM alignment and upper-triangle extraction;
- Spearman similarity and empirical p-value behavior;
- label-shuffle and date-preserving permutation reproducibility;
- class candidate selection and min-feature filtering;
- save behavior with and without `include_debug`.

Integration tests should use small synthetic fixtures to run:

- `run_anchor_batch_effect`;
- `run_rdm_alignment`;
- `run_chemical_class_rsa`.

One real-data smoke test may be kept outside default CI or marked slow. It
should verify that the filtered 202604 workflow can run end-to-end and produce
the expected result object structure. Stochastic regression checks should use a
99th-percentile or null-context tolerance instead of hard-coded exact values.

## Acceptance Criteria

- A user can run the three main analyses from Python without invoking CLI
  commands.
- The new analyses do not require old Stage 1/2/3 or review-output directories
  as inputs.
- Default execution does not write large intermediate tables.
- Saving is explicit and produces only final summaries, figures, and compact
  tables.
- The broad RDM alignment result approximately reproduces the current filtered
  202604 baseline under the agreed 99th-percentile/null-context tolerance.
- The chemical class result recovers `Purine nucleosides` as the leading current
  class under the current data and parameters.
- The code paths used by notebooks are the same code paths covered by tests.

## Resolved Decisions

- `build_analysis_dataset` should read directly from raw neural parquet by
  default. Existing preprocess outputs should not be required for the new public
  API.
- Anchor-stimulus data should be accepted as a separate dataset/input object.
- `scripts/run_rsa.py` can be frozen once the new notebook workflow is confirmed.
- Stochastic comparison tolerance can use the relevant 99th-percentile/null
  context rather than exact-value matching.
- Final aligned RDM matrices used for headline comparisons should be saved by
  default as audit outputs; non-reported candidate RDMs stay debug-only.

## Remaining Open Questions

- Which alignment comparison, beyond direct all-pair/within-date/cross-date RSA
  and sample stability, is worth exploring if raw rank similarity remains modest?
