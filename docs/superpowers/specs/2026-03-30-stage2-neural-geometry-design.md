# Stage 2 Neural Geometry Design

Date: 2026-03-30
Status: Drafted after user-approved design
Topic: Build and validate neural representational distance geometry for bacterial metabolite stimuli

## Goal

Construct a stable neural representational geometry for the stimulus set before attempting biochemical interpretation.

The central Stage 2 claim is:

> the relative stimulus geometry in neural population space can be measured as stimulus-by-stimulus RDMs, and that geometry is reproducible enough across individuals and dates to support later RSA work.

Stage 2 is not the RSA stage. It is the geometry description and stability stage.

## Current Context

Stage 1 is complete and its handoff is already recorded in `state/project-status.json`.

Current handoff status:

- `current_phase = "Stage 2 preparation"`
- `stage1_status = "completed_cautious_pass"`
- formal Stage 1 primary view = `response_window`
- strongest Stage 1 views by distance gap = `response_window`, `on_window`
- major carry-forward risk = date-level generalization is materially weaker than individual-level generalization

Relevant existing references:

- `docs/stage1-reliability-explained.md`
- `state/project-status.json`
- `results/stage1_reliability/run_summary.json`
- `results/stage1_reliability/tables/final_summary.parquet`
- `results/stage1_reliability/tables/stimulus_distance_pairs.parquet`

Stage 2 also still depends on the Stage 0 trial-level inputs because pooled Stage 1 distance summaries are not sufficient to produce per-individual and per-date RDMs:

- `data/processed/trial_level/trial_metadata.parquet`
- `data/processed/trial_level/trial_tensor_baseline_centered.npz`
- `data/processed/trial_level/trial_wide_baseline_centered.parquet`

Dependency rule:

- Stage 2 must be runnable from Stage 0 trial-level inputs plus the Stage 1 handoff state
- existing Stage 1 output tables may be reused as an optimization or cross-check
- Stage 2 must not require prior Stage 1 pair-summary files in order to run

## Scientific Question

What is the relative geometry among the `19` stimuli in neural population space, and how stable is that geometry across individuals and dates?

## Scope

This stage covers:

- construction of neural stimulus-by-stimulus RDMs
- pooled RDM estimation
- per-individual RDM estimation
- per-date RDM estimation
- geometry stability summaries across individuals and dates
- comparison between the approved dual-view MVP representations
- heatmaps and compact geometry-stability figures for interpretation

This stage does not cover:

- biochemical or metabolite-property model RDMs
- RSA or neural-model correlation
- dendrograms or embeddings as primary evidence
- sliding-window or fully time-resolved RDM analysis
- neuron contribution analysis
- behavior linkage

## MVP View Strategy

Stage 2 should be implemented as a dual-view MVP.

Included views:

1. `response_window`
2. `full_trajectory`

Rationale:

- `response_window` is already the formal primary view from Stage 1
- `full_trajectory` tests whether the geometry remains consistent when the full dynamics are retained
- `on_window` is intentionally excluded from the MVP because the user considers it informationally incomplete for the current question

This MVP must still be implemented in a way that can later expand to additional views without restructuring the whole module.

## Primary Design Principle

Stage 2 must reuse the Stage 1 trial-comparison logic wherever possible and only add the minimum new machinery required to express geometry as stable RDM objects.

This means:

- keep the overlap-aware trial comparison rule
- keep the default correlation distance
- keep the same trial-level representation windows
- add grouping, RDM assembly, and RDM-stability logic on top

Stage 2 should not simultaneously change the distance metric family, the missing-data rule, and the output contract. That would make the Stage 1 to Stage 2 transition hard to interpret.

## Analysis Units

### Trial-Level Unit

The base unit remains one trial-level `neuron x time` array aligned to one `trial_id`.

### Stimulus-Pair Unit

The central Stage 2 long-form analysis unit is one aggregated stimulus pair inside a defined group:

`view_name x group_type x group_id x stimulus_left x stimulus_right`

This is the unit stored in the primary Stage 2 pair tables.

### Matrix Unit

The central Stage 2 matrix object is one square `stimulus x stimulus` RDM for a defined view and group.

## Group Definitions

Stage 2 should support exactly three group types in the MVP:

1. `pooled`
   - all eligible trials together
2. `individual`
   - trials within one `individual_id`
3. `date`
   - trials within one `date`

The Stage 1 individual definition carries forward unchanged:

`individual_id = "{date}__{worm_key}"`

## Distance Definition

The default Stage 2 trial-level distance is the same overlap-aware correlation distance used in Stage 1.

When two trial arrays are compared:

- only neurons observed in both arrays are kept
- only shared non-NaN values within those neurons are used
- the valid values are flattened
- `distance = 1 - correlation`

This is the correct default for the MVP because it keeps Stage 2 scientifically continuous with the Stage 1 gate that already justified moving forward.

No new distance family should be introduced in the MVP.

## RDM Construction Contract

### Step 1: Build Trial-Level Pairwise Distances

For each included view:

- compute overlap-aware trial-to-trial distances
- retain the same QC information already produced in Stage 1

This can reuse the existing Stage 1 pairwise distance machinery.

### Step 2: Aggregate Trial Pairs Into Stimulus Pairs

For each view and each group:

- gather eligible trial pairs belonging to that group definition
- aggregate them into unordered stimulus pairs
- compute at least:
  - `mean_distance`
  - `median_distance`
  - `n_trial_pairs`

The MVP default aggregation should use `mean_distance` as the value that enters the RDM.

### Step 3: Materialize RDM Matrices

From the long-form stimulus-pair summaries:

- create symmetric `stimulus x stimulus` matrices
- preserve diagonal entries for same-stimulus distances
- keep matrices directly writable to parquet for pooled outputs

For the MVP, pooled matrices should be materialized as standalone matrix files. Group-level matrices for every individual/date do not need one-file-per-matrix exports in the first version; the long tables are sufficient.

## Eligibility Rules

Stage 2 must not silently treat sparse groups as fully informative.

For each group-level stimulus pair summary, record enough information to detect weak support:

- number of contributing trial pairs
- number of trials contributing from the left stimulus
- number of trials contributing from the right stimulus

If a group cannot support a given stimulus pair, the missing value should remain explicit rather than being imputed.

## Stability Analysis Contract

Stage 2 must answer stability before interpretation.

### A. Individual-Level Geometry Stability

For each view:

- convert each per-individual RDM into an upper-triangle vector
- compare individuals pairwise using RDM similarity
- summarize the distribution of individual-to-individual RDM similarity

### B. Date-Level Geometry Stability

For each view:

- convert each per-date RDM into an upper-triangle vector
- compare dates pairwise using RDM similarity
- summarize the distribution of date-to-date RDM similarity

### C. Pooled-vs-Group Similarity

For each view:

- compare each individual RDM to the pooled RDM
- compare each date RDM to the pooled RDM

This answers whether the pooled geometry reflects the typical group-level geometry or is being distorted by aggregation.

### D. Cross-View Geometry Similarity

For the dual-view MVP:

- compare the pooled `response_window` RDM to the pooled `full_trajectory` RDM
- compare whether the two views show similar stability patterns across individuals and dates

### Default RDM Similarity Statistic

The MVP should use `Spearman` correlation between upper-triangle vectors as the default RDM similarity measure.

Rationale:

- Stage 2 is primarily about relative geometry ordering
- Spearman is less sensitive to scale shifts than Pearson
- it keeps the interpretation focused on whether stimulus relations are preserved

RDM similarity must be computed only on shared non-missing upper-triangle entries. If two RDMs do not share enough valid entries to support a meaningful comparison, that similarity score must be marked invalid and surfaced in QC rather than silently coerced to a number.

## Main Statistics To Report

The primary Stage 2 reporting set should include:

1. pooled RDM per view
2. median individual-to-individual RDM similarity per view
3. median date-to-date RDM similarity per view
4. median pooled-vs-individual RDM similarity per view
5. median pooled-vs-date RDM similarity per view
6. pooled cross-view RDM similarity
7. group coverage summaries needed to interpret sparse or unstable entries

The main conclusion should be framed around RDM stability, not around any one visualization.

## Interpretation Rules

Stage 2 is considered strong enough to support Stage 3 planning if:

- at least one MVP view shows clearly reproducible geometry across individuals
- date-level stability is either acceptable or weak-but-quantified rather than unexplained
- the pooled RDM is broadly representative of the group-level RDMs

Stage 2 can still be considered scientifically useful if:

- individual-level stability is decent
- date-level stability is weaker

In that case the right conclusion is not "Stage 2 failed." The right conclusion is that the geometry exists but date effects remain a major limitation.

Stage 2 should not be treated as ready for Stage 3 if:

- pooled geometry looks structured but group-level RDMs are too inconsistent to support interpretation
- `response_window` and `full_trajectory` produce strongly conflicting pooled geometry without explanation

## Output Contract

Stage 2 should write outputs under:

- `results/stage2_geometry/tables/`
- `results/stage2_geometry/figures/`
- `results/stage2_geometry/qc/`

### Required Tables

- `rdm_pairs__response_window__pooled.parquet`
- `rdm_pairs__response_window__individual.parquet`
- `rdm_pairs__response_window__date.parquet`
- `rdm_pairs__full_trajectory__pooled.parquet`
- `rdm_pairs__full_trajectory__individual.parquet`
- `rdm_pairs__full_trajectory__date.parquet`
- `rdm_matrix__response_window__pooled.parquet`
- `rdm_matrix__full_trajectory__pooled.parquet`
- `rdm_stability_by_individual.parquet`
- `rdm_stability_by_date.parquet`
- `rdm_view_comparison.parquet`

### Required QC Tables

- `rdm_group_coverage.parquet`

### Required Summaries

- `run_summary.json`
- `run_summary.md`

### Required Figures

- pooled RDM heatmap for `response_window`
- pooled RDM heatmap for `full_trajectory`
- individual-level RDM stability summary figure
- date-level RDM stability summary figure
- view-comparison summary figure

The MVP should avoid generating one separate matrix file per individual/date. That would create a large and low-value file surface. Long-form group tables are sufficient for the first version.

## Repository Structure

Stage 2 should extend the existing repository layout rather than inventing a new subsystem.

Recommended additions:

- `src/bacteria_analysis/geometry.py`
- `src/bacteria_analysis/geometry_outputs.py`
- `scripts/run_geometry.py`
- `tests/test_geometry.py`
- `tests/test_geometry_cli_smoke.py`

Responsibilities:

- `geometry.py`: group-aware RDM construction and stability calculations
- `geometry_outputs.py`: parquet, JSON, markdown, and figure writing
- `run_geometry.py`: thin CLI wrapper
- `tests/`: deterministic unit and smoke coverage

## Data Flow

The intended Stage 2 data flow is:

1. read Stage 0 trial metadata/tensor inputs
2. read Stage 1 handoff information needed for view and reporting defaults
3. build the approved views
4. compute or reuse trial-level pairwise distances
5. aggregate to stimulus-pair summaries by group
6. build pooled matrix outputs
7. compute RDM stability summaries
8. write tables, QC outputs, figures, and run summaries

## Error Handling And QC

Stage 2 should fail explicitly rather than silently degrading.

Required checks:

- missing input files must raise a clear CLI error
- unsupported requested views must fail fast
- empty or undersupported groups must be recorded explicitly in QC outputs
- malformed RDM vectors with no valid upper-triangle entries must not be scored as if valid
- run summary must state the actual included views

The output layer should always make it obvious whether a weak result comes from geometry instability or from insufficient support.

## Testing Expectations

The MVP requires both unit coverage and CLI smoke coverage.

### Unit Tests

Unit tests should verify:

- pooled, individual, and date group assignment rules
- correct unordered stimulus-pair aggregation
- symmetric matrix reconstruction from long tables
- correct handling of missing/undersupported group entries
- RDM similarity computation on upper-triangle vectors
- cross-view comparison behavior

### CLI Smoke Test

The smoke test should verify:

- the Stage 2 CLI runs from tiny Stage 0 fixtures
- the expected output files are created
- the run summary records the two included MVP views

## Risks To Carry Forward

The main Stage 2 risks are:

- date effects remain substantial, so pooled geometry may overstate stability
- full-trajectory geometry may be richer but noisier than response-window geometry
- sparse group support may make some stimulus pairs appear unstable only because coverage is weak
- users may over-read heatmaps or embeddings before stability statistics are checked

The MVP design intentionally addresses these risks by making group-level stability tables primary and pooled heatmaps secondary.

## Non-Goals For This Spec

This spec does not define:

- Stage 3 model RDM construction
- RSA significance testing
- crossnobis or Mahalanobis distance adoption
- publication-final figure styling

Those belong to later documents.

## Immediate Next Step After This Spec

The next document should be a Stage 2 implementation plan that specifies:

- exact functions and module boundaries
- exact aggregation formulas
- exact stability-table schemas
- exact CLI arguments
- exact figure and table filenames
- exact test cases and fixture expectations
