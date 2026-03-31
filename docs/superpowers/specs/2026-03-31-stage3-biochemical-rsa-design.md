# Stage 3 Biochemical RSA Design

Date: 2026-03-31
Status: Drafted after user-approved design
Topic: Link pooled neural geometry to metabolite model structure with user-curated biochemical hypotheses

## Goal

Stage 3 should test whether the pooled neural representational geometry aligns with meaningful structure in the metabolite space.

The central Stage 3 claim to test is:

> the pooled neural geometry across the `19` stimuli is better explained by specific metabolite-space models than by label-shuffled null structure.

Stage 3 is not the stage for strong group-level biological generalization. It is the stage for disciplined neural-model comparison under explicitly limited support.

## Current Context

Stage 1 and Stage 2 are complete and the current handoff is already recorded in `state/project-status.json`.

Current handoff status:

- `current_phase = "Stage 3 preparation"`
- Stage 1 status = `completed_cautious_pass`
- Stage 1 formal primary view = `response_window`
- Stage 1 formal primary distance metric = `correlation`
- Stage 2 status = `completed_exploratory_pass`
- Stage 2 primary view = `response_window`
- Stage 2 sensitivity view = `full_trajectory`
- major carry-forward risk = pooled geometry is useful, but grouped stability remains overlap-limited

Relevant existing references:

- `docs/stage1-reliability-explained.md`
- `docs/stage2-geometry-exploration-summary.md`
- `state/project-status.json`
- `results/stage2_geometry/tables/rdm_matrix__response_window__pooled.parquet`
- `results/stage2_geometry/tables/rdm_matrix__full_trajectory__pooled.parquet`
- `results/stage2_geometry/tables/rdm_stability_by_individual.parquet`
- `results/stage2_geometry/tables/rdm_stability_by_date.parquet`
- `results/stage2_geometry/tables/rdm_view_comparison.parquet`
- `results/stage2_geometry/qc/`

Current verified model-input facts:

- `data/matrix.xlsx` is a single-sheet wide table
- sheet extent is `A1:NQ300`
- the table contains one sample identifier column plus approximately `380` metabolite feature columns
- the current neural stimulus panel contains `19` unique `stim_name` labels
- all `19` corresponding `Axxx` sample ids are present in `data/matrix.xlsx`

## Scientific Question

Which metabolite-space models best explain the pooled neural geometry among the `19` stimuli, and are those matches stable across the approved dual-view neural comparison?

## Scope

This stage covers:

- a fixed stimulus-to-sample mapping between neural stimuli and `data/matrix.xlsx` rows
- a fixed model-space contract built from `data/matrix.xlsx`
- user-curated metabolite model definitions as the primary biochemical source of truth
- optional exploratory experience-based model definitions as supplementary-only hypotheses
- pooled RSA using the approved Stage 2 neural RDMs
- cross-view sensitivity comparison between `response_window` and `full_trajectory`
- permutation-based significance testing
- leave-one-stimulus-out sensitivity checks
- QC and reporting needed to distinguish a real model match from a fragile or underspecified one

This stage does not cover:

- primary grouped RSA across individuals or dates
- time-resolved or sliding-window RSA
- replacing the approved neural distance metric
- automated ontology mining as a substitute for user biochemical research
- neuron contribution analysis
- behavior linkage

## Primary Design Principle

Stage 3 must keep the neural side fixed and move the scientific uncertainty onto an explicit model-definition layer.

This means:

- keep the Stage 2 pooled neural RDM as the neural input object
- keep `response_window` as the formal primary neural view
- keep `full_trajectory` as sensitivity only
- treat biochemical model definition as a first-class input contract, not an ad hoc code detail
- separate user-curated primary models from exploratory experience-based supplement models

Stage 3 should not simultaneously change the neural metric family, the neural aggregation contract, and the biochemical model definitions. The neural side is already constrained enough; the open question is which external model best explains it.

## Neural Input Contract

Stage 3 should consume exactly two pooled neural RDM inputs in the MVP:

1. `response_window` pooled neural RDM
2. `full_trajectory` pooled neural RDM

Formal reporting rule:

- the primary reported RSA result must use the pooled `response_window` neural RDM
- `full_trajectory` is a sensitivity check on the same model space
- grouped neural RDMs may be referenced for context, but not promoted to primary RSA evidence in the MVP

## Stimulus-To-Sample Mapping Contract

Stage 3 must make the neural-to-metabolite linkage explicit.

Required mapping unit:

`stimulus x stim_name x sample_id`

Where:

- `stimulus` is the unique neural stimulus key from the processed trial metadata
- `stim_name` is the human-readable neural label
- `sample_id` is the `Axxx` identifier used as the row key in `data/matrix.xlsx`

Required rules:

- the mapping must contain exactly one row per included neural stimulus
- the mapping must preserve the exact neural stimulus ordering used when materializing neural RDMs
- `sample_id` values must be unique within the current `19`-stimulus panel
- all mapped `sample_id` values must exist in `data/matrix.xlsx`
- any mismatch between neural labels and model rows must fail fast

This mapping file is part of the scientific contract, not just implementation plumbing.

## Model-Definition Authority Rule

The project needs two explicitly separated model-definition tiers.

### Tier 1: User-Curated Primary Models

These are the authoritative biochemical models for formal interpretation.

Characteristics:

- created or approved by the user after domain-specific literature or chemistry review
- stored explicitly as curated model definitions
- eligible for primary RSA interpretation and model ranking
- used for any formal biological statements in Stage 3 reporting

### Tier 2: Exploratory Supplement Models

These are allowed, but they are not primary evidence.

Characteristics:

- derived from experience knowledge, broad name-based grouping, or quick exploratory annotation
- clearly flagged as exploratory in the registry and outputs
- useful for hypothesis generation or sanity checks
- not sufficient on their own to support the main biological conclusion

If an exploratory model performs well but no aligned user-curated primary model exists, the correct conclusion is "interesting exploratory alignment" rather than a formal biochemical claim.

## Model Registry Contract

Stage 3 should not hard-code model subsets directly in analysis code.

Instead, it should use an explicit model registry with at least:

- `model_id`
- `model_label`
- `model_tier`
- `model_status`
- `feature_kind`
- `distance_kind`
- `description`
- `authority`
- `notes`

Recommended status values:

- `primary`
- `supplementary`
- `draft`
- `excluded`

Recommended feature kinds:

- `continuous_abundance`
- `binary_presence`

Recommended distance kinds:

- `correlation`
- `euclidean`
- `jaccard`

The MVP should keep this registry small and explicit.

## Model Membership Contract

Curated subset models should be defined through an explicit long-form membership table rather than wide boolean flags embedded in code.

Required membership unit:

`model_id x metabolite_name`

Optional annotation fields:

- `membership_source`
- `review_status`
- `ambiguous_flag`
- `notes`

Rules:

- model membership may be overlapping when biologically justified
- ambiguous metabolites must be allowed to remain excluded from primary models
- models with too few informative features must be surfaced in QC rather than silently scored as if robust

This design lets the user supply research-backed model memberships without requiring code edits for each new hypothesis.

## Initial Model Space Strategy

The MVP should use a layered model space rather than a single monolithic biochemical distance.

### A. Global Profile Model

One model should always be built from the full available metabolite profile after preprocessing.

Purpose:

- provide a neutral "overall metabolite composition" baseline
- test whether the neural geometry tracks broad metabolite-space similarity at all

This model does not depend on curated subset membership.

### B. User-Curated Subset Models

These are the main scientific hypothesis models.

Examples may include bile acids, lipid-related groups, indole-related groups, phenyl/phenol-related groups, sugar and central-carbon groups, nucleotide-energy groups, or other user-defined biochemical subsets.

Important rule:

- the spec supports these model families
- the exact primary subset definitions must come from user research, not from unreviewed experience grouping

### C. Exploratory Supplement Models

Broad experience-based groupings are allowed only as supplementary outputs.

Examples:

- quick name-based bile-acid grouping
- broad fatty-acid-like grouping
- broad aromatic grouping
- broad central-carbon-like grouping

These can help exploration, but must be clearly separated from the user-curated primary model family in all outputs and figures.

## Model Feature Preprocessing Contract

Stage 3 should define preprocessing rules that do not depend on the neural results.

For continuous-abundance model features, the MVP default should:

1. subset `data/matrix.xlsx` to the mapped `19` sample rows
2. keep only the columns required by the current model definition
3. apply a monotonic scale-compression transform such as `log1p`
4. drop zero-variance features within the included `19` samples
5. standardize retained features column-wise before distance calculation

Rationale:

- the current matrix values span multiple orders of magnitude
- untransformed scale differences would dominate many distance calculations
- zero-variance features do not contribute useful geometry

For binary-presence models:

- the threshold rule must be explicit
- the threshold must be recorded in the model registry or run summary

The preprocessing contract must be identical across `response_window` and `full_trajectory` because the neural view should not change the model side.

## Model RDM Construction Contract

For each included model:

1. build a `19 x p_model` feature matrix in the fixed stimulus order
2. compute pairwise sample distances
3. materialize one square `stimulus x stimulus` model RDM

The model RDM unit is:

`model_id x stimulus_left x stimulus_right`

Default distance rules:

- for `continuous_abundance` models, use `correlation distance` in the MVP
- for `binary_presence` models, use `jaccard distance`

These defaults are chosen to keep the model side pattern-focused and robust to scale differences in the first pass.

Any future distance-family expansion belongs to sensitivity analysis, not to the primary Stage 3 contract.

## RSA Contract

The primary RSA comparison unit is one neural-model pair:

`neural_view x model_id`

For each pair:

- vectorize the upper triangle of the neural RDM
- vectorize the upper triangle of the aligned model RDM
- compute RSA similarity using `Spearman` correlation

Rationale:

- Stage 3 is still about relative geometry ordering
- Spearman stays aligned with the Stage 2 ordering-focused interpretation
- it reduces sensitivity to monotonic scaling differences between neural and model distances

## Significance And Robustness Contract

Stage 3 must not stop at raw RSA correlations.

### A. Permutation Null

For each included neural-view and model pair:

- permute stimulus labels on one side
- recompute the RSA statistic
- summarize the null distribution
- report a permutation p-value

### B. Multiple-Model Correction

Because Stage 3 compares multiple models:

- report raw p-values
- report a multiple-testing-adjusted value across the primary model family

The MVP may use false-discovery-rate control for the primary model family.

### C. Leave-One-Stimulus-Out Sensitivity

For each primary model:

- rerun RSA after removing one stimulus at a time
- summarize whether the apparent model match depends heavily on one or two stimuli

### D. Cross-View Sensitivity

For each included model:

- compare the RSA result under pooled `response_window`
- compare the RSA result under pooled `full_trajectory`

This answers whether the claimed biochemical alignment is stable to the approved neural-view sensitivity check.

## Main Statistics To Report

The primary Stage 3 reporting set should include:

1. pooled neural-model RSA statistic for each primary model under `response_window`
2. permutation p-value for each primary model
3. multiple-testing-adjusted value across primary models
4. leave-one-stimulus-out robustness summary for each primary model
5. `response_window` versus `full_trajectory` RSA comparison for each primary model
6. feature-count and retained-variance QC for each model
7. explicit model-tier labels distinguishing primary curated models from exploratory supplement models

The main conclusion should be framed around the best-supported primary curated models, not around the mere existence of some exploratory alignment.

## Interpretation Rules

Stage 3 can support a formal biochemical interpretation only if:

- at least one user-curated primary model shows a clear pooled RSA match under `response_window`
- that match survives the permutation null at the declared multiple-testing threshold
- the result is not obviously driven by a single stimulus in leave-one-stimulus-out checks
- the result is directionally consistent in `full_trajectory`

Stage 3 can still be scientifically useful if:

- the global profile model matches the neural geometry
- curated subset models remain weaker or inconclusive

In that case the right conclusion is:

> the pooled neural geometry aligns with broad metabolite composition, but the current curated biochemical submodels do not yet isolate the dominant explanatory axis.

Stage 3 should not support a strong biochemical claim if:

- only exploratory supplement models perform well
- primary curated models are absent, underdefined, or fail QC
- the apparent top model is highly unstable under leave-one-stimulus-out checks
- `response_window` and `full_trajectory` tell materially conflicting model-ranking stories without explanation

## Output Contract

Stage 3 should write outputs under:

- `results/stage3_rsa/tables/`
- `results/stage3_rsa/figures/`
- `results/stage3_rsa/qc/`

### Required Tables

- `stimulus_sample_map.parquet`
- `model_registry_resolved.parquet`
- `model_membership_resolved.parquet`
- `model_feature_qc.parquet`
- `model_rdm_summary.parquet`
- `rsa_results.parquet`
- `rsa_leave_one_stimulus_out.parquet`
- `rsa_view_comparison.parquet`

### Required QC Tables

- `model_input_coverage.parquet`
- `model_feature_filtering.parquet`

### Required Summaries

- `run_summary.json`
- `run_summary.md`

### Required Figures

- ranked primary-model RSA summary figure
- neural-versus-top-model RDM comparison panel
- leave-one-stimulus-out robustness summary figure
- view-comparison summary figure

Exploratory supplement models may be shown in secondary tables or appendices, but the main figure set should keep the primary curated models visually distinct.

## Repository Structure

Stage 3 should extend the existing repository layout rather than inventing a new subsystem.

Recommended additions:

- `src/bacteria_analysis/model_space.py`
- `src/bacteria_analysis/rsa.py`
- `src/bacteria_analysis/rsa_outputs.py`
- `scripts/run_rsa.py`
- `tests/test_model_space.py`
- `tests/test_rsa.py`
- `tests/test_rsa_cli_smoke.py`

Responsibilities:

- `model_space.py`: stimulus mapping, model registry resolution, model membership resolution, feature preprocessing, and model RDM construction
- `rsa.py`: upper-triangle extraction, RSA statistics, permutation logic, multiple-testing adjustment, and leave-one-stimulus-out summaries
- `rsa_outputs.py`: parquet, JSON, markdown, and figure writing
- `run_rsa.py`: thin CLI wrapper
- `tests/`: deterministic unit and smoke coverage

## Data Flow

The intended Stage 3 data flow is:

1. read Stage 2 pooled neural RDM inputs and Stage 3 handoff state
2. read the stimulus-to-sample mapping
3. read `data/matrix.xlsx`
4. read the model registry and model membership definitions
5. build preprocessed model feature matrices
6. build model RDMs
7. run RSA against pooled `response_window`
8. rerun against pooled `full_trajectory`
9. compute null, correction, and leave-one-stimulus-out summaries
10. write tables, QC outputs, figures, and run summaries

## Error Handling And QC

Stage 3 should fail explicitly rather than silently degrading.

Required checks:

- missing neural RDM inputs must raise a clear CLI error
- missing or duplicated `sample_id` mappings must fail fast
- mapped `sample_id` values absent from `data/matrix.xlsx` must fail fast
- requested primary models with no resolved features must fail fast
- requested primary models with fewer than two informative features must be surfaced clearly and excluded from ranking
- exploratory models must never be silently upgraded into the primary family
- mismatched stimulus ordering between neural and model RDMs must fail fast
- the run summary must state which models were primary and which were supplementary

The output layer should always make it obvious whether a weak result comes from no neural-model alignment, poor model definition, or fragile support.

## Testing Expectations

The MVP requires both unit coverage and CLI smoke coverage.

### Unit Tests

Unit tests should verify:

- correct stimulus-to-sample mapping validation
- correct resolution of model registry and membership rows
- correct filtering of missing or zero-variance model features
- correct model RDM construction in fixed stimulus order
- correct RSA upper-triangle alignment
- correct permutation and multiple-testing summary behavior
- correct leave-one-stimulus-out aggregation
- correct separation of primary versus supplementary model outputs

### CLI Smoke Test

The smoke test should verify:

- the Stage 3 CLI runs from tiny fixtures
- the expected output files are created
- the run summary records the primary neural view and the included model families

## Risks To Carry Forward

The main Stage 3 risks are:

- a strong pooled neural-model match may overstate biological stability if users forget the grouped-support limitations from Stage 2
- weak or incomplete user curation may leave the primary model family too underdeveloped for strong interpretation
- overly broad exploratory models may look interpretable while actually mixing unrelated chemistry
- aggressive feature preprocessing choices may change the apparent model ranking if they are not fixed in advance
- users may over-read exploratory supplement models that were never intended as primary evidence

The MVP design intentionally addresses these risks by:

- fixing the neural side
- making the model-definition layer explicit
- separating curated primary models from exploratory supplements
- requiring null, correction, and leave-one-stimulus-out checks

## Non-Goals For This Spec

This spec does not define:

- the exact user-researched biochemical model memberships
- ontology lookup tooling
- grouped RSA as a primary result
- time-resolved RSA
- neuron contribution analysis
- behavior-linked interpretation
- publication-final figure styling

Those belong to later documents or later project stages.

## Immediate Next Step After This Spec

The next document should be a Stage 3 implementation plan that specifies:

- exact curated input file schemas
- exact model preprocessing formulas
- exact RSA table schemas
- exact CLI arguments
- exact figure and table filenames
- exact test cases and fixture expectations
