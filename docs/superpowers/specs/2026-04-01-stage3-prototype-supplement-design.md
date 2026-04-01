# Stage 3 Prototype Supplement Design

Date: 2026-04-01
Status: Drafted after user-approved design
Topic: Add prototype-level supplementary analyses to Stage 3 RSA without changing the current pooled-RDM primary result

## Goal

Stage 3 currently produces valid pooled neural-RDM-vs-model RSA results and improved per-view RDM figures.

The goal of this design is to add a supplementary prototype-level analysis layer inside `stage3_rsa` so that:

- each `date x view` gets a prototype-level neural-vs-model RSA comparison
- each `view` also gets a pooled prototype neural RDM for descriptive visualization
- the current pooled-RDM Stage 3 result remains the primary analysis and remains unchanged

This is an additive supplement. It should improve interpretability, not replace the current Stage 3 inferential contract.

## Current Context

Current Stage 3 is built around:

- `scripts/run_rsa.py`
- `src/bacteria_analysis/rsa.py`
- `src/bacteria_analysis/rsa_outputs.py`

Current Stage 3 inputs are limited to:

- pooled neural RDMs from Stage 2
- model-space inputs from `data/model_space...`

Current Stage 3 does not ingest trial-level neural response tables, so it cannot build stimulus-level neural prototypes on its own.

Relevant existing neural inputs live in preprocessing outputs:

- `trial_metadata.parquet`
- `trial_wide_baseline_centered.parquet`
- `trial_tensor_baseline_centered.npz`

Current scientific position already accepted in the project:

- Stage 1 and Stage 2 provide enough support to treat trial-level neural structure as real but limited
- the current formal Stage 3 result is pooled neural RDM RSA derived from Stage 2
- prototype-level analyses may be used as supplementary interpretability aids, not as replacements for the primary pooled-RDM result

For the current `202603` batch:

- the current pooled-RDM Stage 3 result is valid
- the two dates do not share stimulus identities
- pooled cross-date prototype visualizations are therefore descriptive only, not evidence for date-generalizable structure

## User-Requested Addition

The requested supplementary analyses are:

1. `Per-date prototype RSA`
2. `Pooled prototype RDM`

The user explicitly approved this framing:

- add both analyses inside the existing `stage3_rsa` output family
- keep `per-date prototype RSA` as a statistical supplementary analysis
- keep `pooled prototype RDM` as descriptive only
- compute them separately for `response_window` and `full_trajectory`

## Options Considered

### Option 1: Add Supplementary Prototype Outputs Inside `stage3_rsa`

Extend the current Stage 3 pipeline so it can optionally read trial-level preprocessing inputs, construct prototype neural representations, and write supplementary prototype tables and figures alongside the existing trial-level outputs.

Pros:

- best matches the requested interpretation workflow
- keeps current primary and supplementary model-comparison outputs in one place
- avoids creating a new stage for a non-primary analysis

Cons:

- requires Stage 3 to accept additional upstream inputs beyond Stage 2 pooled RDMs

### Option 2: Move Prototype Construction Into Stage 2

Extend Stage 2 to produce prototype vectors or prototype RDMs, then let Stage 3 consume them.

Pros:

- cleaner stage separation on paper

Cons:

- expands Stage 2 scope beyond the current need
- creates a larger cross-stage change for what is currently a supplementary analysis

### Option 3: Create A Separate Prototype-RSA Entry Point

Keep the current Stage 3 command untouched and add a second script dedicated to prototype analyses, while writing results into a Stage 3-adjacent output location.

Pros:

- smallest disruption to the current Stage 3 command

Cons:

- fragments the workflow
- makes reporting and maintenance more scattered

## Chosen Approach

Use Option 1.

The design should extend the existing Stage 3 pipeline with an optional supplementary branch:

- pooled-RDM Stage 3 remains the main result
- prototype-level analyses are computed only when trial-level preprocessing inputs are available
- supplementary outputs are written into the same `stage3_rsa` directory family with explicit `prototype_...` names

This preserves the current interpretation hierarchy:

- primary claim = pooled neural RDM vs model RSA from the existing Stage 2 -> Stage 3 handoff
- supplementary interpretability layer = prototype-level RSA and prototype-level pooled neural RDM

## Scope

This design covers:

- optional Stage 3 access to preprocessing-level neural inputs
- construction of stimulus-level prototype neural response vectors
- per-date prototype RSA for each approved neural view
- pooled prototype neural RDM visualization for each approved neural view
- supplementary tables, figures, and run-summary metadata inside `stage3_rsa`
- tests for prototype construction, statistical boundaries, and output contracts

This design does not cover:

- replacing the current primary Stage 3 pooled-RDM RSA
- changing Stage 1 or Stage 2 formal inference
- introducing pooled prototype RSA significance testing
- adding cross-date prototype generalization claims
- adding new neural views beyond `response_window` and `full_trajectory`

## Input Contract

Stage 3 should gain an optional preprocessing input root, for example via:

- `--preprocess-root`

When provided, Stage 3 supplementary prototype analyses should read:

- `trial_level/trial_metadata.parquet`
- `trial_level/trial_wide_baseline_centered.parquet`
- `trial_level/trial_tensor_baseline_centered.npz`

These inputs should be optional:

- if absent, current Stage 3 behavior stays unchanged
- if present, the supplementary prototype branch runs in addition to the existing primary pooled-RDM branch

The current Stage 3 required inputs remain unchanged:

- Stage 2 pooled neural RDMs
- model-space inputs
- metabolite matrix

Authoritative preprocessing inputs for prototype construction:

- `trial_metadata.parquet` is the row-order authority
- `trial_tensor_baseline_centered.npz` is the computational source of truth for prototype construction
- `trial_wide_baseline_centered.parquet` is optional convenience/debug input only and must not become the authoritative numerical source

Alignment rules:

- trial tensor rows must align exactly with `trial_metadata.parquet`
- prototype-to-model alignment must occur by `stimulus`
- `stim_name` is display metadata only and must not be used as the join key

## Prototype Definition

Prototype construction must reuse the existing neural-view contract rather than inventing a new one.

For each approved `view_name`:

- `response_window`
- `full_trajectory`

the code should derive a trial-level neural response vector from the preprocessing tensor using the same temporal interpretation already used elsewhere in the project.

View definition contract:

- reuse `VIEW_WINDOWS` / Stage 2 MVP view selection exactly
- `response_window` uses the existing response-window timepoints
- `full_trajectory` uses the full existing time axis

Vector extraction contract:

- for each trial, slice the tensor by the requested view timepoints
- preserve fixed neuron order and fixed within-view time order
- flatten the resulting neuron-by-time block in deterministic row-major order
- do not derive prototype vectors from `stim_name`

Prototype rule:

- a prototype is the mean neural response vector across all trials belonging to the same stimulus grouping unit

Grouping unit:

- `date x stimulus` for per-date prototype RSA
- `stimulus` for pooled prototype RDM

This means:

- `Per-date prototype RSA` builds one prototype vector per `date x stimulus x view`
- `Pooled prototype RDM` builds one prototype vector per `stimulus x view` pooled across all available trials

The vector should preserve neural-feature structure:

- do not collapse each stimulus to a single scalar
- keep the neuron-major or flattened neuron-time representation appropriate to the selected view

NaN handling contract:

- prototype vectors are computed by elementwise `nanmean` across trials in the grouping unit
- per-feature contributor counts must be tracked for QC
- if a prototype vector is entirely NaN, that prototype cannot enter downstream RDM construction

## Prototype Neural Distance Contract

Prototype vectors must be converted to neural distances using the same default neural distance family already approved in Stage 1 and Stage 2.

Default rule:

- use overlap-aware correlation distance on shared finite values

Comparison contract for two prototype vectors:

- keep only feature positions that are finite in both vectors
- require at least the existing minimum valid-value threshold before scoring
- compute `distance = 1 - correlation`
- if either filtered vector is constant, treat the pair as invalid rather than forcing a finite distance
- if the resulting correlation is non-finite, treat the pair as invalid

Representation contract:

- valid prototype-pair distances are stored as numeric values
- invalid prototype-pair distances are stored as `NaN`
- the diagonal of a prototype RDM is `0.0`

This supplement should not introduce a new neural distance family in the MVP.

## Supplementary Analysis 1: Per-Date Prototype RSA

For each `date x view`:

1. build one prototype vector per stimulus
2. convert the prototype vectors into a neural RDM using the prototype neural distance contract above
3. restrict the model RDM to the same stimulus set
4. compute RSA between the date-specific prototype neural RDM and the restricted model RDM

This analysis is inferential supplementary output, so it should reuse the existing Stage 3 RSA machinery as closely as possible:

- same RSA similarity metric
- same `score_status` behavior
- same `n_shared_entries` logic
- same permutation-null logic

Each row should still be able to become `invalid` if there are too few shared entries after restriction.

The resulting table should be explicitly date-resolved rather than merged into the primary `rsa_results.parquet`.

Model inclusion rules:

- evaluate the same model set that current Stage 3 would otherwise build
- carry forward `model_tier`, `model_status`, and `excluded_from_primary_ranking`
- keep excluded or invalid rows visible in the supplementary result table rather than silently dropping them

## Supplementary Analysis 2: Pooled Prototype RDM

For each `view`:

1. build one pooled prototype vector per stimulus across all available trials
2. convert those prototypes into a pooled neural RDM using the prototype neural distance contract above
3. write the pooled prototype neural RDM as a table and figure

This branch is descriptive only.

It should not:

- produce formal RSA `p` values
- be described as a cross-date generalization result
- replace the current primary Stage 3 tables or plots

For `202603`, the pooled prototype RDM should be treated explicitly as a descriptive pooled view because the two dates do not share stimulus identities.

## Statistical Rules

### Per-Date Prototype RSA

For each `date x view x model_id` row:

- compute `rsa_similarity`
- compute `p_value_raw` from permutations when requested
- compute `p_value_fdr` across all finite `p_value_raw` rows in `prototype_rsa_results__per_date.parquet` as one supplementary family
- compute `is_top_model` within each `date x view` only among rows where:
  - `score_status == "ok"`
  - `excluded_from_primary_ranking == False`
  - `rsa_similarity` is finite

The primary pooled-RDM `rsa_results.parquet` must remain untouched by this supplementary branch.

### Pooled Prototype RDM

For each `view`:

- compute and write the pooled prototype neural RDM
- do not attach formal RSA inference
- mark this output family as descriptive in run-summary metadata and report text

## Output Contract

All supplementary prototype outputs should live inside the existing `stage3_rsa` root.

Required supplementary tables:

- `prototype_rsa_results__per_date.parquet`
- `prototype_rdm__pooled__response_window.parquet`
- `prototype_rdm__pooled__full_trajectory.parquet`

Minimum schema for `prototype_rsa_results__per_date.parquet`:

- `date`
- `view_name`
- `reference_view_name`
- `comparison_scope`
- `model_id`
- `model_label`
- `model_tier`
- `model_status`
- `excluded_from_primary_ranking`
- `score_method`
- `score_status`
- `n_stimuli`
- `n_shared_entries`
- `rsa_similarity`
- `p_value_raw`
- `p_value_fdr`
- `is_top_model`

Minimum schema for pooled prototype RDM tables:

- `stimulus_row`
- one numeric column per pooled stimulus

Required supplementary QC tables:

- `prototype_support__per_date.parquet`
- `prototype_support__pooled.parquet`

Minimum schema for `prototype_support__per_date.parquet`:

- `date`
- `view_name`
- `stimulus`
- `stim_name`
- `n_trials`
- `n_total_features`
- `n_supported_features`
- `n_all_nan_features`

Minimum schema for `prototype_support__pooled.parquet`:

- `view_name`
- `stimulus`
- `stim_name`
- `n_trials`
- `n_dates_contributed`
- `n_total_features`
- `n_supported_features`
- `n_all_nan_features`

Required supplementary figures:

- `prototype_rsa__per_date__response_window.png`
- `prototype_rsa__per_date__full_trajectory.png`
- `prototype_rdm__pooled__response_window.png`
- `prototype_rdm__pooled__full_trajectory.png`

Figure content contract:

- `prototype_rsa__per_date__<view>.png` summarizes per-date prototype RSA for that view only
- `prototype_rdm__pooled__<view>.png` is a neural-only pooled prototype RDM heatmap for that view only

Run-summary additions:

- `prototype_supplement_enabled`
- `prototype_views`
- `prototype_dates`
- `prototype_table_names`
- `prototype_figure_names`
- `prototype_descriptive_outputs`

Existing summary compatibility rules:

- keep current `rsa_table_names` unchanged for the primary Stage 3 outputs
- list prototype supplementary tables in `additional_table_names`
- append prototype supplementary figure names to `figure_names`
- do not remove or rename any existing primary summary fields

## Reporting Rules

The markdown and JSON summary should state clearly:

- pooled-RDM Stage 3 remains the primary result
- per-date prototype RSA is supplementary inferential support
- pooled prototype RDM is descriptive only

Suggested wording rule:

- avoid language such as "stronger evidence than Stage 3"
- use language such as "supplementary prototype-level structure" or "descriptive pooled prototype geometry"

## Error Handling

The supplementary branch should be operationally conservative.

Rules:

- if prototype inputs are absent, skip supplementary prototype analyses and keep primary Stage 3 intact
- if a specific `date x view` lacks enough stimuli or produces too few shared entries, mark that supplementary RSA row `invalid`
- if pooled prototype RDM cannot be built for a view, write an empty-state figure rather than failing the whole Stage 3 run
- if a prototype stimulus set cannot be aligned cleanly back to `stimulus_sample_map`, fail the supplementary branch explicitly and leave the primary branch untouched
- supplementary branch failures must not silently alter the primary pooled-RDM RSA outputs

## Testing Strategy

### Prototype Construction Tests

- verify that prototype vectors are averaged over the correct grouping unit
- verify that `response_window` and `full_trajectory` use the approved view definitions
- verify that pooled prototype construction ignores date boundaries only in the explicitly pooled descriptive branch
- verify that prototype construction uses the tensor as the authoritative numerical source
- verify that tensor rows align to metadata rows before any grouping
- verify that prototype neural RDMs use the approved overlap-aware correlation-distance contract
- verify that invalid prototype-pair distances are written as `NaN` rather than coerced finite values

### RSA Contract Tests

- verify that per-date prototype RSA reuses the Stage 3 scoring rules
- verify that date-specific stimulus restriction is applied before RSA scoring
- verify that insufficient shared entries produce `invalid` rows rather than forced scores
- verify that supplementary `p_value_fdr` is computed across the full supplementary per-date result table and nowhere else
- verify that `is_top_model` excludes rows marked `excluded_from_primary_ranking`

### Output Contract Tests

- verify that supplementary prototype outputs are written only when preprocessing inputs are supplied
- verify that the primary `rsa_results.parquet` is unchanged when the supplementary branch is enabled
- verify that prototype tables and figures use explicit `prototype_...` names
- verify that reruns remove stale prototype figures from the same output family
- verify that `additional_table_names` and `figure_names` expose the supplementary artifacts as specified

### Integration Tests

- verify that `scripts/run_rsa.py` still succeeds without `--preprocess-root`
- verify that `scripts/run_rsa.py` succeeds with `--preprocess-root` and writes the supplementary prototype outputs
- verify that current primary Stage 3 CLI smoke expectations remain valid

## Implementation Boundaries

Primary code changes are expected in:

- `scripts/run_rsa.py`
- `src/bacteria_analysis/rsa.py`
- `src/bacteria_analysis/rsa_outputs.py`
- `tests/test_rsa.py`
- `tests/test_rsa_cli_smoke.py`

No changes should be required in:

- Stage 1 statistical logic
- Stage 2 inferential logic
- current model-space schema

## Acceptance Criteria

This design is complete when:

- current pooled-RDM Stage 3 outputs still run and remain unchanged
- Stage 3 can optionally ingest preprocessing inputs for prototype supplements
- per-date prototype RSA runs for both approved views
- pooled prototype RDM figures and tables are written for both approved views
- supplementary outputs are clearly labeled as supplementary or descriptive where appropriate
- tests cover the prototype construction, scoring boundaries, and output contracts
