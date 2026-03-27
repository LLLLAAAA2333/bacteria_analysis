# Data Preprocessing Design

Date: 2026-03-27
Status: Drafted after user-approved design
Topic: Preprocess calcium-imaging dataframe into a clean long table and standardized trial-level analysis objects

## Goal

Build a reproducible preprocessing pipeline for `data/data.parquet` that:

- preserves a clean analysis-ready long-format table
- generates a standardized trial-level wide table
- generates a standardized trial-level tensor representation
- records QC and filtering decisions explicitly

This stage is the foundation for later reliability, RSA, geometry, and neuron-contribution analyses.

## Verified Dataset Facts

The raw parquet has already been checked in this repository.

- Shape: `591165 x 11`
- Core columns are present: `neuron`, `stimulus`, `time_point`, `delta_F_over_F0`, `worm_key`, `segment_index`, `date`, `stim_name`, `stim_color`
- Extra columns present: `start_time`, `end_time`
- Time grid is fixed: `0..44` with `45` time points
- `start_time=5`, `end_time=15` for all rows
- `date + worm_key + segment_index` defines `678` unique trials
- Each trial contains exactly one stimulus
- There are `22` neurons in total
- Trial-level neuron coverage is incomplete: min `7`, max `22`, mean about `19.38`
- `delta_F_over_F0` has `5860` NaNs
- `130` trial-stimulus-neuron traces are fully NaN and `1` trace is partially NaN

These facts are treated as current expectations and should be revalidated every preprocessing run.

## Verification Note

The current preprocessing run has been verified against the real dataset and the artifact layout in `data/processed/`.

- Verified trial count: `678`
- Verified neuron count: `22`
- Verified timepoints: `45`
- Filtering policy: mild filtering, meaning fully-NaN traces are dropped and partially-NaN traces are retained
- Verified trace outcomes: `130` fully-NaN traces removed, `1` partially-NaN trace retained

## Scope

This spec covers:

- schema validation
- trial key construction
- trace-level filtering
- baseline centering
- output serialization
- QC reporting

This spec does not cover:

- feature extraction beyond baseline centering
- distance metrics, RSA, clustering, or decoding
- imputation of missing neurons
- behavior integration

## Design Principles

- Keep the raw measurement semantics intact
- Do not average across trials during preprocessing
- Do not fill missing neurons with zero
- Fail fast on structural data problems
- Keep outputs simple, explicit, and easy to reuse from `pandas`, `numpy`, and `sklearn`

## Canonical Units

### Trial Key

Use:

`trial_id = "{date}__{worm_key}__{segment_index}"`

This is the canonical trial identifier for all downstream outputs.

### Trace Unit

The smallest analysis trace unit is:

`trial_id × stimulus × neuron`

Given the current dataset, each `trial_id` maps to exactly one stimulus, but `stimulus` must still remain an explicit column rather than being inferred from `trial_id`.

## Preprocessing Rules

### 1. Schema Validation

Required columns:

- `neuron`
- `stimulus`
- `time_point`
- `delta_F_over_F0`
- `worm_key`
- `segment_index`
- `date`
- `stim_name`
- `stim_color`

Optional-but-expected columns:

- `start_time`
- `end_time`

Validation checks:

- required columns must exist
- `time_point` must cover `0..44`
- each observed `trial_id × stimulus × neuron` group must have exactly `45` rows
- each observed `trial_id × stimulus × neuron` group must contain `45` unique `time_point` values
- each `trial_id` must map to exactly one stimulus
- `stimulus -> stim_name` must be one-to-one for each row group
- `stimulus -> stim_color` must be one-to-one for each row group

If any structural validation fails, preprocessing should stop with a clear error message.

### 2. Trace Quality Flags

For each `trial_id × stimulus × neuron` trace, compute:

- `is_all_nan_trace`
- `has_any_nan_trace`
- `n_valid_points`
- `n_valid_baseline_points`

### 3. Filtering Policy

Use the agreed mild filtering strategy:

- drop traces where all 45 signal values are NaN
- keep partially observed traces
- do not interpolate missing values
- do not impute missing neurons
- do not fill missing values with zero

This filtering happens at the trace level, not the whole-trial level.

### 4. Baseline Centering

For each surviving `trial_id × stimulus × neuron` trace:

- baseline window is `time_point in [0, 5]`
- compute `baseline_mean` from available non-NaN baseline points
- compute `dff_baseline_centered = delta_F_over_F0 - baseline_mean`

Additional rules:

- if all baseline points are NaN, keep the trace but mark `baseline_valid = false`
- when `baseline_valid = false`, `dff_baseline_centered` remains NaN for that trace
- do not rescale by baseline variance or baseline mean magnitude

This step is centering only, not normalization.

## Output Design

All generated artifacts should live under:

`data/processed/`

### 1. Clean Long Table

Path:

`data/processed/clean/neuron_segments_clean.parquet`

Row shape:

- one row per original observation after dropping fully-NaN traces

Columns:

- all original raw columns
- `trial_id`
- `baseline_mean`
- `baseline_valid`
- `n_valid_points`
- `n_valid_baseline_points`
- `is_all_nan_trace`
- `has_any_nan_trace`
- `dff_baseline_centered`

Purpose:

- canonical cleaned source for plotting, grouped summaries, and any downstream reshaping

### 2. Trial Metadata Table

Path:

`data/processed/trial_level/trial_metadata.parquet`

One row per `trial_id`

Columns should include:

- `trial_id`
- `date`
- `worm_key`
- `segment_index`
- `stimulus`
- `stim_name`
- `stim_color`
- `n_observed_neurons`
- `n_missing_neurons`
- `n_all_nan_traces_removed`
- `has_partial_nan_trace`

Purpose:

- row index authority for wide and tensor outputs

### 3. Trial-Level Wide Table

Path:

`data/processed/trial_level/trial_wide_baseline_centered.parquet`

One row per `trial_id`

Feature columns:

- one column per `neuron × time_point`
- naming convention: `{neuron}__t{time_point:02d}`
- values are `dff_baseline_centered`

Metadata columns:

- copy the main trial metadata columns into the left side of the table

Missing data:

- if a neuron is unobserved in a trial, all its feature columns remain NaN
- if a trace is partially NaN, only those affected time points remain NaN

Purpose:

- direct use with `pandas`, `scikit-learn`, and distance-based analyses

### 4. Trial-Level Tensor Object

Paths:

- `data/processed/trial_level/trial_tensor_baseline_centered.npz`

Tensor definition:

- shape: `(n_trials, 22, 45)`
- axis 0: trial order aligned exactly with `trial_metadata.parquet`
- axis 1: fixed neuron order from the documented 22-neuron list
- axis 2: fixed time order `0..44`
- the NPZ stores the tensor plus aligned `trial_ids`, `stimulus_labels`, and `stim_name_labels`

Missing data:

- unobserved neurons remain all-NaN along their time axis
- partially missing time points remain NaN

Purpose:

- direct use with time-resolved distance calculations, tensor workflows, and trajectory analysis

### 5. QC Report

Paths:

- `data/processed/qc/preprocessing_report.json`
- `data/processed/qc/preprocessing_report.md`

Must include:

- input row count
- output row count
- number of unique trials
- number of stimuli
- number of neurons
- number of fully-NaN traces removed
- number of partially-NaN traces retained
- neuron coverage distribution across trials
- trials per stimulus summary

Purpose:

- human-readable and machine-readable provenance for preprocessing decisions

## Proposed Repository Structure

Keep the implementation small and explicit.

- `scripts/run_preprocessing.py`
- `src/bacteria_analysis/preprocessing.py`
- `src/bacteria_analysis/io.py`
- `src/bacteria_analysis/constants.py`
- `tests/test_preprocessing.py`

Responsibilities:

- `run_preprocessing.py`: CLI entry point
- `preprocessing.py`: validation, filtering, centering, reshaping
- `io.py`: read/write helpers
- `constants.py`: neuron order, required columns, time grid
- `test_preprocessing.py`: focused checks on preprocessing behavior

## Data Flow

1. Load `data/data.parquet`
2. Validate schema and structural assumptions
3. Build `trial_id`
4. Compute trace-level QC flags
5. Drop fully-NaN traces
6. Compute baseline mean and centered signal
7. Materialize cleaned long table
8. Build trial metadata
9. Pivot to wide table
10. Build tensor object with fixed neuron/time axes
11. Write QC report

## Error Handling

Stop with an error on:

- missing required columns
- broken time grid
- duplicated time points within a trace
- more than one stimulus per trial
- inconsistent `stim_name` or `stim_color` within a stimulus

Proceed with warning on:

- partially NaN traces
- duplicated `stim_name` across multiple stimuli
- duplicated `stim_color` across multiple stimuli
- low neuron coverage in individual trials

## Testing Strategy

Minimum tests for first implementation:

- trial key construction is deterministic
- fully-NaN traces are removed
- partially-NaN traces are retained
- baseline mean uses only available baseline points
- centered signal equals raw minus baseline mean
- wide output shape matches `n_trials × (22 × 45)` plus metadata columns
- tensor output shape matches `(n_trials, 22, 45)`
- metadata row order matches wide and tensor row order

Integration checks against the current dataset:

- output contains `678` trial rows unless later filtering rules intentionally change that
- time axis remains length `45`
- neuron axis remains length `22`

## Success Criteria

The preprocessing stage is complete when:

- the pipeline runs from a single command through Pixi
- all four artifact families are written successfully
- QC output explains exactly what was filtered and why
- the same input produces identical outputs on repeated runs
- downstream analysis can consume either the clean long table, the wide table, or the tensor without additional reshaping

## Open Notes

- `stimulus` must remain the true unique stimulus identifier; `stim_name` is not unique across the current dataset
- this stage intentionally avoids early feature engineering so later representational analyses can choose their own summary strategy
