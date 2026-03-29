# Stage 1 Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 1 reliability analysis to test whether trial-level chemosensory population responses are reproducible for the same stimulus and generalize across individuals and dates.

**Architecture:** Build Stage 1 on top of the completed preprocessing outputs. Keep the analysis trial-first and view-based: generate multiple neural representations from the same preprocessed trials, compute overlap-aware correlation distances, then evaluate reliability with same-vs-different contrasts, leave-one-individual-out, leave-one-date-out, split-half, permutation nulls, and grouped bootstrap. Keep the CLI thin and push all reusable logic into `src/bacteria_analysis/`. `Stage 1` is the project phase name; Python modules and scripts should use responsibility-based names rather than stage-prefixed names.

**Tech Stack:** Python 3.11, pandas, numpy, pyarrow, pytest, matplotlib/seaborn for figures, standard library (`pathlib`, `json`, `argparse`)

---

## Overview

Stage 1 is the scientific gate for the rest of the project.

This stage must answer one question cleanly:

> are trials from the same stimulus more similar than trials from different stimuli, and does that relationship survive held-out individuals and held-out dates?

This plan assumes the current Stage 0 outputs are the canonical inputs:

- `data/processed/trial_level/trial_metadata.parquet`
- `data/processed/trial_level/trial_wide_baseline_centered.parquet`
- `data/processed/trial_level/trial_tensor_baseline_centered.npz`

The implementation should support four representation views:

- full trajectory: `0..44`
- ON window: `6..15`
- response window: `6..20`
- post window: `16..44`

Formal reporting should use a single primary view, with `response_window` as the default. The other three views should still be computed, but they should be treated as supplementary robustness checks rather than co-equal formal conclusion views. The CLI should allow an explicit primary-view override, and the chosen view should be written into the run summary.

## File Structure

- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability.py`
  Responsibility: load Stage 0 outputs into analysis-ready views, compute overlap-aware distances, same-vs-different summaries, and cross-validation results.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability_stats.py`
  Responsibility: permutation nulls, grouped bootstrap, confidence intervals, and null-comparison summaries.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: convert Stage 1 results into tidy tables and figure files, without mixing plotting into the core analysis logic.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_reliability.py`
  Responsibility: thin CLI that orchestrates Stage 1 from preprocessed inputs to saved outputs.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability.py`
  Responsibility: unit tests for view building, overlap-aware distance calculation, same-vs-different summaries, LOIO/LODO logic, and split-half behavior.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_stats.py`
  Responsibility: unit tests for permutation and grouped bootstrap behavior.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_cli_smoke.py`
  Responsibility: end-to-end CLI smoke test on a tiny synthetic Stage 0-like fixture.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`
  Responsibility: add reusable Stage 1 fixture builders derived from the existing synthetic preprocessing fixtures.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml`
  Responsibility: add minimal plotting dependencies and an optional Stage 1 task alias.

## Output Structure

Stage 1 outputs should live under:

- `results/stage1_reliability/`

Recommended subdirectories:

- `results/stage1_reliability/tables/`
- `results/stage1_reliability/figures/`
- `results/stage1_reliability/qc/`

Expected artifact families:

- per-view same-vs-different summary tables
- leave-one-individual-out result tables
- leave-one-date-out result tables
- split-half result tables
- permutation/bootstrap summary tables
- QC tables for overlap-neuron usage and excluded comparisons
- figure panels for the main Stage 1 result families
- one run-summary JSON/Markdown file

Formal result summaries should prioritize:

- selected-primary-view same-vs-different
- selected-primary-view LOIO
- selected-primary-view LODO
- selected-primary-view permutation/bootstrap summary

The following should remain supplementary unless explicitly promoted later:

- split-half summaries
- cross-view comparison figures/tables
- within-date cross-individual same-vs-different analyses
- per-date LOIO analyses
- stimulus-by-stimulus distance matrices

## Sprint 1: Build Stage 1 Analysis Inputs

**Goal:** Turn Stage 0 outputs into consistent Stage 1 analysis views.

**Demo/Validation:**

- run targeted tests for view slicing and individual/date definitions
- inspect one synthetic fixture through all four views

### Task 1.1: Define Stage 1 constants and view windows

- **Location**: `src/bacteria_analysis/reliability.py`
- **Description**: define the four Stage 1 views, their time windows, the `individual_id = date + worm_key` rule, and the canonical correlation-distance-first configuration.
- **Dependencies**: existing Stage 0 constants and preprocessing outputs
- **Acceptance Criteria**:
  - all four view names are explicit and stable
  - `individual_id` is derived from `date + worm_key`, never from `worm_key` alone
  - the default distance metric is correlation distance
- **Validation**:
  - add unit tests for view window definitions
  - add unit tests for `individual_id` construction

### Task 1.2: Build view-specific trial representations

- **Location**: `src/bacteria_analysis/reliability.py`
- **Description**: load Stage 0 trial data and materialize per-trial representations for full, ON, response, and post views while preserving `NaN` values and metadata alignment.
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - each view returns a trial-indexed representation aligned to metadata
  - each view preserves missing neurons and missing time points as `NaN`
  - representation dimensions are deterministic per view
- **Validation**:
  - add unit tests for expected shapes
  - add unit tests that verify time slicing is correct for each view

### Task 1.3: Define overlap-aware comparison records

- **Location**: `src/bacteria_analysis/reliability.py`
- **Description**: create a comparison helper that, for any pair of trials, identifies overlapping neurons, flattens the overlapping `neuron x time` view, and records overlap-neuron count as QC.
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - comparisons use only overlapping neurons
  - comparisons fail clearly or skip clearly if overlap is insufficient
  - overlap-neuron count is recorded on every comparison record
- **Validation**:
  - unit tests for same-trial and cross-trial overlap logic
  - unit tests for missing-neuron handling

## Sprint 2: Implement Core Reliability Statistics

**Goal:** Compute same-vs-different distance structure and held-out generalization.

**Demo/Validation:**

- produce synthetic-data tables showing same-stimulus distances below different-stimulus distances
- run LOIO and LODO on synthetic fixtures

### Task 2.1: Implement correlation-distance comparison engine

- **Location**: `src/bacteria_analysis/reliability.py`
- **Description**: compute correlation distances between trial pairs using the overlap-aware representation from Sprint 1.
- **Dependencies**: Task 1.3
- **Acceptance Criteria**:
  - correlation distance is the default path
  - invalid comparisons are either excluded or labeled explicitly
  - same-stimulus and different-stimulus labels are attached to each comparison
- **Validation**:
  - unit tests for expected distance behavior on simple synthetic vectors
  - unit tests for `NaN`-safe comparison handling

### Task 2.2: Implement same-vs-different summary analysis

- **Location**: `src/bacteria_analysis/reliability.py`
- **Description**: aggregate trial-pair distances into per-view same-vs-different summaries and compute the core distance gap `different - same`.
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - output tables include view name, counts, means, medians, and distance gap
  - summaries preserve QC metadata such as overlap-neuron counts
  - reporting layer can cleanly promote `response_window` as the primary formal summary view
- **Validation**:
  - unit tests for expected summary columns
  - unit tests for deterministic summary values on fixtures

### Task 2.3: Implement leave-one-individual-out reliability

- **Location**: `src/bacteria_analysis/reliability.py`
- **Description**: hold out one `individual_id` at a time, build training stimulus references from the remaining individuals, and score whether held-out trials are closest to their own stimulus more often than chance.
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - training and held-out sets are disjoint by `individual_id`
  - one held-out record is produced per eligible individual and view
  - output clearly states excluded stimuli/trials if coverage is insufficient
- **Validation**:
  - unit tests that verify held-out individual isolation
  - unit tests for expected score behavior on a controlled fixture

### Task 2.4: Implement leave-one-date-out reliability

- **Location**: `src/bacteria_analysis/reliability.py`
- **Description**: repeat the same reliability logic at the date level.
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - training and held-out sets are disjoint by `date`
  - date-level summaries mirror the individual-level output contract
- **Validation**:
  - unit tests for date isolation
  - unit tests for excluded-date handling

### Task 2.5: Implement split-half reliability as supplementary analysis

- **Location**: `src/bacteria_analysis/reliability.py`
- **Description**: repeatedly split eligible trials into balanced halves and compute cross-half reliability summaries.
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - split-half is clearly labeled supplementary
  - the randomization path is seedable and reproducible
  - outputs are per-view and per-repeat
- **Validation**:
  - unit tests for reproducibility with a fixed seed
  - unit tests for balanced split constraints

## Sprint 3: Add Permutation and Bootstrap Inference

**Goal:** Quantify whether reliability exceeds chance and measure uncertainty at the individual level.

**Demo/Validation:**

- produce null distributions on synthetic fixtures
- produce grouped-bootstrap confidence intervals on synthetic fixtures

### Task 3.1: Implement structure-preserving permutation nulls

- **Location**: `src/bacteria_analysis/reliability_stats.py`
- **Description**: shuffle stimulus labels under a scheme that preserves the nested trial structure enough to create a defensible null distribution for the main Stage 1 statistic.
- **Dependencies**: Sprint 2 outputs
- **Acceptance Criteria**:
  - permutation function is explicit about what is shuffled
  - null outputs are returned per view and per statistic
  - observed statistics can be compared to null summaries cleanly
- **Validation**:
  - unit tests that verify trial counts remain unchanged
  - unit tests that null outputs have expected iteration counts

### Task 3.2: Implement grouped bootstrap

- **Location**: `src/bacteria_analysis/reliability_stats.py`
- **Description**: resample by `individual_id = date + worm_key`, carrying all that individual's trials into each bootstrap sample.
- **Dependencies**: Sprint 2 outputs
- **Acceptance Criteria**:
  - bootstrap unit is the individual, not rows or time points
  - outputs include mean, lower CI, upper CI
  - bootstrap results are reproducible with a fixed seed
- **Validation**:
  - unit tests for grouped resampling behavior
  - unit tests that verify the bootstrap does not split individuals

### Task 3.3: Build final Stage 1 statistical summary tables

- **Location**: `src/bacteria_analysis/reliability_stats.py`, `src/bacteria_analysis/reliability_outputs.py`
- **Description**: combine observed reliability statistics, permutation summaries, and bootstrap intervals into per-view final summary tables.
- **Dependencies**: Tasks 3.1 and 3.2
- **Acceptance Criteria**:
  - final summary table is tidy and per-view
  - the main Stage 1 outputs can be read without recomputing internal state
- **Validation**:
  - unit tests for expected output schema
  - fixture-based exact-value checks where deterministic

## Sprint 4: Reporting, Figures, and CLI

**Goal:** Make Stage 1 runnable end-to-end on the real dataset and save figures/tables for inspection.

**Demo/Validation:**

- run Stage 1 on synthetic data through the CLI
- produce all expected tables and figures

### Task 4.1: Implement table and QC writers

- **Location**: `src/bacteria_analysis/reliability_outputs.py`
- **Description**: save all Stage 1 tables and QC summaries into a predictable directory structure under `results/stage1_reliability/`.
- **Dependencies**: Sprint 3 outputs
- **Acceptance Criteria**:
  - per-view tables, LOIO/LODO tables, split-half tables, and QC tables are all saved
  - file naming is deterministic and readable
- **Validation**:
  - unit tests for file creation and schema round-trips

### Task 4.2: Implement Stage 1 figures

- **Location**: `src/bacteria_analysis/reliability_outputs.py`
- **Description**: create the minimum required Stage 1 figure families:
  - same-vs-different distributions
  - LOIO summary
  - LODO summary
  - split-half summary
  - cross-view reliability comparison
  - overlap-neuron-count QC summary
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - figures are saved without manual intervention
  - figure filenames match the output contract
  - PCA, if included, is labeled as supplementary
  - formal figures center the selected primary view as the main display view, defaulting to `response_window`
  - the main same-vs-different figure annotates the permutation `p` value
  - the main same-vs-different figure does not overlay all pairwise dots by default
- **Validation**:
  - smoke test that the expected figure files are produced

### Task 4.3: Implement the Stage 1 CLI

- **Location**: `scripts/run_reliability.py`
- **Description**: build a thin CLI that loads Stage 0 outputs, runs Stage 1, writes all tables/figures, and prints a short run summary.
- **Dependencies**: Tasks 4.1 and 4.2
- **Acceptance Criteria**:
  - CLI uses reusable library functions instead of duplicating logic
  - CLI accepts explicit input paths and output root
  - CLI accepts an optional primary-view override while defaulting to `response_window`
  - CLI fails clearly if Stage 0 inputs are missing
- **Validation**:
  - end-to-end smoke test in `tests/test_reliability_cli_smoke.py`

## Sprint 5: Real-Data Execution and Review

**Goal:** Run Stage 1 on the real dataset and verify whether the reliability gate is met.

**Demo/Validation:**

- run the Stage 1 CLI on the real preprocessed outputs
- inspect the main Stage 1 tables and figures

### Task 5.1: Execute Stage 1 on the real dataset

- **Location**: `scripts/run_reliability.py`
- **Description**: run the Stage 1 CLI against the real Stage 0 outputs and write results under `results/stage1_reliability/`.
- **Dependencies**: Sprint 4 complete
- **Acceptance Criteria**:
  - run completes without code changes caused by real-data edge cases
  - all expected Stage 1 artifacts are produced
- **Validation**:
  - manual inspection of output directory contents
  - rerun CLI from a clean output directory

### Task 5.2: Verify the scientific gate

- **Location**: Stage 1 result tables and figures
- **Description**: inspect the main Stage 1 outputs and determine whether Stage 1 passed, failed, or is inconclusive.
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - a short written summary states whether Stage 1 passed
  - the summary identifies which representation views are strongest
  - any limitations are explicit
- **Validation**:
  - compare observed statistics to nulls and grouped-bootstrap intervals
  - confirm the conclusion matches the Stage 1 exit criteria from the spec

## Testing Strategy

- Unit-test each Stage 1 function on deterministic synthetic fixtures
- Keep synthetic fixtures small but structured enough to represent:
  - multiple stimuli
  - multiple individuals
  - multiple dates
  - overlapping and non-overlapping neuron sets
- Add dedicated tests for:
  - overlap-neuron handling
  - correlation-distance behavior
  - LOIO/LODO isolation
  - permutation iteration counts
  - grouped bootstrap grouping behavior
  - CLI output layout

## Potential Risks & Gotchas

- `worm_key` is not globally unique; using it alone will silently break individual-level inference
- overlap-neuron counts may vary strongly across trial pairs; Stage 1 must report this clearly
- some stimuli may have insufficient held-out coverage in certain views
- correlation distance can fail on degenerate vectors; the implementation must define how those cases are excluded or labeled
- split-half can look better than LOIO; that should not be mistaken for the main evidence
- permutation logic can become invalid if label shuffling destroys the nested data structure
- grouped bootstrap must resample individuals, not rows
- Stage 1 may be inconclusive even if preprocessing succeeded; that is a valid scientific outcome

## Rollback Plan

- If real-data Stage 1 fails because the implementation is wrong, revert only the Stage 1 analysis files and keep Stage 0 preprocessing intact
- If Stage 1 runs correctly but yields weak reliability, do not force Stage 2; record the result as failed or inconclusive and revisit the representation choices or experimental coverage

## Completion Definition

Stage 1 implementation is complete when:

- the Stage 1 CLI runs from preprocessed inputs to saved outputs
- tests cover the full Stage 1 pipeline and pass
- real-data outputs are generated under `results/stage1_reliability/`
- the formal reporting path uses one selected primary view, defaulting to `response_window`, with the other three views clearly marked supplementary
- a short Stage 1 result summary can state one of:
  - passed
  - failed
  - inconclusive
