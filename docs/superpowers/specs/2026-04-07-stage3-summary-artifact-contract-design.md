# Stage 3 Summary Artifact Contract Design

Date: 2026-04-07
Status: Drafted after user-approved design
Topic: Rename the leave-one-stimulus figure artifact, remove redundant Stage 3 histogram figures, and move their full details into the run summary

## Goal

Stage 3 currently writes three summary-style figures:

- `ranked_model_rsa.png`
- `leave_one_stimulus_out_robustness.png`
- `view_comparison_summary.png`

Only one of these still needs to remain as a figure artifact. The other two are better represented as structured summary content than as small histogram or bar-chart outputs.

The goal of this design is to:

- rename `leave_one_stimulus_out_robustness` to the more direct `single_stimulus_sensitivity`
- stop generating `ranked_model_rsa.png`
- stop generating `view_comparison_summary.png`
- preserve the full details behind those two removed figures inside `run_summary.json`
- mirror those details in a readable `run_summary.md`

This is an output-contract redesign only. It must not change Stage 3 RSA statistics, ranking logic, or parquet table schemas.

## Current Context

Current Stage 3 summary-style output writing lives in:

- `src/bacteria_analysis/rsa_outputs.py`

Current behavior:

- `ranked_model_rsa.png` is generated from `rsa_results` for the current `focus_view` and `ranked_models`
- `leave_one_stimulus_out_robustness.png` is generated from `rsa_leave_one_stimulus_out`
- `view_comparison_summary.png` is generated from `rsa_view_comparison`
- `run_summary.json` records these figure names as figure artifacts, but does not carry the full plotted details for the first and third figure

This creates two problems:

- `ranked_model_rsa` and `view_comparison_summary` duplicate information that is more useful as structured summary data
- `leave_one_stimulus_out_robustness` is not a clear artifact name for readers

## User-Requested Changes

The approved output changes are:

1. rename `leave_one_stimulus_out_robustness` to `single_stimulus_sensitivity`
2. remove `ranked_model_rsa.png`
3. remove `view_comparison_summary.png`
4. keep the full details formerly represented by those two figures in `run_summary`
5. keep those details complete, not reduced to a short summary

## Options Considered

### Option 1: Keep Existing Figures And Also Add Summary Details

Pros:

- lowest behavior change
- preserves old file paths

Cons:

- duplicates the same information in both figures and summaries
- keeps the figure directory noisy
- does not clean up the Stage 3 artifact contract

### Option 2: Remove Redundant Figures And Move Their Full Details Into Run Summary

Pros:

- cleanest artifact contract
- summary content stays machine-readable and human-readable
- matches the user request directly

Cons:

- breaks expectations for anyone depending on those old figure file names
- requires markdown summary expansion

### Option 3: Replace Figures With Additional Parquet Tables Instead Of Run Summary Content

Pros:

- keeps output data structured
- reduces run-summary size

Cons:

- pushes readers back into table inspection
- does not satisfy the request to write these details into `run_summary`

## Chosen Approach

Use Option 2.

The Stage 3 writer should:

- keep a figure for stimulus sensitivity, but rename it to `single_stimulus_sensitivity`
- stop writing the `ranked_model_rsa` and `view_comparison_summary` histogram figures
- move the complete detail payloads for those outputs into `run_summary.json`
- mirror the same information in `run_summary.md`

## Scope

This design covers:

- Stage 3 artifact naming for the stimulus-sensitivity figure
- Stage 3 `figure_names` contract
- Stage 3 run-summary schema additions
- Stage 3 markdown summary formatting for ranked-model and cross-view details
- cleanup of stale legacy summary figures in the results directory
- tests for the updated output contract

This design does not cover:

- RSA score computation
- permutation testing
- model ranking logic
- leave-one-stimulus sensitivity computation
- redesign of the `single_stimulus_sensitivity` plot itself
- Stage 3 paired-RDM figure rendering
- Stage 2 outputs

## Artifact Contract

### Stimulus-Sensitivity Figure Rename

The figure artifact currently named `leave_one_stimulus_out_robustness` should be renamed everywhere in Stage 3 output writing to `single_stimulus_sensitivity`.

This includes:

- figure file name: `single_stimulus_sensitivity.png`
- writer return key in `written`
- `figure_names` entries in `run_summary`
- markdown summary artifact listing
- tests that assert figure presence or artifact naming

No backward-compatible alias is required for the old artifact name.

### Removed Figure Artifacts

The writer should stop generating these figures:

- `ranked_model_rsa.png`
- `view_comparison_summary.png`

They should also be removed from:

- `REQUIRED_FIGURES`
- `figure_names`
- figure existence tests
- markdown summary figure listings

### Stale Figure Cleanup

When rewriting a Stage 3 output directory, stale legacy figure files should be deleted if present:

- `ranked_model_rsa.png`
- `view_comparison_summary.png`
- `leave_one_stimulus_out_robustness.png`

This prevents old artifacts from remaining visible after the new writer contract takes effect.

## Run Summary Contract

### JSON Summary Additions

`run_summary.json` should gain two new top-level fields:

- `ranked_model_rsa_details`
- `view_comparison_details`

Both fields should always exist and contain arrays. If no rows are available, they should be empty arrays.

### ranked_model_rsa_details

This field should contain the full ranked-model detail payload that the removed `ranked_model_rsa.png` used to summarize.

Inclusion rules:

- source table: `rsa_results`
- include only rows for models in `ranked_models`
- if `focus_view` is not `None`, include only rows for that `focus_view`
- include only rows with finite `rsa_similarity`

Sort order:

- descending `rsa_similarity`
- ascending `model_id` as a stable tie-breaker

Each object should preserve these fields when available:

- `view_name`
- `model_id`
- `rsa_similarity`
- `p_value_raw`
- `p_value_fdr`
- `n_shared_entries`
- `score_status`
- `is_top_model`

The summary builder may include additional stable fields already present in the selected rows, but it must not omit the fields above when they exist in the input table.

### view_comparison_details

This field should contain the full detail payload that the removed `view_comparison_summary.png` used to summarize.

Inclusion rules:

- source table: `rsa_view_comparison`
- include only rows with finite `rsa_similarity`

Sort order:

- preserve the existing table row order from `rsa_view_comparison`
- do not re-sort these rows inside the summary builder in this round

Each object should preserve these fields when available:

- `view_name`
- `reference_view_name`
- `comparison_scope`
- `rsa_similarity`
- `p_value_raw`
- `p_value_fdr`
- `n_shared_entries`
- `score_status`

The summary builder may include additional stable fields already present in the selected rows, but it must not omit the fields above when they exist in the input table.

## Markdown Summary Contract

`run_summary.md` should mirror the new JSON details in readable text form.

Add two sections:

- `## Ranked Model RSA Details`
- `## View Comparison Details`

Formatting requirements:

- one flat bullet per record
- no nested bullets
- include the key fields needed to interpret each row directly from markdown
- if a section has no rows, emit `- None`

Recommended bullet shape for ranked-model details:

- `<model_id> | view=<view_name> | rsa=<rsa_similarity> | p_raw=<p_value_raw> | p_fdr=<p_value_fdr> | n=<n_shared_entries> | status=<score_status> | top=<is_top_model>`

Recommended bullet shape for view-comparison details:

- `<view_name> vs <reference_view_name> | scope=<comparison_scope> | rsa=<rsa_similarity> | p_raw=<p_value_raw> | p_fdr=<p_value_fdr> | n=<n_shared_entries> | status=<score_status>`

The markdown writer should keep the field order shown above stable for both bullet formats.

Exact numeric formatting does not need to be artificially rounded by this design unless the existing summary writer already enforces a stable format.

## Implementation Boundaries

The implementation should stay within output writing and tests.

Expected primary code touch points:

- `src/bacteria_analysis/rsa_outputs.py`
- `tests/test_rsa.py`
- `tests/test_rsa_cli_smoke.py`

The implementation must not modify:

- `src/bacteria_analysis/rsa.py`
- Stage 3 parquet table schemas
- any upstream computation that feeds `rsa_results`, `rsa_leave_one_stimulus_out`, or `rsa_view_comparison`

The internal plotting helper for stimulus sensitivity may keep its current plotting logic in this round. Only the public artifact name changes.

The old plotting helpers for `ranked_model_rsa` and `view_comparison_summary` may be deleted or left unused, but the preferred outcome is to remove dead code once summary extraction logic replaces them.

## Testing Requirements

Tests should cover:

1. the renamed figure artifact is written as `single_stimulus_sensitivity.png`
2. the old `leave_one_stimulus_out_robustness.png` path is no longer produced
3. `ranked_model_rsa.png` is no longer produced
4. `view_comparison_summary.png` is no longer produced
5. `figure_names` contains only still-written figure artifacts
6. `run_summary.json` includes full `ranked_model_rsa_details`
7. `run_summary.json` includes full `view_comparison_details`
8. `run_summary.md` renders both new detail sections
9. stale legacy figures are deleted during refresh

CLI smoke tests should also be updated so the expected artifact list matches the new output contract.

## Acceptance Criteria

This design is satisfied when:

- Stage 3 writes `single_stimulus_sensitivity.png`
- Stage 3 no longer writes `ranked_model_rsa.png`
- Stage 3 no longer writes `view_comparison_summary.png`
- `run_summary.json` contains complete ranked-model and cross-view detail arrays
- `run_summary.md` exposes the same content in readable text
- Stage 3 refresh removes stale legacy summary figures from existing result directories
- tests reflect the new artifact contract and pass
