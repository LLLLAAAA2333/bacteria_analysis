# Stage 3 RDM Visualization Refresh Design

Date: 2026-04-01
Status: Drafted after user-approved design
Topic: Replace the default Stage 3 RDM comparison figures with per-view outputs that use sample ids and neural-driven clustering

## Goal

Stage 3 currently computes valid RSA statistics, but its default RDM comparison figures are weaker than they need to be for interpretation.

The goal of this design is to replace the current default Stage 3 RDM comparison figure style so that:

- `response_window` and `full_trajectory` are rendered as separate default figures
- figure labels use short `sample_id` values such as `A226`
- both the neural and top-model RDM in each view share a single neural-driven clustered order

This is a visualization-only redesign. It should improve readability without changing any Stage 3 statistics, ranking logic, or model-space inputs.

## Current Context

Current Stage 3 figure writing lives in:

- `src/bacteria_analysis/rsa_outputs.py`

Current default Stage 3 figures include:

- `ranked_primary_model_rsa.png`
- `neural_vs_top_model_rdm_panel.png`
- `leave_one_stimulus_out_robustness.png`
- `view_comparison_summary.png`

The current `neural_vs_top_model_rdm_panel.png` behavior is:

- one multi-row panel covering both views
- labels taken directly from the matrix frame labels
- no explicit neural-driven clustering shared between the neural and model heatmaps

For the current `202603` run, the Stage 3 statistics are already valid and stable:

- `global_profile` is the top primary model in both views
- each view has `325` shared entries
- the user wants the visual outputs to become stronger before reviewing the Stage 1/2/3 design as a whole

## User-Requested Changes

The requested visual changes are:

1. split the `response_window` and `full_trajectory` RDM comparison into separate default figures
2. display short labels such as `A226` instead of long `stim_name` labels
3. reorder the matrices by neural clustering so the structure is easier to see

The user approved replacing the current default figure contract rather than keeping backward-compatible legacy figure content.

## Options Considered

### Option 1: Per-View Default Figures With Neural-Driven Ordering

Produce one default neural-vs-top-model figure per view, render labels with `sample_id`, and apply the same neural-derived clustered order to both the neural and model RDM.

Pros:

- best matches the requested reading workflow
- makes within-view structure easy to inspect
- preserves direct visual comparability between neural and model matrices
- keeps the redesign confined to the Stage 3 output layer

Cons:

- changes the current default figure names and tests

### Option 2: Keep One Panel But Improve Internal Layout

Preserve a single panel file, but visually separate the views inside it and add clustering plus `sample_id` labels.

Pros:

- lower compatibility risk

Cons:

- still mixes two distinct views into one default image
- weaker than the requested “split by view” behavior

### Option 3: Only Change Labels

Leave the current figure layout and ordering intact, but switch labels to `sample_id`.

Pros:

- lowest implementation cost

Cons:

- does not solve the main readability problem
- leaves the current panel visually weak

## Chosen Approach

Use Option 1.

The redesign should replace the current default Stage 3 RDM comparison panel with separate per-view figures and a single ordering rule:

- each view gets its own neural-vs-top-model figure
- the ordering comes from the neural RDM only
- the model RDM is displayed in exactly the same order
- labels default to `sample_id`

This is the cleanest interpretation contract because it prevents the model heatmap from using its own independent clustering, which would make both images look internally tidy but weaken their direct pairwise comparison.

## Scope

This design covers:

- Stage 3 figure generation in `src/bacteria_analysis/rsa_outputs.py`
- per-view default RDM figure output names
- label display policy for Stage 3 RDM figures
- neural-driven clustering order for Stage 3 heatmaps
- run-summary metadata updates for the new figure outputs
- tests that validate the new figure contract and ordering behavior

This design does not cover:

- Stage 3 RSA statistics
- model ranking logic
- Stage 3 tables or parquet schemas
- Stage 2 visual outputs
- model-space curation changes
- new CLI arguments for alternate visualization modes

## Figure Output Contract

The default Stage 3 neural-vs-model figure contract should become:

- `neural_vs_top_model_rdm__response_window.png`
- `neural_vs_top_model_rdm__full_trajectory.png`

Each per-view figure should contain exactly two heatmaps:

- left: the pooled neural RDM for that view
- right: the top primary model RDM for that same view

The old `neural_vs_top_model_rdm_panel.png` should no longer be the default output.

Other default Stage 3 figures remain unchanged:

- `ranked_primary_model_rsa.png`
- `leave_one_stimulus_out_robustness.png`
- `view_comparison_summary.png`

`run_summary.json` and `run_summary.md` should report the new per-view figure names so the output summary reflects the files that are actually written.

## Display Label Contract

Stage 3 RDM figures should use a display-label resolver with this priority:

1. `sample_id`
2. `stim_name`
3. `stimulus`

The preferred source for label resolution is `stimulus_sample_map`, because it already contains all three fields and is the explicit Stage 3 alignment contract.

Rules:

- labels must remain one-to-one within each displayed matrix
- if a preferred label layer is missing or would create duplicates, the code must fall back to the next layer
- the default expected result for the current `202603` run is short labels like `A226`, `A137`, and `A179`

This keeps the default figures compact while preserving a safe fallback path for future batches or partially populated inputs.

## Ordering Contract

Each view should compute one clustered display order from the neural RDM only.

The order should then be applied to:

- the neural heatmap for that view
- the top-model heatmap for that same view

The model heatmap must not compute an independent clustering order.

Reason:

- independent clustering can make both matrices look locally clean while destroying direct cell-by-cell structural comparison
- a shared neural-driven order makes visual agreement or disagreement more honest

The clustering implementation should follow the same spirit as the existing Stage 2 heatmap clustering rather than inventing a second ordering philosophy for Stage 3.

If clustering is impossible because the matrix is too small, incomplete, or non-finite, the figure writer should fall back to the original aligned order rather than failing the entire Stage 3 run.

## Implementation Boundaries

Primary implementation file:

- `src/bacteria_analysis/rsa_outputs.py`

Expected changes in that file:

- replace `_plot_neural_vs_top_model_rdm_panel` with a per-view writer
- extend heatmap coercion so it can:
  - resolve display labels from `stimulus_sample_map`
  - compute neural-driven ordering
  - apply that ordering to both matrices in a pair
- update figure registration in the Stage 3 writer so the new per-view files are recorded in the written outputs and summary

Expected supporting test changes:

- `tests/test_rsa.py`
- `tests/test_rsa_cli_smoke.py`

No changes should be required in:

- `src/bacteria_analysis/rsa.py`
- `src/bacteria_analysis/model_space.py`
- `scripts/run_rsa.py`

## Error Handling And Fallbacks

The redesign should be visually strict but operationally conservative.

Rules:

- missing or duplicate `sample_id` display labels should fall back to `stim_name`, then `stimulus`
- matrices that cannot be clustered should still render in aligned non-clustered order
- figure-generation problems should not silently change the Stage 3 statistics
- the writer should prefer empty-state or fallback figures over crashing the whole Stage 3 pipeline when only the visualization layer is impaired

This preserves the current separation between statistical validity and output rendering quality.

## Testing Strategy

Add or update tests to cover:

- per-view Stage 3 figure files are written instead of the legacy panel file
- display labels use `sample_id` when available
- label resolution falls back safely when `sample_id` is unavailable or duplicated
- the model heatmap follows the neural clustered order rather than its own independent order
- CLI smoke tests expect the new figure files
- `run_summary` reports the new figure outputs

The tests should stay focused on output contract and ordering behavior, not on re-testing Stage 3 statistics.

## Acceptance Criteria

This redesign is complete when:

- Stage 3 writes separate default neural-vs-top-model figures for `response_window` and `full_trajectory`
- those figures display short `sample_id` labels for the current `202603` run
- each figure uses a neural-derived clustered order shared by the neural and model heatmaps
- the current Stage 3 statistical outputs remain unchanged
- tests pass with the new figure contract

## Recommended Follow-Up

After this spec is approved, the implementation plan should keep the work small and sequential:

1. update the Stage 3 figure writer contract
2. add label-resolution and ordering helpers
3. refresh tests and CLI smoke expectations
4. rerun Stage 3 on `results/202603/stage3_rsa` to verify the new visuals
