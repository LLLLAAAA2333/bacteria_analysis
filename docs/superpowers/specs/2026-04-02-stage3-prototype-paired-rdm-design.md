# Stage 3 Prototype Paired RDM Visualization Design

Date: 2026-04-02
Status: Drafted after user-approved design
Topic: Add paired neural-versus-model RDM visuals to the Stage 3 prototype supplementary outputs while keeping the existing summary plots

## Goal

Stage 3 now writes valid prototype supplementary outputs, but the current prototype figures stop short of the most useful visual comparison.

The goal of this design is to strengthen the prototype supplementary visualization layer so that:

- each `per-date` prototype result can be inspected as a neural RDM beside its best-matching model RDM
- each `pooled` prototype RDM figure also shows a model RDM beside the neural matrix
- the existing `prototype_rsa__per_date__<view>.png` summary plots remain in place

This is a visualization-only extension. It must not change Stage 3 scoring, ranking, permutation logic, or the current prototype table schemas.

## Current Context

Current Stage 3 prototype supplementary outputs already include:

- `prototype_rsa_results__per_date.parquet`
- `prototype_support__per_date.parquet`
- `prototype_support__pooled.parquet`
- `prototype_rdm__pooled__response_window.parquet`
- `prototype_rdm__pooled__full_trajectory.parquet`
- `prototype_rsa__per_date__response_window.png`
- `prototype_rsa__per_date__full_trajectory.png`
- `prototype_rdm__pooled__response_window.png`
- `prototype_rdm__pooled__full_trajectory.png`

Current gaps:

- the `per-date` prototype RSA summary plots show which model wins, but do not show the corresponding model RDM
- the pooled prototype RDM figures currently show only the neural side, so users cannot visually inspect what the chosen model geometry looks like under the same ordering

For the current `202603` batch, this gap is especially noticeable because:

- prototype supplementary results are valid
- `response_window` and `full_trajectory` can differ on the neural side
- the top model may be the same across views, which means the visual comparison should make clear that the main difference is often in the neural matrix rather than in a different model object

## User-Requested Changes

The requested changes are:

1. keep the current `prototype_rsa__per_date__<view>.png` summary plots
2. add a `per-date` paired prototype RDM figure family
3. upgrade the current pooled prototype RDM figures so the right side also shows a model RDM

The user approved a two-part design:

- `Per-date paired prototype RDM`
- `Pooled paired prototype RDM`

## Options Considered

### Option 1: Only Add Model RDMs To The Existing Pooled Prototype Figures

Pros:

- smallest code change
- improves the current pooled figures immediately

Cons:

- does not solve the missing visual bridge for the inferential `per-date` prototype RSA results

### Option 2: Add Two Paired Figure Layers While Keeping Summary Plots

Pros:

- matches the user-approved reading workflow
- gives a direct visual companion to `per-date prototype RSA`
- also strengthens the pooled descriptive figures
- preserves the existing summary-plot layer instead of replacing it

Cons:

- increases the number of prototype figures

### Option 3: Build Multi-Model Prototype Comparison Panels

Pros:

- shows more model context at once

Cons:

- visually crowded
- weakens the emphasis on the top model
- adds complexity without a clear need

## Chosen Approach

Use Option 2.

Stage 3 should keep the current summary plots and add paired RDM comparisons in two places:

- `per-date` paired prototype RDM figures as new files
- `pooled` paired prototype RDM figures by upgrading the existing pooled figure files

This gives one statistical summary layer and one structural comparison layer without changing the underlying prototype RSA contract.

## Scope

This design covers:

- Stage 3 prototype figure generation in `src/bacteria_analysis/rsa_outputs.py`
- new `per-date` paired prototype figure names
- upgraded pooled prototype figure layout
- model-selection rules for the right-hand model RDM in both figure families
- shared ordering and label rules
- summary metadata updates for the new figures
- rerun cleanup behavior for the new paired figures
- tests for the new writer contract

This design does not cover:

- changes to `prototype_rsa_results__per_date.parquet`
- changes to prototype scoring, permutation, or FDR logic
- new CLI flags
- new Stage 3 tables beyond what already exists
- replacing the current `prototype_rsa__per_date__<view>.png` summary plots

## Part 1: Per-Date Paired Prototype RDM

### Output Contract

Stage 3 should add these new figures:

- `prototype_rdm_comparison__per_date__response_window.png`
- `prototype_rdm_comparison__per_date__full_trajectory.png`

Each figure should be organized by `date`.

Recommended layout:

- one row per `date`
- left column: the prototype neural RDM for that `date x view`
- right column: the top model RDM for that same `date x view`

If there is only one date, the figure may collapse to a single-row layout without changing the file contract.

### Model Selection Rule

The right-hand model must come from:

- `prototype_rsa_results__per_date.parquet`

For each `date x view`:

- use the row where `is_top_model == True`
- ignore rows where `excluded_from_primary_ranking == True`
- if no eligible top model exists, render an empty-state right-hand panel rather than failing the whole figure

This ensures the visual comparison matches the prototype supplementary inference layer rather than inventing a second selection rule.

### Interpretation Boundary

These figures are supplementary visual companions to the existing `per-date prototype RSA` results.

They should not replace:

- `prototype_rsa__per_date__response_window.png`
- `prototype_rsa__per_date__full_trajectory.png`

The summary plots remain the compact statistical view; the paired RDM figures are the structural follow-through.

## Part 2: Pooled Paired Prototype RDM

### Output Contract

The existing pooled figure files should be preserved:

- `prototype_rdm__pooled__response_window.png`
- `prototype_rdm__pooled__full_trajectory.png`

But their internal layout should become paired:

- left: pooled prototype neural RDM
- right: corresponding model RDM

This is an in-place content upgrade, not a file rename.

### Model Selection Rule

The right-hand pooled model must come from the main Stage 3 primary RSA results:

- `rsa_results.parquet`

For each `view`:

- select the top primary model already used by the main Stage 3 result
- do not derive a new pooled prototype-specific model ranking for this figure

This rule is important because the pooled prototype figure remains descriptive. It should borrow the already-approved primary model ranking rather than creating a second inferential path.

For the current `202603` batch, this likely means both pooled figures will use the same model on the right. That is acceptable and expected.

## Shared Label And Ordering Contract

Both new figure families should follow the same display rules already approved for Stage 3 RDM visualization:

- label priority:
  1. `sample_id`
  2. `stim_name`
  3. `stimulus`
- order source:
  derive the order from the neural RDM only
- reuse:
  apply that same order to the paired model RDM

The model RDM must not compute an independent clustering order.

This applies to:

- each `date x view` row in the new `per-date` paired figures
- each existing pooled paired prototype figure

For pooled prototype figures:

- keep the current neural order logic
- add the model panel in exactly that order

For per-date figures:

- cluster or preserve aligned order from the per-date neural RDM
- reuse the same order on the paired model panel

## Summary Metadata Contract

No new summary fields are required.

Existing summary handling should change like this:

- `prototype_figure_names` appends:
  - `prototype_rdm_comparison__per_date__response_window`
  - `prototype_rdm_comparison__per_date__full_trajectory`
- `figure_names` appends the same new figure names after the existing prototype figures
- `prototype_descriptive_outputs` continues to identify the pooled prototype RDM outputs only:
  - `prototype_rdm__pooled__response_window`
  - `prototype_rdm__pooled__full_trajectory`

This means:

- `per-date paired` figures are supplementary comparison figures
- `pooled paired` figures remain the descriptive pooled outputs already tracked by summary metadata

## Empty-State And Failure Handling

The writer should remain operationally conservative.

Rules:

- if a `date x view` neural prototype RDM exists but no eligible top model exists, write the figure with an empty-state model panel
- if a pooled prototype RDM exists but no eligible main Stage 3 top model exists, write the figure with an empty-state model panel
- if a whole `view` has no prototype data, write an empty-state figure rather than failing the whole Stage 3 run
- reruns with narrower prototype view sets must remove stale `prototype_rdm_comparison__per_date__*.png` files

The current prototype parquet cleanup behavior should remain intact and continue to remove stale prototype files on rerun.

## Testing Strategy

Add or update tests to cover:

- new `prototype_rdm_comparison__per_date__response_window.png` is written
- new `prototype_rdm_comparison__per_date__full_trajectory.png` is written
- pooled prototype figures still write under the existing filenames
- the paired model panel reuses the neural order rather than reclustering independently
- `prototype_figure_names` includes the new `per-date` paired figures
- `prototype_descriptive_outputs` remains limited to the pooled prototype RDM outputs
- reruns with a narrowed prototype view set remove stale `prototype_rdm_comparison__per_date__*.png`

Tests should stay focused on writer contract and pairing behavior, not on recomputing Stage 3 statistics.

## Acceptance Criteria

This design is complete when:

- Stage 3 keeps the current `prototype_rsa__per_date__<view>.png` summary plots
- Stage 3 adds the two `per-date` paired prototype RDM figures
- Stage 3 upgrades both pooled prototype RDM figures so the right side shows a paired model RDM
- both figure families use neural-driven ordering shared by the paired model panel
- labels continue to prefer `sample_id`
- summary metadata reports the new figure files without changing the current primary Stage 3 contract

