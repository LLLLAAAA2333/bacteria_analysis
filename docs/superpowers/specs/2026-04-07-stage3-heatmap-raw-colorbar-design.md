# Stage 3 Heatmap Raw-Value Colorbar Design

Date: 2026-04-07
Status: Drafted after user-approved design
Topic: Replace Stage 3 shared quantile heatmap colorbars with raw-value panel-specific colorbars that preserve diagonal masking and make internal structure more visible

## Goal

Stage 3 paired RDM figures currently render valid structural comparisons, but the present color scaling contract is a poor fit for interpretation.

The goal of this design is to replace the current `Shared off-diagonal quantile` display contract so that:

- each displayed heatmap keeps a colorbar whose ticks still correspond to raw RDM values
- diagonal cells remain masked and do not dominate the color range
- internal block structure is more visible than under a naive full-range linear scale
- Stage 3 RSA statistics remain unchanged

This is a visualization-only redesign. It must not change Stage 3 neural/model ranking, RSA scoring, permutation logic, or parquet table schemas.

## Current Context

Current Stage 3 paired-RDM writing lives in:

- `src/bacteria_analysis/rsa_outputs.py`

Current paired-RDM figure families include:

- `neural_vs_top_model_rdm__response_window.png`
- `neural_vs_top_model_rdm__full_trajectory.png`
- `aggregated_response_rdm_comparison__per_date__response_window.png`
- `aggregated_response_rdm_comparison__per_date__full_trajectory.png`
- `aggregated_response_rdm__pooled__response_window.png`
- `aggregated_response_rdm__pooled__full_trajectory.png`

Current visualization behavior:

- diagonal entries are masked
- the figure gathers all displayed off-diagonal values across the whole panel
- each displayed matrix is remapped into a shared within-figure quantile scale
- one shared colorbar is labeled `Shared off-diagonal quantile`

This solved one problem and created another:

- it makes structure easier to see
- but the colorbar no longer represents raw RDM values
- so readers cannot interpret the scale numerically

The user explicitly wants to keep diagonal masking, but wants each heatmap colorbar to reflect real values while still making structure visually clearer.

## User-Requested Changes

The requested display changes are:

1. keep the diagonal masked
2. stop using the current shared quantile colorbar contract
3. let different heatmaps use different colorbars
4. keep colorbars interpretable in raw-value units
5. adjust the color mapping so internal heatmap structure is more visible

The approved design direction is:

- per-panel raw-value colorbar
- per-panel off-diagonal robust clipping
- monotone non-linear color mapping rather than quantile remapping

## Options Considered

### Option 1: Keep Shared Quantile Colorbars

Pros:

- strongest cross-panel visual comparability
- brings out structure even when raw ranges differ

Cons:

- colorbar values no longer represent actual RDM values
- difficult to explain in figures or text
- feels artificial for a scientific heatmap

### Option 2: Raw-Value Per-Panel Colorbars With Off-Diagonal Robust Clipping And Power Normalization

Pros:

- colorbar remains interpretable in the original value scale
- structure becomes clearer because extreme off-diagonal values no longer consume the whole dynamic range
- non-linear color response can emphasize the middle range without discarding numeric meaning

Cons:

- colors are no longer directly comparable across different heatmaps
- implementation is more complex than a single shared colorbar

### Option 3: Raw-Value Per-Panel Colorbars With Only Linear Clipping

Pros:

- simpler than non-linear normalization
- preserves raw-value colorbar semantics

Cons:

- often still too flat when the informative structure is concentrated in the middle of the value range
- weaker visual improvement than the approved direction

## Chosen Approach

Use Option 2.

Each heatmap panel should:

- keep its own raw-value colorbar
- estimate its display range from off-diagonal values only
- clip display to a robust quantile interval
- apply a monotone `PowerNorm` so the middle range carries more visible contrast

This preserves the scientific meaning of the colorbar while making structure materially easier to see.

## Scope

This design covers:

- Stage 3 paired-RDM figure generation in `src/bacteria_analysis/rsa_outputs.py`
- colorbar semantics for paired Stage 3 heatmaps
- per-panel color scaling rules
- layout updates required to place one colorbar per heatmap without overlap
- tests for the new display contract

This design does not cover:

- RSA score computation
- permutation testing
- model ranking logic
- Stage 3 parquet schemas
- Stage 2 heatmaps
- `ranked_model_rsa.png`
- `leave_one_stimulus_out_robustness.png`
- `view_comparison_summary.png`

## Display Contract

### Diagonal Policy

All displayed RDM heatmaps should continue to mask the diagonal before display.

Reason:

- self-comparison cells are not informative for the neural-versus-model structure comparison
- diagonal zeros otherwise compress the useful dynamic range

The diagonal mask is a display-layer operation only. It must not alter any Stage 3 RSA statistics.

### Colorbar Policy

Each displayed heatmap panel should have its own dedicated colorbar.

Rules:

- the colorbar must be tied to that panel's raw numeric values
- the colorbar label should describe raw dissimilarity, not quantiles
- the current label `Shared off-diagonal quantile` should be removed

Recommended label:

- `RDM dissimilarity`

If later desired, the implementation may use more specific labels such as:

- `Neural RDM dissimilarity`
- `Model RDM dissimilarity`

but the default should stay compact and neutral.

### Display Range Policy

For each heatmap panel:

1. collect finite off-diagonal values only
2. compute a robust display interval from that panel's own values
3. render the heatmap in raw units within that interval

Recommended default interval:

- `vmin = off_diagonal q05`
- `vmax = off_diagonal q95`

Fallback rules:

- if there are too few finite off-diagonal values for stable quantiles, fall back to finite off-diagonal min/max
- if `vmin == vmax`, widen conservatively around that value or fall back to a simple linear min/max rule
- if the panel has no finite off-diagonal values, render the existing empty-state message instead of failing

### Normalization Policy

Within each panel's raw-value display interval, apply a monotone power normalization:

- `PowerNorm(gamma=0.7)`

Reason:

- this improves contrast in the middle of the distribution
- the mapping remains monotone
- the colorbar still reports raw values
- it is easier to explain than histogram equalization or quantile colorbars

The normalization must be used for display only. It must not feed back into any RDM tables or RSA statistics.

### Colormap Policy

Keep a monotone perceptually ordered colormap:

- `viridis`

Do not switch to a diverging colormap.

Reason:

- Stage 3 RDMs encode non-negative dissimilarity
- there is no scientifically meaningful central zero-crossing that would justify a diverging map

## Figure Layout Contract

The previous shared-colorbar layout was already replaced once to stop overlap with the heatmaps.

Under the new contract, each heatmap panel needs its own adjacent colorbar axis.

Requirements:

- each heatmap panel must reserve space for its own colorbar
- colorbars must not overlap any heatmap pixels
- layout must remain readable for:
  - two-row paired figures
  - multi-row per-date paired figures

Recommended implementation direction:

- use explicit `GridSpec`-style layout rather than `tight_layout()` heuristics
- pair each heatmap axis with a narrow neighboring colorbar axis
- keep the heatmap-to-colorbar pairing visually obvious

## Figure Families Affected

The new color policy should apply consistently to:

- `neural_vs_top_model_rdm__<view>.png`
- `aggregated_response_rdm_comparison__per_date__<view>.png`
- `aggregated_response_rdm__pooled__<view>.png`

For repeated appearances of the same underlying matrix under different orderings:

- each displayed panel still keeps its own colorbar under this contract
- the raw-value scale may end up identical between repeated panels, which is acceptable

## Interpretation Boundary

After this redesign:

- colors remain interpretable in raw-value space
- panel-to-panel colors are not guaranteed to be numerically comparable at a glance
- structural readability within each heatmap is intentionally prioritized

This is the explicit tradeoff chosen by the user.

The figures should therefore be interpreted as:

- strong within-panel structure displays
- honest raw-value colorbars
- weaker direct cross-panel color comparability than under the previous shared-quantile mode

## Implementation Boundaries

Primary implementation file:

- `src/bacteria_analysis/rsa_outputs.py`

Expected change areas in that file:

- replace shared figure-level quantile normalization helpers
- add per-panel off-diagonal clipping helpers
- add per-panel `PowerNorm` construction
- update panel rendering so each heatmap receives its own colorbar axis
- remove `Shared off-diagonal quantile` labeling from Stage 3 paired heatmaps

Expected supporting test changes:

- `tests/test_rsa.py`
- `tests/test_rsa_cli_smoke.py` only if figure contract metadata changes

No changes should be required in:

- `src/bacteria_analysis/rsa.py`
- `src/bacteria_analysis/rsa_aggregated_responses.py`
- `src/bacteria_analysis/model_space.py`
- `scripts/run_rsa.py`

## Error Handling And Fallbacks

The writer should remain operationally conservative.

Rules:

- if a panel has no finite off-diagonal values, render the existing empty-state panel
- if robust clipping cannot be estimated safely, fall back to raw finite min/max
- if per-panel colorbar creation fails for one panel, the writer should prefer a degraded but readable figure over crashing the whole Stage 3 pipeline
- display-layer failures must not change or invalidate the Stage 3 statistical tables

## Testing Strategy

Add or update tests to cover:

- diagonal masking remains in place
- the old shared-quantile colorbar label is no longer used
- paired-RDM figures no longer depend on shared quantile normalization helpers
- panel layouts reserve non-overlapping colorbar axes
- per-panel normalization is derived from that panel's own off-diagonal values
- raw-value display fallback works when off-diagonal values are degenerate or sparse

Tests should stay focused on display contract and layout behavior, not on recomputing RSA statistics.

## Acceptance Criteria

This redesign is complete when:

- Stage 3 paired heatmaps still mask the diagonal
- each displayed heatmap panel has its own colorbar
- those colorbars correspond to raw RDM values rather than quantiles
- the display range is estimated from panel-specific off-diagonal values
- `PowerNorm(gamma=0.7)` is used to improve visible internal structure
- no heatmap colorbar overlaps the heatmap itself
- existing Stage 3 RSA statistics and rankings remain unchanged
- tests pass

## Recommended Follow-Up

After this spec is approved, the implementation plan should stay small and sequential:

1. replace shared quantile scaling helpers with panel-local display-range helpers
2. refactor paired-RDM layout to attach one colorbar axis per panel
3. update tests for the new display contract
4. rerun `results/202603/stage3_rsa` and visually inspect all paired heatmap families
