# Stage 3 Heatmap Raw-Value Colorbar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current shared quantile colorbar display in Stage 3 paired heatmaps with per-panel raw-value colorbars that keep the diagonal masked and make internal structure easier to see without changing any RSA statistics.

**Architecture:** Keep the change local to `src/bacteria_analysis/rsa_outputs.py`. Remove the current shared quantile normalization path, add panel-local off-diagonal display-range helpers plus `PowerNorm(gamma=0.7)`, and refactor the paired-RDM layout so each heatmap gets its own adjacent colorbar axis. Refresh display-focused tests in `tests/test_rsa.py` and rerun the real `202603` Stage 3 outputs for visual verification.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib (`Normalize`, `PowerNorm`, colorbar axes), pytest, standard library (`pathlib`)

---

## Overview

This plan implements the approved spec:

- `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-04-07-stage3-heatmap-raw-colorbar-design.md`

The plan intentionally does not change:

- Stage 3 RSA scoring in `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
- aggregated-response generation in `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_aggregated_responses.py`
- CLI arguments in `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
- Stage 3 parquet schemas or summary-table semantics

The plan focuses on one display-layer redesign:

> keep diagonal masking, restore raw-value colorbars, and improve within-panel structural visibility through robust clipping plus monotone non-linear normalization.

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
  Responsibility: remove shared-quantile heatmap rendering, add panel-local display-range and normalization helpers, attach one colorbar to each heatmap panel, and keep figure layout readable without overlap.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
  Responsibility: lock in the new display contract, including raw-value colorbar labeling, panel-local normalization behavior, per-panel colorbar axes, and fallback behavior for sparse or degenerate off-diagonal values.

No new production modules are required. Keep the implementation local to the existing Stage 3 output writer unless a helper split becomes clearly necessary while coding.

## Output Contract To Implement

Affected figure families:

- `neural_vs_top_model_rdm__response_window.png`
- `neural_vs_top_model_rdm__full_trajectory.png`
- `aggregated_response_rdm_comparison__per_date__response_window.png`
- `aggregated_response_rdm_comparison__per_date__full_trajectory.png`
- `aggregated_response_rdm__pooled__response_window.png`
- `aggregated_response_rdm__pooled__full_trajectory.png`

For every displayed heatmap panel:

- keep the diagonal masked
- use panel-local off-diagonal values to derive the display range
- use raw-value colorbars rather than a shared quantile colorbar
- apply `PowerNorm(gamma=0.7)` inside the chosen display range
- keep `viridis`

Explicitly remove:

- the current `Shared off-diagonal quantile` colorbar contract
- the current within-figure shared quantile normalization path

## Task 1: Lock The New Display Contract In Tests

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`

- [ ] **Step 1: Write failing tests for panel-local raw-value normalization inputs**

```python
def test_prepare_rdm_display_parameters_uses_off_diagonal_quantiles():
    frame = pd.DataFrame(
        [[np.nan, 0.1, 0.2], [0.1, np.nan, 0.9], [0.2, 0.9, np.nan]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    params = rsa_outputs._prepare_rdm_display_parameters(
        frame,
        lower_quantile=0.25,
        upper_quantile=0.75,
    )

    assert pytest.approx(params.vmin) == 0.125
    assert pytest.approx(params.vmax) == 0.725
```

- [ ] **Step 2: Write a failing test that the old shared-quantile label is gone**

```python
def test_render_prepared_rdm_panels_uses_raw_value_colorbar_label(tmp_path):
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=1, figsize=(8.0, 4.0))
    frame = pd.DataFrame(
        [[np.nan, 0.2], [0.2, np.nan]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    rsa_outputs._render_prepared_rdm_panels(
        figure,
        axes,
        [(0, 0, frame, "left", "missing"), (0, 1, frame, "right", "missing")],
        colorbar_axes=colorbar_axes,
    )
    assert all(axis.get_ylabel() == "RDM dissimilarity" for axis in colorbar_axes.ravel())
```

- [ ] **Step 3: Write a failing layout test for one-colorbar-per-panel**

```python
def test_create_rdm_panel_figure_allocates_one_colorbar_axis_per_panel():
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=2, figsize=(10.0, 7.0))
    try:
        assert axes.shape == (2, 2)
        assert colorbar_axes.shape == (2, 2)
    finally:
        plt.close(figure)
```

- [ ] **Step 4: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "display_parameters_uses_off_diagonal_quantiles or raw_value_colorbar_label or one_colorbar_axis_per_panel" -v`

Expected: FAIL because the current writer still uses shared quantile normalization, a single shared colorbar label, and a figure-level colorbar slot rather than one per panel.

- [ ] **Step 5: Implement the minimal scaffolding required to make the tests meaningful**

```python
@dataclass(frozen=True)
class RdmDisplayParameters:
    vmin: float
    vmax: float
    norm: matplotlib.colors.Normalize
```

Implementation requirements:

- add a small immutable container for per-panel display parameters
- avoid changing any statistical helper signatures
- keep the helper local to `rsa_outputs.py`

- [ ] **Step 6: Run the targeted tests again to verify progress**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "display_parameters_uses_off_diagonal_quantiles or raw_value_colorbar_label or one_colorbar_axis_per_panel" -v`

Expected: still FAIL or partially PASS, but now against concrete display helpers instead of missing-symbol errors.

- [ ] **Step 7: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "test: lock stage3 raw-value heatmap contract"
```

## Task 2: Replace Shared Quantile Scaling With Panel-Local Display Helpers

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`

- [ ] **Step 1: Write failing tests for robust clipping and degenerate fallback**

```python
def test_prepare_rdm_display_parameters_falls_back_to_finite_min_max_when_quantiles_are_unstable():
    frame = pd.DataFrame(
        [[np.nan, 0.4], [0.4, np.nan]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    params = rsa_outputs._prepare_rdm_display_parameters(frame)

    assert params.vmin < 0.4
    assert params.vmax > 0.4


def test_prepare_rdm_display_parameters_returns_none_when_no_finite_off_diagonal_values():
    frame = pd.DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    assert rsa_outputs._prepare_rdm_display_parameters(frame) is None
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "quantiles_are_unstable or no_finite_off_diagonal_values" -v`

Expected: FAIL because there is no panel-local display-parameter helper with these fallback rules.

- [ ] **Step 3: Implement `_prepare_rdm_display_parameters`**

```python
def _prepare_rdm_display_parameters(
    heatmap_frame: pd.DataFrame,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    gamma: float = 0.7,
) -> RdmDisplayParameters | None:
    off_diagonal = _finite_off_diagonal_values(heatmap_frame)
    if off_diagonal.size == 0:
        return None

    vmin = float(np.quantile(off_diagonal, lower_quantile))
    vmax = float(np.quantile(off_diagonal, upper_quantile))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = float(np.min(off_diagonal))
        vmax = float(np.max(off_diagonal))
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if vmin == vmax:
        raw_min = float(np.min(off_diagonal))
        raw_max = float(np.max(off_diagonal))
        if raw_min < raw_max:
            vmin, vmax = raw_min, raw_max
        else:
            span = max(abs(vmin) * 0.05, 1e-6)
            vmin -= span
            vmax += span
    norm = matplotlib.colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
    return RdmDisplayParameters(vmin=vmin, vmax=vmax, norm=norm)
```

Implementation requirements:

- collect off-diagonal finite values only
- default to `q05/q95`
- use raw-value units for `vmin` and `vmax`
- construct `PowerNorm(gamma=0.7, clip=True)`
- when `vmin == vmax`, first fall back to raw finite min/max if they differ, otherwise widen conservatively around the constant value before constructing `PowerNorm`
- handle sparse, degenerate, or non-finite cases conservatively

- [ ] **Step 4: Remove the shared quantile normalization path**

```python
def _normalize_display_frames(...):
    raise NotImplementedError
```

Implementation requirements:

- delete `_normalize_display_frames` and `_shared_quantile_scale`
- update call sites so no Stage 3 figure depends on cross-panel quantile remapping
- keep diagonal masking behavior unchanged

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "display_parameters or quantiles_are_unstable or no_finite_off_diagonal_values" -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: add panel-local stage3 heatmap display scaling"
```

## Task 3: Refactor Heatmap Rendering To Use One Raw-Value Colorbar Per Panel

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`

- [ ] **Step 1: Replace the legacy single-colorbar layout assertion with paired-colorbar layout tests**

```python
def test_create_rdm_panel_figure_pairs_each_heatmap_with_a_neighbor_colorbar():
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=1, figsize=(10.0, 4.0))
    try:
        assert colorbar_axes[0, 0].get_position().x0 > axes[0, 0].get_position().x1
        assert colorbar_axes[0, 1].get_position().x0 > axes[0, 1].get_position().x1
    finally:
        plt.close(figure)
```

Implementation requirements:

- retire or rewrite the existing legacy test `test_create_rdm_panel_figure_reserves_separate_colorbar_column`
- the updated test contract must assert the new `(nrows, 2)` colorbar-axis grid rather than the old single shared colorbar column

- [ ] **Step 2: Write a failing test that each rendered panel gets its own colorbar**

```python
def test_render_prepared_rdm_panels_writes_one_colorbar_per_heatmap():
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=1, figsize=(10.0, 4.0))
    frame = pd.DataFrame(
        [[np.nan, 0.2, 0.7], [0.2, np.nan, 0.8], [0.7, 0.8, np.nan]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    rsa_outputs._render_prepared_rdm_panels(
        figure,
        axes,
        [(0, 0, frame, "left", "missing"), (0, 1, frame, "right", "missing")],
        colorbar_axes=colorbar_axes,
    )
    assert colorbar_axes[0, 0].has_data()
    assert colorbar_axes[0, 1].has_data()
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "neighbor_colorbar or one_colorbar_per_heatmap" -v`

Expected: FAIL because the current figure factory allocates one shared colorbar region and the renderer only draws a single shared colorbar.

- [ ] **Step 4: Refactor `_create_rdm_panel_figure`**

```python
def _create_rdm_panel_figure(
    *,
    nrows: int,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    grid = figure.add_gridspec(
        nrows=nrows,
        ncols=4,
        width_ratios=(1.0, 0.05, 1.0, 0.05),
        ...
    )
```

Implementation requirements:

- each row gets `heatmap, colorbar, heatmap, colorbar`
- return `axes` with shape `(nrows, 2)`
- return `colorbar_axes` with shape `(nrows, 2)`
- keep enough whitespace to avoid label overlap

- [ ] **Step 5: Refactor `_render_prepared_rdm_panels` and `_render_prepared_rdm_axis`**

```python
params = _prepare_rdm_display_parameters(frame)
image = axis.imshow(values, cmap=cmap, norm=params.norm)
try:
    figure.colorbar(image, cax=colorbar_axes[row_index, col_index], label="RDM dissimilarity")
except Exception:
    colorbar_axes[row_index, col_index].set_visible(False)
```

Implementation requirements:

- compute display parameters per panel
- call `imshow` with `norm=params.norm` instead of shared `vmin=0.0, vmax=1.0`
- create one colorbar per heatmap axis
- if `figure.colorbar(...)` fails for a specific panel, hide that panel's colorbar axis and keep the heatmap panel rendered so the writer degrades gracefully instead of aborting the whole figure
- hide any colorbar axis whose paired panel is empty
- preserve empty-state text behavior

- [ ] **Step 6: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "raw_value_colorbar_label or neighbor_colorbar or one_colorbar_per_heatmap" -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: give each stage3 heatmap its own raw-value colorbar"
```

## Task 4: Verify End-To-End Figures And Real-Data Outputs

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py` only if verification exposes a small display bug
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py` only if verification requires an additional regression test

- [ ] **Step 1: Run the focused RSA test suite**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -q`

Expected: PASS.

- [ ] **Step 2: Run the full repository test suite**

Run: `pixi run pytest -q`

Expected: PASS.

- [ ] **Step 3: Refresh the live `202603` Stage 3 outputs**

Run:

```powershell
$env:PYTHONPATH='src'; @'
from pathlib import Path
from bacteria_analysis.model_space import resolve_model_inputs
from bacteria_analysis.rsa import run_biochemical_rsa
from bacteria_analysis.rsa_outputs import write_rsa_outputs
from bacteria_analysis.rsa_aggregated_responses import load_aggregated_response_context_inputs

resolved_inputs = resolve_model_inputs(Path('data/model_space_202603'), Path('data/matrix.xlsx'))
aggregated_inputs = load_aggregated_response_context_inputs(
    Path('data/20260313_preprocess'),
    view_names=('response_window', 'full_trajectory'),
)
core_outputs = run_biochemical_rsa(
    resolved_inputs,
    aggregated_response_inputs=aggregated_inputs,
    response_aggregation='mean',
    permutations=1000,
    seed=0,
)
write_rsa_outputs(core_outputs, Path('results/202603/stage3_rsa'))
'@ | pixi run python -
```

Expected:

- exit code `0`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__response_window.png` is refreshed
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm_comparison__per_date__response_window.png` is refreshed
- the colorbar label now reports raw-value dissimilarity rather than quantiles

- [ ] **Step 4: Visually inspect all affected figure families**

Inspect:

- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__response_window.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__full_trajectory.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm_comparison__per_date__response_window.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm_comparison__per_date__full_trajectory.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm__pooled__response_window.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm__pooled__full_trajectory.png`

Confirm visually:

- diagonal remains masked
- each heatmap has its own adjacent colorbar
- colorbar labels describe raw dissimilarity values
- panel-internal structure is clearer than under the old flat scaling
- no colorbar overlaps the heatmap pixels

- [ ] **Step 5: Add one regression test if visual verification reveals a missed edge case**

Example:

```python
def test_render_prepared_rdm_panels_hides_colorbar_for_empty_panel():
    ...
```

Only add this if the real-data verification reveals a concrete missed case.

- [ ] **Step 6: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: refresh stage3 raw-value heatmap colorbars"
```

## Final Verification

- [ ] Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -q`
  Expected: PASS.

- [ ] Run: `pixi run pytest -q`
  Expected: PASS.

- [ ] Inspect:
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__response_window.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__full_trajectory.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm_comparison__per_date__response_window.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm_comparison__per_date__full_trajectory.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm__pooled__response_window.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/aggregated_response_rdm__pooled__full_trajectory.png`

- [ ] Confirm:
  - the old `Shared off-diagonal quantile` label is gone
  - heatmap colors now represent panel-local raw dissimilarity values
  - diagonal masking remains intact
  - panel-local structure is easier to read
  - RSA tables and rankings remain unchanged
