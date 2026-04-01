# Stage 3 RDM Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default Stage 3 neural-vs-model RDM panel with per-view figures that use `sample_id` labels and a neural-driven clustered order shared by the neural and model heatmaps.

**Architecture:** Keep Stage 3 statistics untouched and confine the redesign to the output writer in `src/bacteria_analysis/rsa_outputs.py`. Add a small display-label and ordering helper layer, replace the single mixed-view figure with per-view outputs, and update tests plus run-summary metadata to match the new output contract.

**Tech Stack:** Python 3.11, pandas, numpy, scipy hierarchical clustering, matplotlib, pytest, standard library (`pathlib`, `json`)

---

## Overview

This plan implements the approved spec:

- `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-04-01-stage3-rdm-visualization-design.md`

The plan intentionally does not change:

- Stage 3 RSA statistics in `src/bacteria_analysis/rsa.py`
- model-space resolution in `src/bacteria_analysis/model_space.py`
- the CLI surface in `scripts/run_rsa.py`
- Stage 2 outputs

The plan focuses on one output-layer redesign:

> make Stage 3 default figures easier to read by splitting views, shortening labels, and using one neural-driven clustered order per view.

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
  Responsibility: add display-label resolution, derive a neural heatmap order, render per-view neural-vs-top-model figures, and update run-summary figure metadata.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
  Responsibility: add unit coverage for the new per-view figure contract, label fallback behavior, and shared ordering behavior.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`
  Responsibility: update end-to-end smoke expectations for the new Stage 3 figure files.

No new production modules are required. Keep the implementation local to the existing Stage 3 output writer unless a helper split is clearly needed during implementation.

## Output Contract To Implement

Replace the legacy default panel file:

- `neural_vs_top_model_rdm_panel.png`

With per-view default files:

- `neural_vs_top_model_rdm__response_window.png`
- `neural_vs_top_model_rdm__full_trajectory.png`

For each per-view file:

- left subplot: pooled neural RDM for that view
- right subplot: top primary model RDM for that same view
- display labels: prefer `sample_id`, then `stim_name`, then `stimulus`
- display order: cluster the neural RDM once, then apply the same order to both subplots

## Task 1: Lock The New Figure Contract In Unit Tests

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`

- [ ] **Step 1: Write failing tests for the new figure file names**

```python
def test_write_stage3_outputs_writes_per_view_neural_model_figures(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")

    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert not (written["figures_dir"] / "neural_vs_top_model_rdm_panel.png").exists()
```

- [ ] **Step 2: Write a failing test for `run_summary.json` figure names**

```python
def test_write_stage3_outputs_reports_per_view_figure_names(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert "neural_vs_top_model_rdm__response_window" in summary["figure_names"]
    assert "neural_vs_top_model_rdm__full_trajectory" in summary["figure_names"]
    assert "neural_vs_top_model_rdm_panel" not in summary["figure_names"]
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "per_view_neural_model_figures or reports_per_view_figure_names" -v`

Expected: FAIL because the current writer still emits `neural_vs_top_model_rdm_panel.png` and the old figure metadata.

- [ ] **Step 4: Implement the minimal figure-name contract change**

```python
REQUIRED_FIGURES = (
    "ranked_primary_model_rsa",
    "leave_one_stimulus_out_robustness",
    "view_comparison_summary",
)


def _build_neural_vs_model_figure_names(view_names: list[str]) -> list[str]:
    return [f"neural_vs_top_model_rdm__{view_name}" for view_name in view_names]
```

Implementation requirements:

- remove the legacy panel from the default Stage 3 figure list
- write figure names in a deterministic view order
- keep all other figure names unchanged

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "per_view_neural_model_figures or reports_per_view_figure_names" -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: split stage3 rdm figures by view"
```

## Task 2: Add Display-Label Resolution With Safe Fallbacks

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`

- [ ] **Step 1: Write failing tests for label resolution priority**

```python
def test_stage3_heatmap_prefers_sample_id_labels():
    mapping = pd.DataFrame.from_records(
        [
            {"stimulus": "b1", "stim_name": "A001 stationary", "sample_id": "A001"},
            {"stimulus": "b2", "stim_name": "A002 stationary", "sample_id": "A002"},
        ]
    )

    labels = _build_stage3_display_labels(["b1", "b2"], mapping)
    assert labels == ["A001", "A002"]


def test_stage3_heatmap_falls_back_when_sample_ids_duplicate():
    mapping = pd.DataFrame.from_records(
        [
            {"stimulus": "b1", "stim_name": "Stimulus 1", "sample_id": "dup"},
            {"stimulus": "b2", "stim_name": "Stimulus 2", "sample_id": "dup"},
        ]
    )

    labels = _build_stage3_display_labels(["b1", "b2"], mapping)
    assert labels == ["Stimulus 1", "Stimulus 2"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "prefers_sample_id_labels or falls_back_when_sample_ids_duplicate" -v`

Expected: FAIL because the helper does not exist and the current heatmap code uses raw matrix labels.

- [ ] **Step 3: Implement the minimal label resolver**

```python
def _build_stage3_display_labels(stimulus_order: list[str], stimulus_sample_map: pd.DataFrame | None) -> list[str]:
    for column in ("sample_id", "stim_name", "stimulus"):
        labels = _candidate_labels_for_column(stimulus_order, stimulus_sample_map, column)
        if _labels_are_unique_and_non_empty(labels):
            return labels
    return [str(label) for label in stimulus_order]
```

Implementation requirements:

- read labels from `stimulus_sample_map`
- preserve the incoming stimulus order
- require one-to-one non-empty display labels before accepting a candidate layer
- fall back in the exact order `sample_id -> stim_name -> stimulus`
- keep the helper local to `rsa_outputs.py`

- [ ] **Step 4: Thread the label resolver into heatmap preparation**

```python
heatmap_frame.index = pd.Index(display_labels)
heatmap_frame.columns = pd.Index(display_labels)
```

Implementation requirements:

- apply the same labels to rows and columns
- preserve numeric matrix values
- avoid changing the underlying parquet tables

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "prefers_sample_id_labels or falls_back_when_sample_ids_duplicate" -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: add stage3 rdm display label fallback"
```

## Task 3: Apply One Neural-Driven Cluster Order To Both Heatmaps

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`

- [ ] **Step 1: Write failing tests for shared neural ordering**

```python
def test_stage3_model_heatmap_uses_neural_cluster_order():
    neural = pd.DataFrame.from_records(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 1.0, "s3": 5.0, "s4": 5.2},
            {"stimulus_row": "s2", "s1": 1.0, "s2": 0.0, "s3": 4.8, "s4": 5.1},
            {"stimulus_row": "s3", "s1": 5.0, "s2": 4.8, "s3": 0.0, "s4": 0.9},
            {"stimulus_row": "s4", "s1": 5.2, "s2": 5.1, "s3": 0.9, "s4": 0.0},
        ]
    )
    model = pd.DataFrame.from_records(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 8.0, "s3": 1.0, "s4": 2.0},
            {"stimulus_row": "s2", "s1": 8.0, "s2": 0.0, "s3": 3.0, "s4": 4.0},
            {"stimulus_row": "s3", "s1": 1.0, "s2": 3.0, "s3": 0.0, "s4": 7.0},
            {"stimulus_row": "s4", "s1": 2.0, "s2": 4.0, "s3": 7.0, "s4": 0.0},
        ]
    )

    neural_frame, order = _prepare_stage3_heatmap_frame(neural)
    model_frame, reused_order = _prepare_stage3_heatmap_frame(model, order_labels=order)
    assert list(model_frame.index) == list(neural_frame.index)
    assert reused_order == order
```

- [ ] **Step 2: Add a failing fallback test for non-clusterable matrices**

```python
def test_stage3_heatmap_falls_back_to_aligned_order_when_clustering_is_invalid():
    matrix = pd.DataFrame.from_records(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": np.nan},
            {"stimulus_row": "s2", "s1": np.nan, "s2": 0.0},
        ]
    )

    heatmap_frame, order = _prepare_stage3_heatmap_frame(matrix)
    assert list(heatmap_frame.index) == ["s1", "s2"]
    assert order == ["s1", "s2"]
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "uses_neural_cluster_order or falls_back_to_aligned_order_when_clustering_is_invalid" -v`

Expected: FAIL because the current writer does not expose shared-order heatmap preparation.

- [ ] **Step 4: Implement the minimal shared-order helper**

```python
def _prepare_stage3_heatmap_frame(
    matrix_frame: pd.DataFrame,
    *,
    stimulus_sample_map: pd.DataFrame | None = None,
    order_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    heatmap_frame = _coerce_square_numeric_matrix(matrix_frame)
    if order_labels is None:
        order_labels = _cluster_order_or_identity(heatmap_frame)
    heatmap_frame = heatmap_frame.loc[order_labels, order_labels]
    display_labels = _build_stage3_display_labels(order_labels, stimulus_sample_map)
    heatmap_frame.index = pd.Index(display_labels)
    heatmap_frame.columns = pd.Index(display_labels)
    return heatmap_frame, order_labels
```

Implementation requirements:

- derive the order from the neural matrix only
- reuse the same order for the paired model matrix
- if clustering fails, keep the aligned original order
- keep ordering logic deterministic for the same matrix input

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "uses_neural_cluster_order or falls_back_to_aligned_order_when_clustering_is_invalid" -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: align stage3 model heatmaps to neural clustering"
```

## Task 4: Replace The Legacy Panel Writer And Refresh End-To-End Expectations

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Write a failing CLI smoke update for the new figure files**

```python
expected_paths = [
    stage3_root / "figures" / "ranked_primary_model_rsa.png",
    stage3_root / "figures" / "neural_vs_top_model_rdm__response_window.png",
    stage3_root / "figures" / "neural_vs_top_model_rdm__full_trajectory.png",
    stage3_root / "run_summary.json",
    stage3_root / "run_summary.md",
]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "write_stage3_outputs" -v`

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -k "cli_runs_and_writes_stage3_outputs" -v`

Expected: FAIL because the writer still routes through the old panel helper or does not yet register all per-view paths.

- [ ] **Step 3: Replace the panel writer with a per-view figure writer**

```python
def _plot_neural_vs_top_model_rdms_by_view(
    core_outputs: dict[str, pd.DataFrame],
    top_primary_models: dict[str, str],
    figures_dir: Path,
    stimulus_sample_map: pd.DataFrame | None,
) -> dict[str, Path]:
    written = {}
    view_names = _ordered_views(core_outputs["rsa_results"], core_outputs["rsa_view_comparison"])
    for view_name in view_names:
        path = figures_dir / f"neural_vs_top_model_rdm__{view_name}.png"
        written[f"neural_vs_top_model_rdm__{view_name}"] = _plot_single_view_neural_vs_model(
            core_outputs,
            view_name=view_name,
            top_model_id=top_primary_models.get(view_name),
            path=path,
            stimulus_sample_map=stimulus_sample_map,
        )
    return written
```

Implementation requirements:

- remove the legacy panel helper from the write path
- write one figure per view in stable order
- keep the per-view layout to two subplots only
- use the same `stimulus_sample_map` input for both display labels and pairing logic
- update `written` and `run_summary` so they track the actual per-view files

- [ ] **Step 4: Run the updated test suites**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -v`

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -v`

Expected: PASS.

- [ ] **Step 5: Re-run Stage 3 on the real `202603` output root to verify the new visuals**

Run:

```powershell
pixi run python H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py --stage2-root H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage2_geometry --matrix H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx --model-input-root H:/Process_temporary/WJH/bacteria_analysis/data/model_space_202603 --output-root H:/Process_temporary/WJH/bacteria_analysis/results/202603 --permutations 1000
```

Expected:

- exit code `0`
- `results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__response_window.png` exists
- `results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__full_trajectory.png` exists
- `results/202603/stage3_rsa/figures/neural_vs_top_model_rdm_panel.png` is no longer the default artifact
- `rsa_results.parquet` remains unchanged at the statistics level

- [ ] **Step 6: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "feat: refresh stage3 rdm visual outputs"
```

## Final Verification

- [ ] Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -v`
  Expected: PASS.

- [ ] Inspect:
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__response_window.png`
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/neural_vs_top_model_rdm__full_trajectory.png`
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/run_summary.json`

- [ ] Confirm visually:
  - labels render as `Axxx`
  - the neural and model heatmaps in the same view use the same order
  - `response_window` and `full_trajectory` are now clearly separated
