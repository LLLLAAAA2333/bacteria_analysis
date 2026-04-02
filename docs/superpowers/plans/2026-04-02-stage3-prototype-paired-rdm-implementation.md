# Stage 3 Prototype Paired RDM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Stage 3 prototype supplementary visuals so `per-date` prototype results get dedicated neural-vs-top-model paired RDM figures and the existing pooled prototype RDM figures gain a paired model panel without changing any Stage 3 statistics or table schemas.

**Architecture:** Keep the change entirely inside the Stage 3 output writer in `src/bacteria_analysis/rsa_outputs.py`. Reuse the existing heatmap-label and neural-order helpers, add one small figure-name helper plus one top-model selection helper for `date x view`, then route both `per-date` and `pooled` prototype figures through paired neural/model renderers that share the neural-derived order.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, pytest, standard library (`pathlib`, `json`)

---

## Overview

This plan implements the approved spec:

- `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/docs/superpowers/specs/2026-04-02-stage3-prototype-paired-rdm-design.md`

The plan intentionally does not change:

- Stage 3 RSA scoring in `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/src/bacteria_analysis/rsa.py`
- prototype supplementary table generation in `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/src/bacteria_analysis/rsa.py`
- CLI flags in `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/scripts/run_rsa.py`
- prototype parquet schemas already written by Stage 3

The plan focuses on one output-layer extension:

> keep the existing prototype summary plots, add `per-date` paired neural/model RDM figures, and upgrade the pooled prototype figures to the same paired layout.

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/src/bacteria_analysis/rsa_outputs.py`
  Responsibility: define new prototype figure names, select top prototype models per `date x view`, render paired `per-date` and pooled prototype RDM figures, update stale-figure cleanup, and refresh summary metadata.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py`
  Responsibility: lock the writer contract for new figure files, summary metadata, shared ordering, empty-state handling, and rerun cleanup.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa_cli_smoke.py`
  Responsibility: keep the end-to-end Stage 3 smoke expectations aligned with the new prototype figure contract.

No new production modules are needed. Keep the implementation local to the existing output writer unless a helper split becomes necessary during coding.

## Output Contract To Implement

Keep these existing summary plots unchanged:

- `prototype_rsa__per_date__response_window.png`
- `prototype_rsa__per_date__full_trajectory.png`

Add these new `per-date` paired comparison figures:

- `prototype_rdm_comparison__per_date__response_window.png`
- `prototype_rdm_comparison__per_date__full_trajectory.png`

Keep these pooled filenames, but upgrade their content to paired neural/model layout:

- `prototype_rdm__pooled__response_window.png`
- `prototype_rdm__pooled__full_trajectory.png`

Pairing rules:

- `per-date` right-hand model panel comes from `prototype_rsa_results__per_date` top eligible model for the same `date x view`
- pooled right-hand model panel comes from main `rsa_results` top primary model for the same `view`
- model heatmaps must reuse the neural heatmap order
- labels continue to prefer `sample_id`, then `stim_name`, then `stimulus`

## Task 1: Lock The New Prototype Figure Contract In Tests

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/src/bacteria_analysis/rsa_outputs.py`

- [ ] **Step 1: Write failing tests for the new figure file set**

```python
def test_write_stage3_outputs_writes_prototype_paired_rdm_figures(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")

    assert (written["figures_dir"] / "prototype_rsa__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rsa__per_date__full_trajectory.png").exists()
    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__full_trajectory.png").exists()
    assert (written["figures_dir"] / "prototype_rdm__pooled__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm__pooled__full_trajectory.png").exists()
```

- [ ] **Step 2: Write a failing test for run-summary prototype figure names**

```python
def test_write_stage3_outputs_reports_prototype_paired_figure_names(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["prototype_figure_names"] == [
        "prototype_rsa__per_date__response_window",
        "prototype_rsa__per_date__full_trajectory",
        "prototype_rdm_comparison__per_date__response_window",
        "prototype_rdm_comparison__per_date__full_trajectory",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
    assert summary["prototype_descriptive_outputs"] == [
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py -k "prototype_paired_rdm_figures or prototype_paired_figure_names" -v`

Expected: FAIL because the writer currently only emits the summary plots plus pooled prototype figures and does not know the new `prototype_rdm_comparison__per_date__*` files.

- [ ] **Step 4: Implement the minimal figure-name helper and summary registration**

```python
def _build_prototype_rdm_comparison_figure_names(view_names: list[str]) -> list[str]:
    return [f"prototype_rdm_comparison__per_date__{view_name}" for view_name in view_names]


prototype_figure_names = [
    *_build_prototype_rsa_figure_names(prototype_rsa_views),
    *_build_prototype_rdm_comparison_figure_names(prototype_rsa_views),
    *_build_prototype_rdm_figure_names(prototype_rdm_views),
]
```

Implementation requirements:

- keep existing `prototype_rsa__per_date__*` figure names
- append new `per-date` paired figure names in canonical view order
- keep `prototype_descriptive_outputs` limited to pooled prototype figure names
- do not change primary Stage 3 `figure_names` ordering except for adding the new prototype figure names

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py -k "prototype_paired_rdm_figures or prototype_paired_figure_names" -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git -C H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement add src/bacteria_analysis/rsa_outputs.py tests/test_rsa.py
git -C H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement commit -m "feat: add prototype paired figure contract"
```

## Task 2: Add Per-Date Top-Model Selection And Paired Figure Rendering

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py`

- [ ] **Step 1: Write failing tests for `date x view` top-model selection**

```python
def test_build_top_prototype_models_by_date_and_view_uses_is_top_model():
    prototype = pd.DataFrame.from_records(
        [
            {"date": "20260311", "view_name": "response_window", "model_id": "global_profile", "is_top_model": True, "excluded_from_primary_ranking": False},
            {"date": "20260311", "view_name": "response_window", "model_id": "bile_acid", "is_top_model": False, "excluded_from_primary_ranking": False},
            {"date": "20260313", "view_name": "response_window", "model_id": "bile_acid", "is_top_model": True, "excluded_from_primary_ranking": False},
        ]
    )

    assert _build_top_prototype_models_by_date_and_view(prototype) == {
        ("20260311", "response_window"): "global_profile",
        ("20260313", "response_window"): "bile_acid",
    }
```

- [ ] **Step 2: Write a failing writer test for the new `per-date` paired figure**

```python
def test_write_stage3_outputs_writes_per_date_prototype_comparison_figures(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__full_trajectory.png").exists()
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py -k "top_prototype_models_by_date_and_view or per_date_prototype_comparison_figures" -v`

Expected: FAIL because the helper and figure writer do not exist yet.

- [ ] **Step 4: Implement the minimal `date x view` selection helper**

```python
def _build_top_prototype_models_by_date_and_view(
    prototype_rsa_results: pd.DataFrame | None,
) -> dict[tuple[str, str], str]:
    if prototype_rsa_results is None or prototype_rsa_results.empty:
        return {}

    eligible = prototype_rsa_results.copy()
    eligible["date"] = eligible["date"].astype(str)
    eligible["view_name"] = eligible["view_name"].astype(str)
    eligible["model_id"] = eligible["model_id"].astype(str)
    eligible = eligible.loc[_bool_column(eligible, "is_top_model")]
    if "excluded_from_primary_ranking" in eligible.columns:
        eligible = eligible.loc[~_bool_column(eligible, "excluded_from_primary_ranking")]
    return {
        (str(row["date"]), str(row["view_name"])): str(row["model_id"])
        for _, row in eligible.iterrows()
    }
```

Implementation requirements:

- key the mapping by `(date, view_name)`
- require `is_top_model == True`
- drop rows excluded from primary ranking
- return `{}` when the table is missing or unusable

- [ ] **Step 5: Implement the per-date paired figure renderer**

```python
def _plot_prototype_rdm_comparison_per_date(
    core_outputs: dict[str, pd.DataFrame],
    prototype_rsa_results: pd.DataFrame | None,
    *,
    view_name: str,
    path: Path,
) -> Path:
    top_models = _build_top_prototype_models_by_date_and_view(prototype_rsa_results)
    ordered_dates = _prototype_dates(core_outputs)
    figure, axes = plt.subplots(len(ordered_dates), 2, figsize=(10.5, max(4.0, 3.8 * len(ordered_dates))))
    ...
```

Implementation requirements:

- one row per `date`, two columns per row
- left panel uses the per-date prototype neural RDM for that `date x view`
- right panel resolves the paired model RDM using the top model from `prototype_rsa_results__per_date`
- right panel must reuse the neural order labels
- if the model RDM or top model is missing, render an empty-state right panel instead of failing the figure
- if the whole view has no prototype data, write an empty-state figure

- [ ] **Step 6: Thread the new per-date figure family into `_write_prototype_supplementary_figures`**

```python
for figure_name, view_name in zip(
    _build_prototype_rdm_comparison_figure_names(prototype_rsa_views),
    prototype_rsa_views,
    strict=False,
):
    written[figure_name] = _plot_prototype_rdm_comparison_per_date(
        core_outputs,
        prototype_rsa_results,
        view_name=view_name,
        path=dirs["figures_dir"] / f"{figure_name}.png",
    )
```

Implementation requirements:

- keep the existing summary plot write path unchanged
- place the new comparison figures between summary plots and pooled figures in `prototype_figure_names`
- reuse `stimulus_sample_map`-based labels through the existing heatmap helpers

- [ ] **Step 7: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py -k "top_prototype_models_by_date_and_view or per_date_prototype_comparison_figures" -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git -C H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement add src/bacteria_analysis/rsa_outputs.py tests/test_rsa.py
git -C H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement commit -m "feat: add per-date prototype rdm comparisons"
```

## Task 3: Upgrade Pooled Prototype Figures To Paired Neural/Model Layout

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py`

- [ ] **Step 1: Write a failing test that pooled prototype figures still use the existing filenames**

```python
def test_write_stage3_outputs_keeps_pooled_prototype_filenames(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    assert (written["figures_dir"] / "prototype_rdm__pooled__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm__pooled__full_trajectory.png").exists()
```

- [ ] **Step 2: Write a failing helper test for shared neural ordering on pooled prototype figures**

```python
def test_prepare_rdm_heatmap_frame_reuses_neural_order_for_paired_prototype_model():
    neural_frame, order = _prepare_rdm_heatmap_frame(prototype_neural_matrix, stimulus_sample_map)
    model_frame, reused_order = _prepare_rdm_heatmap_frame(model_matrix, stimulus_sample_map, order_labels=order)
    assert list(model_frame.index) == list(neural_frame.index)
    assert reused_order == order
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py -k "keeps_pooled_prototype_filenames or paired_prototype_model" -v`

Expected: FAIL or expose that the current pooled writer still produces a single-axis prototype-only figure.

- [ ] **Step 4: Replace `_plot_prototype_pooled_rdm` with a paired-layout writer**

```python
def _plot_prototype_pooled_rdm(
    prototype_rdm: pd.DataFrame | None,
    *,
    core_outputs: dict[str, pd.DataFrame],
    stimulus_sample_map: pd.DataFrame | None,
    top_primary_models: dict[str, str],
    view_name: str,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    neural_frame, order_labels = _prepare_rdm_heatmap_frame(prototype_rdm, stimulus_sample_map)
    _render_rdm_axis(axes[0], prototype_rdm, stimulus_sample_map=stimulus_sample_map, title=f"{view_name}: pooled prototype", fallback_message="No pooled prototype RDM provided", order_labels=order_labels)
    _render_rdm_axis(axes[1], pooled_model_rdm, stimulus_sample_map=stimulus_sample_map, title=f"{view_name}: {top_model_id or 'no top model'}", fallback_message="No paired model RDM available", order_labels=order_labels)
```

Implementation requirements:

- keep the output filenames unchanged
- choose the right-hand model from the main Stage 3 `top_primary_models_by_view`
- reuse the left-panel neural order for the model panel
- if the top model or its RDM is missing, write an empty-state right panel
- do not invent a new pooled prototype-specific ranking rule

- [ ] **Step 5: Update the pooled writer call site in `_write_prototype_supplementary_figures`**

```python
written[f"figure__{figure_name}"] = _plot_prototype_pooled_rdm(
    _dataframe_or_none(core_outputs, figure_name),
    core_outputs=core_outputs,
    stimulus_sample_map=_dataframe_or_none(core_outputs, "stimulus_sample_map"),
    top_primary_models=_build_top_primary_models_by_view(
        _dataframe_or_none(core_outputs, "rsa_results") or pd.DataFrame(),
        _collect_model_families(_dataframe_or_none(core_outputs, "model_registry") or pd.DataFrame())["primary_models"],
    ),
    view_name=view_name,
    path=dirs["figures_dir"] / f"{figure_name}.png",
)
```

Implementation requirements:

- reuse the already-computed `top_primary_models` mapping if it is available in the surrounding writer flow
- avoid recomputing model families repeatedly inside the plotting loop
- keep pooled figures descriptive only; summary metadata stays unchanged

- [ ] **Step 6: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py -k "keeps_pooled_prototype_filenames or paired_prototype_model" -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git -C H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement add src/bacteria_analysis/rsa_outputs.py tests/test_rsa.py
git -C H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement commit -m "feat: pair pooled prototype rdms with models"
```

## Task 4: Refresh Cleanup, Smoke Expectations, And Real-Data Verification

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Write a failing rerun-cleanup test for stale `per-date` paired figures**

```python
def test_write_stage3_outputs_removes_stale_prototype_comparison_figures(tmp_path, synthetic_stage3_outputs):
    first_written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    stale = first_written["figures_dir"] / "prototype_rdm_comparison__per_date__full_trajectory.png"

    narrowed_outputs = synthetic_stage3_outputs.copy()
    narrowed_outputs["prototype_rsa_results__per_date"] = narrowed_outputs["prototype_rsa_results__per_date"].loc[
        narrowed_outputs["prototype_rsa_results__per_date"]["view_name"].eq("response_window")
    ]

    second_written = write_stage3_outputs(narrowed_outputs, tmp_path / "stage3_rsa")
    assert not stale.exists()
    assert (second_written["figures_dir"] / "prototype_rdm_comparison__per_date__response_window.png").exists()
```

- [ ] **Step 2: Update the CLI smoke test expectations to include the new comparison figures**

```python
expected_paths = [
    stage3_root / "figures" / "prototype_rsa__per_date__response_window.png",
    stage3_root / "figures" / "prototype_rsa__per_date__full_trajectory.png",
    stage3_root / "figures" / "prototype_rdm_comparison__per_date__response_window.png",
    stage3_root / "figures" / "prototype_rdm_comparison__per_date__full_trajectory.png",
    stage3_root / "figures" / "prototype_rdm__pooled__response_window.png",
    stage3_root / "figures" / "prototype_rdm__pooled__full_trajectory.png",
]
```

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py -k "stale_prototype_comparison_figures" -v`

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa_cli_smoke.py -k "prototype" -v`

Expected: FAIL because stale comparison figures are not cleaned yet and smoke expectations do not match the new figure set.

- [ ] **Step 4: Extend stale-figure cleanup and smoke expectations**

```python
for stale_figure in figures_dir.glob("prototype_rdm_comparison__per_date__*.png"):
    if stale_figure.stem not in expected_figure_names:
        stale_figure.unlink()
```

Implementation requirements:

- cleanup must remove stale `prototype_rdm_comparison__per_date__*.png` files on rerun
- existing pooled prototype cleanup behavior must stay intact
- `test_rsa_cli_smoke.py` should validate the new comparison figures without changing Stage 3 table expectations

- [ ] **Step 5: Run the updated test suites**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa_cli_smoke.py -q`

Expected: PASS.

- [ ] **Step 6: Re-run real `202603` Stage 3 with prototype supplement enabled**

Run:

```powershell
pixi run python H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/scripts/run_rsa.py --stage2-root H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage2_geometry --preprocess-root H:/Process_temporary/WJH/bacteria_analysis/data/20260313_preprocess --matrix H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx --model-input-root H:/Process_temporary/WJH/bacteria_analysis/data/model_space_202603 --output-root H:/Process_temporary/WJH/bacteria_analysis/results/202603 --permutations 1000
```

Expected:

- exit code `0`
- new files exist:
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rdm_comparison__per_date__response_window.png`
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rdm_comparison__per_date__full_trajectory.png`
- upgraded pooled files still exist:
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rdm__pooled__response_window.png`
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rdm__pooled__full_trajectory.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/tables/rsa_results.parquet` remains unchanged

- [ ] **Step 7: Commit**

```powershell
git -C H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement add src/bacteria_analysis/rsa_outputs.py tests/test_rsa.py tests/test_rsa_cli_smoke.py
git -C H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement commit -m "feat: add paired prototype rdm visuals"
```

## Final Verification

- [ ] Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/.worktrees/stage3-prototype-supplement/tests/test_rsa_cli_smoke.py -q`
  Expected: PASS.

- [ ] Inspect:
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rsa__per_date__response_window.png`
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rdm_comparison__per_date__response_window.png`
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rdm__pooled__response_window.png`
  - `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/run_summary.json`

- [ ] Confirm visually:
  - `prototype_rsa__per_date__<view>.png` summary plots are still present
  - each `per-date` comparison figure has neural left, model right, one row per date
  - pooled prototype figures now show a right-hand model panel
  - model heatmaps reuse the neural order and `sample_id` labels render when available
