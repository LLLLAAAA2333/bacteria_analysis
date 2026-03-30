# Stage 2 Neural Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Stage 2 dual-view geometry pipeline that builds pooled, per-individual, and per-date neural RDM summaries for `response_window` and `full_trajectory`, then quantifies geometry stability before any RSA work.

**Architecture:** Build Stage 2 on top of the completed Stage 0 trial-level inputs and the existing Stage 1 pairwise-comparison logic. Keep the core analysis in one focused module that reuses Stage 1 view slicing and overlap-aware correlation distance, then add a separate output module and a thin CLI. Emit long-form stimulus-pair tables as the main persisted geometry object, materialize only the pooled matrices as standalone matrix files, and keep group-level stability summaries primary.

**Tech Stack:** Python 3.11, pandas, numpy, pyarrow, pytest, matplotlib/seaborn, scipy for Spearman correlation if already available, standard library (`pathlib`, `json`, `argparse`)

---

## Overview

This plan implements the approved Stage 2 dual-view MVP from:

- `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-03-30-stage2-neural-geometry-design.md`

The implementation must answer one question cleanly:

> can the neural stimulus geometry be represented as stable RDMs across pooled, individual, and date groupings for `response_window` and `full_trajectory`?

The plan intentionally does not include:

- RSA/model-RDM matching
- `on_window` or `post_window` support in the CLI defaults
- sliding-window geometry
- dendrograms or embeddings as primary outputs

## File Structure

- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py`
  Responsibility: Stage 2 core analysis helpers for grouped stimulus-pair summaries, pooled matrix construction, upper-triangle vectorization, and RDM stability summaries.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry_outputs.py`
  Responsibility: write Stage 2 tables, QC outputs, figures, and run summaries without mixing output logic into the analysis code.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_geometry.py`
  Responsibility: thin CLI that loads Stage 0 inputs, reuses Stage 1 view logic, runs the Stage 2 geometry pipeline, and writes `results/stage2_geometry/`.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py`
  Responsibility: unit coverage for grouped stimulus-pair aggregation, matrix reconstruction, missing-value handling, and stability statistics.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py`
  Responsibility: end-to-end CLI smoke test for the dual-view MVP.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`
  Responsibility: provide a tiny Stage 0-like fixture or reuse the existing reliability fixture for Stage 2 smoke coverage.

## Output Structure

Stage 2 outputs should live under:

- `H:/Process_temporary/WJH/bacteria_analysis/results/stage2_geometry/`

Required subdirectories:

- `H:/Process_temporary/WJH/bacteria_analysis/results/stage2_geometry/tables/`
- `H:/Process_temporary/WJH/bacteria_analysis/results/stage2_geometry/figures/`
- `H:/Process_temporary/WJH/bacteria_analysis/results/stage2_geometry/qc/`

Required artifact families:

- dual-view grouped stimulus-pair tables
- pooled RDM matrix files
- stability summary tables for individual and date groupings
- cross-view comparison table
- QC coverage table
- run summary JSON/Markdown
- pooled heatmaps and compact stability figures

## Task 1: Scaffold Stage 2 Test Fixtures And Module Surface

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py`

- [ ] **Step 1: Write the failing fixture-backed tests for grouped stimulus-pair aggregation**

```python
def test_individual_group_summary_excludes_cross_individual_pairs(synthetic_geometry_comparisons):
    summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="individual",
    )
    assert set(summary["group_id"]) == {"2026-01-01__worm_a", "2026-01-02__worm_b"}
    assert (summary["group_type"] == "individual").all()
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py::test_individual_group_summary_excludes_cross_individual_pairs -v`
Expected: FAIL with `ImportError` or `NameError` because `geometry.py` and `summarize_grouped_stimulus_pairs()` do not exist yet.

- [ ] **Step 3: Write the minimal geometry module surface and fixture helpers**

```python
GROUP_TYPES = ("pooled", "individual", "date")

def summarize_grouped_stimulus_pairs(comparisons: pd.DataFrame, view_name: str, group_type: str) -> pd.DataFrame:
    if group_type not in GROUP_TYPES:
        raise ValueError(f"unsupported group_type: {group_type}")
    view_frame = comparisons.loc[comparisons["view_name"] == view_name].copy()
    return _aggregate_grouped_pairs(view_frame, group_type=group_type)
```

Add a tiny comparisons fixture in `tests/conftest.py` with:

- two views: `response_window`, `full_trajectory`
- at least two individuals across two dates
- same-stimulus and different-stimulus pairs
- both valid and invalid comparisons

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py -k grouped -v`
Expected: PASS on the new grouped-summary tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py
git commit -m "feat: scaffold stage 2 geometry aggregation"
```

## Task 2: Implement Grouped Stimulus-Pair Summaries And Pooled RDM Matrices

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py`

- [ ] **Step 1: Write failing tests for pooled, individual, and date aggregation plus matrix reconstruction**

```python
def test_build_pooled_rdm_matrix_is_symmetric(synthetic_geometry_comparisons):
    pair_summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="pooled",
    )
    matrix = build_rdm_matrix(pair_summary, group_id="all").set_index("stimulus_row")
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.equals(matrix.T)
```

```python
def test_date_group_summary_keeps_only_same_date_pairs(synthetic_geometry_comparisons):
    summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="full_trajectory",
        group_type="date",
    )
    assert set(summary["group_id"]) == {"2026-01-01", "2026-01-02"}
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py -k "matrix or date_group" -v`
Expected: FAIL because `build_rdm_matrix()` and the date-group filtering logic are incomplete.

- [ ] **Step 3: Implement minimal grouped-summary and matrix helpers**

```python
def build_rdm_matrix(pair_summary: pd.DataFrame, group_id: str) -> pd.DataFrame:
    group = pair_summary.loc[pair_summary["group_id"] == group_id].copy()
    matrix = _pivot_symmetric_distance_matrix(group, value_column="mean_distance")
    return matrix.reset_index()
```

Implementation requirements:

- `pooled` uses all valid comparisons from the selected view
- `individual` uses only pairs where `individual_id_a == individual_id_b`
- `date` uses only pairs where `date_a == date_b`
- stimulus pairs are unordered
- matrices are symmetric and keep same-stimulus diagonal distances

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py -k "group_summary or matrix" -v`
Expected: PASS on pooled/date/individual aggregation and matrix reconstruction tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py
git commit -m "feat: build stage 2 grouped rdm summaries"
```

## Task 3: Implement Upper-Triangle Vectorization And RDM Stability Statistics

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py`

- [ ] **Step 1: Write failing tests for upper-triangle extraction and Spearman-based stability scoring**

```python
def test_score_rdm_similarity_uses_shared_non_missing_entries_only():
    left = pd.DataFrame(
        {"stimulus_row": ["s1", "s2"], "s1": [0.1, 0.3], "s2": [0.3, 0.2]}
    )
    right = pd.DataFrame(
        {"stimulus_row": ["s1", "s2"], "s1": [0.2, 0.4], "s2": [0.4, 0.1]}
    )
    result = score_rdm_similarity(left, right)
    assert result["score_status"] == "ok"
    assert result["n_shared_entries"] >= 1
```

```python
def test_score_rdm_similarity_marks_invalid_when_no_shared_entries():
    result = score_rdm_similarity(left_empty, right_empty)
    assert result["score_status"] == "invalid"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py -k "similarity or upper_triangle" -v`
Expected: FAIL because vectorization and similarity helpers are not implemented yet.

- [ ] **Step 3: Implement the minimal stability helpers**

```python
def extract_upper_triangle(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix_frame.set_index("stimulus_row")
    return _matrix_upper_triangle_records(matrix)

def score_rdm_similarity(left_matrix: pd.DataFrame, right_matrix: pd.DataFrame) -> dict[str, object]:
    shared = _shared_upper_triangle(left_matrix, right_matrix)
    return _score_shared_triangle(shared, method="spearman")

def summarize_rdm_stability(pair_summary_by_group: pd.DataFrame) -> pd.DataFrame:
    grouped_matrices = build_group_matrices(pair_summary_by_group)
    return _score_group_matrix_sets(grouped_matrices)
```

Implementation requirements:

- compare only shared non-missing upper-triangle entries
- use `Spearman` correlation by default
- emit `score_status`, `n_shared_entries`, and `similarity`
- compute:
  - group-vs-group similarity within individuals
  - group-vs-group similarity within dates
  - pooled-vs-group similarity
  - pooled cross-view similarity

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py -k "similarity or stability" -v`
Expected: PASS on upper-triangle and stability-summary tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py
git commit -m "feat: add stage 2 rdm stability summaries"
```

## Task 4: Implement Output Writers And Run Summary

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py`

- [ ] **Step 1: Write failing tests for Stage 2 output paths and run summary structure**

```python
def test_write_stage2_outputs_writes_required_tables(tmp_path, synthetic_geometry_outputs):
    written = write_stage2_outputs(synthetic_geometry_outputs, tmp_path / "stage2_geometry")
    assert (written["tables_dir"] / "rdm_stability_by_individual.parquet").exists()
    assert (written["figures_dir"] / "rdm_matrix__response_window__pooled.png").exists()
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py -k "write_stage2_outputs" -v`
Expected: FAIL because `geometry_outputs.py` and the writer functions do not exist yet.

- [ ] **Step 3: Write minimal output helpers**

```python
def ensure_stage2_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    return _mkdir_stage2_dirs(root)

def write_stage2_outputs(core_outputs: dict[str, pd.DataFrame], output_root: str | Path) -> dict[str, Path]:
    dirs = ensure_stage2_output_dirs(output_root)
    return _write_stage2_artifacts(core_outputs, dirs)
```

Implementation requirements:

- write the required `tables/`, `figures/`, and `qc/` outputs
- materialize pooled matrix parquet files only
- write `run_summary.json` and `run_summary.md`
- produce figures for:
  - pooled `response_window` RDM
  - pooled `full_trajectory` RDM
  - individual stability summary
  - date stability summary
  - view comparison summary

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py -k "output or summary" -v`
Expected: PASS on output-path and run-summary tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py
git commit -m "feat: add stage 2 geometry outputs"
```

## Task 5: Implement The Thin CLI And End-To-End Smoke Test

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_geometry.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry_outputs.py`

- [ ] **Step 1: Write the failing smoke test for the dual-view CLI**

```python
def test_cli_runs_and_writes_stage2_outputs(tmp_path, stage1_stage0_root):
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_geometry.py",
            "--input-root",
            str(stage1_stage0_root),
            "--output-root",
            str(tmp_path / "results"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py -v`
Expected: FAIL because `scripts/run_geometry.py` does not exist yet.

- [ ] **Step 3: Implement the minimal CLI**

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 neural geometry from Stage 0 trial outputs.")
    parser.add_argument("--input-root", help="Root directory containing Stage 0 trial_level outputs.")
    parser.add_argument("--output-root", default="results", help="Base directory for Stage 2 outputs.")
    parser.add_argument("--views", default="response_window,full_trajectory", help="Comma-separated Stage 2 views.")
    return parser.parse_args(argv)
```

Implementation requirements:

- load Stage 0 trial metadata/tensor inputs
- build or reuse Stage 1 pairwise comparisons for the selected views
- run the Stage 2 geometry pipeline
- write outputs to `results/stage2_geometry/`
- print the included views and key output paths

- [ ] **Step 4: Run the smoke test and targeted unit tests to verify they pass**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py -v`
Expected: PASS with created Stage 2 outputs under the temporary `results/` directory.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/scripts/run_geometry.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry_outputs.py
git commit -m "feat: add stage 2 geometry cli"
```

## Task 6: Full Regression Pass And Real-Data Dry Run

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_geometry.py`

- [ ] **Step 1: Add any missing regression tests discovered during integration**

```python
def test_run_summary_records_actual_included_views(tmp_path, synthetic_geometry_outputs):
    written = write_stage2_outputs(synthetic_geometry_outputs, tmp_path / "stage2_geometry")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["views"] == ["response_window", "full_trajectory"]
```

```python
def test_pooled_vs_group_similarity_marks_invalid_sparse_groups(sparse_group_pair_summary):
    summary = summarize_rdm_stability(sparse_group_pair_summary)
    assert "invalid" in set(summary["score_status"])
```

- [ ] **Step 2: Run the full targeted geometry test suite**

Run: `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py -v`
Expected: PASS on all Stage 2 geometry tests.

- [ ] **Step 3: Run the real-data dry run**

Run: `pixi run python H:/Process_temporary/WJH/bacteria_analysis/scripts/run_geometry.py --output-root H:/Process_temporary/WJH/bacteria_analysis/results`
Expected: exit code `0`, `results/stage2_geometry/` created, pooled matrix files and stability summary tables written.

- [ ] **Step 4: Inspect the run summary and one pooled heatmap**

Run: `Get-Content -Raw H:/Process_temporary/WJH/bacteria_analysis/results/stage2_geometry/run_summary.json`
Expected: the JSON lists `response_window` and `full_trajectory` as the included views and points to the expected table/figure directories.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry_outputs.py H:/Process_temporary/WJH/bacteria_analysis/scripts/run_geometry.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py
git commit -m "test: validate stage 2 geometry pipeline"
```

## Implementation Notes

- Reuse `build_trial_views()` from `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability.py` instead of duplicating view-window definitions.
- Reuse Stage 1 overlap-aware comparison behavior and only add group-aware aggregation on top.
- Do not require `results/stage1_reliability/` to exist at runtime. If Stage 1 artifacts are present, they may be used only for cross-checking.
- Keep the CLI default views fixed to `response_window,full_trajectory`, but parse the value as a comma-separated list so later multi-view expansion is cheap.
- Do not write one matrix file per individual/date in the MVP.
- If `scipy` is not already available, either add the smallest possible dependency change or implement Spearman via rank-transform plus `np.corrcoef` in `geometry.py`.

## Final Validation Checklist

- [ ] `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry.py -v`
- [ ] `pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_geometry_cli_smoke.py -v`
- [ ] `pixi run python H:/Process_temporary/WJH/bacteria_analysis/scripts/run_geometry.py --output-root H:/Process_temporary/WJH/bacteria_analysis/results`
- [ ] Inspect `H:/Process_temporary/WJH/bacteria_analysis/results/stage2_geometry/run_summary.json`
- [ ] Inspect one pooled matrix parquet and one pooled heatmap

## Handoff

When this plan is complete, the next human review should focus on:

- whether `response_window` and `full_trajectory` tell a consistent geometry story
- whether date-level instability is a biological limitation or a support/coverage artifact
- whether Stage 2 is stable enough to justify Stage 3 RSA planning
