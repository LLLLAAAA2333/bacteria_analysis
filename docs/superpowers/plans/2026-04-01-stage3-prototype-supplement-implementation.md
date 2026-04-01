# Stage 3 Prototype Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Stage 3 with optional prototype-level supplementary outputs inside `stage3_rsa` while leaving the current pooled-RDM primary RSA result unchanged.

**Architecture:** Add a small prototype-specific helper module that reuses the existing preprocessing tensor contract, Stage 1/2 view definitions, and overlap-aware correlation-distance rule. Keep `src/bacteria_analysis/rsa.py` as the orchestrator, `src/bacteria_analysis/rsa_outputs.py` as the writer, and `scripts/run_rsa.py` as a thin CLI wrapper that optionally enables the supplementary branch via `--preprocess-root`.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, pytest, standard library (`pathlib`, `dataclasses`, `json`), existing Stage 1/2 helpers in `bacteria_analysis.reliability`

---

## Overview

This plan implements the approved spec:

- `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-04-01-stage3-prototype-supplement-design.md`

The plan intentionally does not change:

- the current primary Stage 3 pooled-RDM RSA scoring contract
- Stage 1 reliability statistics
- Stage 2 inferential outputs
- model-space schemas and curation files

The feature adds one optional supplementary branch:

- `Per-date prototype RSA` for `response_window` and `full_trajectory`
- `Pooled prototype RDM` for `response_window` and `full_trajectory`

When `--preprocess-root` is not supplied, Stage 3 must behave exactly as it does today.

## File Structure

- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_prototypes.py`
  Responsibility: load and validate prototype supplement inputs from preprocessing outputs, build grouped prototype vectors and support QC, and convert prototype vectors into overlap-aware neural RDMs.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
  Responsibility: accept optional prototype inputs, compute per-date prototype RSA, compute pooled prototype neural RDMs, attach supplementary tables/QC to `core_outputs`, and keep the current primary `rsa_results.parquet` untouched.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
  Responsibility: write supplementary prototype tables and figures, clean stale supplementary artifacts on rerun, and expose prototype metadata in `run_summary.json` / `run_summary.md`.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
  Responsibility: add `--preprocess-root`, resolve it safely, load prototype inputs only when supplied, and pass them into the Stage 3 orchestrator.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_prototypes.py`
  Responsibility: focused unit coverage for prototype construction, overlap-aware prototype RDM building, NaN handling, and support QC.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
  Responsibility: orchestration and writer contract tests for supplementary prototype outputs while locking the current primary Stage 3 outputs.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`
  Responsibility: smoke coverage for `--preprocess-root` and end-to-end prototype supplementary output writing.

No changes are planned in:

- `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
- `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability.py`
- `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/geometry.py`

## Data And Output Contract To Implement

Primary Stage 3 output remains:

- pooled neural RDM vs model RSA from Stage 2 pooled matrices

New optional supplementary artifacts to add under `stage3_rsa`:

- tables:
  - `prototype_rsa_results__per_date.parquet`
  - `prototype_rdm__pooled__response_window.parquet`
  - `prototype_rdm__pooled__full_trajectory.parquet`
- QC tables:
  - `prototype_support__per_date.parquet`
  - `prototype_support__pooled.parquet`
- figures:
  - `prototype_rsa__per_date__response_window.png`
  - `prototype_rsa__per_date__full_trajectory.png`
  - `prototype_rdm__pooled__response_window.png`
  - `prototype_rdm__pooled__full_trajectory.png`

Key implementation rules:

- prototype construction must use `trial_metadata.parquet` as row-order authority
- prototype construction must use `trial_tensor_baseline_centered.npz` as the numerical source of truth
- trial-to-model alignment must be by `stimulus`
- prototype neural distances must use the same overlap-aware correlation-distance rule already approved in Stage 1/2
- `prototype_rsa_results__per_date.parquet` gets its own FDR correction family across all finite rows in that table
- current `rsa_results.parquet` must remain byte-for-byte equivalent for the same random seed and permutations

## Task 1: Create Prototype Input And Distance Helpers

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_prototypes.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_prototypes.py`
- Read: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability.py`
- Read: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`

- [ ] **Step 1: Write failing helper tests for grouped prototype construction and overlap-aware prototype RDMs**

```python
from bacteria_analysis.reliability import TrialView
from bacteria_analysis import rsa_prototypes


def test_build_grouped_prototypes_uses_nanmean_and_tracks_support():
    metadata = pd.DataFrame(
        {
            "date": ["2026-03-11", "2026-03-11", "2026-03-11"],
            "stimulus": ["b1_1", "b1_1", "b2_1"],
            "stim_name": ["A001 stationary", "A001 stationary", "A002 stationary"],
        }
    )
    values = np.array(
        [
            [[[1.0, np.nan]], [[2.0, 4.0]]],
            [[[3.0, 5.0]], [[np.nan, 6.0]]],
            [[[7.0, 8.0]], [[9.0, 10.0]]],
        ],
        dtype=float,
    )
    view = TrialView(name="response_window", timepoints=(0, 1), metadata=metadata, values=values)

    prototypes, support = rsa_prototypes.build_grouped_prototypes(view, group_columns=("date", "stimulus", "stim_name"))

    row = prototypes.loc[(prototypes["date"] == "2026-03-11") & (prototypes["stimulus"] == "b1_1")].iloc[0]
    assert row["f000"] == pytest.approx(2.0)
    assert row["f001"] == pytest.approx(5.0)
    support_row = support.loc[(support["date"] == "2026-03-11") & (support["stimulus"] == "b1_1")].iloc[0]
    assert support_row["n_trials"] == 2
    assert support_row["n_supported_features"] == 4


def test_build_pooled_prototype_support_tracks_contributing_dates():
    metadata = pd.DataFrame(
        {
            "date": ["2026-03-11", "2026-03-13", "2026-03-13"],
            "stimulus": ["b1_1", "b1_1", "b2_1"],
            "stim_name": ["A001 stationary", "A001 stationary", "A002 stationary"],
        }
    )
    values = np.array(
        [
            [[[1.0, 2.0]]],
            [[[3.0, 4.0]]],
            [[[5.0, 6.0]]],
        ],
        dtype=float,
    )
    view = TrialView(name="response_window", timepoints=(0, 1), metadata=metadata, values=values)

    _, support = rsa_prototypes.build_grouped_prototypes(view, group_columns=("stimulus", "stim_name"))

    support_row = support.loc[support["stimulus"] == "b1_1"].iloc[0]
    assert support_row["n_trials"] == 2
    assert support_row["n_dates_contributed"] == 2


def test_load_prototype_supplement_inputs_allows_missing_wide_table(stage1_stage0_root):
    wide_path = stage1_stage0_root / "trial_level" / "trial_wide_baseline_centered.parquet"
    wide_path.unlink()

    inputs = rsa_prototypes.load_prototype_supplement_inputs(stage1_stage0_root, view_names=("response_window",))

    assert "response_window" in inputs.views


def test_build_prototype_rdm_uses_overlap_aware_correlation_distance_and_nan_for_invalid_pairs():
    prototypes = pd.DataFrame.from_records(
        [
            {"stimulus": "b1_1", "f000": 1.0, "f001": 2.0, "f002": 3.0},
            {"stimulus": "b2_1", "f000": 2.0, "f001": 4.0, "f002": 6.0},
            {"stimulus": "b3_1", "f000": 1.0, "f001": 1.0, "f002": 1.0},
        ]
    )

    matrix = rsa_prototypes.build_prototype_rdm(prototypes, id_columns=("stimulus",))

    assert matrix.loc[matrix["stimulus_row"] == "b1_1", "b2_1"].iloc[0] == pytest.approx(0.0)
    assert np.isnan(matrix.loc[matrix["stimulus_row"] == "b1_1", "b3_1"].iloc[0])
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_prototypes.py -q`

Expected: FAIL because `rsa_prototypes.py` does not exist yet.

- [ ] **Step 3: Implement the minimal prototype helper module**

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bacteria_analysis.reliability import (
    VIEW_WINDOWS,
    TrialView,
    build_trial_views,
    compute_vector_distance,
    load_reliability_inputs,
)


@dataclass(frozen=True)
class PrototypeSupplementInputs:
    metadata: pd.DataFrame
    views: dict[str, TrialView]


def load_prototype_supplement_inputs(preprocess_root: str | Path, view_names: tuple[str, ...] | list[str]) -> PrototypeSupplementInputs:
    root = Path(preprocess_root)
    wide_path = root / "trial_level" / "trial_wide_baseline_centered.parquet"
    inputs = load_reliability_inputs(
        metadata_path=root / "trial_level" / "trial_metadata.parquet",
        tensor_path=root / "trial_level" / "trial_tensor_baseline_centered.npz",
        wide_path=wide_path if wide_path.exists() else None,
    )
    view_windows = {view_name: VIEW_WINDOWS[view_name] for view_name in view_names}
    views = build_trial_views(inputs.metadata, inputs.tensor, view_windows=view_windows)
    return PrototypeSupplementInputs(metadata=inputs.metadata, views=views)


def build_grouped_prototypes(view: TrialView, group_columns: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...


def build_prototype_rdm(prototypes: pd.DataFrame, *, id_columns: tuple[str, ...]) -> pd.DataFrame:
    ...
```

Implementation requirements:

- use the tensor-backed `TrialView` values as the numerical input
- flatten feature blocks in deterministic row-major order
- compute prototype vectors with elementwise `nanmean`
- emit support QC with `n_trials`, `n_total_features`, `n_supported_features`, and `n_all_nan_features`
- include `n_dates_contributed` whenever the grouping collapses across dates
- compute prototype neural distances with the existing overlap-aware correlation-distance rule via `compute_vector_distance`
- store invalid pair distances as `NaN`
- treat `trial_wide_baseline_centered.parquet` as optional convenience input only

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_prototypes.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_prototypes.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_prototypes.py
git commit -m "feat: add stage3 prototype helper module"
```

## Task 2: Integrate Prototype Supplements Into `run_stage3_rsa`

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Read: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_prototypes.py`

- [ ] **Step 1: Write failing orchestration tests for supplementary prototype results**

```python
from bacteria_analysis.rsa import benjamini_hochberg


def test_run_stage3_rsa_adds_prototype_tables_when_prototype_inputs_are_present(
    stage3_resolved_inputs,
    stage3_neural_rdms,
    stage3_prototype_inputs,
):
    results = run_stage3_rsa(
        stage3_resolved_inputs,
        neural_matrices=stage3_neural_rdms,
        prototype_inputs=stage3_prototype_inputs,
        permutations=10,
        seed=0,
    )

    assert "prototype_rsa_results__per_date" in results
    assert "prototype_support__per_date" in results
    assert "prototype_support__pooled" in results
    assert "prototype_rdm__pooled__response_window" in results
    assert "prototype_rdm__pooled__full_trajectory" in results


def test_run_stage3_rsa_keeps_primary_rsa_results_unchanged_when_prototype_inputs_are_present(
    stage3_resolved_inputs,
    stage3_neural_rdms,
    stage3_prototype_inputs,
):
    baseline = run_stage3_rsa(stage3_resolved_inputs, neural_matrices=stage3_neural_rdms, permutations=10, seed=0)
    supplemented = run_stage3_rsa(
        stage3_resolved_inputs,
        neural_matrices=stage3_neural_rdms,
        prototype_inputs=stage3_prototype_inputs,
        permutations=10,
        seed=0,
    )

    pd.testing.assert_frame_equal(baseline["rsa_results"], supplemented["rsa_results"])


def test_run_stage3_rsa_corrects_prototype_fdr_across_full_table_and_ignores_excluded_rows_for_top_model(
    stage3_resolved_inputs,
    stage3_neural_rdms,
    stage3_prototype_inputs,
):
    results = run_stage3_rsa(
        stage3_resolved_inputs,
        neural_matrices=stage3_neural_rdms,
        prototype_inputs=stage3_prototype_inputs,
        permutations=10,
        seed=0,
    )

    prototype = results["prototype_rsa_results__per_date"]
    finite = prototype.loc[prototype["p_value_raw"].notna()].copy()
    expected_fdr = benjamini_hochberg(finite["p_value_raw"].to_numpy(dtype=float))
    np.testing.assert_allclose(finite["p_value_fdr"].to_numpy(dtype=float), expected_fdr)
    assert not prototype.loc[prototype["excluded_from_primary_ranking"].astype(bool), "is_top_model"].any()

    for (_, _), group in prototype.groupby(["date", "view_name"], sort=False):
        eligible = group.loc[
            group["score_status"].eq("ok")
            & ~group["excluded_from_primary_ranking"].astype(bool)
            & group["rsa_similarity"].notna()
        ]
        if eligible.empty:
            continue
        assert int(eligible["is_top_model"].sum()) == 1


def test_run_stage3_rsa_restricts_model_rdms_by_date_stimulus_set_and_marks_sparse_rows_invalid(
    stage3_resolved_inputs,
    stage3_neural_rdms,
    stage3_sparse_prototype_inputs,
):
    results = run_stage3_rsa(
        stage3_resolved_inputs,
        neural_matrices=stage3_neural_rdms,
        prototype_inputs=stage3_sparse_prototype_inputs,
        permutations=10,
        seed=0,
    )

    prototype = results["prototype_rsa_results__per_date"]
    sparse_rows = prototype.loc[prototype["date"].eq("2026-03-13")]
    assert sparse_rows["score_status"].eq("invalid").all()
    assert sparse_rows["n_shared_entries"].fillna(0).astype(int).max() < 2
```

- [ ] **Step 2: Run the orchestration tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "prototype_tables_when_prototype_inputs_are_present or primary_rsa_results_unchanged_when_prototype_inputs_are_present or prototype_fdr_across_full_table_and_ignores_excluded_rows_for_top_model or restricts_model_rdms_by_date_stimulus_set_and_marks_sparse_rows_invalid" -q`

Expected: FAIL because `run_stage3_rsa()` does not yet accept `prototype_inputs`.

- [ ] **Step 3: Extend `run_stage3_rsa()` with the supplementary prototype branch**

```python
def run_stage3_rsa(
    resolved_inputs: dict[str, pd.DataFrame],
    *,
    neural_matrices: dict[str, pd.DataFrame],
    prototype_inputs: PrototypeSupplementInputs | None = None,
    permutations: int = 0,
    seed: int = 0,
    ...
) -> dict[str, pd.DataFrame]:
    ...
    if prototype_inputs is not None:
        prototype_outputs = build_stage3_prototype_supplement(
            resolved_inputs,
            prototype_inputs=prototype_inputs,
            model_ids=registry["model_id"].astype(str).tolist(),
            permutations=permutations,
            seed=seed,
            view_names=requested_views,
        )
        core_outputs.update(prototype_outputs)
```

Implementation requirements:

- keep the existing primary `rsa_results`, `rsa_view_comparison`, and `rsa_leave_one_stimulus_out` path untouched
- restrict each prototype RSA comparison to the date-specific stimulus subset before scoring against the model RDM
- if too few shared entries remain after that restriction, emit an `invalid` supplementary row instead of forcing a score
- compute `prototype_rsa_results__per_date.parquet` rows with:
  - `date`
  - `view_name`
  - `reference_view_name`
  - `comparison_scope`
  - model metadata columns
  - `excluded_from_primary_ranking`
  - `score_method`
  - `score_status`
  - `n_stimuli`
  - `n_shared_entries`
  - `rsa_similarity`
  - `p_value_raw`
  - `p_value_fdr`
  - `is_top_model`
- compute `p_value_fdr` across all finite rows in the full supplementary per-date table
- compute `is_top_model` within each `date x view`, excluding `excluded_from_primary_ranking == True` rows
- write pooled prototype RDM tables into `core_outputs` under view-specific keys
- ensure pooled support QC carries `n_dates_contributed`

- [ ] **Step 4: Run the orchestration tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "prototype_tables_when_prototype_inputs_are_present or primary_rsa_results_unchanged_when_prototype_inputs_are_present or prototype_fdr_across_full_table_and_ignores_excluded_rows_for_top_model or restricts_model_rdms_by_date_stimulus_set_and_marks_sparse_rows_invalid" -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: add stage3 prototype supplementary rsa"
```

## Task 3: Write Supplementary Prototype Tables, Figures, And Summary Metadata

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`

- [ ] **Step 1: Write failing output tests for prototype supplementary artifacts**

```python
def test_write_stage3_outputs_writes_prototype_supplementary_artifacts(tmp_path, synthetic_stage3_outputs_with_prototypes):
    written = write_stage3_outputs(synthetic_stage3_outputs_with_prototypes, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert (written["tables_dir"] / "prototype_rsa_results__per_date.parquet").exists()
    assert (written["tables_dir"] / "prototype_rdm__pooled__response_window.parquet").exists()
    assert (written["tables_dir"] / "prototype_rdm__pooled__full_trajectory.parquet").exists()
    assert (written["qc_dir"] / "prototype_support__per_date.parquet").exists()
    assert (written["qc_dir"] / "prototype_support__pooled.parquet").exists()
    assert (written["figures_dir"] / "prototype_rsa__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm__pooled__response_window.png").exists()
    assert summary["prototype_supplement_enabled"] is True
    assert summary["prototype_views"] == ["response_window", "full_trajectory"]
    assert summary["prototype_dates"] == ["2026-03-11", "2026-03-13"]
    assert summary["prototype_descriptive_outputs"] == [
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
    assert summary["prototype_table_names"] == [
        "prototype_rsa_results__per_date",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
    assert summary["prototype_figure_names"] == [
        "prototype_rsa__per_date__response_window",
        "prototype_rsa__per_date__full_trajectory",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
    assert "prototype_rsa_results__per_date" in summary["additional_table_names"]
    assert "prototype_rsa__per_date__response_window" in summary["figure_names"]


def test_write_stage3_outputs_removes_stale_prototype_figures_when_view_set_narrows(tmp_path, synthetic_stage3_outputs_with_prototypes):
    output_root = tmp_path / "stage3_rsa"
    write_stage3_outputs(synthetic_stage3_outputs_with_prototypes, output_root)
    narrowed = synthetic_stage3_outputs_with_prototypes.copy()
    narrowed["prototype_rdm__pooled__full_trajectory"] = pd.DataFrame()
    narrowed["prototype_rsa_results__per_date"] = narrowed["prototype_rsa_results__per_date"].loc[
        narrowed["prototype_rsa_results__per_date"]["view_name"].eq("response_window")
    ].reset_index(drop=True)

    written = write_stage3_outputs(narrowed, output_root)

    assert not (written["figures_dir"] / "prototype_rdm__pooled__full_trajectory.png").exists()
```

- [ ] **Step 2: Run the output tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "prototype_supplementary_artifacts or stale_prototype_figures_when_view_set_narrows" -q`

Expected: FAIL because the writer does not yet know about prototype supplementary keys.

- [ ] **Step 3: Extend `rsa_outputs.py` to write prototype supplementary outputs**

```python
def _write_stage3_artifacts(core_outputs: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> dict[str, Path]:
    _remove_matching_files(dirs["figures_dir"], "prototype_*.png")
    ...
    if "prototype_rsa_results__per_date" in core_outputs:
        written["prototype_rsa_results__per_date"] = write_parquet(
            core_outputs["prototype_rsa_results__per_date"],
            dirs["tables_dir"] / "prototype_rsa_results__per_date.parquet",
        )
        written["prototype_support__per_date"] = write_parquet(
            core_outputs["prototype_support__per_date"],
            dirs["qc_dir"] / "prototype_support__per_date.parquet",
        )
        ...
```

Implementation requirements:

- write supplementary tables/QC only when the keys are present in `core_outputs`
- append prototype tables to `additional_table_names`, not `rsa_table_names`
- append prototype figure names to `figure_names`
- preserve existing Stage 3 primary summary fields
- write `prototype_supplement_enabled`, `prototype_views`, `prototype_dates`, `prototype_table_names`, `prototype_figure_names`, and `prototype_descriptive_outputs` exactly as specified in the spec
- render `prototype_rsa__per_date__<view>.png` as a view-specific summary plot over `date`
- render `prototype_rdm__pooled__<view>.png` as a view-specific neural-only heatmap in the table's stimulus order without adding a new clustering requirement
- remove stale `prototype_*.png` artifacts before rewriting current supplementary figures

- [ ] **Step 4: Run the output tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "prototype_supplementary_artifacts or stale_prototype_figures_when_view_set_narrows or write_stage3_outputs" -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: write stage3 prototype supplementary outputs"
```

## Task 4: Add `--preprocess-root` And End-To-End CLI Coverage

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Write a failing CLI smoke test for the supplementary prototype branch**

```python
def test_cli_runs_and_writes_prototype_supplement_outputs(tmp_path, stage3_fixture_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_rsa.py",
            "--stage2-root",
            str(stage3_fixture_root / "stage2_geometry"),
            "--preprocess-root",
            str(stage3_fixture_root / "preprocess"),
            "--matrix",
            str(stage3_fixture_root / "matrix.xlsx"),
            "--model-input-root",
            str(stage3_fixture_root / "model_space"),
            "--output-root",
            str(output_root),
            "--permutations",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stage3_root = output_root / "stage3_rsa"
    assert (stage3_root / "tables" / "prototype_rsa_results__per_date.parquet").exists()
    assert (stage3_root / "figures" / "prototype_rdm__pooled__response_window.png").exists()
```

- [ ] **Step 2: Run the CLI smoke test to verify it fails**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -k "prototype_supplement_outputs" -q`

Expected: FAIL because `run_rsa.py` does not yet accept `--preprocess-root`.

- [ ] **Step 3: Add optional preprocess-root plumbing to the CLI**

```python
parser.add_argument(
    "--preprocess-root",
    default=None,
    help="Optional preprocessing root that enables Stage 3 prototype supplementary outputs.",
)
...
prototype_inputs = None
if args.preprocess_root:
    prototype_inputs = load_prototype_supplement_inputs(args.preprocess_root, view_names=tuple(neural_matrices))
core_outputs = run_stage3_rsa(
    resolved_inputs,
    neural_matrices=neural_matrices,
    prototype_inputs=prototype_inputs,
    permutations=args.permutations,
    seed=args.seed,
)
```

Implementation requirements:

- keep the command valid when `--preprocess-root` is omitted
- load prototype inputs only when `--preprocess-root` is provided
- do not change the current default output root or required-input contract
- update the fixture in `tests/test_rsa_cli_smoke.py` so the smoke dataset includes an aligned `preprocess/` trial-level root

- [ ] **Step 4: Run the CLI smoke tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "feat: add stage3 prototype supplement cli input"
```

## Task 5: Full Regression And Real-Data Verification

**Files:**
- Verify only unless a bug is discovered:
  - `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_prototypes.py`
  - `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
  - `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
  - `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
  - `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_prototypes.py`
  - `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
  - `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Run the full Stage 3 prototype supplement regression suite**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_prototypes.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -q`

Expected: PASS.

- [ ] **Step 2: Capture the primary Stage 3 RSA digest before the real rerun**

Run:

```powershell
@'
import hashlib
from pathlib import Path
import pandas as pd
from pandas.util import hash_pandas_object

path = Path(r"H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/tables/rsa_results.parquet")
df = pd.read_parquet(path)
print(hashlib.sha256(hash_pandas_object(df, index=True).values.tobytes()).hexdigest())
'@ | pixi run python -
```

Expected: one stable SHA256 digest string to compare after the rerun.

- [ ] **Step 3: Re-run the real `202603` Stage 3 command with prototype supplements enabled**

Run:

```powershell
pixi run python H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py --stage2-root H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage2_geometry --preprocess-root H:/Process_temporary/WJH/bacteria_analysis/data/20260313_preprocess --matrix H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx --model-input-root H:/Process_temporary/WJH/bacteria_analysis/data/model_space_202603 --output-root H:/Process_temporary/WJH/bacteria_analysis/results/202603 --permutations 1000
```

Expected:

- exit code `0`
- primary Stage 3 tables still written
- supplementary prototype tables/QC/figures written under `results/202603/stage3_rsa`

- [ ] **Step 4: Verify the real output contract**

Inspect:

- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/tables/prototype_rsa_results__per_date.parquet`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/tables/prototype_rdm__pooled__response_window.parquet`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/tables/prototype_rdm__pooled__full_trajectory.parquet`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/qc/prototype_support__per_date.parquet`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/qc/prototype_support__pooled.parquet`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rsa__per_date__response_window.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/figures/prototype_rdm__pooled__response_window.png`
- `H:/Process_temporary/WJH/bacteria_analysis/results/202603/stage3_rsa/run_summary.json`

Expected:

- the supplementary outputs exist
- `run_summary.json` exposes `prototype_supplement_enabled`, `prototype_views`, `prototype_dates`, `prototype_table_names`, `prototype_figure_names`, and `prototype_descriptive_outputs`
- `rsa_results.parquet` hash matches the pre-rerun digest exactly

- [ ] **Step 5: Commit any verification fixes, or skip commit if verification is clean**

```bash
# Only if a verification bug required code changes:
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_prototypes.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_prototypes.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "fix: finalize stage3 prototype supplement verification"
```

If no code changed during verification, do not create a no-op commit.
