# Stage 3 Biochemical RSA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Stage 3 pooled RSA pipeline that links the approved pooled neural RDMs to curated metabolite-space models, with permutation nulls, leave-one-stimulus-out robustness checks, and `response_window` versus `full_trajectory` sensitivity reporting.

**Architecture:** Build Stage 3 on top of the completed Stage 2 pooled RDM outputs and keep the neural side fixed. Put model-space resolution in one focused module that reads human-editable CSV inputs plus `data/matrix.xlsx`, keep RSA statistics in a separate module, and keep output writing in a separate writer module. The CLI should stay thin, default to the Stage 2/Stage 3 canonical paths, and remain runnable even when only `global_profile` is fully defined, while clearly excluding underpowered draft models from primary ranking.

**Tech Stack:** Python 3.11, pandas, numpy, pyarrow, openpyxl, pytest, matplotlib/seaborn, standard library (`pathlib`, `json`, `argparse`, `zipfile`)

---

## Overview

This plan implements the approved Stage 3 design from:

- `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-03-31-stage3-biochemical-rsa-design.md`

The implementation must answer one question cleanly:

> which metabolite-space models best explain the pooled neural geometry across the current `19` stimuli, and do those rankings survive null and robustness checks?

The plan intentionally does not include:

- grouped RSA as a primary result
- time-resolved or sliding-window RSA
- changing the approved neural distance metric
- ontology mining or automated chemical knowledge retrieval
- neuron contribution analysis
- behavior linkage

## File Structure

- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
  Responsibility: load and validate the Stage 3 human-editable inputs, read the metabolite matrix from `data/matrix.xlsx`, derive resolved annotation/membership tables, preprocess model features, and construct model RDMs plus feature-level QC.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
  Responsibility: align neural/model upper triangles, compute Spearman RSA scores, build permutation nulls, apply false-discovery-rate correction, and summarize leave-one-stimulus-out plus cross-view comparisons.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
  Responsibility: write Stage 3 tables, QC outputs, figures, and run summaries without mixing output logic into the analysis code.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
  Responsibility: thin CLI that loads Stage 2 pooled RDM inputs plus Stage 3 model inputs, runs the Stage 3 RSA pipeline, and writes `results/stage3_rsa/`.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`
  Responsibility: unit coverage for input validation, metabolite matrix loading, annotation/membership resolution, feature preprocessing, and model RDM construction.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
  Responsibility: unit coverage for upper-triangle alignment, Spearman scoring, permutation nulls, FDR correction, and leave-one-stimulus-out summaries.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`
  Responsibility: end-to-end CLI smoke coverage for the Stage 3 MVP.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`
  Responsibility: provide tiny Stage 2-like pooled RDM fixtures plus small Stage 3 model-input fixtures, including a synthetic `.xlsx` matrix file.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml`
  Responsibility: add `openpyxl` and a dedicated `rsa` task so the Stage 3 CLI runs inside the project environment.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/stimulus_sample_map.csv`
  Responsibility: explicit `stimulus x stim_name x sample_id` source-of-truth mapping for the current neural panel.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/metabolite_annotation.csv`
  Responsibility: human-editable metabolite annotation table seeded from the matrix headers and later reviewed by the user.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/model_registry.csv`
  Responsibility: explicit model registry distinguishing primary curated models, supplementary exploratory models, and draft entries.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/model_membership.csv`
  Responsibility: explicit long-form `model_id x metabolite_name` membership definitions for curated and exploratory subset models.

## Input Structure

Stage 3 should read human-editable inputs from:

- `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/`

Required input files:

- `stimulus_sample_map.csv`
- `metabolite_annotation.csv`
- `model_registry.csv`
- `model_membership.csv`

Recommended CSV schemas:

- `stimulus_sample_map.csv`
  Columns: `stimulus`, `stim_name`, `sample_id`
- `metabolite_annotation.csv`
  Columns: `metabolite_name`, `superclass`, `subclass`, `pathway_tag`, `annotation_source`, `review_status`, `ambiguous_flag`, `notes`
- `model_registry.csv`
  Columns: `model_id`, `model_label`, `model_tier`, `model_status`, `feature_kind`, `distance_kind`, `description`, `authority`, `notes`
- `model_membership.csv`
  Columns: `model_id`, `metabolite_name`, `membership_source`, `review_status`, `ambiguous_flag`, `notes`

Implementation rule:

- these files are scientific inputs, so the runtime should validate them aggressively
- the runtime should not silently infer primary model memberships from metabolite names
- template seeding may be offered as an explicit helper, but it must not silently happen during the main RSA run

## Output Structure

Stage 3 outputs should live under:

- `H:/Process_temporary/WJH/bacteria_analysis/results/stage3_rsa/`

Required subdirectories:

- `H:/Process_temporary/WJH/bacteria_analysis/results/stage3_rsa/tables/`
- `H:/Process_temporary/WJH/bacteria_analysis/results/stage3_rsa/figures/`
- `H:/Process_temporary/WJH/bacteria_analysis/results/stage3_rsa/qc/`

Required artifact families:

- resolved model-input tables
- model-level feature QC tables
- model RDM summaries
- RSA result tables across primary and supplementary families
- leave-one-stimulus-out robustness tables
- cross-view comparison tables
- run summary JSON/Markdown
- compact ranked-model and neural-versus-model figures

## Task 1: Add Stage 3 Dependency, Fixture Scaffolding, And Module Surface

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`

- [ ] **Step 1: Write the failing Stage 3 scaffold tests**

```python
def test_load_stimulus_sample_map_requires_unique_sample_ids(stage3_model_input_root):
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        load_stimulus_sample_map(stage3_model_input_root / "duplicate_stimulus_sample_map.csv")


def test_read_metabolite_matrix_loads_expected_sample_ids(stage3_matrix_path):
    matrix = read_metabolite_matrix(stage3_matrix_path)
    assert matrix.index.tolist() == ["A001", "A002", "A003"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py -k "stimulus_sample_map or metabolite_matrix" -v`
Expected: FAIL because `model_space.py`, the Stage 3 fixtures, and the `openpyxl` dependency are not in place yet.

- [ ] **Step 3: Add the minimal dependency, fixture helpers, and loader surface**

```toml
[tasks]
rsa = "python scripts/run_rsa.py"

[dependencies]
openpyxl = ">=3.1,<4"
```

```python
def load_stimulus_sample_map(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    return _validate_stimulus_sample_map(frame)


def read_metabolite_matrix(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(path, engine="openpyxl")
    return _normalize_matrix_frame(frame)
```

Add small Stage 3 fixtures in `tests/conftest.py`:

- a tiny synthetic Stage 2 pooled RDM root with `response_window` and `full_trajectory`
- a tiny `matrix.xlsx` file with sample ids in column `A`
- tiny CSV inputs under a temporary `model_space` directory

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py -k "stimulus_sample_map or metabolite_matrix" -v`
Expected: PASS on the new scaffold tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/pixi.toml H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py
git commit -m "build: scaffold stage 3 model-space inputs"
```

## Task 2: Implement Input Resolution, Annotation Seeding, And Validation Rules

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`

- [ ] **Step 1: Write failing tests for annotation, registry, and membership resolution**

```python
def test_resolve_model_inputs_rejects_primary_model_with_union_like_status(stage3_model_input_root, stage3_matrix_path):
    with pytest.raises(ValueError, match="supplementary"):
        resolve_model_inputs(stage3_model_input_root, stage3_matrix_path)


def test_build_annotation_skeleton_emits_all_matrix_metabolites(stage3_matrix_path):
    annotation = build_metabolite_annotation_skeleton(stage3_matrix_path)
    assert {"metabolite_name", "review_status", "ambiguous_flag"}.issubset(annotation.columns)
    assert "Cholic acid (CA)" in set(annotation["metabolite_name"])
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py -k "resolve_model_inputs or annotation_skeleton" -v`
Expected: FAIL because the registry, membership, and annotation-resolution helpers are incomplete.

- [ ] **Step 3: Implement the minimal resolution and seeding helpers**

```python
def build_metabolite_annotation_skeleton(matrix_path: str | Path) -> pd.DataFrame:
    matrix = read_metabolite_matrix(matrix_path)
    return pd.DataFrame(
        {
            "metabolite_name": matrix.columns.tolist(),
            "superclass": "",
            "subclass": "",
            "pathway_tag": "",
            "annotation_source": "",
            "review_status": "",
            "ambiguous_flag": False,
            "notes": "",
        }
    )


def resolve_model_inputs(model_input_root: str | Path, matrix_path: str | Path) -> dict[str, pd.DataFrame]:
    mapping = load_stimulus_sample_map(Path(model_input_root) / "stimulus_sample_map.csv")
    annotation = load_metabolite_annotation(Path(model_input_root) / "metabolite_annotation.csv")
    registry = load_model_registry(Path(model_input_root) / "model_registry.csv")
    membership = load_model_membership(Path(model_input_root) / "model_membership.csv")
    return _resolve_stage3_inputs(mapping, annotation, registry, membership, matrix_path)
```

Implementation requirements:

- validate required columns and stable dtypes for all four CSV inputs
- require unique `stimulus`, `stim_name`, and `sample_id` rows in the mapping
- require all mapped `sample_id` values to exist in the matrix
- keep `global_profile` resolvable even when membership rows are empty
- prevent broad union models from entering the primary family
- expose a helper that can seed `metabolite_annotation.csv` from `matrix.xlsx` without running the full RSA pipeline

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py -k "resolve_model_inputs or annotation_skeleton" -v`
Expected: PASS on the new input-resolution and annotation-skeleton tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py
git commit -m "feat: validate stage 3 model inputs"
```

## Task 3: Implement Feature Preprocessing And Model RDM Construction

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`

- [ ] **Step 1: Write failing tests for feature filtering, scaling, and model RDM construction**

```python
def test_build_model_feature_matrix_drops_zero_variance_columns(stage3_resolved_inputs):
    feature_matrix, qc = build_model_feature_matrix(stage3_resolved_inputs, model_id="global_profile")
    assert "zero_variance" in set(qc["filter_reason"])
    assert feature_matrix.shape[0] == 3


def test_build_model_rdm_keeps_fixed_stimulus_order(stage3_resolved_inputs):
    matrix_frame = build_model_rdm(stage3_resolved_inputs, model_id="bile_acid")
    assert matrix_frame["stimulus_row"].tolist() == ["b1_1", "b2_1", "b3_1"]
```

```python
def test_binary_presence_model_uses_jaccard_distance(stage3_resolved_inputs):
    matrix_frame = build_model_rdm(stage3_resolved_inputs, model_id="presence_profile")
    assert matrix_frame.shape[0] == 3
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py -k "feature_matrix or build_model_rdm or jaccard" -v`
Expected: FAIL because feature preprocessing and model-RDM builders are not implemented yet.

- [ ] **Step 3: Implement the minimal preprocessing and model-RDM helpers**

```python
def build_model_feature_matrix(resolved_inputs: dict[str, pd.DataFrame], model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _select_model_columns(resolved_inputs, model_id)
    transformed = np.log1p(raw.astype(float))
    filtered, qc = _drop_and_record_uninformative_features(transformed)
    standardized = _zscore_columns(filtered)
    return standardized, qc


def build_model_rdm(resolved_inputs: dict[str, pd.DataFrame], model_id: str) -> pd.DataFrame:
    feature_matrix, feature_qc = build_model_feature_matrix(resolved_inputs, model_id=model_id)
    return _pairwise_distance_matrix(feature_matrix, distance_kind=_model_distance_kind(resolved_inputs, model_id))
```

Implementation requirements:

- continuous-abundance models use `log1p`, zero-variance drop, then column-wise standardization
- binary-presence models use an explicit threshold and preserve the threshold in QC
- `global_profile` uses all matrix columns after preprocessing
- subset models use explicit membership rows only
- primary models with fewer than `5` retained informative features should be marked `excluded_from_primary_ranking = True`
- every model should produce both a matrix frame and feature-level QC rows

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py -k "feature_matrix or build_model_rdm or jaccard" -v`
Expected: PASS on continuous-abundance, binary-presence, fixed-order, and feature-filtering tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py
git commit -m "feat: build stage 3 model rdms"
```

## Task 4: Implement RSA Statistics, Permutation Nulls, And Robustness Summaries

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`

- [ ] **Step 1: Write failing tests for RSA scoring, FDR correction, and leave-one-stimulus-out summaries**

```python
def test_compute_rsa_score_uses_shared_upper_triangle_entries_only():
    result = compute_rsa_score(neural_matrix_frame, model_matrix_frame)
    assert result["score_status"] == "ok"
    assert result["n_shared_entries"] == 3


def test_benjamini_hochberg_returns_monotonic_adjusted_values():
    adjusted = benjamini_hochberg(np.array([0.01, 0.03, 0.20]))
    assert np.all(np.diff(adjusted) >= 0)


def test_leave_one_stimulus_out_summary_records_excluded_stimulus():
    summary = summarize_leave_one_stimulus_out(neural_matrix_frame, model_matrix_frame)
    assert set(summary["excluded_stimulus"]) == {"b1_1", "b2_1", "b3_1"}
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -v`
Expected: FAIL because `rsa.py` and the Stage 3 RSA helpers do not exist yet.

- [ ] **Step 3: Implement the minimal RSA helpers**

```python
def compute_rsa_score(neural_matrix: pd.DataFrame, model_matrix: pd.DataFrame) -> dict[str, object]:
    shared = align_rdm_upper_triangles(neural_matrix, model_matrix)
    return _score_spearman(shared)


def build_permutation_null(neural_matrix: pd.DataFrame, model_matrix: pd.DataFrame, n_iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _permute_model_labels_and_score(neural_matrix, model_matrix, rng, n_iterations=n_iterations)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    return _bh_adjust(p_values)
```

Implementation requirements:

- align by `stimulus_row` labels, not by row position alone
- use rank-transform plus `np.corrcoef` for Spearman to avoid unnecessary new dependencies
- record `rsa_similarity`, `score_status`, `n_shared_entries`, `p_value_raw`, and `p_value_fdr`
- summarize leave-one-stimulus-out for every included stimulus and model
- compute cross-view comparison rows for `response_window` and `full_trajectory`

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -v`
Expected: PASS on RSA scoring, null, FDR, and leave-one-stimulus-out tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: add stage 3 rsa statistics"
```

## Task 5: Implement Output Writers, Figures, And Run Summary

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`

- [ ] **Step 1: Write failing tests for Stage 3 output paths and run summary**

```python
def test_write_stage3_outputs_writes_required_tables(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    assert (written["tables_dir"] / "rsa_results.parquet").exists()
    assert (written["tables_dir"] / "model_registry_resolved.parquet").exists()
    assert (written["figures_dir"] / "ranked_primary_model_rsa.png").exists()


def test_write_stage3_outputs_records_primary_and_supplementary_models(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["primary_models"] == ["global_profile", "bile_acid"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "write_stage3_outputs or run_summary" -v`
Expected: FAIL because `rsa_outputs.py` and the Stage 3 writer functions do not exist yet.

- [ ] **Step 3: Implement the minimal Stage 3 output helpers**

```python
def ensure_stage3_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    return _mkdir_stage3_dirs(root)


def write_stage3_outputs(core_outputs: dict[str, pd.DataFrame], output_root: str | Path) -> dict[str, Path]:
    dirs = ensure_stage3_output_dirs(output_root)
    return _write_stage3_artifacts(core_outputs, dirs)
```

Implementation requirements:

- write all required `tables/`, `figures/`, and `qc/` artifacts
- keep resolved input tables and RSA result tables distinct
- write figures for:
  - ranked primary-model RSA summary
  - neural-versus-top-model RDM comparison panel
  - leave-one-stimulus-out robustness summary
  - `response_window` versus `full_trajectory` comparison
- write `run_summary.json` and `run_summary.md`
- make the summary explicitly list primary versus supplementary models and any excluded models

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -k "write_stage3_outputs or run_summary" -v`
Expected: PASS on output-path and run-summary tests.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py
git commit -m "feat: add stage 3 rsa outputs"
```

## Task 6: Implement The Thin CLI And End-To-End Smoke Test

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
def test_cli_runs_and_writes_stage3_outputs(tmp_path, stage3_fixture_root):
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_rsa.py",
            "--stage2-root",
            str(stage3_fixture_root / "stage2_geometry"),
            "--matrix",
            str(stage3_fixture_root / "matrix.xlsx"),
            "--model-input-root",
            str(stage3_fixture_root / "model_space"),
            "--output-root",
            str(tmp_path / "results"),
            "--permutations",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -v`
Expected: FAIL because `scripts/run_rsa.py` does not exist yet.

- [ ] **Step 3: Implement the minimal CLI**

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 pooled biochemical RSA from Stage 2 pooled RDM outputs.")
    parser.add_argument("--stage2-root", default="results/stage2_geometry", help="Stage 2 output root containing pooled RDM parquet files.")
    parser.add_argument("--matrix", default="data/matrix.xlsx", help="Path to the metabolite matrix workbook.")
    parser.add_argument("--model-input-root", default="data/model_space", help="Directory containing Stage 3 model input CSVs.")
    parser.add_argument("--output-root", default="results", help="Base directory for Stage 3 outputs.")
    parser.add_argument("--permutations", type=int, default=1000, help="Number of stimulus-label permutations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for permutation and robustness summaries.")
    return parser.parse_args(argv)
```

Implementation requirements:

- load pooled `response_window` and `full_trajectory` neural RDMs from `results/stage2_geometry/tables/`
- load and resolve model-space inputs from `data/model_space/`
- run RSA across all included models
- write outputs to `results/stage3_rsa/`
- print included primary models and key output paths

- [ ] **Step 4: Run the smoke test and targeted unit tests to verify they pass**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -v`
Expected: PASS with created Stage 3 outputs under the temporary `results/` directory.

- [ ] **Step 5: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_space.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/rsa_outputs.py
git commit -m "feat: add stage 3 rsa cli"
```

## Task 7: Seed Real-Data Model Inputs And Run The Full Regression Pass

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/stimulus_sample_map.csv`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/metabolite_annotation.csv`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/model_registry.csv`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/data/model_space/model_membership.csv`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py`

- [ ] **Step 1: Add regression tests for draft versus excluded versus ranked models**

```python
def test_primary_model_with_too_few_features_is_excluded_from_primary_ranking(stage3_resolved_inputs):
    results = run_stage3_rsa(stage3_resolved_inputs, neural_rdms=stage3_neural_rdms, permutations=10, seed=0)
    excluded = results["model_registry_resolved"].loc[lambda df: df["model_id"] == "tiny_primary"]
    assert excluded["excluded_from_primary_ranking"].iloc[0]
```

```python
def test_global_profile_can_run_when_curated_subset_membership_is_empty(stage3_global_only_inputs):
    results = run_stage3_rsa(stage3_global_only_inputs, neural_rdms=stage3_neural_rdms, permutations=10, seed=0)
    assert "global_profile" in set(results["rsa_results"]["model_id"])
```

- [ ] **Step 2: Seed the real-data scientific input templates**

Create `stimulus_sample_map.csv` with the reviewed `19`-row mapping, for example:

```csv
stimulus,stim_name,sample_id
b10_1,A216 stationary,A216
b15_0,A085 stationary,A085
b19_0,A186 stationary,A186
b20_0,A097 stationary,A097
```

Seed `model_registry.csv` with at least:

```csv
model_id,model_label,model_tier,model_status,feature_kind,distance_kind,description,authority,notes
global_profile,Global Metabolite Profile,primary,primary,continuous_abundance,correlation,All metabolite columns after preprocessing,user,Always included
bile_acid,Bile Acid Model,primary,draft,continuous_abundance,correlation,Curated bile acid family,user,Needs reviewed membership
broad_aromatic_union,Broad Aromatic Union,supplementary,supplementary,continuous_abundance,correlation,Exploratory broad aromatic grouping,exploratory,Supplement only
```

Seed `metabolite_annotation.csv` from the matrix headers using the helper from Task 2, then review it outside the neural results loop.

- [ ] **Step 3: Run the full targeted Stage 3 test suite**

Run: `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -v`
Expected: PASS on all Stage 3 model-space, RSA, and CLI smoke tests.

- [ ] **Step 4: Run the real-data dry run**

Run: `pixi run python H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py --stage2-root H:/Process_temporary/WJH/bacteria_analysis/results/stage2_geometry --matrix H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx --model-input-root H:/Process_temporary/WJH/bacteria_analysis/data/model_space --output-root H:/Process_temporary/WJH/bacteria_analysis/results`
Expected: exit code `0`, `results/stage3_rsa/` created, `rsa_results.parquet` written, and draft or underpowered models surfaced clearly rather than silently ranked.

- [ ] **Step 5: Inspect the run summary and top-model table**

Run: `Get-Content -Raw H:/Process_temporary/WJH/bacteria_analysis/results/stage3_rsa/run_summary.json`
Expected: the JSON lists the primary neural view as `response_window`, lists included and excluded models separately, and points to the expected `tables/`, `figures/`, and `qc/` directories.

- [ ] **Step 6: Commit**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/data/model_space/stimulus_sample_map.csv H:/Process_temporary/WJH/bacteria_analysis/data/model_space/metabolite_annotation.csv H:/Process_temporary/WJH/bacteria_analysis/data/model_space/model_registry.csv H:/Process_temporary/WJH/bacteria_analysis/data/model_space/model_membership.csv H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py
git commit -m "test: validate stage 3 rsa pipeline"
```

## Implementation Notes

- Reuse `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/io.py` write helpers instead of inventing new filesystem utilities.
- Reuse the Stage 2 pooled RDM parquet contract exactly: `rdm_matrix__response_window__pooled.parquet` and `rdm_matrix__full_trajectory__pooled.parquet`.
- Keep the neural side fixed. Do not add new neural distance families or grouped neural RSA in the MVP.
- Keep `global_profile` as the always-runnable baseline model.
- Do not silently seed or modify `data/model_space/*.csv` during the main CLI run. Seeding should be explicit and reviewable.
- Use CSV for editable scientific inputs and parquet for resolved runtime outputs.
- Prefer pure-numpy or pandas implementations for Spearman and FDR to keep dependency expansion minimal beyond `openpyxl`.
- If the real-data curated subset membership is not yet reviewed, the pipeline must still run, but it must rank only the eligible primary models and mark draft or underpowered models explicitly.

## Final Validation Checklist

- [ ] `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_space.py -v`
- [ ] `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa.py -v`
- [ ] `pixi run pytest H:/Process_temporary/WJH/bacteria_analysis/tests/test_rsa_cli_smoke.py -v`
- [ ] `pixi run python H:/Process_temporary/WJH/bacteria_analysis/scripts/run_rsa.py --stage2-root H:/Process_temporary/WJH/bacteria_analysis/results/stage2_geometry --matrix H:/Process_temporary/WJH/bacteria_analysis/data/matrix.xlsx --model-input-root H:/Process_temporary/WJH/bacteria_analysis/data/model_space --output-root H:/Process_temporary/WJH/bacteria_analysis/results`
- [ ] Inspect `H:/Process_temporary/WJH/bacteria_analysis/results/stage3_rsa/run_summary.json`
- [ ] Inspect one top-model row in `H:/Process_temporary/WJH/bacteria_analysis/results/stage3_rsa/tables/rsa_results.parquet`
- [ ] Inspect `H:/Process_temporary/WJH/bacteria_analysis/results/stage3_rsa/figures/ranked_primary_model_rsa.png`

## Handoff

When this plan is complete, the next human review should focus on:

- whether the resolved `data/model_space/` inputs match the intended scientific hypotheses
- whether `global_profile` versus curated subset models tell a coherent story
- whether any apparent top model survives permutation and leave-one-stimulus-out checks
- whether `response_window` and `full_trajectory` give materially consistent model rankings
