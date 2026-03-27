# Data Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible preprocessing pipeline that reads `data/data.parquet`, validates its structure, applies mild trace-level filtering plus baseline centering, and writes clean long-table, wide-table, tensor, metadata, and QC artifacts under `data/processed/`.

**Architecture:** Keep the implementation small and explicit. Put domain constants and validation rules in focused modules, keep all dataframe reshaping in a single preprocessing module, and make the CLI a thin orchestration layer. Outputs are trial-major and aligned through a single metadata table so both table-style and tensor-style downstream analyses can reuse the same preprocessing results.

**Tech Stack:** Python 3.11, pandas, numpy, pyarrow, pytest, standard library (`json`, `pathlib`)

---

## File Structure

- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/__init__.py`
  Responsibility: mark the package and expose a version string only if needed.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/constants.py`
  Responsibility: store neuron order, required columns, time grid, baseline window, and output-relative paths.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/io.py`
  Responsibility: read raw parquet, ensure output directories exist, and write parquet/npz/json/markdown outputs.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/preprocessing.py`
  Responsibility: schema validation, trial-id construction, trace QC flags, filtering, baseline centering, metadata/wide/tensor/QC builders.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_preprocessing.py`
  Responsibility: CLI entry point that wires loading, processing, writing, and terminal summary together.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`
  Responsibility: small synthetic dataframe fixtures for deterministic preprocessing tests.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_preprocessing.py`
  Responsibility: unit tests for validation, filtering, baseline centering, and output shaping.
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_cli_smoke.py`
  Responsibility: smoke test for the CLI on a tiny fixture parquet written to a temp directory.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml`
  Responsibility: add test dependency and convenient tasks for preprocessing and test execution.

## Task 1: Scaffold the Package and Tooling

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/__init__.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml`

- [ ] **Step 1: Write the failing smoke test expectation for the CLI command**

```python
def test_cli_smoke_help():
    assert True  # placeholder until CLI exists
```

- [ ] **Step 2: Add test tooling to Pixi**

Update `H:/Process_temporary/WJH/bacteria_analysis/pixi.toml` to include:

```toml
[tasks]
test = "pytest -q"
preprocess = "python scripts/run_preprocessing.py"

[dependencies]
pytest = ">=8.0,<9"
```

- [ ] **Step 3: Create the minimal package marker**

```python
"""Core package for bacteria_analysis."""
```

- [ ] **Step 4: Install and run the empty test suite**

Run: `pixi install`

Run: `pixi run pytest -q`

Expected: tests collect successfully or fail only because later files are still missing.

- [ ] **Step 5: Commit**

```bash
git add pixi.toml pixi.lock src/bacteria_analysis/__init__.py
git commit -m "chore: scaffold preprocessing package"
```

## Task 2: Define Constants and Synthetic Fixtures

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/constants.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/conftest.py`
- Test: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests for canonical constants**

```python
from bacteria_analysis.constants import BASELINE_TIMEPOINTS, EXPECTED_TIMEPOINTS, NEURON_ORDER

def test_expected_timepoints_cover_full_window():
    assert EXPECTED_TIMEPOINTS == tuple(range(45))

def test_baseline_window_matches_spec():
    assert BASELINE_TIMEPOINTS == (0, 1, 2, 3, 4, 5)

def test_neuron_order_has_22_entries():
    assert len(NEURON_ORDER) == 22
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: FAIL with import error for `bacteria_analysis.constants`.

- [ ] **Step 3: Implement the shared constants**

```python
REQUIRED_COLUMNS = (
    "neuron",
    "stimulus",
    "time_point",
    "delta_F_over_F0",
    "worm_key",
    "segment_index",
    "date",
    "stim_name",
    "stim_color",
)
EXPECTED_TIMEPOINTS = tuple(range(45))
BASELINE_TIMEPOINTS = (0, 1, 2, 3, 4, 5)
NEURON_ORDER = (
    "ADFL", "ADFR", "ADLL", "ADLR", "ASEL", "ASER", "ASGL", "ASGR",
    "ASHL", "ASHR", "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
    "AWAL", "AWAR", "AWBL", "AWBR", "AWCOFF", "AWCON",
)
```

- [ ] **Step 4: Add a deterministic fixture dataframe**

Create a fixture with:
- two trials
- one complete trace
- one fully-NaN trace
- one partially-NaN trace
- full `0..44` time grid

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: PASS for constant tests.

- [ ] **Step 6: Commit**

```bash
git add src/bacteria_analysis/constants.py tests/conftest.py tests/test_preprocessing.py
git commit -m "test: add preprocessing constants fixtures"
```

## Task 3: Implement Validation and Trial ID Construction

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/preprocessing.py`
- Test: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests for structural validation**

```python
from bacteria_analysis.preprocessing import add_trial_id, validate_input_dataframe

def test_add_trial_id_uses_date_worm_segment(sample_df):
    out = add_trial_id(sample_df)
    assert out["trial_id"].iloc[0] == "20260106__wormA__1"

def test_validate_rejects_missing_required_columns(sample_df):
    broken = sample_df.drop(columns=["stimulus"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_input_dataframe(broken)
```

- [ ] **Step 2: Add tests for trace completeness**

```python
def test_validate_rejects_broken_time_grid(sample_df):
    broken = sample_df[sample_df["time_point"] != 44]
    with pytest.raises(ValueError, match="45 unique time_point"):
        validate_input_dataframe(broken)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: FAIL because `preprocessing.py` does not exist yet.

- [ ] **Step 4: Implement minimal validation functions**

Implement:
- `add_trial_id(df: pd.DataFrame) -> pd.DataFrame`
- `validate_input_dataframe(df: pd.DataFrame) -> None`

Key behaviors:
- assert required columns exist
- assert global time grid covers `0..44`
- assert each observed `trial_id × stimulus × neuron` has 45 rows and 45 unique time points
- assert each trial maps to exactly one stimulus

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: PASS for validation tests.

- [ ] **Step 6: Commit**

```bash
git add src/bacteria_analysis/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add preprocessing validation and trial ids"
```

## Task 4: Implement Trace QC Flags, Filtering, and Baseline Centering

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/preprocessing.py`
- Test: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests for trace-level QC**

```python
from bacteria_analysis.preprocessing import annotate_trace_quality, filter_traces, center_by_baseline

def test_full_nan_trace_is_flagged(sample_df):
    annotated = annotate_trace_quality(sample_df)
    assert annotated.loc[annotated["neuron"] == "ADFR", "is_all_nan_trace"].all()

def test_filter_drops_fully_nan_trace_only(sample_df):
    annotated = annotate_trace_quality(sample_df)
    filtered = filter_traces(annotated)
    assert "ADFR" not in filtered["neuron"].unique()
    assert "ADLL" in filtered["neuron"].unique()
```

- [ ] **Step 2: Add failing tests for baseline centering**

```python
def test_baseline_centering_subtracts_trace_baseline(sample_df):
    centered = center_by_baseline(filter_traces(annotate_trace_quality(sample_df)))
    trace = centered[centered["neuron"] == "ADFL"].sort_values("time_point")
    assert trace["baseline_mean"].iloc[0] == pytest.approx(2.5)
    assert trace.loc[trace["time_point"] == 0, "dff_baseline_centered"].iloc[0] == pytest.approx(-2.5)

def test_missing_baseline_marks_baseline_invalid(sample_df_missing_baseline):
    centered = center_by_baseline(filter_traces(annotate_trace_quality(sample_df_missing_baseline)))
    assert not centered["baseline_valid"].all()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: FAIL because QC and centering functions are not implemented.

- [ ] **Step 4: Implement the minimal QC and centering logic**

Implement:
- `annotate_trace_quality(df)`
- `filter_traces(df)`
- `center_by_baseline(df)`

Required columns added:
- `is_all_nan_trace`
- `has_any_nan_trace`
- `n_valid_points`
- `n_valid_baseline_points`
- `baseline_mean`
- `baseline_valid`
- `dff_baseline_centered`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: PASS for QC and centering tests.

- [ ] **Step 6: Commit**

```bash
git add src/bacteria_analysis/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add trace qc filtering and baseline centering"
```

## Task 5: Build Trial Metadata, Wide Table, and Tensor Outputs

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/preprocessing.py`
- Test: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests for output builders**

```python
from bacteria_analysis.preprocessing import build_trial_metadata, build_trial_wide_table, build_trial_tensor

def test_trial_metadata_has_one_row_per_trial(processed_df):
    metadata = build_trial_metadata(processed_df)
    assert metadata["trial_id"].is_unique

def test_wide_table_has_trial_rows_and_neuron_time_columns(processed_df):
    metadata = build_trial_metadata(processed_df)
    wide = build_trial_wide_table(processed_df, metadata)
    assert "ADFL__t00" in wide.columns
    assert len(wide) == len(metadata)

def test_tensor_shape_matches_metadata_neurons_time(processed_df):
    metadata = build_trial_metadata(processed_df)
    tensor = build_trial_tensor(processed_df, metadata)
    assert tensor.shape == (len(metadata), 22, 45)
```

- [ ] **Step 2: Add test for row-order alignment**

```python
def test_tensor_and_metadata_share_trial_order(processed_df):
    metadata = build_trial_metadata(processed_df)
    tensor = build_trial_tensor(processed_df, metadata)
    assert tensor.shape[0] == len(metadata)
    assert metadata.iloc[0]["trial_id"].startswith("20260106__")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: FAIL because output builders are not implemented.

- [ ] **Step 4: Implement the output builders**

Implement:
- `build_trial_metadata(df)`
- `build_trial_wide_table(df, metadata)`
- `build_trial_tensor(df, metadata)`

Rules:
- metadata is the authority for trial ordering
- wide feature names use `{neuron}__t{time_point:02d}`
- tensor shape is `(n_trials, 22, 45)`
- missing neurons/timepoints remain `NaN`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: PASS for metadata, wide, and tensor tests.

- [ ] **Step 6: Commit**

```bash
git add src/bacteria_analysis/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: build trial metadata wide table and tensor outputs"
```

## Task 6: Add IO Helpers and QC Report Writers

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/io.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/preprocessing.py`
- Test: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests for serialization and QC summary**

```python
from bacteria_analysis.preprocessing import build_qc_report

def test_qc_report_counts_removed_and_retained_traces(processed_df):
    report = build_qc_report(raw_df=sample_df, processed_df=processed_df, metadata=metadata)
    assert report["n_unique_trials"] == len(metadata)
    assert report["n_fully_nan_traces_removed"] >= 1
```

- [ ] **Step 2: Add failing test for output directory creation**

```python
from bacteria_analysis.io import ensure_output_dirs

def test_ensure_output_dirs_creates_expected_tree(tmp_path):
    paths = ensure_output_dirs(tmp_path)
    assert paths["clean_dir"].exists()
    assert paths["trial_level_dir"].exists()
    assert paths["qc_dir"].exists()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: FAIL because IO and QC functions are not implemented.

- [ ] **Step 4: Implement IO and report generation**

Implement:
- `ensure_output_dirs(output_root)`
- `write_parquet(df, path)`
- `write_tensor_npz(path, tensor, trial_ids, stimulus_labels, stim_name_labels)`
- `write_json(obj, path)`
- `write_markdown_report(report, path)`
- `build_qc_report(raw_df, processed_df, metadata)`

QC report must include:
- input/output row counts
- unique trials, stimuli, neurons
- number of fully-NaN traces removed
- number of partially-NaN traces retained
- neuron coverage distribution
- trials-per-stimulus summary

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run pytest tests/test_preprocessing.py -q`

Expected: PASS for IO and QC tests.

- [ ] **Step 6: Commit**

```bash
git add src/bacteria_analysis/io.py src/bacteria_analysis/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add preprocessing io and qc reporting"
```

## Task 7: Wire the CLI End-to-End

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/run_preprocessing.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_cli_smoke.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/io.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/preprocessing.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
def test_cli_runs_and_writes_outputs(tmp_path, tiny_parquet_path):
    result = subprocess.run(
        ["pixi", "run", "python", "scripts/run_preprocessing.py", "--input", str(tiny_parquet_path), "--output-root", str(tmp_path / "processed")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert (tmp_path / "processed" / "clean" / "neuron_segments_clean.parquet").exists()
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `pixi run pytest tests/test_cli_smoke.py -q`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement the thin CLI**

CLI responsibilities:
- parse `--input` and `--output-root`
- load parquet
- run validation
- run preprocessing pipeline
- write all outputs
- print a short summary

Suggested command body:

```python
raw_df = read_parquet(input_path)
validated = add_trial_id(raw_df)
validate_input_dataframe(validated)
annotated = annotate_trace_quality(validated)
filtered = filter_traces(annotated)
centered = center_by_baseline(filtered)
metadata = build_trial_metadata(centered)
wide = build_trial_wide_table(centered, metadata)
tensor = build_trial_tensor(centered, metadata)
report = build_qc_report(raw_df, centered, metadata)
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `pixi run pytest tests/test_cli_smoke.py -q`

Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `pixi run pytest -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_preprocessing.py tests/test_cli_smoke.py src/bacteria_analysis/io.py src/bacteria_analysis/preprocessing.py
git commit -m "feat: add preprocessing cli"
```

## Task 8: Run the Pipeline on the Real Dataset and Verify Artifact Counts

**Files:**
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/preprocessing.py`
- Review outputs under: `H:/Process_temporary/WJH/bacteria_analysis/data/processed/`

- [ ] **Step 1: Execute preprocessing on the real dataset**

Run: `pixi run python scripts/run_preprocessing.py --input data/data.parquet --output-root data/processed`

Expected: SUCCESS with a short terminal summary.

- [ ] **Step 2: Verify output files exist**

Check:
- `data/processed/clean/neuron_segments_clean.parquet`
- `data/processed/trial_level/trial_metadata.parquet`
- `data/processed/trial_level/trial_wide_baseline_centered.parquet`
- `data/processed/trial_level/trial_tensor_baseline_centered.npz`
- `data/processed/qc/preprocessing_report.json`
- `data/processed/qc/preprocessing_report.md`

- [ ] **Step 3: Verify real-data invariants**

Expected:
- `trial_metadata.parquet` has `678` rows unless a later explicit rule changes this
- tensor shape is `(678, 22, 45)` unless a later explicit rule changes this
- fully-NaN traces removed equals current observed count of about `130`
- partially-NaN retained count equals current observed count of about `1`

- [ ] **Step 4: Fix any mismatch with the spec**

If output mismatches:
- inspect whether the bug is in validation, filtering, centering, or reshaping
- add a targeted regression test before changing implementation

- [ ] **Step 5: Re-run the full suite**

Run: `pixi run pytest -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add data/processed src/bacteria_analysis tests scripts
git commit -m "feat: generate standardized preprocessing artifacts"
```

## Task 9: Final Documentation Pass

**Files:**
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-03-27-data-preprocessing-design.md`
- Optionally create: `H:/Process_temporary/WJH/bacteria_analysis/docs/preprocessing-usage.md`

- [ ] **Step 1: Add a short usage document**

Include:
- command to run preprocessing
- artifact paths
- meaning of `trial_id`
- meaning of missing values in wide/tensor outputs

- [ ] **Step 2: Add a verification note**

Document current verified counts:
- 678 trials
- 22 neurons
- 45 timepoints
- mild filtering policy

- [ ] **Step 3: Run the CLI once more from a clean state**

Run: `pixi run python scripts/run_preprocessing.py --input data/data.parquet --output-root data/processed`

Expected: SUCCESS and identical artifact layout.

- [ ] **Step 4: Commit**

```bash
git add docs/preprocessing-usage.md docs/superpowers/specs/2026-03-27-data-preprocessing-design.md
git commit -m "docs: add preprocessing usage notes"
```

## Implementation Notes

- Keep `stimulus` as the true unique stimulus key. Do not collapse onto `stim_name`.
- Keep all reshaping trial-major. `trial_metadata.parquet` is the single authority for row order.
- Do not introduce interpolation or neuron imputation in this stage.
- Preserve NaNs in wide/tensor outputs.
- Prefer a few small pure functions over one large pipeline function.

## Verification Checklist

- `pixi run pytest -q`
- `pixi run python scripts/run_preprocessing.py --input data/data.parquet --output-root data/processed`
- Open the QC JSON/Markdown report and confirm counts match expectations
- Load the wide parquet and tensor metadata once to verify trial-order alignment

## Handoff

When this plan is executed successfully, the repository will have a stable preprocessing foundation for:

- split-half reliability checks
- time-resolved RDM construction
- RSA/model-RDM comparison
- neuron leave-one-out contribution analysis
