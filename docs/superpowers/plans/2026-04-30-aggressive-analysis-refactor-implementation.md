# Aggressive Analysis Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a function-first analysis workflow for anchor batch-effect review, neural-chemical RDM alignment, and chemical-class RSA without depending on Stage 1/2/3 or exploratory review outputs.

**Architecture:** Add a new small analysis layer under `src/bacteria_analysis/` that reads raw neural parquet plus matrix/metadata inputs, builds neural and chemical RDMs in memory, returns inspectable result objects, and writes only explicit final outputs. Keep legacy scripts as reference only; do not make the new code import generated result directories.

**Tech Stack:** Python 3.11, pandas, numpy, scipy-free statistics where possible, matplotlib, pytest, existing project fixtures and raw parquet/matrix conventions.

---

## Scientific Contract

The implementation must follow these decisions from
`docs/superpowers/specs/2026-04-30-aggressive-analysis-refactor-design.md`:

- `build_analysis_dataset` reads directly from raw neural parquet by default.
- Anchor stimuli use a separate `AnchorDataset` or equivalent input object.
- New APIs are Python functions, not CLI-first commands.
- Default execution does not write large intermediate outputs.
- `date-stratified rank RSA` is not part of the new core workflow. It is a post-hoc sensitivity analysis with an unclear primary scientific estimand and should not be implemented unless a future explicit question requests it.
- Primary RDM alignment summaries are all-pair RSA, within-date RSA, cross-date RSA, sample/stimulus stability, and null context.
- Within-date and cross-date RSA are date-structure summaries, not clean controls, because stimulus identity and date are confounded.
- Final aligned RDM matrices used for headline comparisons are compact reported outputs and should be saved by default when saving a result; non-reported candidate RDMs and provenance/audit records remain opt-in.
- `scripts/run_rsa.py` may be frozen after the notebook workflow is validated.

## File Structure

Current target layout after the 2026-05-13 method-boundary cleanup:

```text
src/bacteria_analysis/
  io.py
  preprocessing.py
  features/
    neural.py
    chemical.py
    taxonomy.py
    anchor.py
  analyses/
    rdm/
      core.py
      builders.py
      stats.py
      plots.py
      neural_chemical.py
      chemical_class.py
      anchor_batch.py
```

`features/` owns reusable feature tables and metadata. `analyses/rdm/` owns RDM
building, RDM comparison, shuffle/resample statistics, RDM-specific plotting,
and the maintained RDM analysis entry points.

Create:

- `src/bacteria_analysis/analysis_dataset.py`
  - Load raw neural parquet, metabolite matrix, metabolite metadata, date filters, stimulus/sample mapping, and separate anchor datasets.
- `src/bacteria_analysis/neural_features.py`
  - Build current neural reference features and RDMs from raw neural rows.
- `src/bacteria_analysis/chemical_features.py`
  - Build full chemical and taxonomy-class chemical RDMs from `matrix.xlsx` and `metabolism_raw_data.xlsx`.
- `src/bacteria_analysis/rdm.py`
  - Shared RDM alignment, upper-triangle extraction, and similarity utilities.
- `src/bacteria_analysis/stats.py`
  - Shared permutation, empirical p-value, and sample/stimulus stability helpers.
- `src/bacteria_analysis/analysis_results.py`
  - Result dataclasses, reported RDM outputs, optional source manifests, and explicit save behavior.
- `src/bacteria_analysis/analysis_plotting.py`
  - Optional shared plotting helpers.
- `src/bacteria_analysis/analysis_plot_scripts.py`
  - Thin loader for existing useful review plotting scripts.
- `src/bacteria_analysis/analyses/__init__.py`
  - Public analysis exports.
- `src/bacteria_analysis/analyses/anchor_batch_effect.py`
  - Anchor-stimulus batch/date-effect analysis.
- `src/bacteria_analysis/analyses/rdm_alignment.py`
  - Full neural-chemical RDM comparison and significance/stability analysis.
- `src/bacteria_analysis/analyses/chemical_class_rsa.py`
  - Taxonomy class RDM scoring, significance, and stability.
- `tests/test_analysis_dataset.py`
- `tests/test_neural_features.py`
- `tests/test_chemical_features.py`
- `tests/test_rdm_utils.py`
- `tests/test_stats_utils.py`
- `tests/test_analysis_results.py`
- `tests/test_anchor_batch_effect.py`
- `tests/test_rdm_alignment_analysis.py`
- `tests/test_chemical_class_rsa_analysis.py`
- `notebook/anchor_batch_effect.ipynb`
- `notebook/rdm_alignment.ipynb`
- `notebook/chemical_class_rsa.ipynb`
- `docs/legacy-analysis-notes.md`

Modify:

- `src/bacteria_analysis/__init__.py`
  - Optionally expose the new high-level functions.
- `docs/current-neural-chemical-rsa-summary-2026-04-24.md`
  - Add a short note pointing future work to the new workflow after it exists.

Do not modify in the first implementation:

- `scripts/run_rsa.py`
- legacy review scripts under `scripts/`
- existing reliability/geometry/RSA modules except to reuse stable helper logic by copying or importing pure functions when appropriate.

## Task 1: Dataset Foundation

**Files:**
- Create: `src/bacteria_analysis/analysis_dataset.py`
- Test: `tests/test_analysis_dataset.py`

- [ ] **Step 1: Write failing tests for raw dataset loading**

Create a tiny raw-neural fixture DataFrame in the test with columns matching the current raw parquet contract used by `data/202604/202604_data.parquet`. Include at least:

- `date`
- `stimulus`
- `stim_name`
- `worm_key`
- `segment_index`
- a minimal set of neuron/time columns or rows matching the project's raw neural format

Test expectations:

- `build_analysis_dataset(...)` reads raw neural rows from parquet.
- `exclude_dates=["20260331"]` removes that date.
- returned metadata records included and excluded dates.
- no preprocess root is required.

Run:

```bash
pixi run pytest tests/test_analysis_dataset.py -q
```

Expected: fail because `analysis_dataset.py` does not exist.

- [ ] **Step 2: Write failing tests for separate anchor dataset loading**

Add tests for:

```python
anchor = build_anchor_dataset(
    neural_path=anchor_path,
    anchor_stimuli=("s3_0", "s6_1"),
)
```

Expected behavior:

- only requested anchor stimuli are retained;
- anchor dataset can have a different raw file from the main dataset;
- date filters are local to the anchor dataset.

- [ ] **Step 3: Implement minimal dataset dataclasses**

Implement:

```python
@dataclass(frozen=True)
class AnalysisDataset:
    neural: pd.DataFrame
    matrix: pd.DataFrame
    metadata: pd.DataFrame
    stimulus_sample_map: pd.DataFrame
    included_dates: tuple[str, ...]
    excluded_dates: tuple[str, ...]
    parameters: dict[str, object]

@dataclass(frozen=True)
class AnchorDataset:
    neural: pd.DataFrame
    anchor_stimuli: tuple[str, ...]
    included_dates: tuple[str, ...]
    excluded_dates: tuple[str, ...]
    parameters: dict[str, object]
```

Implement:

- `build_analysis_dataset(...)`
- `build_anchor_dataset(...)`
- small internal helpers for date filtering and string normalization.

Reuse existing `read_metabolite_matrix` from `src/bacteria_analysis/model_space.py` if it cleanly fits.

- [ ] **Step 4: Run dataset tests**

Run:

```bash
pixi run pytest tests/test_analysis_dataset.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/bacteria_analysis/analysis_dataset.py tests/test_analysis_dataset.py
git commit -m "feat: add analysis dataset loaders"
```

## Task 2: Shared RDM and Statistics Utilities

**Files:**
- Create: `src/bacteria_analysis/rdm.py`
- Create: `src/bacteria_analysis/stats.py`
- Test: `tests/test_rdm_utils.py`
- Test: `tests/test_stats_utils.py`

- [ ] **Step 1: Write failing RDM utility tests**

Cover:

- shared-label alignment preserves label order from the neural RDM by default;
- upper-triangle extraction excludes the diagonal;
- non-shared labels are dropped explicitly;
- Spearman similarity returns expected values for simple arrays;
- NaN behavior is explicit and tested.

- [ ] **Step 2: Implement RDM utilities**

Implement public functions:

```python
align_square_rdms(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]
upper_triangle_values(matrix: pd.DataFrame) -> pd.Series
rdm_pair_values(neural: pd.DataFrame, chemical: pd.DataFrame) -> pd.DataFrame
spearman_similarity(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float
pearson_similarity(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float
rank_normalize(values: np.ndarray | pd.Series) -> np.ndarray
```

Do not implement date-stratified rank RSA.

- [ ] **Step 3: Write failing stats tests**

Cover:

- empirical one-sided p-value;
- label-shuffle null reproducibility with a fixed seed;
- sample/stimulus subset resampling reproducibility;
- date-preserving label shuffle keeps date counts unchanged.

- [ ] **Step 4: Implement stats utilities**

Implement:

```python
empirical_p_value(observed: float, null_values: np.ndarray, *, side: str = "greater") -> float
label_shuffle_null(neural: pd.DataFrame, chemical: pd.DataFrame, *, n_permutations: int, seed: int) -> np.ndarray
date_preserving_label_shuffle_null(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    date_map: Mapping[str, str],
    *,
    n_permutations: int,
    seed: int,
) -> np.ndarray
stimulus_subset_rsa(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    *,
    subset_count: int,
    subset_fraction: float,
    seed: int,
) -> pd.DataFrame
```

Keep full null arrays in memory; saving is handled later.

- [ ] **Step 5: Run utility tests**

```bash
pixi run pytest tests/test_rdm_utils.py tests/test_stats_utils.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/bacteria_analysis/rdm.py src/bacteria_analysis/stats.py tests/test_rdm_utils.py tests/test_stats_utils.py
git commit -m "feat: add rdm and stats utilities"
```

## Task 3: Neural Feature and RDM Construction

**Files:**
- Create: `src/bacteria_analysis/neural_features.py`
- Test: `tests/test_neural_features.py`

- [ ] **Step 1: Write failing tests for L/R merge and trial aggregation**

Use a small synthetic raw neural table with paired neurons such as `ADFL/ADFR`
and unmerged neurons such as `ASEL/ASER`.

Test expectations:

- raw neural validation mirrors `preprocessing.validate_input_dataframe`;
- all-NaN traces are removed before feature construction;
- baseline centering mirrors `preprocessing.center_by_baseline`;
- trial IDs mirror `preprocessing.add_trial_id`;
- non-ASE L/R neurons merge by mean;
- `ASEL` and `ASER` remain separate;
- trial aggregation supports `median` and `mean`;
- `response_window` and `full_trajectory` select expected timepoints;
- output labels are stimulus IDs.

- [ ] **Step 2: Implement neural feature helpers**

Implement:

```python
build_trial_feature_matrix(dataset_or_frame, *, view: str, merge_lr: bool = True) -> pd.DataFrame
build_stimulus_prototypes(features: pd.DataFrame, *, aggregation: str = "median") -> pd.DataFrame
build_neural_rdm(dataset, *, view: str, aggregation: str = "median", merge_lr: bool = True, distance: str = "correlation") -> RdmResult
```

Define a small `RdmResult` dataclass either here or in `analysis_results.py`.

The raw-to-neural feature contract is scientifically important. Reuse or mirror
the semantics of these existing preprocessing helpers:

- `preprocessing.validate_input_dataframe`
- `preprocessing.add_trial_id`
- `preprocessing.annotate_trace_quality`
- `preprocessing.filter_traces`
- `preprocessing.center_by_baseline`
- `preprocessing.build_trial_metadata`
- `preprocessing.build_trial_tensor`

The new API should not require preprocess output files, but it should reproduce
the current reference preprocessing behavior from raw parquet. Add a real-data
smoke comparison, outside default unit tests if needed, that compares the new
neural RDM against the legacy current-reference RDM within the agreed tolerance.

- [ ] **Step 3: Run neural feature tests**

```bash
pixi run pytest tests/test_neural_features.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/bacteria_analysis/neural_features.py tests/test_neural_features.py
git commit -m "feat: build neural rdms from raw data"
```

## Task 4: Chemical Feature and RDM Construction

**Files:**
- Create: `src/bacteria_analysis/chemical_features.py`
- Test: `tests/test_chemical_features.py`

- [ ] **Step 1: Write failing tests for QC filtering and transforms**

Use synthetic matrix and metadata frames.

Cover:

- `QCRSD <= 0.2` filtering uses fractional thresholds;
- `log2` transform is applied to retained matrix values;
- nonpositive values fail clearly before `log2`;
- Euclidean RDM has expected shape and labels;
- taxonomy class grouping respects `min_features`;
- missing taxonomy values are ignored.

- [ ] **Step 2: Implement chemical RDM APIs**

Implement:

```python
build_chemical_rdm(dataset, *, qc_threshold: float, transform: str, distance: str) -> RdmResult
build_chemical_class_rdms(
    dataset,
    *,
    taxonomy_level: str = "Class",
    qc_threshold: float = 0.2,
    min_features: int = 3,
    transform: str = "log2",
    distance: str = "euclidean",
) -> dict[str, RdmResult]
```

Return feature counts and retained metabolite names in metadata, but do not save candidate RDMs by default.

- [ ] **Step 3: Run chemical feature tests**

```bash
pixi run pytest tests/test_chemical_features.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/bacteria_analysis/chemical_features.py tests/test_chemical_features.py
git commit -m "feat: build chemical rdms from matrix metadata"
```

## Task 5: Result Objects and Save Policy

**Files:**
- Create: `src/bacteria_analysis/analysis_results.py`
- Test: `tests/test_analysis_results.py`

- [ ] **Step 1: Write failing tests for result saving**

Test:

- `AnalysisResult` stores parameters, summary, tables, figures, diagnostics, and debug tables.
- `AnalysisResult` stores final aligned RDM matrices used for headline comparisons.
- `AnalysisResult` stores compact audit metadata and audit tables.
- `save_analysis_result(..., include_debug=False)` writes only final summaries/tables/figures.
- default saving writes final aligned RDM matrices but does not write `audit/`.
- `include_audit=True` writes aligned stimulus order, retained feature lists, source manifest, seeds, permutation counts, and `n_pairs` by scope where present.
- `include_debug=True` writes debug tables under `debug/`.
- no files are written when analysis functions are called without save.

- [ ] **Step 2: Implement result dataclasses and save helper**

Implement:

```python
@dataclass
class AnalysisResult:
    analysis_id: str
    parameters: dict[str, object]
    summary: dict[str, object] | pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    rdms: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Figure | Callable[[Path], None]] = field(default_factory=dict)
    audit: dict[str, object | pd.DataFrame] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
    debug_tables: dict[str, pd.DataFrame] = field(default_factory=dict)

save_analysis_result(
    result: AnalysisResult,
    output_root: str | Path,
    *,
    include_debug: bool = False,
    include_audit: bool = False,
) -> dict[str, Path]
```

Write `summary.json`, `summary.md`, `parameters.json`, final tables, final
RDMs, final figures, optional audit artifacts, and optional debug tables.
When `include_audit=True`, `audit/source_manifest.json` should include source
paths, file hashes when cheap to compute, git commit when available, package
version when available, seed values, permutation counts, and key parameters.

- [ ] **Step 3: Run result tests**

```bash
pixi run pytest tests/test_analysis_results.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/bacteria_analysis/analysis_results.py tests/test_analysis_results.py
git commit -m "feat: add explicit analysis result saving"
```

## Task 6: RDM Alignment Analysis

**Files:**
- Create: `src/bacteria_analysis/analyses/__init__.py`
- Create: `src/bacteria_analysis/analyses/rdm_alignment.py`
- Create or update: `src/bacteria_analysis/analysis_plot_scripts.py`
- Test: `tests/test_rdm_alignment_analysis.py`

- [ ] **Step 1: Write failing tests for the high-level alignment result**

Use synthetic data where chemical and neural RDMs have known positive alignment.

Test expectations:

- `run_rdm_alignment(...)` returns `AnalysisResult`;
- summary contains all-pair RSA;
- summary contains within-date and cross-date RSA as descriptive date-structure summaries when date metadata are available;
- summary reports `n_pairs` by all-pair, within-date, and cross-date scope;
- audit can report date and date-pair coverage when `include_audit=True`;
- summary caveats state that cross-date RSA is a stress test, not proof of generalization;
- summary contains label-shuffle p-value;
- summary contains sample/stimulus stability summary;
- final aligned neural and chemical RDMs are stored in `result.rdms`;
- result does not contain date-stratified rank RSA fields.

- [ ] **Step 2: Implement analysis function**

Implement:

```python
run_rdm_alignment(
    dataset: AnalysisDataset,
    *,
    neural_view: str = "response_window",
    neural_aggregation: str = "median",
    chemical_qc_threshold: float = 0.2,
    chemical_transform: str = "log2",
    chemical_distance: str = "euclidean",
    permutations: int = 2000,
    subset_count: int = 200,
    subset_fraction: float = 0.8,
    seed: int = 0,
    include_debug: bool = False,
) -> AnalysisResult
```

Required summaries:

- `all_pairs_rsa`;
- `within_date_rsa`;
- `cross_date_rsa`;
- `n_pairs_all`;
- `n_pairs_within_date`;
- `n_pairs_cross_date`;
- `label_shuffle_p_value`;
- `date_preserving_p_value` when date labels allow it;
- `subset_rsa_median`;
- `subset_rsa_q01`;
- `subset_rsa_q99`.

Optional audit artifacts when `include_audit=True`:

- aligned stimulus order;
- date/date-pair coverage;
- retained metabolite list for the broad chemical RDM;
- source manifest;
- provenance records for final aligned neural and chemical RDM matrices.

Debug-only tables:

- pair-level RDM values;
- full null values;
- subset draw values.

- [ ] **Step 3: Wire the current RDM foundation figure logic**

Use the existing `plot_neural_chemical_rdm_foundation.py` plotting functions
as save-time figure writers. The default saved figures should be:

- label-shuffle RDM heatmap panel;
- label-shuffle distribution;
- label-shuffle fraction distribution;
- random-subset distribution.

- [ ] **Step 4: Run alignment tests**

```bash
pixi run pytest tests/test_rdm_alignment_analysis.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/bacteria_analysis/analyses/__init__.py src/bacteria_analysis/analyses/rdm_alignment.py src/bacteria_analysis/analysis_plot_scripts.py tests/test_rdm_alignment_analysis.py
git commit -m "feat: add rdm alignment analysis"
```

## Task 7: Anchor Batch-Effect Analysis

**Files:**
- Create: `src/bacteria_analysis/analyses/anchor_batch_effect.py`
- Modify: `src/bacteria_analysis/analysis_plot_scripts.py`
- Test: `tests/test_anchor_batch_effect.py`

- [ ] **Step 1: Write failing anchor analysis tests**

Use a synthetic anchor dataset with repeated anchors across dates.

Test expectations:

- coverage table reports anchor/date counts;
- same-anchor cross-date distances are summarized;
- stimulus-effect versus date-effect contrast is present;
- figure writers are returned, not saved;
- result can be saved through `save_analysis_result`.

- [ ] **Step 2: Implement anchor batch-effect analysis**

Implement:

```python
run_anchor_batch_effect(
    anchor_dataset: AnchorDataset,
    *,
    views: tuple[str, ...] = ("response_window", "full_trajectory"),
    aggregation: str = "median",
    seed: int = 0,
    include_debug: bool = False,
) -> AnalysisResult
```

Primary outputs:

- `anchor_stimulus_coverage`;
- `anchor_stimulus_cross_date_anchor_summary`;
- `anchor_stimulus_date_pair_same_vs_other_contrasts`;
- the current nine anchor-stimulus review figures.

Keep this focused on batch/date effect assessment. Do not add broad RSA logic here.

- [ ] **Step 3: Run anchor tests**

```bash
pixi run pytest tests/test_anchor_batch_effect.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/bacteria_analysis/analyses/anchor_batch_effect.py src/bacteria_analysis/analysis_plot_scripts.py tests/test_anchor_batch_effect.py
git commit -m "feat: add anchor batch effect analysis"
```

## Task 8: Chemical Class RSA Analysis

**Files:**
- Create: `src/bacteria_analysis/analyses/chemical_class_rsa.py`
- Modify: `src/bacteria_analysis/analysis_plot_scripts.py`
- Test: `tests/test_chemical_class_rsa_analysis.py`

- [ ] **Step 1: Write failing class RSA tests**

Use synthetic chemical class RDMs where one class is known to match the neural RDM.

Test expectations:

- observed class scores are ranked correctly;
- fixed-class permutation summary is produced;
- fixed-class summary includes adjusted p/q values when multiple classes are evaluated;
- reselection stability summary is produced;
- reselection reports date composition per draw or uses date-aware resampling;
- search-corrected diagnostic summary is produced only when debug outputs are requested and is marked diagnostic;
- top class metadata includes feature count;
- result saves only reported top-class/final-shortlist RDMs by default and does not write all class RDM matrices.

- [ ] **Step 2: Implement class RSA analysis**

Implement:

```python
run_chemical_class_rsa(
    dataset: AnalysisDataset,
    *,
    neural_rdm: pd.DataFrame | None = None,
    neural_view: str = "response_window",
    taxonomy_level: str = "Class",
    qc_threshold: float = 0.2,
    min_features: int = 3,
    fixed_permutations: int = 2000,
    resamples: int = 500,
    search_permutations: int = 2000,
    top_k: int = 5,
    seed: int = 0,
    include_debug: bool = False,
) -> AnalysisResult
```

If `neural_rdm` is `None`, build it from the dataset using the current default neural reference.

Required outputs:

- `observed_class_scores`;
- `fixed_class_permutation_summary`;
- `reselection_stability_summary`;
- `final_class_shortlist`;
- `class_vs_full_chemical_rdm_similarity`.

Scientific constraints:

- fixed-class permutation p-values should be adjusted across evaluated classes;
- reselection should either resample within date strata or report date
  composition per draw so stability is not silently driven by date imbalance;
- search-corrected results remain diagnostic and should not be merged into the
  fixed-class evidence layer;
- save the broad full-chemical RDM and reported top-class/final-shortlist RDMs,
  but keep all non-reported candidate RDMs debug-only.
- keep search-corrected diagnostics and class-to-class pairwise similarity
  tables debug-only unless a later analysis explicitly needs them.

The main figure contract should match the current useful taxonomy review figures:

- fixed-class permutation score plot;
- reselection stability plot;
- top-class neural/full-chemical/top-class RDM comparison;
- final shortlist scorecard;
- class chemical RDM similarity matrix.
- class-versus-full chemical RDM similarity.

- [ ] **Step 3: Run class RSA tests**

```bash
pixi run pytest tests/test_chemical_class_rsa_analysis.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/bacteria_analysis/analyses/chemical_class_rsa.py src/bacteria_analysis/analysis_plot_scripts.py tests/test_chemical_class_rsa_analysis.py
git commit -m "feat: add chemical class rsa analysis"
```

## Task 9: Public Exports and Notebook Entry Points

**Files:**
- Modify: `src/bacteria_analysis/__init__.py`
- Modify: `src/bacteria_analysis/analyses/__init__.py`
- Create: `notebook/anchor_batch_effect.ipynb`
- Create: `notebook/rdm_alignment.ipynb`
- Create: `notebook/chemical_class_rsa.ipynb`

- [ ] **Step 1: Export public APIs**

Expose:

```python
build_analysis_dataset
build_anchor_dataset
run_anchor_batch_effect
run_rdm_alignment
run_chemical_class_rsa
save_analysis_result
```

- [ ] **Step 2: Add thin notebooks**

Each notebook should:

- define paths and parameters in the first code cell;
- call the relevant function API;
- display summary tables and render figure writers inline;
- save outputs only in a final optional cell.

Do not put core scientific logic in notebooks.

- [ ] **Step 3: Smoke-test notebook imports**

Run a simple import command:

```bash
pixi run python -c "from bacteria_analysis import build_analysis_dataset; from bacteria_analysis.analyses import run_rdm_alignment; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/bacteria_analysis/__init__.py src/bacteria_analysis/analyses/__init__.py notebook/anchor_batch_effect.ipynb notebook/rdm_alignment.ipynb notebook/chemical_class_rsa.ipynb
git commit -m "docs: add notebook workflow entry points"
```

## Task 10: Legacy Notes and Workflow Freeze

**Files:**
- Create: `docs/legacy-analysis-notes.md`
- Modify: `docs/current-neural-chemical-rsa-summary-2026-04-24.md`

- [ ] **Step 1: Document legacy boundary**

Create `docs/legacy-analysis-notes.md` explaining:

- old Stage 1/2/3 scripts and review scripts remain audit/reference material;
- new work should use function-first APIs;
- generated review-output directories are not valid inputs for new analyses;
- `scripts/run_rsa.py` is frozen after the notebook workflow is validated.

- [ ] **Step 2: Add pointer from current summary**

Add a short note to `docs/current-neural-chemical-rsa-summary-2026-04-24.md`
that future reruns should use the new APIs once implemented.

- [ ] **Step 3: Commit**

```bash
git add docs/legacy-analysis-notes.md docs/current-neural-chemical-rsa-summary-2026-04-24.md
git commit -m "docs: mark legacy analysis boundary"
```

## Task 11: Validation and Real-Data Smoke

**Files:**
- Optional create: `tests/test_analysis_real_data_smoke.py`

- [ ] **Step 1: Run full unit suite**

```bash
pixi run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run small real-data smoke manually**

Run a short Python snippet that builds the filtered dataset and executes reduced-iteration analyses:

```bash
pixi run python - <<'PY'
from bacteria_analysis import build_analysis_dataset
from bacteria_analysis.analyses import run_rdm_alignment, run_chemical_class_rsa

dataset = build_analysis_dataset(
    neural_path="data/202604/202604_data.parquet",
    matrix_path="data/matrix.xlsx",
    metadata_path="data/metabolism_raw_data.xlsx",
    exclude_dates=["20260331"],
)

alignment = run_rdm_alignment(dataset, seed=1)
classes = run_chemical_class_rsa(
    dataset,
    neural_rdm=alignment.rdms["neural"],
    fixed_permutations=20,
    resamples=20,
    search_permutations=20,
    seed=1,
)

print(alignment.summary)
print(classes.summary)
PY
```

Expected:

- both functions return summaries;
- broad alignment RSA is close to the current filtered 202604 baseline under the agreed 99th-percentile/null-context tolerance;
- `Purine nucleosides` is recovered as the leading or expected top current class under deterministic observed-score ranking;
- no large output directories are created;
- no old review-output directory is read.

- [ ] **Step 3: Save one explicit output smoke**

Use `save_analysis_result` to write one temporary reduced-iteration run under
`results/202604_without_20260331/refactor_smoke/`.

Expected:

- `summary.md`, `summary.json`, `parameters.json`, final figures, and compact tables exist;
- no `debug/` directory exists unless `include_debug=True`.

- [ ] **Step 4: Commit final validation note if needed**

Only commit docs if validation reveals an important permanent note. Do not commit generated result outputs unless explicitly requested.

## Task 12: Scientific Review Gate

**Files:**
- Review: `docs/superpowers/specs/2026-04-30-aggressive-analysis-refactor-design.md`
- Review: `docs/superpowers/plans/2026-04-30-aggressive-analysis-refactor-implementation.md`

- [ ] **Step 1: Ask a scientific reviewer subagent to review the spec and plan**

Review focus:

- Are the three analysis layers scientifically coherent?
- Is removing date-stratified rank RSA justified?
- Are within-date and cross-date RSA framed with the right caveats?
- Does the output-minimization policy preserve enough auditability?
- Are the class-RSA significance and stability layers separated correctly?

- [ ] **Step 2: Revise plan/spec based on accepted review findings**

Apply only changes that improve scientific validity or implementation clarity.

- [ ] **Step 3: Commit review-driven revisions**

```bash
git add docs/superpowers/specs/2026-04-30-aggressive-analysis-refactor-design.md docs/superpowers/plans/2026-04-30-aggressive-analysis-refactor-implementation.md
git commit -m "docs: revise analysis refactor plan after scientific review"
```

Skip this commit if no revisions are needed.

## Final Acceptance Checklist

- [ ] The plan creates a callable Python workflow for the three main analyses.
- [ ] No new core analysis depends on Stage 1/2/3 outputs.
- [ ] No new core analysis depends on generated review-output directories.
- [ ] Default functions return in-memory result objects and do not write large intermediates.
- [ ] `date-stratified rank RSA` is absent from core implementation.
- [ ] Within-date and cross-date RSA are presented as direct date-structure summaries with the stimulus/date-confounding caveat.
- [ ] Class-RSA fixed significance, reselection stability, and search diagnostic remain separate evidence layers.
- [ ] Notebooks are thin wrappers around tested `src/` APIs.
- [ ] Full test suite passes.
