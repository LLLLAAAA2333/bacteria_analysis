# Model Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a focused diagnostic review package that estimates neural RDM reliability ceilings, selects reliable stimulus pairs, and diagnoses model residuals for filtered `202604_without_20260331`.

**Architecture:** Add one testable module for diagnostic math and data assembly, plus one thin script that runs the real review and writes ignored result artifacts. Keep the existing RSA command and production defaults unchanged; reuse existing preprocess, aggregated-response, model-space, and RSA helpers wherever they fit.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, parquet IO, openpyxl, pytest, existing `bacteria_analysis` modules. Do not add new dependencies; use numpy SVD for PCA.

---

## File Map

- `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py`
  Responsibility: core model-diagnosis helpers: RDM upper triangles, ranks, split-half ceiling, pair reliability tiers, current neural-reference construction, QC20 log2 Euclidean model RDMs, PCA embedding RDMs, residual tables, and date-controlled scoring.
- `H:/Process_temporary/WJH/bacteria_analysis/scripts/model_diagnosis_review.py`
  Responsibility: CLI wrapper for the filtered-batch review. It parses paths, calls `model_diagnosis.py`, writes tables, figures, QC files, and `run_summary.md`.
- `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py`
  Responsibility: unit and smoke coverage for diagnostic math and small synthetic end-to-end outputs. Tests must not depend on the full real dataset.

Generated but ignored outputs:

- `H:/Process_temporary/WJH/bacteria_analysis/results/202604_without_20260331/model_diagnosis`
  Responsibility: real-data diagnostic tables, figures, QC artifacts, and summary. Do not commit this directory because `results/` is ignored.

## Shared Constants and Naming

Use these defaults in the module and script:

```python
VIEW_NAMES = ("response_window", "full_trajectory")
DEFAULT_PREPROCESS_ROOT = Path("data/202604/202604_preprocess_without_20260331")
DEFAULT_MATRIX_PATH = Path("data/matrix.xlsx")
DEFAULT_RAW_METADATA_PATH = Path("data/metabolism_raw_data.xlsx")
DEFAULT_BASE_RESULTS_ROOT = Path("results/202604_without_20260331")
DEFAULT_OUTPUT_ROOT = DEFAULT_BASE_RESULTS_ROOT / "model_diagnosis"
DEFAULT_QC_THRESHOLD = 0.2
DEFAULT_SPLIT_ITERATIONS = 200
DEFAULT_PERMUTATIONS = 2000
DEFAULT_SEED = 20260420
KEEP_SEPARATE_NEURONS = {"ASEL", "ASER"}
PRIMARY_RELIABILITY_TIERS = {"high", "medium"}
```

Use stable model IDs:

```python
global_profile_default_correlation
global_qc20_log2_euclidean
best_weighted_fusion_fixed_weights
pca_qc20_log2_euclidean__k02
pca_qc20_log2_euclidean__k03
pca_qc20_log2_euclidean__k05
pca_qc20_log2_euclidean__k10
pca_qc20_log2_euclidean__k20
```

### Task 1: Add Core RDM, Rank, Ceiling, and Residual Math

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py`
- Create: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing tests for core math helpers**

Add `tests/test_model_diagnosis.py` with small synthetic RDMs:

```python
import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.model_diagnosis import (
    assign_reliability_tiers,
    compute_rank_residuals,
    extract_upper_triangle,
    pct_rank,
    spearman_brown,
    stratified_pct_rank,
)


def _rdm(labels, values):
    frame = pd.DataFrame(values, index=labels, columns=labels)
    frame.insert(0, "stimulus_row", frame.index.astype(str))
    return frame.reset_index(drop=True)


def test_extract_upper_triangle_returns_pairs_in_matrix_order():
    matrix = _rdm(["b", "a", "c"], [[0.0, 0.2, 0.4], [0.2, 0.0, 0.1], [0.4, 0.1, 0.0]])

    pairs = extract_upper_triangle(matrix, value_name="distance")

    assert pairs.to_dict("records") == [
        {"stimulus_left": "b", "stimulus_right": "a", "distance": 0.2},
        {"stimulus_left": "b", "stimulus_right": "c", "distance": 0.4},
        {"stimulus_left": "a", "stimulus_right": "c", "distance": 0.1},
    ]


def test_spearman_brown_handles_valid_and_invalid_values():
    assert spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)
    assert np.isnan(spearman_brown(np.nan))
    assert np.isnan(spearman_brown(-1.0))


def test_pct_rank_uses_average_rank_scaled_to_unit_interval():
    ranks = pct_rank(np.array([10.0, 30.0, 30.0, 20.0]))
    np.testing.assert_allclose(ranks, [0.125, 0.75, 0.75, 0.375])


def test_stratified_pct_rank_ranks_within_date_pair():
    frame = pd.DataFrame({"date_pair": ["d1", "d1", "d2", "d2"], "value": [2.0, 4.0, 100.0, 50.0]})

    ranks = stratified_pct_rank(frame, value_column="value", group_column="date_pair")

    np.testing.assert_allclose(ranks, [0.25, 0.75, 0.75, 0.25])
```

Also test tiering and residual directions:

```python
def test_assign_reliability_tiers_keeps_insufficient_pairs_out_of_primary_pool():
    pairs = pd.DataFrame(
        {
            "valid_split_count": [20, 20, 20, 2],
            "rank_stability": [0.95, 0.75, 0.40, 0.99],
            "split_distance_sd": [0.01, 0.03, 0.10, 0.01],
        }
    )

    tiered = assign_reliability_tiers(pairs, min_valid_splits=5)

    assert tiered["reliability_tier"].tolist() == ["high", "medium", "low", "insufficient"]


def test_compute_rank_residuals_labels_directions():
    neural = pd.DataFrame(
        {
            "stimulus_left": ["A", "A", "B"],
            "stimulus_right": ["B", "C", "C"],
            "date_pair": ["d1", "d1", "d2"],
            "neural_distance": [0.9, 0.2, 0.5],
            "reliability_tier": ["high", "medium", "low"],
        }
    )
    model = pd.DataFrame(
        {
            "stimulus_left": ["A", "A", "B"],
            "stimulus_right": ["B", "C", "C"],
            "model_distance": [0.1, 0.8, 0.5],
        }
    )

    residuals = compute_rank_residuals(neural, model, model_id="test_model", view_name="response_window")

    ab = residuals.loc[lambda df: df["stimulus_left"].eq("A") & df["stimulus_right"].eq("B")].iloc[0]
    ac = residuals.loc[lambda df: df["stimulus_left"].eq("A") & df["stimulus_right"].eq("C")].iloc[0]
    assert ab["residual_direction"] == "neural_far_model_near"
    assert ac["residual_direction"] == "neural_near_model_far"
```

- [ ] **Step 2: Run the focused test file to verify it fails**

Run: `pixi run pytest tests/test_model_diagnosis.py -q`

Expected: FAIL because `bacteria_analysis.model_diagnosis` does not exist.

- [ ] **Step 3: Implement core helpers in `model_diagnosis.py`**

Add:

```python
def prepare_square_rdm(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    ...


def extract_upper_triangle(matrix_frame: pd.DataFrame, *, value_name: str = "value") -> pd.DataFrame:
    ...


def score_values(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray, *, method: str = "spearman") -> float:
    ...


def pct_rank(values: pd.Series | np.ndarray) -> np.ndarray:
    ...


def stratified_pct_rank(frame: pd.DataFrame, *, value_column: str, group_column: str) -> np.ndarray:
    ...


def spearman_brown(split_half_r: float) -> float:
    if not np.isfinite(split_half_r) or split_half_r <= -1.0:
        return float("nan")
    return float((2.0 * split_half_r) / (1.0 + split_half_r))
```

Implement `compute_rank_residuals(...)` so it:

- merges neural and model pair tables on `stimulus_left`, `stimulus_right`
- computes global percentile ranks
- computes date-pair-stratified percentile ranks
- writes `residual` and `date_pair_stratified_residual`
- assigns direction using a small threshold, e.g. `0.10`

Implement `assign_reliability_tiers(...)` with quantile-based thresholds:

- insufficient when `valid_split_count < min_valid_splits`
- high when rank stability is at least the 75th percentile among supported pairs and split SD is no worse than the median
- medium when rank stability is at least the 50th percentile among supported pairs
- low otherwise

- [ ] **Step 4: Run the core tests**

Run: `pixi run pytest tests/test_model_diagnosis.py -q`

Expected: PASS for Task 1 tests.

- [ ] **Step 5: Commit core diagnostic helpers**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py
git commit -m "feat: add model diagnosis core math"
```

### Task 2: Build Current Neural Reference and Split-Half Reliability

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing tests for L/R merge and split RDM construction**

Add tests using synthetic `TrialView` objects:

```python
from bacteria_analysis.reliability import TrialView
from bacteria_analysis.model_diagnosis import (
    build_neural_split_diagnostics,
    merge_non_ase_lr_view,
)


def test_merge_non_ase_lr_view_keeps_asel_aser_separate_and_merges_other_pairs():
    metadata = pd.DataFrame({"trial_id": ["t1"], "stimulus": ["A"], "stim_name": ["A"], "date": ["d1"], "worm_key": ["w1"]})
    values = np.arange(22 * 3, dtype=float).reshape(1, 22, 3)
    view = TrialView(name="response_window", timepoints=(0, 1, 2), metadata=metadata, values=values)

    merged = merge_non_ase_lr_view(view)

    assert merged.values.shape[1] == 13
    assert np.allclose(merged.values[:, 0, :], np.nanmean(values[:, [0, 1], :], axis=1))
    assert np.allclose(merged.values[:, 3, :], values[:, 4, :])
    assert np.allclose(merged.values[:, 4, :], values[:, 5, :])


def test_build_neural_split_diagnostics_reports_ceiling_and_pair_reliability():
    context = _tiny_aggregated_context_with_replicates()

    diagnostics = build_neural_split_diagnostics(
        context,
        view_names=("response_window",),
        n_iterations=12,
        seed=123,
        min_valid_splits=3,
    )

    ceiling = diagnostics.ceiling_summary
    pairs = diagnostics.pair_reliability
    assert set(ceiling["split_kind"]) == {"trial", "individual"}
    assert {"split_half_r", "spearman_brown_ceiling"}.issubset(ceiling.columns)
    assert {"stimulus_left", "stimulus_right", "rank_stability", "reliability_tier"}.issubset(pairs.columns)
```

The `_tiny_aggregated_context_with_replicates()` helper should create at least three stimuli, two worms, and enough repeated trials per stimulus for both split kinds.

- [ ] **Step 2: Run the neural diagnostic tests to verify they fail**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "merge_non_ase_lr_view or neural_split" -q`

Expected: FAIL because neural split functions are not implemented.

- [ ] **Step 3: Implement neural-reference helpers**

Add:

```python
@dataclass(frozen=True)
class NeuralSplitDiagnostics:
    observed_rdms: dict[str, pd.DataFrame]
    observed_pair_distances: pd.DataFrame
    ceiling_summary: pd.DataFrame
    pair_reliability: pd.DataFrame
    pair_reliability_by_date_pair: pd.DataFrame
```

Implement:

- `_build_lr_merge_plan()` from `constants.NEURON_ORDER`, keeping `ASEL` and `ASER` separate
- `merge_non_ase_lr_view(view: TrialView) -> TrialView`
- `build_observed_neural_rdms(context, view_names) -> dict[str, pd.DataFrame]`
- `build_neural_split_diagnostics(...) -> NeuralSplitDiagnostics`

For split generation:

- trial split: shuffle trial indices within each stimulus and split half A/B
- individual split: use `individual_id = date + "__" + worm_key`; split individual IDs within each stimulus when at least two are present
- skip a stimulus in a split half if the half has no trials
- skip an iteration for ceiling scoring when fewer than two valid pair distances remain

For pair reliability:

- collect one row per split iteration, view, split kind, and pair with `distance_a`, `distance_b`, `mean_distance`, `abs_distance_delta`, `rank_a`, `rank_b`, `abs_rank_delta`
- summarize to one row per view, split kind, and pair
- compute `rank_stability = 1.0 - median(abs_rank_delta)` after ranks are scaled to `[0, 1]`
- compute split distance SD and central 95 percent interval
- assign reliability tiers per view and split kind, then derive an overall per-view tier by preferring the weaker of trial and individual tiers

Keep date split out of the main ceiling. Only attach `date_left`, `date_right`, `date_pair`, and `date_pair_type` using the observed stimulus-date map. For the first pass, fail clearly if a stimulus maps to more than one date.

- [ ] **Step 4: Run neural diagnostic tests**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "merge_non_ase_lr_view or neural_split" -q`

Expected: PASS.

- [ ] **Step 5: Commit neural split diagnostics**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py
git commit -m "feat: add neural RDM reliability diagnostics"
```

### Task 3: Build Chemical Baseline and PCA Embedding RDMs

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing tests for QC20 log2 Euclidean and PCA RDMs**

Add tests:

```python
from bacteria_analysis.model_diagnosis import (
    build_pca_embedding_rdms,
    build_qc20_log2_euclidean_rdm,
)


def test_build_qc20_log2_euclidean_rdm_filters_by_qcrsd_and_builds_distances():
    matrix = pd.DataFrame(
        {"m1": [1.0, 2.0, 4.0], "m2": [4.0, 2.0, 1.0], "m3": [10.0, 10.0, 10.0]},
        index=pd.Index(["A001", "A002", "A003"], name="sample_id"),
    )
    mapping = pd.DataFrame({"stimulus": ["s1", "s2", "s3"], "sample_id": ["A001", "A002", "A003"]})
    taxonomy_qc = pd.DataFrame({"normalized_name": ["m1", "m2", "m3"], "QCRSD": [0.2, 0.1, 0.4]})

    rdm, retained = build_qc20_log2_euclidean_rdm(matrix, mapping, taxonomy_qc, qc_threshold=0.2)

    assert retained["metabolite_name"].tolist() == ["m1", "m2"]
    assert rdm["stimulus_row"].tolist() == ["s1", "s2", "s3"]


def test_build_pca_embedding_rdms_uses_numpy_svd_without_neural_labels():
    feature_frame = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [4.0, 3.0, 2.0]],
        index=pd.Index(["s1", "s2", "s3", "s4"], name="stimulus"),
        columns=["m1", "m2", "m3"],
    )

    rdms, summary = build_pca_embedding_rdms(feature_frame, dimensions=(2, 3, 5))

    assert sorted(rdms) == ["pca_qc20_log2_euclidean__k02", "pca_qc20_log2_euclidean__k03"]
    assert summary["n_components"].tolist() == [2, 3]
```

- [ ] **Step 2: Run chemical model tests to verify they fail**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "qc20 or pca_embedding" -q`

Expected: FAIL because model builders are not implemented.

- [ ] **Step 3: Implement chemical model builders**

Add:

```python
def load_taxonomy_qc(raw_metadata_path: str | Path) -> pd.DataFrame:
    ...


def build_qc20_feature_frame(
    matrix: pd.DataFrame,
    stimulus_sample_map: pd.DataFrame,
    taxonomy_qc: pd.DataFrame,
    *,
    qc_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...


def build_qc20_log2_euclidean_rdm(...) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...


def build_pca_embedding_rdms(feature_frame: pd.DataFrame, *, dimensions: tuple[int, ...]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    ...
```

Implementation details:

- load raw metadata sheet with `RAW_METADATA_SHEET_NAME` from `model_space_seed`
- normalize metabolite names with `_normalize_header_text` from `model_space_seed`, matching existing local scripts
- retain matrix columns present in taxonomy QC and `QCRSD <= 0.2`
- coerce matrix values to numeric
- drop features with missing or nonpositive values before `np.log2`
- write a retained-feature summary with drop reasons
- build Euclidean distance by direct numpy broadcasting
- build PCA by z-scoring log2 features, using `np.linalg.svd`, and computing Euclidean distance on scores
- skip requested PCA dimensions greater than the matrix rank or feature count

- [ ] **Step 4: Add model RDM bundle builder**

Implement:

```python
@dataclass(frozen=True)
class ModelRdmBundle:
    model_rdms: dict[str, pd.DataFrame]
    model_summary: pd.DataFrame
    feature_qc: pd.DataFrame
    pca_summary: pd.DataFrame
```

`build_model_rdm_bundle(...)` should include:

- current `global_profile_default_correlation` built through `resolve_direct_global_profile_inputs(...)` plus `build_model_rdm(...)`
- `global_qc20_log2_euclidean` built from matrix and QCRSD metadata
- `best_weighted_fusion_fixed_weights` loaded from `results/202604_without_20260331/neural_common_response_residualization_review/model_rdm__best_weighted_fusion_fixed_weights.parquet` if present
- selected taxonomy comparator RDMs loaded from `neural_common_response_residualization_review/model_rdm__*.parquet` if present
- PCA embedding RDMs built from the QC20 log2 feature frame

Skip optional comparator RDMs when files are absent; record `status=missing` in `model_summary` rather than failing.

- [ ] **Step 5: Run chemical model tests**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "qc20 or pca_embedding or model_rdm_bundle" -q`

Expected: PASS.

- [ ] **Step 6: Commit chemical model builders**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py
git commit -m "feat: add model diagnosis chemical RDM builders"
```

### Task 4: Add Residual Tables and Date-Controlled Model Scoring

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing tests for residual summaries and scoped RSA**

Add tests:

```python
from bacteria_analysis.model_diagnosis import build_residual_diagnostics, score_model_scopes


def test_score_model_scopes_reports_reliable_and_date_pair_stratified_scores():
    pairs = pd.DataFrame(
        {
            "stimulus_left": ["A", "A", "B", "C"],
            "stimulus_right": ["B", "C", "C", "D"],
            "date_pair_type": ["within_date", "cross_date", "cross_date", "within_date"],
            "date_pair": ["d1|d1", "d1|d2", "d1|d2", "d2|d2"],
            "neural_distance": [0.1, 0.9, 0.8, 0.2],
            "model_distance": [0.2, 0.7, 0.6, 0.3],
            "reliability_tier": ["high", "medium", "low", "high"],
        }
    )

    summary = score_model_scopes(pairs, view_name="response_window", model_id="model")

    assert set(summary["scope"]) >= {
        "all_pairs",
        "reliable_pairs",
        "high_reliability_pairs",
        "within_date",
        "cross_date",
        "date_pair_stratified_rank_all",
    }


def test_build_residual_diagnostics_keeps_primary_pool_separate_from_excluded_pairs():
    observed_pairs, reliability, model_rdms = _tiny_residual_inputs()

    outputs = build_residual_diagnostics(
        observed_pair_distances=observed_pairs,
        pair_reliability=reliability,
        model_rdms=model_rdms,
        n_permutations=5,
        seed=123,
    )

    assert not outputs.residual_pairs.empty
    assert not outputs.top_residual_pairs.empty
    assert outputs.top_residual_pairs["reliability_pool"].isin(["primary", "high"]).all()
```

- [ ] **Step 2: Run residual tests to verify they fail**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "residual or score_model_scopes" -q`

Expected: FAIL because residual output builders are not implemented.

- [ ] **Step 3: Implement scoped scoring**

Add:

```python
def score_model_scopes(pair_table: pd.DataFrame, *, view_name: str, model_id: str) -> pd.DataFrame:
    ...
```

Scopes:

- `all_pairs`
- `reliable_pairs`
- `high_reliability_pairs`
- `within_date`
- `cross_date`
- `date_pair_stratified_rank_all`
- `date_pair_stratified_rank_reliable`

Each row should include:

- `view_name`
- `model_id`
- `scope`
- `score_method`
- `n_pairs`
- `rsa_similarity`
- `score_status`

Use Spearman for raw-distance scopes and Pearson on within-date-pair percentile ranks for stratified scopes.

- [ ] **Step 4: Implement date-preserving permutation summary**

Add:

```python
def date_preserving_permutation_summary(
    pair_table: pd.DataFrame,
    model_rdm: pd.DataFrame,
    *,
    n_iterations: int,
    seed: int,
    scopes: tuple[str, ...] = ("all_pairs", "reliable_pairs", "date_pair_stratified_rank_all"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

Reuse the pattern from `scripts/date_controlled_rsa_review.py`:

- permute model labels within each date group
- preserve the neural distances and date-pair composition
- compute observed score and null distribution
- write one summary row per scope

For speed, default to 2000 real-data permutations but keep tests at 10 or fewer.

- [ ] **Step 5: Implement residual outputs**

Add:

```python
@dataclass(frozen=True)
class ResidualDiagnostics:
    residual_pairs: pd.DataFrame
    top_residual_pairs: pd.DataFrame
    residual_summary_by_date_pair: pd.DataFrame
    residual_summary_by_model: pd.DataFrame
    model_scope_summary: pd.DataFrame
    permutation_summary: pd.DataFrame
    permutation_null: pd.DataFrame
```

Rules:

- only `high + medium` pairs enter the primary residual pool
- `high` pairs get a separate high-confidence top list
- low and insufficient pairs remain in `residual_pairs` but not in `top_residual_pairs`
- top residuals should include both tails per view/model: largest positive and largest negative residuals
- attach a boolean `is_extreme_in_multiple_models` after all models are processed

- [ ] **Step 6: Run residual tests**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "residual or score_model_scopes or permutation" -q`

Expected: PASS.

- [ ] **Step 7: Commit residual and date-controlled scoring**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py
git commit -m "feat: add residual model diagnosis scoring"
```

### Task 5: Add Output Writer, Figures, and Summary Text

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing smoke test for output writing**

Add:

```python
from bacteria_analysis.model_diagnosis import write_model_diagnosis_outputs


def test_write_model_diagnosis_outputs_writes_expected_tables_figures_and_summary(tmp_path):
    outputs = _tiny_model_diagnosis_outputs()

    written = write_model_diagnosis_outputs(outputs, tmp_path / "model_diagnosis")

    assert (written["tables_dir"] / "neural_rdm_ceiling_summary.csv").exists()
    assert (written["tables_dir"] / "neural_pair_reliability.csv").exists()
    assert (written["tables_dir"] / "residual_summary_by_date_pair.csv").exists()
    assert (written["figures_dir"] / "neural_ceiling_by_view.png").exists()
    assert (written["figures_dir"] / "residual_by_date_pair.png").exists()
    assert (written["output_root"] / "run_summary.md").exists()
```

- [ ] **Step 2: Run writer test to verify it fails**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "write_model_diagnosis_outputs" -q`

Expected: FAIL because the writer is not implemented.

- [ ] **Step 3: Implement output dataclass and writer**

Add:

```python
@dataclass(frozen=True)
class ModelDiagnosisOutputs:
    observed_rdms: dict[str, pd.DataFrame]
    model_rdms: dict[str, pd.DataFrame]
    ceiling_summary: pd.DataFrame
    pair_reliability: pd.DataFrame
    pair_reliability_by_date_pair: pd.DataFrame
    residual_pairs: pd.DataFrame
    top_residual_pairs: pd.DataFrame
    residual_summary_by_date_pair: pd.DataFrame
    residual_summary_by_model: pd.DataFrame
    model_scope_summary: pd.DataFrame
    permutation_summary: pd.DataFrame
    permutation_null: pd.DataFrame
    model_summary: pd.DataFrame
    feature_qc: pd.DataFrame
    pca_summary: pd.DataFrame
```

Write tables:

- `tables/neural_rdm_ceiling_summary.csv`
- `tables/neural_pair_reliability.csv`
- `tables/neural_pair_reliability_by_date_pair.csv`
- `tables/model_summary.csv`
- `tables/model_scope_summary.csv`
- `tables/date_preserving_permutation_summary.csv`
- `tables/residual_pairs__<view>__<model>.csv`
- `tables/top_residual_pairs__<view>__<model>.csv`
- `tables/residual_summary_by_date_pair.csv`
- `tables/residual_summary_by_model.csv`
- `tables/pca_embedding_summary.csv`

Write QC:

- `qc/model_feature_qc.csv`
- `qc/date_preserving_permutation_null.csv`
- `qc/neural_rdm__<view>.parquet`
- `qc/model_rdm__<model>.parquet`

Use slug-safe filenames for model IDs.

- [ ] **Step 4: Implement figures**

Keep figures simple:

- `figures/neural_ceiling_by_view.png`: point/bar plot of split-half and Spearman-Brown ceiling by view and split kind
- `figures/pair_reliability_vs_distance.png`: scatter of neural distance vs rank stability, colored by tier
- `figures/residual_by_date_pair.png`: date-pair summary of residual mean and spread
- `figures/residual_scatter__<view>__<model>.png`: neural rank vs model rank, colored by reliability tier
- `figures/top_residual_pair_heatmap__<view>__<model>.png`: compact heatmap/table-like plot of top residuals

No elaborate visual system is needed. Keep labels readable and avoid expensive plotting.

- [ ] **Step 5: Implement `run_summary.md` writer**

Include:

- fixed analysis contract
- neural ceiling summary table
- reliability tier counts by view
- top fixed-model comparison rows
- top PCA rows, if any
- top residual examples for global profile and QC20 log2 Euclidean
- interpretation caveat that filtered batch lacks clean LODO and date claims are date-controlled diagnostics

- [ ] **Step 6: Run output tests**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "write_model_diagnosis_outputs" -q`

Expected: PASS.

- [ ] **Step 7: Commit output writer and figures**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py
git commit -m "feat: write model diagnosis outputs"
```

### Task 6: Add CLI Script and Synthetic End-to-End Smoke Test

**Files:**
- Create: `H:/Process_temporary/WJH/bacteria_analysis/scripts/model_diagnosis_review.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing CLI smoke test**

Add a synthetic end-to-end test that writes a tiny preprocess root, matrix workbook, and raw metadata workbook:

```python
import subprocess


def test_model_diagnosis_review_cli_writes_outputs_on_tiny_fixture(tmp_path):
    fixture = _write_tiny_model_diagnosis_fixture(tmp_path)
    output_root = tmp_path / "model_diagnosis"

    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/model_diagnosis_review.py",
            "--preprocess-root",
            str(fixture["preprocess_root"]),
            "--matrix-path",
            str(fixture["matrix_path"]),
            "--raw-metadata-path",
            str(fixture["raw_metadata_path"]),
            "--base-results-root",
            str(fixture["base_results_root"]),
            "--output-root",
            str(output_root),
            "--split-iterations",
            "8",
            "--permutations",
            "5",
            "--seed",
            "123",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "tables" / "neural_rdm_ceiling_summary.csv").exists()
    assert (output_root / "tables" / "model_scope_summary.csv").exists()
    assert (output_root / "run_summary.md").exists()
```

- [ ] **Step 2: Run CLI smoke test to verify it fails**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "review_cli" -q`

Expected: FAIL because `scripts/model_diagnosis_review.py` does not exist.

- [ ] **Step 3: Implement CLI script**

Create `scripts/model_diagnosis_review.py`:

```python
"""Run filtered-batch model diagnosis beyond global profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.model_diagnosis import (
    DEFAULT_BASE_RESULTS_ROOT,
    DEFAULT_MATRIX_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREPROCESS_ROOT,
    DEFAULT_RAW_METADATA_PATH,
    run_model_diagnosis_review,
    write_model_diagnosis_outputs,
)
```

CLI options:

- `--preprocess-root`
- `--matrix-path`
- `--raw-metadata-path`
- `--base-results-root`
- `--output-root`
- `--qc-threshold`
- `--split-iterations`
- `--permutations`
- `--seed`
- `--pca-dimensions`

The default invocation should be:

```powershell
pixi run python scripts/model_diagnosis_review.py
```

- [ ] **Step 4: Implement orchestration helper**

In `model_diagnosis.py`, add:

```python
def run_model_diagnosis_review(
    *,
    preprocess_root: str | Path,
    matrix_path: str | Path,
    raw_metadata_path: str | Path,
    base_results_root: str | Path,
    qc_threshold: float = DEFAULT_QC_THRESHOLD,
    split_iterations: int = DEFAULT_SPLIT_ITERATIONS,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    pca_dimensions: tuple[int, ...] = (2, 3, 5, 10, 20),
) -> ModelDiagnosisOutputs:
    ...
```

Flow:

1. load aggregated-response context from preprocess root
2. build neural split diagnostics
3. build model RDM bundle
4. build residual diagnostics
5. return `ModelDiagnosisOutputs`

- [ ] **Step 5: Run CLI smoke test**

Run: `pixi run pytest tests/test_model_diagnosis.py -k "review_cli" -q`

Expected: PASS.

- [ ] **Step 6: Commit CLI script**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/scripts/model_diagnosis_review.py H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py
git commit -m "feat: add model diagnosis review CLI"
```

### Task 7: Run Full Tests and Real Filtered-Batch Review

**Files:**
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/scripts/model_diagnosis_review.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Run the new focused test file**

Run: `pixi run pytest tests/test_model_diagnosis.py -q`

Expected: PASS.

- [ ] **Step 2: Run adjacent regression tests**

Run: `pixi run pytest tests/test_model_space.py tests/test_rsa.py tests/test_rsa_cli_smoke.py tests/test_reliability.py -q`

Expected: PASS.

- [ ] **Step 3: Fix only regressions caused by this work**

If tests fail, only modify:

- `src/bacteria_analysis/model_diagnosis.py`
- `scripts/model_diagnosis_review.py`
- `tests/test_model_diagnosis.py`

Do not refactor production RSA or reliability code unless a direct integration issue requires a small compatibility adjustment.

- [ ] **Step 4: Re-run focused and adjacent tests**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -q
pixi run pytest tests/test_model_space.py tests/test_rsa.py tests/test_rsa_cli_smoke.py tests/test_reliability.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the real filtered-batch diagnosis**

Run:

```powershell
pixi run python scripts/model_diagnosis_review.py --split-iterations 200 --permutations 2000 --seed 20260420
```

Expected:

- exits with code 0
- writes `results/202604_without_20260331/model_diagnosis/run_summary.md`
- writes `tables/neural_rdm_ceiling_summary.csv`
- writes `tables/neural_pair_reliability.csv`
- writes `tables/model_scope_summary.csv`
- writes residual CSVs for at least `global_profile_default_correlation` and `global_qc20_log2_euclidean`
- writes figures under `figures/`

- [ ] **Step 6: Inspect real output for obvious contract failures**

Run:

```powershell
Get-ChildItem results/202604_without_20260331/model_diagnosis -Recurse | Select-Object FullName
Get-Content results/202604_without_20260331/model_diagnosis/run_summary.md -TotalCount 120
```

Check:

- both views appear
- `global_profile_default_correlation` appears
- `global_qc20_log2_euclidean` appears
- reliability tier counts are nonempty
- PCA rows are clearly marked as unsupervised embedding models
- date caveat is present

- [ ] **Step 7: Commit final validation fixes**

If the real run exposed small issues, commit them:

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/scripts/model_diagnosis_review.py H:/Process_temporary/WJH/bacteria_analysis/tests/test_model_diagnosis.py
git commit -m "test: validate model diagnosis review"
```

If no fixes were needed, skip this commit.

### Task 8: Update Project Memory and Handoff Notes

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/memory/2026-04-20.md`
- Modify if durable: `H:/Process_temporary/WJH/bacteria_analysis/MEMORY.md`
- Modify if durable: `H:/Process_temporary/WJH/bacteria_analysis/memory/INDEX.md`
- Modify if current state changed: `H:/Process_temporary/WJH/bacteria_analysis/state/project-status.json`

- [ ] **Step 1: Append a daily memory log entry**

Record:

- script path
- output root
- tests run
- real-data run status
- high-level finding only if the real output has been inspected

- [ ] **Step 2: Promote durable memory only if warranted**

Promote only if the real diagnosis establishes a durable conclusion, such as:

- neural ceiling meaningfully constrains interpretation
- residual mismatch concentrates in a specific model/date/taxonomy pattern
- unsupervised PCA consistently helps or clearly fails under controls

Do not promote transient implementation details.

- [ ] **Step 3: Commit memory updates only if made**

Memory files are ignored, so this step is normally local-only unless the user explicitly asks to track memory changes.

## Final Verification Checklist

- [ ] `pixi run pytest tests/test_model_diagnosis.py -q` passes.
- [ ] `pixi run pytest tests/test_model_space.py tests/test_rsa.py tests/test_rsa_cli_smoke.py tests/test_reliability.py -q` passes.
- [ ] `pixi run python scripts/model_diagnosis_review.py --split-iterations 200 --permutations 2000 --seed 20260420` succeeds.
- [ ] `results/202604_without_20260331/model_diagnosis/run_summary.md` exists and includes the fixed filtered-batch contract.
- [ ] `tables/neural_rdm_ceiling_summary.csv` includes both `response_window` and `full_trajectory`.
- [ ] `tables/neural_pair_reliability.csv` includes reliability tiers.
- [ ] `tables/model_scope_summary.csv` includes all-pairs, reliable-pairs, high-reliability, within-date, cross-date, and date-pair-stratified scopes.
- [ ] Residual outputs exist for `global_profile_default_correlation` and `global_qc20_log2_euclidean`.
- [ ] PCA embedding outputs are clearly labeled as unsupervised and exploratory.
- [ ] Generated `results/` artifacts are not staged.
