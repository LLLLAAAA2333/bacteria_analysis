# Supervised Chemical Subspace and Metabolite Subset Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Archive status, 2026-04-22:** This line is archived as a negative result. The implemented search tends to select only 1-2 chemicals, produces only moderate final RSA, and requires too many validation-scope distinctions to stay readable. Preserve outputs for audit, but do not continue extending this route.

**Goal:** Build a supervised but held-out-validated search package that identifies neural-aligned candidate chemical subspaces and metabolite subsets from `QC20 + log2(matrix)` features.

**Architecture:** Add a focused `supervised_subspace_search` module and a thin CLI script. Reuse the model-diagnosis branch for neural RDM construction, reliability tiers, scoped RSA scoring, and date metadata. Keep the first production pass dependency-free and interpretable: taxonomy category exhaustive search first, within-category metabolite refinement second, and global metabolite search as exploratory.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, parquet IO, openpyxl, pytest, existing `bacteria_analysis` modules. Do not add `sklearn` or `scipy`; implement greedy selection and simple numpy scoring directly.

---

## Prerequisites

- Base this work on the `codex/model-diagnosis` branch or merge/cherry-pick commit `31a2b39` first. Current `master` does not contain `src/bacteria_analysis/model_diagnosis.py`.
- Use the filtered batch by default: `data/202604/202604_preprocess_without_20260331`, `data/matrix.xlsx`, `data/metabolism_raw_data.xlsx`, and `results/202604_without_20260331/model_diagnosis`.
- Treat this as supervised model selection. Training scores are diagnostic only; main evidence must come from held-out date, held-out stimulus, and cross-view validation.
- Do not describe selected subsets as the true chemical space. Use `neural-aligned candidate chemical subspace` unless held-out validation is strong and stable.

## File Map

- Create: `src/bacteria_analysis/supervised_subspace_search.py`
  Responsibility: taxonomy candidate construction, folds, RDM construction, supervised search, held-out scoring, null controls, stability summaries, and output writing.
- Create: `scripts/supervised_subspace_search.py`
  Responsibility: CLI wrapper for the real filtered-batch run.
- Create: `tests/test_supervised_subspace_search.py`
  Responsibility: unit tests with synthetic feature/neural tables; no real-data dependency.
- Modify only if necessary: `src/bacteria_analysis/model_diagnosis.py`
  Responsibility: expose small reusable helpers if they are not already exported. Avoid expanding this large module with search-specific code.

Generated but ignored output:

- `results/202604_without_20260331/supervised_subspace_search`
  Responsibility: real-data tables, figures, QC artifacts, and `run_summary.md`.

## Constants and Defaults

Use these defaults in the module and CLI:

```python
VIEW_NAMES = ("response_window", "full_trajectory")
DEFAULT_PREPROCESS_ROOT = Path("data/202604/202604_preprocess_without_20260331")
DEFAULT_MATRIX_PATH = Path("data/matrix.xlsx")
DEFAULT_RAW_METADATA_PATH = Path("data/metabolism_raw_data.xlsx")
DEFAULT_MODEL_DIAGNOSIS_ROOT = Path("results/202604_without_20260331/model_diagnosis")
DEFAULT_OUTPUT_ROOT = Path("results/202604_without_20260331/supervised_subspace_search")
DEFAULT_QC_THRESHOLD = 0.2
DEFAULT_MIN_CATEGORY_FEATURES = 5
DEFAULT_MAX_COMBO_SIZE = 3
DEFAULT_TOP_CATEGORY_COMBOS = 20
DEFAULT_TOP_METABOLITE_SCREEN = 50
DEFAULT_MAX_SELECTED_METABOLITES = 20
DEFAULT_RANDOM_NULL_ITERATIONS = 500
DEFAULT_SEED = 20260421
DEFAULT_STIMULUS_SPLITS = 20
DEFAULT_STIMULUS_TEST_FRACTION = 0.25
PRIMARY_RELIABILITY_TIERS = {"high", "medium"}
TAXONOMY_LEVELS = ("SuperClass", "Class", "SubClass")
MISSING_TAXONOMY_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown"}
```

Use stable search IDs:

```text
taxonomy_union
within_taxonomy_greedy_metabolite
global_greedy_metabolite
weighted_category_fusion_optional
```

## Search Definition

### Layer A: Taxonomy Category Exhaustive Search

Search unit:

```text
SuperClass::<category>
Class::<category>
SubClass::<category>
```

Candidate filter:

- metabolite is present in `matrix.xlsx`
- `QCRSD <= 0.2`
- taxonomy category is non-empty after stripping and lowercasing
- taxonomy category is not a missing-like token: `""`, `na`, `n/a`, `nan`, `none`, `null`, or `unknown`
- category has at least `DEFAULT_MIN_CATEGORY_FEATURES` retained features

Search space:

```text
single categories: C(n, 1)
category pairs: C(n, 2)
category triples: C(n, 3)
```

Model construction:

```text
combo features = union(category_1 features, category_2 features, ...)
chemical RDM = Euclidean(log2(matrix[combo features]))
training score = Spearman(neural_distance_train, chemical_distance_train)
```

### Layer B: Within-Taxonomy Metabolite Refinement

For each top taxonomy combo from Layer A, search individual metabolites inside the union of selected categories.

Method:

```text
1. Split the outer-training fold into inner train/validation folds.
2. For each inner fold, screen individual metabolites on inner-train pairs only.
3. Keep top M metabolites per inner fold.
4. Greedy forward-select metabolites on inner-train pairs up to k_max.
5. Score each prefix length k on the inner-validation pairs.
6. Pick k using the smallest k within one standard error of the best mean inner-validation score.
7. Refit screen + greedy selection on the full outer-training fold, then keep the first k metabolites.
8. Evaluate once on held-out pairs.
```

Do not choose `k` by validating prefixes from a greedy path that was already
fit on the full outer-training fold. That leaks outer-training selection
information into the inner validation step.

### Layer C: Global Metabolite Search

Exploratory only. Search individual metabolites from all retained QC20 features, not limited to taxonomy categories.

Use the same screen + greedy + inner-validation pattern as Layer B. Mark all outputs as exploratory because the search space is much larger and less interpretable.

## Validation Contract

### Primary: Leave-One-Date-Out

For each held-out date:

```text
train pairs: pairs with neither stimulus from held-out date
heldout_within_date: both stimuli from held-out date
heldout_cross_date: one stimulus from held-out date and one from training dates
heldout_touching_date: any pair with at least one stimulus from held-out date
```

Selection uses only `train pairs`.

### Secondary: Date-Stratified Held-Out Stimulus Split

For each split:

```text
train stimuli: stratified subset within each date
test stimuli: held-out subset within each date
train pairs: both stimuli in train set
test_test pairs: both stimuli in test set
train_test pairs: one train stimulus and one test stimulus
```

Main stimulus-generalization readout is `test_test`; `train_test` is supplementary.

### Cross-View Transfer

For each selected subset:

```text
select on response_window, evaluate full_trajectory
select on full_trajectory, evaluate response_window
```

This catches subsets that only fit one neural summary.

Every search and validation row must carry both:

```text
selection_view: view used for taxonomy or metabolite selection
evaluation_view: view used for held-out scoring
```

Within-view validation uses the same value for both columns.

### Null Controls

Use two nulls:

- `matched_feature_null`: random metabolite sets matched to the selected subset's feature count.
- `matched_category_null`: random taxonomy combos matched to combo size and approximate feature count.

For each held-out fold, compare the selected subset's held-out score against null held-out scores. Report percentile and empirical p-value.

## Promotion Rules

A subset can be called a `stable neural-aligned candidate chemical subspace` only if:

- median held-out delta versus `global_qc20_log2_euclidean` is positive,
- improvement appears in more than one held-out fold,
- performance is not driven by a single date or a single view,
- selected categories or metabolites recur across folds,
- matched random null does not explain the held-out improvement.

If train score is high but held-out score is weak, report it as selection bias.

## Output Contract

Tables:

- `tables/candidate_category_summary.csv`
- `tables/taxonomy_train_search_scores.csv`
- `tables/fold_selected_taxonomy_subspaces.csv`
- `tables/metabolite_refinement_paths.csv`
- `tables/metabolite_inner_validation_scores.csv`
- `tables/fold_selected_metabolite_subsets.csv`
- `tables/heldout_validation_scores.csv`
- `tables/selection_stability_summary.csv`
- `tables/matched_random_null_summary.csv`
- `tables/final_candidate_subspace_summary.csv`

Required validation metadata columns in selection and held-out tables:

- `fold_id`
- `fold_kind`
- `selection_view`
- `evaluation_view`
- `selection_scope`
- `scope`
- `search_layer`

Required taxonomy overlap columns wherever category combos are reported:

- `n_sum_category_features`
- `n_union_features`
- `n_overlap_features`
- `feature_redundancy_ratio`
- `max_pairwise_category_jaccard`
- `mean_pairwise_category_jaccard`

Figures:

- `figures/heldout_rsa_by_fold.png`
- `figures/heldout_delta_vs_baseline.png`
- `figures/selection_frequency_by_category.png`
- `figures/selection_frequency_by_metabolite.png`
- `figures/candidate_vs_matched_null.png`
- `figures/within_cross_date_validation.png`
- `figures/final_subspace_rdm_comparison.png`

Summary:

- `run_summary.md` with concise sections:
  - inputs and search settings
  - best held-out taxonomy candidates
  - best held-out metabolite-refined candidates
  - null control readout
  - stability readout
  - interpretation guardrails

## Task 1: Add Module Skeleton and Core Data Structures

**Files:**
- Create: `src/bacteria_analysis/supervised_subspace_search.py`
- Create: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write failing import and dataclass tests**

Create tests for `SearchSettings`, `CategoryCandidate`, and `SearchFold`.

```python
from bacteria_analysis.supervised_subspace_search import CategoryCandidate, SearchFold, SearchSettings


def test_search_settings_defaults_are_conservative():
    settings = SearchSettings()
    assert settings.qc_threshold == 0.2
    assert settings.min_category_features == 5
    assert settings.max_combo_size == 3
    assert settings.max_selected_metabolites == 20
    assert settings.stimulus_splits == 20
    assert settings.stimulus_test_fraction == 0.25


def test_category_candidate_has_stable_id():
    candidate = CategoryCandidate(
        category_id="Class::Indoles",
        taxonomy_level="Class",
        category="Indoles",
        metabolites=("indole", "tryptophan"),
    )
    assert candidate.category_id == "Class::Indoles"
    assert candidate.n_features == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Expected: FAIL because module does not exist.

- [ ] **Step 3: Add minimal module skeleton**

Implement frozen dataclasses and constants in `src/bacteria_analysis/supervised_subspace_search.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/bacteria_analysis/supervised_subspace_search.py tests/test_supervised_subspace_search.py
git commit -m "feat: add supervised subspace search skeleton"
```

## Task 2: Build QC20 Log2 Feature Frame and Taxonomy Candidate Pool

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write tests for category candidate construction**

Test that `build_category_candidates()`:

- keeps only metabolites present in the feature frame
- keeps only `QCRSD <= qc_threshold`
- drops blank taxonomy labels
- drops missing-like taxonomy labels: `na`, `n/a`, `nan`, `none`, `null`, and `unknown`
- enforces `min_features`
- preserves feature-frame column order in `metabolites`

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -k category_candidates -q`

Expected: FAIL because function is missing.

- [ ] **Step 3: Implement candidate builder**

Implement:

```python
def normalize_taxonomy_label(value: object) -> str:
    ...


def build_category_candidates(
    feature_frame: pd.DataFrame,
    taxonomy_metadata: pd.DataFrame,
    *,
    taxonomy_levels: tuple[str, ...] = TAXONOMY_LEVELS,
    qc_threshold: float = 0.2,
    min_features: int = 5,
) -> list[CategoryCandidate]:
    ...
```

Rules for `normalize_taxonomy_label()`:

- strip surrounding whitespace
- collapse blank and missing-like tokens from `MISSING_TAXONOMY_TOKENS` to `""`
- preserve display spelling for real labels
- use the normalized label before building stable `category_id` values

- [ ] **Step 4: Add real-data metadata loader**

Add `load_taxonomy_metadata(raw_metadata_path)` and reuse normalization logic from existing modules where safe.

- [ ] **Step 5: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: build taxonomy candidate pool"` after staging changed files.

## Task 3: Add RDM Construction, Pair Alignment, and Scoring

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write tests for RDM and score helpers**

Test `build_feature_union_rdm()` on a tiny feature frame and test `score_candidate_rdm()` with a pair mask that excludes one high-scoring pair.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -k "rdm or score" -q`

Expected: FAIL because helpers are missing.

- [ ] **Step 3: Implement RDM builder**

Implement:

```python
def build_feature_union_rdm(feature_frame: pd.DataFrame, metabolites: tuple[str, ...]) -> pd.DataFrame:
    ...
```

Rules:

- require at least one metabolite
- require all metabolites exist in `feature_frame`
- compute Euclidean pairwise distances with numpy
- return first column `stimulus_row`, matching `model_diagnosis` RDM shape

- [ ] **Step 4: Implement pair scoring**

Implement:

```python
def score_candidate_rdm(
    pair_table: pd.DataFrame,
    candidate_rdm: pd.DataFrame,
    *,
    pair_mask: pd.Series,
    reliability_tiers: set[str] | None = PRIMARY_RELIABILITY_TIERS,
) -> dict[str, object]:
    ...
```

Use `model_diagnosis.score_values` for Spearman. Return `n_pairs`, `rsa_similarity`, and `score_status`.

- [ ] **Step 5: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: add subspace RDM scoring helpers"` after staging changed files.

## Task 4: Build Held-Out Fold Definitions

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write fold tests**

Acceptance cases:

- LODO creates one fold per date.
- A held-out date fold has no held-out-date stimuli in `train_stimuli`.
- Stimulus split keeps train and test disjoint.
- Split is deterministic for a fixed seed.

- [ ] **Step 2: Implement LODO fold builder**

Implement `build_leave_one_date_out_folds(stimulus_dates)`.

- [ ] **Step 3: Implement stimulus split builder**

Implement `build_date_stratified_stimulus_folds(stimulus_dates, n_splits, test_fraction, seed)` using numpy RNG.

- [ ] **Step 4: Implement pair masks**

Implement `fold_pair_masks(pair_table, fold)` with masks:

- `train`
- `heldout_within_date`
- `heldout_cross_date`
- `heldout_touching_date`
- `test_test`
- `train_test`

- [ ] **Step 5: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: add supervised search fold definitions"` after staging changed files.

## Task 5: Implement Taxonomy Exhaustive Search

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write tests for combination search**

Create three category candidates and a synthetic pair table where one category combo clearly matches neural distances. Verify search ranks that combo first using only the `train` mask.

- [ ] **Step 2: Implement combo generator**

Implement:

```python
def iter_category_combos(
    candidates: list[CategoryCandidate],
    *,
    max_combo_size: int,
) -> Iterator[tuple[CategoryCandidate, ...]]:
    ...
```

Sort combos deterministically and include singles, pairs, and triples.

- [ ] **Step 3: Implement taxonomy search**

Implement:

```python
def search_taxonomy_unions(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    candidates: list[CategoryCandidate],
    fold: SearchFold,
    *,
    selection_view: str,
    max_combo_size: int,
    top_n: int,
) -> pd.DataFrame:
    ...
```

Output columns:

- `fold_id`
- `fold_kind`
- `selection_view`
- `search_layer`
- `combo_size`
- `combo_id`
- `taxonomy_levels`
- `categories`
- `n_sum_category_features`
- `n_union_features`
- `n_overlap_features`
- `feature_redundancy_ratio`
- `max_pairwise_category_jaccard`
- `mean_pairwise_category_jaccard`
- `metabolites`
- `train_n_pairs`
- `train_rsa`
- `train_score_status`

- [ ] **Step 4: Add baseline scoring**

For every fold and view, score `global_qc20_log2_euclidean` on the same masks. This is needed for held-out delta.

- [ ] **Step 5: Add taxonomy overlap helper**

Implement:

```python
def category_combo_overlap_metrics(combo: tuple[CategoryCandidate, ...]) -> dict[str, object]:
    ...
```

Rules:

- `n_sum_category_features` is the sum of category feature counts before unioning.
- `n_union_features` is the size of the union.
- `n_overlap_features = n_sum_category_features - n_union_features`.
- `feature_redundancy_ratio = n_overlap_features / n_sum_category_features`, or `0.0` for empty sums.
- pairwise Jaccard values are computed between category metabolite sets.
- singles report `max_pairwise_category_jaccard = 0.0` and `mean_pairwise_category_jaccard = 0.0`.

- [ ] **Step 6: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: add taxonomy union search"` after staging changed files.

## Task 6: Implement Held-Out Scoring for Selected Taxonomy Combos

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write held-out scoring tests**

Use a synthetic fold where the best training combo is known. Verify:

- selected combo is scored on `heldout_within_date`
- selected combo is scored on `heldout_cross_date`
- selected combo is scored on `test_test` for a date-stratified stimulus fold
- scoring supports `selection_view != evaluation_view`
- held-out rows include baseline score and delta

- [ ] **Step 2: Implement selected combo scoring**

Implement:

```python
def score_selected_taxonomy_combos(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    selected: pd.DataFrame,
    fold: SearchFold,
    *,
    selection_view: str,
    evaluation_view: str,
    baseline_rdm: pd.DataFrame,
) -> pd.DataFrame:
    ...
```

Output columns:

- `fold_id`
- `fold_kind`
- `selection_view`
- `evaluation_view`
- `search_layer`
- `model_id`
- `selection_scope`
- `scope`
- `n_pairs`
- `rsa_similarity`
- `baseline_rsa_similarity`
- `delta_vs_baseline`
- `score_status`

- [ ] **Step 3: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: score held-out taxonomy subspaces"` after staging changed files.

## Task 7: Add Within-Taxonomy Greedy Metabolite Refinement

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write greedy selection tests**

Synthetic case:

- feature `a` explains neural distances
- feature `b` is noise
- feature `c` helps after `a`
- greedy path starts with `a` and improves with `c`

Nested-validation case:

- outer train contains dates `d1`, `d2`, and `d3`
- each inner split holds out one training date
- screen + greedy is recomputed inside each inner split
- a deliberately overfit full-outer greedy path cannot determine the chosen `k`

- [ ] **Step 2: Implement univariate screen**

Implement:

```python
def screen_metabolites(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    metabolites: tuple[str, ...],
    train_mask: pd.Series,
    *,
    top_m: int,
) -> pd.DataFrame:
    ...
```

Output columns: `metabolite`, `screen_rank`, `train_rsa`, `n_pairs`.

- [ ] **Step 3: Implement greedy forward selection**

Implement:

```python
def greedy_select_metabolites(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    candidate_metabolites: tuple[str, ...],
    train_mask: pd.Series,
    *,
    max_selected: int,
) -> pd.DataFrame:
    ...
```

Output path columns: `step`, `added_metabolite`, `selected_metabolites`, `n_selected`, `train_rsa`, `train_delta`.

- [ ] **Step 4: Implement inner folds over the outer-training set**

Implement:

```python
def build_inner_training_masks(
    pair_table: pd.DataFrame,
    outer_fold: SearchFold,
) -> dict[str, dict[str, pd.Series]]:
    ...
```

For LODO outer folds, use leave-one-date-out over the dates that remain in
`outer_fold.train_stimuli`. For date-stratified stimulus outer folds, create
deterministic inner stimulus splits only within `outer_fold.train_stimuli`.
Each returned entry must include `inner_train` and `inner_validation` masks.

- [ ] **Step 5: Implement nested validation for k**

Implement:

```python
def choose_metabolite_subset_size_nested(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    candidate_metabolites: tuple[str, ...],
    inner_masks: dict[str, dict[str, pd.Series]],
    *,
    top_m: int,
    max_selected: int,
) -> dict[str, object]:
    ...
```

For each inner split, run `screen_metabolites()` on `inner_train`, then run
`greedy_select_metabolites()` on the screened metabolites and `inner_train`.
Score each prefix on `inner_validation`. Choose the smallest `k` within one
standard error of the best mean inner-validation score.

Do not pass a full-outer-fold `greedy_path` into this function.

- [ ] **Step 6: Implement outer refit after k selection**

Implement:

```python
def refine_metabolites_nested(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    candidate_metabolites: tuple[str, ...],
    fold: SearchFold,
    *,
    top_m: int,
    max_selected: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ...
```

Expected behavior:

- build inner masks from the outer-training stimuli
- choose `k` with `choose_metabolite_subset_size_nested()`
- rerun screen + greedy on the full outer-training mask
- return the full outer refit path, inner-validation score rows, and a selected subset row using the first `k` metabolites
- include `chosen_k`, `inner_mean_rsa`, `inner_se_rsa`, and `selection_scope = "outer_train"` in outputs

- [ ] **Step 7: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: refine taxonomy subspaces with greedy metabolite selection"` after staging changed files.

## Task 8: Add Global Greedy Metabolite Search as Exploratory Supplement

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write tests for global search labeling**

Verify global search rows are marked:

```text
search_layer = global_greedy_metabolite
interpretation_tier = exploratory
selection_scope = outer_train
```

- [ ] **Step 2: Implement global search wrapper**

Implement:

```python
def search_global_metabolite_subset(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    fold: SearchFold,
    *,
    selection_view: str,
    top_m: int,
    max_selected: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ...
```

This should reuse `refine_metabolites_nested()` with all retained QC20
metabolites as the candidate set. It must keep global-search rows marked
`interpretation_tier = exploratory`.

- [ ] **Step 3: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: add exploratory global metabolite search"` after staging changed files.

## Task 9: Add Matched Random Nulls and Selection Stability

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write tests for matched random feature null**

Verify random subsets:

- match selected feature count
- are deterministic for seed
- return empirical percentile and p-value

- [ ] **Step 2: Implement matched feature null**

Implement:

```python
def matched_feature_null(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    selected_metabolites: tuple[str, ...],
    mask: pd.Series,
    *,
    n_iterations: int,
    seed: int,
) -> pd.DataFrame:
    ...
```

- [ ] **Step 3: Implement matched category null**

Implement:

```python
def matched_category_null(
    pair_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    candidates: list[CategoryCandidate],
    selected_combo_size: int,
    selected_feature_count: int,
    mask: pd.Series,
    *,
    n_iterations: int,
    seed: int,
) -> pd.DataFrame:
    ...
```

Match feature count within a tolerance first. If too few candidates exist, widen tolerance and record the tolerance in output.

- [ ] **Step 4: Implement selection stability summaries**

Implement:

```python
def summarize_selection_stability(selected_rows: pd.DataFrame) -> pd.DataFrame:
    ...
```

Output frequency for taxonomy category, taxonomy level, metabolite, selection
view, evaluation view, and fold kind. Include fold-level and cross-fold
Jaccard summaries:

- `category_jaccard_vs_other_folds`
- `metabolite_jaccard_vs_other_folds`
- `mean_pairwise_category_jaccard`
- `max_pairwise_category_jaccard`
- `feature_redundancy_ratio`

Add tests that nested taxonomy categories produce nonzero overlap metrics and
that disjoint categories produce zero Jaccard.

- [ ] **Step 5: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: add null controls and selection stability"` after staging changed files.

## Task 10: Add Orchestration and Output Writer

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write tiny end-to-end test**

Create synthetic feature frame, metadata, pair table, and dates. Verify orchestration returns all required tables and writes CSV outputs.

Also verify:

- both `lodo` and `date_stratified_stimulus` fold kinds are present
- within-view rows have `selection_view == evaluation_view`
- cross-view rows have `selection_view != evaluation_view`
- held-out rows include `heldout_within_date`, `heldout_cross_date`, and `test_test`
- selected taxonomy rows include overlap/Jaccard columns

- [ ] **Step 2: Add output dataclass**

Implement:

```python
@dataclass(frozen=True)
class SupervisedSubspaceSearchOutputs:
    candidate_category_summary: pd.DataFrame
    taxonomy_train_search_scores: pd.DataFrame
    fold_selected_taxonomy_subspaces: pd.DataFrame
    metabolite_refinement_paths: pd.DataFrame
    metabolite_inner_validation_scores: pd.DataFrame
    fold_selected_metabolite_subsets: pd.DataFrame
    heldout_validation_scores: pd.DataFrame
    selection_stability_summary: pd.DataFrame
    matched_random_null_summary: pd.DataFrame
    final_candidate_subspace_summary: pd.DataFrame
```

- [ ] **Step 3: Add orchestration function**

Implement:

```python
def run_supervised_subspace_search(
    *,
    preprocess_root: str | Path,
    matrix_path: str | Path,
    raw_metadata_path: str | Path,
    model_diagnosis_root: str | Path,
    settings: SearchSettings = SearchSettings(),
    view_names: tuple[str, ...] = VIEW_NAMES,
) -> SupervisedSubspaceSearchOutputs:
    ...
```

Expected behavior:

- build/load `QC20 + log2` feature frame
- load observed neural pair table and pair reliability from model diagnosis outputs, or rebuild via `model_diagnosis` helpers
- build LODO folds
- build date-stratified held-out stimulus folds using `settings.stimulus_splits`, `settings.stimulus_test_fraction`, and `settings.seed`
- for each `selection_view`, run taxonomy search using only that view's training pairs
- run metabolite refinement for top taxonomy candidates with nested validation inside the outer-training fold
- run exploratory global metabolite search with the same nested validation contract
- for each selected subset, score held-out scopes for every `evaluation_view`
- mark cross-view rows with `selection_view != evaluation_view`
- compute nulls and stability
- summarize final candidates

Use this loop shape:

```python
for fold in lodo_folds + stimulus_folds:
    for selection_view in view_names:
        selected = select_subsets(pair_tables[selection_view], fold)
        for evaluation_view in view_names:
            score_selected_subsets(
                pair_tables[evaluation_view],
                selected,
                fold,
                selection_view=selection_view,
                evaluation_view=evaluation_view,
            )
```

Never select on `evaluation_view` unless it is also the `selection_view` for
that row.

- [ ] **Step 4: Add writer**

Implement:

```python
def write_supervised_subspace_search_outputs(
    outputs: SupervisedSubspaceSearchOutputs,
    output_root: str | Path,
) -> dict[str, Path]:
    ...
```

Write all required CSVs and `run_summary.md`.

`run_summary.md` must include:

- fold counts by `fold_kind`
- within-view versus cross-view held-out readout
- date-stratified stimulus readout for `test_test`
- taxonomy overlap/redundancy summary for selected category combos
- interpretation guardrails for exploratory global metabolite rows

- [ ] **Step 5: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: orchestrate supervised subspace search outputs"` after staging changed files.

## Task 11: Add Figures

**Files:**
- Modify: `src/bacteria_analysis/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Add smoke test for figure writing**

Use tiny output frames and verify the figure paths exist after writer runs.

- [ ] **Step 2: Implement compact matplotlib figures**

Create these plot helpers:

```python
def plot_heldout_rsa_by_fold(...)
def plot_heldout_delta_vs_baseline(...)
def plot_selection_frequency_by_category(...)
def plot_selection_frequency_by_metabolite(...)
def plot_candidate_vs_matched_null(...)
def plot_within_cross_date_validation(...)
def plot_final_subspace_rdm_comparison(...)
```

Keep plots simple and readable. Do not create per-combo figure explosions.

- [ ] **Step 3: Run tests and commit**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Commit: `git commit -m "feat: add supervised subspace search figures"` after staging changed files.

## Task 12: Add CLI Script

**Files:**
- Create: `scripts/supervised_subspace_search.py`
- Modify: `tests/test_supervised_subspace_search.py`

- [ ] **Step 1: Write CLI smoke test**

If existing CLI smoke patterns are present, follow them. Otherwise test `_parse_args` and `_resolve_repo_path` only.

- [ ] **Step 2: Implement CLI**

Add arguments:

```text
--preprocess-root
--matrix-path
--raw-metadata-path
--model-diagnosis-root
--output-root
--qc-threshold
--min-category-features
--max-combo-size
--top-category-combos
--top-metabolite-screen
--max-selected-metabolites
--random-null-iterations
--stimulus-splits
--stimulus-test-fraction
--seed
```

Use the same worktree-aware path resolution pattern as `scripts/model_diagnosis_review.py`.

- [ ] **Step 3: Run CLI smoke**

Run:

```powershell
pixi run python scripts/supervised_subspace_search.py --random-null-iterations 5 --top-category-combos 3 --max-selected-metabolites 5
```

Expected: writes output tables and figures to `results/202604_without_20260331/supervised_subspace_search`.

- [ ] **Step 4: Commit**

Run:

```powershell
git add scripts/supervised_subspace_search.py tests/test_supervised_subspace_search.py
git commit -m "feat: add supervised subspace search CLI"
```

## Task 13: Real Run and Interpretation Guardrails

**Files:**
- Generated only: `results/202604_without_20260331/supervised_subspace_search`
- Optional modify: `docs/superpowers/specs/2026-04-20-model-diagnosis-design.md` if adding a brief pointer to this follow-up

- [ ] **Step 1: Run focused tests**

Run: `pixi run pytest tests/test_supervised_subspace_search.py -q`

Expected: all pass.

- [ ] **Step 2: Run adjacent regression tests**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py tests/test_model_space.py tests/test_rsa.py -q
```

Expected: all pass. If `tests/test_model_diagnosis.py` does not exist because the branch was not based on `codex/model-diagnosis`, stop and fix the branch base.

- [ ] **Step 3: Run real analysis**

Run:

```powershell
pixi run python scripts/supervised_subspace_search.py --random-null-iterations 500 --seed 20260421
```

Expected outputs:

- all required tables exist
- all required figures exist
- `run_summary.md` exists
- held-out validation rows include taxonomy, within-taxonomy metabolite, and global metabolite layers

- [ ] **Step 4: Sanity-check leakage**

Inspect `heldout_validation_scores.csv` and verify:

- each LODO fold has training rows excluding the held-out date
- date-stratified stimulus folds include `test_test` rows
- cross-view rows have `selection_view != evaluation_view`
- selected model IDs differ by fold where expected
- held-out scopes are not used in `taxonomy_train_search_scores.csv`
- metabolite `chosen_k` was selected by inner validation, not by held-out score
- selected taxonomy combos report overlap and Jaccard columns

- [ ] **Step 5: Summarize outcomes cautiously**

In `run_summary.md`, include one of these labels:

```text
stable candidate: held-out positive and stable across folds
date-limited candidate: held-out positive only for some dates
selection-biased candidate: train positive but held-out weak
exploratory-only candidate: global metabolite search result
```

- [ ] **Step 6: Commit code, not ignored outputs**

Run:

```powershell
git add src/bacteria_analysis/supervised_subspace_search.py scripts/supervised_subspace_search.py tests/test_supervised_subspace_search.py docs/superpowers/plans/2026-04-21-subspace-search.md
git commit -m "feat: add supervised chemical subspace search"
```

## Testing Strategy

- Unit tests cover candidate construction, RDM construction, fold masks, taxonomy search, greedy metabolite selection, null matching, stability summaries, and writer outputs.
- Synthetic tests must include at least one case where training and held-out disagree, to ensure the implementation does not silently promote training-only winners.
- Real-data smoke should run with small null iterations first, then full `500` null iterations.
- Adjacent regression tests should include model diagnosis, model space, and RSA modules because this feature reuses their data contracts.

## Potential Risks and Gotchas

- **Leakage risk:** The most serious failure mode is using held-out pairs during subset selection. Keep fold masks explicit and test them.
- **Search bias:** Exhaustive taxonomy search and greedy metabolite selection can find high training RSA by chance. Treat held-out validation and matched nulls as primary.
- **Date confounding:** LODO tests date generalization, but held-out cross-date pairs include one training-date stimulus. Report `heldout_within_date` and `heldout_cross_date` separately.
- **Feature-count confounding:** Larger subsets can improve distance stability simply by including more features. Always report `n_features` and matched feature-count nulls.
- **Taxonomy redundancy:** SuperClass, Class, and SubClass can overlap. Report category overlap and metabolite Jaccard so repeated selection is interpretable.
- **Global metabolite search interpretability:** Global individual-metabolite search is less interpretable and more overfit-prone. Keep it exploratory unless it independently validates.
- **Branch base risk:** If the worker starts from `master`, model diagnosis helpers will be missing. Start from `codex/model-diagnosis` or merge that branch first.

## Rollback Plan

- Revert only the new files:
  - `src/bacteria_analysis/supervised_subspace_search.py`
  - `scripts/supervised_subspace_search.py`
  - `tests/test_supervised_subspace_search.py`
  - this plan file, if needed
- Remove generated ignored outputs under `results/202604_without_20260331/supervised_subspace_search`.
- Do not revert `model_diagnosis` changes unless the user explicitly requests it; this plan depends on them but does not own them.
