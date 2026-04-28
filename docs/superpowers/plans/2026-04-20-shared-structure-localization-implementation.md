# Shared Structure Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the filtered-batch `model_diagnosis` workflow with Section 3 anchor-local diagnostics that localize stable neural-chemical overlap across anchor stimuli, neighborhood structure, and cross-date support.

**Architecture:** Reuse the existing `bacteria_analysis.model_diagnosis` package rather than creating a second review pipeline. Build the new layer on top of the already-written `residual_pairs` table, because it already aligns neural and model distances, date metadata, and reliability tiers. Keep the first pass narrow: only `global_profile_default_correlation`, `global_qc20_log2_euclidean`, and `best_weighted_fusion_fixed_weights` get localization outputs.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib, pytest, existing `bacteria_analysis.model_diagnosis` helpers and current `pixi` test/runtime workflow. No new dependencies.

**Execution Root:** Run this plan from the dedicated worktree `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis`, not from the main checkout.

---

## File Map

- `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py`
  Responsibility: add anchor-local helper functions, localization dataclasses, summary builders, writer hooks, figure builders, and `run_summary.md` integration.
- `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py`
  Responsibility: unit coverage for anchor-local pair expansion, top-k overlap, triplet consistency, localization summaries, and output-writing smoke coverage.
- `H:/Process_temporary/WJH/bacteria_analysis/docs/superpowers/specs/2026-04-20-model-diagnosis-design.md`
  Responsibility: already updated spec; read for scope and naming only, do not modify during implementation unless the user asks for plan/spec sync changes.

Generated but ignored outputs:

- `H:/Process_temporary/WJH/bacteria_analysis/results/202604_without_20260331/model_diagnosis`
  Responsibility: localization tables, figures, and updated summary markdown from the real filtered-batch run.

## Shared Constants and Naming

Add or reuse these constants inside `model_diagnosis.py`:

```python
LOCALIZATION_MODEL_IDS = (
    "global_profile_default_correlation",
    "global_qc20_log2_euclidean",
    "best_weighted_fusion_fixed_weights",
)
LOCALIZATION_TOPK = (3, 5)
LOCALIZATION_PRIMARY_POOL = "primary"
LOCALIZATION_HIGH_POOL = "high"
MIN_LOCAL_NEIGHBORS = 3
MIN_TRIPLETS = 3
```

Expected output tables:

```text
tables/anchor_localization.csv
tables/anchor_localization_summary.csv
tables/anchor_triplet_summary.csv
```

Expected output figures:

```text
figures/anchor_rank_alignment__<view>__<model>.png
figures/anchor_neighborhood_overlap__<view>__<model>.png
figures/localization_hotspots__<view>__<model>.png
```

## Data Contract

The localization layer must run from `residual_pairs`, not from a reconstructed
sub-RDM. This is important because `high_reliability_pairs` and
`reliable_pairs` are pair subsets, not guaranteed complete square matrices.

Anchor-local logic should therefore:

1. start from aligned pair rows that already include `view_name`, `model_id`,
   `stimulus_left`, `stimulus_right`, `neural_distance`, `model_distance`,
   `date_pair_type`, and `reliability_tier`
2. expand each pair into two directed anchor rows `(A -> B)` and `(B -> A)`
3. compute local metrics by grouping on `view_name`, `model_id`, `anchor`,
   `selection_pool`, and optional `scope`

### Task 1: Add Pure Anchor-Local Helper Tests and Math

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing tests for anchor expansion, top-k overlap, and triplet consistency**

Add tests like:

```python
from bacteria_analysis.model_diagnosis import (
    expand_pairs_to_anchor_rows,
    topk_neighbor_overlap,
    triplet_order_consistency,
)


def test_expand_pairs_to_anchor_rows_duplicates_each_pair_for_both_anchors():
    pairs = pd.DataFrame(
        {
            "view_name": ["response_window", "response_window"],
            "model_id": ["model", "model"],
            "stimulus_left": ["A", "A"],
            "stimulus_right": ["B", "C"],
            "date_pair_type": ["within_date", "cross_date"],
            "neural_distance": [0.2, 0.9],
            "model_distance": [0.3, 0.8],
            "reliability_pool": ["primary", "primary"],
        }
    )

    anchor_rows = expand_pairs_to_anchor_rows(pairs)

    assert set(anchor_rows["anchor_stimulus"]) == {"A", "B", "C"}
    assert {"anchor_stimulus", "neighbor_stimulus", "anchor_scope"}.issubset(anchor_rows.columns)
    assert len(anchor_rows) == 4


def test_topk_neighbor_overlap_uses_nearest_neighbors_and_scales_by_k():
    frame = pd.DataFrame(
        {
            "neighbor_stimulus": ["B", "C", "D", "E", "F"],
            "neural_distance": [0.1, 0.2, 0.3, 0.4, 0.5],
            "model_distance": [0.2, 0.1, 0.3, 0.4, 0.5],
        }
    )

    assert topk_neighbor_overlap(frame, k=3) == pytest.approx(1.0)
    assert topk_neighbor_overlap(frame, k=5) == pytest.approx(1.0)


def test_triplet_order_consistency_reports_partial_agreement():
    frame = pd.DataFrame(
        {
            "neighbor_stimulus": ["B", "C", "D"],
            "neural_distance": [0.1, 0.4, 0.8],
            "model_distance": [0.1, 0.9, 0.2],
        }
    )

    consistency, n_triplets = triplet_order_consistency(frame)

    assert n_triplets == 3
    assert consistency == pytest.approx(1.0 / 3.0)
```

- [ ] **Step 2: Run the focused helper tests to verify they fail**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -k "anchor_rows or topk_neighbor_overlap or triplet_order_consistency" -q
```

Expected: FAIL because the helper functions do not exist yet.

- [ ] **Step 3: Implement the pure helper functions in `model_diagnosis.py`**

Add:

```python
def expand_pairs_to_anchor_rows(pair_table: pd.DataFrame) -> pd.DataFrame:
    ...


def topk_neighbor_overlap(frame: pd.DataFrame, *, k: int) -> float:
    ...


def triplet_order_consistency(frame: pd.DataFrame) -> tuple[float, int]:
    ...
```

Implementation rules:

- `expand_pairs_to_anchor_rows(...)` should turn `(A, B)` into two rows:
  `anchor=A, neighbor=B` and `anchor=B, neighbor=A`
- keep `view_name`, `model_id`, `date_pair_type`, `reliability_tier`,
  `reliability_pool`, and `is_high_reliability`
- add `anchor_scope` with values:
  - `overall`
  - `within_date`
  - `cross_date`
- `topk_neighbor_overlap(...)` should sort by ascending distance and break ties
  deterministically with `neighbor_stimulus`
- if fewer than `k` neighbors are available, return `np.nan`
- `triplet_order_consistency(...)` should compare all unordered `(B, C)` pairs,
  skip exact ties, and return `(consistency, n_valid_triplets)`

- [ ] **Step 4: Run the focused helper tests**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -k "anchor_rows or topk_neighbor_overlap or triplet_order_consistency" -q
```

Expected: PASS.

- [ ] **Step 5: Commit the pure helper layer**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py
git commit -m "feat: add anchor localization core helpers"
```

### Task 2: Build Anchor Localization Tables from Residual Pairs

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing tests for anchor-local summary builders**

Add tests like:

```python
from bacteria_analysis.model_diagnosis import build_anchor_localization_diagnostics


def test_build_anchor_localization_diagnostics_reports_primary_and_high_pools():
    residuals = _tiny_localization_residual_pairs()

    outputs = build_anchor_localization_diagnostics(residuals)

    assert not outputs.anchor_localization.empty
    assert not outputs.anchor_localization_summary.empty
    assert not outputs.anchor_triplet_summary.empty
    assert set(outputs.anchor_localization["selection_pool"]) >= {"primary", "high"}
    assert set(outputs.anchor_localization["scope"]) >= {"overall", "within_date", "cross_date"}


def test_build_anchor_localization_diagnostics_restricts_to_target_models():
    residuals = _tiny_localization_residual_pairs()
    residuals.loc[:, "model_id"] = [
        "global_profile_default_correlation",
        "global_qc20_log2_euclidean",
        "best_weighted_fusion_fixed_weights",
        "pca_qc20_log2_euclidean__k02",
    ]

    outputs = build_anchor_localization_diagnostics(residuals)

    assert "pca_qc20_log2_euclidean__k02" not in set(outputs.anchor_localization["model_id"])
```

The `_tiny_localization_residual_pairs()` fixture should include:

- at least two views
- at least three anchors with enough neighbors for `k=3`
- both `within_date` and `cross_date` rows
- `high`, `medium`, and `low` tiers

- [ ] **Step 2: Run the localization summary tests to verify they fail**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -k "anchor_localization_diagnostics" -q
```

Expected: FAIL because the diagnostics builder and dataclass do not exist.

- [ ] **Step 3: Implement localization dataclass and builder**

Add:

```python
@dataclass(frozen=True)
class AnchorLocalizationDiagnostics:
    anchor_localization: pd.DataFrame
    anchor_localization_summary: pd.DataFrame
    anchor_triplet_summary: pd.DataFrame
```

Add:

```python
def build_anchor_localization_diagnostics(
    residual_pairs: pd.DataFrame,
    *,
    model_ids: tuple[str, ...] = LOCALIZATION_MODEL_IDS,
    topk_values: tuple[int, ...] = LOCALIZATION_TOPK,
    min_neighbors: int = MIN_LOCAL_NEIGHBORS,
    min_triplets: int = MIN_TRIPLETS,
) -> AnchorLocalizationDiagnostics:
    ...
```

Implementation rules:

- filter to `model_ids`
- define `selection_pool` as:
  - `primary`: `reliability_pool == "primary"`
  - `high`: `is_high_reliability == True`
- expand filtered rows with `expand_pairs_to_anchor_rows(...)`
- build one row per `view_name x model_id x anchor_stimulus x selection_pool x scope`
- for each row calculate:
  - `n_neighbors`
  - `n_unique_dates`
  - `anchor_rank_spearman`
  - `topk_overlap_k3`
  - `topk_overlap_k5`
  - `triplet_consistency`
  - `n_triplets`
  - `score_status`
- require `n_neighbors >= min_neighbors` for rank and top-k scores
- require `n_triplets >= min_triplets` for triplet reporting
- write `anchor_localization_summary` aggregated by `view_name`, `model_id`,
  `selection_pool`, and `scope` with:
  - `n_anchor_rows`
  - `n_valid_rank_rows`
  - `n_valid_triplet_rows`
  - `mean_anchor_rank_spearman`
  - `median_anchor_rank_spearman`
  - `mean_topk_overlap_k3`
  - `mean_topk_overlap_k5`
  - `mean_triplet_consistency`
- write `anchor_triplet_summary` in long format with one row per
  `view_name x model_id x anchor_stimulus x selection_pool x scope`

- [ ] **Step 4: Run the localization summary tests**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -k "anchor_localization_diagnostics" -q
```

Expected: PASS.

- [ ] **Step 5: Commit the anchor-local diagnostics**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py
git commit -m "feat: add anchor localization diagnostics"
```

### Task 3: Wire Localization into Outputs, Figures, and Summary Text

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write failing output-writer tests for localization artifacts**

Extend the existing writer smoke test:

```python
def test_write_model_diagnosis_outputs_writes_localization_tables_and_figures(tmp_path):
    outputs = _tiny_model_diagnosis_outputs()

    written = write_model_diagnosis_outputs(outputs, tmp_path / "model_diagnosis")

    assert (written["tables_dir"] / "anchor_localization.csv").exists()
    assert (written["tables_dir"] / "anchor_localization_summary.csv").exists()
    assert (written["tables_dir"] / "anchor_triplet_summary.csv").exists()
    assert any(path.name.startswith("anchor_rank_alignment__") for path in written["figures_dir"].iterdir())
    assert any(path.name.startswith("anchor_neighborhood_overlap__") for path in written["figures_dir"].iterdir())
    assert any(path.name.startswith("localization_hotspots__") for path in written["figures_dir"].iterdir())
```

Add small synthetic localization frames into `_tiny_model_diagnosis_outputs()`.

- [ ] **Step 2: Run the writer smoke test to verify it fails**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -k "localization_tables_and_figures" -q
```

Expected: FAIL because the outputs dataclass and writer do not include the new artifacts.

- [ ] **Step 3: Extend the output dataclass and writer**

In `model_diagnosis.py`, add these fields to `ModelDiagnosisOutputs`:

```python
anchor_localization: pd.DataFrame
anchor_localization_summary: pd.DataFrame
anchor_triplet_summary: pd.DataFrame
```

Update `write_model_diagnosis_outputs(...)` to write:

- `tables/anchor_localization.csv`
- `tables/anchor_localization_summary.csv`
- `tables/anchor_triplet_summary.csv`

- [ ] **Step 4: Add the three localization figure builders**

Implement:

```python
def _plot_anchor_rank_alignment(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    ...


def _plot_anchor_neighborhood_overlap(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    ...


def _plot_localization_hotspots(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    ...
```

Figure rules:

- all three figure types should default to `selection_pool == "primary"` so
  each anchor appears once per `view x model`
- `anchor_rank_alignment`: use primary-pool `scope == "overall"` as x and
  primary-pool `scope == "cross_date"` as y after pivoting one row per anchor
- `anchor_neighborhood_overlap`: rank anchors by primary-pool
  `topk_overlap_k5`; keep the top 15 anchors so labels stay readable
- `localization_hotspots`: heatmap on the top 15 primary-pool anchors ranked by
  `anchor_rank_spearman`, with columns:
  - `anchor_rank_spearman_overall`
  - `anchor_rank_spearman_cross_date`
  - `topk_overlap_k3_overall`
  - `topk_overlap_k5_overall`
  - `triplet_consistency_overall`

Do not build figures for non-target models.

- [ ] **Step 5: Extend `run_summary.md` with a Shared Structure section**

Update `_write_model_diagnosis_summary(...)` so it includes a short section with:

- the best anchors by `anchor_rank_spearman`
- the best anchors by `topk_overlap_k5`
- a note about whether `cross_date` localization is preserved or collapses
- a reminder that Section 3 is anchor-local and does not imply a full globally
  aligned RDM

- [ ] **Step 6: Run the writer smoke test**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -k "localization_tables_and_figures" -q
```

Expected: PASS.

- [ ] **Step 7: Commit localization outputs and figures**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py
git commit -m "feat: write shared structure localization outputs"
```

### Task 4: Extend Orchestration and End-to-End Coverage

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py`
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Write a failing tiny end-to-end test that expects localization outputs**

Extend the existing CLI smoke test or orchestration smoke test:

```python
def test_run_model_diagnosis_review_populates_localization_outputs_on_tiny_fixture(tmp_path):
    fixture = _write_tiny_model_diagnosis_fixture(tmp_path)

    outputs = run_model_diagnosis_review(
        preprocess_root=fixture["preprocess_root"],
        matrix_path=fixture["matrix_path"],
        raw_metadata_path=fixture["raw_metadata_path"],
        base_results_root=fixture["base_results_root"],
        split_iterations=8,
        permutations=5,
        seed=123,
    )

    assert not outputs.anchor_localization.empty
    assert not outputs.anchor_localization_summary.empty
    assert not outputs.anchor_triplet_summary.empty
```

- [ ] **Step 2: Run the orchestration test to verify it fails**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -k "populates_localization_outputs" -q
```

Expected: FAIL because `run_model_diagnosis_review(...)` does not yet call the localization builder.

- [ ] **Step 3: Wire the localization builder into orchestration**

Inside `run_model_diagnosis_review(...)`:

1. keep current neural diagnostics
2. keep current model bundle builder
3. keep current residual diagnostics
4. call `build_anchor_localization_diagnostics(residuals.residual_pairs, ...)`
5. return the new localization tables inside `ModelDiagnosisOutputs`

No CLI flag changes are required for the first pass.

- [ ] **Step 4: Run the orchestration/localization tests**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -k "populates_localization_outputs or anchor_localization_diagnostics or localization_tables_and_figures" -q
```

Expected: PASS.

- [ ] **Step 5: Commit the orchestration hookup**

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py
git commit -m "feat: integrate shared structure localization"
```

### Task 5: Full Validation and Real Filtered-Batch Run

**Files:**
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py`
- Modify if needed: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py`

- [ ] **Step 1: Run the full focused test file**

Run:

```powershell
pixi run pytest tests/test_model_diagnosis.py -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent regression tests**

Run:

```powershell
pixi run pytest tests/test_model_space.py tests/test_rsa.py tests/test_rsa_cli_smoke.py tests/test_reliability.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the real filtered-batch diagnosis**

Run:

```powershell
pixi run python scripts/model_diagnosis_review.py --split-iterations 200 --permutations 2000 --seed 20260420
```

Expected:

- exit code `0`
- updated `results/202604_without_20260331/model_diagnosis/run_summary.md`
- new localization tables under `tables/`
- new localization figures under `figures/`

- [ ] **Step 4: Inspect the real localization outputs for contract failures**

Run:

```powershell
Get-ChildItem results/202604_without_20260331/model_diagnosis/tables/anchor_*.csv | Select-Object Name
Get-ChildItem results/202604_without_20260331/model_diagnosis/figures/*localization* , results/202604_without_20260331/model_diagnosis/figures/anchor_* | Select-Object Name
Get-Content results/202604_without_20260331/model_diagnosis/run_summary.md -TotalCount 200
```

Check:

- all three target models appear in localization tables
- both views appear
- at least some anchors have valid `cross_date` localization metrics
- the summary text distinguishes pooled overlap from anchor-local overlap

- [ ] **Step 5: Commit any real-run fixes**

If the real run exposed output-contract or edge-case issues:

```powershell
git add H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/src/bacteria_analysis/model_diagnosis.py H:/Process_temporary/WJH/bacteria_analysis/.worktrees/model-diagnosis/tests/test_model_diagnosis.py
git commit -m "test: validate shared structure localization"
```

If no fixes were needed, skip this commit.

## Final Verification Checklist

- [ ] `pixi run pytest tests/test_model_diagnosis.py -q` passes.
- [ ] `pixi run pytest tests/test_model_space.py tests/test_rsa.py tests/test_rsa_cli_smoke.py tests/test_reliability.py -q` passes.
- [ ] `pixi run python scripts/model_diagnosis_review.py --split-iterations 200 --permutations 2000 --seed 20260420` succeeds.
- [ ] `tables/anchor_localization.csv` exists and includes both views plus the three target models.
- [ ] `tables/anchor_localization_summary.csv` reports overall, within-date, and cross-date scopes.
- [ ] `tables/anchor_triplet_summary.csv` reports triplet counts and consistency.
- [ ] `figures/anchor_rank_alignment__<view>__<model>.png` exist for the three target models.
- [ ] `figures/anchor_neighborhood_overlap__<view>__<model>.png` exist for the three target models.
- [ ] `figures/localization_hotspots__<view>__<model>.png` exist for the three target models.
- [ ] `run_summary.md` contains a Shared Structure section that distinguishes anchor-local overlap from pooled RSA.
