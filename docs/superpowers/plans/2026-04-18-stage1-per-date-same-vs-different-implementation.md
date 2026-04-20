# Stage 1 Per-Date Same-vs-Different Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one focus-view same-vs-different reliability figure per date without changing the underlying reliability statistics.

**Architecture:** Keep the change isolated to `reliability_outputs.py`. Reuse the existing same-vs-different plotting path, add a date-specific comparison filter, write one figure per date, and expose the figure names in the run summary.

**Tech Stack:** Python 3.11, pandas, matplotlib, seaborn, pytest

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: filter focus-view comparisons by date, write the new per-date figures, and include their names in the run summary.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_outputs.py`
  Responsibility: cover the new date-level comparison filtering.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_cli_smoke.py`
  Responsibility: assert the new per-date figures and run summary field are written.

## Completion Status

Implemented. Automated verification on 2026-04-20: `pixi run test` passed with
254 tests.

### Task 1: Add Per-Date Reliability Figure Support

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability_outputs.py`
- Test: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_outputs.py`

- [x] **Step 1: Write the failing unit test**

```python
def test_build_focus_view_same_vs_different_plot_frame_for_date_filters_to_within_date_pairs():
    ...
```

- [x] **Step 2: Run it to verify it fails**

Run: `pixi run pytest tests/test_reliability_outputs.py -k per_date -q`
Expected: FAIL because the date filter helper does not exist yet.

- [x] **Step 3: Implement the date filter and per-date figure writer**

```python
def _build_focus_view_same_vs_different_plot_frame_for_date(...):
    ...
```

- [x] **Step 4: Run the targeted unit tests**

Run: `pixi run pytest tests/test_reliability_outputs.py -q`
Expected: PASS

### Task 2: Update CLI Output Coverage

**Files:**
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_cli_smoke.py`

- [x] **Step 1: Update the smoke expectations**

Add the per-date figure paths and assert they are listed in `run_summary.json`.

- [x] **Step 2: Run the CLI smoke test**

Run: `pixi run pytest tests/test_reliability_cli_smoke.py -q`
Expected: PASS

### Task 3: Real-Data Verification

**Files:**
- No code changes

- [ ] **Step 1: Rerun reliability on the current 202604 batch**

Run: `pixi run reliability --input-root data/202604/202604_preprocess --output-root results/202604`

- [ ] **Step 2: Confirm the per-date figures exist**

Check: `results/202604/reliability/figures/same_vs_different_by_date__*.png`

- [ ] **Step 3: Review the new run summary**

Check: `results/202604/reliability/run_summary.json`

Note: this cleanup commit verified the implementation with the full automated
test suite. Real-data output review remains an optional rerun because generated
`results/` files are ignored and not part of this commit.
