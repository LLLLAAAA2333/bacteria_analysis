# Stage 1 Per-Stimulus Same-vs-Different Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore pooled and per-date per-stimulus same-vs-different reliability figures without changing reliability statistics.

**Architecture:** Keep the change in `reliability_outputs.py`. Reuse the existing per-stimulus summary and plotting logic, add a date-filtered path for per-date figures, and expose the filenames in the run summary.

**Tech Stack:** Python 3.11, pandas, matplotlib, seaborn, pytest

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: write pooled and per-date per-stimulus figures and include them in summaries.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_outputs.py`
  Responsibility: cover date-filtered per-stimulus summaries.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/tests/test_reliability_cli_smoke.py`
  Responsibility: assert the new pooled and per-date figure files and summary fields.

## Completion Status

Implemented. Automated verification on 2026-04-20: `pixi run test` passed with
254 tests.

### Task 1: Add Per-Stimulus Writer Support

- [x] Add a date-filtered per-stimulus summary helper.
- [x] Write pooled and per-date per-stimulus figures from the reliability writer.

### Task 2: Update Tests

- [x] Add a unit test for the date-filtered per-stimulus summary.
- [x] Update the CLI smoke test expectations and run summary checks.

### Task 3: Real-Data Verification

- [x] Run targeted reliability tests.
- [ ] Rerun `202604` reliability and confirm the pooled and per-date per-stimulus figures.

Note: this cleanup commit verified the implementation with the full automated
test suite. Real-data output review remains an optional rerun because generated
`results/` files are ignored and not part of this commit.
