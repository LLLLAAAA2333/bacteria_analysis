# Stage 1 Cross-View Slope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep only the selected same-vs-different candidate figures and redesign `cross_view_reliability_comparison.png` as a clean paired slope plot.

**Architecture:** Reuse the existing focus-view per-stimulus summary helper and update only the reliability figure writer. Narrow the same-vs-different figure contract at the writer boundary and keep the rest of the output schema stable.

**Tech Stack:** Python 3.11, pandas, matplotlib, seaborn, pytest

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: narrow the same-vs-different output set and redraw the per-stimulus comparison figure.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability_cli_smoke.py`
  Responsibility: align smoke expectations with the narrowed figure contract.

## Task 1: Narrow Same-vs-Different Output Contract

- [ ] Stop emitting the `raincloud` and `violin_clean` candidate files.
- [ ] Keep only `boxen_points` and `ecdf` as same-vs-different candidates.

## Task 2: Redesign Per-Stimulus Comparison

- [ ] Replace the horizontal dumbbell layout with a vertical paired slope plot.
- [ ] Remove stimulus labels and stimulus-color encoding.
- [ ] Keep a compact summary overlay so the median shift remains obvious.

## Task 3: Verify

- [ ] Update smoke coverage.
- [ ] Run targeted reliability tests.
- [ ] Re-run the real-data reliability CLI and inspect the refreshed figures.
