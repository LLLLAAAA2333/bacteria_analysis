# Stage 1 Figure Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the weakest reliability figures with clearer focus-view visualizations while preserving the existing reliability tables and statistics.

**Architecture:** Keep the change isolated to the reliability output writer. Add small helper functions for focus-view per-stimulus summaries and date-by-stimulus availability, reuse the existing `focus_view` argument, and keep all filenames stable so downstream consumers do not need a contract change.

**Tech Stack:** Python 3.11, pandas, matplotlib, seaborn, pytest

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: replace the three figure builders and add any small helper functions needed for the new focus-view plots.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability.py`
  Responsibility: add focused coverage for any new figure-preparation helpers.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability_cli_smoke.py`
  Responsibility: keep smoke coverage aligned with the updated figure contract.

## Task 1: Add Focus-View Figure Data Helpers

- [ ] Add a helper that prepares per-stimulus same-versus-different mean distances for one focus view.
- [ ] Add a helper that builds a date-by-stimulus trial-count matrix from metadata.
- [ ] Write focused unit tests for both helpers before wiring them into the plotting layer.

## Task 2: Replace Weak Reliability Figures

- [ ] Rewrite `same_vs_different_distributions.png` as a focus-view violin plot with summary annotations.
- [ ] Replace the content of `cross_view_reliability_comparison.png` with a focus-view per-stimulus dumbbell plot.
- [ ] Replace the content of `overlap_neuron_qc_summary.png` with a date-by-stimulus availability heatmap.
- [ ] Keep filenames unchanged so the outward figure contract remains stable.

## Task 3: Verify

- [ ] Run the reliability-focused unit and CLI smoke tests.
- [ ] Run the full test suite if the targeted checks pass.
- [ ] Confirm that no reliability tables or summary keys changed as part of the figure refresh.
