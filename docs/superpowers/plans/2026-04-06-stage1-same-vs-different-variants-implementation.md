# Stage 1 Same-vs-Different Variant Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit three focus-view same-versus-different candidate figures with suffixed filenames so the user can choose the final publication-facing visual.

**Architecture:** Keep the change isolated to the reliability output writer. Add a small shared preparation layer for focus-view plotting, then render three figure variants from the same prepared data and update smoke coverage to match the new file contract.

**Tech Stack:** Python 3.11, pandas, matplotlib, seaborn, pytest

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: add shared same-vs-different plot helpers and emit the three suffixed candidate figures.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability_outputs.py`
  Responsibility: cover any new shared helper logic used by the plot variants.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability_cli_smoke.py`
  Responsibility: align smoke expectations with the new suffixed figure outputs.

## Task 1: Add Shared Plot Preparation

- [ ] Add a helper that filters comparisons to one focus view and prepares `same`/`different` plot labels.
- [ ] Add a helper that summarizes group counts, medians, and IQR for the candidate plots.
- [ ] Add a helper that deterministically downsamples points for overlay layers.

## Task 2: Emit Three Candidate Figures

- [ ] Replace the single `same_vs_different_distributions.png` output with `__raincloud`, `__violin_clean`, and `__boxen_points` variants.
- [ ] Keep palette, subtitle statistics, and focus-view-only behavior consistent across the three variants.
- [ ] Keep the rest of the reliability writer unchanged.

## Task 3: Verify

- [ ] Update unit coverage for the shared helper layer.
- [ ] Update CLI smoke coverage for the three suffixed files.
- [ ] Run targeted reliability tests, then run a real-data reliability CLI check to confirm the new figures are written.
