# Stage 1 Raincloud Refinement and ECDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the raincloud candidate cleaner and more difference-focused, and add an ECDF candidate for focus-view same-versus-different comparisons.

**Architecture:** Reuse the existing focus-view same-vs-different preparation helpers and adjust only the figure-rendering layer. Keep the candidate-set contract additive by preserving the prior `violin_clean` and `boxen_points` variants while refining `raincloud` and adding `ECDF`.

**Tech Stack:** Python 3.11, pandas, matplotlib, seaborn, pytest

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: refine the raincloud renderer, add an ECDF renderer, and keep shared focus-view styling coherent.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability_outputs.py`
  Responsibility: cover any helper added for direct center-gap annotation.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability_cli_smoke.py`
  Responsibility: require the ECDF candidate in the figure output contract.

## Task 1: Add Shared Annotation Support

- [ ] Add a helper that computes the direct center-gap annotation from the same-vs-different summary table.
- [ ] Keep the result based on medians so the annotation is robust to the strong sample-size imbalance.

## Task 2: Refine Candidate Rendering

- [ ] Convert the raincloud figure to a lighter half-violin treatment with sparse point overlays and simplified subtitle text.
- [ ] Add an ECDF candidate with the same palette family and simplified title/subtitle hierarchy.
- [ ] Keep the existing `violin_clean` and `boxen_points` candidates available.

## Task 3: Verify

- [ ] Update smoke coverage for the ECDF candidate.
- [ ] Run targeted reliability tests.
- [ ] Run a real-data reliability CLI check and inspect the resulting candidate figures.
