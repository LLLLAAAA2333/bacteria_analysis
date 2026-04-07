# Stage 1 Boxen-Points Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the focus-view boxen-points candidate cleaner and more difference-focused without changing any reliability statistics.

**Architecture:** Keep the change isolated to the same-vs-different plotting helpers in `reliability_outputs.py`. Reuse the existing focus-view summary helpers and only adjust point sampling, boxen styling, and the title/subtitle text for the boxen candidate.

**Tech Stack:** Python 3.11, pandas, matplotlib, seaborn, pytest

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: tune the `boxen_points` renderer.
- Verify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability_outputs.py`
  Responsibility: keep helper coverage passing after the visual adjustment.

## Task 1: Refine Boxen-Points Rendering

- [ ] Reduce the sampled point layer to a smaller equal-count overlay.
- [ ] Soften box outlines and fill emphasis.
- [ ] Change the subtitle to use `Delta median` plus permutation `p`.

## Task 2: Verify

- [ ] Run targeted reliability tests.
- [ ] Re-run the real-data reliability CLI and inspect the refreshed boxen-points figure.
