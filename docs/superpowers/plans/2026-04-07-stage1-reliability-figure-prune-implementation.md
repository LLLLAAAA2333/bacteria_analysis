# Stage 1 Reliability Figure Prune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove low-value reliability figures while keeping all reliability tables and summaries unchanged.

**Architecture:** Narrow the reliability writer's figure contract and add a stale-file cleanup helper for retired figures. Keep the table-writing path and run summary path untouched.

**Tech Stack:** Python 3.11, pathlib, pandas, matplotlib, seaborn, pytest

---

## File Structure

- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/src/bacteria_analysis/reliability_outputs.py`
  Responsibility: stop writing retired figure files and remove stale retired files on rerun.
- Modify: `H:/Process_temporary/WJH/bacteria_analysis/.worktrees/neutral-analysis-contract/tests/test_reliability_cli_smoke.py`
  Responsibility: align smoke coverage with the smaller reliability figure contract.

## Task 1: Prune Reliability Figure Outputs

- [ ] Remove retired figure writes from the reliability writer.
- [ ] Add stale retired figure cleanup for reruns in the same output directory.

## Task 2: Verify

- [ ] Update the CLI smoke test expectations.
- [ ] Run targeted reliability tests.
- [ ] Re-run the real-data reliability CLI and confirm the pruned figure list.
