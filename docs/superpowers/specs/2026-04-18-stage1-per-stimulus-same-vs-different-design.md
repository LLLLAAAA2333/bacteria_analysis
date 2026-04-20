# Stage 1 Per-Stimulus Same-vs-Different Figures

Date: 2026-04-18
Status: Implemented
Topic: Restore pooled and per-date per-stimulus same-vs-different reliability figures

## Goal

Add one pooled and one per-date per-stimulus same-vs-different figure so the
user can inspect which stimuli and which dates show weak separation.

## User-Approved Scope

The approved scope is:

1. write one pooled per-stimulus figure for the selected `focus_view`
2. write one additional per-stimulus figure per available `date`
3. keep the existing same-vs-different pooled and per-date distribution figures
   unchanged
4. do not change the underlying reliability statistics or tables

## Chosen Approach

Reuse the existing focus-view per-stimulus summary helper and plotting function
already present in `reliability_outputs.py`.

The pooled figure uses all valid focus-view comparisons. Each per-date figure
filters to valid focus-view comparisons where both trials come from the same
date and then computes stimulus-level same and different mean distances for that
date only.

## Success Criteria

The change is complete when:

- `reliability` writes `per_stimulus_same_vs_different__pooled.png`
- `reliability` writes `per_stimulus_same_vs_different__<date>.png` for each
  available date
- `run_summary.json` lists the pooled and per-date per-stimulus figure names
- tests pass and a 202604 reliability rerun produces the figures

## Verification

- 2026-04-20: `pixi run test` passed with 254 tests.
