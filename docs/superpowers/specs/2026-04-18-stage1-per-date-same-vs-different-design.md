# Stage 1 Per-Date Same-vs-Different Figures

Date: 2026-04-18
Status: Implemented
Topic: Restore date-level same-vs-different visibility in reliability outputs

## Goal

Add one focus-view `same vs different` distribution figure per `date` so the
user can quickly judge whether one recording date is driving weak reliability.

## User-Approved Scope

The approved scope is:

1. keep the existing pooled reliability figures unchanged
2. add one new figure per `date`
3. each per-date figure should show the overall `same` versus `different`
   distance distribution for the selected `focus_view`
4. do not further split the per-date figures by `stimulus`

## Chosen Approach

Reuse the existing focus-view same-vs-different plotting style and add a
date-specific filter at the writer boundary.

For each date present in `metadata`:

- keep only comparisons from the selected `focus_view`
- keep only valid comparisons where both trials come from the same date
- write one boxen-plus-points summary figure under the reliability `figures/`
  directory
- if a date has no valid comparisons, still emit a placeholder figure so the
  missing support is obvious

## Success Criteria

The change is complete when:

- `reliability` writes one per-date same-vs-different figure per available date
- the current pooled figure contract remains intact
- `run_summary.json` lists the new per-date figure names
- tests pass and a real-data reliability rerun produces the new outputs

## Verification

- 2026-04-20: `pixi run test` passed with 254 tests.
