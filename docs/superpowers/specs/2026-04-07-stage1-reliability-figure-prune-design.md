# Stage 1 Reliability Figure Prune Design

Date: 2026-04-07
Status: Approved in chat before implementation
Topic: Remove redundant reliability figures and keep only the concise visual set

## Goal

Prune the reliability figure set so it keeps only the figures that add visual value beyond the accompanying summary tables and later-stage outputs.

The statistics and tables remain unchanged.

## User-Approved Scope

Remove automatic generation of:

- `cross_view_reliability_comparison.png`
- `leave_one_individual_out_summary.png`
- `leave_one_date_out_summary.png`
- `split_half_summary.png`
- all `stimulus_distance_matrix__*.png`

Keep the remaining reliability figures unchanged.

## Chosen Approach

Stop emitting the retired figures from the writer and add a small cleanup step so reruns in the same output directory remove stale retired figure files.

This keeps the figure contract minimal without changing any underlying result tables.

## Success Criteria

The change is complete when:

- the retired figure files are no longer emitted
- reruns remove any stale copies of those retired files
- reliability tables remain intact
- tests pass and a real-data reliability rerun shows only the kept figure set
