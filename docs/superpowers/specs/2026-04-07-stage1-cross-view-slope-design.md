# Stage 1 Cross-View Slope Redesign

Date: 2026-04-07
Status: Approved in chat before implementation
Topic: Narrow the same-vs-different candidate set and redesign the per-stimulus comparison figure

## Goal

Keep only the two preferred same-vs-different candidate figures and replace the current per-stimulus dumbbell plot with a cleaner paired slope plot that emphasizes the consistent `same < different` pattern across stimuli.

The change remains in the figure writer only. Reliability statistics and tables stay unchanged.

## User-Approved Scope

The approved scope is:

1. retain only `same_vs_different_distributions__boxen_points.png` and `same_vs_different_distributions__ecdf.png`
2. stop emitting the `raincloud` and `violin_clean` same-vs-different variants
3. redesign `cross_view_reliability_comparison.png` so:
   - x-axis is `same` vs `different`
   - y-axis is distance
   - each stimulus remains a faint paired line
   - stimulus identity is not encoded or labeled
4. keep the rest of the reliability output contract unchanged

## Chosen Approach

Reuse the existing focus-view per-stimulus summary table and redraw the figure as an overlaid paired slope plot:

- one faint neutral line per stimulus
- small endpoint markers aligned to `same` and `different`
- no stimulus labels and no stimulus colors
- an optional compact black summary overlay for median and IQR

This makes the direction of the effect the first thing the reader sees.

## Success Criteria

The change is complete when:

- only `boxen_points` and `ecdf` remain in the same-vs-different candidate set
- `cross_view_reliability_comparison.png` becomes a vertical paired slope plot
- the new figure no longer labels or colors individual stimuli
- tests pass and a real-data reliability re-run succeeds
