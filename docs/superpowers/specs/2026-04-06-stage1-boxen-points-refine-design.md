# Stage 1 Boxen-Points Refinement Design

Date: 2026-04-06
Status: Approved in chat before implementation
Topic: Refine the focus-view boxen-points candidate for cleaner visual emphasis

## Goal

Keep `same_vs_different_distributions__boxen_points.png` as one of the preferred focus-view candidates, but reduce visual clutter so the distribution shift is more obvious at first glance.

This remains a figure-only change. No reliability statistics or table outputs should change.

## User-Approved Scope

The approved scope is:

1. reduce the visual weight of the sampled point layer
2. soften the boxen layer so it reads as structure, not as heavy outlines
3. simplify the title area so it emphasizes median shift rather than general gap text
4. keep the same output filename and the rest of the candidate set unchanged

## Chosen Approach

Refine only the `boxen_points` renderer:

- cap both groups to the same small sampled-point count
- make the points smaller and more transparent
- use softer dark-gray outlines instead of near-black emphasis
- simplify the title to the core comparison and move `Delta median` plus permutation `p` into the subtitle

## Success Criteria

The change is complete when:

- `same_vs_different_distributions__boxen_points.png` looks visibly cleaner than the prior version
- the plot still communicates the same-vs-different separation clearly
- all reliability tests pass and a real-data re-run succeeds
