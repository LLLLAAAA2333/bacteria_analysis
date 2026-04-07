# Stage 1 Raincloud Refinement and ECDF Design

Date: 2026-04-06
Status: Approved in chat before implementation
Topic: Refine the same-vs-different raincloud candidate and add an ECDF candidate

## Goal

Improve the publication-readiness of the focus-view same-versus-different candidate figures by making the group shift easier to see and the overall layout cleaner.

The change remains purely visual. Reliability statistics, tables, and the rest of the Stage 1 figure set stay unchanged.

## Current Context

The first multi-version candidate set improved flexibility, but the user identified two remaining issues in `same_vs_different_distributions__raincloud.png`:

- the difference signal is diluted by too many layers and too many points
- the visual hierarchy at the top of the figure is too dense

The user also requested an ECDF comparison because it can make the overall distribution shift clearer when group sizes are highly imbalanced.

## User-Approved Scope

The approved scope is:

1. refine the `raincloud` candidate using the lighter visual treatment recommended in chat
2. add a new `ECDF` candidate figure
3. keep the existing `violin_clean` and `boxen_points` candidates available
4. keep all outputs restricted to `focus_view`
5. keep all tables and non-target figures unchanged

## Chosen Approach

Use the existing focus-view plot-preparation helpers and change only the rendering layer:

- refine the raincloud figure into a lighter half-violin layout with very sparse points
- simplify the title/subtitle hierarchy
- add a direct median-gap annotation
- add a separate ECDF figure as a fourth candidate

This keeps the contract stable at the analysis level while expanding the editorial options for figure selection.

## Design Details

### Refined Raincloud

The refined `same_vs_different_distributions__raincloud.png` should:

- use a half violin instead of a full violin
- keep a narrow box overlay for quartiles and median
- use a much smaller sampled point layer so points act as texture, not noise
- use a white background with very light horizontal grid lines
- use a cleaner title plus a shorter subtitle focused on `Gap` and permutation `p`
- add a direct in-plot median-gap label so the group shift is explicit

The styling goal is clarity first: readers should see the center shift before they inspect density detail.

### ECDF Candidate

Add `same_vs_different_distributions__ecdf.png` as a fourth candidate.

The ECDF figure should:

- compare only `same` and `different` for the selected `focus_view`
- use one line per group with the same palette family as the other candidates
- optionally mark the group medians
- keep the same simplified title/subtitle hierarchy

This figure is intended to foreground the overall rightward shift of the `different` distribution.

## Testing

Update the reliability CLI smoke test to require the new `ECDF` file.

Add focused unit coverage for any new helper used to compute direct center-gap annotations.

## Success Criteria

The change is complete when:

- `same_vs_different_distributions__raincloud.png` is visibly lighter and more focused on group separation
- `same_vs_different_distributions__ecdf.png` is emitted
- the existing `violin_clean` and `boxen_points` candidates still emit
- all outputs remain focus-view-only
- targeted reliability tests and a real-data CLI run pass
