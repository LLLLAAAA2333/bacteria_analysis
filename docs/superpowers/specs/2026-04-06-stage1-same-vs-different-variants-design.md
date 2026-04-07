# Stage 1 Same-vs-Different Variant Figures Design

Date: 2026-04-06
Status: Approved in chat before implementation
Topic: Emit multiple focus-view same-vs-different figure variants for manual selection

## Goal

Generate several candidate versions of the focus-view same-versus-different figure so the user can manually choose the most publication-ready visual treatment.

The change must stay at the figure-writer layer only. No Stage 1 statistics, tables, or inference rules should change.

## Current Context

The current focus-view violin plot improved readability over the old boxplot, but the user still finds the two-group comparison visually weak.

The main issues are:

- the large sample imbalance between `same` and `different` makes the pooled distributions feel visually heavy
- the on-plot annotation box competes with the distribution itself
- a single plotting style makes it harder to judge which visual treatment is best for publication

## User-Approved Scope

The approved scope is:

1. generate multiple candidate figures instead of one default `same_vs_different_distributions.png`
2. keep the figure generic across all reliability runs
3. keep the figure restricted to `focus_view`
4. use suffixed filenames so the user can manually select the final figure
5. leave reliability tables and all non-target figures unchanged

## Chosen Approach

Add a shared focus-view plotting prep layer, then render three separate candidate figures from the same prepared data:

- `same_vs_different_distributions__raincloud.png`
- `same_vs_different_distributions__violin_clean.png`
- `same_vs_different_distributions__boxen_points.png`

This keeps the statistical contract stable while making the figure contract more flexible for editorial selection.

## Design Details

### Shared Plot Preparation

The writer should prepare the following once and reuse it across all three variants:

- valid focus-view comparisons
- `same`/`different` labels
- per-group counts, medians, and IQR
- formatted subtitle text with `distance_gap`, permutation `p`, and the central y-axis window
- a deterministic downsample of points for point overlays

The plot layer should move summary text out of a large in-panel annotation box and into lighter title or subtitle text.

### Variant 1: Raincloud

`same_vs_different_distributions__raincloud.png` should use:

- violin density as the base layer
- a narrow box overlay for quartiles and median
- a deterministic sampled point layer for texture

This is the most expressive candidate and should emphasize separation while still showing distribution shape.

### Variant 2: Clean Violin

`same_vs_different_distributions__violin_clean.png` should use:

- a cleaner violin plot without the old annotation box
- explicit median and IQR markers
- normalized width and a tighter central quantile window

This is the most conservative candidate because it stays closest to the current figure form.

### Variant 3: Boxen Plus Points

`same_vs_different_distributions__boxen_points.png` should use:

- a boxen-style summary layer
- deterministic sampled points
- the same title, subtitle, palette, and axis treatment as the other variants

This candidate should make the location difference more direct, even if it sacrifices some density detail.

## Output Contract

The writer should stop emitting the unsuffixed `same_vs_different_distributions.png` file.

Instead, it should emit the three suffixed files above and expose them as separate written-path entries.

## Testing

Update reliability CLI smoke coverage so it checks for the three suffixed same-versus-different figures and no longer requires the unsuffixed file.

Add focused unit coverage for any new shared preparation helpers introduced for the plot variants.

## Success Criteria

The change is complete when:

- all three suffixed same-versus-different candidate figures are emitted
- the unsuffixed same-versus-different figure is no longer required
- each candidate uses only `focus_view`
- reliability tables and run summary fields remain unchanged
- targeted reliability tests still pass
