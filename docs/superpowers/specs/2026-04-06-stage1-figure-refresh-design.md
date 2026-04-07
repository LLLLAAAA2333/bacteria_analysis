# Stage 1 Figure Refresh Design

Date: 2026-04-06
Status: Approved in chat before implementation
Topic: Replace weak reliability figures with clearer focus-view visualizations

## Goal

Refresh the reliability figure layer so the plots communicate stability more directly without changing any Stage 1 statistics, table schemas, or inference rules.

The main goals are:

- make the focus-view same-versus-different separation easier to see
- expose per-stimulus same-versus-different gaps instead of only pooled distributions
- show date-by-stimulus coverage directly so cross-date limits are visually obvious
- keep the change generic across all reliability runs

## Current Context

The existing reliability writer already computes the right statistics, but several current figures are weak:

- `same_vs_different_distributions.png` uses a boxplot on pooled distances, which hides the distribution shape and downplays visible separation when the y-scale is wide
- `cross_view_reliability_comparison.png` emphasizes view ranking, which is not the main interpretive need for this figure refresh
- `overlap_neuron_qc_summary.png` reports overlap counts but does not explain when date panels fail to share stimuli

The user wants a stronger visual story while keeping the figures portable across datasets. The new figures should therefore key off the configured `focus_view` instead of expanding into one plot per view.

## User-Approved Scope

The approved scope for this figure refresh is:

1. replace `same_vs_different_distributions.png` with a focus-view violin plot
2. replace `cross_view_reliability_comparison.png` with a focus-view per-stimulus same-versus-different comparison plot
3. replace `overlap_neuron_qc_summary.png` with a date-by-stimulus availability heatmap
4. keep the rest of the reliability figure set unchanged
5. keep the implementation generic across all reliability runs

## Chosen Approach

Use the existing reliability tables as the single source of truth and redesign only the writer layer.

This avoids changing the core analysis or output tables and keeps the new figures interpretable as pure visual summaries of already-computed results.

## Design Details

### Focus-View Same-Versus-Different Distribution

Keep the filename `same_vs_different_distributions.png`, but redraw it as a single focus-view violin plot.

The plot should:

- include only valid comparisons for the selected `focus_view`
- show `same` and `different` as the two categories
- use violin shapes so the distribution spread is visible
- overlay compact summary markers so the center is still legible
- annotate the plot with `distance_gap` and permutation `p_value`

This keeps the figure focused on the formal Stage 1 evidence chain while making the separation easier to read.

### Focus-View Per-Stimulus Same-Versus-Different Gap

Reuse the filename `cross_view_reliability_comparison.png`, but replace the content with a focus-view per-stimulus dumbbell plot.

For each stimulus:

- compute the mean distance for comparisons where that stimulus appears in `same` pairs
- compute the mean distance for comparisons where that stimulus appears in `different` pairs
- draw one line connecting the two means
- color the line and endpoints using the stimulus color when available
- sort stimuli by `different_mean_distance - same_mean_distance`

This gives a direct answer to whether the pooled same-versus-different effect is broad-based or driven by only a few stimuli.

### Date-By-Stimulus Availability Heatmap

Reuse the filename `overlap_neuron_qc_summary.png`, but replace the content with a heatmap of trial coverage.

The heatmap should:

- use `date` as rows
- use `stimulus` as columns
- encode the number of trials per `date × stimulus`
- remain readable when many stimuli exist by scaling the figure width

This figure is intentionally descriptive. Its job is to make panel overlap and date coverage obvious, especially in runs where leave-one-date-out scoring becomes weak or impossible.

## Data Requirements

The refresh must work from existing outputs already available to the writer:

- `core_outputs["comparisons"]`
- `core_outputs["metadata"]`
- `stats_outputs["final_summary"]`
- the existing `focus_view` argument

No new tables or core pipeline outputs are required.

## Testing

Update reliability smoke coverage so it still checks the expected figure filenames after the redesign.

Add unit coverage for any new helper functions that summarize:

- per-stimulus same-versus-different means for the focus view
- date-by-stimulus availability counts

## Success Criteria

The change is complete when:

- the three replaced figure filenames are still emitted
- `same_vs_different_distributions.png` becomes a focus-view violin plot
- `cross_view_reliability_comparison.png` becomes a focus-view per-stimulus same-versus-different figure
- `overlap_neuron_qc_summary.png` becomes a date-by-stimulus coverage heatmap
- all reliability tests pass without changing Stage 1 statistics
