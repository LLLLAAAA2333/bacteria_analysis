# Neural overview: analysis decisions and review notes

Executed with `.pixi/envs/default/bin/python reports/exploration_20260929/code/02_neural_overview.py`; seed 20260929. Outputs were regenerated after the preparation tables were updated at 23:02 on 2026-09-29. The JSON metadata records hashes of the exact intermediate input files.

## Questions and scope

1. Which neuron classes show animal-consistent differences between strains within an observed date, large enough in the measured calcium units to motivate chemical follow-up?
2. Do the six two-date strains support transferring those observations across dates?
3. Is an attractive broad-window response mostly due to three particularly large strain effects?

All 13 classes and three fixed broad windows (0–10 s stimulus present, 10–30 s after stimulus, 0–40 s full response) were examined. No peak-time optimization, clustering, chemical feature selection, significance tests, or p-value selection were performed here. The choices of ADF/AWCON for follow-up and the example curves are data-driven exploratory choices.

## Units and correlation checks

The data contain 106 strains, 112 strain–date combinations, nine dates and 49 actual animals identified by `(date, worm_key)`. Animal means average available bilateral channels within each trial, then repeated trials. An animal receives 11–13 strains. It is never resampled separately for each strain, neuron or time window.

For each of 500 balanced random partitions, all observations from the same actual animal enter the same half, jointly across every strain and neuron. Animal-half means are formed for each strain–date and neuron. Within each date, subtract the two halves' respective mean over the identical observed strain set. The reported correlation pools these centered strain–date effects. The 5th–95th partition percentiles describe sensitivity to an animal split, **not confidence intervals or independent validation**. Dates were not randomly sampled; strain and date effects are not identified by this calculation.

A potentially falsifying sensitivity check drops the three strains with the largest absolute date-centered full-animal mean, for each neuron/window. Dates are re-centered **after dropping these strains**, preventing removed extremes from leaving artificially correlated date offsets. A second sensitivity check requires at least two measured animals in both halves. Primary split pair counts range from 86–112 across classes/partitions; the stricter check ranges from 49–112. Full counts are retained in the tables.

## Judgments for prioritization

- **ADF intensity warrants follow-up.** Stimulus-window centered split r has median 0.868 (partition 5–95%: 0.833–0.908), dropping the three largest date-centered effects leaves 0.824. The broad post-stimulus and full-window correlations remain 0.795 and 0.800. A024 and A025 have similar high ADF date means, but the uncertainty is wide; this is limited replication support, not proof of equivalence.
- **AWCON strong responses warrant a distinct follow-up.** Stimulus split r is 0.924 (0.881–0.955); dropping the three strongest date-centered effects leaves 0.879. Strong responses occur in several animals for several strains, rather than a single outlier curve. For A247, the four measured animals have stimulus means 2.838, 2.611, 3.068 and 3.138 ΔF/F0. Their ADF is weak. This motivates testing whether AWCON activation is simply overall response gain versus a distinct response composition. None of the six cross-date strains covers the strong AWCON group; cross-date robustness of that group remains untested.
- **Do not elevate all attractive time courses to strain-stable identities.** AWB post-stimulus animal consistency is substantial (r 0.759, drop-three 0.730), yet the limited repeat strains show large date changes. A044 ADF is also a clear counterexample to broad stability: 0–10 s mean falls from 0.315 (five animals, 20260414) to 0.010 (six animals, 20260429), difference −0.305; animal-bootstrap 95% percentile interval conditional on these two dates is −0.354 to −0.248. This cannot distinguish date, growth/stimulus preparation, cohort or other session factors.

AWA and ASH are also animal-consistent candidates (stimulus split medians 0.917 and 0.873), but no additional chemical route was opened merely to increase the number of candidate findings. ASG has weak evidence for strain differentiation in these windows (stimulus split median 0.014); this is lack of support under the current coverage and summary, not absence of neural function.

## Cross-date uncertainty

The six strains A011, A013, A014, A024, A025 and A044 each have only two dates, and are not a random set of independent validation strains. The cross-date table gives all 234 neuron/window contrasts. Its 1,000 bootstrap replicates sample whole animals within each observed date using common weights across strains. Percentile intervals are conditional on those dates and do not estimate a population-of-dates effect. The intervals are descriptive and not corrected for selection or 234 comparisons. No significance claims are made from them.

## Figures and traceability

- `figures/01_neural_overview.png` and `.pdf`: 112 rows preserve both dates of repeated strains; separate stimulus/post-stimulus panels; same linear ΔF/F0 color scale for every neuron and panel. Horizontal separators and date labels expose the batch layout. Each cell is an equal-animal mean (2–7 animals, coverage in `overview_heatmap_data.csv`). No uncertainty is encoded in this overview; animal curves supply direct evidence.
- `figures/01_selected_animal_curves.png` and `.pdf`: selected A024/A025/A044/A247 examples, ADF/AWB/AWCON columns. Every available animal curve is shown, thin lines; date means, thick lines; no bands. Y axes are shared within a neuron column, and may differ across columns. Grey marks 0–10 s stimulus. The data table is `overview_selected_animal_curves.csv`. These are calcium curves processed upstream with fitted F0 and then averaged over trials/channels, not raw fluorescence or firing traces.
- `overview_diagnostic.csv`: compact 39-row summary including effect range, animal dispersion, split sensitivity, excluded strain identities, and descriptive between-date variance fraction. The latter also reflects the non-random strain composition by date and is not a causal batch variance estimate.
- `overview_split_half_draws.csv` retains all 58,500 records, so every aggregate can be checked.
- `overview_cross_date.csv` retains all six two-date contrasts by neuron/window.

## Verification and repairs

Executed invariants: all 7,063 animal curves reproduce all three metric-table windows to 1e-12 tolerance; expected 58,500 split rows, 39 diagnostic rows, 234 cross-date rows and 1,456 heatmap cells; every date–strain has 13 heatmap cells; correlations are within [−1,1]; bootstrap interval endpoints are ordered. Both figures were opened and visually checked. Overlapping titles in the first drafts were corrected and the outputs regenerated. A review of the drop-three sensitivity found that date centering must be repeated after exclusion; the script was corrected before final results were delivered. No input data or unrelated files were changed.
