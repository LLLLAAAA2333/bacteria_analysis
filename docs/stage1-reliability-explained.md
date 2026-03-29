# Stage 1 Reliability Explained

Date: 2026-03-29
Audience: project notes for method explanation and result interpretation

## What Stage 1 Is Trying To Answer

Stage 1 is the reliability gate for the project.

It asks one main question:

> does the neural population contain a reproducible representation of stimulus identity?

In more concrete terms:

- are trials from the same stimulus more similar than trials from different stimuli?
- does that pattern still hold when we hold out one individual?
- does that pattern still hold when we hold out one date?

This stage is not yet about clustering, biochemical interpretation, or neuron importance.

## Inputs Used By Stage 1

Stage 1 reads the Stage 0 outputs:

- `data/processed/trial_level/trial_metadata.parquet`
- `data/processed/trial_level/trial_wide_baseline_centered.parquet`
- `data/processed/trial_level/trial_tensor_baseline_centered.npz`

Current real-data run used:

- `678` trials
- `22` neurons
- `45` time points
- `19` stimuli
- `37` individuals
- `6` dates

In Stage 1, the individual unit is not `worm_key` alone. It is:

`individual_id = "{date}__{worm_key}"`

This matters because the same `worm_key` can appear on different dates, and Stage 1 wants the held-out unit to be a real recording unit.

## The Four Representation Views

Stage 1 computes the same analysis in four different time windows:

1. `full_trajectory`: time points `0..44`
2. `on_window`: time points `6..15`
3. `response_window`: time points `6..20`
4. `post_window`: time points `16..44`

Formal reporting currently uses `response_window` as the default primary view, but all four views are computed.

## How Distance Is Computed

Each trial is a `neuron x time` array.

When two trials are compared:

1. only neurons observed in both trials are kept
2. within those neurons, only shared non-NaN time points are used
3. the remaining values are flattened into one vector per trial
4. correlation distance is computed

Correlation distance is:

`distance = 1 - correlation`

Correlation is computed with `np.corrcoef` on the two vectors. This is pearson correlation, which is scale-invariant and shift-invariant. The calculation fomula is:
`correlation = cov(X, Y) / (std(X) * std(Y))` or equivalently `correlation = (X_centered * Y_centered).sum() / (std(X) * std(Y) * (n - 1))`

Interpretation:

- smaller distance means more similar response patterns
- larger distance means less similar response patterns

Important:

- missing neurons are not imputed
- if overlap is insufficient, that comparison is excluded
- overlap counts are recorded as QC

## Same-vs-Different Distance

This is the simplest Stage 1 statistic.

For each view:

1. compute all pairwise trial distances
2. split them into:
   - `same`: the two trials have the same stimulus
   - `different`: the two trials have different stimuli
3. compare the two distributions

The key summary is:

`distance_gap = mean(different distances) - mean(same distances)`

Interpretation:

- if `distance_gap > 0`, same-stimulus trials are closer than different-stimulus trials
- larger positive values mean a clearer stimulus structure

For the current real-data run:

- `full_trajectory`: `0.1234`
- `on_window`: `0.1436`
- `response_window`: `0.1478`
- `post_window`: `0.1133`

All four are positive, which supports the basic Stage 1 claim.

## How LOIO Accuracy Is Computed

`LOIO` means `Leave-One-Individual-Out`.

This is the main cross-individual validation.

### Step-by-step

For one view and one held-out individual:

1. pick one `individual_id` and hold out all its trials
2. use all remaining individuals as the training set
3. for each stimulus in the training set, average the training trials to build one stimulus reference pattern
4. for each held-out trial:
   - compute its distance to every training stimulus reference
   - choose the stimulus with the smallest distance
   - this chosen stimulus is the prediction
5. mark the trial as correct if:

`predicted_stimulus == true_stimulus`

### Trial-level accuracy

For one held-out individual, if there are `n` scored trials:

`accuracy_individual = (# correct held-out trials) / n`

### Final LOIO mean accuracy

After repeating this for every held-out individual:

`LOIO mean accuracy = mean(individual accuracies)`

So this is not one giant pooled trial accuracy. It is the mean of held-out-individual accuracies.

That matters because it gives each held-out individual equal weight, instead of letting an individual with more trials dominate the result.

### Small example

Suppose one held-out individual has 3 scored trials:

- trial 1 true stimulus = `b3_2`, predicted = `b3_2` -> correct
- trial 2 true stimulus = `b4_2`, predicted = `b5_2` -> wrong
- trial 3 true stimulus = `b10_1`, predicted = `b10_1` -> correct

Then that individual's LOIO accuracy is:

`2 / 3 = 0.667`

If the next held-out individual has accuracy `0.333`, the mean across those two individuals is:

`(0.667 + 0.333) / 2 = 0.500`

### Current real-data LOIO mean accuracies

- `full_trajectory`: `0.3887`
- `on_window`: `0.2794`
- `response_window`: `0.3031`
- `post_window`: `0.3546`

How to interpret this:

- these values are not compared to a 50% baseline
- this is a roughly 19-class identification problem, so a rough random baseline is about `1 / 19 = 0.0526`

So `response_window = 0.3031` is modest in absolute value, but still much higher than random guessing.

## How LODO Accuracy Is Computed

Below I use the exact Stage 1 term `LODO`, which means `Leave-One-Date-Out`. If you write `LDIO`, this is the quantity we are referring to.

The logic is the same as LOIO, but the held-out unit is now `date`.

### Step-by-step

For one view and one held-out date:

1. hold out all trials from one date
2. train on all other dates
3. build one stimulus reference pattern per stimulus from the training dates
4. for each held-out trial on the held-out date:
   - compute distance to every training stimulus reference
   - choose the closest reference as the prediction
5. compute accuracy for that held-out date

### Final LODO mean accuracy

`LODO mean accuracy = mean(date accuracies)`

Again, this is the mean across held-out dates, not one pooled trial accuracy.

### Current real-data LODO mean accuracies

- `full_trajectory`: `0.1540`
- `on_window`: `0.1337`
- `response_window`: `0.1108`
- `post_window`: `0.1185`

These are clearly lower than LOIO.

Interpretation:

- the stimulus structure generalizes across dates, because these numbers are still above rough random chance
- but generalization across dates is much weaker than generalization across individuals
- that usually means date-level effects or date-linked sample composition matter

## What Happens To Excluded Trials

A held-out trial is not always scorable.

Examples:

- the training set may not contain that trial's true stimulus
- overlap with reference patterns may be too poor
- the true stimulus reference may be invalid

In those cases, the trial is marked excluded and does not count toward accuracy.

Stage 1 writes those exclusions to:

- `results/stage1_reliability/qc/excluded_holdout_trials.parquet`

## What Permutation Means Here

`Permutation` is the label-shuffle null test.

It answers:

> if there were actually no real stimulus structure, how large a distance gap might we get just by chance?

### What is shuffled

Stage 1 permutes `stimulus` labels within each `date`.

That means:

- the pairwise distances are left unchanged
- the date structure is preserved
- the number of trials per stimulus within each date is preserved

Then, for each permutation:

1. relabel trials with shuffled stimulus labels
2. recompute which pairs count as `same` and `different`
3. recompute the distance gap

This creates a null distribution of `distance_gap` values expected under random labels.

### The permutation p-value

The code computes:

`p = (1 + # null gaps >= observed gap) / (n_permutations + 1)`

For the current run, `n_permutations = 100`, so the smallest possible nonzero p-value is:

`1 / 101 = 0.009901`

Current results:

- all four views have `p_value = 0.009901`

Interpretation:

- the observed same-vs-different structure is stronger than every shuffled iteration
- with only 100 permutations, this means "significant within this permutation resolution"
- if you want finer p-values later, increase the number of permutations

## What Bootstrap Means Here

`Bootstrap` is not a null test. It is an uncertainty estimate.

It answers:

> if the dataset were re-sampled at the individual level, how much would the LOIO result vary?

### Why grouped bootstrap is needed

Trials from the same individual are not independent.

So Stage 1 does not resample rows one by one.

Instead, it resamples whole held-out individuals with replacement.

### What Stage 1 bootstraps

The current bootstrap is applied to LOIO scored results.

For one view:

1. take the scored LOIO trial results
2. group them by held-out individual
3. resample held-out individuals with replacement
4. compute each sampled individual's trial accuracy
5. average those sampled individual accuracies
6. repeat many times

This gives a bootstrap distribution of LOIO mean accuracy.

The reported confidence interval is the central 95% interval:

- lower bound = 2.5th percentile
- upper bound = 97.5th percentile

### Current LOIO bootstrap intervals

- `full_trajectory`: `0.3055` to `0.4744`
- `on_window`: `0.2125` to `0.3530`
- `response_window`: `0.2339` to `0.3780`
- `post_window`: `0.2723` to `0.4396`

Interpretation:

- bootstrap does not say whether the effect is random noise
- permutation handles that question
- bootstrap says how stable the LOIO estimate is across re-sampled individuals

So:

- `permutation` = "is there real structure beyond chance?"
- `bootstrap` = "how uncertain is the LOIO effect size estimate?"

## Split-Half Reliability

Split-half is supplementary.

For each stimulus, Stage 1 randomly splits its trials into two halves, builds one average pattern per half, and asks whether the two halves match on stimulus identity.

This is easier than LOIO or LODO because it does not require leaving out an entire individual or date.

That is why split-half can be high even when LOIO or LODO is only moderate.

## How To Read The Current Stage 1 Conclusion

For the default primary view `response_window`, the main numbers are:

- `distance_gap = 0.1478`
- `LOIO mean accuracy = 0.3031`
- `LODO mean accuracy = 0.1108`
- `permutation p_value = 0.009901`
- `bootstrap LOIO CI = 0.2339 .. 0.3780`

This supports the following interpretation:

1. same-stimulus trials are consistently closer than different-stimulus trials
2. the structure generalizes across held-out individuals
3. the structure also generalizes across held-out dates, but more weakly
4. the signal is real, but date-related variation is nontrivial

So the current Stage 1 result is best described as:

`a cautious pass`

Not because the effect is absent, but because:

- LOIO is clearly above chance yet not extremely high
- LODO is weaker still
- date-linked variation remains an important limitation

## Supplementary Analyses Added After The Main Stage 1 Run

Three supplementary analyses are now also implemented:

1. within-date cross-individual same-vs-different
2. per-date LOIO
3. stimulus-by-stimulus distance matrices

These do not change the formal Stage 1 pass/fail logic, but they help explain where the date effect is coming from and which stimuli are close to each other.

## Main Files To Inspect

Core output summary:

- `results/stage1_reliability/tables/final_summary.parquet`
- `results/stage1_reliability/run_summary.md`

Main formal figures:

- `results/stage1_reliability/figures/same_vs_different_distributions.png`
- `results/stage1_reliability/figures/leave_one_individual_out_summary.png`
- `results/stage1_reliability/figures/leave_one_date_out_summary.png`

Supplementary outputs:

- `results/stage1_reliability/tables/within_date_cross_individual_same_vs_different_summary.parquet`
- `results/stage1_reliability/tables/per_date_loio_summary.parquet`
- `results/stage1_reliability/tables/stimulus_distance_pairs.parquet`

## One-Sentence Summary

Stage 1 asks whether stimulus identity is reproducibly encoded in the neural population; `LOIO` and `LODO` measure how well that identity can be recovered on held-out individuals or dates, `permutation` tests whether the observed structure is stronger than chance, and `bootstrap` measures how uncertain the LOIO estimate is.
