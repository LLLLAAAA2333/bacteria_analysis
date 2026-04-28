# Current Neural-Chemical RSA Summary

Date: 2026-04-24
Status: interim interpretation summary for slides and discussion; not paper-ready wording

## One-Sentence Conclusion

Gut bacterial chemical space shows a modest but statistically reliable alignment
with the C. elegans neural response space. The alignment is partial rather than
global, and recent taxonomy-class analyses suggest that the shared structure is
concentrated in specific chemical subspaces, especially Purine nucleosides.

## Recommended Slide Summary

- Gut bacterial metabolite profiles define a chemical representational space.
- C. elegans neural responses define a neural response space.
- The two spaces show a modest but significant RDM alignment, around `0.2`,
  indicating non-random shared structure rather than a strong global match.
- The signal is stronger within dates and weaker or mixed across dates, so the
  relationship is partly date-sensitive.
- Taxonomy-class analysis localizes the signal: Purine nucleosides is the most
  stable neural-associated chemical class across fixed-class permutation and
  resampling tests.

Suggested slide title:

> Partial alignment between gut bacterial chemical space and neural response space

## Current Evidence

### 1. Full chemical space: weak but non-random alignment

On the filtered `202604_without_20260331` batch, the current broad baseline is:

- neural RDM: non-ASE L/R-merged trial-median correlation RDM
- chemical RDM: `QCRSD <= 0.2 + log2(matrix) + Euclidean`
- retained stimuli: `52`
- retained metabolites in full QC20 chemical RDM: `267`

Key RSA values:

| Neural view | Scope | RSA |
| --- | --- | ---: |
| response_window | all pairs | `0.1950` |
| response_window | within-date only | `0.3111` |
| response_window | cross-date only | `0.0003` |
| response_window | date-pair-stratified all pairs | `0.1679` |
| full_trajectory | all pairs | `0.2221` |
| full_trajectory | within-date only | `0.3502` |
| full_trajectory | cross-date only | `0.0210` |
| full_trajectory | date-pair-stratified all pairs | `0.1344` |

Interpretation:

The full chemical RDM does not strongly explain the neural RDM, but the observed
alignment is above null expectations in the relevant permutation analyses. The
right conclusion is therefore "modest but reliable", not "no relationship" and
not "strong chemical explanation".

### 2. Date structure: within-date alignment is stronger

Across the current date-controlled review, within-date RSA is consistently
higher than cross-date RSA. This means the chemical-neural relationship is not
uniform across all stimulus pairs.

The safest interpretation is:

- there is real shared neural-chemical structure;
- part of it is date-sensitive or pair-composition-sensitive;
- repeated anchor-stimulus controls argue against a dominant global date
  artifact, but they do not remove the possibility of localized date-pair drift.

### 3. Taxonomy classes: signal is concentrated in stable subspaces

The taxonomy class stability review used:

- `2000` fixed-class permutations
- `500` reselection resamples
- `2000` full-search permutations as an internal diagnostic

The strongest and most stable class is Purine nucleosides:

| Class | response_window RSA | observed percentile | top-3 reselection frequency | fixed signal pass |
| --- | ---: | ---: | ---: | --- |
| Purine nucleosides | `0.2193` | `100.00` | `0.888` | true |
| Pyridines and derivatives | `0.1961` | `100.00` | `0.712` | true |
| Pyrimidine nucleotides | `0.1857` | `99.70` | `0.624` | true |
| Indoles and derivatives | `0.1669` | `99.90` | `0.324` | true |
| Organooxygen compounds | `0.1504` | `98.45` | `0.252` | true |

The internal full-search diagnostic gives the best-class search-corrected
p-value as `0.012994`. Because the current output is still exploratory, this
should be used as support for stability, not as final paper-level evidence.

Interpretation:

The global chemical space is only modestly aligned with neural space, but the
alignment is not evenly distributed across the metabolome. Purine nucleosides
appear to carry a stable part of the neural-associated chemical structure.

### 4. Subspace and supervised routes

Subset-style models can raise RSA above the broad full-chemical baseline. For
example, the fixed weighted-fusion model reaches:

| Neural view | Scope | RSA |
| --- | --- | ---: |
| response_window | all pairs | `0.2513` |
| response_window | within-date only | `0.3462` |
| full_trajectory | all pairs | `0.2867` |
| full_trajectory | within-date only | `0.3953` |

However, the supervised chemical subspace search is currently archived as a
negative route. It selected very small chemical sets, was hard to explain across
validation scopes, and should not be presented as the main result.

## Recommended Wording

For PPT:

> Gut bacterial chemical space is modestly but significantly aligned with the
> C. elegans neural response space. The effect is partial rather than global,
> and taxonomy-class analysis suggests that the shared structure is concentrated
> in specific chemical subspaces, especially Purine nucleosides.

More cautious version:

> We find exploratory evidence for a non-random relationship between gut
> bacterial metabolite space and neural response space. The global RDM alignment
> is modest, but stable class-level signals suggest that part of the shared
> structure is localized to Purine nucleosides and related chemical subspaces.

## What Not To Overclaim

Avoid saying:

- the chemical space explains neural geometry;
- Purine nucleosides cause the neural response pattern;
- the current taxonomy-class result is paper-ready;
- cross-date generalization is solved;
- the supervised selected-metabolite model is the main evidence.

Prefer saying:

- modest but reliable alignment;
- partial neural-chemical shared structure;
- localized chemical subspaces;
- exploratory taxonomy-class evidence;
- Purine nucleosides as the strongest current candidate class.

## Main Source Files

- `results/202604_without_20260331/date_controlled_rsa_review/run_summary.md`
- `results/202604_without_20260331/date_controlled_rsa_review/tables/rsa_all_vs_within_date_vs_cross_date.csv`
- `results/202604_without_20260331/taxonomy_class_stability_review/run_summary.md`
- `results/202604_without_20260331/taxonomy_class_stability_review/figures/03_top_class_rdm_comparison.png`
- `results/202604_without_20260331/taxonomy_class_stability_review/figures/04_taxonomy_class_stability_summary.png`
- `results/202604_without_20260331/taxonomy_class_stability_review/figures/05_class_chemical_rdm_similarity_matrix.png`
- `results/202604_without_20260331/date_controlled_rsa_review/figures/neural_chemical_rdm_foundation__response_window_full_vs_top_class_rdms.png`

## Next Decisions

- Decide whether the current slide language should use "chemical space" or
  "metabolite profile space" as the main term.
- Decide whether Purine nucleosides should be presented as the single headline
  class or as part of a broader purine/pyridine/pyrimidine/indole candidate set.
- Keep the current result framed as exploratory until the collaborator confirms
  the raw-to-matrix preprocessing semantics and the reporting dataset is fixed.
