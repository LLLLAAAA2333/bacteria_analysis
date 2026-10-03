# Population pair response signal: measured results

All 147 previously eligible comparisons were retained without neural-outcome selection. They cover 72 strains, 49 animals and 9 acquisition blocks. There are 137 pairs with all 13 neuron classes measured in at least 3 paired animals. Ten pairs support only 10 classes; ASI, ASEL and AWCON have only 2 paired animals there, so those cells remain missing.

The response score is a mean inner product of paired difference vectors from **different animals**. It excludes self products, averages five calcium bins within each neuron, and then gives eligible neurons equal weight in one fixed coordinate system. It measures consistency of a difference under the recorded protocol. It is not a classifier, a cross-date replication statistic, a nonnegative distance, or evidence for a specific molecular cause. Negative estimates are retained; they cannot establish response equivalence.

## Coverage and magnitude

| Subset | Pairs | Strains | Animals | Blocks | Complete 13-cell pairs | Score median | Score min/max |
|---|---:|---:|---:|---:|---:|---:|---:|
| All eligible pairs | 147 | 72 | 49 | 9 | 137 | 0.595 | -0.241 / 4.465 |
| Nearest in either direction | 45 | 72 | 49 | 9 | 39 | 0.318 | -0.241 / 2.721 |
| Nearest, at least 3 eligible strains in group | 28 | 40 | 38 | 7 | 26 | 0.315 | -0.024 / 2.393 |

The last restriction removes automatic nearest neighbors from groups containing only two strains. It is a chemical candidate-set restriction, not a neural selection. The two incomplete-panel pairs among these 28 are A246/A248 and A238/A240. They remain in the table and should have explicit coverage markers in a figure.

## Checks that could change the picture

Across all 147 pairs, the Spearman correlation between the fixed-scale score and the raw dF/F0-squared score is 0.857; the same-block, pair-excluded-scale sensitivity is 0.855; the fixed common 10-neuron panel is 0.981. For the 28 nonautomatic nearest neighbors these are 0.910, 0.800 and 0.978 respectively. These are descriptive concordances, without pair-independence assumptions or p-values; individual rankings do change. The pair-excluded scale uses at least 9 other strains and at least 3 animals for every eligible neuron. Its units differ across pairs, so it is a sensitivity and not a replacement common plotting coordinate system.

Only 77 pairs have at least three animals complete for every one of the 13 neurons. On those 77, the complete-animal sensitivity and available-cell/animal primary score have rank correlation 0.864. This restricted check does not establish general robustness for the other 70 pairs.

The score remains positive after deletion of any one whole animal for 120/147 pairs, 31/45 nearest pairs, and 19/28 nonautomatic nearest pairs. This is a deletion diagnostic, **not** a count of significant effects. Exported min/max ranges are not confidence intervals. The same original cell panel and scales are retained after deletion; a cell with 3 animals can have 2 after deletion.

## Numerical verification

The fixed cell scales were reconstructed from the aligned animal table and match the prior full-catalogue scales to 1.11e-16. Explicit double loops over distinct animal pairs match the fast inner-product formula to 1.78e-15. The identity `score = squared mean difference - sample variance / n` matches to 3.33e-16. Ineligible cells have missing scores, never zero. No negative score was truncated.

Remaining scientific alternatives are fixed stimulus order or preceding-stimulus effects, shared acquisition context, the finite animal sample, scaling uncertainty, and systematic missing cells. The statistic alone does not resolve them. First/later trial sensitivity and context associations are separate bounded analyses in this round.
